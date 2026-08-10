"""Strict, additive loading of custom HassIL sentences.

The built-in grammar always comes from the installed ``home-assistant-intents``
package.  Local YAML files may only add sentence data to intent names already
present in that package.  They cannot replace global lists, rules, responses,
or parser settings.
"""

from __future__ import annotations

import copy
import logging
import unicodedata
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode

LOGGER = logging.getLogger(__name__)

BaseGrammarLoader = Callable[[str], Mapping[str, Any] | None]
GrammarCompiler = Callable[[dict[str, Any]], object]

_TOP_LEVEL_KEYS = frozenset({"language", "intents"})
_INTENT_KEYS = frozenset({"data"})
_DATA_KEYS = frozenset(
    {
        "sentences",
        "slots",
        "requires_context",
        "excludes_context",
        "lists",
        "expansion_rules",
    }
)


class GrammarLoadError(ValueError):
    """Raised when an OHF grammar or custom sentence overlay is invalid."""


class GrammarDependencyError(RuntimeError):
    """Raised when the packaged grammar or HassIL compiler is unavailable."""


@dataclass(frozen=True, slots=True)
class SentenceProvenance:
    """Source location for one custom sentence template."""

    intent_name: str
    template: str
    source: Path
    data_index: int
    sentence_index: int


@dataclass(frozen=True, slots=True)
class LoadedIntentGrammar:
    """One immutable-at-publication compiled grammar snapshot."""

    intents: object
    language: str
    custom_files: tuple[Path, ...]
    provenance: tuple[SentenceProvenance, ...]

    @property
    def custom_sentence_count(self) -> int:
        """Return the number of custom templates in this snapshot."""

        return len(self.provenance)


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader which rejects mappings with duplicate keys."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    loader.flatten_mapping(node)
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _load_packaged_ohf_grammar(language: str) -> Mapping[str, Any] | None:
    try:
        from home_assistant_intents import get_intents
    except ImportError as exc:  # pragma: no cover - exercised in an uninstalled environment
        raise GrammarDependencyError(
            "The 'home-assistant-intents' package is required to load the OHF grammar"
        ) from exc

    return get_intents(language)


def _compile_hassil_grammar(grammar: dict[str, Any]) -> object:
    try:
        from hassil.intents import Intents
    except ImportError as exc:  # pragma: no cover - exercised in an uninstalled environment
        raise GrammarDependencyError(
            "The 'hassil' package is required to compile deterministic intents"
        ) from exc

    return Intents.from_dict(grammar)


def _expect_mapping(value: object, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GrammarLoadError(f"{location} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise GrammarLoadError(f"{location} keys must be strings")
    return value


def _expect_optional_mapping(container: Mapping[str, Any], key: str, location: str) -> None:
    if key in container:
        _expect_mapping(container[key], f"{location}.{key}")


def _reject_unknown_keys(
    value: Mapping[str, Any],
    allowed: frozenset[str],
    location: str,
) -> None:
    unknown = sorted(set(value).difference(allowed))
    if unknown:
        rendered = ", ".join(unknown)
        raise GrammarLoadError(f"{location} contains unsupported keys: {rendered}")


def _canonical_template(template: str) -> str:
    normalized = unicodedata.normalize("NFKC", template)
    return " ".join(normalized.split()).casefold()


def _base_templates(grammar: Mapping[str, Any]) -> set[str]:
    templates: set[str] = set()
    intents = _expect_mapping(grammar.get("intents"), "packaged OHF grammar.intents")
    for intent_value in intents.values():
        if not isinstance(intent_value, Mapping):
            continue
        data_items = intent_value.get("data", ())
        if not isinstance(data_items, list):
            continue
        for data_item in data_items:
            if not isinstance(data_item, Mapping):
                continue
            sentences = data_item.get("sentences", ())
            if not isinstance(sentences, list):
                continue
            templates.update(
                _canonical_template(sentence)
                for sentence in sentences
                if isinstance(sentence, str) and sentence.strip()
            )
    return templates


def _read_yaml(path: Path) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as source_file:
            value = yaml.load(source_file, Loader=_UniqueKeySafeLoader)
    except (OSError, yaml.YAMLError) as exc:
        raise GrammarLoadError(f"Unable to load custom sentences from {path}: {exc}") from exc
    return _expect_mapping(value, str(path))


def _validate_local_rules(data: Mapping[str, Any], location: str) -> None:
    rules = data.get("expansion_rules")
    if rules is None:
        return
    rules_mapping = _expect_mapping(rules, f"{location}.expansion_rules")
    for rule_name, rule_body in rules_mapping.items():
        if not rule_name.strip() or not isinstance(rule_body, str) or not rule_body.strip():
            raise GrammarLoadError(
                f"{location}.expansion_rules must map non-empty names to templates"
            )


def _validate_local_lists(data: Mapping[str, Any], location: str) -> None:
    lists = data.get("lists")
    if lists is None:
        return
    lists_mapping = _expect_mapping(lists, f"{location}.lists")
    for list_name, list_definition in lists_mapping.items():
        if not list_name.strip():
            raise GrammarLoadError(f"{location}.lists contains an empty list name")
        definition = _expect_mapping(list_definition, f"{location}.lists.{list_name}")
        list_kinds = {"values", "range", "wildcard"}.intersection(definition)
        if len(list_kinds) != 1:
            raise GrammarLoadError(
                f"{location}.lists.{list_name} must define exactly one of "
                "values, range, or wildcard"
            )


def _validate_data_item(
    value: object,
    *,
    location: str,
    intent_name: str,
    source: Path,
    data_index: int,
    seen_templates: set[str],
) -> tuple[dict[str, Any], list[SentenceProvenance]]:
    data = _expect_mapping(value, location)
    _reject_unknown_keys(data, _DATA_KEYS, location)

    sentences = data.get("sentences")
    if not isinstance(sentences, list) or not sentences:
        raise GrammarLoadError(f"{location}.sentences must be a non-empty list")

    provenance: list[SentenceProvenance] = []
    for sentence_index, sentence in enumerate(sentences):
        sentence_location = f"{location}.sentences[{sentence_index}]"
        if not isinstance(sentence, str) or not sentence.strip():
            raise GrammarLoadError(f"{sentence_location} must be a non-empty string")
        canonical = _canonical_template(sentence)
        if canonical in seen_templates:
            raise GrammarLoadError(
                f"Duplicate sentence template at {sentence_location}: {sentence}"
            )
        seen_templates.add(canonical)
        provenance.append(
            SentenceProvenance(
                intent_name=intent_name,
                template=sentence,
                source=source,
                data_index=data_index,
                sentence_index=sentence_index,
            )
        )

    for mapping_key in ("slots", "requires_context", "excludes_context"):
        _expect_optional_mapping(data, mapping_key, location)
    _validate_local_rules(data, location)
    _validate_local_lists(data, location)
    return copy.deepcopy(dict(data)), provenance


class IntentGrammarLoader:
    """Build one validated HassIL snapshot from OHF and additive local YAML."""

    def __init__(
        self,
        *,
        language: str,
        custom_sentences_path: str | Path,
        base_loader: BaseGrammarLoader | None = None,
        compiler: GrammarCompiler | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._language = language.strip()
        self._custom_sentences_path = Path(custom_sentences_path)
        self._base_loader = base_loader or _load_packaged_ohf_grammar
        self._compiler = compiler or _compile_hassil_grammar
        self._logger = logger or LOGGER

    def load(self) -> LoadedIntentGrammar:
        """Validate all sources and compile them as one atomic startup snapshot."""

        if not self._language:
            raise GrammarLoadError("Deterministic intent language must not be empty")

        base_value = self._base_loader(self._language)
        if base_value is None:
            raise GrammarLoadError(
                f"The packaged OHF grammar does not support language {self._language!r}"
            )
        base = copy.deepcopy(dict(_expect_mapping(base_value, "packaged OHF grammar")))
        if base.get("language") != self._language:
            raise GrammarLoadError(
                "Packaged OHF grammar language does not match configured language "
                f"{self._language!r}"
            )
        base_intents = _expect_mapping(base.get("intents"), "packaged OHF grammar.intents")
        if not base_intents:
            raise GrammarLoadError("Packaged OHF grammar contains no supported intents")

        custom_path = self._custom_sentences_path
        if not custom_path.exists():
            self._logger.warning(
                "Custom sentence directory does not exist; continuing with packaged OHF "
                "grammar only: %s",
                custom_path,
            )
            custom_files: tuple[Path, ...] = ()
        elif not custom_path.is_dir():
            raise GrammarLoadError(f"Custom sentence path is not a directory: {custom_path}")
        else:
            try:
                custom_files = tuple(
                    sorted(
                        (path for path in custom_path.rglob("*.yaml") if path.is_file()),
                        key=lambda path: (
                            path.relative_to(custom_path).as_posix().casefold(),
                            path.relative_to(custom_path).as_posix(),
                        ),
                    )
                )
            except OSError as exc:
                raise GrammarLoadError(
                    f"Unable to search custom sentence directory {custom_path}: {exc}"
                ) from exc

        seen_templates = _base_templates(base)
        additions: list[tuple[str, dict[str, Any]]] = []
        provenance: list[SentenceProvenance] = []

        # Stage every addition first. Nothing is merged or compiled until all files pass.
        for source in custom_files:
            overlay = _read_yaml(source)
            _reject_unknown_keys(overlay, _TOP_LEVEL_KEYS, str(source))
            if overlay.get("language") != self._language:
                raise GrammarLoadError(
                    f"{source}.language must equal configured language {self._language!r}"
                )
            overlay_intents = _expect_mapping(overlay.get("intents"), f"{source}.intents")
            if not overlay_intents:
                raise GrammarLoadError(f"{source}.intents must not be empty")

            for intent_name, intent_value in overlay_intents.items():
                intent_location = f"{source}.intents.{intent_name}"
                if intent_name not in base_intents:
                    raise GrammarLoadError(
                        f"{intent_location} is not an existing supported OHF intent"
                    )
                intent = _expect_mapping(intent_value, intent_location)
                _reject_unknown_keys(intent, _INTENT_KEYS, intent_location)
                data_items = intent.get("data")
                if not isinstance(data_items, list) or not data_items:
                    raise GrammarLoadError(f"{intent_location}.data must be a non-empty list")

                for data_index, data_value in enumerate(data_items):
                    data, data_provenance = _validate_data_item(
                        data_value,
                        location=f"{intent_location}.data[{data_index}]",
                        intent_name=intent_name,
                        source=source,
                        data_index=data_index,
                        seen_templates=seen_templates,
                    )
                    additions.append((intent_name, data))
                    provenance.extend(data_provenance)

        mutable_intents = base["intents"]
        for intent_name, data in additions:
            mutable_intents[intent_name]["data"].append(data)

        try:
            compiled = self._compiler(base)
        except GrammarDependencyError:
            raise
        except Exception as exc:
            raise GrammarLoadError(f"Unable to compile combined HassIL grammar: {exc}") from exc

        if custom_files:
            self._logger.info(
                "Loaded %d custom sentence templates from %d files under %s",
                len(provenance),
                len(custom_files),
                custom_path,
            )

        return LoadedIntentGrammar(
            intents=compiled,
            language=self._language,
            custom_files=custom_files,
            provenance=tuple(provenance),
        )


def load_intent_grammar(
    *,
    language: str,
    custom_sentences_path: str | Path,
    base_loader: BaseGrammarLoader | None = None,
    compiler: GrammarCompiler | None = None,
    logger: logging.Logger | None = None,
) -> LoadedIntentGrammar:
    """Convenience entry point for loading a deterministic grammar snapshot."""

    return IntentGrammarLoader(
        language=language,
        custom_sentences_path=custom_sentences_path,
        base_loader=base_loader,
        compiler=compiler,
        logger=logger,
    ).load()


__all__ = [
    "GrammarDependencyError",
    "GrammarLoadError",
    "IntentGrammarLoader",
    "LoadedIntentGrammar",
    "SentenceProvenance",
    "load_intent_grammar",
]
