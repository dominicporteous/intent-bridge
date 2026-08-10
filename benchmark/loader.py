"""Strict, deterministic loader for the checked-in voice benchmark corpus."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from benchmark.models import (
    BenchmarkCorpus,
    BenchmarkExample,
    BenchmarkRequest,
    BenchmarkScenario,
    Home,
    HomeArea,
    HomeEntity,
    HomeFloor,
    Operation,
)

DEFAULT_DATASET_ROOT = Path(__file__).resolve().parent / "datasets"

_ACTION_ATTRIBUTE_KEYS = frozenset(
    {
        "brightness",
        "color",
        "color_temp",
        "hs_color",
        "percentage",
        "position",
        "rgb_color",
        "temperature",
        "volume_level",
    }
)
_DURATION_KEYS = frozenset(
    {
        "days",
        "hours",
        "milliseconds",
        "minutes",
        "seconds",
        "start_hours",
        "start_minutes",
        "start_seconds",
    }
)
_DOMAIN_WORDS: dict[str, tuple[str, ...]] = {
    "binary_sensor": ("sensor",),
    "climate": ("climate", "thermostat"),
    "cover": ("blind", "blinds", "cover", "covers", "curtain", "curtains", "door"),
    "fan": ("fan", "fans"),
    "light": ("lamp", "lamps", "light", "lights"),
    "lock": ("door", "lock", "locks"),
    "media_player": ("player", "speaker", "stereo", "tv"),
    "scene": ("scene",),
    "script": ("script",),
    "switch": ("switch",),
    "timer": ("timer",),
}
_COLLECTIVE_WORDS = frozenset({"all", "both", "every", "everything", "them"})


class CorpusLoadError(ValueError):
    """Raised when a fixture does not conform to the benchmark schema."""


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False):
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise CorpusLoadError(f"duplicate YAML key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)


def _read_yaml(path: Path) -> Any:
    try:
        return yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise CorpusLoadError(f"cannot load {path}: {exc}") from exc


def _text(value: Any, *, field: str, path: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CorpusLoadError(f"{path}: {field} must be a non-empty string")
    return value.strip()


def _normal_words(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _dedupe(values: Iterable[str]) -> tuple[str, ...]:
    found: dict[str, str] = {}
    for value in values:
        normal = _normal_words(value)
        if normal:
            found.setdefault(normal, normal)
    return tuple(found[key] for key in sorted(found))


def _entity_from_config(raw: Mapping[str, Any], *, prefix: str | None = None) -> HomeEntity:
    raw_id = str(raw["id"])
    entity_id = raw_id if "." in raw_id or prefix is None else f"{prefix}.{raw_id}"
    domain = entity_id.split(".", 1)[0]
    name = str(raw.get("name") or raw_id.replace("_", " ")).strip()
    aliases = raw.get("aliases") or []
    if not isinstance(aliases, list):
        aliases = []
    state = raw.get("state")
    attributes = raw.get("attributes")
    return HomeEntity(
        entity_id=entity_id,
        name=name,
        domain=domain,
        area_id=str(raw["area_id"]) if raw.get("area_id") else None,
        state=str(state) if state is not None else None,
        attributes=dict(attributes) if isinstance(attributes, Mapping) else {},
        aliases=_dedupe([name, raw_id.replace("_", " "), *map(str, aliases)]),
    )


def _load_home(path: Path) -> Home:
    raw = _read_yaml(path)
    if not isinstance(raw, Mapping):
        raise CorpusLoadError(f"{path}: home_config.yaml must contain a mapping")
    home_id = path.parent.name
    floors = tuple(
        HomeFloor(
            floor_id=_text(item.get("id"), field="floor.id", path=path),
            name=_text(item.get("name"), field="floor.name", path=path),
            level=int(item["level"]) if item.get("level") is not None else None,
            aliases=_dedupe(
                [str(item.get("name") or ""), str(item.get("id") or "").replace("_", " ")]
            ),
        )
        for item in raw.get("floors", [])
    )
    areas = tuple(
        HomeArea(
            area_id=_text(item.get("id"), field="area.id", path=path),
            name=_text(item.get("name"), field="area.name", path=path),
            floor_id=str(item["floor"]) if item.get("floor") else None,
            aliases=_dedupe(
                [str(item.get("name") or ""), str(item.get("id") or "").replace("_", " ")]
            ),
        )
        for item in raw.get("areas", [])
    )
    entities: list[HomeEntity] = [_entity_from_config(item) for item in raw.get("devices", [])]
    for collection, prefix in (
        ("scenes", "scene"),
        ("scripts", "script"),
        ("timers", "timer"),
        ("persons", "person"),
        ("lists", "todo"),
    ):
        entities.extend(
            _entity_from_config(item, prefix=prefix) for item in raw.get(collection, [])
        )
    return Home(
        home_id=home_id,
        name=_text(raw.get("name"), field="name", path=path),
        difficulty=str(raw.get("difficulty") or "unspecified"),
        floors=floors,
        areas=areas,
        entities=tuple(sorted(entities, key=lambda entity: entity.entity_id)),
        metadata={
            key: value
            for key, value in raw.items()
            if key not in {"name", "difficulty", "floors", "areas", "devices"}
        },
    )


def _entity_id(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, Mapping) and isinstance(value.get("id"), str):
        return value["id"].strip()
    return None


def _target_phrases(
    home: Home, entity_ids: tuple[str, ...], area_id: str | None, domain: str | None
) -> tuple[str, ...]:
    phrases: list[str] = []
    for entity_id in entity_ids:
        entity = home.entity(entity_id)
        if entity is None:
            phrases.append(entity_id.split(".", 1)[-1].replace("_", " "))
            unknown_domain = entity_id.split(".", 1)[0]
            phrases.extend(_DOMAIN_WORDS.get(unknown_domain, (unknown_domain,)))
            continue
        phrases.extend((entity.name, *entity.aliases))
        area = home.area(entity.area_id) if entity.area_id else None
        if area:
            for area_alias in (area.name, *area.aliases):
                for domain_word in _DOMAIN_WORDS.get(entity.domain, (entity.domain,)):
                    phrases.append(f"{area_alias} {domain_word}")
                    phrases.append(f"{domain_word} in {area_alias}")
    if area_id:
        area = home.area(area_id)
        if area:
            phrases.extend((area.name, *area.aliases))
            for area_alias in (area.name, *area.aliases):
                for domain_word in _DOMAIN_WORDS.get(domain or "", (domain or "",)):
                    phrases.append(f"{area_alias} {domain_word}")
                    phrases.append(f"{domain_word} in {area_alias}")
    return _dedupe(phrases)


def _operation(
    raw: Mapping[str, Any],
    home: Home,
    *,
    setup: bool = False,
    category: str | None = None,
) -> Operation:
    entity_id = _entity_id(raw.get("entity_id"))
    area_id = str(raw["area"]) if raw.get("area") else None
    domain = str(raw["domain"]) if raw.get("domain") else None
    entity_ids: tuple[str, ...] = (entity_id,) if entity_id else ()
    if area_id and domain:
        entity_ids = tuple(entity.entity_id for entity in home.entities_in(area_id, domain))

    raw_kind = str(raw.get("type") or "").strip()
    if setup:
        kind = "setup"
    elif raw_kind:
        kind = raw_kind
    elif category == "timers":
        kind = "timer"
    else:
        kind = "action"

    state = raw.get("state")
    attributes = raw.get("attributes")
    payload: dict[str, Any] = dict(attributes) if isinstance(attributes, Mapping) else {}
    muted = payload.get("is_volume_muted")
    if isinstance(muted, str) and muted.casefold() in {"true", "false"}:
        payload["is_volume_muted"] = muted.casefold() == "true"
    reserved = {"type", "entity_id", "area", "domain", "state", "attributes", "intent"}
    for key, value in raw.items():
        if key in reserved:
            continue
        if (
            key in _ACTION_ATTRIBUTE_KEYS
            or key in _DURATION_KEYS
            or kind
            in {
                "shopping_list",
                "todo_list",
                "timer",
                "setup",
            }
        ):
            payload[key] = value

    target_phrases = _target_phrases(home, entity_ids, area_id, domain)
    if area_id is None and len(entity_ids) == 1:
        entity = home.entity(entity_ids[0])
        area_id = entity.area_id if entity is not None else None

    return Operation(
        kind=kind,
        entity_ids=entity_ids,
        area_id=area_id,
        domain=domain,
        state=str(state) if state is not None else None,
        payload=payload,
        intent=str(raw["intent"]) if raw.get("intent") else None,
        target_phrases=target_phrases,
    )


def _automation_operation(raw: Mapping[str, Any]) -> Operation:
    expected = raw.get("expected")
    if not isinstance(expected, Mapping):
        raise CorpusLoadError("automation scenario expected must be a mapping")
    return Operation(kind="automation", payload={"definition": dict(expected)})


def _activation_state(action: str) -> str | None:
    domain, _, service = action.partition(".")
    if service in {"turn_on", "turn_off"}:
        state = "on" if service == "turn_on" else "off"
    elif service in {"open", "open_cover"}:
        state = "open"
    elif service in {"close", "close_cover"}:
        state = "closed"
    elif service == "lock":
        state = "locked"
    elif service == "unlock":
        state = "unlocked"
    elif service in {"media_pause", "pause"}:
        state = "paused"
    elif service in {"media_play", "play"}:
        state = "playing"
    else:
        state = None
    if domain == "cover" and service in {"turn_on", "turn_off"}:
        return "open" if service == "turn_on" else "closed"
    if domain == "lock" and service in {"turn_on", "turn_off"}:
        return "locked" if service == "turn_on" else "unlocked"
    return state


def _expanded_invocation(operation: Operation, home: Home) -> tuple[Operation, ...]:
    """Expand a scene/script invocation into its configured observable effects."""

    if len(operation.entity_ids) != 1:
        return (operation,)
    entity_id = operation.entity_ids[0]
    domain, _, object_id = entity_id.partition(".")
    if domain == "scene":
        collection = home.metadata.get("scenes")
        raw = next(
            (
                item
                for item in collection or ()
                if isinstance(item, Mapping) and str(item.get("id")) == object_id
            ),
            None,
        )
        entities = raw.get("entities") if isinstance(raw, Mapping) else None
        if not isinstance(entities, Mapping):
            return (operation,)
        effects: list[Operation] = []
        for target, value in entities.items():
            if isinstance(value, Mapping):
                state = value.get("state")
                payload = {key: item for key, item in value.items() if key != "state"}
            else:
                state = value
                payload = {}
            effects.append(
                Operation(
                    kind="action",
                    entity_ids=(str(target),),
                    state=str(state) if state is not None else None,
                    payload=payload,
                    target_phrases=operation.target_phrases,
                )
            )
        return tuple(effects)

    if domain != "script":
        return (operation,)
    collection = home.metadata.get("scripts")
    raw = next(
        (
            item
            for item in collection or ()
            if isinstance(item, Mapping) and str(item.get("id")) == object_id
        ),
        None,
    )
    actions = raw.get("actions") if isinstance(raw, Mapping) else None
    if not isinstance(actions, list):
        return (operation,)
    effects = []
    for action_item in actions:
        if not isinstance(action_item, Mapping):
            continue
        target = action_item.get("target")
        raw_targets = target.get("entity_id") if isinstance(target, Mapping) else None
        if raw_targets is None:
            raw_targets = action_item.get("entity_id")
        if isinstance(raw_targets, str):
            entity_ids = (raw_targets,)
        elif isinstance(raw_targets, list):
            entity_ids = tuple(str(item) for item in raw_targets)
        else:
            entity_ids = ()
        action = str(action_item.get("action") or action_item.get("service") or "")
        data = action_item.get("data")
        effects.append(
            Operation(
                kind="action",
                entity_ids=tuple(sorted(entity_ids)),
                state=_activation_state(action),
                payload=dict(data) if isinstance(data, Mapping) else {},
                target_phrases=operation.target_phrases,
            )
        )
    return tuple(effects) or (operation,)


def _contains_phrase(words: str, phrase: str) -> bool:
    # Articles are optional in natural device references (``blinds in the
    # master bedroom`` and ``blinds in master bedroom`` are equivalent).
    # Keeping this normalization local avoids weakening the corpus-wide input
    # identity used by conflict validation.
    def without_articles(value: str) -> str:
        return " ".join(word for word in value.split() if word not in {"a", "an", "the"})

    return f" {without_articles(phrase)} " in f" {without_articles(words)} "


def _project_expected(
    turns: tuple[str, ...], expected: tuple[Operation, ...]
) -> tuple[Operation, ...]:
    """Project combined case conditions onto one independently spoken example.

    Some source cases describe two devices but contain alternative sentences
    naming only one of them.  Explicit target phrases select the per-example
    subset.  A collective or a wording with no resolvable explicit target keeps
    the whole expectation, which is appropriate for pronoun-based dialogue and
    broad area wording.
    """

    if len(expected) <= 1:
        return expected
    words = _normal_words(" ".join(turns))
    # A collective explicitly refers to the whole coordinated target set.
    # Check it before partial phrase matches so, for example, ``both the front
    # door and garage entry`` cannot be projected onto just the first lock.
    if _COLLECTIVE_WORDS.intersection(words.split()):
        return expected

    explicit = tuple(
        operation
        for operation in expected
        if operation.target_phrases
        and any(_contains_phrase(words, phrase) for phrase in operation.target_phrases)
    )
    # Coordinated area wording commonly elides the repeated domain noun:
    # ``lights in bedroom 3 and the kitchen``.  Once two distinct expected
    # areas are named, retain every expected operation in those named areas.
    mentioned_areas = {
        operation.area_id
        for operation in expected
        if operation.area_id and _contains_phrase(words, _normal_words(operation.area_id))
    }
    coordinated = tuple(
        operation
        for operation in expected
        if len(mentioned_areas) >= 2 and operation.area_id in mentioned_areas
    )
    targetless = tuple(operation for operation in expected if not operation.target_phrases)
    if explicit or coordinated:
        projected: list[Operation] = []
        seen: set[tuple[Any, ...]] = set()
        for operation in (*explicit, *coordinated, *targetless):
            key = operation.semantic_key()
            if key not in seen:
                seen.add(key)
                projected.append(operation)
        return tuple(projected)
    return expected


def _turns(value: Any, *, path: Path) -> tuple[str, ...]:
    raw_turns = value if isinstance(value, list) else [value]
    if not raw_turns:
        raise CorpusLoadError(f"{path}: a dialogue cannot be empty")
    return tuple(_text(turn, field="sentence turn", path=path) for turn in raw_turns)


def _load_scenario_item(
    raw: Mapping[str, Any],
    *,
    home: Home,
    path: Path,
    dataset_root: Path,
    item_index: int,
) -> BenchmarkScenario:
    name = _text(raw.get("name"), field="name", path=path)
    source = path.relative_to(dataset_root).as_posix()
    scenario_id = f"{source}::{item_index}:{name}"
    setup_raw = raw.get("setup") or []
    if not isinstance(setup_raw, list) or not all(isinstance(item, Mapping) for item in setup_raw):
        raise CorpusLoadError(f"{path}: setup must be a list of mappings")
    setup = tuple(_operation(item, home, setup=True) for item in setup_raw)

    if "expected" in raw:
        expected = (_automation_operation(raw),)
    else:
        conditions = raw.get("conditions")
        if not isinstance(conditions, list) or not conditions:
            raise CorpusLoadError(f"{path}: conditions must be a non-empty list")
        if not all(isinstance(item, Mapping) for item in conditions):
            raise CorpusLoadError(f"{path}: every condition must be a mapping")
        # Clarification fixtures encode the selected action first and use the
        # remaining conditions as invariants proving that similarly named
        # devices were not changed.
        if "clarifications" in path.relative_to(dataset_root).parts:
            conditions = conditions[:1]
        category = path.relative_to(dataset_root).parts[1]
        expected = tuple(
            effect
            for item in conditions
            for effect in _expanded_invocation(
                _operation(item, home, category=category),
                home,
            )
        )

    sentences = raw.get("sentences")
    if not isinstance(sentences, list) or not sentences:
        raise CorpusLoadError(f"{path}: sentences must be a non-empty list")
    examples: list[BenchmarkExample] = []
    for example_index, sentence in enumerate(sentences):
        turns = _turns(sentence, path=path)
        examples.append(
            BenchmarkExample(
                diagnostic_id=f"{scenario_id}::example-{example_index}",
                request=BenchmarkRequest(turns=turns, home=home, setup=setup),
                expected=_project_expected(turns, expected),
            )
        )
    return BenchmarkScenario(
        diagnostic_id=scenario_id,
        name=name,
        source=source,
        home=home,
        setup=setup,
        expected=expected,
        examples=tuple(examples),
    )


def _request_key(example: BenchmarkExample) -> tuple[Any, ...]:
    return (
        example.request.home.home_id,
        tuple(_normal_words(turn) for turn in example.request.turns),
        tuple(
            sorted(
                (operation.semantic_key() for operation in example.request.setup),
                key=repr,
            )
        ),
        tuple(
            sorted((str(key), repr(value)) for key, value in example.request.origin_context.items())
        ),
    )


def _project_duplicate_inputs(
    scenarios: list[BenchmarkScenario],
) -> list[BenchmarkScenario]:
    """Give identical runtime inputs one deterministic answer key.

    Generated combined-device files repeat wordings from focused single-device
    files. Their common operation is the gold answer for that particular
    wording. Empty intersections are retained for validation to report as real
    fixture contradictions rather than silently choosing an answer.
    """

    groups: dict[tuple[Any, ...], list[BenchmarkExample]] = {}
    for scenario in scenarios:
        for example in scenario.examples:
            groups.setdefault(_request_key(example), []).append(example)

    projected: dict[str, tuple[Operation, ...]] = {}
    for examples in groups.values():
        if len(examples) < 2:
            continue
        variants = [
            {operation.semantic_key() for operation in example.expected} for example in examples
        ]
        common = set.intersection(*variants)
        if not common or all(variant == common for variant in variants):
            continue
        for example in examples:
            projected[example.diagnostic_id] = tuple(
                operation for operation in example.expected if operation.semantic_key() in common
            )

    if not projected:
        return scenarios
    return [
        replace(
            scenario,
            examples=tuple(
                replace(example, expected=projected[example.diagnostic_id])
                if example.diagnostic_id in projected
                else example
                for example in scenario.examples
            ),
        )
        for scenario in scenarios
    ]


def load_corpus(dataset_root: Path | str = DEFAULT_DATASET_ROOT) -> BenchmarkCorpus:
    """Load every YAML fixture below ``dataset_root`` in a stable order."""

    root = Path(dataset_root).resolve()
    if not root.is_dir():
        raise CorpusLoadError(f"dataset directory does not exist: {root}")
    config_paths = sorted(root.glob("*/home_config.yaml"), key=lambda path: path.as_posix())
    if not config_paths:
        raise CorpusLoadError(f"no home_config.yaml files found below {root}")
    loaded_homes = tuple(_load_home(path) for path in config_paths)
    homes = {home.home_id: home for home in loaded_homes}

    scenarios: list[BenchmarkScenario] = []
    scenario_paths = sorted(
        (path for path in root.rglob("*.yaml") if path.name != "home_config.yaml"),
        key=lambda path: path.as_posix(),
    )
    for path in scenario_paths:
        relative = path.relative_to(root)
        home = homes.get(relative.parts[0])
        if home is None:
            raise CorpusLoadError(f"{path}: no sibling home_config.yaml")
        document = _read_yaml(path)
        items = document if isinstance(document, list) else [document]
        if not items or not all(isinstance(item, Mapping) for item in items):
            raise CorpusLoadError(
                f"{path}: scenario document must be a mapping or list of mappings"
            )
        scenarios.extend(
            _load_scenario_item(
                item,
                home=home,
                path=path,
                dataset_root=root,
                item_index=index,
            )
            for index, item in enumerate(items)
        )
    scenarios = _project_duplicate_inputs(scenarios)
    return BenchmarkCorpus(
        root=str(root),
        homes=tuple(homes[key] for key in sorted(homes)),
        scenarios=tuple(scenarios),
    )


__all__ = ["DEFAULT_DATASET_ROOT", "CorpusLoadError", "load_corpus"]
