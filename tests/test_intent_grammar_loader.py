from __future__ import annotations

import copy
import logging
from pathlib import Path

import pytest

from intent_bridge.intent_engine import (
    GrammarDependencyError,
    GrammarLoadError,
    IntentGrammarLoader,
    load_intent_grammar,
)


def _base_grammar() -> dict[str, object]:
    return {
        "language": "en",
        "intents": {
            "HassTurnOn": {
                "data": [
                    {
                        "sentences": ["turn on [the] {name}"],
                        "slots": {"domain": "light"},
                    }
                ]
            },
            "HassTurnOff": {"data": [{"sentences": ["turn off [the] {name}"]}]},
        },
        "lists": {"name": {"wildcard": True}},
        "expansion_rules": {"please": "[please]"},
    }


def _write(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def test_missing_custom_directory_warns_and_loads_packaged_grammar(tmp_path, caplog):
    base = _base_grammar()
    compiled_inputs: list[dict[str, object]] = []

    def compiler(grammar):
        compiled_inputs.append(grammar)
        return "compiled"

    with caplog.at_level(logging.WARNING):
        result = load_intent_grammar(
            language="en",
            custom_sentences_path=tmp_path / "missing",
            base_loader=lambda _language: base,
            compiler=compiler,
        )

    assert result.intents == "compiled"
    assert result.language == "en"
    assert result.custom_files == ()
    assert result.provenance == ()
    assert result.custom_sentence_count == 0
    assert compiled_inputs == [base]
    assert "continuing with packaged OHF grammar only" in caplog.text


def test_recursively_sorted_files_are_added_with_local_rules_lists_and_provenance(tmp_path, caplog):
    custom_path = tmp_path / "custom"
    later = _write(
        custom_path / "z_lights.yaml",
        """
language: en
intents:
  HassTurnOff:
    data:
      - sentences:
          - "kill [the] {area} light[s]"
        slots:
          domain: light
        requires_context:
          area: true
        excludes_context:
          domain: fan
""",
    )
    earlier = _write(
        custom_path / "nested" / "a_lights.yaml",
        """
language: en
intents:
  HassTurnOn:
    data:
      - sentences:
          - "<power_up> [the] {area} {lamp_word}"
          - "illuminate [the] {area}"
        slots:
          domain: light
        expansion_rules:
          power_up: "(power up|wake up)"
        lists:
          lamp_word:
            values:
              - lamp
              - lights
""",
    )
    base = _base_grammar()
    original_base = copy.deepcopy(base)
    compiled_inputs = []

    with caplog.at_level(logging.INFO):
        result = IntentGrammarLoader(
            language="en",
            custom_sentences_path=custom_path,
            base_loader=lambda _language: base,
            compiler=lambda grammar: compiled_inputs.append(grammar) or grammar,
        ).load()

    assert result.custom_files == (earlier, later)
    assert result.custom_sentence_count == 3
    assert [item.intent_name for item in result.provenance] == [
        "HassTurnOn",
        "HassTurnOn",
        "HassTurnOff",
    ]
    assert result.provenance[0].source == earlier
    assert result.provenance[0].data_index == 0
    assert result.provenance[1].sentence_index == 1
    merged = compiled_inputs[0]
    assert len(merged["intents"]["HassTurnOn"]["data"]) == 2
    assert len(merged["intents"]["HassTurnOff"]["data"]) == 2
    assert merged["lists"] == base["lists"]
    assert base == original_base
    assert "Loaded 3 custom sentence templates from 2 files" in caplog.text


@pytest.mark.parametrize(
    ("extra_yaml", "unsupported_key"),
    [
        ("responses: {}", "responses"),
        ("settings: {}", "settings"),
        ("skip_words: []", "skip_words"),
        ("lists: {}", "lists"),
        ("expansion_rules: {}", "expansion_rules"),
    ],
)
def test_global_customization_is_rejected(tmp_path, extra_yaml, unsupported_key):
    _write(
        tmp_path / "custom.yaml",
        f"""
language: en
intents:
  HassTurnOn:
    data:
      - sentences: ["switch on {{name}}"]
{extra_yaml}
""",
    )

    with pytest.raises(GrammarLoadError, match=f"unsupported keys: {unsupported_key}"):
        load_intent_grammar(
            language="en",
            custom_sentences_path=tmp_path,
            base_loader=lambda _language: _base_grammar(),
            compiler=lambda grammar: grammar,
        )


@pytest.mark.parametrize("unsupported_key", ["response", "metadata", "settings"])
def test_custom_data_only_allows_recognition_fields(tmp_path, unsupported_key):
    _write(
        tmp_path / "custom.yaml",
        f"""
language: en
intents:
  HassTurnOn:
    data:
      - sentences: ["switch on {{name}}"]
        {unsupported_key}: value
""",
    )

    with pytest.raises(GrammarLoadError, match=f"unsupported keys: {unsupported_key}"):
        load_intent_grammar(
            language="en",
            custom_sentences_path=tmp_path,
            base_loader=lambda _language: _base_grammar(),
            compiler=lambda grammar: grammar,
        )


def test_custom_intent_name_is_rejected(tmp_path):
    _write(
        tmp_path / "custom.yaml",
        """
language: en
intents:
  MyCustomIntent:
    data:
      - sentences: ["do a new thing"]
""",
    )

    with pytest.raises(GrammarLoadError, match="not an existing supported OHF intent"):
        load_intent_grammar(
            language="en",
            custom_sentences_path=tmp_path,
            base_loader=lambda _language: _base_grammar(),
            compiler=lambda grammar: grammar,
        )


def test_overlay_language_must_match_exactly(tmp_path):
    _write(
        tmp_path / "custom.yaml",
        """
language: en-GB
intents:
  HassTurnOn:
    data:
      - sentences: ["switch on {name}"]
""",
    )

    with pytest.raises(GrammarLoadError, match="must equal configured language 'en'"):
        load_intent_grammar(
            language="en",
            custom_sentences_path=tmp_path,
            base_loader=lambda _language: _base_grammar(),
            compiler=lambda grammar: grammar,
        )


@pytest.mark.parametrize(
    "sentence",
    [
        "turn on [the] {name}",
        "  TURN   ON [THE] {NAME}  ",
        "turn\u00a0on [the] {name}",
    ],
)
def test_template_duplicate_of_packaged_grammar_is_rejected(tmp_path, sentence):
    _write(
        tmp_path / "custom.yaml",
        f"""
language: en
intents:
  HassTurnOn:
    data:
      - sentences:
          - "{sentence}"
""",
    )

    with pytest.raises(GrammarLoadError, match="Duplicate sentence template"):
        load_intent_grammar(
            language="en",
            custom_sentences_path=tmp_path,
            base_loader=lambda _language: _base_grammar(),
            compiler=lambda grammar: grammar,
        )


def test_duplicate_templates_across_custom_files_are_rejected_before_compile(tmp_path):
    body = """
language: en
intents:
  HassTurnOn:
    data:
      - sentences: ["energise {name}"]
"""
    _write(tmp_path / "a.yaml", body)
    _write(tmp_path / "b.yaml", body)
    compile_calls = []

    with pytest.raises(GrammarLoadError, match="Duplicate sentence template"):
        load_intent_grammar(
            language="en",
            custom_sentences_path=tmp_path,
            base_loader=lambda _language: _base_grammar(),
            compiler=lambda grammar: compile_calls.append(grammar),
        )

    assert compile_calls == []


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("language: [", "Unable to load custom sentences"),
        (
            "language: !!python/object/apply:os.system ['echo unsafe']",
            "Unable to load custom sentences",
        ),
        (
            """
language: en
language: en
intents: {}
""",
            "duplicate key",
        ),
    ],
)
def test_malformed_or_unsafe_yaml_is_rejected(tmp_path, body, message):
    _write(tmp_path / "custom.yaml", body)

    with pytest.raises(GrammarLoadError, match=message):
        load_intent_grammar(
            language="en",
            custom_sentences_path=tmp_path,
            base_loader=lambda _language: _base_grammar(),
            compiler=lambda grammar: grammar,
        )


def test_all_files_must_validate_before_the_compiler_is_called(tmp_path):
    _write(
        tmp_path / "a_valid.yaml",
        """
language: en
intents:
  HassTurnOn:
    data:
      - sentences: ["energise {name}"]
""",
    )
    _write(
        tmp_path / "z_invalid.yaml",
        """
language: en
intents:
  HassTurnOff:
    data:
      - sentences: not-a-list
""",
    )
    base = _base_grammar()
    original = copy.deepcopy(base)
    compile_calls = []

    with pytest.raises(GrammarLoadError, match="sentences must be a non-empty list"):
        load_intent_grammar(
            language="en",
            custom_sentences_path=tmp_path,
            base_loader=lambda _language: base,
            compiler=lambda grammar: compile_calls.append(grammar),
        )

    assert compile_calls == []
    assert base == original


@pytest.mark.parametrize(
    ("fragment", "message"),
    [
        ("sentences: []", "sentences must be a non-empty list"),
        ("sentences: [12]", "must be a non-empty string"),
        ("sentences: ['energise {name}']\n        slots: []", "slots must be a mapping"),
        (
            "sentences: ['energise {name}']\n        expansion_rules:\n          empty: ''",
            "must map non-empty names to templates",
        ),
        (
            "sentences: ['energise {name}']\n        lists:\n          local:\n            values: []\n            wildcard: true",
            "must define exactly one",
        ),
    ],
)
def test_invalid_local_data_is_rejected(tmp_path, fragment, message):
    _write(
        tmp_path / "custom.yaml",
        f"""
language: en
intents:
  HassTurnOn:
    data:
      - {fragment}
""",
    )

    with pytest.raises(GrammarLoadError, match=message):
        load_intent_grammar(
            language="en",
            custom_sentences_path=tmp_path,
            base_loader=lambda _language: _base_grammar(),
            compiler=lambda grammar: grammar,
        )


@pytest.mark.parametrize(
    ("base", "language", "message"),
    [
        (None, "xx", "does not support language"),
        ({"language": "fr", "intents": {"HassTurnOn": {"data": []}}}, "en", "does not match"),
        ({"language": "en", "intents": {}}, "en", "contains no supported intents"),
    ],
)
def test_packaged_grammar_must_support_the_configured_language(tmp_path, base, language, message):
    with pytest.raises(GrammarLoadError, match=message):
        load_intent_grammar(
            language=language,
            custom_sentences_path=tmp_path,
            base_loader=lambda _language: base,
            compiler=lambda grammar: grammar,
        )


def test_configured_language_must_not_be_blank(tmp_path):
    with pytest.raises(GrammarLoadError, match="must not be empty"):
        load_intent_grammar(
            language="  ",
            custom_sentences_path=tmp_path,
            base_loader=lambda _language: _base_grammar(),
            compiler=lambda grammar: grammar,
        )


def test_existing_custom_path_must_be_a_directory(tmp_path):
    custom_file = _write(tmp_path / "custom.yaml", "language: en")

    with pytest.raises(GrammarLoadError, match="is not a directory"):
        load_intent_grammar(
            language="en",
            custom_sentences_path=custom_file,
            base_loader=lambda _language: _base_grammar(),
            compiler=lambda grammar: grammar,
        )


def test_compiler_errors_are_reported_as_grammar_load_errors(tmp_path):
    def fail_compile(_grammar):
        raise RuntimeError("bad sentence expression")

    with pytest.raises(GrammarLoadError, match="Unable to compile combined HassIL grammar"):
        load_intent_grammar(
            language="en",
            custom_sentences_path=tmp_path,
            base_loader=lambda _language: _base_grammar(),
            compiler=fail_compile,
        )


def test_dependency_errors_from_compiler_remain_actionable(tmp_path):
    def fail_compile(_grammar):
        raise GrammarDependencyError("install hassil")

    with pytest.raises(GrammarDependencyError, match="install hassil"):
        load_intent_grammar(
            language="en",
            custom_sentences_path=tmp_path,
            base_loader=lambda _language: _base_grammar(),
            compiler=fail_compile,
        )


def test_pinned_ohf_package_and_hassil_compile_a_real_overlay(tmp_path):
    _write(
        tmp_path / "lighting.yaml",
        """
language: en
intents:
  HassTurnOn:
    data:
      - sentences:
          - "<power_up> [the] {device_alias}"
        slots:
          domain: light
        expansion_rules:
          power_up: "(energise|wake up)"
        lists:
          device_alias:
            values:
              - reading lamp
""",
    )

    result = load_intent_grammar(
        language="en",
        custom_sentences_path=tmp_path,
    )

    assert type(result.intents).__module__ == "hassil.intents"
    custom_data = result.intents.intents["HassTurnOn"].data[-1]
    assert custom_data.sentence_texts == ["<power_up> [the] {device_alias}"]
    assert "power_up" in custom_data.expansion_rules
    assert "device_alias" in custom_data.slot_lists
