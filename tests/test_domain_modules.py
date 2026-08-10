"""Focused tests for dependency-free domain boundaries."""

from intent_bridge.core.text import extract_client_history, sanitise_spoken_response
from intent_bridge.home_assistant import compact_service_definition, normalise_service_data


def test_compact_service_definition_limits_ui_schema_to_tool_contract():
    result = compact_service_definition(
        "light",
        "turn_on",
        {
            "name": "Turn on",
            "description": "Turns a light on",
            "target": {"entity": {"domain": ["light"]}},
            "fields": {
                "brightness_pct": {
                    "required": False,
                    "default": 100,
                    "description": "Brightness percentage",
                }
            },
            "response": {"optional": True},
        },
    )
    assert result["target"] == {
        "required": True,
        "entity_domains": ["light"],
        "accepts_entity_id": True,
    }
    assert result["parameters"]["brightness_pct"]["default"] == 100
    assert result["returns_data"] is True


def test_service_retry_preserves_only_known_previous_fields():
    result, repairs, validation = normalise_service_data(
        "climate",
        "set_temperature",
        {"fields": {"temperature": {}, "hvac_mode": {}}},
        {"temperature": 20},
        {"hvac_mode": "heat", "invented": "discard me"},
    )
    assert result == {"temperature": 20, "hvac_mode": "heat"}
    assert repairs == ["preserved previous valid field 'hvac_mode'"]
    assert validation is None


def test_history_window_is_expressed_as_complete_turns():
    body = {
        "messages": [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "recent"},
            {"role": "assistant", "content": "recent answer"},
            {"role": "user", "content": "current"},
        ]
    }
    assert extract_client_history(body, turns=1) == [
        {"role": "user", "content": "recent"},
        {"role": "assistant", "content": "recent answer"},
    ]


def test_spoken_response_has_explicit_length_policy():
    assert sanitise_spoken_response("one two three four", max_chars=13) == "one two."
