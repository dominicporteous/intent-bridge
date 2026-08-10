"""Characterization tests for the legacy public and pure-function contracts."""

from types import SimpleNamespace

import pytest

from intent_bridge import application
from intent_bridge.agents import results as tool_results
from intent_bridge.api import conversation
from intent_bridge.config import settings
from intent_bridge.runtime import execution as state


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("  TURN   ON the Light!!! ", "turn on the light"),
        ("What's the temperature?", "what's the temperature"),
        ("Kitchen, please.", "kitchen, please"),
    ],
)
def test_normalize_command(value, expected):
    assert state.normalize_command(value) == expected


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        ("http://ha.local:8123", "ws://ha.local:8123/api/websocket"),
        ("https://example.test/ha/", "wss://example.test/ha/api/websocket"),
        ("ws://ha.local", "ws://ha.local/api/websocket"),
    ],
)
def test_ha_websocket_url(base_url, expected):
    assert state._ha_websocket_url(base_url) == expected


def test_ha_websocket_url_rejects_unsupported_scheme():
    with pytest.raises(ValueError, match="http:// or https://"):
        state._ha_websocket_url("ftp://ha.local")


def test_service_schema_repairs_a_related_key_with_valid_enum_value(monkeypatch):
    monkeypatch.setattr(settings.home_assistant, "schema_auto_repair_enabled", True)
    definition = {
        "fields": {
            "hvac_mode": {
                "required": True,
                "selector": {"select": {"options": ["heat", "cool"]}},
            },
            "temperature": {"required": False},
        }
    }

    data, repairs, validation = state._normalise_service_data_from_schema(
        "climate", "set_hvac_mode", definition, {"new_hvac_mode": "heat"}, None
    )

    assert data == {"hvac_mode": "heat"}
    assert repairs and "renamed invalid field" in repairs[0]
    assert validation is None


def test_service_schema_reports_unknown_missing_and_invalid_values(monkeypatch):
    monkeypatch.setattr(settings.home_assistant, "schema_auto_repair_enabled", False)
    definition = {
        "description": "Set a mode",
        "fields": {
            "mode": {
                "required": True,
                "selector": {"select": {"options": ["auto", "manual"]}},
            }
        },
    }

    data, repairs, validation = state._normalise_service_data_from_schema(
        "demo", "set_mode", definition, {"mode": "invalid", "extra": 1}, None
    )

    assert data == {"mode": "invalid", "extra": 1}
    assert repairs == []
    assert validation["unknown_parameters"] == ["extra"]
    assert validation["missing_required_parameters"] == []
    assert validation["invalid_values"]["mode"]["allowed"] == ["auto", "manual"]


def test_message_text_supports_openai_multimodal_text_parts():
    message = {
        "content": [
            {"type": "text", "text": "turn on"},
            {"type": "image_url", "image_url": "ignored"},
            {"type": "input_text", "text": "the lamp"},
        ]
    }
    assert conversation._message_text(message) == "turn on the lamp"


def test_extract_client_history_excludes_system_and_latest_user(monkeypatch):
    monkeypatch.setattr(settings.conversation, "history_turns", 2)
    body = {
        "messages": [
            {"role": "system", "content": "secret system context"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "current"},
        ]
    }
    assert conversation.extract_client_history(body) == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer"},
    ]


def test_conversation_key_precedence_and_fallback(monkeypatch):
    request = SimpleNamespace(headers={"X-Conversation-ID": " header-id "})
    assert conversation.get_conversation_key(request, {"conversation_id": "body-id"}) == "header-id"

    assert conversation.get_conversation_key(SimpleNamespace(headers={}), {}) == "site:default"


def test_sanitise_spoken_response_removes_markup_and_followup(monkeypatch):
    monkeypatch.setattr(settings.api, "spoken_response_max_chars", 180)
    value = "**It is [18 degrees](https://example.test).** Would you like details?"
    assert tool_results.sanitise_spoken_response(value) == "It is 18 degrees."


@pytest.mark.asyncio
async def test_conversation_memory_keeps_only_configured_number_of_turns(monkeypatch):
    monkeypatch.setattr(settings.conversation, "enabled", True)
    monkeypatch.setattr(settings.conversation, "history_turns", 2)
    await conversation.clear_conversation_history("test")

    for number in range(3):
        await conversation.remember_conversation_turn("test", f"u{number}", f"a{number}")

    assert await conversation.get_conversation_history("test") == [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]


@pytest.mark.asyncio
async def test_models_endpoint_preserves_openai_compatible_shape():
    response = await application.models()
    assert response["object"] == "list"
    assert response["data"][0] == {
        "id": settings.api.model_name,
        "object": "model",
        "created": 0,
        "owned_by": "home-intent",
    }
