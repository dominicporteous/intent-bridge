from pathlib import Path

import pytest
from dotenv import dotenv_values

from intent_bridge.config import ConfigurationError, load_settings


def test_canonical_environment_maps_to_typed_domains():
    settings = load_settings(
        {
            "INTENT_BRIDGE_API_MODEL_NAME": "voice-actions",
            "INTENT_BRIDGE_BASE_URL": "https://bridge.example/proxy/",
            "INTENT_BRIDGE_TIMEZONE": "UTC",
            "INTENT_BRIDGE_LOCALE": "en-US",
            "INTENT_BRIDGE_LOCATION": "New York, United States",
            "INTENT_BRIDGE_VOICE_FAILURE_RESPONSE": "Please try another request.",
            "INTENT_BRIDGE_DETERMINISTIC_LANGUAGE": "en-GB",
            "INTENT_BRIDGE_DETERMINISTIC_CUSTOM_SENTENCES_PATH": "config/sentences/en",
            "INTENT_BRIDGE_DETERMINISTIC_ERROR_PHRASES": "not found; Try Again ",
            "INTENT_BRIDGE_LLM_ENABLED": "off",
            "INTENT_BRIDGE_LLM_AMBIGUOUS_TARGET_FALLBACK_ENABLED": "off",
            "INTENT_BRIDGE_LLM_API_KEY": "secret",
            "INTENT_BRIDGE_MCP_CONFIG_PATH": "config/mcp.json",
            "INTENT_BRIDGE_MCP_CONNECT_TIMEOUT_SECONDS": "12",
            "INTENT_BRIDGE_HA_SEARCH_DEFAULT_LIMIT": "4",
            "INTENT_BRIDGE_HA_SEARCH_MAX_LIMIT": "9",
            "INTENT_BRIDGE_HA_CATALOG_REFRESH_SECONDS": "45",
            "INTENT_BRIDGE_HA_CATALOG_EVENT_DEBOUNCE_SECONDS": "0.25",
            "INTENT_BRIDGE_HA_CATALOG_MINIMUM_REFRESH_SECONDS": "2",
            "INTENT_BRIDGE_HA_ADVANCED_ARGS": "-m tool --flag 'two words'",
            "INTENT_BRIDGE_HA_ADVANCED_PINNED_TOOLS": "one,two",
            "INTENT_BRIDGE_HA_STATE_CHANGING_SERVICES": "turn_on,custom_action",
            "INTENT_BRIDGE_MA_RADIO_SEED_STRATEGY": "FIRST",
            "INTENT_BRIDGE_MA_PREFER_NATIVE_PLAYBACK": "off",
            "INTENT_BRIDGE_ASSISTANT_LED_DOMAINS": "Light, SWITCH",
            "INTENT_BRIDGE_ASSISTANT_SOUNDS_ENABLED": "true",
            "INTENT_BRIDGE_CONVERSATION_TTL_SECONDS": "45",
        }
    )

    assert settings.api.model_name == "voice-actions"
    assert settings.api.base_url == "https://bridge.example/proxy"
    assert settings.assistant.sounds_enabled is True
    assert settings.assistant.led_domains == ("light", "switch")
    assert settings.api.timezone == "UTC"
    assert settings.api.locale == "en-US"
    assert settings.api.location == "New York, United States"
    assert settings.api.timezone_explicit is True
    assert settings.api.locale_explicit is True
    assert settings.api.location_explicit is True
    assert settings.api.voice_failure_response == "Please try another request."
    assert settings.deterministic.language == "en-GB"
    assert settings.deterministic.custom_sentences_path == Path("config/sentences/en")
    assert settings.deterministic.error_phrases == ("not found", "try again")
    assert settings.llm.enabled is False
    assert settings.llm.ambiguous_target_fallback_enabled is False
    assert settings.llm.api_key == "secret"
    assert settings.mcp.config_path == Path("config/mcp.json")
    assert settings.mcp.connect_timeout_seconds == 12
    assert settings.home_assistant.search_default_limit == 4
    assert settings.home_assistant.search_max_limit == 9
    assert settings.home_assistant.websocket.catalog_refresh_seconds == 45
    assert settings.home_assistant.websocket.catalog_event_debounce_seconds == 0.25
    assert settings.home_assistant.websocket.catalog_minimum_refresh_seconds == 2
    assert settings.home_assistant.advanced.args[-1] == "two words"
    assert settings.home_assistant.advanced.pinned_tools == ("one", "two")
    assert settings.home_assistant.ignored_entity_domains == ("update",)
    assert settings.home_assistant.state_changing_services == frozenset(
        {"turn_on", "custom_action"}
    )
    assert settings.music_assistant.prefer_native_playback is False


def test_home_assistant_ignored_entity_domains_can_be_configured_by_env():
    settings = load_settings(
        {"INTENT_BRIDGE_HA_IGNORED_ENTITY_DOMAINS": "update,select"}
    )

    assert settings.home_assistant.ignored_entity_domains == ("update", "select")


def test_deterministic_sentence_settings_have_safe_defaults():
    settings = load_settings({})
    blank_path_settings = load_settings(
        {"INTENT_BRIDGE_DETERMINISTIC_CUSTOM_SENTENCES_PATH": "   "}
    )

    assert settings.deterministic.language == "en"
    assert settings.deterministic.custom_sentences_path == Path("custom_sentences/en")
    assert blank_path_settings.deterministic.custom_sentences_path == Path("custom_sentences/en")
    assert settings.llm.ambiguous_target_fallback_enabled is True
    assert settings.music_assistant.prefer_native_playback is True
    assert settings.api.timezone_explicit is False
    assert settings.api.locale_explicit is False
    assert settings.api.location_explicit is False

def test_env_example_contains_only_canonical_valid_configuration():
    environment = {
        key: value or "" for key, value in dotenv_values(".env.example").items() if key is not None
    }
    assert environment
    assert all(name.startswith("INTENT_BRIDGE_") for name in environment)
    load_settings(environment)


@pytest.mark.parametrize(
    ("environment", "message"),
    [
        ({"INTENT_BRIDGE_LLM_ENABLED": "perhaps"}, "must be a boolean"),
        ({"INTENT_BRIDGE_LLM_TIMEOUT_SECONDS": "nope"}, "must be a number"),
        ({"INTENT_BRIDGE_LLM_TIMEOUT_SECONDS": "0"}, "must be at least"),
        (
            {
                "INTENT_BRIDGE_HA_SEARCH_DEFAULT_LIMIT": "10",
                "INTENT_BRIDGE_HA_SEARCH_MAX_LIMIT": "5",
            },
            "HA_SEARCH_MAX_LIMIT",
        ),
        (
            {
                "INTENT_BRIDGE_MA_PLAY_ACK_TIMEOUT_SECONDS": "10",
                "INTENT_BRIDGE_MA_PLAY_COMPLETION_TIMEOUT_SECONDS": "5",
            },
            "MA_PLAY_COMPLETION_TIMEOUT_SECONDS",
        ),
        (
            {
                "INTENT_BRIDGE_MA_FIRST_AUDIO_TIMEOUT_SECONDS": "10",
                "INTENT_BRIDGE_MA_BACKGROUND_TIMEOUT_SECONDS": "5",
            },
            "MA_BACKGROUND_TIMEOUT_SECONDS",
        ),
        (
            {"INTENT_BRIDGE_MA_RADIO_SEED_STRATEGY": "unknown"},
            "MA_RADIO_SEED_STRATEGY",
        ),
        (
            {"INTENT_BRIDGE_BASE_URL": "bridge.local"},
            "BASE_URL",
        ),
        (
            {"INTENT_BRIDGE_ASSISTANT_SOUNDS_ENABLED": "true"},
            "BASE_URL is required",
        ),
    ],
)
def test_invalid_canonical_configuration_fails_fast(environment, message):
    with pytest.raises(ConfigurationError, match=message):
        load_settings(environment)
