"""Canonical environment loader for Intent Bridge configuration."""

import os
import shlex
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlparse

from intent_bridge.config.models import (
    ApiSettings,
    AssistantSettings,
    BridgeSettings,
    ConversationSettings,
    DeterministicSettings,
    HomeAssistantAdvancedSettings,
    HomeAssistantSettings,
    HomeAssistantWebSocketSettings,
    LlmSettings,
    McpSettings,
    MusicAssistantSettings,
    VoiceOriginSettings,
)

PREFIX = "INTENT_BRIDGE_"
TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
FALSE_VALUES = frozenset({"0", "false", "no", "off"})


class ConfigurationError(ValueError):
    """Raised when a canonical environment value cannot be parsed."""


def _name(suffix: str) -> str:
    return PREFIX + suffix


def _text(environ: Mapping[str, str], suffix: str, default: str) -> str:
    return environ.get(_name(suffix), default).strip()


def _optional_text(
    environ: Mapping[str, str], suffix: str, default: str | None = None
) -> str | None:
    value = environ.get(_name(suffix))
    return default if value is None else value.strip() or None


def _path(environ: Mapping[str, str], suffix: str, default: Path) -> Path:
    value = _text(environ, suffix, str(default))
    return Path(value) if value else default


def _boolean(environ: Mapping[str, str], suffix: str, default: bool) -> bool:
    name = _name(suffix)
    raw = environ.get(name)
    if raw is None:
        return default
    value = raw.strip().casefold()
    if value in TRUE_VALUES:
        return True
    if value in FALSE_VALUES:
        return False
    raise ConfigurationError(f"{name} must be a boolean")


def _integer(
    environ: Mapping[str, str],
    suffix: str,
    default: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    name = _name(suffix)
    try:
        value = int(environ.get(name, str(default)))
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer") from exc
    if minimum is not None and value < minimum:
        raise ConfigurationError(f"{name} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ConfigurationError(f"{name} must be at most {maximum}")
    return value


def _number(
    environ: Mapping[str, str],
    suffix: str,
    default: float,
    *,
    minimum: float | None = None,
) -> float:
    name = _name(suffix)
    try:
        value = float(environ.get(name, str(default)))
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number") from exc
    if minimum is not None and value < minimum:
        raise ConfigurationError(f"{name} must be at least {minimum}")
    return value


def _csv(environ: Mapping[str, str], suffix: str, default: str) -> tuple[str, ...]:
    return tuple(
        item.strip() for item in environ.get(_name(suffix), default).split(",") if item.strip()
    )


def load_settings(environ: Mapping[str, str] | None = None) -> BridgeSettings:
    """Load and validate settings from canonical ``INTENT_BRIDGE_*`` names."""
    environ = os.environ if environ is None else environ

    base_url = _text(environ, "BASE_URL", "").rstrip("/")
    sounds_enabled = _boolean(environ, "ASSISTANT_SOUNDS_ENABLED", False)
    if base_url:
        parsed_base_url = urlparse(base_url)
        if parsed_base_url.scheme not in {"http", "https"} or not parsed_base_url.netloc:
            raise ConfigurationError(
                "INTENT_BRIDGE_BASE_URL must be an absolute http:// or https:// URL"
            )
    if sounds_enabled and not base_url:
        raise ConfigurationError(
            "INTENT_BRIDGE_BASE_URL is required when "
            "INTENT_BRIDGE_ASSISTANT_SOUNDS_ENABLED is true"
        )

    ha_search_default = _integer(environ, "HA_SEARCH_DEFAULT_LIMIT", 8, minimum=1)
    ha_search_max = _integer(environ, "HA_SEARCH_MAX_LIMIT", 20, minimum=1)
    if ha_search_max < ha_search_default:
        raise ConfigurationError(
            "INTENT_BRIDGE_HA_SEARCH_MAX_LIMIT must be greater than or equal to "
            "INTENT_BRIDGE_HA_SEARCH_DEFAULT_LIMIT"
        )

    play_ack_timeout = _number(environ, "MA_PLAY_ACK_TIMEOUT_SECONDS", 2.0, minimum=0.1)
    play_completion_timeout = _number(
        environ, "MA_PLAY_COMPLETION_TIMEOUT_SECONDS", 90.0, minimum=0.1
    )
    if play_completion_timeout < play_ack_timeout:
        raise ConfigurationError(
            "INTENT_BRIDGE_MA_PLAY_COMPLETION_TIMEOUT_SECONDS must be greater than or "
            "equal to INTENT_BRIDGE_MA_PLAY_ACK_TIMEOUT_SECONDS"
        )

    first_audio_timeout = _number(environ, "MA_FIRST_AUDIO_TIMEOUT_SECONDS", 15.0, minimum=1.0)
    background_timeout = _number(environ, "MA_BACKGROUND_TIMEOUT_SECONDS", 300.0, minimum=1.0)
    if background_timeout < first_audio_timeout:
        raise ConfigurationError(
            "INTENT_BRIDGE_MA_BACKGROUND_TIMEOUT_SECONDS must be greater than or equal "
            "to INTENT_BRIDGE_MA_FIRST_AUDIO_TIMEOUT_SECONDS"
        )

    radio_strategy = _text(environ, "MA_RADIO_SEED_STRATEGY", "weighted").casefold()
    if radio_strategy not in {"weighted", "random", "first"}:
        raise ConfigurationError(
            "INTENT_BRIDGE_MA_RADIO_SEED_STRATEGY must be weighted, random, or first"
        )

    return BridgeSettings(
        api=ApiSettings(
            base_url=base_url,
            model_name=_text(environ, "API_MODEL_NAME", "home-intent"),
            timezone=_text(environ, "TIMEZONE", "Europe/London"),
            spoken_response_max_chars=_integer(environ, "VOICE_RESPONSE_MAX_CHARS", 180, minimum=1),
            action_confirmation=_text(environ, "VOICE_ACTION_CONFIRMATION", ""),
            voice_failure_response=(
                _text(
                    environ,
                    "VOICE_FAILURE_RESPONSE",
                    "Sorry, I couldn't handle that request.",
                )
            ),
            log_level=_text(environ, "LOG_LEVEL", "INFO").upper(),
        ),
        deterministic=DeterministicSettings(
            language=_text(environ, "DETERMINISTIC_LANGUAGE", "en") or "en",
            custom_sentences_path=_path(
                environ,
                "DETERMINISTIC_CUSTOM_SENTENCES_PATH",
                Path("custom_sentences/en"),
            ),
            timeout_seconds=_number(environ, "DETERMINISTIC_TIMEOUT_SECONDS", 10.0, minimum=0.0),
            response_grace_seconds=_number(
                environ, "DETERMINISTIC_RESPONSE_GRACE_SECONDS", 0.75, minimum=0.0
            ),
            default_response=_text(environ, "DETERMINISTIC_DEFAULT_RESPONSE", "Done"),
            minimum_confidence=_number(environ, "DETERMINISTIC_MIN_CONFIDENCE", 1.0, minimum=0.0),
            error_phrases=tuple(
                phrase.strip().casefold()
                for phrase in environ.get(_name("DETERMINISTIC_ERROR_PHRASES"), "").split(";")
                if phrase.strip()
            ),
        ),
        llm=LlmSettings(
            enabled=_boolean(environ, "LLM_ENABLED", True),
            ambiguous_target_fallback_enabled=_boolean(
                environ,
                "LLM_AMBIGUOUS_TARGET_FALLBACK_ENABLED",
                True,
            ),
            base_url=_text(environ, "LLM_BASE_URL", "http://192.168.0.159:13305/v1").rstrip("/"),
            api_key=_text(environ, "LLM_API_KEY", "not-used"),
            model=_text(environ, "LLM_MODEL", ""),
            max_turns=_integer(environ, "LLM_MAX_TURNS", 6, minimum=1),
            timeout_seconds=_number(environ, "LLM_TIMEOUT_SECONDS", 120.0, minimum=0.1),
            data_recovery_enabled=_boolean(environ, "LLM_DATA_RECOVERY_ENABLED", True),
            data_recovery_max_chars=_integer(
                environ, "LLM_DATA_RECOVERY_MAX_CHARS", 16000, minimum=1000
            ),
        ),
        mcp=McpSettings(
            config_path=_path(environ, "MCP_CONFIG_PATH", Path("mcp.json")),
            connect_timeout_seconds=_number(
                environ, "MCP_CONNECT_TIMEOUT_SECONDS", 30.0, minimum=0.1
            ),
            cleanup_timeout_seconds=_number(
                environ, "MCP_CLEANUP_TIMEOUT_SECONDS", 10.0, minimum=0.1
            ),
        ),
        voice_origin=VoiceOriginSettings(
            enabled=_boolean(environ, "VOICE_ORIGIN_ENABLED", True),
            area_bias_enabled=_boolean(environ, "VOICE_ORIGIN_AREA_BIAS_ENABLED", True),
            soft_ranking_enabled=_boolean(environ, "VOICE_ORIGIN_SOFT_RANKING_ENABLED", True),
            system_prompt_fallback_enabled=_boolean(
                environ, "VOICE_ORIGIN_SYSTEM_PROMPT_FALLBACK_ENABLED", True
            ),
        ),
        home_assistant=HomeAssistantSettings(
            base_url=_text(environ, "HA_BASE_URL", ""),
            access_token=_text(environ, "HA_ACCESS_TOKEN", ""),
            websocket=HomeAssistantWebSocketSettings(
                enabled=_boolean(environ, "HA_WS_ENABLED", True),
                connect_timeout_seconds=_number(
                    environ, "HA_WS_CONNECT_TIMEOUT_SECONDS", 20.0, minimum=0.0
                ),
                command_timeout_seconds=_number(
                    environ, "HA_WS_COMMAND_TIMEOUT_SECONDS", 15.0, minimum=0.0
                ),
                reconnect_delay_seconds=_number(
                    environ, "HA_WS_RECONNECT_DELAY_SECONDS", 2.0, minimum=0.0
                ),
                service_cache_ttl_seconds=_number(
                    environ, "HA_SERVICE_CACHE_TTL_SECONDS", 600.0, minimum=0.0
                ),
                registry_cache_ttl_seconds=_number(
                    environ, "HA_REGISTRY_CACHE_TTL_SECONDS", 600.0, minimum=0.0
                ),
                state_confirm_timeout_seconds=_number(
                    environ, "HA_STATE_CONFIRM_TIMEOUT_SECONDS", 1.0, minimum=0.0
                ),
            ),
            advanced=HomeAssistantAdvancedSettings(
                enabled=_boolean(environ, "HA_ADVANCED_ENABLED", True),
                connect_timeout_seconds=_number(
                    environ, "HA_ADVANCED_CONNECT_TIMEOUT_SECONDS", 120.0, minimum=0.1
                ),
                cleanup_timeout_seconds=_number(
                    environ, "HA_ADVANCED_CLEANUP_TIMEOUT_SECONDS", 30.0, minimum=0.1
                ),
                max_turns=_integer(environ, "HA_ADVANCED_MAX_TURNS", 8, minimum=1),
                command=_text(environ, "HA_ADVANCED_COMMAND", "python3"),
                args=tuple(
                    shlex.split(
                        environ.get(
                            _name("HA_ADVANCED_ARGS"),
                            "-m uv tool run ha-mcp@latest",
                        )
                    )
                ),
                tool_search_enabled=_boolean(environ, "HA_ADVANCED_TOOL_SEARCH_ENABLED", True),
                tool_search_max_results=_integer(
                    environ, "HA_ADVANCED_TOOL_SEARCH_MAX_RESULTS", 5, minimum=1
                ),
                pinned_tools=_csv(
                    environ,
                    "HA_ADVANCED_PINNED_TOOLS",
                    "ha_search,ha_get_state,ha_call_service,ha_get_overview",
                ),
            ),
            schema_auto_repair_enabled=_boolean(environ, "HA_SCHEMA_AUTO_REPAIR_ENABLED", True),
            search_default_limit=ha_search_default,
            search_max_limit=ha_search_max,
            penalize_indicator_lights=_boolean(environ, "HA_PENALIZE_INDICATOR_LIGHTS", True),
            ignored_entity_domains=_csv(environ, "HA_IGNORED_ENTITY_DOMAINS", "update"),
            state_changing_services=frozenset(
                _csv(
                    environ,
                    "HA_STATE_CHANGING_SERVICES",
                    ",".join(sorted(HomeAssistantSettings().state_changing_services)),
                )
            ),
        ),
        music_assistant=MusicAssistantSettings(
            enabled=_boolean(environ, "MA_ENABLED", True),
            base_url=_text(environ, "MA_BASE_URL", ""),
            access_token=_text(environ, "MA_ACCESS_TOKEN", ""),
            area_player_map=_text(environ, "MA_AREA_PLAYER_MAP", ""),
            connect_timeout_seconds=_number(
                environ, "MA_CONNECT_TIMEOUT_SECONDS", 15.0, minimum=1.0
            ),
            command_timeout_seconds=_number(
                environ, "MA_COMMAND_TIMEOUT_SECONDS", 20.0, minimum=1.0
            ),
            reconnect_delay_seconds=_number(
                environ, "MA_RECONNECT_DELAY_SECONDS", 2.0, minimum=0.25
            ),
            search_default_limit=_integer(
                environ, "MA_SEARCH_DEFAULT_LIMIT", 10, minimum=1, maximum=50
            ),
            post_action_settle_seconds=_number(
                environ, "MA_POST_ACTION_SETTLE_SECONDS", 0.25, minimum=0.0
            ),
            play_ack_timeout_seconds=play_ack_timeout,
            play_completion_timeout_seconds=play_completion_timeout,
            first_audio_timeout_seconds=first_audio_timeout,
            first_audio_poll_seconds=_number(
                environ, "MA_FIRST_AUDIO_POLL_SECONDS", 0.2, minimum=0.05
            ),
            background_timeout_seconds=background_timeout,
            radio_seed_top_n=_integer(environ, "MA_RADIO_SEED_TOP_N", 8, minimum=1, maximum=50),
            radio_seed_strategy=radio_strategy,
            terminal_actions_enabled=_boolean(environ, "MA_TERMINAL_ACTIONS_ENABLED", True),
            replay_guard_seconds=_number(environ, "MA_REPLAY_GUARD_SECONDS", 4.0, minimum=0.0),
        ),
        assistant=AssistantSettings(
            led_enabled=_boolean(environ, "ASSISTANT_LED_ENABLED", True),
            led_domains=tuple(
                item.casefold()
                for item in _csv(environ, "ASSISTANT_LED_DOMAINS", "light,switch")
            ),
            led_color=_text(environ, "ASSISTANT_LED_COLOR", "green"),
            led_effect=_text(environ, "ASSISTANT_LED_EFFECT", "pulse"),
            led_software_pulse_enabled=_boolean(
                environ, "ASSISTANT_LED_SOFTWARE_PULSE_ENABLED", True
            ),
            led_pulse_interval_seconds=_number(
                environ, "ASSISTANT_LED_PULSE_INTERVAL_SECONDS", 0.7, minimum=0.2
            ),
            sounds_enabled=sounds_enabled,
        ),
        conversation=ConversationSettings(
            enabled=_boolean(environ, "CONVERSATION_ENABLED", True),
            history_turns=_integer(environ, "CONVERSATION_HISTORY_TURNS", 4, minimum=1),
            ttl_seconds=_number(environ, "CONVERSATION_TTL_SECONDS", 300.0, minimum=1.0),
            max_sessions=_integer(environ, "CONVERSATION_MAX_SESSIONS", 32, minimum=1),
        ),
    )


__all__ = ["ConfigurationError", "PREFIX", "load_settings"]
