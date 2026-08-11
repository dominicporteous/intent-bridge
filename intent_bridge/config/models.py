"""Typed configuration models aligned to package responsibilities."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class ApiSettings:
    version: str = "6.9.0"
    base_url: str = ""
    model_name: str = "home-intent"
    timezone: str = "Europe/London"
    spoken_response_max_chars: int = 180
    action_confirmation: str = ""
    voice_failure_response: str = "Sorry, I couldn't handle that request."
    log_level: str = "INFO"


@dataclass(slots=True)
class DeterministicSettings:
    language: str = "en"
    custom_sentences_path: Path = Path("custom_sentences/en")
    timeout_seconds: float = 10.0
    response_grace_seconds: float = 0.75
    default_response: str = "Done"
    minimum_confidence: float = 1.0
    error_phrases: tuple[str, ...] = ()


@dataclass(slots=True)
class LlmSettings:
    enabled: bool = True
    ambiguous_target_fallback_enabled: bool = True
    base_url: str = "http://192.168.0.159:13305/v1"
    api_key: str = field(default="not-used", repr=False)
    model: str = ""
    max_turns: int = 6
    timeout_seconds: float = 120.0
    data_recovery_enabled: bool = True
    data_recovery_max_chars: int = 16000


@dataclass(slots=True)
class McpSettings:
    config_path: Path = Path("mcp.json")
    connect_timeout_seconds: float = 30.0
    cleanup_timeout_seconds: float = 10.0


@dataclass(slots=True)
class VoiceOriginSettings:
    enabled: bool = True
    area_bias_enabled: bool = True
    soft_ranking_enabled: bool = True
    system_prompt_fallback_enabled: bool = True


@dataclass(slots=True)
class HomeAssistantWebSocketSettings:
    enabled: bool = True
    connect_timeout_seconds: float = 20.0
    command_timeout_seconds: float = 15.0
    reconnect_delay_seconds: float = 2.0
    service_cache_ttl_seconds: float = 600.0
    registry_cache_ttl_seconds: float = 600.0
    state_confirm_timeout_seconds: float = 1.0


@dataclass(slots=True)
class HomeAssistantAdvancedSettings:
    enabled: bool = True
    connect_timeout_seconds: float = 120.0
    cleanup_timeout_seconds: float = 30.0
    max_turns: int = 8
    command: str = "python3"
    args: tuple[str, ...] = ("-m", "uv", "tool", "run", "ha-mcp@latest")
    tool_search_enabled: bool = True
    tool_search_max_results: int = 5
    pinned_tools: tuple[str, ...] = (
        "ha_search",
        "ha_get_state",
        "ha_call_service",
        "ha_get_overview",
    )


@dataclass(slots=True)
class HomeAssistantSettings:
    base_url: str = ""
    access_token: str = field(default="", repr=False)
    websocket: HomeAssistantWebSocketSettings = field(
        default_factory=HomeAssistantWebSocketSettings
    )
    advanced: HomeAssistantAdvancedSettings = field(default_factory=HomeAssistantAdvancedSettings)
    schema_auto_repair_enabled: bool = True
    search_default_limit: int = 8
    search_max_limit: int = 20
    penalize_indicator_lights: bool = True
    ignored_entity_domains: tuple[str, ...] = ("update",)
    state_changing_services: frozenset[str] = frozenset(
        {
            "turn_on",
            "turn_off",
            "toggle",
            "open",
            "close",
            "open_cover",
            "close_cover",
            "lock",
            "unlock",
            "set_temperature",
            "set_hvac_mode",
            "set_fan_mode",
            "set_percentage",
            "set_preset_mode",
            "select_option",
            "set_value",
            "set_datetime",
            "set_cover_position",
            "set_position",
            "play_media",
            "media_play",
            "media_pause",
            "media_stop",
        }
    )


@dataclass(slots=True)
class MusicAssistantSettings:
    enabled: bool = True
    base_url: str = ""
    access_token: str = field(default="", repr=False)
    area_player_map: str = ""
    connect_timeout_seconds: float = 15.0
    command_timeout_seconds: float = 20.0
    reconnect_delay_seconds: float = 2.0
    search_default_limit: int = 10
    post_action_settle_seconds: float = 0.25
    play_ack_timeout_seconds: float = 2.0
    play_completion_timeout_seconds: float = 90.0
    first_audio_timeout_seconds: float = 15.0
    first_audio_poll_seconds: float = 0.2
    background_timeout_seconds: float = 300.0
    radio_seed_top_n: int = 8
    radio_seed_strategy: str = "weighted"
    terminal_actions_enabled: bool = True
    replay_guard_seconds: float = 4.0


@dataclass(slots=True)
class AssistantSettings:
    """Transport-neutral visual and audible assistant feedback settings."""

    led_enabled: bool = True
    led_domains: tuple[str, ...] = ("light", "switch")
    led_color: str = "green"
    led_effect: str = "pulse"
    led_software_pulse_enabled: bool = True
    led_pulse_interval_seconds: float = 0.7
    sounds_enabled: bool = False


@dataclass(slots=True)
class ConversationSettings:
    enabled: bool = True
    history_turns: int = 4
    ttl_seconds: float = 300.0
    max_sessions: int = 32


@dataclass(slots=True)
class BridgeSettings:
    api: ApiSettings = field(default_factory=ApiSettings)
    deterministic: DeterministicSettings = field(default_factory=DeterministicSettings)
    llm: LlmSettings = field(default_factory=LlmSettings)
    mcp: McpSettings = field(default_factory=McpSettings)
    voice_origin: VoiceOriginSettings = field(default_factory=VoiceOriginSettings)
    home_assistant: HomeAssistantSettings = field(default_factory=HomeAssistantSettings)
    music_assistant: MusicAssistantSettings = field(default_factory=MusicAssistantSettings)
    assistant: AssistantSettings = field(default_factory=AssistantSettings)
    conversation: ConversationSettings = field(default_factory=ConversationSettings)


__all__ = [
    "ApiSettings",
    "AssistantSettings",
    "BridgeSettings",
    "ConversationSettings",
    "DeterministicSettings",
    "HomeAssistantAdvancedSettings",
    "HomeAssistantSettings",
    "HomeAssistantWebSocketSettings",
    "LlmSettings",
    "McpSettings",
    "MusicAssistantSettings",
    "VoiceOriginSettings",
]
