import asyncio
import difflib
import json
import logging
import os
import random
import re
import shlex
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal
from urllib.parse import urlparse, urlunparse
from zoneinfo import ZoneInfo

import paho.mqtt.client as mqtt
import websockets
from agents import (
    Agent,
    AsyncOpenAI,
    FunctionToolResult,
    OpenAIChatCompletionsModel,
    RunContextWrapper,
    Runner,
    function_tool,
    set_tracing_disabled,
)
from agents.agent import ToolsToFinalOutputResult
from agents.exceptions import MaxTurnsExceeded
from agents.mcp import MCPServerManager, MCPServerStdio
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request

from intent_bridge import home_assistant as ha_domain
from intent_bridge import text as text_policy

try:
    from music_assistant_client import MusicAssistantClient
    from music_assistant_models.enums import MediaType, QueueOption, RepeatMode
    MUSIC_ASSISTANT_CLIENT_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - deployment dependency guard
    MusicAssistantClient = None
    MediaType = None
    QueueOption = None
    RepeatMode = None
    MUSIC_ASSISTANT_CLIENT_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

load_dotenv()

PROXY_VERSION = "6.9.0"

# Deterministic Home Intent / Rhasspy MQTT path.
MQTT_HOST = os.getenv("MQTT_HOST", "home-intent")
MQTT_PORT = int(os.getenv("MQTT_PORT", "12183"))
MQTT_USERNAME = os.getenv("MQTT_USERNAME")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD")

HOME_INTENT_TIMEOUT = float(os.getenv("HOME_INTENT_TIMEOUT", "10"))
HOME_INTENT_RESPONSE_GRACE = float(
    os.getenv("HOME_INTENT_RESPONSE_GRACE", "0.75")
)
HOME_INTENT_FALLBACK_RESPONSE = os.getenv(
    "HOME_INTENT_FALLBACK_RESPONSE", "Done"
)
HOME_INTENT_MIN_CONFIDENCE = float(
    os.getenv("HOME_INTENT_MIN_CONFIDENCE", "1.0")
)

MODEL_NAME = os.getenv("MODEL_NAME", "home-intent")
SITE_ID = os.getenv("SITE_ID", "default")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

HOME_INTENT_ERROR_PHRASES = tuple(
    phrase.strip().casefold()
    for phrase in os.getenv("HOME_INTENT_ERROR_PHRASES", "").split(";")
    if phrase.strip()
)

# Local LLM fallback through Lemonade.
MCP_FALLBACK_ENABLED = os.getenv("MCP_FALLBACK_ENABLED", "true").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
LEMONADE_BASE_URL = os.getenv(
    "LEMONADE_BASE_URL",
    "http://192.168.0.159:13305/v1",
).rstrip("/")
LEMONADE_API_KEY = os.getenv("LEMONADE_API_KEY", "not-used")
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "")
FALLBACK_MAX_TURNS = int(os.getenv("FALLBACK_MAX_TURNS", "6"))
FALLBACK_TIMEOUT = float(os.getenv("FALLBACK_TIMEOUT", "120"))

# Home Assistant connection used by the direct persistent WebSocket fast path.
HOMEASSISTANT_URL = os.getenv("HOMEASSISTANT_URL", "")
HOMEASSISTANT_TOKEN = os.getenv("HOMEASSISTANT_TOKEN", "")

HA_WS_ENABLED = os.getenv("HA_WS_ENABLED", "true").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
HA_WS_CONNECT_TIMEOUT = float(os.getenv("HA_WS_CONNECT_TIMEOUT", "20"))
HA_WS_COMMAND_TIMEOUT = float(os.getenv("HA_WS_COMMAND_TIMEOUT", "15"))
HA_WS_RECONNECT_DELAY = float(os.getenv("HA_WS_RECONNECT_DELAY", "2"))
HA_WS_SERVICE_CACHE_TTL = float(os.getenv("HA_WS_SERVICE_CACHE_TTL", "600"))
HA_WS_REGISTRY_CACHE_TTL = float(os.getenv("HA_WS_REGISTRY_CACHE_TTL", "600"))
HA_WS_SEARCH_DEFAULT_LIMIT = max(
    1, int(os.getenv("HA_WS_SEARCH_DEFAULT_LIMIT", "8"))
)
HA_WS_SEARCH_MAX_LIMIT = max(
    HA_WS_SEARCH_DEFAULT_LIMIT,
    int(os.getenv("HA_WS_SEARCH_MAX_LIMIT", "20")),
)
HA_WS_STATE_CONFIRM_TIMEOUT = max(
    0.0, float(os.getenv("HA_WS_STATE_CONFIRM_TIMEOUT", "1.0"))
)

# Schema-aware service validation/repair. These are local only: they avoid
# wasting HA round-trips and LLM turns on mechanically invalid service calls.
HA_SERVICE_SCHEMA_AUTO_REPAIR = os.getenv(
    "HA_SERVICE_SCHEMA_AUTO_REPAIR", "true"
).lower() in {"1", "true", "yes", "on"}

# Voice-satellite origin context. When the caller forwards a Home Assistant
# device/area identifier, unqualified requests are softly biased toward that
# room. Explicit user room names always win.
VOICE_ORIGIN_CONTEXT_ENABLED = os.getenv(
    "VOICE_ORIGIN_CONTEXT_ENABLED", "true"
).lower() in {"1", "true", "yes", "on"}
VOICE_ORIGIN_AREA_BIAS = os.getenv(
    "VOICE_ORIGIN_AREA_BIAS", "true"
).lower() in {"1", "true", "yes", "on"}

# Voice origin should bias ranking, not hide entities that are named for the
# room but have no explicit HA area assignment.
VOICE_ORIGIN_SOFT_AREA_RANKING = os.getenv(
    "VOICE_ORIGIN_SOFT_AREA_RANKING", "true"
).lower() in {"1", "true", "yes", "on"}

# Generic room-light commands should not select status/ring/indicator LEDs.
GENERIC_LIGHT_INDICATOR_PENALTY = os.getenv(
    "GENERIC_LIGHT_INDICATOR_PENALTY", "true"
).lower() in {"1", "true", "yes", "on"}

# Home Assistant's OpenAI-compatible conversation caller may not forward a
# structured device_id/area_id. It does, however, include a trusted system line
# such as:
#   "You are in area Office (floor First) and all generic commands..."
# Parse that as a fallback room hint. System messages are never copied into the
# Lemonade conversation history, so HA's large static-context prompt is not
# forwarded to the local model.
VOICE_ORIGIN_SYSTEM_PROMPT_FALLBACK = os.getenv(
    "VOICE_ORIGIN_SYSTEM_PROMPT_FALLBACK", "true"
).lower() in {"1", "true", "yes", "on"}

# If Home Assistant already returned useful data but the model exhausts its
# tool-loop turns or produces an empty final message, make one final no-tools
# Lemonade summarisation attempt rather than returning HTTP 502.
DATA_RESPONSE_RECOVERY_ENABLED = os.getenv(
    "DATA_RESPONSE_RECOVERY_ENABLED", "true"
).lower() in {"1", "true", "yes", "on"}
DATA_RESPONSE_RECOVERY_MAX_CHARS = max(
    1000, int(os.getenv("DATA_RESPONSE_RECOVERY_MAX_CHARS", "16000"))
)

# Optional full ha-mcp specialist. It is NOT exposed directly to the normal
# voice agent. The main agent sees one function tool, ha_advanced, which invokes
# this specialist only for genuinely advanced/admin/history/configuration work.
HA_MCP_ADVANCED_ENABLED = os.getenv(
    "HA_MCP_ADVANCED_ENABLED", "true"
).lower() in {"1", "true", "yes", "on"}
MCP_CONNECT_TIMEOUT = float(os.getenv("MCP_CONNECT_TIMEOUT", "120"))
MCP_CLEANUP_TIMEOUT = float(os.getenv("MCP_CLEANUP_TIMEOUT", "30"))
ADVANCED_MAX_TURNS = int(os.getenv("ADVANCED_MAX_TURNS", "8"))

HA_MCP_COMMAND = os.getenv("HA_MCP_COMMAND", "python3")
HA_MCP_ARGS = shlex.split(
    os.getenv("HA_MCP_ARGS", "-m uv tool run ha-mcp@latest")
)

# Tool-search is useful for the advanced specialist because ha-mcp has a large
# administrative tool surface. The normal voice agent does not see that surface.
HA_MCP_TOOL_SEARCH = os.getenv("HA_MCP_TOOL_SEARCH", "true").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
HA_MCP_TOOL_SEARCH_MAX_RESULTS = os.getenv("HA_MCP_TOOL_SEARCH_MAX_RESULTS", "5")
HA_MCP_PINNED_TOOLS = os.getenv(
    "HA_MCP_PINNED_TOOLS",
    "ha_search,ha_get_state,ha_call_service,ha_get_overview",
)

# Native Music Assistant WebSocket integration. v6.8.2 keeps Music Assistant
# on the official client, treats long play_media operations optimistically at the
# voice boundary, deduplicates in-flight playback, and can drive an activity LED
# belonging to the originating Home Assistant assist_satellite device.
# The client keeps player and queue topology warm from WebSocket events.
MUSIC_ASSISTANT_ENABLED = os.getenv(
    "MUSIC_ASSISTANT_ENABLED", "true"
).lower() in {"1", "true", "yes", "on"}
MUSIC_ASSISTANT_URL = os.getenv("MUSIC_ASSISTANT_URL", "").strip()
MUSIC_ASSISTANT_TOKEN = os.getenv("MUSIC_ASSISTANT_TOKEN", "").strip()

# Optional explicit HA-area -> Music Assistant player ID mapping.
# Format: Office=apb0f7c4c29cd1,Kitchen=kitchen
MUSIC_ASSISTANT_AREA_PLAYER_MAP = os.getenv(
    "MUSIC_ASSISTANT_AREA_PLAYER_MAP", ""
).strip()
MUSIC_ASSISTANT_CONNECT_TIMEOUT_SECONDS = max(
    1.0, float(os.getenv("MUSIC_ASSISTANT_CONNECT_TIMEOUT_SECONDS", "15"))
)
MUSIC_ASSISTANT_COMMAND_TIMEOUT_SECONDS = max(
    1.0, float(os.getenv("MUSIC_ASSISTANT_COMMAND_TIMEOUT_SECONDS", "20"))
)
# player_queues/play_media on MA 2.8.x is intentionally long-running: the server
# may resolve an artist/playlist, build the queue, resolve stream details and
# prepare the audio buffer before returning its command result. For voice use we
# only wait briefly for an immediate ACK/error; if the server is still preparing
# playback, the command future remains alive and its eventual result is logged.
MUSIC_ASSISTANT_PLAY_ACK_WAIT_SECONDS = max(
    0.1, float(os.getenv("MUSIC_ASSISTANT_PLAY_ACK_WAIT_SECONDS", "2.0"))
)
MUSIC_ASSISTANT_PLAY_COMPLETION_TIMEOUT_SECONDS = max(
    MUSIC_ASSISTANT_PLAY_ACK_WAIT_SECONDS,
    float(os.getenv("MUSIC_ASSISTANT_PLAY_COMPLETION_TIMEOUT_SECONDS", "90")),
)

# Optional visual acknowledgement while a long-running Music Assistant play_media
# command is still preparing. Resolution is deterministic: origin device -> its
# assist_satellite entity -> same-device indicator sibling; if only an origin area
# is known, exactly one assist_satellite in that area may be used. No confident
# match means no visual feedback and never blocks/fails the music request.
MUSIC_ASSISTANT_ACTIVITY_INDICATOR_ENABLED = os.getenv(
    "MUSIC_ASSISTANT_ACTIVITY_INDICATOR_ENABLED", "true"
).lower() in {"1", "true", "yes", "on"}
# v6.8.2: configurable visual state. COLOR accepts a common colour name,
# #RRGGBB, or R,G,B. Empty/none/current preserves the indicator's current
# colour while the activity state is active. EFFECT is matched against the
# light's advertised effect_list (case-insensitive, then shortest containing
# match). Empty/none disables a native effect. The legacy GREEN flag remains a
# backwards-compatible default only when COLOR is not explicitly configured.
MUSIC_ASSISTANT_ACTIVITY_INDICATOR_GREEN = os.getenv(
    "MUSIC_ASSISTANT_ACTIVITY_INDICATOR_GREEN", "true"
).lower() in {"1", "true", "yes", "on"}
_indicator_color_env = os.getenv("MUSIC_ASSISTANT_ACTIVITY_INDICATOR_COLOR")
MUSIC_ASSISTANT_ACTIVITY_INDICATOR_COLOR = (
    _indicator_color_env.strip()
    if _indicator_color_env is not None
    else ("green" if MUSIC_ASSISTANT_ACTIVITY_INDICATOR_GREEN else "")
)
MUSIC_ASSISTANT_ACTIVITY_INDICATOR_EFFECT = os.getenv(
    "MUSIC_ASSISTANT_ACTIVITY_INDICATOR_EFFECT", "pulse"
).strip()
MUSIC_ASSISTANT_ACTIVITY_INDICATOR_SOFTWARE_PULSE = os.getenv(
    "MUSIC_ASSISTANT_ACTIVITY_INDICATOR_SOFTWARE_PULSE", "true"
).lower() in {"1", "true", "yes", "on"}
MUSIC_ASSISTANT_ACTIVITY_INDICATOR_PULSE_INTERVAL_SECONDS = max(
    0.2, float(os.getenv("MUSIC_ASSISTANT_ACTIVITY_INDICATOR_PULSE_INTERVAL_SECONDS", "0.7"))
)
MUSIC_ASSISTANT_ACTIVITY_INDICATOR_DOMAINS = tuple(
    item.strip().casefold()
    for item in os.getenv("MUSIC_ASSISTANT_ACTIVITY_INDICATOR_DOMAINS", "light,switch").split(",")
    if item.strip()
)
MUSIC_ASSISTANT_RECONNECT_DELAY_SECONDS = max(
    0.25, float(os.getenv("MUSIC_ASSISTANT_RECONNECT_DELAY_SECONDS", "2"))
)
MUSIC_ASSISTANT_POST_ACTION_SETTLE_SECONDS = max(
    0.0, float(os.getenv("MUSIC_ASSISTANT_POST_ACTION_SETTLE_SECONDS", "0.25"))
)
MUSIC_ASSISTANT_SEARCH_DEFAULT_LIMIT = max(
    1, min(50, int(os.getenv("MUSIC_ASSISTANT_SEARCH_DEFAULT_LIMIT", "10")))
)

# v6.9 radio fast-start policy. Voice radio requests deliberately optimize for
# time-to-first-audio on every supported MA schema: resolve an artist, ask MA for
# its top tracks, REPLACE the queue with one weighted-random seed track, then
# enable MA Don't Stop The Music only after that seed is audibly playing. MA
# remains authoritative for provider lookup, queue state and radio continuation.
MUSIC_ASSISTANT_FIRST_AUDIO_TIMEOUT_SECONDS = max(
    1.0, float(os.getenv("MUSIC_ASSISTANT_FIRST_AUDIO_TIMEOUT_SECONDS", "15"))
)
MUSIC_ASSISTANT_BACKGROUND_TIMEOUT_SECONDS = max(
    MUSIC_ASSISTANT_FIRST_AUDIO_TIMEOUT_SECONDS,
    float(os.getenv("MUSIC_ASSISTANT_BACKGROUND_TIMEOUT_SECONDS", "300")),
)
MUSIC_ASSISTANT_FIRST_AUDIO_POLL_SECONDS = max(
    0.05, float(os.getenv("MUSIC_ASSISTANT_FIRST_AUDIO_POLL_SECONDS", "0.20"))
)
MUSIC_ASSISTANT_RADIO_SEED_TOP_N = max(
    1, min(50, int(os.getenv("MUSIC_ASSISTANT_RADIO_SEED_TOP_N", "8")))
)
MUSIC_ASSISTANT_RADIO_SEED_STRATEGY = os.getenv(
    "MUSIC_ASSISTANT_RADIO_SEED_STRATEGY", "weighted"
).strip().casefold()
if MUSIC_ASSISTANT_RADIO_SEED_STRATEGY not in {"weighted", "random", "first"}:
    MUSIC_ASSISTANT_RADIO_SEED_STRATEGY = "weighted"

# Successful Music Assistant writes are terminal. This avoids a second model
# turn after a confirmed server acknowledgement and prevents client retries from
# replaying a state-changing media action.
MUSIC_ASSISTANT_TERMINAL_ACTIONS = os.getenv(
    "MUSIC_ASSISTANT_TERMINAL_ACTIONS", "true"
).lower() in {"1", "true", "yes", "on"}
MUSIC_ASSISTANT_REPLAY_GUARD_SECONDS = float(
    os.getenv("MUSIC_ASSISTANT_REPLAY_GUARD_SECONDS", "4.0")
)

FAST_ACTION_RESPONSE = os.getenv("FAST_ACTION_RESPONSE", "Done.").strip() or "Done."
FAST_STATE_CHANGING_SERVICES = {
    name.strip()
    for name in os.getenv(
        "FAST_STATE_CHANGING_SERVICES",
        (
            "turn_on,turn_off,toggle,open,close,open_cover,close_cover,"
            "lock,unlock,set_temperature,set_hvac_mode,set_fan_mode,"
            "set_percentage,set_preset_mode,select_option,set_value,"
            "set_datetime,set_cover_position,set_position,play_media,"
            "media_play,media_pause,media_stop"
        ),
    ).split(",")
    if name.strip()
}

# Expected primary states for a small subset of commands. These are only used
# as a very short post-call confirmation when the state cache can verify them.
EXPECTED_PRIMARY_STATES = {
    "turn_on": "on",
    "turn_off": "off",
    "open": "open",
    "close": "closed",
    "open_cover": "open",
    "close_cover": "closed",
    "lock": "locked",
    "unlock": "unlocked",
    "media_play": "playing",
    "media_pause": "paused",
    "media_stop": "idle",
}

SPOKEN_MAX_CHARS = int(os.getenv("SPOKEN_MAX_CHARS", "180"))

# Short in-process conversation memory for quick spoken follow-ups.
CONVERSATION_HISTORY_ENABLED = os.getenv(
    "CONVERSATION_HISTORY_ENABLED", "true"
).lower() in {"1", "true", "yes", "on"}
CONVERSATION_HISTORY_TURNS = max(
    1, int(os.getenv("CONVERSATION_HISTORY_TURNS", "4"))
)
CONVERSATION_HISTORY_TTL_SECONDS = max(
    1.0, float(os.getenv("CONVERSATION_HISTORY_TTL_SECONDS", "300"))
)
CONVERSATION_HISTORY_MAX_SESSIONS = max(
    1, int(os.getenv("CONVERSATION_HISTORY_MAX_SESSIONS", "32"))
)
LOCAL_TIMEZONE = os.getenv("LOCAL_TIMEZONE", "Europe/London").strip() or "Europe/London"

# The SDK tracing backend is OpenAI-specific by default. This application uses
# a local OpenAI-compatible endpoint.
set_tracing_disabled(True)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("home-intent-proxy")


event_loop: asyncio.AbstractEventLoop | None = None
mqtt_ready: asyncio.Event | None = None
fallback_lock = asyncio.Lock()

mcp_manager: MCPServerManager | None = None
advanced_agent: Agent | None = None
fallback_agent: Agent | None = None

# v6.8.2 native Music Assistant runtime. Typed as Any here because the dependency
# is optional at import time; the concrete class is defined below.
music_assistant_native: Any | None = None


@dataclass
class VoiceToolRunState:
    request_text: str = ""
    origin_device_id: str | None = None
    origin_device_name: str | None = None
    origin_area_id: str | None = None
    origin_area_name: str | None = None
    origin_floor_name: str | None = None
    origin_source: str | None = None
    last_entity_by_domain: dict[str, str] | None = None
    last_area_id: str | None = None
    last_service_call: dict[str, Any] | None = None
    last_successful_data: dict[str, Any] | None = None
    last_successful_music_action: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.last_entity_by_domain is None:
            self.last_entity_by_domain = {}


voice_tool_run_state = VoiceToolRunState()


def _reset_voice_tool_run_state(
    request_text: str,
    origin_context: dict[str, Any] | None = None,
) -> None:
    global voice_tool_run_state
    origin_context = origin_context or {}
    voice_tool_run_state = VoiceToolRunState(
        request_text=request_text.strip(),
        origin_device_id=origin_context.get("device_id"),
        origin_device_name=origin_context.get("device_name"),
        origin_area_id=origin_context.get("area_id"),
        origin_area_name=origin_context.get("area_name"),
        origin_floor_name=origin_context.get("floor_name"),
        origin_source=origin_context.get("source"),
    )


@dataclass
class PendingRequest:
    response_future: asyncio.Future[str]
    intent_future: asyncio.Future[dict]
    original_text: str
    normalized_text: str
    request_id: str
    created_at: float


pending: dict[str, PendingRequest] = {}


@dataclass
class ConversationMemory:
    messages: list[dict[str, str]]
    updated_at: float


conversation_memories: dict[str, ConversationMemory] = {}
conversation_history_lock = asyncio.Lock()

# Successful Music Assistant action responses retained only for the very short
# replay-guard window. Access occurs under fallback_lock.
recent_music_action_responses: dict[str, tuple[float, str]] = {}

def normalize_command(text: str) -> str:
    """Normalise STT formatting without semantic rewriting."""
    return text_policy.normalize_command(text)

    text = text.strip().lower().replace("’", "'")
    text = re.sub(r"[.!?,;:]+$", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()



def is_home_intent_error_response(text: str) -> bool:
    if not HOME_INTENT_ERROR_PHRASES:
        return False
    normalized = text.strip().casefold()
    return any(phrase in normalized for phrase in HOME_INTENT_ERROR_PHRASES)



def _normalise_search_text(value: Any) -> str:
    text = str(value or "").casefold().replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9. ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()



def _ha_websocket_url(base_url: str) -> str:
    return ha_domain.websocket_url(base_url)

    parsed = urlparse(base_url.strip())
    if parsed.scheme not in {"http", "https", "ws", "wss"}:
        raise ValueError("HOMEASSISTANT_URL must use http:// or https://")
    scheme = "wss" if parsed.scheme in {"https", "wss"} else "ws"
    base_path = parsed.path.rstrip("/")
    path = f"{base_path}/api/websocket"
    return urlunparse((scheme, parsed.netloc, path, "", "", ""))



def _json_tool_result(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)



def _compact_attributes(attributes: dict[str, Any]) -> dict[str, Any]:
    """Keep useful state attributes while avoiding giant lists/config blobs."""
    compact: dict[str, Any] = {}
    for key, value in attributes.items():
        if key in {"attribution"}:
            continue
        if isinstance(value, str):
            compact[key] = value if len(value) <= 500 else value[:497] + "..."
        elif isinstance(value, (int, float, bool)) or value is None:
            compact[key] = value
        elif isinstance(value, list):
            if len(value) <= 20:
                compact[key] = value
        elif isinstance(value, dict):
            if len(value) <= 20:
                compact[key] = value
    return compact


def _truncate_text(value: Any, limit: int = 220) -> str | None:
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    if not text:
        return None
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def _selector_allowed_values(field_definition: dict[str, Any]) -> list[Any]:
    """Extract simple enum/select values from a Home Assistant field schema."""
    selector = field_definition.get("selector")
    if not isinstance(selector, dict):
        return []

    select = selector.get("select")
    if not isinstance(select, dict):
        return []

    options = select.get("options")
    if not isinstance(options, list):
        return []

    values: list[Any] = []
    for option in options:
        if isinstance(option, dict):
            if "value" in option:
                values.append(option["value"])
            elif "label" in option:
                values.append(option["label"])
        elif isinstance(option, (str, int, float, bool)):
            values.append(option)
    return values[:50]


def _target_entity_domains(target_definition: Any) -> list[str]:
    if not isinstance(target_definition, dict):
        return []
    entity = target_definition.get("entity")
    if not isinstance(entity, dict):
        return []
    domain = entity.get("domain")
    if isinstance(domain, str):
        return [domain]
    if isinstance(domain, list):
        return [str(item) for item in domain if isinstance(item, str)]
    return []


def _compact_service_definition(
    domain: str,
    service: str,
    definition: dict[str, Any],
) -> dict[str, Any]:
    """Project HA's UI-oriented service schema into a compact tool-call schema."""
    fields = definition.get("fields")
    if not isinstance(fields, dict):
        fields = {}

    parameters: dict[str, Any] = {}
    for field_name, raw_definition in fields.items():
        field_def = raw_definition if isinstance(raw_definition, dict) else {}
        item: dict[str, Any] = {
            "required": bool(field_def.get("required", False)),
        }
        allowed = _selector_allowed_values(field_def)
        if allowed:
            item["allowed"] = allowed
        if "default" in field_def:
            item["default"] = field_def.get("default")
        if "example" in field_def:
            item["example"] = field_def.get("example")
        description = _truncate_text(field_def.get("description"), 180)
        if description:
            item["description"] = description
        parameters[str(field_name)] = item

    target = definition.get("target")
    entity_domains = _target_entity_domains(target)

    result: dict[str, Any] = {
        "domain": domain,
        "service": service,
        "name": definition.get("name"),
        "description": _truncate_text(definition.get("description"), 240),
        "target": {
            "required": bool(target),
            "entity_domains": entity_domains,
            "accepts_entity_id": bool(target),
        },
        "parameters": parameters,
        "required_parameters": [
            key
            for key, value in parameters.items()
            if isinstance(value, dict) and value.get("required") is True
        ],
        "returns_data": bool(definition.get("response")),
    }
    return result


def _get_cached_service_definition(
    client: "HomeAssistantWebSocket",
    domain: str,
    service: str,
) -> dict[str, Any] | None:
    domain_services = client.services.get(domain)
    if not isinstance(domain_services, dict):
        return None
    definition = domain_services.get(service)
    return definition if isinstance(definition, dict) else None


def _normalise_service_data_from_schema(
    domain: str,
    service: str,
    definition: dict[str, Any] | None,
    supplied_data: dict[str, Any] | None,
    previous_data: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[str], dict[str, Any] | None]:
    """Validate/repair service_data without asking the model to rediscover basics.

    Safe repair is deliberately narrow:
    - preserve previously-valid schema keys omitted by a retry;
    - if exactly one unknown key and one required key is missing, rename the
    unknown key only when its value is valid for the missing field.
    """
    return ha_domain.normalise_service_data(
        domain,
        service,
        definition,
        supplied_data,
        previous_data,
        auto_repair=HA_SERVICE_SCHEMA_AUTO_REPAIR,
    )

    data = dict(supplied_data or {})
    repairs: list[str] = []

    if not isinstance(definition, dict):
        return data, repairs, None

    fields = definition.get("fields")
    if not isinstance(fields, dict):
        fields = {}

    # Retry preservation: keep earlier valid parameters unless the new call
    # explicitly replaces them. Unknown earlier keys are intentionally not kept.
    if isinstance(previous_data, dict):
        for key, value in previous_data.items():
            if key in fields and key not in data:
                data[key] = value
                repairs.append(f"preserved previous valid field '{key}'")

    required = [
        str(name)
        for name, field_def in fields.items()
        if isinstance(field_def, dict) and field_def.get("required") is True
    ]
    unknown = [str(key) for key in data if key not in fields]
    missing = [name for name in required if name not in data]

    if (
        HA_SERVICE_SCHEMA_AUTO_REPAIR
        and len(unknown) == 1
        and len(missing) == 1
    ):
        unknown_key = unknown[0]
        missing_key = missing[0]
        field_def = fields.get(missing_key)
        field_def = field_def if isinstance(field_def, dict) else {}
        allowed = _selector_allowed_values(field_def)
        candidate = data.get(unknown_key)

        key_related = (
            unknown_key.casefold().endswith("_" + missing_key.casefold())
            or missing_key.casefold() in unknown_key.casefold()
            or difflib.SequenceMatcher(
                None, unknown_key.casefold(), missing_key.casefold()
            ).ratio() >= 0.55
        )
        value_is_strong_match = bool(allowed) and candidate in allowed

        if value_is_strong_match and key_related:
            data[missing_key] = data.pop(unknown_key)
            repairs.append(
                f"renamed invalid field '{unknown_key}' to required field "
                f"'{missing_key}' because its value matches the allowed schema"
            )

    unknown = [str(key) for key in data if key not in fields]
    missing = [name for name in required if name not in data]
    invalid_values: dict[str, Any] = {}
    for key, value in data.items():
        field_def = fields.get(key)
        if not isinstance(field_def, dict):
            continue
        allowed = _selector_allowed_values(field_def)
        if allowed and value not in allowed:
            invalid_values[key] = {
                "value": value,
                "allowed": allowed,
            }

    if unknown or missing or invalid_values:
        compact = _compact_service_definition(domain, service, definition)
        validation = {
            "unknown_parameters": unknown,
            "missing_required_parameters": missing,
            "invalid_values": invalid_values,
            "service_schema": compact,
        }
        return data, repairs, validation

    return data, repairs, None


def _single_cached_entity_for_domain(
    client: "HomeAssistantWebSocket",
    domain: str,
) -> str | None:
    matches = [
        entity_id
        for entity_id in client.states
        if entity_id.split(".", 1)[0].casefold() == domain.casefold()
    ]
    return matches[0] if len(matches) == 1 else None


class HomeAssistantWebSocket:

    def __init__(self, base_url: str, token: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.ws_url = _ha_websocket_url(base_url)

        self.ready = asyncio.Event()
        self._stopping = asyncio.Event()
        self._ws = None
        self._supervisor_task: asyncio.Task | None = None
        self._reader_task: asyncio.Task | None = None
        self._send_lock = asyncio.Lock()
        self._request_id = 0
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}

        self.states: dict[str, dict[str, Any]] = {}
        self.services: dict[str, Any] = {}
        self.entity_registry: dict[str, dict[str, Any]] = {}
        self.devices: dict[str, dict[str, Any]] = {}
        self.areas: dict[str, dict[str, Any]] = {}

        self._services_loaded_at = 0.0
        self._registries_loaded_at = 0.0
        self._service_refresh_lock = asyncio.Lock()
        self._registry_refresh_lock = asyncio.Lock()

        self.connected_at: float | None = None
        self.reconnect_count = 0
        self.last_error: str | None = None
        self.state_event_count = 0

    async def start(self) -> None:
        if self._supervisor_task is not None:
            return
        self._stopping.clear()
        self._supervisor_task = asyncio.create_task(
            self._supervisor(), name="ha-ws-supervisor"
        )
        try:
            await asyncio.wait_for(
                self.ready.wait(), timeout=HA_WS_CONNECT_TIMEOUT
            )
        except asyncio.TimeoutError as exc:
            raise RuntimeError(
                f"Timed out connecting to Home Assistant WebSocket {self.ws_url}"
            ) from exc

    async def stop(self) -> None:
        self._stopping.set()
        self.ready.clear()

        ws = self._ws
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                pass

        task = self._supervisor_task
        self._supervisor_task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                log.exception("HA WebSocket supervisor shutdown error")

        self._fail_pending(RuntimeError("Home Assistant WebSocket stopped"))

    async def _supervisor(self) -> None:
        while not self._stopping.is_set():
            try:
                await self._run_connection()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.last_error = str(exc)
                self.ready.clear()
                log.warning("HA WebSocket disconnected/error: %s", exc)

            if self._stopping.is_set():
                break

            await asyncio.sleep(HA_WS_RECONNECT_DELAY)

    async def _run_connection(self) -> None:
        log.info("Connecting Home Assistant WebSocket %s", self.ws_url)

        async with websockets.connect(
            self.ws_url,
            open_timeout=HA_WS_CONNECT_TIMEOUT,
            ping_interval=20,
            ping_timeout=20,
            close_timeout=5,
            max_size=32 * 1024 * 1024,
        ) as ws:
            auth_required = json.loads(await ws.recv())
            if auth_required.get("type") != "auth_required":
                raise RuntimeError(
                    f"Unexpected HA WebSocket auth message: {auth_required}"
                )

            await ws.send(
                json.dumps({"type": "auth", "access_token": self.token})
            )
            auth_reply = json.loads(await ws.recv())
            if auth_reply.get("type") != "auth_ok":
                raise RuntimeError(
                    f"Home Assistant WebSocket authentication failed: {auth_reply}"
                )

            self._ws = ws
            if self.connected_at is not None:
                self.reconnect_count += 1
            self.connected_at = time.time()
            self.last_error = None
            self._reader_task = asyncio.create_task(
                self._reader_loop(ws), name="ha-ws-reader"
            )

            try:
                await self._initialise_connection_caches()
                self.ready.set()
                log.info(
                    "HA WebSocket ready states=%d service_domains=%d entities=%d",
                    len(self.states),
                    len(self.services),
                    len(self.entity_registry),
                )
                await self._reader_task
            finally:
                self.ready.clear()
                self._ws = None
                self._fail_pending(
                    ConnectionError("Home Assistant WebSocket connection closed")
                )

    async def _reader_loop(self, ws) -> None:
        async for raw in ws:
            try:
                message = json.loads(raw)
            except Exception:
                log.debug("Ignoring non-JSON HA WebSocket frame")
                continue

            message_type = message.get("type")
            message_id = message.get("id")

            if message_type == "result" and isinstance(message_id, int):
                future = self._pending.pop(message_id, None)
                if future is not None and not future.done():
                    future.set_result(message)
                continue

            if message_type == "event":
                self._handle_event(message)
                continue

    def _handle_event(self, message: dict[str, Any]) -> None:
        event = message.get("event")
        if not isinstance(event, dict):
            return

        event_type = event.get("event_type")
        data = event.get("data")
        if not isinstance(data, dict):
            data = {}

        if event_type == "state_changed":
            entity_id = data.get("entity_id")
            if not isinstance(entity_id, str):
                return
            new_state = data.get("new_state")
            if isinstance(new_state, dict):
                self.states[entity_id] = new_state
            else:
                self.states.pop(entity_id, None)
            self.state_event_count += 1
            return

        if event_type in {"service_registered", "service_removed"}:
            self._services_loaded_at = 0.0
            return

        if event_type in {
            "entity_registry_updated",
            "device_registry_updated",
            "area_registry_updated",
        }:
            self._registries_loaded_at = 0.0

    def _fail_pending(self, exc: Exception) -> None:
        for request_id, future in list(self._pending.items()):
            if not future.done():
                future.set_exception(exc)
            self._pending.pop(request_id, None)

    async def _send_current(
        self,
        payload: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        ws = self._ws
        if ws is None:
            raise ConnectionError("Home Assistant WebSocket is not connected")

        loop = asyncio.get_running_loop()
        async with self._send_lock:
            self._request_id += 1
            request_id = self._request_id
            message = {"id": request_id, **payload}
            future: asyncio.Future[dict[str, Any]] = loop.create_future()
            self._pending[request_id] = future
            try:
                await ws.send(json.dumps(message))
            except Exception:
                self._pending.pop(request_id, None)
                raise

        try:
            return await asyncio.wait_for(
                future,
                timeout=timeout or HA_WS_COMMAND_TIMEOUT,
            )
        except Exception:
            self._pending.pop(request_id, None)
            raise

    async def command(
        self,
        payload: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        # Wait for an established/reconnected socket. One retry handles the
        # narrow race where the connection drops between ready.wait and send.
        for attempt in range(2):
            await asyncio.wait_for(
                self.ready.wait(), timeout=timeout or HA_WS_COMMAND_TIMEOUT
            )
            try:
                return await self._send_current(payload, timeout=timeout)
            except (ConnectionError, websockets.ConnectionClosed):
                self.ready.clear()
                if attempt == 1:
                    raise
                await asyncio.sleep(0)
        raise RuntimeError("Home Assistant WebSocket command failed")

    @staticmethod
    def _require_success(message: dict[str, Any], operation: str) -> Any:
        if message.get("success") is True:
            return message.get("result")
        error = message.get("error")
        raise RuntimeError(f"HA WebSocket {operation} failed: {error}")

    async def _initialise_connection_caches(self) -> None:
        states_reply = await self._send_current({"type": "get_states"})
        states = self._require_success(states_reply, "get_states")
        if not isinstance(states, list):
            raise RuntimeError("HA get_states returned an unexpected payload")
        self.states = {
            item["entity_id"]: item
            for item in states
            if isinstance(item, dict) and isinstance(item.get("entity_id"), str)
        }

        services_reply = await self._send_current({"type": "get_services"})
        services = self._require_success(services_reply, "get_services")
        self.services = services if isinstance(services, dict) else {}
        self._services_loaded_at = time.monotonic()

        await self._refresh_registries_current()

        # Subscriptions are cheap and keep normal state reads network-free.
        for event_type in (
            "state_changed",
            "service_registered",
            "service_removed",
            "entity_registry_updated",
            "device_registry_updated",
            "area_registry_updated",
        ):
            reply = await self._send_current(
                {"type": "subscribe_events", "event_type": event_type}
            )
            if reply.get("success") is not True:
                log.debug(
                    "HA event subscription unavailable event_type=%s error=%s",
                    event_type,
                    reply.get("error"),
                )

    async def refresh_services(self, *, force: bool = False) -> None:
        if not force and (
            time.monotonic() - self._services_loaded_at
            < HA_WS_SERVICE_CACHE_TTL
        ):
            return
        async with self._service_refresh_lock:
            if not force and (
                time.monotonic() - self._services_loaded_at
                < HA_WS_SERVICE_CACHE_TTL
            ):
                return
            reply = await self.command({"type": "get_services"})
            services = self._require_success(reply, "get_services")
            if isinstance(services, dict):
                self.services = services
                self._services_loaded_at = time.monotonic()

    async def refresh_registries(self, *, force: bool = False) -> None:
        if not force and (
            time.monotonic() - self._registries_loaded_at
            < HA_WS_REGISTRY_CACHE_TTL
        ):
            return
        async with self._registry_refresh_lock:
            if not force and (
                time.monotonic() - self._registries_loaded_at
                < HA_WS_REGISTRY_CACHE_TTL
            ):
                return
            await self._refresh_registries_via_command()

    async def _refresh_registries_current(self) -> None:
        async def send(payload: dict[str, Any]) -> dict[str, Any]:
            return await self._send_current(payload)
        await self._refresh_registries(send)

    async def _refresh_registries_via_command(self) -> None:
        async def send(payload: dict[str, Any]) -> dict[str, Any]:
            return await self.command(payload)
        await self._refresh_registries(send)

    async def _refresh_registries(self, sender) -> None:
        # entity_registry/list_for_display is documented and compact. The area
        # and device list commands are long-standing frontend WebSocket APIs; if
        # either is unavailable we simply search without that decoration.
        try:
            reply = await sender({"type": "config/entity_registry/list_for_display"})
            result = self._require_success(
                reply, "config/entity_registry/list_for_display"
            )
            entries = result.get("entities", []) if isinstance(result, dict) else []
            self.entity_registry = {
                item["ei"]: item
                for item in entries
                if isinstance(item, dict) and isinstance(item.get("ei"), str)
            }
        except Exception as exc:
            log.debug("Entity display registry cache unavailable: %s", exc)

        try:
            reply = await sender({"type": "config/device_registry/list"})
            result = self._require_success(reply, "config/device_registry/list")
            if isinstance(result, list):
                self.devices = {
                    item["id"]: item
                    for item in result
                    if isinstance(item, dict) and isinstance(item.get("id"), str)
                }
        except Exception as exc:
            log.debug("Device registry cache unavailable: %s", exc)

        try:
            reply = await sender({"type": "config/area_registry/list"})
            result = self._require_success(reply, "config/area_registry/list")
            if isinstance(result, list):
                self.areas = {
                    item["area_id"]: item
                    for item in result
                    if isinstance(item, dict)
                    and isinstance(item.get("area_id"), str)
                }
        except Exception as exc:
            log.debug("Area registry cache unavailable: %s", exc)

        self._registries_loaded_at = time.monotonic()

    def _entity_context(self, entity_id: str, state: dict[str, Any]) -> dict[str, Any]:
        registry = self.entity_registry.get(entity_id, {})
        attributes = state.get("attributes") if isinstance(state, dict) else {}
        if not isinstance(attributes, dict):
            attributes = {}

        device_id = registry.get("di")
        device = self.devices.get(device_id, {}) if isinstance(device_id, str) else {}

        area_id = registry.get("ai")
        if not area_id and isinstance(device, dict):
            area_id = device.get("area_id")
        area = self.areas.get(area_id, {}) if isinstance(area_id, str) else {}

        friendly_name = attributes.get("friendly_name")
        registry_name = registry.get("en")
        device_name = None
        if isinstance(device, dict):
            device_name = device.get("name_by_user") or device.get("name")
        area_name = area.get("name") if isinstance(area, dict) else None

        return {
            "friendly_name": friendly_name,
            "registry_name": registry_name,
            "device_id": device_id,
            "device_name": device_name,
            "area_id": area_id,
            "area_name": area_name,
        }

    def search_cached_states(
        self,
        query: str,
        *,
        domain_filter: str | None,
        area_filter: str | None,
        preferred_area_filter: str | None = None,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Search cached states.

        area_filter is a hard user-requested area restriction.
        preferred_area_filter is a soft voice-origin ranking preference.
        """
        query_norm = _normalise_search_text(query)
        query_tokens = set(query_norm.split())
        hard_area_norm = _normalise_search_text(area_filter) if area_filter else ""
        preferred_area_norm = (
            _normalise_search_text(preferred_area_filter)
            if preferred_area_filter
            else ""
        )
        domain_norm = (domain_filter or "").strip().casefold()
        domain_words = domain_norm.replace("_", " ")

        generic_domain_query = bool(
            domain_norm
            and query_norm
            and query_norm
            in {
                domain_words,
                domain_words.rstrip("s"),
                f"{domain_words} device",
                f"{domain_words} devices",
            }
        )

        scored: list[tuple[float, dict[str, Any]]] = []

        for entity_id, state in list(self.states.items()):
            domain = entity_id.split(".", 1)[0].casefold()
            if domain_norm and domain != domain_norm:
                continue

            ctx = self._entity_context(entity_id, state)
            local_entity_name = entity_id.split(".", 1)[-1]

            # IMPORTANT: do not normally include the full entity_id here.
            # Otherwise query="light" matches every "light.*" entity merely
            # because of its domain prefix.
            identity_parts = [
                _normalise_search_text(local_entity_name),
                _normalise_search_text(ctx.get("friendly_name")),
                _normalise_search_text(ctx.get("registry_name")),
            ]
            identity_parts = [p for p in identity_parts if p]

            context_parts = [
                _normalise_search_text(ctx.get("device_name")),
                _normalise_search_text(ctx.get("area_name")),
            ]
            context_parts = [p for p in context_parts if p]

            searchable_parts = identity_parts + context_parts
            if "." in query_norm:
                searchable_parts.append(_normalise_search_text(entity_id))

            searchable = " ".join(searchable_parts)
            identity_text = " ".join(identity_parts)
            device_text = _normalise_search_text(ctx.get("device_name"))
            entity_area = _normalise_search_text(ctx.get("area_name"))
            entity_area_id = _normalise_search_text(ctx.get("area_id"))

            # Explicit/user requested area remains a hard restriction.
            if hard_area_norm:
                if hard_area_norm not in {entity_area, entity_area_id} and (
                    hard_area_norm not in entity_area
                ):
                    continue

            score = 0.0
            reasons: list[str] = []

            if query_norm and query_norm in set(searchable_parts):
                score = 100.0
                reasons.append("exact_name")
            elif query_norm and query_norm in searchable:
                score = 75.0
                reasons.append("name_contains_query")
            elif query_tokens:
                combined_tokens = set(searchable.split())
                matched = len(query_tokens & combined_tokens)
                if matched:
                    score = 45.0 + (matched / len(query_tokens)) * 30.0
                    reasons.append("token_match")

            # Generic domain requests still consider all entities of that domain,
            # but only with a modest baseline score.
            if generic_domain_query:
                score = max(score, 35.0)
                reasons.append("generic_domain_candidate")

            if score < 45.0 and query_norm and searchable_parts:
                best_ratio = max(
                    (
                        difflib.SequenceMatcher(
                            None, query_norm, candidate
                        ).ratio()
                        for candidate in searchable_parts
                        if candidate
                    ),
                    default=0.0,
                )
                if best_ratio >= 0.58:
                    score = max(score, best_ratio * 60.0)
                    reasons.append("fuzzy_match")

            if not query_norm:
                score = 1.0

            # Soft room preference. Actual HA area membership helps, but an
            # entity explicitly named after the room helps more. Thus
            # office_light can win even if it has no HA area assignment.
            if preferred_area_norm:
                actual_area_match = (
                    preferred_area_norm in {entity_area, entity_area_id}
                    or (
                        entity_area
                        and preferred_area_norm in entity_area
                    )
                )
                room_name_in_identity = any(
                    preferred_area_norm in part
                    for part in identity_parts
                )

                if actual_area_match:
                    score += 30.0
                    reasons.append("voice_origin_area")
                if room_name_in_identity:
                    score += 45.0
                    reasons.append("room_name_in_entity")

            attributes = state.get("attributes", {})
            if not isinstance(attributes, dict):
                attributes = {}

            device_class = _normalise_search_text(attributes.get("device_class"))

            # "Turn the light on" means room illumination, not a ring/status LED.
            # Only apply this to a generic light query; explicit "ring LED" or
            # "indicator light" requests remain available.
            if (
                GENERIC_LIGHT_INDICATOR_PENALTY
                and domain == "light"
                and generic_domain_query
            ):
                explicit_indicator_words = {
                    "led", "ring", "indicator", "status",
                    "notification", "backlight",
                }
                if not (query_tokens & explicit_indicator_words):
                    indicator_text = " ".join(
                        p for p in (identity_text, device_text) if p
                    )
                    indicator_phrases = (
                        "led ring",
                        "ring led",
                        "indicator",
                        "status led",
                        "status light",
                        "notification led",
                        "notification light",
                        "backlight",
                    )
                    indicator_like = any(
                        phrase in indicator_text
                        for phrase in indicator_phrases
                    ) or (
                        "voice" in indicator_text
                        and "led" in indicator_text
                    )
                    if indicator_like:
                        score -= 90.0
                        reasons.append("indicator_light_penalty")

            # Helpful for generic local media playback too.
            if (
                domain == "media_player"
                and preferred_area_norm
                and generic_domain_query
            ):
                if device_class == "speaker":
                    score += 20.0
                    reasons.append("speaker_preference")
                elif device_class == "tv":
                    score -= 10.0
                    reasons.append("tv_penalty")

            if score <= 0:
                continue

            scored.append(
                (
                    score,
                    {
                        "entity_id": entity_id,
                        "name": ctx.get("friendly_name")
                        or ctx.get("registry_name")
                        or entity_id,
                        "domain": domain,
                        "state": state.get("state"),
                        "area": ctx.get("area_name"),
                        "area_id": ctx.get("area_id"),
                        "device": ctx.get("device_name"),
                        "device_class": attributes.get("device_class"),
                        "unit": attributes.get("unit_of_measurement"),
                        "match_score": round(score, 2),
                        "match_reasons": reasons,
                    },
                )
            )

        scored.sort(key=lambda item: (-item[0], item[1]["entity_id"]))
        return [record for _, record in scored[:limit]]

    def resolve_area_reference(
        self,
        *,
        area_id: str | None = None,
        area_name: str | None = None,
    ) -> tuple[str | None, str | None]:
        """Resolve an HA area ID/name pair from cached registry data."""
        if isinstance(area_id, str) and area_id.strip():
            area_id = area_id.strip()
            area = self.areas.get(area_id)
            if isinstance(area, dict):
                return area_id, area.get("name") or area_name

        if isinstance(area_name, str) and area_name.strip():
            wanted = _normalise_search_text(area_name)
            matches = [
                (candidate_id, area.get("name"))
                for candidate_id, area in self.areas.items()
                if isinstance(area, dict)
                and _normalise_search_text(area.get("name")) == wanted
            ]
            if len(matches) == 1:
                return matches[0]

        return None, None

    def resolve_device_origin(
        self,
        *,
        device_id: str | None = None,
        device_name: str | None = None,
        area_id: str | None = None,
        area_name: str | None = None,
    ) -> dict[str, Any]:
        """Resolve a calling voice device and its room from cached registries."""
        device: dict[str, Any] | None = None
        resolved_device_id: str | None = None

        if isinstance(device_id, str) and device_id.strip():
            candidate = self.devices.get(device_id.strip())
            if isinstance(candidate, dict):
                resolved_device_id = device_id.strip()
                device = candidate

        if device is None and isinstance(device_name, str) and device_name.strip():
            wanted = _normalise_search_text(device_name)
            matches = [
                (candidate_id, candidate)
                for candidate_id, candidate in self.devices.items()
                if isinstance(candidate, dict)
                and _normalise_search_text(
                    candidate.get("name_by_user") or candidate.get("name")
                ) == wanted
            ]
            if len(matches) == 1:
                resolved_device_id, device = matches[0]

        resolved_device_name = device_name
        if isinstance(device, dict):
            resolved_device_name = (
                device.get("name_by_user") or device.get("name") or device_name
            )
            if not area_id:
                area_id = device.get("area_id")

        # Some devices inherit useful area context only through their entities.
        if resolved_device_id and not area_id:
            entity_area_ids = {
                entry.get("ai")
                for entry in self.entity_registry.values()
                if isinstance(entry, dict)
                and entry.get("di") == resolved_device_id
                and isinstance(entry.get("ai"), str)
                and entry.get("ai")
            }
            if len(entity_area_ids) == 1:
                area_id = next(iter(entity_area_ids))

        resolved_area_id, resolved_area_name = self.resolve_area_reference(
            area_id=area_id, area_name=area_name
        )
        if not resolved_area_id and isinstance(area_id, str) and area_id.strip():
            # Keep a caller-provided ID even if registry refresh lags.
            resolved_area_id = area_id.strip()
            resolved_area_name = area_name

        return {
            "device_id": resolved_device_id or (device_id.strip() if isinstance(device_id, str) and device_id.strip() else None),
            "device_name": resolved_device_name,
            "area_id": resolved_area_id,
            "area_name": resolved_area_name,
        }

    def area_mentioned_in_text(self, text: str) -> tuple[str, str] | None:
        """Return one explicitly named HA area found in natural-language text."""
        haystack = f" {_normalise_search_text(text)} "
        matches: list[tuple[str, str]] = []
        for candidate_id, area in self.areas.items():
            if not isinstance(area, dict):
                continue
            name = area.get("name")
            normalised = _normalise_search_text(name)
            if normalised and f" {normalised} " in haystack:
                matches.append((candidate_id, str(name)))
        return matches[0] if len(matches) == 1 else None

    def entities_in_area(self, domain: str, area_id: str) -> list[str]:
        """Return cached entities in a specific area/domain."""
        matches: list[str] = []
        for entity_id, state in self.states.items():
            if entity_id.split(".", 1)[0].casefold() != domain.casefold():
                continue
            ctx = self._entity_context(entity_id, state)
            if ctx.get("area_id") == area_id:
                matches.append(entity_id)
        return sorted(matches)

    async def wait_for_expected_state(
        self,
        entity_id: str,
        expected_state: str,
        timeout: float,
    ) -> str | None:
        if timeout <= 0:
            return None
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = self.states.get(entity_id)
            if isinstance(state, dict) and state.get("state") == expected_state:
                return expected_state
            await asyncio.sleep(0.04)
        return None


ha_ws: HomeAssistantWebSocket | None = None


async def _require_ha_ws() -> HomeAssistantWebSocket:
    if ha_ws is None:
        raise RuntimeError("Home Assistant WebSocket fast path is unavailable")
    await asyncio.wait_for(ha_ws.ready.wait(), timeout=HA_WS_COMMAND_TIMEOUT)
    return ha_ws


@dataclass
class SatelliteIndicatorTarget:
    entity_id: str
    domain: str
    satellite_entity_id: str
    device_id: str
    device_name: str | None
    area_id: str | None
    area_name: str | None
    match_reason: str


@dataclass
class SatelliteIndicatorSnapshot:
    entity_id: str
    domain: str
    state: str
    attributes: dict[str, Any]


@dataclass
class SatelliteIndicatorSession:
    target: SatelliteIndicatorTarget
    snapshot: SatelliteIndicatorSnapshot
    refcount: int = 1
    pulse_task: asyncio.Task[Any] | None = None
    native_effect: str | None = None


@dataclass(frozen=True)
class SatelliteIndicatorHandle:
    entity_id: str


def _voice_origin_snapshot() -> dict[str, Any]:
    """Capture request-scoped voice origin before background work outlives it."""
    return {
        "device_id": voice_tool_run_state.origin_device_id,
        "device_name": voice_tool_run_state.origin_device_name,
        "area_id": voice_tool_run_state.origin_area_id,
        "area_name": voice_tool_run_state.origin_area_name,
        "source": voice_tool_run_state.origin_source,
    }


def _indicator_identity_text(client: HomeAssistantWebSocket, entity_id: str) -> str:
    state = client.states.get(entity_id, {})
    attrs = state.get("attributes", {}) if isinstance(state, dict) else {}
    if not isinstance(attrs, dict):
        attrs = {}
    registry = client.entity_registry.get(entity_id, {})
    device_id = registry.get("di") if isinstance(registry, dict) else None
    device = client.devices.get(device_id, {}) if isinstance(device_id, str) else {}
    device_name = None
    if isinstance(device, dict):
        device_name = device.get("name_by_user") or device.get("name")
    # Include the owning device name as well as the entity identity. Integrations
    # commonly expose a generic light entity on a connected device named e.g.
    # "Office Dot ring", which is exactly the topology signal we want to use.
    parts = [
        entity_id.split(".", 1)[-1],
        attrs.get("friendly_name"),
        registry.get("en") if isinstance(registry, dict) else None,
        device_name,
    ]
    return " ".join(_normalise_search_text(part) for part in parts if part)


def _device_via_device_id(client: HomeAssistantWebSocket, device_id: str) -> str | None:
    """Return the HA parent/via device ID, supporting current and legacy keys."""
    device = client.devices.get(device_id, {})
    if not isinstance(device, dict):
        return None
    # Home Assistant 2026.8 prefers via_device_id. Keep via_device as a defensive
    # compatibility fallback for older registry payloads/custom implementations.
    for key in ("via_device_id", "via_device"):
        value = device.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _connected_satellite_device_scope(
    client: HomeAssistantWebSocket,
    anchor_device_id: str,
    *,
    max_depth: int = 2,
) -> dict[str, tuple[int, str]]:
    """Return a bounded HA device-topology scope around an Assist satellite.

    Home Assistant integrations may model one physical satellite as a parent device
    plus endpoint devices such as assistant, microphone, playback and ring. The
    endpoint devices are linked through via_device_id. Treat those topology edges as
    undirected for discovery, but bound traversal to two hops so an Assist endpoint
    can reach its parent and sibling endpoints without wandering through a large HA
    device graph.
    """
    adjacency: dict[str, set[str]] = {device_id: set() for device_id in client.devices}
    for child_id in list(client.devices):
        parent_id = _device_via_device_id(client, child_id)
        if not parent_id or parent_id not in client.devices:
            continue
        adjacency.setdefault(child_id, set()).add(parent_id)
        adjacency.setdefault(parent_id, set()).add(child_id)

    distances: dict[str, int] = {anchor_device_id: 0}
    frontier = [anchor_device_id]
    while frontier:
        current = frontier.pop(0)
        depth = distances[current]
        if depth >= max_depth:
            continue
        for neighbour in sorted(adjacency.get(current, set())):
            if neighbour in distances:
                continue
            distances[neighbour] = depth + 1
            frontier.append(neighbour)

    anchor_parent = _device_via_device_id(client, anchor_device_id)
    result: dict[str, tuple[int, str]] = {}
    for device_id, depth in distances.items():
        if depth == 0:
            relation = "same_device"
        elif device_id == anchor_parent:
            relation = "parent_device"
        elif _device_via_device_id(client, device_id) == anchor_device_id:
            relation = "child_device"
        elif (
            anchor_parent
            and device_id != anchor_device_id
            and _device_via_device_id(client, device_id) == anchor_parent
        ):
            relation = "sibling_device"
        else:
            relation = f"connected_depth_{depth}"
        result[device_id] = (depth, relation)
    return result


def _indicator_relation_bonus(depth: int, relation: str) -> float:
    """Prefer the exact device while allowing a clearly named ring sibling to win."""
    bonuses = {
        "same_device": 30.0,
        "child_device": 20.0,
        "sibling_device": 18.0,
        "parent_device": 10.0,
    }
    return bonuses.get(relation, max(0.0, 10.0 - (depth * 2.0)))


def _indicator_score(client: HomeAssistantWebSocket, entity_id: str) -> tuple[float, list[str]]:
    identity = _indicator_identity_text(client, entity_id)
    domain = entity_id.split(".", 1)[0].casefold()
    score = 0.0
    reasons: list[str] = []
    weighted = (
        ("led", 120.0),
        ("ring", 105.0),
        ("indicator", 95.0),
        ("notification", 70.0),
        ("status", 55.0),
        ("pixel", 50.0),
    )
    for word, points in weighted:
        if word in identity.split() or word in identity:
            score += points
            reasons.append(word)
    if domain == "light":
        score += 12.0
        reasons.append("light_domain")
    elif domain == "switch":
        score += 4.0
        reasons.append("switch_domain")
    return score, reasons


async def _ha_internal_service_call(
    domain: str,
    service: str,
    entity_id: str,
    data: dict[str, Any] | None = None,
) -> None:
    """Call a tightly-scoped HA service without mutating LLM tool scratch state."""
    client = await _require_ha_ws()
    payload: dict[str, Any] = {
        "type": "call_service",
        "domain": domain,
        "service": service,
        "service_data": dict(data or {}),
        "target": {"entity_id": entity_id},
        "return_response": False,
    }
    reply = await client.command(payload, timeout=HA_WS_COMMAND_TIMEOUT)
    if reply.get("success") is not True:
        error = reply.get("error")
        raise RuntimeError(f"HA {domain}.{service} failed for {entity_id}: {error}")


_INDICATOR_NAMED_RGB: dict[str, tuple[int, int, int]] = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "orange": (255, 165, 0),
    "purple": (128, 0, 128),
    "magenta": (255, 0, 255),
    "pink": (255, 105, 180),
    "cyan": (0, 255, 255),
    "teal": (0, 128, 128),
    "lime": (0, 255, 0),
}


def _configured_indicator_rgb() -> list[int] | None:
    """Parse configured indicator colour without adding a new dependency."""
    raw = MUSIC_ASSISTANT_ACTIVITY_INDICATOR_COLOR.strip()
    if not raw or raw.casefold() in {"none", "off", "disabled", "current", "preserve"}:
        return None
    normalized = raw.casefold().replace(" ", "_").replace("-", "_")
    named = _INDICATOR_NAMED_RGB.get(normalized)
    if named is not None:
        return list(named)
    if raw.startswith("#") and len(raw) == 7:
        try:
            return [int(raw[1:3], 16), int(raw[3:5], 16), int(raw[5:7], 16)]
        except ValueError:
            return None
    csv = raw.strip("[]() ")
    parts = [part.strip() for part in csv.split(",")]
    if len(parts) == 3:
        try:
            values = [int(part) for part in parts]
        except ValueError:
            return None
        if all(0 <= value <= 255 for value in values):
            return values
    return None


def _find_configured_native_effect(attributes: dict[str, Any]) -> str | None:
    requested = MUSIC_ASSISTANT_ACTIVITY_INDICATOR_EFFECT.strip()
    if not requested or requested.casefold() in {"none", "off", "disabled", "current", "preserve"}:
        return None
    if requested.casefold() == "auto":
        requested = "pulse"
    effects = attributes.get("effect_list")
    if not isinstance(effects, (list, tuple)):
        return None
    names = [str(item) for item in effects if str(item).strip()]
    wanted = requested.casefold()
    exact = next((name for name in names if name.casefold() == wanted), None)
    if exact:
        return exact
    # Useful for integrations exposing e.g. "Slow Pulse" when configured as
    # "pulse", or "Rainbow Cycle" when configured as "rainbow".
    containing = [name for name in names if wanted in name.casefold()]
    return min(containing, key=len) if containing else None


def _configured_effect_wants_software_pulse() -> bool:
    requested = MUSIC_ASSISTANT_ACTIVITY_INDICATOR_EFFECT.strip().casefold()
    return requested == "auto" or "pulse" in requested


def _find_neutral_native_effect(attributes: dict[str, Any]) -> str | None:
    """Best-effort advertised value that disables/neutralises an active effect."""
    effects = attributes.get("effect_list")
    if not isinstance(effects, (list, tuple)):
        return None
    names = [str(item) for item in effects if str(item).strip()]
    for neutral in ("none", "off", "static", "solid"):
        exact = next((name for name in names if name.casefold() == neutral), None)
        if exact:
            return exact
    return None


def _light_supports_colour(attributes: dict[str, Any]) -> bool:
    modes = attributes.get("supported_color_modes")
    if not isinstance(modes, (list, tuple, set)):
        return False
    colour_modes = {"hs", "rgb", "rgbw", "rgbww", "xy"}
    return any(str(mode).casefold() in colour_modes for mode in modes)


def _snapshot_restore_light_data(snapshot: SatelliteIndicatorSnapshot) -> dict[str, Any]:
    """Build best-effort light.turn_on data for the exact prior visual state."""
    attrs = snapshot.attributes
    data: dict[str, Any] = {}
    brightness = attrs.get("brightness")
    if isinstance(brightness, (int, float)):
        data["brightness"] = max(1, min(255, int(brightness)))

    effect = attrs.get("effect")
    if isinstance(effect, str) and effect.strip():
        data["effect"] = effect

    color_mode = str(attrs.get("color_mode") or "").casefold()
    mode_to_key = {
        "rgb": "rgb_color",
        "rgbw": "rgbw_color",
        "rgbww": "rgbww_color",
        "hs": "hs_color",
        "xy": "xy_color",
        "color_temp": "color_temp_kelvin",
    }
    preferred = mode_to_key.get(color_mode)
    candidate_keys = [
        preferred,
        "rgb_color",
        "rgbw_color",
        "rgbww_color",
        "hs_color",
        "xy_color",
        "color_temp_kelvin",
    ]
    seen: set[str] = set()
    for key in candidate_keys:
        if not key or key in seen:
            continue
        seen.add(key)
        value = attrs.get(key)
        if value is None:
            continue
        if key in {"rgb_color", "rgbw_color", "rgbww_color", "hs_color", "xy_color"}:
            if isinstance(value, (list, tuple)):
                data[key] = list(value)
                break
        elif key == "color_temp_kelvin" and isinstance(value, (int, float)):
            data[key] = int(value)
            break
    return data


async def _resolve_satellite_indicator(
    origin_context: dict[str, Any] | None,
) -> SatelliteIndicatorTarget | None:
    """Resolve an indicator only through a real assist_satellite device relation."""
    if not MUSIC_ASSISTANT_ACTIVITY_INDICATOR_ENABLED or not origin_context:
        return None
    client = await _require_ha_ws()
    await client.refresh_registries()

    origin_device_id = str(origin_context.get("device_id") or "").strip() or None
    origin_device_name = str(origin_context.get("device_name") or "").strip() or None
    origin_area_id = str(origin_context.get("area_id") or "").strip() or None
    origin_area_name = str(origin_context.get("area_name") or "").strip() or None

    # Resolve any supplied device name/id first, but only trust it as a satellite
    # anchor if that HA device actually owns an assist_satellite.* entity.
    resolved_origin = client.resolve_device_origin(
        device_id=origin_device_id,
        device_name=origin_device_name,
        area_id=origin_area_id,
        area_name=origin_area_name,
    )
    candidate_device_id = resolved_origin.get("device_id")
    satellite_entity_id: str | None = None
    match_reason: str | None = None

    if isinstance(candidate_device_id, str) and candidate_device_id:
        device_satellites = [
            entity_id
            for entity_id in client.states
            if entity_id.startswith("assist_satellite.")
            and client.entity_registry.get(entity_id, {}).get("di") == candidate_device_id
        ]
        if len(device_satellites) == 1:
            satellite_entity_id = device_satellites[0]
            match_reason = "origin_device_assist_satellite"

    # HA's OpenAI conversation caller often gives us only an area through its
    # trusted system prompt. In that case require exactly one assist_satellite in
    # the area before traversing to its owning device.
    if satellite_entity_id is None:
        resolved_area_id, resolved_area_name = client.resolve_area_reference(
            area_id=origin_area_id or resolved_origin.get("area_id"),
            area_name=origin_area_name or resolved_origin.get("area_name"),
        )
        area_satellites: list[str] = []
        for entity_id, state in client.states.items():
            if not entity_id.startswith("assist_satellite."):
                continue
            ctx = client._entity_context(entity_id, state)
            if resolved_area_id and ctx.get("area_id") == resolved_area_id:
                area_satellites.append(entity_id)
            elif (
                not resolved_area_id
                and resolved_area_name
                and _normalise_search_text(ctx.get("area_name"))
                == _normalise_search_text(resolved_area_name)
            ):
                area_satellites.append(entity_id)
        if len(area_satellites) != 1:
            log.info(
                "VOICE INDICATOR skipped: area satellite resolution ambiguous area=%r matches=%s",
                resolved_area_name or origin_area_name or origin_area_id,
                area_satellites,
            )
            return None
        satellite_entity_id = area_satellites[0]
        match_reason = "sole_assist_satellite_in_origin_area"
        candidate_device_id = client.entity_registry.get(satellite_entity_id, {}).get("di")

    if not isinstance(candidate_device_id, str) or not candidate_device_id:
        log.info(
            "VOICE INDICATOR skipped: assist_satellite has no registry device satellite=%s",
            satellite_entity_id,
        )
        return None

    # Resolve indicator controls across the Assist satellite's bounded HA device
    # topology. A number of integrations model one physical voice satellite as
    # connected endpoint devices (assistant/microphone/playback/ring), so an LED
    # can legitimately live on a sibling device rather than the assist_satellite's
    # own device entry.
    device_scope = _connected_satellite_device_scope(
        client, candidate_device_id, max_depth=2
    )
    if len(device_scope) > 1:
        scope_log = []
        for scoped_device_id, (depth, relation) in sorted(
            device_scope.items(), key=lambda item: (item[1][0], item[0])
        ):
            scoped_device = client.devices.get(scoped_device_id, {})
            scoped_name = None
            if isinstance(scoped_device, dict):
                scoped_name = scoped_device.get("name_by_user") or scoped_device.get("name")
            scope_log.append((scoped_name or scoped_device_id, relation, depth))
        log.info(
            "VOICE INDICATOR connected-device scope satellite=%s anchor_device=%s devices=%s",
            satellite_entity_id,
            candidate_device_id,
            scope_log,
        )

    # Tuple: final_score, entity_id, reasons, owning_device_id, relation, depth.
    sibling_controls: list[tuple[float, str, list[str], str, str, int]] = []
    for entity_id, state in client.states.items():
        domain = entity_id.split(".", 1)[0].casefold()
        if domain not in MUSIC_ASSISTANT_ACTIVITY_INDICATOR_DOMAINS:
            continue
        registry = client.entity_registry.get(entity_id, {})
        owning_device_id = registry.get("di") if isinstance(registry, dict) else None
        if not isinstance(owning_device_id, str) or owning_device_id not in device_scope:
            continue
        depth, relation = device_scope[owning_device_id]
        score, reasons = _indicator_score(client, entity_id)
        if score > 0:
            relation_bonus = _indicator_relation_bonus(depth, relation)
            sibling_controls.append(
                (
                    score + relation_bonus,
                    entity_id,
                    [*reasons, relation],
                    owning_device_id,
                    relation,
                    depth,
                )
            )

    # Generic domain-only controls (for example remote_adb or bluetooth_proxy
    # switches) should not prevent traversal to a clearly named connected ring
    # endpoint. If no indicator-like name exists anywhere in the topology, retain
    # the old conservative sole-control fallback but only when the entire bounded
    # topology exposes exactly one light/switch control.
    if not sibling_controls:
        sole_controls: list[tuple[str, str, int, str]] = []
        for entity_id in client.states:
            if entity_id.split(".", 1)[0].casefold() not in MUSIC_ASSISTANT_ACTIVITY_INDICATOR_DOMAINS:
                continue
            registry = client.entity_registry.get(entity_id, {})
            owning_device_id = registry.get("di") if isinstance(registry, dict) else None
            if not isinstance(owning_device_id, str) or owning_device_id not in device_scope:
                continue
            depth, relation = device_scope[owning_device_id]
            sole_controls.append((entity_id, owning_device_id, depth, relation))
        if len(sole_controls) == 1:
            entity_id, owning_device_id, depth, relation = sole_controls[0]
            sibling_controls.append(
                (
                    10.0 + _indicator_relation_bonus(depth, relation),
                    entity_id,
                    ["sole_connected_light_control", relation],
                    owning_device_id,
                    relation,
                    depth,
                )
            )

    if not sibling_controls:
        log.info(
            "VOICE INDICATOR skipped: no indicator control in connected satellite topology "
            "satellite=%s device=%s",
            satellite_entity_id,
            candidate_device_id,
        )
        return None

    sibling_controls.sort(key=lambda item: (-item[0], item[5], item[1]))
    top_score, entity_id, reasons, indicator_device_id, relation, depth = sibling_controls[0]
    if len(sibling_controls) > 1 and sibling_controls[1][0] == top_score:
        log.info(
            "VOICE INDICATOR skipped: connected indicator tie satellite=%s candidates=%s",
            satellite_entity_id,
            [
                (item[1], item[0], item[4], item[3])
                for item in sibling_controls
                if item[0] == top_score
            ],
        )
        return None

    if indicator_device_id != candidate_device_id:
        indicator_device = client.devices.get(indicator_device_id, {})
        indicator_device_name = None
        if isinstance(indicator_device, dict):
            indicator_device_name = (
                indicator_device.get("name_by_user") or indicator_device.get("name")
            )
        log.info(
            "VOICE INDICATOR selected connected control satellite=%s "
            "satellite_device=%s indicator_device=%s indicator_device_name=%r "
            "relation=%s depth=%d entity=%s score=%.1f",
            satellite_entity_id,
            candidate_device_id,
            indicator_device_id,
            indicator_device_name,
            relation,
            depth,
            entity_id,
            top_score,
        )

    state = client.states.get(satellite_entity_id, {})
    sat_ctx = client._entity_context(satellite_entity_id, state)
    indicator_device = client.devices.get(indicator_device_id, {})
    device_name = None
    if isinstance(indicator_device, dict):
        device_name = indicator_device.get("name_by_user") or indicator_device.get("name")

    return SatelliteIndicatorTarget(
        entity_id=entity_id,
        domain=entity_id.split(".", 1)[0].casefold(),
        satellite_entity_id=satellite_entity_id,
        device_id=indicator_device_id,
        device_name=device_name,
        area_id=sat_ctx.get("area_id"),
        area_name=sat_ctx.get("area_name"),
        match_reason=(
            f"{match_reason}:{relation}:depth{depth}:{'+'.join(reasons)}"
        ),
    )


class VoiceSatelliteActivityIndicators:
    """Reference-counted optional LED feedback for optimistic MA playback."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._sessions: dict[str, SatelliteIndicatorSession] = {}
        self.last_error: str | None = None
        self.last_target: dict[str, Any] | None = None

    @property
    def active_count(self) -> int:
        return len(self._sessions)

    async def begin(
        self, origin_context: dict[str, Any] | None
    ) -> SatelliteIndicatorHandle | None:
        try:
            target = await _resolve_satellite_indicator(origin_context)
            if target is None:
                return None
            async with self._lock:
                existing = self._sessions.get(target.entity_id)
                if existing is not None:
                    existing.refcount += 1
                    log.info(
                        "VOICE INDICATOR reuse entity=%s refcount=%d",
                        target.entity_id,
                        existing.refcount,
                    )
                    return SatelliteIndicatorHandle(target.entity_id)

                client = await _require_ha_ws()
                state = client.states.get(target.entity_id)
                if not isinstance(state, dict):
                    return None
                attrs = state.get("attributes")
                if not isinstance(attrs, dict):
                    attrs = {}
                snapshot = SatelliteIndicatorSnapshot(
                    entity_id=target.entity_id,
                    domain=target.domain,
                    state=str(state.get("state") or "unknown"),
                    attributes=dict(attrs),
                )
                session = SatelliteIndicatorSession(target=target, snapshot=snapshot)
                self._sessions[target.entity_id] = session

            try:
                await self._activate(session)
            except Exception:
                async with self._lock:
                    self._sessions.pop(target.entity_id, None)
                raise

            self.last_error = None
            self.last_target = {
                "entity_id": target.entity_id,
                "satellite_entity_id": target.satellite_entity_id,
                "device_id": target.device_id,
                "device_name": target.device_name,
                "area_name": target.area_name,
                "match_reason": target.match_reason,
                "native_effect": session.native_effect,
            }
            return SatelliteIndicatorHandle(target.entity_id)
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            log.warning("VOICE INDICATOR start failed: %s", self.last_error)
            return None

    async def _activate(self, session: SatelliteIndicatorSession) -> None:
        target = session.target
        attrs = session.snapshot.attributes
        if target.domain == "light":
            data: dict[str, Any] = {}
            configured_rgb = _configured_indicator_rgb()
            if configured_rgb is not None and _light_supports_colour(attrs):
                # Home Assistant translates rgb_color to another supported colour
                # mode when required, so RGB is a safe common input.
                data["rgb_color"] = configured_rgb
            configured_effect = _find_configured_native_effect(attrs)
            if configured_effect:
                data["effect"] = configured_effect
                session.native_effect = configured_effect
            await _ha_internal_service_call("light", "turn_on", target.entity_id, data)
            if (
                not configured_effect
                and MUSIC_ASSISTANT_ACTIVITY_INDICATOR_SOFTWARE_PULSE
                and _configured_effect_wants_software_pulse()
            ):
                session.pulse_task = asyncio.create_task(
                    self._software_pulse_light(session),
                    name=f"voice-indicator-pulse-{target.entity_id}",
                )
        elif target.domain == "switch":
            await _ha_internal_service_call("switch", "turn_on", target.entity_id)
            if (
                MUSIC_ASSISTANT_ACTIVITY_INDICATOR_SOFTWARE_PULSE
                and _configured_effect_wants_software_pulse()
            ):
                session.pulse_task = asyncio.create_task(
                    self._software_pulse_switch(session),
                    name=f"voice-indicator-pulse-{target.entity_id}",
                )
        log.info(
            "VOICE INDICATOR active entity=%s satellite=%s device=%s area=%r "
            "color=%r effect_requested=%r native_effect=%r software_pulse=%s "
            "previous_state=%s previous_effect=%r",
            target.entity_id,
            target.satellite_entity_id,
            target.device_id,
            target.area_name,
            MUSIC_ASSISTANT_ACTIVITY_INDICATOR_COLOR,
            MUSIC_ASSISTANT_ACTIVITY_INDICATOR_EFFECT,
            session.native_effect,
            session.pulse_task is not None,
            session.snapshot.state,
            session.snapshot.attributes.get("effect"),
        )

    async def _software_pulse_light(self, session: SatelliteIndicatorSession) -> None:
        entity_id = session.target.entity_id
        attrs = session.snapshot.attributes
        on_data: dict[str, Any] = {}
        configured_rgb = _configured_indicator_rgb()
        if configured_rgb is not None and _light_supports_colour(attrs):
            on_data["rgb_color"] = configured_rgb
        try:
            while True:
                await asyncio.sleep(MUSIC_ASSISTANT_ACTIVITY_INDICATOR_PULSE_INTERVAL_SECONDS)
                await _ha_internal_service_call("light", "turn_off", entity_id)
                await asyncio.sleep(MUSIC_ASSISTANT_ACTIVITY_INDICATOR_PULSE_INTERVAL_SECONDS)
                await _ha_internal_service_call("light", "turn_on", entity_id, on_data)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("VOICE INDICATOR software light pulse stopped entity=%s error=%s", entity_id, exc)

    async def _software_pulse_switch(self, session: SatelliteIndicatorSession) -> None:
        entity_id = session.target.entity_id
        try:
            while True:
                await asyncio.sleep(MUSIC_ASSISTANT_ACTIVITY_INDICATOR_PULSE_INTERVAL_SECONDS)
                await _ha_internal_service_call("switch", "turn_off", entity_id)
                await asyncio.sleep(MUSIC_ASSISTANT_ACTIVITY_INDICATOR_PULSE_INTERVAL_SECONDS)
                await _ha_internal_service_call("switch", "turn_on", entity_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("VOICE INDICATOR software switch pulse stopped entity=%s error=%s", entity_id, exc)

    async def end(self, handle: SatelliteIndicatorHandle | None) -> None:
        if handle is None:
            return
        session: SatelliteIndicatorSession | None = None
        async with self._lock:
            current = self._sessions.get(handle.entity_id)
            if current is None:
                return
            current.refcount -= 1
            if current.refcount > 0:
                log.info(
                    "VOICE INDICATOR retain entity=%s refcount=%d",
                    handle.entity_id,
                    current.refcount,
                )
                return
            session = self._sessions.pop(handle.entity_id, None)
        if session is not None:
            await self._restore(session)

    async def _restore(self, session: SatelliteIndicatorSession) -> None:
        pulse_task = session.pulse_task
        if pulse_task is not None and not pulse_task.done():
            pulse_task.cancel()
            await asyncio.gather(pulse_task, return_exceptions=True)

        snap = session.snapshot
        try:
            if snap.domain == "light":
                restore_data = _snapshot_restore_light_data(snap)
                # If the light previously had no effect and v6.8.2 temporarily
                # enabled one, explicitly select an advertised neutral effect when
                # possible so the configured activity effect does not become the
                # light's next-on/default effect. Existing non-empty effects are
                # already restored verbatim by _snapshot_restore_light_data().
                previous_effect = snap.attributes.get("effect")
                if (
                    session.native_effect
                    and not (isinstance(previous_effect, str) and previous_effect.strip())
                    and (neutral_effect := _find_neutral_native_effect(snap.attributes))
                ):
                    restore_data["effect"] = neutral_effect
                if snap.state == "on":
                    await _ha_internal_service_call("light", "turn_on", snap.entity_id, restore_data)
                else:
                    # Restore colour/effect as well as off state when possible.
                    # This may briefly turn the indicator on, but prevents our
                    # temporary green/pulse settings becoming its next-on state.
                    if restore_data:
                        await _ha_internal_service_call("light", "turn_on", snap.entity_id, restore_data)
                    await _ha_internal_service_call("light", "turn_off", snap.entity_id)
            elif snap.domain == "switch":
                await _ha_internal_service_call(
                    "switch",
                    "turn_on" if snap.state == "on" else "turn_off",
                    snap.entity_id,
                )
            log.info(
                "VOICE INDICATOR restored entity=%s state=%s colour_mode=%r effect=%r",
                snap.entity_id,
                snap.state,
                snap.attributes.get("color_mode"),
                snap.attributes.get("effect"),
            )
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            log.warning("VOICE INDICATOR restore failed entity=%s error=%s", snap.entity_id, exc)

    async def stop_all(self) -> None:
        async with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            session.refcount = 0
            await self._restore(session)


voice_activity_indicators = VoiceSatelliteActivityIndicators()


# ---------------------------------------------------------------------------
# Direct fast Home Assistant tools exposed to the normal voice agent
# ---------------------------------------------------------------------------


@function_tool
async def ha_search(
    query: str,
    domain_filter: str | None = None,
    area_filter: str | None = None,
    limit: int = HA_WS_SEARCH_DEFAULT_LIMIT,
) -> str:
    """Search cached Home Assistant entities by name, entity ID, device or area.

    Args:
        query: Natural-language entity/device name to find, such as "office light".
        domain_filter: Optional exact entity domain such as light, switch, climate, weather.
        area_filter: Optional explicit Home Assistant area name such as Office or Kitchen.
        limit: Maximum matches to return.
    """
    try:
        client = await _require_ha_ws()
        await client.refresh_registries()
        bounded_limit = min(max(1, int(limit)), HA_WS_SEARCH_MAX_LIMIT)

        effective_area_filter = area_filter
        preferred_area_filter: str | None = None
        area_context_source = "explicit" if area_filter else None

        # Explicit room names override the calling satellite room.
        if not effective_area_filter:
            mentioned = client.area_mentioned_in_text(query)
            if mentioned is not None:
                _, mentioned_name = mentioned
                effective_area_filter = mentioned_name
                area_context_source = "query"

        # The voice-origin room is only a preference. Do not hide room-named
        # entities which lack an explicit HA area assignment.
        if (
            not effective_area_filter
            and VOICE_ORIGIN_CONTEXT_ENABLED
            and VOICE_ORIGIN_AREA_BIAS
            and (
                voice_tool_run_state.origin_area_name
                or voice_tool_run_state.origin_area_id
            )
        ):
            origin_area = (
                voice_tool_run_state.origin_area_name
                or voice_tool_run_state.origin_area_id
            )
            if VOICE_ORIGIN_SOFT_AREA_RANKING:
                preferred_area_filter = origin_area
                area_context_source = "voice_origin_soft"
            else:
                effective_area_filter = origin_area
                area_context_source = "voice_origin"

        results = client.search_cached_states(
            query,
            domain_filter=domain_filter,
            area_filter=effective_area_filter,
            preferred_area_filter=preferred_area_filter,
            limit=bounded_limit,
        )

        # Backward-compatible fallback if hard origin mode is manually enabled.
        if not results and area_context_source == "voice_origin":
            results = client.search_cached_states(
                query,
                domain_filter=domain_filter,
                area_filter=None,
                preferred_area_filter=None,
                limit=bounded_limit,
            )
            area_context_source = "voice_origin_global_fallback"

        if results:
            top_preview = [
                {
                    "entity_id": item.get("entity_id"),
                    "score": item.get("match_score"),
                    "reasons": item.get("match_reasons"),
                }
                for item in results[:3]
            ]
            log.info(
                "HA SEARCH query=%r domain=%r hard_area=%r preferred_area=%r top=%s",
                query,
                domain_filter,
                effective_area_filter,
                preferred_area_filter,
                top_preview,
            )

        if len(results) == 1:
            entity_id = results[0].get("entity_id")
            domain = results[0].get("domain")
            if isinstance(entity_id, str) and isinstance(domain, str):
                voice_tool_run_state.last_entity_by_domain[domain] = entity_id
            area_id = results[0].get("area_id")
            if isinstance(area_id, str) and area_id:
                voice_tool_run_state.last_area_id = area_id

        return _json_tool_result(
            {
                "success": True,
                "query": query,
                "domain_filter": domain_filter,
                "area_filter": effective_area_filter,
                "preferred_area": preferred_area_filter,
                "area_context_source": area_context_source,
                "origin_area": voice_tool_run_state.origin_area_name,
                "count": len(results),
                "recommended_entity_id": (
                    results[0].get("entity_id") if results else None
                ),
                "results": results,
                "source": "local_state_cache",
            }
        )
    except Exception as exc:
        log.warning("DIRECT TOOL ha_search failed: %s", exc)
        return _json_tool_result({"success": False, "error": str(exc)})


@function_tool
async def ha_get_state(
    entity_id: str,
    attribute_keys: list[str] | None = None,
) -> str:
    """Read one Home Assistant entity from the local state cache.

    Args:
        entity_id: Exact Home Assistant entity ID previously found with ha_search.
        attribute_keys: Optional attribute names to return. Omit for compact useful attributes.
    """
    try:
        client = await _require_ha_ws()
        state = client.states.get(entity_id)
        if not isinstance(state, dict):
            return _json_tool_result(
                {
                    "success": False,
                    "error": f"Entity not found in current state cache: {entity_id}",
                    "suggestion": "Use ha_search to resolve the entity ID.",
                }
            )

        domain = entity_id.split(".", 1)[0]
        voice_tool_run_state.last_entity_by_domain[domain] = entity_id

        attributes = state.get("attributes", {})
        if not isinstance(attributes, dict):
            attributes = {}

        if attribute_keys is not None:
            selected_attributes = {
                key: attributes.get(key)
                for key in attribute_keys
                if key in attributes
            }
        else:
            selected_attributes = _compact_attributes(attributes)

        return _json_tool_result(
            {
                "success": True,
                "entity_id": entity_id,
                "state": state.get("state"),
                "attributes": selected_attributes,
                "last_changed": state.get("last_changed"),
                "last_updated": state.get("last_updated"),
                "source": "local_state_cache",
            }
        )
    except Exception as exc:
        log.warning("DIRECT TOOL ha_get_state failed: %s", exc)
        return _json_tool_result({"success": False, "error": str(exc)})


@function_tool
async def ha_list_services(
    domain: str | None = None,
    query: str | None = None,
    detail_level: Literal["summary", "full"] = "summary",
    limit: int = 50,
) -> str:
    """Discover actions from the cached Home Assistant service catalogue.

    Full detail returns a compact CALL schema rather than Home Assistant's
    UI-oriented raw schema. Exact keys under "parameters" are the keys that must
    be placed in ha_call_service(data=...).

    Args:
        domain: Optional exact service domain such as weather, light, calendar.
        query: Optional service-name/description search text.
        detail_level: summary or full. Full includes exact call parameters.
        limit: Maximum matching services to return.
    """
    try:
        client = await _require_ha_ws()
        await client.refresh_services()
        domain_norm = domain.strip().casefold() if isinstance(domain, str) else None
        query_norm = _normalise_search_text(query) if query else ""
        bounded_limit = min(max(1, int(limit)), 200)

        matches: list[dict[str, Any]] = []
        for service_domain, domain_services in client.services.items():
            if domain_norm and service_domain.casefold() != domain_norm:
                continue
            if not isinstance(domain_services, dict):
                continue

            for service_name, definition in domain_services.items():
                if not isinstance(definition, dict):
                    definition = {}
                searchable = _normalise_search_text(
                    f"{service_domain} {service_name} "
                    f"{definition.get('name', '')} {definition.get('description', '')}"
                )
                if query_norm and query_norm not in searchable:
                    continue

                if detail_level == "full":
                    record = _compact_service_definition(
                        service_domain,
                        service_name,
                        definition,
                    )
                else:
                    record = {
                        "domain": service_domain,
                        "service": service_name,
                        "name": definition.get("name"),
                        "description": _truncate_text(
                            definition.get("description"), 220
                        ),
                    }

                matches.append(record)
                if len(matches) >= bounded_limit:
                    break
            if len(matches) >= bounded_limit:
                break

        return _json_tool_result(
            {
                "success": True,
                "domain": domain,
                "query": query,
                "detail_level": detail_level,
                "count": len(matches),
                "services": matches,
                "schema_note": (
                    "For full detail, use exact keys under parameters as data keys. "
                    "Do not derive parameter names from human-readable labels."
                    if detail_level == "full"
                    else None
                ),
                "source": "cached_websocket_service_catalogue",
            }
        )
    except Exception as exc:
        log.warning("DIRECT TOOL ha_list_services failed: %s", exc)
        return _json_tool_result({"success": False, "error": str(exc)})


# This tool accepts arbitrary Home Assistant service_data keys.
# OpenAI Agents SDK strict JSON schemas reject free-form dict objects,
# so this one tool must use a non-strict schema.
@function_tool(strict_mode=False)
async def ha_call_service(
    domain: str,
    service: str,
    entity_id: str | None = None,
    area_id: str | None = None,
    data: dict[str, Any] | None = None,
    return_response: bool = False,
) -> str:
    """Call a Home Assistant action over the persistent WebSocket.

    Args:
        domain: Exact service domain such as light, climate, weather, calendar.
        service: Exact service/action name such as turn_off or get_forecasts.
        entity_id: Optional exact target entity ID.
        area_id: Optional exact Home Assistant area ID for an area-wide action.
        data: Service-specific fields using EXACT keys from ha_list_services.
        return_response: True for actions that return information/data.
    """
    try:
        client = await _require_ha_ws()
        await client.refresh_services()

        domain = str(domain or "").strip()
        service = str(service or "").strip()
        if not domain or not service:
            return _json_tool_result(
                {
                    "success": False,
                    "error_type": "local_schema_validation",
                    "error": "domain and service are required",
                }
            )

        definition = _get_cached_service_definition(client, domain, service)

        previous = voice_tool_run_state.last_service_call
        same_previous_service = (
            isinstance(previous, dict)
            and previous.get("domain") == domain
            and previous.get("service") == service
        )

        # Preserve a target already resolved in this SAME request while the model
        # repairs service_data. This prevents retries from accidentally dropping
        # weather.forecast_home, a light, calendar, etc.
        if not entity_id and not area_id and same_previous_service:
            previous_entity = previous.get("entity_id")
            previous_area = previous.get("area_id")
            if isinstance(previous_entity, str) and previous_entity:
                entity_id = previous_entity
                log.info(
                    "HA SERVICE RETRY preserved previous entity target=%s",
                    entity_id,
                )
            elif isinstance(previous_area, str) and previous_area:
                area_id = previous_area
                log.info(
                    "HA SERVICE RETRY preserved previous area target=%s",
                    area_id,
                )

        if (
            not entity_id
            and not area_id
            and VOICE_ORIGIN_CONTEXT_ENABLED
            and VOICE_ORIGIN_AREA_BIAS
            and voice_tool_run_state.origin_area_id
        ):
            origin_candidates = client.entities_in_area(
                domain, voice_tool_run_state.origin_area_id
            )
            if len(origin_candidates) == 1:
                entity_id = origin_candidates[0]
                log.info(
                    "HA SERVICE TARGET auto-selected sole origin-area entity domain=%s area=%s entity=%s",
                    domain,
                    voice_tool_run_state.origin_area_name or voice_tool_run_state.origin_area_id,
                    entity_id,
                )

        if not entity_id and not area_id:
            remembered = voice_tool_run_state.last_entity_by_domain.get(domain)
            if isinstance(remembered, str) and remembered:
                entity_id = remembered
                log.info(
                    "HA SERVICE TARGET reused unambiguous run target domain=%s entity=%s",
                    domain,
                    entity_id,
                )

        # For read/data actions only, selecting the sole entity of the required
        # target domain is deterministic and safe. Never do this for writes.
        if not entity_id and not area_id and return_response and definition:
            target_domains = _target_entity_domains(definition.get("target"))
            candidate_domains = target_domains or [domain]
            candidates = [
                _single_cached_entity_for_domain(client, target_domain)
                for target_domain in candidate_domains
            ]
            candidates = [candidate for candidate in candidates if candidate]
            unique_candidates = list(dict.fromkeys(candidates))
            if len(unique_candidates) == 1:
                entity_id = unique_candidates[0]
                log.info(
                    "HA DATA SERVICE TARGET auto-selected sole entity=%s",
                    entity_id,
                )

        previous_data = (
            previous.get("data")
            if same_previous_service and isinstance(previous.get("data"), dict)
            else None
        )
        normalised_data, repairs, validation = _normalise_service_data_from_schema(
            domain,
            service,
            definition,
            data,
            previous_data,
        )

        # Preserve return_response=True on a repair retry. The default False in
        # a fresh model tool call otherwise makes it easy to lose returned data.
        if (
            same_previous_service
            and previous.get("return_response") is True
            and return_response is False
        ):
            return_response = True
            repairs.append("preserved return_response=True from previous retry")

        voice_tool_run_state.last_service_call = {
            "domain": domain,
            "service": service,
            "entity_id": entity_id,
            "area_id": area_id,
            "data": dict(normalised_data),
            "return_response": bool(return_response),
        }

        if repairs:
            log.info(
                "HA SERVICE LOCAL REPAIR domain=%s service=%s repairs=%s",
                domain,
                service,
                repairs,
            )

        if validation is not None:
            log.warning(
                "HA SERVICE LOCAL VALIDATION FAILED domain=%s service=%s validation=%s",
                domain,
                service,
                {
                    "unknown": validation.get("unknown_parameters"),
                    "missing": validation.get("missing_required_parameters"),
                    "invalid_values": validation.get("invalid_values"),
                },
            )
            return _json_tool_result(
                {
                    "success": False,
                    "domain": domain,
                    "service": service,
                    "entity_id": entity_id,
                    "area_id": area_id,
                    "error_type": "local_schema_validation",
                    "error": "Service arguments do not match the cached Home Assistant schema.",
                    **validation,
                    "repairs_applied": repairs,
                    "retry_instruction": (
                        "Change only the invalid service-data fields. Preserve the "
                        "domain, service, target, return_response, and every already-valid field."
                    ),
                }
            )

        payload: dict[str, Any] = {
            "type": "call_service",
            "domain": domain,
            "service": service,
            "service_data": normalised_data,
            "return_response": bool(return_response),
        }

        target: dict[str, Any] = {}
        if entity_id:
            if isinstance(entity_id, str):
                entity_ids = [
                    item.strip()
                    for item in entity_id.split(",")
                    if item.strip()
                ]
            elif isinstance(entity_id, list):
                entity_ids = [str(item).strip() for item in entity_id if str(item).strip()]
            else:
                entity_ids = [str(entity_id).strip()]
            if entity_ids:
                target["entity_id"] = (
                    entity_ids[0] if len(entity_ids) == 1 else entity_ids
                )
                entity_id = ",".join(entity_ids)

        if area_id:
            target["area_id"] = area_id
        if target:
            payload["target"] = target

        reply = await client.command(payload, timeout=HA_WS_COMMAND_TIMEOUT)
        if reply.get("success") is not True:
            error = reply.get("error")
            message = error.get("message") if isinstance(error, dict) else str(error)
            code = error.get("code") if isinstance(error, dict) else None
            log.warning(
                "HA WS SERVICE FAILED domain=%s service=%s entity=%s code=%s error=%s",
                domain,
                service,
                entity_id,
                code,
                message,
            )
            compact_schema = (
                _compact_service_definition(domain, service, definition)
                if isinstance(definition, dict)
                else None
            )
            return _json_tool_result(
                {
                    "success": False,
                    "domain": domain,
                    "service": service,
                    "entity_id": entity_id,
                    "area_id": area_id,
                    "data": normalised_data,
                    "return_response": bool(return_response),
                    "error": message or "Home Assistant service call failed",
                    "error_code": code,
                    "service_schema": compact_schema,
                    "retry_instruction": (
                        "Do not repeat this call unchanged. Preserve its valid target "
                        "and valid fields; change only what the error identifies."
                    ),
                }
            )

        result = reply.get("result")
        if not isinstance(result, dict):
            result = {}

        response: dict[str, Any] = {
            "success": True,
            "domain": domain,
            "service": service,
            "entity_id": entity_id,
            "area_id": area_id,
            "repairs_applied": repairs,
            "source": "home_assistant_websocket",
        }

        if return_response:
            response["service_response"] = result.get("response")
            voice_tool_run_state.last_successful_data = {
                "domain": domain,
                "service": service,
                "entity_id": entity_id,
                "area_id": area_id,
                "service_response": result.get("response"),
            }

        expected_state = EXPECTED_PRIMARY_STATES.get(service)
        if entity_id and expected_state and "," not in entity_id:
            verified = await client.wait_for_expected_state(
                entity_id,
                expected_state,
                HA_WS_STATE_CONFIRM_TIMEOUT,
            )
            if verified is not None:
                response["verified_state"] = verified

        if entity_id and "," not in entity_id:
            voice_tool_run_state.last_entity_by_domain[domain] = entity_id

        log.info(
            "HA WS SERVICE COMPLETE domain=%s service=%s entity=%s return_response=%s repairs=%d",
            domain,
            service,
            entity_id,
            return_response,
            len(repairs),
        )
        return _json_tool_result(response)
    except Exception as exc:
        log.warning(
            "DIRECT TOOL ha_call_service failed domain=%s service=%s: %s",
            domain,
            service,
            exc,
        )
        return _json_tool_result(
            {
                "success": False,
                "domain": domain,
                "service": service,
                "entity_id": entity_id,
                "area_id": area_id,
                "error": str(exc),
            }
        )


# ---------------------------------------------------------------------------
# Optional advanced ha-mcp specialist
# ---------------------------------------------------------------------------


ADVANCED_INSTRUCTIONS = """
You are the advanced Home Assistant specialist behind a spoken voice assistant.

The normal agent has already decided this request needs capabilities beyond its
fast cached state/search/service tools. Use the available ha-mcp tools to perform
or investigate the requested Home Assistant operation.

Use tool search when available rather than guessing obscure tool names. Never
invent entity IDs or claim success without tool confirmation. Minimize tool calls.
Avoid destructive administration unless the user's request clearly requires it.
Return a concise factual result to the parent agent. Do not add offers for more
help, troubleshooting advice, Markdown, URLs, or long explanations unless the
request itself explicitly asks for technical detail.
""".strip()



def _make_lemonade_model() -> OpenAIChatCompletionsModel:
    client = AsyncOpenAI(
        base_url=LEMONADE_BASE_URL,
        api_key=LEMONADE_API_KEY,
        timeout=FALLBACK_TIMEOUT,
    )
    return OpenAIChatCompletionsModel(
        model=FALLBACK_MODEL,
        openai_client=client,
    )



def make_ha_mcp_server() -> MCPServerStdio:
    child_env = dict(os.environ)
    child_env.update(
        {
            "HOMEASSISTANT_URL": HOMEASSISTANT_URL,
            "HOMEASSISTANT_TOKEN": HOMEASSISTANT_TOKEN,
            "ENABLE_TOOL_SEARCH": "true" if HA_MCP_TOOL_SEARCH else "false",
            "TOOL_SEARCH_MAX_RESULTS": HA_MCP_TOOL_SEARCH_MAX_RESULTS,
            "PINNED_TOOLS": HA_MCP_PINNED_TOOLS,
        }
    )
    return MCPServerStdio(
        name="Home Assistant Advanced",
        params={
            "command": HA_MCP_COMMAND,
            "args": HA_MCP_ARGS,
            "env": child_env,
        },
        cache_tools_list=True,
        client_session_timeout_seconds=FALLBACK_TIMEOUT,
    )



def _parse_music_area_player_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    raw = MUSIC_ASSISTANT_AREA_PLAYER_MAP.strip()
    if not raw:
        return mapping

    for item in raw.split(","):
        item = item.strip()
        if not item or "=" not in item:
            continue
        area, player_id = item.split("=", 1)
        area = area.strip()
        player_id = player_id.strip()
        if area and player_id:
            mapping[area.casefold()] = player_id
    return mapping


def _music_assistant_agent_instructions() -> str:
    mapping = _parse_music_area_player_map()
    mapping_text = ""
    if mapping:
        friendly = ", ".join(
            f"{area} -> {player_id}" for area, player_id in mapping.items()
        )
        mapping_text = (
            "\nConfigured authoritative area-to-player mapping: "
            f"{friendly}. Use it unless the user explicitly names another player."
        )

    return (
        """
MUSIC ASSISTANT AUTHORITY

Native Music Assistant WebSocket tools are connected and are authoritative for
music and speaker playback operations. Do not use Home Assistant media_player
services for Music Assistant music playback.

For ordinary requests of the form "play <music> in/on <room/player>", prefer
ma_play_query. It resolves the player, searches Music Assistant and starts the
selected result in ONE tool call. Pass the explicitly named room in area. If the
request has no explicit room, pass the trusted voice-origin area when available.
Do not call ma_search or ma_list_players before ma_play_query unless the request
actually requires browsing or disambiguation. For an artist-radio request such as
"play Foo Fighters radio", pass query="Foo Fighters radio" and radio_mode=true;
v6.9 will optimize time-to-first-audio and leave continuation to Music Assistant.

Use the lower-level native tools when needed:
- ma_list_players: inspect the event-updated in-memory player topology.
- ma_search / ma_browse: discover media URIs.
- ma_play_media: play a known URI on a known player/queue.
- ma_playback: play/pause/stop/next/previous/toggle.
- ma_volume: set/adjust/mute Music Assistant players.
- ma_group: join/leave speaker sync groups.
- ma_queue / ma_queue_item / ma_transfer_queue: queue operations.

An explicitly named room/player always overrides voice-origin context. Do not
call ha_search merely to discover a Music Assistant player.

After any successful Music Assistant write/dispatch, stop. Do not verify it with
another model-driven write and do not repeat it. The proxy treats the native
result as terminal and returns the spoken confirmation directly.
""".strip()
        + mapping_text
    )


def make_advanced_agent(active_servers) -> Agent:
    return Agent(
        name="Home Assistant advanced specialist",
        model=_make_lemonade_model(),
        instructions=ADVANCED_INSTRUCTIONS,
        mcp_servers=list(active_servers),
    )


@function_tool
async def ha_advanced(request: str) -> str:
    """Use full ha-mcp for advanced Home Assistant work.

    Use only for requests that truly need configuration, automation editing,
    history/traces, dashboards, helpers, integrations, system diagnostics or
    another capability unavailable from the fast state/service tools.

    Args:
        request: Concise self-contained description of the advanced operation.
    """
    if advanced_agent is None:
        return _json_tool_result(
            {
                "success": False,
                "error": "Advanced Home Assistant tools are unavailable.",
            }
        )

    try:
        result = await asyncio.wait_for(
            Runner.run(
                advanced_agent,
                f"{_runtime_context()}\n{_origin_runtime_context({
                    'device_name': voice_tool_run_state.origin_device_name,
                    'area_name': voice_tool_run_state.origin_area_name,
                    'area_id': voice_tool_run_state.origin_area_id,
                })}\n\nAdvanced request: {request.strip()}",
                max_turns=ADVANCED_MAX_TURNS,
            ),
            timeout=FALLBACK_TIMEOUT,
        )
        output = str(result.final_output or "").strip()
        return _json_tool_result(
            {
                "success": bool(output),
                "result": output or "No advanced result was returned.",
            }
        )
    except Exception as exc:
        log.exception("Advanced ha-mcp tool failed")
        return _json_tool_result({"success": False, "error": str(exc)})


# ---------------------------------------------------------------------------
# Fast-agent tool-result handling and spoken cleanup
# ---------------------------------------------------------------------------



def _serialise_tool_output(output) -> str:
    if output is None:
        return ""
    if hasattr(output, "model_dump"):
        try:
            output = output.model_dump()
        except Exception:
            pass
    if isinstance(output, (dict, list, tuple)):
        try:
            return json.dumps(output, ensure_ascii=False, default=str)
        except Exception:
            return str(output)
    return str(output)



def _tool_output_mapping(output) -> dict | None:
    if output is None:
        return None
    if hasattr(output, "model_dump"):
        try:
            output = output.model_dump()
        except Exception:
            pass
    if isinstance(output, dict):
        if set(output.keys()) >= {"text"} and isinstance(output.get("text"), str):
            nested = _tool_output_mapping(output["text"])
            return nested if nested is not None else output
        return output
    if isinstance(output, (list, tuple)):
        for item in output:
            nested = _tool_output_mapping(item)
            if nested is not None:
                return nested
        return None
    if isinstance(output, str):
        text = output.strip()
        if not text:
            return None
        try:
            decoded = json.loads(text)
        except Exception:
            return None
        return decoded if isinstance(decoded, dict) else None
    return None



def _tool_output_failed(output) -> bool:
    payload = _tool_output_mapping(output)
    if isinstance(payload, dict):
        if payload.get("success") is False:
            return True
        if payload.get("success") is True:
            return False
        if payload.get("error"):
            return True
    text = _serialise_tool_output(output).casefold()
    return any(
        marker in text
        for marker in (
            '"success": false',
            "'success': false",
            "error calling tool",
            "toolerror",
            "unauthorized",
            "forbidden",
            "invalid parameter",
            "timed out",
        )
    )

def _ma_enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _ma_name(value: Any) -> str:
    return str(getattr(value, "name", "") or "").strip()


def _ma_uri(value: Any) -> str:
    return str(getattr(value, "uri", "") or "").strip()


def _ma_media_type(value: Any, fallback: str | None = None) -> str | None:
    media_type = getattr(value, "media_type", None)
    if media_type is None:
        return fallback
    return str(_ma_enum_value(media_type))


def _ma_media_summary(item: Any, fallback_type: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": _ma_name(item),
        "uri": _ma_uri(item),
        "media_type": _ma_media_type(item, fallback_type),
    }
    for attr in ("provider", "item_id", "version"):
        value = getattr(item, attr, None)
        if value not in (None, ""):
            payload[attr] = _ma_enum_value(value)

    artists = getattr(item, "artists", None)
    if artists:
        payload["artists"] = [
            _ma_name(artist) for artist in artists if _ma_name(artist)
        ]
    album = getattr(item, "album", None)
    if album is not None and _ma_name(album):
        payload["album"] = _ma_name(album)
    return payload


def _ma_queue_item_summary(item: Any) -> dict[str, Any]:
    media_item = getattr(item, "media_item", None)
    payload = {
        "queue_item_id": str(
            getattr(item, "queue_item_id", None)
            or getattr(item, "item_id", None)
            or ""
        ),
        "name": str(getattr(item, "name", "") or ""),
    }
    if media_item is not None:
        payload["media"] = _ma_media_summary(media_item)
    else:
        uri = str(getattr(item, "uri", "") or "")
        if uri:
            payload["uri"] = uri
    return payload


def _ma_player_summary(client: Any, player: Any) -> dict[str, Any]:
    player_id = str(getattr(player, "player_id", "") or "")
    active_source = str(getattr(player, "active_source", "") or "")
    local_queue = client.player_queues.get(active_source or player_id)
    queue_id = str(getattr(local_queue, "queue_id", "") or "") if local_queue else ""
    payload: dict[str, Any] = {
        "player_id": player_id,
        "name": str(getattr(player, "name", "") or player_id),
        "available": bool(getattr(player, "available", True)),
        "powered": getattr(player, "powered", None),
        "state": _ma_enum_value(getattr(player, "state", None)),
        "volume_level": getattr(player, "volume_level", None),
        "volume_muted": getattr(player, "volume_muted", None),
        "active_source": active_source or None,
        "queue_id": queue_id or active_source or player_id,
        "synced_to": getattr(player, "synced_to", None),
    }
    group_childs = getattr(player, "group_childs", None)
    if group_childs:
        payload["group_childs"] = list(group_childs)
    return payload


def _ma_queue_summary(client: Any, queue_id: str) -> dict[str, Any]:
    queue = client.player_queues.get(queue_id)
    if queue is None:
        return {"queue_id": queue_id, "cached": False}
    current_item = getattr(queue, "current_item", None)
    return {
        "queue_id": str(getattr(queue, "queue_id", queue_id)),
        "display_name": str(
            getattr(queue, "display_name", None)
            or getattr(queue, "name", None)
            or ""
        ),
        "state": _ma_enum_value(getattr(queue, "state", None)),
        "current_index": getattr(queue, "current_index", None),
        "elapsed_time": getattr(queue, "elapsed_time", None),
        "shuffle_enabled": getattr(queue, "shuffle_enabled", None),
        "repeat_mode": _ma_enum_value(getattr(queue, "repeat_mode", None)),
        "current_item": (
            _ma_queue_item_summary(current_item) if current_item is not None else None
        ),
        "cached": True,
    }


@dataclass
class InFlightMusicPlayback:
    queue_id: str
    fingerprint: str
    task: asyncio.Task[Any]
    label: str
    started_at: float


@dataclass(frozen=True)
class MusicPlayDispatchResult:
    command_acknowledged: bool
    command_dispatched: bool
    still_processing: bool
    duplicate_suppressed: bool


class NativeMusicAssistant:
    """Persistent official Music Assistant WebSocket client with reconnects.

    The official client owns the receive loop and maintains players/queues from
    server events. State-changing calls are deliberately NOT auto-retried: once a
    command has been sent, an ambiguous disconnect must not cause duplicate media
    actions. The next user request can execute after the supervisor reconnects.
    """

    def __init__(self, server_url: str, token: str) -> None:
        self.server_url = server_url
        self.token = token
        self.client: Any | None = None
        self.ready = asyncio.Event()
        self._stopping = asyncio.Event()
        self._supervisor_task: asyncio.Task | None = None
        self._tool_lock = asyncio.Lock()
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._inflight_playbacks: dict[tuple[str, str], InFlightMusicPlayback] = {}
        # Voice UX state only: last chosen artist-radio seed avoids immediate
        # repetition, while queue generations stop superseded background radio
        # lifecycles from enabling DSTM for a newer playback request.
        self._last_radio_seed_by_artist: dict[str, str] = {}
        self._queue_playback_generation: dict[str, int] = {}
        self.last_error: str | None = None
        self.last_connected_at: float | None = None
        self.connection_count = 0
        self.reconnect_count = 0

    @property
    def connected(self) -> bool:
        client = self.client
        return bool(
            client is not None
            and self.ready.is_set()
            and getattr(getattr(client, "connection", None), "connected", False)
        )

    async def start(self) -> bool:
        if MusicAssistantClient is None:
            self.last_error = MUSIC_ASSISTANT_CLIENT_IMPORT_ERROR or "music-assistant-client unavailable"
            log.error("Music Assistant native client unavailable: %s", self.last_error)
            return False
        if self._supervisor_task is None or self._supervisor_task.done():
            self._stopping.clear()
            self._supervisor_task = asyncio.create_task(
                self._supervisor(), name="music-assistant-native-supervisor"
            )
        try:
            await asyncio.wait_for(
                self.ready.wait(), timeout=MUSIC_ASSISTANT_CONNECT_TIMEOUT_SECONDS
            )
            return True
        except asyncio.TimeoutError:
            log.warning(
                "Music Assistant native initial connection not ready after %.1fs; "
                "supervisor will keep retrying",
                MUSIC_ASSISTANT_CONNECT_TIMEOUT_SECONDS,
            )
            return False

    async def stop(self) -> None:
        self._stopping.set()
        task = self._supervisor_task
        self._supervisor_task = None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        background = list(self._background_tasks)
        self._background_tasks.clear()
        self._inflight_playbacks.clear()
        self._last_radio_seed_by_artist.clear()
        self._queue_playback_generation.clear()
        for pending in background:
            pending.cancel()
        if background:
            await asyncio.gather(*background, return_exceptions=True)
        self.ready.clear()
        self.client = None

    @property
    def inflight_playback_count(self) -> int:
        return sum(1 for item in self._inflight_playbacks.values() if not item.task.done())

    def get_inflight_playback(
        self, queue_id: str, fingerprint: str
    ) -> InFlightMusicPlayback | None:
        key = (queue_id, fingerprint)
        record = self._inflight_playbacks.get(key)
        if record is not None and record.task.done():
            self._inflight_playbacks.pop(key, None)
            return None
        return record

    def register_inflight_playback(
        self,
        queue_id: str,
        fingerprint: str,
        task: asyncio.Task[Any],
        label: str,
    ) -> InFlightMusicPlayback:
        record = InFlightMusicPlayback(
            queue_id=queue_id,
            fingerprint=fingerprint,
            task=task,
            label=label,
            started_at=time.time(),
        )
        self._inflight_playbacks[(queue_id, fingerprint)] = record
        return record

    def clear_inflight_playback(
        self, queue_id: str, fingerprint: str, task: asyncio.Task[Any]
    ) -> None:
        key = (queue_id, fingerprint)
        current = self._inflight_playbacks.get(key)
        if current is not None and current.task is task:
            self._inflight_playbacks.pop(key, None)

    def begin_queue_playback_generation(self, queue_id: str) -> int:
        generation = self._queue_playback_generation.get(queue_id, 0) + 1
        self._queue_playback_generation[queue_id] = generation
        return generation

    def is_current_queue_playback_generation(self, queue_id: str, generation: int) -> bool:
        return self._queue_playback_generation.get(queue_id, 0) == generation

    def last_radio_seed(self, artist_uri: str) -> str | None:
        return self._last_radio_seed_by_artist.get(artist_uri)

    def remember_radio_seed(self, artist_uri: str, seed_uri: str) -> None:
        if artist_uri and seed_uri:
            self._last_radio_seed_by_artist[artist_uri] = seed_uri

    def track_background_task(self, task: asyncio.Task[Any], *, label: str) -> None:
        """Keep a long-running MA command alive and log its eventual result."""
        self._background_tasks.add(task)

        def _done(completed: asyncio.Task[Any]) -> None:
            self._background_tasks.discard(completed)
            if completed.cancelled():
                log.info("MA NATIVE ASYNC COMMAND CANCELLED label=%s", label)
                return
            try:
                completed.result()
            except Exception as exc:
                log.error(
                    "MA NATIVE ASYNC COMMAND FAILED label=%s error=%s: %s",
                    label,
                    type(exc).__name__,
                    exc,
                )
            else:
                log.info("MA NATIVE ASYNC COMMAND COMPLETE label=%s", label)

        task.add_done_callback(_done)

    async def _supervisor(self) -> None:
        first_connection = True
        while not self._stopping.is_set():
            client: Any | None = None
            listener_task: asyncio.Task | None = None
            ready_wait_task: asyncio.Task | None = None
            try:
                assert MusicAssistantClient is not None
                client = MusicAssistantClient(
                    self.server_url,
                    None,
                    token=self.token,
                )
                self.client = client
                initial_state_ready = asyncio.Event()
                listener_task = asyncio.create_task(
                    client.start_listening(initial_state_ready),
                    name="music-assistant-native-listener",
                )
                ready_wait_task = asyncio.create_task(initial_state_ready.wait())

                done, _ = await asyncio.wait(
                    {listener_task, ready_wait_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if listener_task in done:
                    # Propagate connection/auth/schema errors immediately.
                    await listener_task
                    raise RuntimeError(
                        "Music Assistant listener exited before initial state was ready"
                    )

                await ready_wait_task
                self.ready.set()
                self.last_error = None
                self.last_connected_at = time.time()
                self.connection_count += 1
                if not first_connection:
                    self.reconnect_count += 1
                first_connection = False

                info = getattr(client, "server_info", None)
                log.info(
                    "Music Assistant native WebSocket ready url=%r version=%r schema=%r "
                    "players=%s queues=%s",
                    self.server_url,
                    getattr(info, "server_version", None),
                    getattr(info, "schema_version", None),
                    len(getattr(client.players, "players", []) or []),
                    len(getattr(client.player_queues, "player_queues", []) or []),
                )

                # Blocks until the websocket is closed or stop() cancels us.
                await listener_task
                if not self._stopping.is_set():
                    raise RuntimeError("Music Assistant WebSocket listener disconnected")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                log.warning("Music Assistant native connection lost: %s", self.last_error)
            finally:
                self.ready.clear()
                if ready_wait_task is not None and not ready_wait_task.done():
                    ready_wait_task.cancel()
                    await asyncio.gather(ready_wait_task, return_exceptions=True)
                if listener_task is not None and not listener_task.done():
                    listener_task.cancel()
                    await asyncio.gather(listener_task, return_exceptions=True)
                if client is not None:
                    try:
                        await client.disconnect()
                    except Exception:
                        pass
                if self.client is client:
                    self.client = None

            if not self._stopping.is_set():
                try:
                    await asyncio.wait_for(
                        self._stopping.wait(),
                        timeout=MUSIC_ASSISTANT_RECONNECT_DELAY_SECONDS,
                    )
                except asyncio.TimeoutError:
                    pass

    async def wait_ready(self) -> Any:
        if not self.connected:
            try:
                await asyncio.wait_for(
                    self.ready.wait(),
                    timeout=MUSIC_ASSISTANT_COMMAND_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError as exc:
                raise RuntimeError("Music Assistant is not connected") from exc
        client = self.client
        if client is None or not self.connected:
            raise RuntimeError("Music Assistant is not connected")
        return client

    async def run_serialized(self, tool_name: str, operation) -> Any:
        # Wait outside the lock so one disconnected request does not block all
        # callers from observing a reconnect.
        await self.wait_ready()
        async with self._tool_lock:
            client = await self.wait_ready()
            log.info("MA NATIVE CALL tool=%s", tool_name)
            return await asyncio.wait_for(
                operation(client),
                timeout=MUSIC_ASSISTANT_COMMAND_TIMEOUT_SECONDS,
            )


def _ma_native_manager() -> NativeMusicAssistant:
    manager = music_assistant_native
    if manager is None:
        raise RuntimeError("Music Assistant native client is unavailable")
    return manager


async def _ma_dispatch_play_media(
    manager: NativeMusicAssistant,
    client: Any,
    *,
    queue_id: str,
    media: str | list[str],
    option: Any,
    radio_mode: bool,
    label: str,
    origin_context: dict[str, Any] | None = None,
) -> MusicPlayDispatchResult:
    """Optimistically dispatch long-running play_media for voice interactions.

    Music Assistant 2.8.x may resolve an artist/playlist, generate radio tracks,
    build the queue, resolve stream details and prepare audio before the command
    returns. A short ACK window catches immediate success/errors. If the command
    is still running, the voice request succeeds optimistically while the MA
    future stays alive. Identical in-flight writes for the same queue/media are
    suppressed to avoid impatient voice retries duplicating playback requests.
    """
    media_payload = media if isinstance(media, list) else [media]
    fingerprint_payload = {
        "queue_id": queue_id,
        "media": [str(item) for item in media_payload],
        "option": str(getattr(option, "value", option)),
        "radio_mode": bool(radio_mode),
    }
    fingerprint = json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":"))

    existing = manager.get_inflight_playback(queue_id, fingerprint)
    if existing is not None:
        log.info(
            "MA NATIVE PLAY_MEDIA DUPLICATE SUPPRESSED queue_id=%r label=%s age=%.1fs",
            queue_id,
            existing.label,
            max(0.0, time.time() - existing.started_at),
        )
        return MusicPlayDispatchResult(
            command_acknowledged=False,
            command_dispatched=False,
            still_processing=True,
            duplicate_suppressed=True,
        )

    generation = manager.begin_queue_playback_generation(queue_id)

    log.info(
        "MA NATIVE PLAY_MEDIA DISPATCH queue_id=%r media_payload=%r option=%s "
        "radio_mode=%s ack_wait=%.1fs",
        queue_id,
        media_payload,
        getattr(option, "value", option),
        radio_mode,
        MUSIC_ASSISTANT_PLAY_ACK_WAIT_SECONDS,
    )

    async def _complete() -> None:
        await asyncio.wait_for(
            client.player_queues.play_media(
                queue_id=queue_id,
                media=media_payload,
                option=option,
                radio_mode=bool(radio_mode),
            ),
            timeout=MUSIC_ASSISTANT_PLAY_COMPLETION_TIMEOUT_SECONDS,
        )

    task = asyncio.create_task(_complete(), name=f"ma-play-media-{queue_id}")
    manager.register_inflight_playback(queue_id, fingerprint, task, label)
    try:
        await asyncio.wait_for(
            asyncio.shield(task),
            timeout=MUSIC_ASSISTANT_PLAY_ACK_WAIT_SECONDS,
        )
        manager.clear_inflight_playback(queue_id, fingerprint, task)
        log.info("MA NATIVE PLAY_MEDIA ACKNOWLEDGED queue_id=%r", queue_id)
        return MusicPlayDispatchResult(
            command_acknowledged=True,
            command_dispatched=True,
            still_processing=False,
            duplicate_suppressed=False,
        )
    except asyncio.TimeoutError:
        async def _background_lifecycle() -> None:
            indicator_handle: SatelliteIndicatorHandle | None = None
            try:
                # LED resolution/control happens only after the voice ACK window
                # and never delays the optimistic spoken response.
                indicator_handle = await voice_activity_indicators.begin(origin_context)
                await task
            finally:
                manager.clear_inflight_playback(queue_id, fingerprint, task)
                await voice_activity_indicators.end(indicator_handle)

        lifecycle = asyncio.create_task(
            _background_lifecycle(),
            name=f"ma-play-lifecycle-{queue_id}",
        )
        manager.track_background_task(lifecycle, label=label)
        log.info(
            "MA NATIVE PLAY_MEDIA STILL PROCESSING queue_id=%r; optimistic voice ACK; "
            "inflight=%d",
            queue_id,
            manager.inflight_playback_count,
        )
        return MusicPlayDispatchResult(
            command_acknowledged=False,
            command_dispatched=True,
            still_processing=True,
            duplicate_suppressed=False,
        )
    except Exception:
        manager.clear_inflight_playback(queue_id, fingerprint, task)
        raise


def _ma_media_provider_and_item_id(item: Any) -> tuple[str, str]:
    """Return provider + item ID needed by MA item-specific controller calls."""
    provider = str(_ma_enum_value(getattr(item, "provider", "")) or "").strip()
    item_id = str(getattr(item, "item_id", "") or "").strip()
    uri = _ma_uri(item)
    if uri and "://" in uri:
        scheme, rest = uri.split("://", 1)
        if not provider:
            provider = scheme.strip()
        if not item_id and "/" in rest:
            # MA URI shape: provider://media_type/item_id. Preserve any
            # additional slashes in provider-specific item IDs.
            item_id = rest.split("/", 1)[1].strip()
    return provider, item_id


def _ma_queue_playback_marker(client: Any, queue_id: str) -> tuple[str, str, str, str]:
    """Stable-enough cached marker used to distinguish new audio from old audio."""
    queue = client.player_queues.get(queue_id)
    if queue is None:
        return ("", "", "", "")
    current = getattr(queue, "current_item", None)
    if current is None:
        return (
            str(_ma_enum_value(getattr(queue, "state", "")) or "").casefold(),
            "",
            "",
            "",
        )
    media_item = getattr(current, "media_item", None)
    uri = _ma_uri(media_item) if media_item is not None else ""
    if not uri:
        uri = str(getattr(current, "uri", "") or "").strip()
    queue_item_id = str(
        getattr(current, "queue_item_id", None)
        or getattr(current, "item_id", None)
        or ""
    )
    name = str(getattr(current, "name", "") or _ma_name(media_item) or "").strip()
    state = str(_ma_enum_value(getattr(queue, "state", "")) or "").casefold()
    return (state, queue_item_id, uri, name.casefold())


def _ma_marker_is_playing(marker: tuple[str, str, str, str]) -> bool:
    return marker[0] == "playing"


def _ma_first_audio_matches(
    marker: tuple[str, str, str, str],
    *,
    baseline: tuple[str, str, str, str],
    expected_uri: str | None = None,
    expected_name: str | None = None,
) -> bool:
    if not _ma_marker_is_playing(marker):
        return False
    # If the player was not already playing, any new PLAYING state is enough.
    if not _ma_marker_is_playing(baseline):
        return True
    # When replacing already-playing audio, require evidence that the current
    # item changed so an old PLAYING state cannot satisfy the watcher instantly.
    if marker[1:] != baseline[1:]:
        if expected_uri and marker[2] and marker[2] == expected_uri:
            return True
        if expected_name and marker[3] and marker[3] == expected_name.casefold():
            return True
        # Queue item IDs are regenerated by REPLACE and are the most reliable
        # generic signal when MA normalizes a library URI to a provider URI.
        if marker[1] and marker[1] != baseline[1]:
            return True
        if marker[2] and marker[2] != baseline[2]:
            return True
        if marker[3] and marker[3] != baseline[3]:
            return True
    return False


async def _ma_wait_for_first_audio(
    client: Any,
    *,
    queue_id: str,
    baseline: tuple[str, str, str, str],
    expected_uri: str | None,
    expected_name: str | None,
    timeout: float,
    manager: NativeMusicAssistant | None = None,
    generation: int | None = None,
) -> tuple[bool, float | None, tuple[str, str, str, str]]:
    """Watch MA's event-updated queue cache for the requested audio to start."""
    started = time.monotonic()
    deadline = started + timeout
    last_marker = baseline
    while True:
        if (
            manager is not None
            and generation is not None
            and not manager.is_current_queue_playback_generation(queue_id, generation)
        ):
            return False, None, last_marker
        marker = _ma_queue_playback_marker(client, queue_id)
        last_marker = marker
        if _ma_first_audio_matches(
            marker,
            baseline=baseline,
            expected_uri=expected_uri,
            expected_name=expected_name,
        ):
            elapsed = max(0.0, time.monotonic() - started)
            return True, elapsed, marker
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False, None, last_marker
        await asyncio.sleep(min(MUSIC_ASSISTANT_FIRST_AUDIO_POLL_SECONDS, remaining))


def _ma_radio_seed_candidates(tracks: list[Any]) -> list[Any]:
    candidates: list[Any] = []
    seen: set[str] = set()
    for track in tracks:
        uri = _ma_uri(track)
        if not uri or uri in seen:
            continue
        if getattr(track, "available", True) is False:
            continue
        seen.add(uri)
        candidates.append(track)
        if len(candidates) >= MUSIC_ASSISTANT_RADIO_SEED_TOP_N:
            break
    return candidates


def _ma_choose_radio_seed(
    manager: NativeMusicAssistant,
    *,
    artist_uri: str,
    tracks: list[Any],
) -> tuple[Any, int, list[dict[str, Any]]]:
    candidates = _ma_radio_seed_candidates(tracks)
    if not candidates:
        raise ValueError("Music Assistant returned no available top tracks for the artist")

    previous_uri = manager.last_radio_seed(artist_uri)
    pool = candidates
    if previous_uri and len(candidates) > 1:
        without_previous = [item for item in candidates if _ma_uri(item) != previous_uri]
        if without_previous:
            pool = without_previous

    strategy = MUSIC_ASSISTANT_RADIO_SEED_STRATEGY
    if strategy == "first":
        selected = pool[0]
    elif strategy == "random":
        selected = random.choice(pool)
    else:
        # Weighted randomness preserves MA's top-track ordering without always
        # starting track #1. Candidate rank is taken from MA, not invented here.
        original_rank = {_ma_uri(item): idx for idx, item in enumerate(candidates)}
        weights = [max(1, MUSIC_ASSISTANT_RADIO_SEED_TOP_N - original_rank[_ma_uri(item)]) for item in pool]
        selected = random.choices(pool, weights=weights, k=1)[0]

    selected_uri = _ma_uri(selected)
    manager.remember_radio_seed(artist_uri, selected_uri)
    rank = next((idx + 1 for idx, item in enumerate(candidates) if _ma_uri(item) == selected_uri), 1)
    return selected, rank, [_ma_media_summary(item, "track") for item in candidates]


@dataclass(frozen=True)
class FastRadioDispatchResult:
    command_acknowledged: bool
    command_dispatched: bool
    still_processing: bool
    duplicate_suppressed: bool
    first_audio_observed: bool
    first_audio_seconds: float | None
    seed: dict[str, Any] | None
    seed_rank: int | None
    seed_pool: list[dict[str, Any]]


async def _ma_dispatch_fast_artist_radio(
    manager: NativeMusicAssistant,
    client: Any,
    *,
    queue_id: str,
    artist: Any,
    origin_context: dict[str, Any] | None,
    label: str,
) -> FastRadioDispatchResult:
    """Start one MA-provided artist seed ASAP, then let MA own radio continuation."""
    artist_uri = _ma_uri(artist)
    provider, artist_item_id = _ma_media_provider_and_item_id(artist)
    if not artist_uri or not provider or not artist_item_id:
        raise ValueError("Selected Music Assistant artist is missing URI/provider/item_id")

    fingerprint = json.dumps(
        {"queue_id": queue_id, "mode": "fast_artist_radio", "artist": artist_uri},
        sort_keys=True,
        separators=(",", ":"),
    )
    existing = manager.get_inflight_playback(queue_id, fingerprint)
    if existing is not None:
        log.info(
            "MA FAST RADIO DUPLICATE SUPPRESSED queue_id=%r artist=%r age=%.1fs",
            queue_id,
            artist_uri,
            max(0.0, time.time() - existing.started_at),
        )
        return FastRadioDispatchResult(
            command_acknowledged=False,
            command_dispatched=False,
            still_processing=True,
            duplicate_suppressed=True,
            first_audio_observed=False,
            first_audio_seconds=None,
            seed=None,
            seed_rank=None,
            seed_pool=[],
        )

    bootstrap_started = time.monotonic()
    top_tracks_started = bootstrap_started
    tracks = await client.music.get_artist_tracks(
        item_id=artist_item_id,
        provider_instance_id_or_domain=provider,
        in_library_only=False,
    )
    top_tracks_ms = (time.monotonic() - top_tracks_started) * 1000.0
    seed, seed_rank, seed_pool = _ma_choose_radio_seed(
        manager,
        artist_uri=artist_uri,
        tracks=list(tracks),
    )
    seed_uri = _ma_uri(seed)
    seed_name = _ma_name(seed)
    baseline = _ma_queue_playback_marker(client, queue_id)
    generation = manager.begin_queue_playback_generation(queue_id)

    log.info(
        "MA FAST RADIO SEED queue_id=%r artist=%r provider=%r top_tracks_ms=%.1f "
        "seed=%r seed_name=%r seed_rank=%s pool=%s strategy=%s generation=%s",
        queue_id,
        artist_uri,
        provider,
        top_tracks_ms,
        seed_uri,
        seed_name,
        seed_rank,
        len(seed_pool),
        MUSIC_ASSISTANT_RADIO_SEED_STRATEGY,
        generation,
    )

    assert QueueOption is not None

    async def _play_seed() -> None:
        await asyncio.wait_for(
            client.player_queues.play_media(
                queue_id=queue_id,
                media=[seed_uri],
                option=QueueOption.REPLACE,
                radio_mode=False,
            ),
            timeout=MUSIC_ASSISTANT_BACKGROUND_TIMEOUT_SECONDS,
        )

    command_task = asyncio.create_task(_play_seed(), name=f"ma-fast-radio-play-{queue_id}")

    dedupe_release = asyncio.Event()

    async def _hold_dedupe() -> None:
        await dedupe_release.wait()

    dedupe_task = asyncio.create_task(_hold_dedupe(), name=f"ma-fast-radio-dedupe-{queue_id}")
    manager.register_inflight_playback(queue_id, fingerprint, dedupe_task, label)
    manager.track_background_task(dedupe_task, label=f"{label}:dedupe")

    audio_task = asyncio.create_task(
        _ma_wait_for_first_audio(
            client,
            queue_id=queue_id,
            baseline=baseline,
            expected_uri=seed_uri,
            expected_name=seed_name,
            timeout=MUSIC_ASSISTANT_BACKGROUND_TIMEOUT_SECONDS,
            manager=manager,
            generation=generation,
        ),
        name=f"ma-fast-radio-first-audio-{queue_id}",
    )

    # Give immediate command errors or first-audio success a short opportunity
    # to surface, but never make recommendation/radio generation part of the
    # synchronous voice response path.
    done, _ = await asyncio.wait(
        {command_task, audio_task},
        timeout=MUSIC_ASSISTANT_PLAY_ACK_WAIT_SECONDS,
        return_when=asyncio.FIRST_COMPLETED,
    )
    if command_task in done:
        try:
            command_task.result()
        except Exception:
            manager.clear_inflight_playback(queue_id, fingerprint, dedupe_task)
            dedupe_release.set()
            audio_task.cancel()
            await asyncio.gather(audio_task, return_exceptions=True)
            raise

    immediate_audio = False
    immediate_audio_seconds: float | None = None
    if audio_task in done:
        immediate_audio, immediate_audio_seconds, marker = audio_task.result()
        if immediate_audio:
            log.info(
                "MA TIME_TO_FIRST_AUDIO queue_id=%r artist=%r seed=%r seconds=%.3f marker=%r",
                queue_id,
                artist_uri,
                seed_uri,
                immediate_audio_seconds or 0.0,
                marker,
            )

    async def _command_cleanup() -> None:
        await command_task

    if not command_task.done():
        cleanup = asyncio.create_task(
            _command_cleanup(), name=f"ma-fast-radio-command-cleanup-{queue_id}"
        )
        manager.track_background_task(cleanup, label=f"{label}:seed_play")

    async def _radio_lifecycle() -> None:
        indicator_handle: SatelliteIndicatorHandle | None = None
        first_audio = immediate_audio
        first_audio_seconds = immediate_audio_seconds
        try:
            if not first_audio:
                # Indicator represents waiting for audible playback, not waiting
                # for MA's recommendation engine or play_media response to end.
                indicator_handle = await voice_activity_indicators.begin(origin_context)
                try:
                    ux_remaining = max(
                        0.05,
                        MUSIC_ASSISTANT_FIRST_AUDIO_TIMEOUT_SECONDS
                        - (time.monotonic() - bootstrap_started),
                    )
                    first_audio, first_audio_seconds, marker = await asyncio.wait_for(
                        asyncio.shield(audio_task),
                        timeout=ux_remaining,
                    )
                except asyncio.TimeoutError:
                    log.warning(
                        "MA FIRST AUDIO UX TIMEOUT queue_id=%r artist=%r seed=%r timeout=%.1fs; "
                        "indicator will restore while background watcher continues",
                        queue_id,
                        artist_uri,
                        seed_uri,
                        MUSIC_ASSISTANT_FIRST_AUDIO_TIMEOUT_SECONDS,
                    )
                    await voice_activity_indicators.end(indicator_handle)
                    indicator_handle = None
                    # Continue waiting without holding the LED indefinitely.
                    first_audio, first_audio_seconds, marker = await audio_task
                if first_audio:
                    log.info(
                        "MA TIME_TO_FIRST_AUDIO queue_id=%r artist=%r seed=%r seconds=%.3f marker=%r",
                        queue_id,
                        artist_uri,
                        seed_uri,
                        first_audio_seconds or 0.0,
                        marker,
                    )

            if not first_audio:
                return
            if not manager.is_current_queue_playback_generation(queue_id, generation):
                log.info(
                    "MA FAST RADIO continuation skipped: request superseded queue_id=%r generation=%s",
                    queue_id,
                    generation,
                )
                return

            # MA owns radio continuation. On 2.8.7 this is a quick setting change
            # that schedules the expensive similar-track fill asynchronously.
            await asyncio.wait_for(
                client.player_queues.dont_stop_the_music(
                    queue_id=queue_id,
                    dont_stop_the_music_enabled=True,
                ),
                timeout=MUSIC_ASSISTANT_COMMAND_TIMEOUT_SECONDS,
            )
            log.info(
                "MA FAST RADIO DSTM ENABLED queue_id=%r artist=%r seed=%r generation=%s",
                queue_id,
                artist_uri,
                seed_uri,
                generation,
            )
        finally:
            await voice_activity_indicators.end(indicator_handle)
            manager.clear_inflight_playback(queue_id, fingerprint, dedupe_task)
            dedupe_release.set()

    lifecycle = asyncio.create_task(
        _radio_lifecycle(), name=f"ma-fast-radio-lifecycle-{queue_id}"
    )
    manager.track_background_task(lifecycle, label=f"{label}:radio_lifecycle")

    return FastRadioDispatchResult(
        command_acknowledged=command_task.done() and not command_task.cancelled() and command_task.exception() is None,
        command_dispatched=True,
        still_processing=not command_task.done(),
        duplicate_suppressed=False,
        first_audio_observed=immediate_audio,
        first_audio_seconds=immediate_audio_seconds,
        seed=_ma_media_summary(seed, "track"),
        seed_rank=seed_rank,
        seed_pool=seed_pool,
    )


def _ma_media_type_lookup() -> dict[str, Any]:
    if MediaType is None:
        raise RuntimeError("music-assistant-models is unavailable")
    return {
        "artist": MediaType.ARTIST,
        "album": MediaType.ALBUM,
        "track": MediaType.TRACK,
        "playlist": MediaType.PLAYLIST,
        "radio": MediaType.RADIO,
        "podcast": MediaType.PODCAST,
        "audiobook": MediaType.AUDIOBOOK,
    }


def _ma_default_play_query_media_types() -> list[Any]:
    lookup = _ma_media_type_lookup()
    return [
        lookup["artist"],
        lookup["album"],
        lookup["track"],
        lookup["playlist"],
    ]


def _ma_media_types(media_types: list[str] | None) -> list[Any]:
    lookup = _ma_media_type_lookup()
    if not media_types:
        return _ma_default_play_query_media_types()
    result = []
    aliases = {
        "song": "track",
        "songs": "track",
        "tracks": "track",
        "artists": "artist",
        "albums": "album",
        "playlists": "playlist",
        "radio_station": "radio",
        "radio_stations": "radio",
        "podcasts": "podcast",
        "audiobooks": "audiobook",
    }
    for raw in media_types:
        value = str(raw or "").strip().casefold().replace(" ", "_")
        value = aliases.get(value, value)
        enum_value = lookup.get(value)
        if enum_value is None:
            raise ValueError(
                "Unsupported Music Assistant media type: "
                f"{raw!r}; allowed values are {', '.join(sorted(lookup))}"
            )
        if enum_value not in result:
            result.append(enum_value)
    return result


def _ma_search_groups(results: Any) -> list[tuple[str, str, list[Any]]]:
    groups: list[tuple[str, str, list[Any]]] = []
    specs = (
        ("artists", "artist", ("artists",)),
        ("albums", "album", ("albums",)),
        ("tracks", "track", ("tracks",)),
        ("playlists", "playlist", ("playlists",)),
        ("radio", "radio", ("radio", "radios")),
        ("podcasts", "podcast", ("podcasts",)),
        ("audiobooks", "audiobook", ("audiobooks",)),
    )
    for label, media_type, attrs in specs:
        values = None
        for attr in attrs:
            values = getattr(results, attr, None)
            if values is not None:
                break
        if values:
            groups.append((label, media_type, list(values)))
    return groups


def _ma_search_payload(results: Any, limit: int) -> dict[str, list[dict[str, Any]]]:
    payload: dict[str, list[dict[str, Any]]] = {}
    for label, media_type, items in _ma_search_groups(results):
        payload[label] = [
            _ma_media_summary(item, media_type) for item in items[:limit]
        ]
    return payload


def _ma_media_type_label(media_type: Any) -> str:
    value = getattr(media_type, "value", media_type)
    return str(value or "").strip().casefold()


async def _ma_search_compatible(
    client: Any,
    *,
    query: str,
    media_types: list[Any],
    limit: int,
) -> Any:
    try:
        return await client.music.search(
            search_query=query,
            media_types=media_types,
            limit=limit,
        )
    except Exception as exc:
        if "NotImplementedError" not in str(exc):
            raise
        log.warning(
            "MA NATIVE SEARCH compatibility fallback query=%r media_types=%s error=%s",
            query,
            [_ma_media_type_label(item) for item in media_types],
            exc,
        )

    class _MergedSearchResults:
        pass

    merged = _MergedSearchResults()
    # Match all attribute names understood by _ma_search_groups().
    for attr in (
        "artists",
        "albums",
        "tracks",
        "playlists",
        "radio",
        "radios",
        "podcasts",
        "audiobooks",
    ):
        setattr(merged, attr, [])

    successes = 0
    skipped: list[str] = []
    for media_type in media_types:
        label = _ma_media_type_label(media_type)
        try:
            result = await client.music.search(
                search_query=query,
                media_types=[media_type],
                limit=limit,
            )
        except Exception as exc:
            if "NotImplementedError" in str(exc):
                skipped.append(label)
                log.info(
                    "MA NATIVE SEARCH media type unsupported on server query=%r media_type=%s",
                    query,
                    label,
                )
                continue
            raise

        successes += 1
        for attr in (
            "artists",
            "albums",
            "tracks",
            "playlists",
            "radio",
            "radios",
            "podcasts",
            "audiobooks",
        ):
            values = getattr(result, attr, None)
            if values:
                target = getattr(merged, attr)
                seen = {
                    (_ma_uri(existing), _ma_name(existing))
                    for existing in target
                }
                for item in values:
                    key = (_ma_uri(item), _ma_name(item))
                    if key not in seen:
                        target.append(item)
                        seen.add(key)

    if not successes:
        raise RuntimeError(
            "Music Assistant search is not implemented for the requested media types"
        )

    if skipped:
        log.info(
            "MA NATIVE SEARCH compatibility result query=%r skipped_media_types=%s",
            query,
            skipped,
        )
    return merged


def _ma_select_search_item(results: Any, query: str) -> tuple[Any, float] | tuple[None, float]:
    q = _normalise_search_text(query)
    if not q:
        return None, 0.0

    type_bias = {
        "artist": 25.0,
        "track": 20.0,
        "playlist": 10.0,
        "album": 5.0,
        "radio": 0.0,
        "podcast": -5.0,
        "audiobook": -5.0,
    }
    explicit_hint: str | None = None
    q_words = set(q.split())
    for word, media_type in (
        ("artist", "artist"),
        ("album", "album"),
        ("playlist", "playlist"),
        ("song", "track"),
        ("track", "track"),
        ("radio", "radio"),
    ):
        if word in q_words:
            explicit_hint = media_type
            break

    best_item = None
    best_score = float("-inf")
    for _, fallback_type, items in _ma_search_groups(results):
        for index, item in enumerate(items):
            name = _normalise_search_text(_ma_name(item))
            if not name:
                continue
            media_type = _ma_media_type(item, fallback_type) or fallback_type
            similarity = difflib.SequenceMatcher(None, q, name).ratio() * 100.0
            score = similarity + type_bias.get(media_type, 0.0)
            if name == q:
                score += 100.0
            elif q in name or name in q:
                score += 30.0
            if explicit_hint:
                score += 100.0 if media_type == explicit_hint else -20.0
            uri = _ma_uri(item)
            if uri.startswith("library://"):
                score += 4.0
            score -= index * 0.05
            if score > best_score:
                best_item = item
                best_score = score
    return best_item, best_score


def _ma_player_match_score(player: Any, target: str) -> float:
    target_norm = _normalise_search_text(target)
    player_id = _normalise_search_text(getattr(player, "player_id", ""))
    name = _normalise_search_text(getattr(player, "name", ""))
    if not target_norm:
        return 0.0
    if target_norm in {player_id, name}:
        score = 1000.0
    elif name.startswith(target_norm) or player_id.startswith(target_norm):
        score = 850.0
    elif re.search(rf"\\b{re.escape(target_norm)}\\b", name):
        score = 800.0
    elif target_norm in name or target_norm in player_id:
        score = 700.0
    else:
        score = max(
            difflib.SequenceMatcher(None, target_norm, name).ratio(),
            difflib.SequenceMatcher(None, target_norm, player_id).ratio(),
        ) * 500.0
    if not bool(getattr(player, "available", True)):
        score -= 400.0
    return score


def _ma_resolve_player(client: Any, *, area: str | None = None, player_id: str | None = None) -> tuple[Any, str]:
    players = list(client.players)
    if not players:
        raise RuntimeError("Music Assistant has no players")

    mapping = _parse_music_area_player_map()
    if area:
        mapped = mapping.get(str(area).strip().casefold())
        if mapped:
            player = client.players.get(mapped)
            if player is not None:
                return player, "configured_area_map"
            log.warning(
                "MA NATIVE configured area mapping points to missing player area=%r player_id=%r",
                area,
                mapped,
            )

    if player_id:
        exact = client.players.get(str(player_id))
        if exact is not None:
            return exact, "explicit_player_id"

    target = str(player_id or area or "").strip()
    if not target:
        origin = str(voice_tool_run_state.origin_area_name or "").strip()
        if origin:
            target = origin
    if not target:
        available = [p for p in players if bool(getattr(p, "available", True))]
        if len(available) == 1:
            return available[0], "single_available_player"
        raise ValueError("A Music Assistant room/player is required")

    ranked = sorted(
        ((_ma_player_match_score(player, target), player) for player in players),
        key=lambda item: item[0],
        reverse=True,
    )
    score, player = ranked[0]
    if score < 275.0:
        raise ValueError(f"No Music Assistant player matches {target!r}")
    return player, "name_match"


async def _ma_resolve_queue_id(client: Any, player_id: str) -> str:
    active = await client.player_queues.get_active_queue(player_id)
    if active is not None and getattr(active, "queue_id", None):
        return str(active.queue_id)
    player = client.players.get(player_id)
    if player is not None:
        active_source = str(getattr(player, "active_source", "") or "")
        if active_source and client.player_queues.get(active_source) is not None:
            return active_source
    return player_id


async def _ma_post_action_verification(client: Any, queue_id: str) -> dict[str, Any]:
    if MUSIC_ASSISTANT_POST_ACTION_SETTLE_SECONDS:
        await asyncio.sleep(MUSIC_ASSISTANT_POST_ACTION_SETTLE_SECONDS)
    snapshot = _ma_queue_summary(client, queue_id)
    try:
        items = await client.player_queues.get_queue_items(queue_id, limit=5, offset=0)
        snapshot["first_items"] = [_ma_queue_item_summary(item) for item in items]
    except Exception as exc:
        # Never turn a confirmed state-changing server ACK into a failure merely
        # because post-action inspection failed. That could cause a replay.
        snapshot["verification_error"] = f"{type(exc).__name__}: {exc}"
    return snapshot


@function_tool
async def ma_list_players() -> str:
    """List Music Assistant players from the live event-updated WebSocket state."""
    try:
        manager = _ma_native_manager()
        client = await manager.wait_ready()
        players = [_ma_player_summary(client, player) for player in client.players]
        return _json_tool_result(
            {
                "success": True,
                "transport": "native_websocket",
                "players": players,
            }
        )
    except Exception as exc:
        log.warning("MA NATIVE ma_list_players failed: %s", exc)
        return _json_tool_result({"success": False, "error": str(exc)})


@function_tool
async def ma_search(
    query: str,
    media_types: list[
        Literal["artist", "album", "track", "playlist", "radio", "podcast", "audiobook"]
    ] | None = None,
    limit: int = 10,
) -> str:
    """Search Music Assistant directly over its persistent WebSocket client."""
    query = str(query or "").strip()
    if not query:
        return _json_tool_result({"success": False, "error": "query is required"})
    limit = max(1, min(50, int(limit or MUSIC_ASSISTANT_SEARCH_DEFAULT_LIMIT)))
    try:
        selected_types = _ma_media_types(media_types)
        manager = _ma_native_manager()
        log.info(
            "MA NATIVE SEARCH request query=%r requested_media_types=%r enum_media_types=%s limit=%s",
            query,
            media_types,
            [_ma_media_type_label(item) for item in selected_types],
            limit,
        )

        async def operation(client: Any):
            return await _ma_search_compatible(
                client,
                query=query,
                media_types=selected_types,
                limit=limit,
            )

        results = await manager.run_serialized("ma_search", operation)
        return _json_tool_result(
            {
                "success": True,
                "query": query,
                "results": _ma_search_payload(results, limit),
            }
        )
    except Exception as exc:
        log.warning("MA NATIVE ma_search failed query=%r: %s", query, exc)
        return _json_tool_result({"success": False, "error": str(exc)})


@function_tool
async def ma_browse(
    path: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> str:
    """Browse Music Assistant directly over its persistent WebSocket client."""
    path = str(path).strip() if path is not None else None
    limit = max(1, min(100, int(limit)))
    offset = max(0, int(offset))
    try:
        manager = _ma_native_manager()

        async def operation(client: Any):
            return await client.music.browse(path=path)

        items = await manager.run_serialized("ma_browse", operation)
        selected = list(items)[offset : offset + limit]
        return _json_tool_result(
            {
                "success": True,
                "path": path,
                "offset": offset,
                "limit": limit,
                "items": [_ma_media_summary(item) for item in selected],
            }
        )
    except Exception as exc:
        log.warning("MA NATIVE ma_browse failed path=%r: %s", path, exc)
        return _json_tool_result({"success": False, "error": str(exc)})


@function_tool
async def ma_play_query(
    query: str,
    area: str | None = None,
    player_id: str | None = None,
    radio_mode: bool = False,
) -> str:
    """Search and optimistically start music on a room/player in one native call.

    Prefer this for normal voice requests like "play Taylor Swift in the office".
    The proxy resolves player/queue/media deterministically. Artist-radio requests
    use v6.9 fast-start: one MA top-track seed is played first, then MA Don't Stop
    Music Assistant owns radio continuation. Long preparation never holds the voice turn.
    """
    query = str(query or "").strip()
    if not query:
        return _json_tool_result({"success": False, "error": "query is required"})
    try:
        selected_types = _ma_default_play_query_media_types()
        manager = _ma_native_manager()
        # "X radio" is an instruction to enable radio mode, not part of the
        # artist name. Removing the trailing word improves deterministic search.
        search_query = query
        if radio_mode:
            stripped = re.sub(r"\s+radio\s*$", "", query, flags=re.IGNORECASE).strip()
            if stripped:
                search_query = stripped
        origin_context = _voice_origin_snapshot()
        log.info(
            "MA NATIVE PLAY_QUERY search_types=%s query=%r search_query=%r area=%r player_id=%r radio_mode=%s",
            [_ma_media_type_label(item) for item in selected_types],
            query,
            search_query,
            area,
            player_id,
            radio_mode,
        )

        async def operation(client: Any):
            player, match_reason = _ma_resolve_player(
                client, area=area, player_id=player_id
            )
            resolved_player_id = str(getattr(player, "player_id", "") or "")
            queue_id = await _ma_resolve_queue_id(client, resolved_player_id)
            results = await _ma_search_compatible(
                client,
                query=search_query,
                media_types=selected_types,
                limit=MUSIC_ASSISTANT_SEARCH_DEFAULT_LIMIT,
            )
            selected, score = _ma_select_search_item(results, search_query)
            if selected is None or not _ma_uri(selected):
                raise ValueError(f"No Music Assistant result found for {search_query!r}")
            media_uri = _ma_uri(selected)
            selected_summary = _ma_media_summary(selected)
            log.info(
                "MA NATIVE PLAY_QUERY query=%r area=%r requested_player_id=%r "
                "resolved_player_id=%r queue_id=%r media=%r media_type=%r score=%.2f radio_mode=%s",
                query,
                area,
                player_id,
                resolved_player_id,
                queue_id,
                media_uri,
                selected_summary.get("media_type"),
                score,
                radio_mode,
            )
            selected_media_type = str(selected_summary.get("media_type") or "").casefold()

            # v6.9: artist radio always uses the same fast-start contract on all
            # MA schema versions. Start one MA-ranked seed track immediately,
            # then enable MA Don't Stop The Music once that seed is PLAYING.
            if radio_mode and selected_media_type == "artist":
                fast_radio = await _ma_dispatch_fast_artist_radio(
                    manager,
                    client,
                    queue_id=queue_id,
                    artist=selected,
                    origin_context=origin_context,
                    label=f"ma_play_query:{queue_id}:{media_uri}:fast_artist_radio",
                )
                verification = _ma_queue_summary(client, queue_id)
                message = (
                    f"Playing {search_query} radio."
                    if fast_radio.first_audio_observed
                    else f"Starting {search_query} radio."
                )
                return {
                    "success": True,
                    "message": message,
                    "command_acknowledged": fast_radio.command_acknowledged,
                    "command_dispatched": fast_radio.command_dispatched,
                    "still_processing": fast_radio.still_processing,
                    "duplicate_suppressed": fast_radio.duplicate_suppressed,
                    "first_audio_observed": fast_radio.first_audio_observed,
                    "first_audio_seconds": fast_radio.first_audio_seconds,
                    "query": query,
                    "search_query": search_query,
                    "area": area,
                    "player_id": resolved_player_id,
                    "player_name": str(getattr(player, "name", "") or resolved_player_id),
                    "player_match": match_reason,
                    "queue_id": queue_id,
                    "selected_media": selected_summary,
                    "radio_seed": fast_radio.seed,
                    "radio_seed_rank": fast_radio.seed_rank,
                    "radio_seed_pool": fast_radio.seed_pool,
                    "radio_seed_strategy": MUSIC_ASSISTANT_RADIO_SEED_STRATEGY,
                    "option": "replace",
                    "radio_mode": True,
                    "radio_fast_start": True,
                    "radio_continuation": "music_assistant_dont_stop_the_music",
                    "queue_after": verification,
                }

            assert QueueOption is not None
            dispatch = await _ma_dispatch_play_media(
                manager,
                client,
                queue_id=queue_id,
                media=media_uri,
                option=QueueOption.PLAY,
                radio_mode=bool(radio_mode),
                label=f"ma_play_query:{queue_id}:{media_uri}:radio={bool(radio_mode)}",
                origin_context=origin_context,
            )
            verification = (
                await _ma_post_action_verification(client, queue_id)
                if dispatch.command_acknowledged
                else _ma_queue_summary(client, queue_id)
            )
            if dispatch.command_acknowledged:
                message = "Playing."
            elif radio_mode:
                message = f"Starting {search_query} radio."
            else:
                message = f"Starting {search_query}."
            return {
                "success": True,
                "message": message,
                "command_acknowledged": dispatch.command_acknowledged,
                "command_dispatched": dispatch.command_dispatched,
                "still_processing": dispatch.still_processing,
                "duplicate_suppressed": dispatch.duplicate_suppressed,
                "query": query,
                "search_query": search_query,
                "area": area,
                "player_id": resolved_player_id,
                "player_name": str(getattr(player, "name", "") or resolved_player_id),
                "player_match": match_reason,
                "queue_id": queue_id,
                "selected_media": selected_summary,
                "option": "play",
                "radio_mode": bool(radio_mode),
                "radio_fast_start": False,
                "queue_after": verification,
            }

        payload = await manager.run_serialized("ma_play_query", operation)
        return _json_tool_result(payload)
    except Exception as exc:
        log.exception("MA NATIVE ma_play_query failed query=%r area=%r", query, area)
        return _json_tool_result({"success": False, "error": str(exc)})


@function_tool
async def ma_play_media(
    queue_id: str,
    media: str | list[str],
    option: Literal["play", "replace", "next", "add"] = "play",
    radio_mode: bool = False,
) -> str:
    """Play a known Music Assistant media URI on a player/queue."""
    queue_id = str(queue_id or "").strip()
    if not queue_id:
        return _json_tool_result({"success": False, "error": "queue_id is required"})
    try:
        manager = _ma_native_manager()

        async def operation(client: Any):
            resolved_queue = await _ma_resolve_queue_id(client, queue_id)
            assert QueueOption is not None
            option_value = QueueOption(str(option))
            log.info(
                "MA NATIVE PLAY_MEDIA requested_queue=%r resolved_queue=%r media=%r option=%s radio_mode=%s",
                queue_id,
                resolved_queue,
                media,
                option,
                radio_mode,
            )
            dispatch = await _ma_dispatch_play_media(
                manager,
                client,
                queue_id=resolved_queue,
                media=media,
                option=option_value,
                radio_mode=bool(radio_mode),
                label=f"ma_play_media:{resolved_queue}:{option}",
                origin_context=_voice_origin_snapshot(),
            )
            verification = (
                await _ma_post_action_verification(client, resolved_queue)
                if dispatch.command_acknowledged
                else _ma_queue_summary(client, resolved_queue)
            )
            return {
                "success": True,
                "message": (
                    "Added as next."
                    if option == "next"
                    else "Added to queue."
                    if option == "add"
                    else "Playing."
                    if dispatch.command_acknowledged
                    else "Starting playback."
                ),
                "command_acknowledged": dispatch.command_acknowledged,
                "command_dispatched": dispatch.command_dispatched,
                "still_processing": dispatch.still_processing,
                "duplicate_suppressed": dispatch.duplicate_suppressed,
                "queue_id": resolved_queue,
                "media": media,
                "option": option,
                "radio_mode": bool(radio_mode),
                "queue_after": verification,
            }

        return _json_tool_result(
            await manager.run_serialized("ma_play_media", operation)
        )
    except Exception as exc:
        log.exception("MA NATIVE ma_play_media failed queue_id=%r", queue_id)
        return _json_tool_result({"success": False, "error": str(exc)})


@function_tool
async def ma_playback(
    queue_id: str,
    command: Literal["play", "pause", "stop", "toggle", "next", "previous"],
    seek_seconds: int | None = None,
) -> str:
    """Control Music Assistant queue playback directly over WebSocket."""
    queue_id = str(queue_id or "").strip()
    try:
        manager = _ma_native_manager()

        async def operation(client: Any):
            resolved_queue = await _ma_resolve_queue_id(client, queue_id)
            queues = client.player_queues
            log.info(
                "MA NATIVE PLAYBACK requested_queue=%r resolved_queue=%r command=%s seek=%r",
                queue_id,
                resolved_queue,
                command,
                seek_seconds,
            )
            if command == "play":
                await queues.play(resolved_queue)
                if seek_seconds is not None:
                    await queues.seek(resolved_queue, max(0, int(seek_seconds)))
                message = "Playing."
            elif command == "pause":
                await queues.pause(resolved_queue)
                message = "Paused."
            elif command == "stop":
                await queues.stop(resolved_queue)
                message = "Stopped."
            elif command == "toggle":
                await queues.play_pause(resolved_queue)
                message = "Playback toggled."
            elif command == "next":
                await queues.next(resolved_queue)
                message = "Skipped to next."
            elif command == "previous":
                await queues.previous(resolved_queue)
                message = "Previous."
            else:
                raise ValueError(f"Unsupported playback command: {command}")
            return {
                "success": True,
                "message": message,
                "command_acknowledged": True,
                "queue_id": resolved_queue,
                "command": command,
            }

        return _json_tool_result(
            await manager.run_serialized("ma_playback", operation)
        )
    except Exception as exc:
        log.exception("MA NATIVE ma_playback failed queue_id=%r command=%r", queue_id, command)
        return _json_tool_result({"success": False, "error": str(exc)})


@function_tool
async def ma_volume(
    player_id: str,
    level: int | None = None,
    adjust: Literal["up", "down"] | None = None,
    mute: bool | None = None,
    group: bool = False,
) -> str:
    """Set, adjust, mute or unmute a Music Assistant player."""
    supplied = sum(value is not None for value in (level, adjust, mute))
    if supplied != 1:
        return _json_tool_result(
            {"success": False, "error": "Provide exactly one of level, adjust or mute"}
        )
    if mute is not None and group:
        return _json_tool_result(
            {"success": False, "error": "group=true is not supported for mute"}
        )
    try:
        manager = _ma_native_manager()

        async def operation(client: Any):
            players = client.players
            if client.players.get(player_id) is None:
                raise ValueError(f"Music Assistant player not found: {player_id}")
            if level is not None:
                value = max(0, min(100, int(level)))
                if group:
                    await players.group_volume(player_id, value)
                else:
                    await players.volume_set(player_id, value)
                message = f"Volume set to {value}%."
            elif adjust == "up":
                if group:
                    await players.group_volume_up(player_id)
                else:
                    await players.volume_up(player_id)
                message = "Volume increased."
            elif adjust == "down":
                if group:
                    await players.group_volume_down(player_id)
                else:
                    await players.volume_down(player_id)
                message = "Volume decreased."
            else:
                await players.volume_mute(player_id, bool(mute))
                message = "Muted." if mute else "Unmuted."
            log.info(
                "MA NATIVE VOLUME player_id=%r level=%r adjust=%r mute=%r group=%s",
                player_id,
                level,
                adjust,
                mute,
                group,
            )
            return {
                "success": True,
                "message": message,
                "command_acknowledged": True,
                "player_id": player_id,
            }

        return _json_tool_result(await manager.run_serialized("ma_volume", operation))
    except Exception as exc:
        log.exception("MA NATIVE ma_volume failed player_id=%r", player_id)
        return _json_tool_result({"success": False, "error": str(exc)})


@function_tool
async def ma_group(
    action: Literal["join", "leave"],
    player_ids: list[str],
    target_player_id: str | None = None,
) -> str:
    """Join Music Assistant players to a leader or remove them from groups."""
    ids = [str(value).strip() for value in player_ids if str(value).strip()]
    if not ids:
        return _json_tool_result({"success": False, "error": "player_ids is required"})
    if action == "join" and not target_player_id:
        return _json_tool_result(
            {"success": False, "error": "target_player_id is required for join"}
        )
    try:
        manager = _ma_native_manager()

        async def operation(client: Any):
            if action == "join":
                target = str(target_player_id)
                children = [item for item in ids if item != target]
                if not children:
                    raise ValueError("No child players supplied to join")
                log.info("MA NATIVE GROUP join target=%r children=%r", target, children)
                await client.players.group_many(target, children)
                message = "Speakers grouped."
            else:
                log.info("MA NATIVE GROUP leave players=%r", ids)
                await client.players.ungroup_many(ids)
                message = "Speakers removed from group."
            return {
                "success": True,
                "message": message,
                "command_acknowledged": True,
                "action": action,
                "player_ids": ids,
                "target_player_id": target_player_id,
            }

        return _json_tool_result(await manager.run_serialized("ma_group", operation))
    except Exception as exc:
        log.exception("MA NATIVE ma_group failed action=%r", action)
        return _json_tool_result({"success": False, "error": str(exc)})


@function_tool
async def ma_queue(
    queue_id: str,
    get_items: bool = True,
    shuffle: bool | None = None,
    repeat: Literal["off", "one", "all"] | None = None,
    clear: bool = False,
) -> str:
    """Read queue state and optionally change shuffle/repeat/clear settings."""
    queue_id = str(queue_id or "").strip()
    try:
        manager = _ma_native_manager()

        async def operation(client: Any):
            resolved_queue = await _ma_resolve_queue_id(client, queue_id)
            changes: list[str] = []
            if shuffle is not None:
                await client.player_queues.shuffle(resolved_queue, bool(shuffle))
                changes.append(f"shuffle {'enabled' if shuffle else 'disabled'}")
            if repeat is not None:
                assert RepeatMode is not None
                await client.player_queues.repeat(resolved_queue, RepeatMode(str(repeat)))
                changes.append(f"repeat set to {repeat}")
            if clear:
                await client.player_queues.clear(resolved_queue)
                changes.append("queue cleared")
            payload: dict[str, Any] = {
                "success": True,
                "changed": bool(changes),
                "changes_applied": changes,
                "queue": _ma_queue_summary(client, resolved_queue),
            }
            if get_items:
                items = await client.player_queues.get_queue_items(
                    resolved_queue, limit=50, offset=0
                )
                payload["items"] = [_ma_queue_item_summary(item) for item in items]
            return payload

        return _json_tool_result(await manager.run_serialized("ma_queue", operation))
    except Exception as exc:
        log.exception("MA NATIVE ma_queue failed queue_id=%r", queue_id)
        return _json_tool_result({"success": False, "error": str(exc)})


@function_tool
async def ma_queue_item(
    queue_id: str,
    item_id: str,
    action: Literal["move_up", "move_down", "move_next", "remove"],
) -> str:
    """Move or remove an individual Music Assistant queue item."""
    try:
        manager = _ma_native_manager()

        async def operation(client: Any):
            resolved_queue = await _ma_resolve_queue_id(client, queue_id)
            queues = client.player_queues
            log.info(
                "MA NATIVE QUEUE_ITEM queue_id=%r item_id=%r action=%s",
                resolved_queue,
                item_id,
                action,
            )
            if action == "move_up":
                await queues.move_up(resolved_queue, item_id)
            elif action == "move_down":
                await queues.move_down(resolved_queue, item_id)
            elif action == "move_next":
                await queues.move_next(resolved_queue, item_id)
            elif action == "remove":
                await queues.delete_item(resolved_queue, item_id)
            return {
                "success": True,
                "message": "Queue updated.",
                "command_acknowledged": True,
                "queue_id": resolved_queue,
                "item_id": item_id,
                "action": action,
            }

        return _json_tool_result(
            await manager.run_serialized("ma_queue_item", operation)
        )
    except Exception as exc:
        log.exception("MA NATIVE ma_queue_item failed queue_id=%r", queue_id)
        return _json_tool_result({"success": False, "error": str(exc)})


@function_tool
async def ma_transfer_queue(source_queue_id: str, target_queue_id: str) -> str:
    """Transfer Music Assistant playback from one queue/player to another."""
    try:
        manager = _ma_native_manager()

        async def operation(client: Any):
            source = await _ma_resolve_queue_id(client, source_queue_id)
            target = await _ma_resolve_queue_id(client, target_queue_id)
            log.info("MA NATIVE TRANSFER source=%r target=%r", source, target)
            await client.player_queues.transfer(source, target)
            return {
                "success": True,
                "message": "Playback moved.",
                "command_acknowledged": True,
                "source_queue_id": source,
                "target_queue_id": target,
            }

        return _json_tool_result(
            await manager.run_serialized("ma_transfer_queue", operation)
        )
    except Exception as exc:
        log.exception(
            "MA NATIVE ma_transfer_queue failed source=%r target=%r",
            source_queue_id,
            target_queue_id,
        )
        return _json_tool_result({"success": False, "error": str(exc)})


MUSIC_ASSISTANT_ALWAYS_WRITE_TOOLS = {
    "ma_play_query",
    "ma_play_media",
    "ma_volume",
    "ma_playback",
    "ma_group",
    "ma_queue_item",
    "ma_transfer_queue",
}


def _tool_result_name(result: FunctionToolResult) -> str:
    return str(getattr(getattr(result, "tool", None), "name", "") or "").strip()


def _music_assistant_output_failed(output: Any) -> bool:
    """Recognize native Music Assistant errors in addition to generic failures."""
    if _tool_output_failed(output):
        return True

    text = _serialise_tool_output(output).strip().casefold()
    if not text:
        return True

    failure_markers = (
        "error:",
        "error calling tool",
        "failed to connect",
        "connector is closed",
        "queue not found:",
        "unknown command:",
        "unknown action:",
        "no changes made",
        "invalid player",
        "player not found",
    )
    return any(marker in text for marker in failure_markers)


def _is_music_assistant_write_result(tool_name: str, output: Any) -> bool:
    """Return True only for Music Assistant calls which changed playback/state."""
    if tool_name in MUSIC_ASSISTANT_ALWAYS_WRITE_TOOLS:
        return True

    # ma_queue is both a read and write tool. Native v6.8 returns an explicit
    # changed boolean; retain text matching for compatibility with older output.
    if tool_name == "ma_queue":
        payload = _tool_output_mapping(output)
        if isinstance(payload, dict) and payload.get("changed") is True:
            return True
        text = _serialise_tool_output(output).casefold()
        return any(
            marker in text
            for marker in (
                "shuffle enabled",
                "shuffle disabled",
                "repeat set to",
                "queue cleared",
            )
        )

    return False


def _music_assistant_terminal_speech(
    tool_name: str,
    output: Any,
    *,
    multiple_actions: bool = False,
) -> str:
    """Turn a confirmed MA write result into concise TTS without another LLM call."""
    if multiple_actions:
        return FAST_ACTION_RESPONSE

    text = _serialise_tool_output(output).strip()
    lower = text.casefold()

    if tool_name in {"ma_play_query", "ma_play_media"}:
        payload = _tool_output_mapping(output)
        if isinstance(payload, dict):
            message = payload.get("message")
            if isinstance(message, str) and message.strip():
                return message.strip()
        if "added as next" in lower:
            return "Added next."
        if "added to queue" in lower:
            return "Added to the queue."
        if "starting playback" in lower:
            return "Starting playback."
        return "Playing."

    if tool_name == "ma_volume":
        if "unmuted" in lower:
            return "Unmuted."
        if "muted" in lower:
            return "Muted."
        if "increased" in lower:
            return "Volume up."
        if "decreased" in lower:
            return "Volume down."
        match = re.search(r"volume set to\s+(\d+)%", lower)
        if match:
            return f"Volume set to {match.group(1)} percent."
        return "Volume set."

    if tool_name == "ma_playback":
        if "paused" in lower:
            return "Paused."
        if "stopped" in lower:
            return "Stopped."
        if "skipped to next" in lower:
            return "Next."
        if "previous" in lower:
            return "Previous."
        if "playing" in lower:
            return "Playing."
        return FAST_ACTION_RESPONSE

    if tool_name == "ma_transfer_queue":
        return "Playback moved."

    if tool_name == "ma_group":
        if "removed from" in lower:
            return "Speakers ungrouped."
        return "Speakers grouped."

    if tool_name in {"ma_queue", "ma_queue_item"}:
        return FAST_ACTION_RESPONSE

    return FAST_ACTION_RESPONSE


def _music_replay_cache_key(
    conversation_key: str,
    text: str,
    origin_context: dict[str, Any] | None,
) -> str:
    origin_context = origin_context or {}
    area = str(
        origin_context.get("area_id")
        or origin_context.get("area_name")
        or ""
    ).strip().casefold()
    return "|".join(
        (
            conversation_key.strip(),
            area,
            normalize_command(text),
        )
    )


def _get_recent_music_action_response(
    conversation_key: str,
    text: str,
    origin_context: dict[str, Any] | None,
) -> str | None:
    if MUSIC_ASSISTANT_REPLAY_GUARD_SECONDS <= 0:
        return None

    now = time.monotonic()
    cutoff = now - MUSIC_ASSISTANT_REPLAY_GUARD_SECONDS

    # Opportunistic cleanup; this cache is intentionally tiny and short-lived.
    expired = [
        key
        for key, (created_at, _) in recent_music_action_responses.items()
        if created_at < cutoff
    ]
    for key in expired:
        recent_music_action_responses.pop(key, None)

    key = _music_replay_cache_key(conversation_key, text, origin_context)
    cached = recent_music_action_responses.get(key)
    if cached is None:
        return None

    created_at, response = cached
    if created_at < cutoff:
        recent_music_action_responses.pop(key, None)
        return None

    return response


def _remember_music_action_response(
    conversation_key: str,
    text: str,
    origin_context: dict[str, Any] | None,
    response: str,
) -> None:
    if MUSIC_ASSISTANT_REPLAY_GUARD_SECONDS <= 0:
        return
    key = _music_replay_cache_key(conversation_key, text, origin_context)
    recent_music_action_responses[key] = (time.monotonic(), response)


def fast_tool_result_handler(
    context: RunContextWrapper,
    tool_results: list[FunctionToolResult],
) -> ToolsToFinalOutputResult:
    """End successful write actions immediately; return read/error results to the LLM."""
    del context

    # ------------------------------------------------------------------
    # Music Assistant
    # ------------------------------------------------------------------
    music_results = [
        result
        for result in tool_results
        if _tool_result_name(result).startswith("ma_")
    ]

    music_write_results = [
        result
        for result in music_results
        if _is_music_assistant_write_result(
            _tool_result_name(result),
            result.output,
        )
    ]

    if MUSIC_ASSISTANT_TERMINAL_ACTIONS and music_write_results:
        # Any failed write stays non-terminal so Qwen can inspect/correct it.
        failed_writes = [
            result
            for result in music_write_results
            if _music_assistant_output_failed(result.output)
        ]
        if failed_writes:
            log.info(
                "MUSIC ASSISTANT action batch includes failure tools=%s; "
                "returning results to model",
                [_tool_result_name(result) for result in failed_writes],
            )
            return ToolsToFinalOutputResult(
                is_final_output=False,
                final_output=None,
            )

        # All MA write calls in this tool batch have completed successfully.
        # This is intentionally terminal: no post-action LLM turn means there
        # is no empty-final-output window which can cause an HTTP retry/replay.
        final_result = music_write_results[-1]
        final_tool_name = _tool_result_name(final_result)
        speech = _music_assistant_terminal_speech(
            final_tool_name,
            final_result.output,
            multiple_actions=len(music_write_results) > 1,
        )

        voice_tool_run_state.last_successful_music_action = {
            "tool": final_tool_name,
            "output": _serialise_tool_output(final_result.output),
            "spoken": speech,
            "write_count": len(music_write_results),
        }

        log.info(
            "FAST MUSIC ACTION BATCH COMPLETE tools=%s response=%r",
            [_tool_result_name(result) for result in music_write_results],
            speech,
        )
        return ToolsToFinalOutputResult(
            is_final_output=True,
            final_output=speech,
        )

    # ------------------------------------------------------------------
    # Direct Home Assistant WebSocket
    # ------------------------------------------------------------------
    service_results = [
        result
        for result in tool_results
        if _tool_result_name(result) == "ha_call_service"
    ]
    if not service_results:
        # Read-only MA calls such as ma_list_players / ma_search / ma_browse /
        # normal ma_queue reads are deliberately returned to Qwen.
        return ToolsToFinalOutputResult(
            is_final_output=False,
            final_output=None,
        )

    # If any service failed, let the model inspect/correct it. Likewise if any
    # call returned data; response data must be interpreted into spoken output.
    for result in service_results:
        if _tool_output_failed(result.output):
            log.info("HA service batch includes failure; returning results to model")
            return ToolsToFinalOutputResult(
                is_final_output=False,
                final_output=None,
            )
        payload = _tool_output_mapping(result.output)
        if isinstance(payload, dict) and "service_response" in payload:
            voice_tool_run_state.last_successful_data = {
                "domain": payload.get("domain"),
                "service": payload.get("service"),
                "entity_id": payload.get("entity_id"),
                "area_id": payload.get("area_id"),
                "service_response": payload.get("service_response"),
            }
            log.info("HA data service complete; returning service_response to model")
            return ToolsToFinalOutputResult(
                is_final_output=False,
                final_output=None,
            )

    # Only short-circuit when every HA service result is a known state-changing
    # action. This also makes multi-device commands safe: all calls execute first.
    for result in service_results:
        payload = _tool_output_mapping(result.output) or {}
        service_name = str(payload.get("service") or "").strip()
        if (
            payload.get("verified_state") is None
            and service_name not in FAST_STATE_CHANGING_SERVICES
        ):
            return ToolsToFinalOutputResult(
                is_final_output=False,
                final_output=None,
            )

    log.info(
        "FAST HA ACTION BATCH COMPLETE calls=%d response=%r",
        len(service_results),
        FAST_ACTION_RESPONSE,
    )
    return ToolsToFinalOutputResult(
        is_final_output=True,
        final_output=FAST_ACTION_RESPONSE,
    )



def sanitise_spoken_response(text: str) -> str:
    return text_policy.sanitise_spoken_response(text, SPOKEN_MAX_CHARS)

    text = str(text or "").strip()
    if not text:
        return text

    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    text = re.sub(r"https?://\S+", "", text)
    text = text.replace("**", "").replace("__", "").replace("`", "")
    text = re.sub(r"^[#>*\-]+\s*", "", text)
    text = re.sub(r"\s*\([^()]{0,160}\)", "", text)
    text = re.sub(r"\s*\[[^\[\]]{0,160}\]", "", text)
    text = text.replace("°C", " degrees Celsius")
    text = text.replace("°F", " degrees Fahrenheit")
    text = re.sub(r"\s*\n+\s*", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    limitation_markers = (
        "i don't have access",
        "i do not have access",
        "through the available tools",
        "available home assistant tools",
        "unable to access",
        "cannot access",
    )
    lowered = text.casefold()
    if any(marker in lowered for marker in limitation_markers):
        return "I can't check that."

    text = re.split(
        r"\b(?:if you'd like|if you would like|if you want|you could|you can also|would you like)\b",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip(" ,;:-")

    if SPOKEN_MAX_CHARS > 0 and len(text) > SPOKEN_MAX_CHARS:
        shortened = text[:SPOKEN_MAX_CHARS].rsplit(" ", 1)[0].rstrip(" ,;:-")
        text = shortened + "."
    return text


# ---------------------------------------------------------------------------
# Main voice fallback instructions
# ---------------------------------------------------------------------------


FALLBACK_INSTRUCTIONS = """
You are a fast Home Assistant voice assistant used only when a deterministic
intent system could not handle the user's request.

Your response will be spoken aloud.

GENERAL BEHAVIOR

Use Home Assistant tools whenever the request depends on Home Assistant state,
devices, entities, services, automations, integrations, or data.

Not every request needs a Home Assistant tool. If the answer is available from
the supplied runtime date/time context or recent conversation context, answer
directly without searching Home Assistant.

Do not invent entity IDs, service names, capabilities, values, or results.

Prefer the smallest number of tool calls necessary.

Do not repeat a successful tool call.

Do not verify a successful state-changing action again unless its tool result
explicitly indicates uncertainty or failure.

For normal successful control actions, the runtime may finish immediately with
"Done." without another model turn. Do not fight this behavior by making extra
verification calls.

Do not explain which tools you used.


FAST TOOLS AND ADVANCED TOOLS

The normal Home Assistant tools are intentionally fast:

- ha_search searches a local cache of current Home Assistant entities.
- ha_get_state reads current state from that local cache.
- ha_list_services reads a cached Home Assistant service catalogue.
- ha_call_service executes an action over a persistent Home Assistant WebSocket.

Use these four tools for ordinary Home Assistant household control,
current-state questions, forecasts, calendar/service queries, and other normal
Home Assistant operations.

When Music Assistant tools are connected, they are authoritative for music
search/playback, music pause/resume/skip/queue operations, and Music Assistant
speaker volume/mute. Do not route those requests through Home Assistant
media_player services.

Do not use ha_advanced merely because a request needs a service you have not
seen before. First use ha_list_services with a narrow domain and full detail.

Use ha_advanced only when the request genuinely needs capabilities beyond the
fast tools, for example configuration or editing of automations, scripts,
helpers, dashboards, integrations, traces, historical data, system diagnostics,
or other administrative Home Assistant operations.

When using ha_advanced, give it a concise self-contained request containing the
important target and requested outcome. Do not use it for ordinary light,
switch, climate, current-state, forecast, simple service calls, or Music
Assistant playback/volume requests.


CONVERSATION CONTEXT

Recent user and assistant turns may be supplied before the latest request.
Use them only when the latest request is a natural continuation, for example:
"what about tomorrow", "and the bedroom", "make it brighter", or "turn it off".

Resolve pronouns and omitted targets from the most recent relevant context.
The latest user message is always the current request.

Do not repeat a previous action merely because it appears in history.
Do not carry an old target into a new request when the latest message is clear
on its own. If the latest request is not a follow-up, ignore unrelated history.


DATE AND TIME

A trusted current local date and time is supplied with every request.
Use it for now, today, tomorrow, tonight, this morning, this afternoon, this
evening, and direct date/time questions.

Do not search Home Assistant merely to discover the current clock or date.


REQUESTING VOICE AREA CONTEXT

A request may include the Home Assistant voice satellite/device and area from
which the user is speaking. Treat this area as an implicit preference only when
the user did not name another location.

Examples: if the request originates in Office, "turn the light on" should
prefer an Office light, "what is the temperature" should prefer an Office
temperature entity. When Music Assistant is connected, "play Taylor Swift"
or "turn the volume down" should prefer the Music Assistant player matching
Office rather than a Home Assistant media_player.

This is a soft preference, not an absolute restriction. An explicitly named
room, area, floor, entity, device, or global request always overrides the voice
origin. Never claim a device is unavailable solely because it is outside the
origin area.

If exactly one sensible entity of the requested domain exists in the origin
area, prefer it without asking a follow-up question.

When ha_search returns recommended_entity_id, prefer that entity for an
unqualified room-local request unless the user explicitly named another
entity or area.

For a generic light request, interpret "the light" or "the lights" as normal
room illumination. Do not choose a status LED, ring LED, indicator LED,
notification LED, display backlight, or the voice satellite's own LED unless
the user explicitly asks for that LED or indicator.


ENTITY DISCOVERY

Use ha_search when you need to resolve a natural-language device, entity, area,
or integration name.

Use the narrowest relevant domain_filter whenever possible, such as light,
switch, climate, cover, media_player, weather, sensor, binary_sensor, calendar,
or vacuum.

Use area_filter when the user clearly names a Home Assistant area.

If an exact entity ID is already known from a current tool result or recent
conversation, reuse it. Do not search for it again without a reason.

Do not treat a zero-result search in one guessed domain as proof the request is
impossible. If the user's wording could reasonably map to another normal domain,
try one better-targeted search before giving up. Avoid broad repeated searches.


CURRENT STATE VERSUS AVAILABLE CAPABILITIES

Current entity state does not necessarily contain every kind of information an
integration can provide.

Some information is available only through Home Assistant services/actions that
return data. Do not conclude information is unavailable merely because
ha_get_state does not contain it.

If requested information is absent from current state and no direct tool returns
it, use ha_list_services once with the narrowest relevant domain to discover a
suitable action.

Examples include forecasts, event queries, diagnostics, listings,
integration-specific lookups, calculations, and other data-returning actions.

Weather is one example: a weather entity may contain current conditions while a
forecast comes from a weather action. This is only an example of the general
rule.


SERVICE DISCOVERY

When discovering a service because you intend to call it, use
ha_list_services with detail_level="full".

Read the returned compact CALL schema before calling the service.
The exact keys under "parameters" are the ONLY names to put in data.
Do not derive parameter names from a human-readable field label or description.
For example, if the schema key is "type", use "type"; do not invent
"forecast_type".

Include every required parameter and use allowed values exactly as described.

Use the narrowest domain filter. Do not browse unrelated service domains.
Do not repeatedly call ha_list_services for the same domain during one request
unless the first result failed or was incomplete.


SERVICE CALL RULES

Home Assistant services may require service-specific parameters. Finding the
correct service name is not enough.

When calling ha_call_service:

- Put a target entity in entity_id.
- For an area-wide action, prefer one exact area_id returned by ha_search rather than separate calls for every entity.
- Put service-specific parameters in data.
- Include every required field from the discovered schema.
- Use allowed values exactly as described.
- Do not guess parameter names when the full schema is available.
- Set return_response=True for actions used to retrieve information.
- For ordinary state-changing actions, return_response is usually unnecessary.

If a service call succeeds and changes the requested state, do not call it
again.

When repairing a failed service call, preserve every part that was already
valid: domain, service, entity_id or area_id, return_response, and valid data
fields. Change ONLY the argument identified as invalid. Never drop a previously
resolved target merely because one service-data field was wrong.

If a service returns service_response data, use that returned data directly.
Do not discard it and then say the information is unavailable.


SERVICE ERRORS

If ha_call_service returns a validation error, invalid parameter error, missing
field error, or similar argument-related failure:

1. Do not repeat the same call unchanged.
2. Preserve its already-valid target and arguments.
3. If the returned error already includes service_schema, use that schema
directly; do not call ha_list_services again.
4. Otherwise, if you have not already done so, call ha_list_services for that
exact domain with detail_level="full".
5. Re-read the exact parameter keys, required fields, and allowed values.
6. Correct only the invalid argument.
7. Retry the corrected service call once.

If the corrected call still fails, stop. Do not enter a repeated discovery and
retry loop.


STATE-CHANGING ACTIONS

For normal Home Assistant runtime control, prefer ha_call_service.

Exception: when Music Assistant tools are connected, use them instead for music
playback/queue actions and Music Assistant speaker playback volume/mute.

Never claim an action succeeded unless the relevant tool result confirms
success.

Once a requested state-changing action succeeds, immediately finish.
Do not perform another state lookup merely to reassure yourself.
Do not issue the same state-changing service twice.


INFORMATION REQUESTS

For questions about current state, use ha_get_state after resolving the entity.

Do not call ha_get_state as a ritual before every request. If the user clearly
asks for information that is normally produced by an action rather than current
state, such as a forecast, event list, or integration lookup, resolve the target
and move directly to narrow service discovery/calling.

If the answer requires information beyond current state, discover and call an
appropriate information-returning service when one exists.

Do not mention implementation details such as "the sensor only exposes", "the
integration does not store", "the available tools", or "through Home Assistant"
unless that detail is essential to the answer.

Only say information cannot be checked after the relevant entity lookup and,
when appropriate, one narrow service-discovery attempt fail to provide a normal
route. If the user is explicitly asking for historical/configuration/admin data,
ha_advanced may be appropriate before giving up.


SPOKEN RESPONSE STYLE

The final response is for text-to-speech.

Keep it concise, natural, and conversational. Normally use one short sentence.
Aim for fewer than twelve words when possible.

Do not use Markdown, bullet points, headings, parentheses, square brackets,
URLs, JSON, entity IDs, tool names, service names, parameter names, or technical
Home Assistant terminology unless the user explicitly asks for technical detail.

Do not give both Celsius and Fahrenheit unless explicitly requested. Use the
units already provided by Home Assistant unless the user asks for conversion.

Do not add explanatory caveats when a direct answer is available.

Prefer:
"Tomorrow will be cloudy with a high of eighteen degrees."

Not:
"Based on your weather sensor, tomorrow is expected to be cloudy with a high of
eighteen degrees Celsius, although I don't have access to..."

For unavailable information prefer:
"I can't check that."

Do not say "Unfortunately", "Based on your sensor", "According to Home
Assistant", "It appears that", or "I don't have access to" unless genuinely
necessary.


FOLLOW-UP QUESTIONS

Do not ask follow-up questions unless genuinely necessary.
Only ask when proceeding would be unsafe, destructive, security-sensitive, or
seriously ambiguous.

For ordinary household control, state queries, forecasts, media requests, and
similar tasks, make the best safe interpretation from Home Assistant and recent
conversation context.

Do not ask whether the user wants more information. Do not suggest third-party
services, apps, websites, new integrations, configuration changes, or additional
setup unless the user explicitly asks for troubleshooting or advice.


FAILURE HANDLING

If a requested action cannot be completed, keep the spoken response brief.
Preferred forms are "I couldn't do that.", "I can't find that device.", or
"I can't check that right now."

Do not expose internal errors, HTTP status codes, WebSocket details, service
schemas, or implementation details in the spoken answer.
""".strip()



def make_fallback_agent(music_enabled: bool = False) -> Agent:
    tools = [ha_search, ha_get_state, ha_list_services, ha_call_service]
    if advanced_agent is not None:
        tools.append(ha_advanced)

    if music_enabled:
        tools.extend(
            [
                ma_play_query,
                ma_list_players,
                ma_search,
                ma_browse,
                ma_play_media,
                ma_playback,
                ma_volume,
                ma_group,
                ma_queue,
                ma_queue_item,
                ma_transfer_queue,
            ]
        )

    instructions = FALLBACK_INSTRUCTIONS
    if music_enabled:
        instructions = instructions + "\n\n" + _music_assistant_agent_instructions()

    return Agent(
        name="Home Assistant + Music Assistant fast voice fallback",
        model=_make_lemonade_model(),
        instructions=instructions,
        tools=tools,
        tool_use_behavior=fast_tool_result_handler,
    )


# ---------------------------------------------------------------------------
# Runtime time and conversation memory
# ---------------------------------------------------------------------------



def _runtime_context() -> str:
    try:
        tz = ZoneInfo(LOCAL_TIMEZONE)
    except Exception:
        tz = timezone.utc
    now = datetime.now(tz)
    zone_label = LOCAL_TIMEZONE if tz is not timezone.utc else "UTC"
    return (
        "Trusted runtime context: the current local date and time is "
        f"{now.strftime('%A %d %B %Y at %H:%M')} in {zone_label}. "
        "Use this for relative dates and direct clock/date questions."
    )



def _message_text(message: dict) -> str:
    return text_policy.message_text(message)

    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") not in {"text", "input_text", "output_text"}:
                continue
            value = item.get("text")
            if isinstance(value, str):
                parts.append(value)
        return " ".join(parts).strip()
    return ""



def extract_client_history(body: dict) -> list[dict[str, str]]:
    return text_policy.extract_client_history(body, CONVERSATION_HISTORY_TURNS)

    messages = body.get("messages", [])
    if not isinstance(messages, list):
        return []

    compact: list[dict[str, str]] = []
    last_user_index = None
    for index, message in enumerate(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            last_user_index = index

    for index, message in enumerate(messages):
        if not isinstance(message, dict) or index == last_user_index:
            continue
        role = message.get("role")
        if role not in {"user", "assistant"}:
            continue
        value = _message_text(message)
        if value:
            compact.append({"role": role, "content": value})

    return compact[-(CONVERSATION_HISTORY_TURNS * 2) :]



def _first_nonempty_string(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _nested_mapping(body: dict[str, Any], key: str) -> dict[str, Any]:
    value = body.get(key)
    return value if isinstance(value, dict) else {}


_HA_AREA_SYSTEM_PROMPT_RE = re.compile(
    r"""
    \bYou\s+are\s+in\s+area\s+
    (?P<area>.+?)
    (?:\s+\(floor\s+(?P<floor>[^)]+)\))?
    \s+and\s+all\s+generic\s+commands\b
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _extract_ha_system_origin_hint(body: dict[str, Any]) -> dict[str, str] | None:
    """Extract Home Assistant's trusted area/floor hint from its system prompt.

    HA's OpenAI-compatible conversation integration commonly sends a system line:
        You are in area Office (floor First) and all generic commands...

    We use only that small location hint. The system message itself is not added
    to extract_client_history(), so the large HA static context is not forwarded
    to Lemonade.
    """
    if not VOICE_ORIGIN_SYSTEM_PROMPT_FALLBACK:
        return None

    messages = body.get("messages")
    if not isinstance(messages, list):
        return None

    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "system":
            continue
        content = _message_text(message)
        if not content:
            continue

        match = _HA_AREA_SYSTEM_PROMPT_RE.search(content)
        if not match:
            continue

        area_name = (match.group("area") or "").strip()
        floor_name = (match.group("floor") or "").strip()
        if not area_name:
            continue

        result = {"area_name": area_name}
        if floor_name:
            result["floor_name"] = floor_name
        log.info(
            "VOICE ORIGIN system prompt hint parsed area=%r floor=%r",
            area_name,
            floor_name or None,
        )
        return result

    return None


async def extract_voice_origin_context(
    request: Request,
    body: dict[str, Any],
) -> dict[str, Any] | None:
    """Extract and resolve calling HA voice-satellite context.

    Prefer structured request metadata. If HA's OpenAI-compatible caller does
    not forward a device/area field, fall back to the trusted area/floor line in
    its system prompt.
    """
    if not VOICE_ORIGIN_CONTEXT_ENABLED:
        return None

    metadata = _nested_mapping(body, "metadata")
    context = _nested_mapping(body, "context")
    assist = _nested_mapping(body, "assist_context")
    ha_context = _nested_mapping(body, "home_assistant")

    device_id = _first_nonempty_string(
        request.headers.get("X-HA-Device-ID"),
        request.headers.get("X-Home-Assistant-Device-ID"),
        request.headers.get("X-Hass-Device-ID"),
        request.headers.get("X-Assist-Device-ID"),
        request.headers.get("X-Voice-Device-ID"),
        body.get("device_id"),
        body.get("assist_device_id"),
        body.get("satellite_device_id"),
        metadata.get("device_id"),
        metadata.get("assist_device_id"),
        context.get("device_id"),
        assist.get("device_id"),
        ha_context.get("device_id"),
    )
    device_name = _first_nonempty_string(
        request.headers.get("X-HA-Device-Name"),
        request.headers.get("X-Assist-Device-Name"),
        body.get("device_name"),
        body.get("satellite_name"),
        metadata.get("device_name"),
        context.get("device_name"),
        assist.get("device_name"),
        ha_context.get("device_name"),
    )
    area_id = _first_nonempty_string(
        request.headers.get("X-HA-Area-ID"),
        request.headers.get("X-Assist-Area-ID"),
        body.get("area_id"),
        metadata.get("area_id"),
        context.get("area_id"),
        assist.get("area_id"),
        ha_context.get("area_id"),
    )
    area_name = _first_nonempty_string(
        request.headers.get("X-HA-Area-Name"),
        request.headers.get("X-Assist-Area-Name"),
        body.get("area_name"),
        metadata.get("area_name"),
        context.get("area_name"),
        assist.get("area_name"),
        ha_context.get("area_name"),
    )
    floor_name = _first_nonempty_string(
        request.headers.get("X-HA-Floor-Name"),
        request.headers.get("X-Assist-Floor-Name"),
        body.get("floor_name"),
        metadata.get("floor_name"),
        context.get("floor_name"),
        assist.get("floor_name"),
        ha_context.get("floor_name"),
    )

    structured_present = any((device_id, device_name, area_id, area_name))
    system_hint = _extract_ha_system_origin_hint(body)

    # Structured values always win. Use the HA system prompt only to fill room
    # context that the HTTP request did not provide explicitly.
    used_system_hint = False
    if system_hint:
        if not area_id and not area_name:
            area_name = system_hint.get("area_name")
            used_system_hint = bool(area_name)
        if not floor_name:
            floor_name = system_hint.get("floor_name")

    if not any((device_id, device_name, area_id, area_name)):
        return None

    if structured_present and used_system_hint:
        source = "request+ha_system_prompt"
    elif structured_present:
        source = "request"
    elif used_system_hint:
        source = "ha_system_prompt"
    else:
        source = "request"

    if ha_ws is not None:
        try:
            await ha_ws.refresh_registries()
            resolved = ha_ws.resolve_device_origin(
                device_id=device_id,
                device_name=device_name,
                area_id=area_id,
                area_name=area_name,
            )
        except Exception as exc:
            log.warning("VOICE ORIGIN registry resolution failed: %s", exc)
            resolved = {
                "device_id": device_id,
                "device_name": device_name,
                "area_id": area_id,
                "area_name": area_name,
            }
    else:
        resolved = {
            "device_id": device_id,
            "device_name": device_name,
            "area_id": area_id,
            "area_name": area_name,
        }

    resolved["floor_name"] = floor_name
    resolved["source"] = source
    log.info(
        "VOICE ORIGIN resolved source=%s device=%r area=%r area_id=%r floor=%r",
        source,
        resolved.get("device_name"),
        resolved.get("area_name"),
        resolved.get("area_id"),
        floor_name,
    )
    return resolved


def get_conversation_key(request: Request, body: dict) -> str:
    metadata = body.get("metadata")
    metadata_id = (
        metadata.get("conversation_id") if isinstance(metadata, dict) else None
    )
    candidates = (
        request.headers.get("X-Conversation-ID"),
        body.get("conversation_id"),
        metadata_id,
        body.get("user"),
    )
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()[:160]
    return f"site:{SITE_ID}"



def _prune_conversation_memories_locked(now: float) -> None:
    expired = [
        key
        for key, memory in conversation_memories.items()
        if now - memory.updated_at > CONVERSATION_HISTORY_TTL_SECONDS
    ]
    for key in expired:
        conversation_memories.pop(key, None)

    if len(conversation_memories) <= CONVERSATION_HISTORY_MAX_SESSIONS:
        return
    oldest = sorted(
        conversation_memories.items(), key=lambda item: item[1].updated_at
    )
    excess = len(conversation_memories) - CONVERSATION_HISTORY_MAX_SESSIONS
    for key, _ in oldest[:excess]:
        conversation_memories.pop(key, None)


async def clear_conversation_history(conversation_key: str) -> None:
    async with conversation_history_lock:
        conversation_memories.pop(conversation_key, None)


async def seed_conversation_history(
    conversation_key: str,
    messages: list[dict[str, str]],
) -> None:
    if not CONVERSATION_HISTORY_ENABLED or not messages:
        return
    now = time.monotonic()
    async with conversation_history_lock:
        _prune_conversation_memories_locked(now)
        conversation_memories[conversation_key] = ConversationMemory(
            messages=messages[-(CONVERSATION_HISTORY_TURNS * 2) :],
            updated_at=now,
        )


async def get_conversation_history(
    conversation_key: str,
) -> list[dict[str, str]]:
    if not CONVERSATION_HISTORY_ENABLED:
        return []
    now = time.monotonic()
    async with conversation_history_lock:
        _prune_conversation_memories_locked(now)
        memory = conversation_memories.get(conversation_key)
        if memory is None:
            return []
        return [dict(item) for item in memory.messages]


async def remember_conversation_turn(
    conversation_key: str,
    user_text: str,
    assistant_text: str,
) -> None:
    if not CONVERSATION_HISTORY_ENABLED:
        return
    now = time.monotonic()
    async with conversation_history_lock:
        _prune_conversation_memories_locked(now)
        memory = conversation_memories.get(conversation_key)
        messages = list(memory.messages) if memory else []
        messages.extend(
            [
                {"role": "user", "content": user_text.strip()},
                {"role": "assistant", "content": assistant_text.strip()},
            ]
        )
        conversation_memories[conversation_key] = ConversationMemory(
            messages=messages[-(CONVERSATION_HISTORY_TURNS * 2) :],
            updated_at=now,
        )


# ---------------------------------------------------------------------------
# MQTT deterministic Home Intent bridge
# ---------------------------------------------------------------------------


mqtt_client = mqtt.Client(
    mqtt.CallbackAPIVersion.VERSION2,
    client_id="home-intent-proxy-" + uuid.uuid4().hex[:8],
)



def mqtt_on_connect(client, userdata, flags, reason_code, properties):
    global event_loop, mqtt_ready
    if reason_code != 0:
        log.error("MQTT connection failed: %s", reason_code)
        return

    log.info(
        "MQTT CONNECTED host=%s port=%s reason=%s",
        MQTT_HOST,
        MQTT_PORT,
        reason_code,
    )
    subscriptions = [
        ("hermes/nlu/query", 0),
        ("hermes/intent/#", 0),
        ("hermes/dialogueManager/endSession", 0),
        ("hermes/nlu/intentNotRecognized", 0),
    ]
    result, mid = client.subscribe(subscriptions)
    log.info(
        "MQTT SUBSCRIBE result=%s mid=%s topics=%s",
        result,
        mid,
        [topic for topic, _ in subscriptions],
    )
    if event_loop is not None and mqtt_ready is not None:
        event_loop.call_soon_threadsafe(mqtt_ready.set)



def mqtt_on_subscribe(client, userdata, mid, reason_code_list, properties):
    log.info("MQTT SUBSCRIBED mid=%s reasons=%s", mid, reason_code_list)



def mqtt_on_disconnect(client, userdata, disconnect_flags, reason_code, properties):
    log.warning("MQTT DISCONNECTED reason=%s", reason_code)



def mark_intent_recognised(session_id: str, payload: dict):
    request = pending.get(session_id)
    if request is None:
        return
    if not request.intent_future.done():
        request.intent_future.set_result(payload)



def resolve_request(session_id: str, text: str):
    request = pending.get(session_id)
    if request is None or request.response_future.done():
        return
    log.info("RESOLVING session=%s text=%r", session_id, text)
    request.response_future.set_result(text)



def fail_request(session_id: str, message: str):
    request = pending.get(session_id)
    if request is None:
        return
    log.warning("FAILING session=%s reason=%s", session_id, message)
    if not request.intent_future.done():
        request.intent_future.set_exception(RuntimeError(message))
    if not request.response_future.done():
        request.response_future.set_exception(RuntimeError(message))



def mqtt_on_message(client, userdata, message):
    global event_loop
    raw = message.payload.decode("utf-8", errors="replace")
    log.info("MQTT RX topic=%s payload=%s", message.topic, raw)
    if event_loop is None:
        return
    try:
        payload = json.loads(raw)
    except Exception:
        log.warning("Ignoring non-JSON MQTT payload topic=%s", message.topic)
        return
    if not isinstance(payload, dict):
        return

    session_id = payload.get("sessionId")
    if message.topic == "hermes/nlu/query":
        log.info(
            "NLU QUERY OBSERVED session=%s input=%r site=%r",
            session_id,
            payload.get("input"),
            payload.get("siteId"),
        )
        return

    if message.topic.startswith("hermes/intent/"):
        intent = payload.get("intent", {})
        log.info(
            "INTENT RECOGNISED topic=%s session=%s intent=%s confidence=%s slots=%s",
            message.topic,
            session_id,
            intent.get("intentName"),
            intent.get("confidenceScore"),
            payload.get("slots", []),
        )
        if session_id:
            event_loop.call_soon_threadsafe(
                mark_intent_recognised, session_id, payload
            )
        return

    if message.topic == "hermes/dialogueManager/endSession":
        if not session_id:
            return
        text = str(payload.get("text", "")).strip() or HOME_INTENT_FALLBACK_RESPONSE
        log.info("HOME INTENT RESPONSE session=%s text=%r", session_id, text)
        event_loop.call_soon_threadsafe(resolve_request, session_id, text)
        return

    if message.topic == "hermes/nlu/intentNotRecognized":
        if not session_id:
            return
        log.warning(
            "INTENT NOT RECOGNISED session=%s input=%r",
            session_id,
            payload.get("input"),
        )
        event_loop.call_soon_threadsafe(
            fail_request,
            session_id,
            "Home Intent did not recognise the command.",
        )


mqtt_client.on_connect = mqtt_on_connect
mqtt_client.on_subscribe = mqtt_on_subscribe
mqtt_client.on_disconnect = mqtt_on_disconnect
mqtt_client.on_message = mqtt_on_message


async def process_home_intent(text: str) -> str:
    if not mqtt_client.is_connected():
        raise RuntimeError("MQTT is not connected")

    original_text = text
    normalized_text = normalize_command(text)
    if not normalized_text:
        raise RuntimeError("Command was empty after normalisation")

    if original_text != normalized_text:
        log.info(
            "NORMALISED COMMAND original=%r normalized=%r",
            original_text,
            normalized_text,
        )

    session_id = str(uuid.uuid4())
    request_id = str(uuid.uuid4())
    loop = asyncio.get_running_loop()
    response_future: asyncio.Future[str] = loop.create_future()
    intent_future: asyncio.Future[dict] = loop.create_future()
    pending[session_id] = PendingRequest(
        response_future=response_future,
        intent_future=intent_future,
        original_text=original_text,
        normalized_text=normalized_text,
        request_id=request_id,
        created_at=time.monotonic(),
    )

    payload = {
        "input": normalized_text,
        "id": request_id,
        "sessionId": session_id,
        "siteId": SITE_ID,
    }
    payload_json = json.dumps(payload)
    log.info(
        "NLU REQUEST session=%s request=%s site=%s text=%r pending=%d",
        session_id,
        request_id,
        SITE_ID,
        normalized_text,
        len(pending),
    )
    result = mqtt_client.publish("hermes/nlu/query", payload_json, qos=0)
    log.info(
        "MQTT TX topic=hermes/nlu/query rc=%s mid=%s payload=%s",
        result.rc,
        result.mid,
        payload_json,
    )
    if result.rc != mqtt.MQTT_ERR_SUCCESS:
        pending.pop(session_id, None)
        raise RuntimeError(f"MQTT publish failed: {result.rc}")

    try:
        try:
            intent_payload = await asyncio.wait_for(
                asyncio.shield(intent_future), timeout=HOME_INTENT_TIMEOUT
            )
        except asyncio.TimeoutError as exc:
            raise RuntimeError(
                "Timed out waiting for Home Intent to recognise the command"
            ) from exc

        intent_info = intent_payload.get("intent", {})
        intent_name = str(intent_info.get("intentName", ""))
        confidence = float(intent_info.get("confidenceScore", 0) or 0)
        log.info(
            "COMMAND RECOGNISED session=%s intent=%s confidence=%s",
            session_id,
            intent_name,
            confidence,
        )

        if confidence < HOME_INTENT_MIN_CONFIDENCE:
            raise RuntimeError(
                "Home Intent matched a low-confidence intent "
                f"({confidence:.2f}); command rejected"
            )

        if response_future.done():
            response = response_future.result()
        else:
            try:
                response = await asyncio.wait_for(
                    asyncio.shield(response_future),
                    timeout=HOME_INTENT_RESPONSE_GRACE,
                )
            except asyncio.TimeoutError:
                response = HOME_INTENT_FALLBACK_RESPONSE

        if is_home_intent_error_response(response):
            raise RuntimeError(
                f"Home Intent returned a configured error response: {response}"
            )
        return response
    finally:
        request = pending.pop(session_id, None)
        if request is not None:
            for future in (request.intent_future, request.response_future):
                if future.done() and not future.cancelled():
                    try:
                        future.exception()
                    except Exception:
                        pass


def validate_fallback_config() -> list[str]:
    missing: list[str] = []
    if not FALLBACK_MODEL:
        missing.append("FALLBACK_MODEL")
    if not HOMEASSISTANT_URL:
        missing.append("HOMEASSISTANT_URL")
    if not HOMEASSISTANT_TOKEN:
        missing.append("HOMEASSISTANT_TOKEN")
    return missing



def validate_music_assistant_config() -> list[str]:
    if not MUSIC_ASSISTANT_ENABLED:
        return []
    missing: list[str] = []
    if not MUSIC_ASSISTANT_URL:
        missing.append("MUSIC_ASSISTANT_URL")
    if not MUSIC_ASSISTANT_TOKEN:
        missing.append("MUSIC_ASSISTANT_TOKEN")
    return missing


async def _recover_spoken_answer_from_successful_data(
    user_text: str,
    successful_data: dict[str, Any] | None,
) -> str | None:
    """One no-tools recovery turn after HA has already returned useful data."""
    if not DATA_RESPONSE_RECOVERY_ENABLED or not isinstance(successful_data, dict):
        return None

    raw_payload = json.dumps(
        successful_data,
        ensure_ascii=False,
        default=str,
    )
    if len(raw_payload) > DATA_RESPONSE_RECOVERY_MAX_CHARS:
        raw_payload = raw_payload[:DATA_RESPONSE_RECOVERY_MAX_CHARS] + "...[truncated]"

    prompt = (
        "You are producing the final spoken answer for a Home Assistant voice request.\n"
        "Home Assistant already returned successful data. Do not call tools and do not "
        "question whether the data is available.\n"
        "Answer only the user's request from the supplied data. Use one short natural "
        "sentence, ideally under twelve words. No markdown, URLs, brackets, tool names, "
        "entity IDs, implementation details, follow-up offers, or extra explanation.\n\n"
        f"{_runtime_context()}\n\n"
        f"User request: {user_text.strip()}\n\n"
        f"Successful Home Assistant data: {raw_payload}"
    )

    try:
        client = AsyncOpenAI(
            base_url=LEMONADE_BASE_URL,
            api_key=LEMONADE_API_KEY,
            timeout=FALLBACK_TIMEOUT,
        )
        completion = await client.chat.completions.create(
            model=FALLBACK_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        content = completion.choices[0].message.content if completion.choices else None
        response = sanitise_spoken_response(str(content or "").strip())
        if response:
            log.info(
                "LLM DATA RECOVERY COMPLETE response=%r",
                response,
            )
            return response
    except Exception:
        log.exception("LLM data-response recovery summarisation failed")

    return None


def _origin_runtime_context(origin_context: dict[str, Any] | None) -> str:
    if not origin_context:
        return "Voice origin context: not supplied by the caller."
    parts: list[str] = []
    if origin_context.get("device_name"):
        parts.append(f"device={origin_context['device_name']}")
    if origin_context.get("area_name"):
        parts.append(f"area={origin_context['area_name']}")
    elif origin_context.get("area_id"):
        parts.append(f"area_id={origin_context['area_id']}")
    if origin_context.get("floor_name"):
        parts.append(f"floor={origin_context['floor_name']}")
    if not parts:
        return "Voice origin context: caller supplied no resolvable room."
    return (
        "Trusted voice origin context: "
        + ", ".join(parts)
        + ". Use the area as a soft default only for requests that omit a location; "
        "an explicit user location always overrides it."
    )


async def process_llm_fallback(
    text: str,
    conversation_key: str,
    client_history: list[dict[str, str]] | None = None,
    origin_context: dict[str, Any] | None = None,
) -> str:
    if not MCP_FALLBACK_ENABLED:
        raise RuntimeError("LLM fallback is disabled")
    if fallback_agent is None:
        raise RuntimeError("LLM fallback agent is unavailable")

    if client_history:
        await seed_conversation_history(conversation_key, client_history)
        history = client_history[-(CONVERSATION_HISTORY_TURNS * 2) :]
        history_source = "client"
    else:
        history = await get_conversation_history(conversation_key)
        history_source = "proxy" if history else "none"

    current_input = (
        f"{_runtime_context()}\n"
        f"{_origin_runtime_context(origin_context)}\n\n"
        f"Latest user request: {text.strip()}"
    )
    agent_input: list[dict[str, str]] = [
        *history,
        {"role": "user", "content": current_input},
    ]

    log.info(
        "LLM FALLBACK START text=%r history_source=%s history_messages=%d ws_ready=%s origin_device=%r origin_area=%r origin_source=%r",
        text,
        history_source,
        len(history),
        bool(ha_ws and ha_ws.ready.is_set()),
        (origin_context or {}).get("device_name"),
        (origin_context or {}).get("area_name"),
        (origin_context or {}).get("source"),
    )

    response: str | None = None
    llm_calls = 0

    async with fallback_lock:
        _reset_voice_tool_run_state(text, origin_context)

        # HA/OpenAI-compatible clients can automatically retry a request when a
        # prior HTTP response was interrupted. If the same request just completed
        # a confirmed Music Assistant write, return its cached spoken response
        # instead of executing the MCP action again.
        replay_response = _get_recent_music_action_response(
            conversation_key,
            text,
            origin_context,
        )
        if replay_response:
            log.warning(
                "MUSIC ACTION REPLAY GUARD HIT text=%r response=%r",
                text,
                replay_response,
            )
            return replay_response

        try:
            result = await asyncio.wait_for(
                Runner.run(
                    fallback_agent,
                    agent_input,
                    max_turns=FALLBACK_MAX_TURNS,
                ),
                timeout=FALLBACK_TIMEOUT,
            )
            llm_calls = len(getattr(result, "raw_responses", []) or [])
            response = sanitise_spoken_response(
                str(result.final_output or "").strip()
            )

            # v6.5 safety net: if a Music Assistant write somehow completed but
            # the runner still returned an empty final output, never turn that
            # into HTTP 502. The write already happened.
            if (
                not response
                and voice_tool_run_state.last_successful_music_action
            ):
                response = str(
                    voice_tool_run_state.last_successful_music_action.get("spoken")
                    or FAST_ACTION_RESPONSE
                )
                log.warning(
                    "Recovered empty final output from successful Music Assistant "
                    "action response=%r",
                    response,
                )

            # A model can occasionally consume a successful service_response but
            # emit no final text. Recover from the already-successful data rather
            # than turning that into HTTP 502.
            if not response and voice_tool_run_state.last_successful_data:
                log.warning(
                    "LLM returned empty final output after successful HA data; "
                    "attempting one no-tools recovery summary"
                )
                response = await _recover_spoken_answer_from_successful_data(
                    text,
                    voice_tool_run_state.last_successful_data,
                )

        except MaxTurnsExceeded:
            if voice_tool_run_state.last_successful_music_action:
                response = str(
                    voice_tool_run_state.last_successful_music_action.get("spoken")
                    or FAST_ACTION_RESPONSE
                )
                log.warning(
                    "LLM max turns reached AFTER successful Music Assistant "
                    "action; returning confirmed action response=%r",
                    response,
                )
            elif voice_tool_run_state.last_successful_data:
                log.warning(
                    "LLM max turns reached AFTER successful HA data; "
                    "attempting one no-tools recovery summary"
                )
                response = await _recover_spoken_answer_from_successful_data(
                    text,
                    voice_tool_run_state.last_successful_data,
                )
            else:
                raise
        except Exception:
            # A Music Assistant write is a side effect. If it already succeeded,
            # return its confirmed action response instead of emitting a 502 that
            # could cause the caller to replay the original request.
            if voice_tool_run_state.last_successful_music_action:
                response = str(
                    voice_tool_run_state.last_successful_music_action.get("spoken")
                    or FAST_ACTION_RESPONSE
                )
                log.exception(
                    "LLM runner failed after successful Music Assistant action; "
                    "returning confirmed response=%r",
                    response,
                )
            # Only recover generic runner errors when HA has definitely already
            # returned useful data for this same serialized run.
            elif voice_tool_run_state.last_successful_data:
                log.exception(
                    "LLM runner failed after successful HA data; "
                    "attempting one no-tools recovery summary"
                )
                response = await _recover_spoken_answer_from_successful_data(
                    text,
                    voice_tool_run_state.last_successful_data,
                )
                if not response:
                    raise
            else:
                raise

    if not response:
        raise RuntimeError("LLM fallback returned no response")

    if voice_tool_run_state.last_successful_music_action:
        _remember_music_action_response(
            conversation_key,
            text,
            origin_context,
            response,
        )

    log.info(
        "LLM FALLBACK COMPLETE text=%r response=%r llm_calls=%d history_messages=%d",
        text,
        response,
        llm_calls,
        len(history),
    )
    return response


# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    global event_loop, mqtt_ready, ha_ws, mcp_manager
    global advanced_agent, fallback_agent, music_assistant_native

    event_loop = asyncio.get_running_loop()
    mqtt_ready = asyncio.Event()

    if MQTT_USERNAME:
        mqtt_client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)

    log.info("Connecting to MQTT broker %s:%s", MQTT_HOST, MQTT_PORT)
    mqtt_client.connect_async(MQTT_HOST, MQTT_PORT, keepalive=60)
    mqtt_client.loop_start()
    try:
        await asyncio.wait_for(mqtt_ready.wait(), timeout=10)
    except asyncio.TimeoutError as exc:
        mqtt_client.loop_stop()
        raise RuntimeError(
            f"Timed out connecting to MQTT broker {MQTT_HOST}:{MQTT_PORT}"
        ) from exc
    log.info("MQTT ready; deterministic Home Intent path is available")

    missing = validate_fallback_config() if MCP_FALLBACK_ENABLED else []

    if MCP_FALLBACK_ENABLED and not missing and HA_WS_ENABLED:
        try:
            ha_ws = HomeAssistantWebSocket(HOMEASSISTANT_URL, HOMEASSISTANT_TOKEN)
            await ha_ws.start()
        except Exception:
            log.exception("Failed to initialise direct Home Assistant WebSocket")

    # v6.8.2: direct official Music Assistant client. Its WebSocket receive loop
    # maintains players/queues in memory and the supervisor reconnects with a new
    # client object if the connection is lost.
    music_missing = validate_music_assistant_config()
    music_assistant_native = None
    if MCP_FALLBACK_ENABLED and not missing and MUSIC_ASSISTANT_ENABLED:
        if music_missing:
            log.warning(
                "Music Assistant native integration unavailable; missing configuration: %s",
                ", ".join(music_missing),
            )
        elif MusicAssistantClient is None:
            log.error(
                "Music Assistant native integration unavailable; install "
                "music-assistant-client==1.4.3 (%s)",
                MUSIC_ASSISTANT_CLIENT_IMPORT_ERROR,
            )
        else:
            music_assistant_native = NativeMusicAssistant(
                MUSIC_ASSISTANT_URL, MUSIC_ASSISTANT_TOKEN
            )
            initial_ready = await music_assistant_native.start()
            log.info(
                "Music Assistant native transport configured url=%r ready=%s area_map=%s",
                MUSIC_ASSISTANT_URL,
                initial_ready,
                bool(_parse_music_area_player_map()),
            )

    # HA advanced specialist remains the only MCP subprocess in v6.8.2.
    manager_context = None
    if MCP_FALLBACK_ENABLED and not missing and HA_MCP_ADVANCED_ENABLED:
        try:
            ha_mcp_server = make_ha_mcp_server()
            log.info(
                "Starting advanced HA MCP command=%r args=%r",
                HA_MCP_COMMAND,
                HA_MCP_ARGS,
            )
            mcp_manager = MCPServerManager(
                [ha_mcp_server],
                connect_timeout_seconds=MCP_CONNECT_TIMEOUT,
                cleanup_timeout_seconds=MCP_CLEANUP_TIMEOUT,
                strict=False,
                drop_failed_servers=True,
            )
            manager_context = mcp_manager
            await manager_context.__aenter__()
            active_ha = next(
                (
                    server
                    for server in mcp_manager.active_servers
                    if server.name == "Home Assistant Advanced"
                ),
                None,
            )
            if active_ha is not None:
                advanced_agent = make_advanced_agent([active_ha])
                log.info("Advanced ha-mcp specialist ready")
            else:
                advanced_agent = None
                log.warning("Advanced ha-mcp unavailable; MCP errors=%s", mcp_manager.errors)
        except Exception:
            advanced_agent = None
            log.exception("Failed to initialise advanced ha-mcp server")

    if MCP_FALLBACK_ENABLED:
        if missing:
            log.warning(
                "LLM fallback unavailable; missing configuration: %s",
                ", ".join(missing),
            )
        else:
            try:
                music_tools_enabled = music_assistant_native is not None
                fallback_agent = make_fallback_agent(music_tools_enabled)
                log.info(
                    "LLM fallback ready model=%s direct_ws=%s advanced=%s "
                    "music_assistant=%s music_transport=native_websocket native_ready=%s",
                    FALLBACK_MODEL,
                    bool(ha_ws),
                    advanced_agent is not None,
                    music_tools_enabled,
                    bool(music_assistant_native and music_assistant_native.connected),
                )
            except Exception:
                fallback_agent = None
                log.exception("Failed to initialise LLM fallback agent")

    try:
        yield
    finally:
        log.info("Shutting down proxy")

        for session_id, request in list(pending.items()):
            message = "Proxy is shutting down"
            if not request.intent_future.done():
                request.intent_future.set_exception(RuntimeError(message))
            if not request.response_future.done():
                request.response_future.set_exception(RuntimeError(message))
        pending.clear()

        fallback_agent = None
        advanced_agent = None

        if music_assistant_native is not None:
            try:
                await music_assistant_native.stop()
            except Exception:
                log.exception("Error shutting down native Music Assistant client")
            music_assistant_native = None

        try:
            await voice_activity_indicators.stop_all()
        except Exception:
            log.exception("Error restoring voice activity indicators")

        if ha_ws is not None:
            try:
                await ha_ws.stop()
            except Exception:
                log.exception("Error shutting down HA WebSocket")
            ha_ws = None

        if manager_context is not None:
            try:
                await manager_context.__aexit__(None, None, None)
            except Exception:
                log.exception("Error shutting down MCP manager")
        mcp_manager = None

        mqtt_client.disconnect()
        mqtt_client.loop_stop()


# ---------------------------------------------------------------------------
# FastAPI / OpenAI-compatible API
# ---------------------------------------------------------------------------


app = FastAPI(
    title="Home Intent + HA WebSocket + Native Music Assistant OpenAI Proxy",
    version=PROXY_VERSION,
    lifespan=lifespan,
)



def extract_user_message(body: dict) -> str:
    for message in reversed(body.get("messages", [])):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        return _message_text(message)
    return ""


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    if body.get("stream"):
        raise HTTPException(status_code=400, detail="Streaming is not supported")

    text = extract_user_message(body)
    if not text:
        raise HTTPException(status_code=400, detail="No user message supplied")

    conversation_key = get_conversation_key(request, body)
    client_history = extract_client_history(body)
    origin_context = await extract_voice_origin_context(request, body)

    if body.get("reset_conversation"):
        await clear_conversation_history(conversation_key)
        client_history = []
        log.info("CONVERSATION RESET key=%r", conversation_key)

    route = "home-intent"
    try:
        speech = await process_home_intent(text)
        log.info(
            "ROUTE COMPLETE route=home-intent text=%r response=%r",
            text,
            speech,
        )
    except Exception as intent_exc:
        log.warning("HOME INTENT FALLBACK text=%r reason=%s", text, intent_exc)
        try:
            speech = await process_llm_fallback(
                text,
                conversation_key=conversation_key,
                client_history=client_history,
                origin_context=origin_context,
            )
            route = "llm-ha-ws"
            log.info(
                "ROUTE COMPLETE route=llm-ha-ws text=%r response=%r",
                text,
                speech,
            )
        except Exception as fallback_exc:
            log.exception("LLM/Home Assistant fallback failed")
            raise HTTPException(
                status_code=502,
                detail=(
                    "Both Home Intent and LLM fallback failed. "
                    f"Home Intent: {intent_exc}; "
                    f"LLM fallback: {fallback_exc}"
                ),
            ) from fallback_exc

    if client_history:
        await seed_conversation_history(conversation_key, client_history)
    await remember_conversation_turn(conversation_key, text, speech)
    history_messages = len(await get_conversation_history(conversation_key))

    return {
        "id": "chatcmpl-homeintent-" + uuid.uuid4().hex,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": speech},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        "home_intent_proxy": {
            "route": route,
            "conversation_history_messages": history_messages,
            "voice_origin": {
                "device_name": (origin_context or {}).get("device_name"),
                "area_name": (origin_context or {}).get("area_name"),
                "area_id": (origin_context or {}).get("area_id"),
                "floor_name": (origin_context or {}).get("floor_name"),
                "source": (origin_context or {}).get("source"),
            },
        },
    }


@app.get("/v1/models")
async def models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": 0,
                "owned_by": "home-intent",
            }
        ],
    }


@app.get("/health")
async def health():
    mqtt_connected = mqtt_client.is_connected()
    ws_ready = bool(ha_ws and ha_ws.ready.is_set())

    ma_manager = music_assistant_native
    ma_client = ma_manager.client if ma_manager is not None else None
    ma_info = getattr(ma_client, "server_info", None) if ma_client is not None else None

    return {
        "status": "ok" if mqtt_connected else "degraded",
        "version": PROXY_VERSION,
        "mqtt_connected": mqtt_connected,
        "mqtt_host": MQTT_HOST,
        "mqtt_port": MQTT_PORT,
        "site_id": SITE_ID,
        "pending_requests": len(pending),
        "home_intent_timeout_seconds": HOME_INTENT_TIMEOUT,
        "response_grace_seconds": HOME_INTENT_RESPONSE_GRACE,
        "minimum_confidence": HOME_INTENT_MIN_CONFIDENCE,
        "fallback_response": HOME_INTENT_FALLBACK_RESPONSE,
        "configured_error_phrases": list(HOME_INTENT_ERROR_PHRASES),
        "llm_fallback_enabled": MCP_FALLBACK_ENABLED,
        "llm_fallback_ready": fallback_agent is not None,
        "fallback_model": FALLBACK_MODEL or None,
        "music_assistant_enabled": MUSIC_ASSISTANT_ENABLED,
        "music_assistant_transport": "native_websocket",
        "music_assistant_client_package_available": MusicAssistantClient is not None,
        "music_assistant_client_import_error": MUSIC_ASSISTANT_CLIENT_IMPORT_ERROR,
        "music_assistant_ready": bool(ma_manager and ma_manager.connected),
        "music_assistant_url": MUSIC_ASSISTANT_URL or None,
        "music_assistant_server_version": getattr(ma_info, "server_version", None),
        "music_assistant_schema_version": getattr(ma_info, "schema_version", None),
        "music_assistant_players_cached": (
            len(getattr(ma_client.players, "players", []) or []) if ma_client is not None else 0
        ),
        "music_assistant_queues_cached": (
            len(getattr(ma_client.player_queues, "player_queues", []) or [])
            if ma_client is not None
            else 0
        ),
        "music_assistant_connection_count": (
            ma_manager.connection_count if ma_manager is not None else 0
        ),
        "music_assistant_reconnect_count": (
            ma_manager.reconnect_count if ma_manager is not None else 0
        ),
        "music_assistant_last_connected_at": (
            ma_manager.last_connected_at if ma_manager is not None else None
        ),
        "music_assistant_last_error": (
            ma_manager.last_error if ma_manager is not None else None
        ),
        "music_assistant_area_player_map_configured": bool(
            _parse_music_area_player_map()
        ),
        "music_assistant_terminal_actions": MUSIC_ASSISTANT_TERMINAL_ACTIONS,
        "music_assistant_replay_guard_seconds": MUSIC_ASSISTANT_REPLAY_GUARD_SECONDS,
        "music_assistant_connect_timeout_seconds": MUSIC_ASSISTANT_CONNECT_TIMEOUT_SECONDS,
        "music_assistant_command_timeout_seconds": MUSIC_ASSISTANT_COMMAND_TIMEOUT_SECONDS,
        "music_assistant_play_ack_wait_seconds": MUSIC_ASSISTANT_PLAY_ACK_WAIT_SECONDS,
        "music_assistant_play_completion_timeout_seconds": MUSIC_ASSISTANT_PLAY_COMPLETION_TIMEOUT_SECONDS,
        "music_assistant_first_audio_timeout_seconds": MUSIC_ASSISTANT_FIRST_AUDIO_TIMEOUT_SECONDS,
        "music_assistant_background_timeout_seconds": MUSIC_ASSISTANT_BACKGROUND_TIMEOUT_SECONDS,
        "music_assistant_first_audio_poll_seconds": MUSIC_ASSISTANT_FIRST_AUDIO_POLL_SECONDS,
        "music_assistant_radio_seed_top_n": MUSIC_ASSISTANT_RADIO_SEED_TOP_N,
        "music_assistant_radio_seed_strategy": MUSIC_ASSISTANT_RADIO_SEED_STRATEGY,
        "music_assistant_inflight_playbacks": (
            ma_manager.inflight_playback_count if ma_manager is not None else 0
        ),
        "music_assistant_activity_indicator_enabled": MUSIC_ASSISTANT_ACTIVITY_INDICATOR_ENABLED,
        "music_assistant_activity_indicator_active": voice_activity_indicators.active_count,
        "music_assistant_activity_indicator_color": MUSIC_ASSISTANT_ACTIVITY_INDICATOR_COLOR,
        "music_assistant_activity_indicator_effect": MUSIC_ASSISTANT_ACTIVITY_INDICATOR_EFFECT,
        "music_assistant_activity_indicator_green_legacy": MUSIC_ASSISTANT_ACTIVITY_INDICATOR_GREEN,
        "music_assistant_activity_indicator_software_pulse": MUSIC_ASSISTANT_ACTIVITY_INDICATOR_SOFTWARE_PULSE,
        "music_assistant_activity_indicator_pulse_interval_seconds": MUSIC_ASSISTANT_ACTIVITY_INDICATOR_PULSE_INTERVAL_SECONDS,
        "music_assistant_activity_indicator_domains": list(MUSIC_ASSISTANT_ACTIVITY_INDICATOR_DOMAINS),
        "music_assistant_activity_indicator_last_target": voice_activity_indicators.last_target,
        "music_assistant_activity_indicator_last_error": voice_activity_indicators.last_error,
        "music_assistant_post_action_settle_seconds": MUSIC_ASSISTANT_POST_ACTION_SETTLE_SECONDS,
        "lemonade_base_url": LEMONADE_BASE_URL if MCP_FALLBACK_ENABLED else None,
        "ha_ws_enabled": HA_WS_ENABLED,
        "ha_ws_ready": ws_ready,
        "ha_ws_url": ha_ws.ws_url if ha_ws is not None else None,
        "ha_ws_states_cached": len(ha_ws.states) if ha_ws is not None else 0,
        "ha_ws_service_domains_cached": len(ha_ws.services) if ha_ws is not None else 0,
        "ha_ws_entity_registry_cached": len(ha_ws.entity_registry) if ha_ws is not None else 0,
        "ha_ws_device_registry_cached": len(ha_ws.devices) if ha_ws is not None else 0,
        "ha_ws_area_registry_cached": len(ha_ws.areas) if ha_ws is not None else 0,
        "voice_origin_context_enabled": VOICE_ORIGIN_CONTEXT_ENABLED,
        "voice_origin_area_bias": VOICE_ORIGIN_AREA_BIAS,
        "voice_origin_soft_area_ranking": VOICE_ORIGIN_SOFT_AREA_RANKING,
        "generic_light_indicator_penalty": GENERIC_LIGHT_INDICATOR_PENALTY,
        "voice_origin_system_prompt_fallback": VOICE_ORIGIN_SYSTEM_PROMPT_FALLBACK,
        "ha_ws_state_events_seen": ha_ws.state_event_count if ha_ws is not None else 0,
        "ha_ws_reconnect_count": ha_ws.reconnect_count if ha_ws is not None else 0,
        "ha_ws_last_error": ha_ws.last_error if ha_ws is not None else None,
        "ha_ws_connect_timeout_seconds": HA_WS_CONNECT_TIMEOUT,
        "ha_ws_command_timeout_seconds": HA_WS_COMMAND_TIMEOUT,
        "ha_ws_service_cache_ttl_seconds": HA_WS_SERVICE_CACHE_TTL,
        "ha_ws_registry_cache_ttl_seconds": HA_WS_REGISTRY_CACHE_TTL,
        "ha_service_schema_auto_repair": HA_SERVICE_SCHEMA_AUTO_REPAIR,
        "data_response_recovery_enabled": DATA_RESPONSE_RECOVERY_ENABLED,
        "data_response_recovery_max_chars": DATA_RESPONSE_RECOVERY_MAX_CHARS,
        "ha_mcp_advanced_enabled": HA_MCP_ADVANCED_ENABLED,
        "ha_mcp_advanced_ready": advanced_agent is not None,
        "ha_mcp_command": HA_MCP_COMMAND if HA_MCP_ADVANCED_ENABLED else None,
        "ha_mcp_args": HA_MCP_ARGS if HA_MCP_ADVANCED_ENABLED else None,
        "ha_mcp_tool_search": HA_MCP_TOOL_SEARCH,
        "ha_mcp_pinned_tools": HA_MCP_PINNED_TOOLS,
        "mcp_active_servers": (
            [server.name for server in mcp_manager.active_servers]
            if mcp_manager is not None
            else []
        ),
        "mcp_failed_servers": (
            [server.name for server in mcp_manager.failed_servers]
            if mcp_manager is not None
            else []
        ),
        "fast_action_response": FAST_ACTION_RESPONSE,
        "spoken_max_chars": SPOKEN_MAX_CHARS,
        "fallback_max_turns": FALLBACK_MAX_TURNS,
        "advanced_max_turns": ADVANCED_MAX_TURNS,
        "conversation_history_enabled": CONVERSATION_HISTORY_ENABLED,
        "conversation_history_turns": CONVERSATION_HISTORY_TURNS,
        "conversation_history_ttl_seconds": CONVERSATION_HISTORY_TTL_SECONDS,
        "conversation_history_active_sessions": len(conversation_memories),
        "conversation_history_max_sessions": CONVERSATION_HISTORY_MAX_SESSIONS,
        "local_timezone": LOCAL_TIMEZONE,
    }
