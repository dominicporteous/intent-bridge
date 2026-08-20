"""Request-local execution state and compatibility policies."""

import json
import re
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Protocol

from intent_bridge.config import settings
from intent_bridge.core import text as text_policy
from intent_bridge.home_assistant import policy as ha_domain
from intent_bridge.runtime.stores import (
    ConversationMemory as ConversationMemory,
)
from intent_bridge.runtime.stores import (
    PendingRequest as PendingRequest,
)
from intent_bridge.runtime.stores import (
    conversation_history_lock as conversation_history_lock,
)
from intent_bridge.runtime.stores import (
    conversation_memories as conversation_memories,
)
from intent_bridge.runtime.stores import (
    pending_requests,
)
from intent_bridge.runtime.stores import (
    recent_music_action_responses as recent_music_action_responses,
)


@dataclass
class VoiceToolRunState:
    request_text: str = ""
    allow_conversation_websocket: bool = False
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
    last_successful_ha_action: dict[str, Any] | None = None
    last_successful_music_action: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.last_entity_by_domain is None:
            self.last_entity_by_domain = {}


_voice_tool_run_state: ContextVar[VoiceToolRunState | None] = ContextVar(
    "voice_tool_run_state",
    default=None,
)


def current_voice_tool_run_state() -> VoiceToolRunState:
    """Return state local to the current async execution context."""
    current = _voice_tool_run_state.get()
    if current is None:
        current = VoiceToolRunState()
        _voice_tool_run_state.set(current)
    return current


class _VoiceToolRunStateProxy:
    """Compatibility facade while tools migrate to explicit run contexts."""

    def __getattr__(self, name: str) -> Any:
        return getattr(current_voice_tool_run_state(), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(current_voice_tool_run_state(), name, value)


voice_tool_run_state = _VoiceToolRunStateProxy()


def _reset_voice_tool_run_state(
    request_text: str,
    origin_context: dict[str, Any] | None = None,
    *,
    allow_conversation_websocket: bool = False,
) -> None:
    origin_context = origin_context or {}
    fresh = VoiceToolRunState(
        request_text=request_text.strip(),
        allow_conversation_websocket=allow_conversation_websocket,
        origin_device_id=origin_context.get("device_id"),
        origin_device_name=origin_context.get("device_name"),
        origin_area_id=origin_context.get("area_id"),
        origin_area_name=origin_context.get("area_name"),
        origin_floor_name=origin_context.get("floor_name"),
        origin_source=origin_context.get("source"),
    )
    _voice_tool_run_state.set(fresh)


# Compatibility alias for the former state-module store name.
pending = pending_requests


class _HomeAssistantCache(Protocol):
    states: dict[str, dict[str, Any]]
    services: dict[str, Any]


def normalize_command(text: str) -> str:
    """Normalise STT formatting without semantic rewriting."""
    return text_policy.normalize_command(text)


def is_home_intent_error_response(text: str) -> bool:
    if not settings.deterministic.error_phrases:
        return False
    normalized = text.strip().casefold()
    return any(phrase in normalized for phrase in settings.deterministic.error_phrases)


def _normalise_search_text(value: Any) -> str:
    """Compatibility wrapper for the extracted text policy."""
    return text_policy.normalize_search_text(value)


def _ha_websocket_url(base_url: str) -> str:
    return ha_domain.websocket_url(base_url)


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
    client: _HomeAssistantCache,
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
        auto_repair=settings.home_assistant.schema_auto_repair_enabled,
    )


def _single_cached_entity_for_domain(
    client: _HomeAssistantCache,
    domain: str,
) -> str | None:
    matches = [
        entity_id
        for entity_id in client.states
        if entity_id.split(".", 1)[0].casefold() == domain.casefold()
    ]
    return matches[0] if len(matches) == 1 else None
