"""HTTP conversation and voice-origin support."""

import re
import time
from typing import Any

from fastapi import Request

from intent_bridge.config import log, settings
from intent_bridge.core import text as text_policy
from intent_bridge.runtime.context import runtime_context
from intent_bridge.runtime.dependencies import runtime
from intent_bridge.runtime.stores import (
    ConversationMemory,
    conversation_history_lock,
    conversation_memories,
)

# ---------------------------------------------------------------------------
# Runtime time and conversation memory
# ---------------------------------------------------------------------------


def _runtime_context() -> str:
    return runtime_context(settings.api.timezone)


def _message_text(message: dict) -> str:
    return text_policy.message_text(message)


def extract_client_history(body: dict) -> list[dict[str, str]]:
    return text_policy.extract_client_history(body, settings.conversation.history_turns)


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
    if not settings.voice_origin.system_prompt_fallback_enabled:
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
    if not settings.voice_origin.enabled:
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

    if runtime.ha_ws is not None:
        try:
            await runtime.ha_ws.refresh_registries()
            resolved = runtime.ha_ws.resolve_device_origin(
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
    metadata_id = metadata.get("conversation_id") if isinstance(metadata, dict) else None
    candidates = (
        request.headers.get("X-Conversation-ID"),
        body.get("conversation_id"),
        metadata_id,
        body.get("user"),
    )
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()[:160]
    return "site:default"


def _prune_conversation_memories_locked(now: float) -> None:
    expired = [
        key
        for key, memory in conversation_memories.items()
        if now - memory.updated_at > settings.conversation.ttl_seconds
    ]
    for key in expired:
        conversation_memories.pop(key, None)

    if len(conversation_memories) <= settings.conversation.max_sessions:
        return
    oldest = sorted(conversation_memories.items(), key=lambda item: item[1].updated_at)
    excess = len(conversation_memories) - settings.conversation.max_sessions
    for key, _ in oldest[:excess]:
        conversation_memories.pop(key, None)


async def clear_conversation_history(conversation_key: str) -> None:
    async with conversation_history_lock:
        conversation_memories.pop(conversation_key, None)


async def seed_conversation_history(
    conversation_key: str,
    messages: list[dict[str, str]],
) -> None:
    if not settings.conversation.enabled or not messages:
        return
    now = time.monotonic()
    async with conversation_history_lock:
        _prune_conversation_memories_locked(now)
        conversation_memories[conversation_key] = ConversationMemory(
            messages=messages[-(settings.conversation.history_turns * 2) :],
            updated_at=now,
        )


async def get_conversation_history(
    conversation_key: str,
) -> list[dict[str, str]]:
    if not settings.conversation.enabled:
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
    if not settings.conversation.enabled:
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
            messages=messages[-(settings.conversation.history_turns * 2) :],
            updated_at=now,
        )
