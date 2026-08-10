"""Agent tool-result classification and terminal response policy."""

import re
import time
from typing import Any

from agents import (
    FunctionToolResult,
    RunContextWrapper,
)
from agents.agent import ToolsToFinalOutputResult

from intent_bridge.config import log, settings
from intent_bridge.core import text as text_policy
from intent_bridge.core.tool_output import (
    serialise_tool_output as _serialise_tool_output,
)
from intent_bridge.core.tool_output import (
    tool_output_failed as _tool_output_failed,
)
from intent_bridge.core.tool_output import (
    tool_output_mapping as _tool_output_mapping,
)
from intent_bridge.music_assistant.policy import (
    MUSIC_ASSISTANT_ALWAYS_WRITE_TOOLS,
)
from intent_bridge.runtime.execution import (
    normalize_command,
    voice_tool_run_state,
)
from intent_bridge.runtime.stores import recent_music_action_responses


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
        return settings.api.action_confirmation

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
        return settings.api.action_confirmation

    if tool_name == "ma_transfer_queue":
        return "Playback moved."

    if tool_name == "ma_group":
        if "removed from" in lower:
            return "Speakers ungrouped."
        return "Speakers grouped."

    if tool_name in {"ma_queue", "ma_queue_item"}:
        return settings.api.action_confirmation

    return settings.api.action_confirmation


def _music_replay_cache_key(
    conversation_key: str,
    text: str,
    origin_context: dict[str, Any] | None,
) -> str:
    origin_context = origin_context or {}
    area = (
        str(origin_context.get("area_id") or origin_context.get("area_name") or "")
        .strip()
        .casefold()
    )
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
    if settings.music_assistant.replay_guard_seconds <= 0:
        return None

    now = time.monotonic()
    cutoff = now - settings.music_assistant.replay_guard_seconds

    # Opportunistic cleanup; this cache is intentionally tiny and short-lived.
    expired = [
        key for key, (created_at, _) in recent_music_action_responses.items() if created_at < cutoff
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
    if settings.music_assistant.replay_guard_seconds <= 0:
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
        result for result in tool_results if _tool_result_name(result).startswith("ma_")
    ]

    music_write_results = [
        result
        for result in music_results
        if _is_music_assistant_write_result(
            _tool_result_name(result),
            result.output,
        )
    ]

    if settings.music_assistant.terminal_actions_enabled and music_write_results:
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
        result for result in tool_results if _tool_result_name(result) == "ha_call_service"
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
            and service_name not in settings.home_assistant.state_changing_services
        ):
            return ToolsToFinalOutputResult(
                is_final_output=False,
                final_output=None,
            )

    log.info(
        "FAST HA ACTION BATCH COMPLETE calls=%d response=%r",
        len(service_results),
        settings.api.action_confirmation,
    )
    return ToolsToFinalOutputResult(
        is_final_output=True,
        final_output=settings.api.action_confirmation,
    )


def sanitise_spoken_response(text: str) -> str:
    return text_policy.sanitise_spoken_response(text, settings.api.spoken_response_max_chars)
