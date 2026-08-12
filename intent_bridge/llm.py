"""LLM fallback handling for intent bridge voice requests."""

import asyncio
import json
from typing import Any

from agents import AsyncOpenAI, Runner
from agents.exceptions import MaxTurnsExceeded

from intent_bridge.agents.results import (
    _get_recent_music_action_response,
    _remember_music_action_response,
    sanitise_spoken_response,
)
from intent_bridge.api.conversation import (
    _runtime_context,
    get_conversation_history,
    seed_conversation_history,
)
from intent_bridge.config import log, settings
from intent_bridge.runtime.context import informational_runtime_context, origin_runtime_context
from intent_bridge.runtime.dependencies import runtime
from intent_bridge.runtime.execution import (
    _reset_voice_tool_run_state,
    voice_tool_run_state,
)


async def process_llm_fallback(
    text: str,
    conversation_key: str,
    client_history: list[dict[str, str]] | None = None,
    origin_context: dict[str, Any] | None = None,
) -> str:
    if not settings.llm.enabled:
        raise RuntimeError("LLM fallback is disabled")
    if runtime.fallback_agent is None:
        raise RuntimeError("LLM fallback agent is unavailable")

    if client_history:
        await seed_conversation_history(conversation_key, client_history)
        history = client_history[-(settings.conversation.history_turns * 2) :]
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
        bool(runtime.ha_ws and runtime.ha_ws.ready.is_set()),
        (origin_context or {}).get("device_name"),
        (origin_context or {}).get("area_name"),
        (origin_context or {}).get("source"),
    )

    response: str | None = None
    llm_calls = 0

    async with runtime.fallback_lock:
        _reset_voice_tool_run_state(text, origin_context)

        replay_response = _get_recent_music_action_response(
            conversation_key,
            text,
            origin_context,
        )
        if replay_response is not None:
            log.warning(
                "MUSIC ACTION REPLAY GUARD HIT text=%r response=%r",
                text,
                replay_response,
            )
            return replay_response

        try:
            result = await asyncio.wait_for(
                Runner.run(
                    runtime.fallback_agent,
                    agent_input,
                    max_turns=settings.llm.max_turns,
                ),
                timeout=settings.llm.timeout_seconds,
            )
            llm_calls = len(getattr(result, "raw_responses", []) or [])
            final_output = result.final_output
            response = (
                None
                if final_output is None
                else sanitise_spoken_response(str(final_output).strip())
            )

            if not response and voice_tool_run_state.last_successful_music_action:
                response = str(
                    voice_tool_run_state.last_successful_music_action.get("spoken")
                    or settings.api.action_confirmation
                )
                log.warning(
                    "Recovered empty final output from successful Music Assistant "
                    "action response=%r",
                    response,
                )

            if not response and voice_tool_run_state.last_successful_ha_action:
                response = settings.api.action_confirmation
                log.warning(
                    "Recovered empty final output from successful Home Assistant "
                    "action response=%r",
                    response,
                )

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
                    or settings.api.action_confirmation
                )
                log.warning(
                    "LLM max turns reached AFTER successful Music Assistant "
                    "action; returning confirmed action response=%r",
                    response,
                )
            elif voice_tool_run_state.last_successful_ha_action:
                response = settings.api.action_confirmation
                log.warning(
                    "LLM max turns reached AFTER successful Home Assistant "
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
            if voice_tool_run_state.last_successful_music_action:
                response = str(
                    voice_tool_run_state.last_successful_music_action.get("spoken")
                    or settings.api.action_confirmation
                )
                log.exception(
                    "LLM runner failed after successful Music Assistant action; "
                    "returning confirmed response=%r",
                    response,
                )
            elif voice_tool_run_state.last_successful_ha_action:
                response = settings.api.action_confirmation
                log.exception(
                    "LLM runner failed after successful Home Assistant action; "
                    "returning confirmed response=%r",
                    response,
                )
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

    successful_silent_action = bool(
        voice_tool_run_state.last_successful_ha_action
        or voice_tool_run_state.last_successful_music_action
    )
    if response is None or (not response and not successful_silent_action):
        raise RuntimeError("LLM fallback returned no response")

    if voice_tool_run_state.last_successful_music_action:
        _remember_music_action_response(
            conversation_key,
            text,
            origin_context,
            response,
        )

    log.info(
        "LLM FALLBACK COMPLETE text=%r response=%r calls=%d",
        text,
        response,
        llm_calls,
    )
    return response


async def process_informational_query(
    text: str,
    conversation_key: str,
    client_history: list[dict[str, str]] | None = None,
    origin_context: dict[str, Any] | None = None,
) -> str:
    """Run a general query through the agent that has no household tools."""
    if not settings.llm.enabled:
        raise RuntimeError("LLM fallback is disabled")
    if runtime.informational_agent is None:
        raise RuntimeError("Informational LLM agent is unavailable")

    if client_history:
        await seed_conversation_history(conversation_key, client_history)
        history = client_history[-(settings.conversation.history_turns * 2) :]
        history_source = "client"
    else:
        history = await get_conversation_history(conversation_key)
        history_source = "proxy" if history else "none"

    agent_input: list[dict[str, str]] = [
        *history,
        {
            "role": "user",
            "content": (
                f"{informational_runtime_context(
                    settings.api.timezone,
                    settings.api.locale,
                    settings.api.location,
                    origin_context,
                    home_assistant_config=(
                        getattr(runtime.ha_ws, "config", None)
                        if runtime.ha_ws is not None
                        else None
                    ),
                    timezone_explicit=settings.api.timezone_explicit,
                    locale_explicit=settings.api.locale_explicit,
                    location_explicit=settings.api.location_explicit,
                )}\n\n"
                f"Latest user request: {text.strip()}"
            ),
        },
    ]
    log.info(
        "INFORMATIONAL LLM START text=%r history_source=%s history_messages=%d",
        text,
        history_source,
        len(history),
    )

    async with runtime.fallback_lock:
        result = await asyncio.wait_for(
            Runner.run(
                runtime.informational_agent,
                agent_input,
                max_turns=settings.llm.max_turns,
            ),
            timeout=settings.llm.timeout_seconds,
        )

    final_output = result.final_output
    response = (
        None
        if final_output is None
        else sanitise_spoken_response(str(final_output).strip())
    )
    if not response:
        raise RuntimeError("Informational LLM agent returned no response")
    log.info(
        "INFORMATIONAL LLM COMPLETE text=%r response=%r calls=%d",
        text,
        response,
        len(getattr(result, "raw_responses", []) or []),
    )
    return response


async def _recover_spoken_answer_from_successful_data(
    user_text: str,
    successful_data: dict[str, Any] | None,
) -> str | None:
    if not settings.llm.data_recovery_enabled or not isinstance(successful_data, dict):
        return None

    raw_payload = json.dumps(
        successful_data,
        ensure_ascii=False,
        default=str,
    )
    if len(raw_payload) > settings.llm.data_recovery_max_chars:
        raw_payload = raw_payload[: settings.llm.data_recovery_max_chars] + "...[truncated]"

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
            base_url=settings.llm.base_url,
            api_key=settings.llm.api_key,
            timeout=settings.llm.timeout_seconds,
        )
        completion = await client.chat.completions.create(
            model=settings.llm.model,
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
    return origin_runtime_context(origin_context)


def validate_fallback_config() -> list[str]:
    missing: list[str] = []
    if not settings.llm.model:
        missing.append("INTENT_BRIDGE_LLM_MODEL")
    if not settings.home_assistant.base_url:
        missing.append("INTENT_BRIDGE_HA_BASE_URL")
    if not settings.home_assistant.access_token:
        missing.append("INTENT_BRIDGE_HA_ACCESS_TOKEN")
    return missing


def validate_music_assistant_config() -> list[str]:
    if not settings.music_assistant.enabled:
        return []
    missing: list[str] = []
    if not settings.music_assistant.base_url:
        missing.append("INTENT_BRIDGE_MA_BASE_URL")
    if not settings.music_assistant.access_token:
        missing.append("INTENT_BRIDGE_MA_ACCESS_TOKEN")
    return missing
