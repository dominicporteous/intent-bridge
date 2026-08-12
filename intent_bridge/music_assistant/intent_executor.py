"""Deterministic Music Assistant execution for recognized OHF media intents."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any, Protocol

from intent_bridge.config import settings
from intent_bridge.intent_engine.models import ExecutionResult, OhfIntentCall
from intent_bridge.intent_engine.ports import IntentExecutor
from intent_bridge.music_assistant.tools import play_query_native

LOGGER = logging.getLogger(__name__)


class NativePlayQuery(Protocol):
    async def __call__(
        self,
        query: str,
        area: str | None = None,
        player_id: str | None = None,
        radio_mode: bool = True,
        origin_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...


class MusicAssistantIntentExecutor:
    """Prefer native MA for deterministic media search and delegate other intents."""

    def __init__(
        self,
        fallback: IntentExecutor,
        *,
        native_playback_available: Callable[[], bool],
        play_query: NativePlayQuery = play_query_native,
    ) -> None:
        self._fallback = fallback
        self._native_playback_available = native_playback_available
        self._play_query = play_query

    async def execute(self, call: OhfIntentCall) -> ExecutionResult:
        if (
            call.intent_name != "HassMediaSearchAndPlay"
            or not self._native_playback_available()
        ):
            return await self._fallback.execute(call)

        query = _optional_text(call.data.get("search_query"))
        if not query:
            raise ValueError("HassMediaSearchAndPlay requires search_query")
        area = _optional_text(call.data.get("area"))

        LOGGER.info(
            "Executing deterministic Music Assistant intent intent=%s query=%r area=%r",
            call.intent_name,
            query,
            area,
        )
        payload = await self._play_query(
            query=query,
            area=area,
            player_id=None,
            radio_mode=True,
        )
        if not isinstance(payload, Mapping) or payload.get("success") is not True:
            detail = payload.get("error") if isinstance(payload, Mapping) else None
            raise RuntimeError(
                f"Native Music Assistant playback failed: {detail or 'invalid response'}"
            )

        speech = _optional_text(payload.get("message")) or settings.api.action_confirmation
        return ExecutionResult(speech=speech, response=dict(payload))


def _optional_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


__all__ = ["MusicAssistantIntentExecutor"]
