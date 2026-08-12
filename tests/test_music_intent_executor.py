from unittest.mock import AsyncMock

import pytest

from intent_bridge.intent_engine.models import ExecutionResult, OhfIntentCall
from intent_bridge.music_assistant.intent_executor import MusicAssistantIntentExecutor


@pytest.mark.asyncio
async def test_media_search_executes_directly_through_native_music_assistant():
    fallback = AsyncMock()
    native_play = AsyncMock(
        return_value={
            "success": True,
            "message": "Starting Lana Del Rey radio.",
            "player_id": "office",
        }
    )
    executor = MusicAssistantIntentExecutor(
        fallback,
        native_playback_available=lambda: True,
        play_query=native_play,
    )

    result = await executor.execute(
        OhfIntentCall(
            "HassMediaSearchAndPlay",
            {"search_query": " Lana Del Rey ", "area": " Office "},
        )
    )

    assert result.speech == "Starting Lana Del Rey radio."
    assert result.response["player_id"] == "office"
    native_play.assert_awaited_once_with(
        query="Lana Del Rey",
        area="Office",
        player_id=None,
        radio_mode=True,
    )
    fallback.execute.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "call",
    [
        OhfIntentCall("HassTurnOn", {"name": "Office"}),
        OhfIntentCall("HassMediaPause", {"area": "Office"}),
        OhfIntentCall("HassMediaSearchAndPlay", {"search_query": "Song"}),
    ],
)
async def test_non_search_or_unavailable_native_playback_delegates(call):
    delegated = ExecutionResult(speech="Handled by Home Assistant.")
    fallback = AsyncMock()
    fallback.execute.return_value = delegated
    native_play = AsyncMock()
    executor = MusicAssistantIntentExecutor(
        fallback,
        native_playback_available=lambda: False,
        play_query=native_play,
    )

    assert await executor.execute(call) is delegated
    fallback.execute.assert_awaited_once_with(call)
    native_play.assert_not_awaited()


@pytest.mark.asyncio
async def test_native_playback_failure_does_not_delegate_and_risk_duplicate_playback():
    fallback = AsyncMock()
    native_play = AsyncMock(return_value={"success": False, "error": "MA is offline"})
    executor = MusicAssistantIntentExecutor(
        fallback,
        native_playback_available=lambda: True,
        play_query=native_play,
    )

    with pytest.raises(RuntimeError, match="MA is offline"):
        await executor.execute(
            OhfIntentCall("HassMediaSearchAndPlay", {"search_query": "Foo Fighters"})
        )

    fallback.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_native_media_search_requires_a_query_before_dispatch():
    fallback = AsyncMock()
    native_play = AsyncMock()
    executor = MusicAssistantIntentExecutor(
        fallback,
        native_playback_available=lambda: True,
        play_query=native_play,
    )

    with pytest.raises(ValueError, match="requires search_query"):
        await executor.execute(OhfIntentCall("HassMediaSearchAndPlay", {"area": "Office"}))

    native_play.assert_not_awaited()
    fallback.execute.assert_not_awaited()
