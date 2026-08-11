from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from intent_bridge.assistant.feedback import AssistantFeedback


@pytest.mark.asyncio
async def test_feedback_lifecycle_can_select_both_channels():
    led_handle = object()
    sound_target = object()
    leds = SimpleNamespace(
        begin=AsyncMock(return_value=led_handle),
        end=AsyncMock(),
        stop_all=AsyncMock(),
    )
    sounds = SimpleNamespace(
        resolve=AsyncMock(return_value=sound_target),
        play=AsyncMock(return_value=True),
    )
    feedback = AssistantFeedback(leds, sounds)
    origin = {"device_id": "satellite"}

    handle = await feedback.begin(origin, led=True, sounds=True)
    await feedback.complete(handle, success=True)
    await feedback.stop_all()

    leds.begin.assert_awaited_once_with(origin)
    sounds.resolve.assert_awaited_once_with(origin)
    assert [call.args for call in sounds.play.await_args_list] == [
        (sound_target, "processing"),
        (sound_target, "success"),
    ]
    leds.end.assert_awaited_once_with(led_handle)
    leds.stop_all.assert_awaited_once()


@pytest.mark.asyncio
async def test_feedback_lifecycle_supports_independent_channels_and_errors():
    leds = SimpleNamespace(
        begin=AsyncMock(return_value=None),
        end=AsyncMock(),
        stop_all=AsyncMock(),
    )
    sounds = SimpleNamespace(
        resolve=AsyncMock(return_value=None),
        play=AsyncMock(return_value=False),
    )
    feedback = AssistantFeedback(leds, sounds)

    sound_handle = await feedback.begin(None, led=False, sounds=True)
    await feedback.complete(sound_handle, success=False)
    led_handle = await feedback.begin(None, led=True, sounds=False)
    await feedback.complete(led_handle, success=True)
    no_terminal_handle = await feedback.begin(None, led=False, sounds=True)
    await feedback.complete(
        no_terminal_handle,
        success=True,
        play_terminal_sound=False,
    )
    await feedback.complete(None, success=False)

    leds.begin.assert_awaited_once_with(None)
    assert sounds.resolve.await_count == 2
    assert [call.args[1] for call in sounds.play.await_args_list] == [
        "processing",
        "error",
        "processing",
    ]
    assert leds.end.await_count == 3
