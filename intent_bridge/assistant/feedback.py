"""Adapter-facing orchestration for asynchronous assistant feedback channels."""

import asyncio
from dataclasses import dataclass
from typing import Any, Protocol

from intent_bridge.indicators.controller import (
    AssistantLedHandle,
    assistant_leds,
)
from intent_bridge.sounds.controller import (
    AssistantSoundTarget,
    assistant_sounds,
)


@dataclass(frozen=True, slots=True)
class AssistantFeedbackHandle:
    """Resources acquired for one adapter-owned asynchronous operation."""

    led_handle: AssistantLedHandle | None = None
    sound_target: AssistantSoundTarget | None = None
    sounds_requested: bool = False
    processing_task: asyncio.Task[None] | None = None


class LedFeedback(Protocol):
    async def begin(self, origin_context: dict[str, Any] | None) -> AssistantLedHandle | None: ...

    async def end(self, handle: AssistantLedHandle | None) -> None: ...

    async def stop_all(self) -> None: ...


class SoundFeedback(Protocol):
    async def resolve(
        self, origin_context: dict[str, Any] | None
    ) -> AssistantSoundTarget | None: ...

    async def play(self, target: AssistantSoundTarget | None, sound_name: str) -> bool: ...


class AssistantFeedback:
    """Coordinate LED and sound feedback without coupling it to an adapter."""

    def __init__(
        self,
        leds: LedFeedback = assistant_leds,
        sounds: SoundFeedback = assistant_sounds,
        *,
        processing_sound_delay_seconds: float = 0.5,
    ) -> None:
        self.leds = leds
        self.sounds = sounds
        self.processing_sound_delay_seconds = processing_sound_delay_seconds
        self._processing_tasks: set[asyncio.Task[None]] = set()

    async def _play_processing_after_delay(
        self, sound_target: AssistantSoundTarget | None
    ) -> None:
        await asyncio.sleep(self.processing_sound_delay_seconds)
        await self.sounds.play(sound_target, "processing")

    def _schedule_processing_sound(
        self, sound_target: AssistantSoundTarget | None
    ) -> asyncio.Task[None] | None:
        if self.processing_sound_delay_seconds <= 0:
            return None
        task = asyncio.create_task(
            self._play_processing_after_delay(sound_target),
            name="assistant-processing-sound",
        )
        self._processing_tasks.add(task)
        task.add_done_callback(self._processing_tasks.discard)
        return task

    async def _cancel_processing_sound(self, task: asyncio.Task[None] | None) -> None:
        if task is None or task.done():
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def begin(
        self,
        origin_context: dict[str, Any] | None,
        *,
        led: bool,
        sounds: bool,
    ) -> AssistantFeedbackHandle:
        led_handle = await self.leds.begin(origin_context) if led else None
        sound_target = await self.sounds.resolve(origin_context) if sounds else None
        processing_task = None
        if sounds:
            processing_task = self._schedule_processing_sound(sound_target)
            if processing_task is None:
                await self.sounds.play(sound_target, "processing")
        return AssistantFeedbackHandle(
            led_handle=led_handle,
            sound_target=sound_target,
            sounds_requested=sounds,
            processing_task=processing_task,
        )

    async def complete(
        self,
        handle: AssistantFeedbackHandle | None,
        *,
        success: bool,
        play_terminal_sound: bool = True,
    ) -> None:
        if handle is None:
            return
        try:
            await self._cancel_processing_sound(handle.processing_task)
            if handle.sounds_requested and play_terminal_sound:
                await self.sounds.play(
                    handle.sound_target,
                    "success" if success else "error",
                )
        finally:
            await self.leds.end(handle.led_handle)

    async def stop_all(self) -> None:
        pending = tuple(self._processing_tasks)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        await self.leds.stop_all()


assistant_feedback = AssistantFeedback()


__all__ = [
    "AssistantFeedback",
    "AssistantFeedbackHandle",
    "LedFeedback",
    "SoundFeedback",
    "assistant_feedback",
]
