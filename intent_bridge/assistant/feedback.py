"""Adapter-facing orchestration for asynchronous assistant feedback channels."""

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
    ) -> None:
        self.leds = leds
        self.sounds = sounds

    async def begin(
        self,
        origin_context: dict[str, Any] | None,
        *,
        led: bool,
        sounds: bool,
    ) -> AssistantFeedbackHandle:
        led_handle = await self.leds.begin(origin_context) if led else None
        sound_target = await self.sounds.resolve(origin_context) if sounds else None
        if sounds:
            await self.sounds.play(sound_target, "processing")
        return AssistantFeedbackHandle(
            led_handle=led_handle,
            sound_target=sound_target,
            sounds_requested=sounds,
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
            if handle.sounds_requested and play_terminal_sound:
                await self.sounds.play(
                    handle.sound_target,
                    "success" if success else "error",
                )
        finally:
            await self.leds.end(handle.led_handle)

    async def stop_all(self) -> None:
        await self.leds.stop_all()


assistant_feedback = AssistantFeedback()


__all__ = [
    "AssistantFeedback",
    "AssistantFeedbackHandle",
    "LedFeedback",
    "SoundFeedback",
    "assistant_feedback",
]
