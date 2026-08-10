"""Voice-route adapter for the deterministic intent engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from intent_bridge.core.voice import VoiceRequest


class IntentRequestHandler(Protocol):
    async def handle(self, request: VoiceRequest) -> str: ...


@dataclass(frozen=True, slots=True)
class DeterministicVoiceRoute:
    engine: IntentRequestHandler
    name: str = "ohf-hassil"

    async def handle(self, request: VoiceRequest) -> str:
        return await self.engine.handle(request)


__all__ = ["DeterministicVoiceRoute", "IntentRequestHandler"]
