"""Core voice-to-action orchestration.

This module contains no FastAPI, MQTT, Home Assistant, Music Assistant, or LLM
SDK dependencies. Integrations implement ``VoiceRoute`` and are composed in an
ordered pipeline by the application boundary.
"""

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True, slots=True)
class VoiceRequest:
    """Transport-neutral request passed through the action pipeline."""

    text: str
    conversation_key: str
    client_history: tuple[dict[str, str], ...] = ()
    origin_context: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class RouteFailure:
    route: str
    error: Exception


@dataclass(frozen=True, slots=True)
class VoiceResult:
    speech: str
    route: str
    failures: tuple[RouteFailure, ...] = ()


class VoiceRoute(Protocol):
    """One independently replaceable way to satisfy a voice request."""

    name: str

    async def handle(self, request: VoiceRequest) -> str: ...


class VoiceRequestHandler(Protocol):
    """Application-facing port implemented by an ordered voice pipeline."""

    async def handle(self, request: VoiceRequest) -> VoiceResult: ...


class VoicePipelineError(RuntimeError):
    def __init__(self, failures: Sequence[RouteFailure]) -> None:
        self.failures = tuple(failures)
        detail = "; ".join(f"{item.route}: {item.error}" for item in failures)
        super().__init__(detail or "No voice routes are configured")


class RouteDeclined(RuntimeError):
    """A route made no external changes and explicitly permits fallback."""


class RouteExecutionError(RuntimeError):
    """A route failed after execution may have begun and must not fall through."""


@dataclass(frozen=True, slots=True)
class FunctionVoiceRoute:
    """Adapter for an async function, useful at integration boundaries."""

    name: str
    handler: Callable[[VoiceRequest], Awaitable[str]] = field(repr=False)

    async def handle(self, request: VoiceRequest) -> str:
        response = str(await self.handler(request)).strip()
        if not response:
            raise RuntimeError("route returned an empty response")
        return response


class VoiceActionPipeline:
    """Try routes in priority order until one produces spoken output."""

    def __init__(self, routes: Sequence[VoiceRoute]) -> None:
        self._routes = tuple(routes)

    @property
    def route_names(self) -> tuple[str, ...]:
        return tuple(route.name for route in self._routes)

    async def handle(self, request: VoiceRequest) -> VoiceResult:
        failures: list[RouteFailure] = []
        for route in self._routes:
            try:
                speech = await route.handle(request)
            except RouteExecutionError as exc:
                failures.append(RouteFailure(route.name, exc))
                raise VoicePipelineError(failures) from exc
            except Exception as exc:
                failures.append(RouteFailure(route.name, exc))
                continue
            return VoiceResult(speech=speech, route=route.name, failures=tuple(failures))
        raise VoicePipelineError(failures)


__all__ = [
    "FunctionVoiceRoute",
    "RouteDeclined",
    "RouteExecutionError",
    "RouteFailure",
    "VoiceActionPipeline",
    "VoicePipelineError",
    "VoiceRequest",
    "VoiceRequestHandler",
    "VoiceResult",
    "VoiceRoute",
]
