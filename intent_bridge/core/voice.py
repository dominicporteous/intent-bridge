"""Core voice-to-action orchestration.

This module contains no FastAPI, MQTT, Home Assistant, Music Assistant, or LLM
SDK dependencies. Integrations implement ``VoiceRoute`` and are composed in an
ordered pipeline by the application boundary.
"""

import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

LOGGER = logging.getLogger(__name__)


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


class RouteDeclinedWithFallback(RouteDeclined):
    """Decline to later routes while retaining a safe response if they fail."""

    def __init__(
        self,
        fallback_response: str,
        *,
        on_alternative_success: Callable[[], None] | None = None,
    ) -> None:
        self.fallback_response = fallback_response
        self._on_alternative_success = on_alternative_success
        super().__init__(fallback_response)

    def alternative_succeeded(self) -> None:
        if self._on_alternative_success is not None:
            self._on_alternative_success()


class RouteExecutionError(RuntimeError):
    """A route failed after execution may have begun and must not fall through."""


@dataclass(frozen=True, slots=True)
class FunctionVoiceRoute:
    """Adapter for an async function, useful at integration boundaries."""

    name: str
    handler: Callable[[VoiceRequest], Awaitable[str]] = field(repr=False)

    async def handle(self, request: VoiceRequest) -> str:
        response = await self.handler(request)
        if response is None:
            raise RuntimeError("route returned no response")
        return str(response).strip()


class VoiceActionPipeline:
    """Try routes in priority order until one produces spoken output."""

    def __init__(
        self,
        routes: Sequence[VoiceRoute],
        *,
        failure_response: str | None = None,
        failure_route_name: str = "voice-error-response",
    ) -> None:
        self._routes = tuple(routes)
        self._failure_response = (failure_response or "").strip()
        self._failure_route_name = failure_route_name

    @property
    def route_names(self) -> tuple[str, ...]:
        return tuple(route.name for route in self._routes)

    async def handle(self, request: VoiceRequest) -> VoiceResult:
        failures: list[RouteFailure] = []
        for route in self._routes:
            LOGGER.info(
                "VOICE PIPELINE trying route=%s text=%r prior_failures=%s",
                route.name,
                request.text,
                tuple(item.route for item in failures),
            )
            try:
                speech = await route.handle(request)
            except RouteExecutionError as exc:
                LOGGER.info(
                    "VOICE PIPELINE execution_error route=%s text=%r error=%s",
                    route.name,
                    request.text,
                    exc,
                )
                failures.append(RouteFailure(route.name, exc))
                raise VoicePipelineError(failures) from exc
            except Exception as exc:
                LOGGER.info(
                    "VOICE PIPELINE declined route=%s text=%r error_type=%s error=%s",
                    route.name,
                    request.text,
                    type(exc).__name__,
                    exc,
                )
                failures.append(RouteFailure(route.name, exc))
                continue
            LOGGER.info(
                "VOICE PIPELINE selected route=%s text=%r response=%r",
                route.name,
                request.text,
                speech,
            )
            for failure in failures:
                if isinstance(failure.error, RouteDeclinedWithFallback):
                    failure.error.alternative_succeeded()
            return VoiceResult(speech=speech, route=route.name, failures=tuple(failures))
        deferred = next(
            (
                failure
                for failure in reversed(failures)
                if isinstance(failure.error, RouteDeclinedWithFallback)
            ),
            None,
        )
        if deferred is not None:
            error = deferred.error
            assert isinstance(error, RouteDeclinedWithFallback)
            LOGGER.info(
                "VOICE PIPELINE using deferred response route=%s text=%r response=%r",
                deferred.route,
                request.text,
                error.fallback_response,
            )
            return VoiceResult(
                speech=error.fallback_response,
                route=deferred.route,
                failures=tuple(failures),
            )
        if self._failure_response:
            LOGGER.warning(
                "VOICE PIPELINE using failure response route=%s text=%r failures=%s",
                self._failure_route_name,
                request.text,
                tuple(
                    (failure.route, type(failure.error).__name__, str(failure.error))
                    for failure in failures
                ),
            )
            return VoiceResult(
                speech=self._failure_response,
                route=self._failure_route_name,
                failures=tuple(failures),
            )
        raise VoicePipelineError(failures)


__all__ = [
    "FunctionVoiceRoute",
    "RouteDeclined",
    "RouteDeclinedWithFallback",
    "RouteExecutionError",
    "RouteFailure",
    "VoiceActionPipeline",
    "VoicePipelineError",
    "VoiceRequest",
    "VoiceRequestHandler",
    "VoiceResult",
    "VoiceRoute",
]
