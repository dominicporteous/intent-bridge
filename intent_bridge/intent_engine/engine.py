"""Application service for deterministic recognition, resolution, and execution."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from intent_bridge.core.voice import RouteDeclined, RouteExecutionError, VoiceRequest
from intent_bridge.intent_engine.models import (
    CatalogSnapshot,
    IntentMatch,
    IntentPlan,
    OhfIntentCall,
    PlannedIntent,
)
from intent_bridge.intent_engine.ports import (
    CatalogProvider,
    IntentExecutor,
    IntentPlanner,
    IntentRecognizer,
)
from intent_bridge.intent_engine.resolution import ResolvedCandidate, resolve_candidate
from intent_bridge.runtime.dependencies import runtime

_CONTEXT_RELATIVE_INTENTS = frozenset(
    {
        "HassTurnOn",
        "HassTurnOff",
        "HassGetState",
        "HassLightSet",
        "HassClimateSetTemperature",
        "HassClimateGetTemperature",
        "HassSetPosition",
        "HassFanSetSpeed",
    }
)
_EXPLICIT_TARGET_SLOTS = frozenset({"name", "area", "floor"})
_ENTITY_TARGETING_INTENTS = frozenset(
    {
        "HassClimateGetTemperature",
        "HassClimateSetTemperature",
        "HassFanSetSpeed",
        "HassGetState",
        "HassLawnMowerDock",
        "HassLawnMowerStartMowing",
        "HassLightSet",
        "HassMediaNext",
        "HassMediaPause",
        "HassMediaPlayerMute",
        "HassMediaPlayerUnmute",
        "HassMediaPrevious",
        "HassMediaSearchAndPlay",
        "HassMediaUnpause",
        "HassSetPosition",
        "HassSetVolume",
        "HassSetVolumeRelative",
        "HassTurnOff",
        "HassTurnOn",
        "HassVacuumCleanArea",
        "HassVacuumReturnToBase",
        "HassVacuumStart",
    }
)


def _origin_area(origin_context: Mapping[str, object] | None) -> str | None:
    if not origin_context:
        return None
    for key in ("area_name", "area_id"):
        value = origin_context.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _requires_origin_area(match: IntentMatch) -> bool:
    return bool(
        match.intent_name in _CONTEXT_RELATIVE_INTENTS
        and "domain" in match.slots
        and not (_EXPLICIT_TARGET_SLOTS & match.slots.keys())
    )


def _call_for_match(
    match: IntentMatch,
    origin_context: Mapping[str, object] | None,
) -> OhfIntentCall:
    data: dict[str, Any] = {name: slot.value for name, slot in match.slots.items()}
    if _requires_origin_area(match):
        area = _origin_area(origin_context)
        if area:
            data["area"] = area
    return OhfIntentCall(intent_name=match.intent_name, data=data)


class DeterministicIntentEngine:
    """Coordinate pure recognition/resolution before invoking one side-effect boundary."""

    def __init__(
        self,
        recognizer: IntentRecognizer,
        catalog_provider: CatalogProvider,
        executor: IntentExecutor,
        *,
        preferred_planner: IntentPlanner | None = None,
        fallback_planner: IntentPlanner | None = None,
        default_response: str = "Done.",
        ambiguity_response: str = (
            "I found more than one possible target. Please be more specific."
        ),
    ) -> None:
        self._recognizer = recognizer
        self._catalog_provider = catalog_provider
        self._executor = executor
        self._preferred_planner = preferred_planner
        self._fallback_planner = fallback_planner
        self._default_response = default_response
        self._ambiguity_response = ambiguity_response

    def plan(self, request: VoiceRequest) -> IntentPlan:
        """Recognize and resolve a request without executing any operations."""

        catalog = self._catalog_provider.snapshot()
        preferred = self._try_planner(self._preferred_planner, request, catalog)
        if preferred is not None:
            return preferred

        matches = self._recognizer.recognize(
            request.text,
            catalog,
            request.origin_context,
        )
        if not matches:
            if fallback := self._try_planner(self._fallback_planner, request, catalog):
                return fallback
            raise RouteDeclined("No deterministic intent matched the request")

        origin_area = _origin_area(request.origin_context)
        if any(_requires_origin_area(candidate) for candidate in matches) and not origin_area:
            return IntentPlan(response="Which area should I use?")

        resolved = [
            resolve_candidate(candidate, catalog, request.origin_context) for candidate in matches
        ]
        resolved = [
            candidate
            for candidate in resolved
            if candidate.match.intent_name not in _ENTITY_TARGETING_INTENTS or candidate.entity_ids
        ]
        if not resolved:
            if fallback := self._try_planner(self._fallback_planner, request, catalog):
                return fallback
            raise RouteDeclined("No deterministic intent resolved to a catalog target")

        by_semantics: dict[tuple[Any, ...], list[ResolvedCandidate]] = {}
        for candidate in resolved:
            by_semantics.setdefault(candidate.semantic_key, []).append(candidate)

        if len(by_semantics) != 1:
            if fallback := self._try_planner(self._fallback_planner, request, catalog):
                return fallback
            return IntentPlan(response=self._ambiguity_response)

        equivalent_candidates = next(iter(by_semantics.values()))
        selected = max(equivalent_candidates, key=lambda candidate: candidate.specificity)
        call = _call_for_match(selected.match, request.origin_context)
        return IntentPlan(
            steps=(
                PlannedIntent(
                    call=call,
                    entity_ids=tuple(sorted(selected.entity_ids)),
                ),
            )
        )

    @staticmethod
    def _try_planner(
        planner: IntentPlanner | None,
        request: VoiceRequest,
        catalog: CatalogSnapshot,
    ) -> IntentPlan | None:
        if planner is None:
            return None
        try:
            plan = planner.plan(request.text, catalog, request.origin_context)
        except RouteDeclined:
            return None
        return plan if plan.steps or plan.response is not None else None

    async def handle(self, request: VoiceRequest) -> str:
        """Plan a request, then execute each planned operation in order."""

        plan = self.plan(request)
        if plan.response is not None:
            return plan.response

        responses: list[str] = []
        for step in plan.steps:
            try:
                # Short-circuit simple state queries using the HA WebSocket cache
                if step.call.intent_name in ("HassGetState", "HassClimateGetTemperature"):
                    try:
                        # Only short-circuit for explicit measurement queries (temperature, humidity, pressure).
                        # Avoid short-circuiting broad "weather" or forecast requests which require richer
                        # attribute inspection or natural-language summarisation.
                        qtext = (request.text or "").lower()
                        if any(keyword in qtext for keyword in ("weather", "forecast", "today", "tomorrow", "conditions")) and "temperature" not in qtext and "temp" not in qtext:
                            result = await self._executor.execute(step.call)
                        else:
                            ha_ws = runtime.ha_ws
                            if ha_ws is not None and ha_ws.ready.is_set() and step.entity_ids:
                                # Prefer the first resolved entity target
                                entity_id = step.entity_ids[0]
                                state = ha_ws.states.get(entity_id)
                                if isinstance(state, dict):
                                    attributes = state.get("attributes") or {}
                                    unit = attributes.get("unit_of_measurement") or attributes.get("temperature_unit") or ""
                                    friendly = attributes.get("friendly_name") or entity_id
                                    value = state.get("state")
                                    domain = entity_id.split(".", 1)[0] if "." in entity_id else ""
                                    # If the entity is a simple measurement, return concise value. Otherwise defer.
                                    if domain in ("sensor",) or "temperature" in (attributes.get("device_class") or ""):
                                        speech = f"{value}{(' ' + unit) if unit else ''}"
                                        result = type("R", (), {"speech": speech, "response": state})()
                                    else:
                                        result = await self._executor.execute(step.call)
                                else:
                                    result = await self._executor.execute(step.call)
                            else:
                                result = await self._executor.execute(step.call)
                    except Exception:
                        result = await self._executor.execute(step.call)
                else:
                    result = await self._executor.execute(step.call)
            except RouteDeclined:
                raise
            except Exception as exc:
                raise RouteExecutionError(f"Deterministic intent execution failed: {exc}") from exc
            if speech := result.speech.strip():
                responses.append(speech)

        return " ".join(responses) or self._default_response


__all__ = ["DeterministicIntentEngine"]
