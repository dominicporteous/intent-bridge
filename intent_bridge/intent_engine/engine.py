"""Application service for deterministic recognition, resolution, and execution."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Mapping
from typing import Any

from intent_bridge.core.voice import RouteDeclined, RouteExecutionError, VoiceRequest
from intent_bridge.intent_engine.models import (
    CatalogSnapshot,
    IntentMatch,
    IntentPlan,
    OhfIntentCall,
    PlannedIntent,
    semantic_effect_for_call,
)
from intent_bridge.intent_engine.natural_language import (
    normalize_lexical_operator,
    singular_generic_target,
    split_compound_request,
)
from intent_bridge.intent_engine.outcomes import AmbiguousTarget, Resolved
from intent_bridge.intent_engine.ports import (
    CatalogProvider,
    IntentExecutor,
    IntentPlanner,
    IntentRecognizer,
)
from intent_bridge.intent_engine.resolution import ResolvedCandidate, resolve_candidate
from intent_bridge.runtime.dependencies import runtime
from intent_bridge.runtime.execution import _reset_voice_tool_run_state

LOGGER = logging.getLogger(__name__)

_AUTOMATION_REQUEST_RE = re.compile(r"\b(?:automation|automate|routine|rule)\b", re.IGNORECASE)

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


def _reading_speech(step: PlannedIntent) -> str:
    reading = step.reading
    if reading is None:
        return ""
    measurement = reading.measurement
    unit = (measurement.unit or "").strip()
    separator = "" if unit in {"%", "°"} else " "
    rendered_value = f"{measurement.value}{separator}{unit}" if unit else measurement.value
    return f"{reading.entity_name} is {rendered_value}"


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


def _exact_call_for_entity(match: IntentMatch, entity_id: str, name: str) -> OhfIntentCall:
    """Materialize a topology-filtered generic target as an exact HA target."""

    data = {
        slot_name: slot.value
        for slot_name, slot in match.slots.items()
        if slot_name not in _EXPLICIT_TARGET_SLOTS and slot_name != "domain"
    }
    data.update({"name": name, "entity_id": entity_id})
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
        step_observer: Callable[[PlannedIntent], None] | None = None,
        default_response: str = "",
        ambiguity_response: str = (
            "I found more than one possible target. Please be more specific."
        ),
    ) -> None:
        self._recognizer = recognizer
        self._catalog_provider = catalog_provider
        self._executor = executor
        self._preferred_planner = preferred_planner
        self._fallback_planner = fallback_planner
        self._step_observer = step_observer
        self._default_response = default_response
        self._ambiguity_response = ambiguity_response

    def plan(self, request: VoiceRequest) -> IntentPlan:
        """Recognize and resolve a request without executing any operations."""

        catalog = self._catalog_provider.snapshot()
        structural_automation = bool(_AUTOMATION_REQUEST_RE.search(request.text))
        clauses = (request.text,) if structural_automation else split_compound_request(request.text)
        LOGGER.info(
            "INTENT PLAN start text=%r conversation=%r clauses=%r catalog_entities=%d "
            "catalog_areas=%d catalog_floors=%d origin=%r",
            request.text,
            request.conversation_key,
            clauses,
            len(catalog.entities),
            len(catalog.areas),
            len(catalog.floors),
            request.origin_context,
        )
        if len(clauses) > 1:
            steps: list[PlannedIntent] = []
            context = dict(request.origin_context or {})
            for clause in clauses:
                clause_request = VoiceRequest(
                    text=clause,
                    conversation_key=request.conversation_key,
                    client_history=request.client_history,
                    origin_context=context,
                )
                clause_plan = self._plan_single(clause_request, catalog)
                # A compound is one transaction: never execute the clauses that
                # happened to resolve when another clause still needs clarification.
                if clause_plan.response is not None:
                    return IntentPlan(response=clause_plan.response)
                if not clause_plan.steps:
                    raise RouteDeclined(f"No deterministic intent matched clause: {clause}")
                steps.extend(clause_plan.steps)
                context["last_entity_ids"] = tuple(
                    dict.fromkeys(
                        entity_id
                        for step in clause_plan.steps
                        for entity_id in step.entity_ids
                    )
                )
            return IntentPlan(steps=tuple(steps))

        return self._plan_single(request, catalog)

    def catalog_snapshot(self) -> CatalogSnapshot:
        """Return the same production catalog used for request planning."""

        return self._catalog_provider.snapshot()

    def _plan_single(
        self,
        request: VoiceRequest,
        catalog: CatalogSnapshot,
    ) -> IntentPlan:
        """Plan one structural clause using the configured recognizer chain."""

        # Property-bearing language must retain its requested effect before a
        # broad power grammar can consume words such as ``on`` or ``off``.
        # This is capability-level precedence, independent of device type.
        normalized_text = request.text.casefold()
        structural_automation = bool(_AUTOMATION_REQUEST_RE.search(request.text))
        lexical_operator = normalize_lexical_operator(request.text)
        property_semantics = bool(
            re.search(r"\b(?:audio|mute|sound)\b", normalized_text)
            or (
                re.search(
                    r"\b(?:brightness|bright|color|colour|position|speed|"
                    r"temperature|temp|volume)\b",
                    normalized_text,
                )
                and re.search(
                    r"\b(?:adjust|change|dim|increase|lower|make|raise|restore|set|turn)\b",
                    normalized_text,
                )
            )
        )
        if not structural_automation and (lexical_operator is not None or property_semantics) and (
            semantic := self._try_planner(self._fallback_planner, request, catalog)
        ):
            LOGGER.info(
                "INTENT PLAN selected semantic-first planner text=%r operator=%r "
                "property_semantics=%s",
                request.text,
                lexical_operator,
                property_semantics,
            )
            return self._validate_target_cardinality(request.text, semantic, catalog)

        preferred = self._try_planner(self._preferred_planner, request, catalog)
        if preferred is not None:
            return self._validate_target_cardinality(request.text, preferred, catalog)

        matches = self._recognizer.recognize(
            request.text,
            catalog,
            request.origin_context,
        )
        LOGGER.info(
            "INTENT PLAN recognizer text=%r candidates=%s",
            request.text,
            tuple(
                {
                    "intent": match.intent_name,
                    "slots": {
                        name: {"value": slot.value, "metadata": dict(slot.metadata)}
                        for name, slot in match.slots.items()
                    },
                    "response": match.response_key,
                }
                for match in matches
            ),
        )
        if not matches:
            if fallback := self._try_planner(self._fallback_planner, request, catalog):
                return self._validate_target_cardinality(request.text, fallback, catalog)
            raise RouteDeclined("No deterministic intent matched the request")

        origin_area = _origin_area(request.origin_context)
        if any(_requires_origin_area(candidate) for candidate in matches) and not origin_area:
            return IntentPlan(response="Which area should I use?")

        resolved = [
            resolve_candidate(candidate, catalog, request.origin_context) for candidate in matches
        ]
        LOGGER.info(
            "INTENT PLAN resolved_candidates text=%r candidates=%s",
            request.text,
            tuple(
                {
                    "intent": candidate.match.intent_name,
                    "entity_ids": tuple(sorted(candidate.entity_ids)),
                    "semantic_key": candidate.semantic_key,
                    "specificity": candidate.specificity,
                }
                for candidate in resolved
            ),
        )
        resolved = [
            candidate
            for candidate in resolved
            if candidate.match.intent_name not in _ENTITY_TARGETING_INTENTS or candidate.entity_ids
        ]
        if not resolved:
            if fallback := self._try_planner(self._fallback_planner, request, catalog):
                return self._validate_target_cardinality(request.text, fallback, catalog)
            raise RouteDeclined("No deterministic intent resolved to a catalog target")

        by_semantics: dict[tuple[Any, ...], list[ResolvedCandidate]] = {}
        for candidate in resolved:
            by_semantics.setdefault(candidate.semantic_key, []).append(candidate)

        if len(by_semantics) != 1:
            if fallback := self._try_planner(self._fallback_planner, request, catalog):
                return self._validate_target_cardinality(request.text, fallback, catalog)
            LOGGER.info(
                "INTENT PLAN ambiguous reason=distinct_semantics text=%r alternatives=%s",
                request.text,
                tuple(by_semantics),
            )
            return IntentPlan(response=self._ambiguity_response)

        equivalent_candidates = next(iter(by_semantics.values()))
        selected = max(equivalent_candidates, key=lambda candidate: candidate.specificity)
        LOGGER.info(
            "INTENT PLAN selected recognizer_candidate text=%r intent=%s entity_ids=%s "
            "specificity=%d",
            request.text,
            selected.match.intent_name,
            tuple(sorted(selected.entity_ids)),
            selected.specificity,
        )
        selected_ids = tuple(sorted(selected.entity_ids))
        by_id = {entity.entity_id: entity for entity in catalog.entities}
        selected_domains = {
            by_id[entity_id].domain for entity_id in selected_ids if entity_id in by_id
        }
        # HA's area/domain intent would independently expand the target again
        # and re-include status LEDs which the catalog deliberately filtered.
        # Materialize the safe set as exact entity calls whenever that can
        # happen. Explicitly named indicator requests remain available.
        materialize_exact = bool(
            "name" not in selected.match.slots
            and selected_domains == {"light"}
            and any(entity.domain == "light" and entity.is_indicator for entity in catalog.entities)
        )
        if materialize_exact:
            steps = tuple(
                PlannedIntent(
                    call=(
                        call := _exact_call_for_entity(
                            selected.match,
                            entity_id,
                            by_id[entity_id].name,
                        )
                    ),
                    entity_ids=(entity_id,),
                    effect=semantic_effect_for_call(call),
                )
                for entity_id in selected_ids
                if entity_id in by_id
            )
            LOGGER.info(
                "INTENT PLAN materialized filtered light group text=%r entity_ids=%s",
                request.text,
                selected_ids,
            )
            return self._validate_target_cardinality(
                request.text,
                IntentPlan(steps=steps),
                catalog,
            )

        call = _call_for_match(selected.match, request.origin_context)
        return self._validate_target_cardinality(
            request.text,
            IntentPlan(
                steps=(
                    PlannedIntent(
                        call=call,
                        entity_ids=selected_ids,
                        effect=semantic_effect_for_call(call),
                    ),
                )
            ),
            catalog,
        )

    def _validate_target_cardinality(
        self,
        text: str,
        plan: IntentPlan,
        catalog: CatalogSnapshot,
    ) -> IntentPlan:
        """Prevent a singular generic target from silently becoming a group."""

        if plan.response is not None or len(plan.steps) != 1:
            return plan
        step = plan.steps[0]
        if len(step.entity_ids) < 2:
            return plan
        by_id = {entity.entity_id: entity for entity in catalog.entities}
        domains = {
            by_id[entity_id].domain for entity_id in step.entity_ids if entity_id in by_id
        }
        if len(domains) == 1 and singular_generic_target(text, next(iter(domains))):
            LOGGER.info(
                "INTENT PLAN ambiguous reason=singular_generic_cardinality text=%r "
                "domain=%s entity_ids=%s",
                text,
                next(iter(domains)),
                step.entity_ids,
            )
            return IntentPlan(response=self._ambiguity_response)
        return plan

    def _try_planner(
        self,
        planner: IntentPlanner | None,
        request: VoiceRequest,
        catalog: CatalogSnapshot,
    ) -> IntentPlan | None:
        if planner is None:
            return None
        planner_name = type(planner).__name__
        resolver = getattr(planner, "resolve", None)
        if callable(resolver):
            outcome = resolver(request.text, catalog, request.origin_context)
            LOGGER.info(
                "INTENT PLAN planner_outcome planner=%s text=%r outcome=%s detail=%r",
                planner_name,
                request.text,
                type(outcome).__name__,
                outcome,
            )
            if isinstance(outcome, Resolved):
                return outcome.plan
            if isinstance(outcome, AmbiguousTarget):
                return IntentPlan(
                    response=self._ambiguity_response
                )
            # Unsupported/incomplete/capability/no-target outcomes are declines
            # here, allowing the next recognizer or the external voice fallback.
            return None
        try:
            plan = planner.plan(request.text, catalog, request.origin_context)
        except RouteDeclined as exc:
            LOGGER.info(
                "INTENT PLAN planner_declined planner=%s text=%r reason=%s",
                planner_name,
                request.text,
                exc,
            )
            return None
        LOGGER.info(
            "INTENT PLAN planner_result planner=%s text=%r steps=%d response=%r",
            planner_name,
            request.text,
            len(plan.steps),
            plan.response,
        )
        return plan if plan.steps or plan.response is not None else None

    async def handle(self, request: VoiceRequest) -> str:
        """Plan a request, then execute each planned operation in order."""

        return await self.execute_plan(request, self.plan(request))

    async def execute_plan(self, request: VoiceRequest, plan: IntentPlan) -> str:
        """Execute a previously produced plan through the production boundary."""

        _reset_voice_tool_run_state(request.text, dict(request.origin_context or {}))
        if plan.response is not None:
            return plan.response

        responses: list[str] = []
        for step in plan.steps:
            if speech := _reading_speech(step):
                reading = step.reading
                assert reading is not None
                LOGGER.info(
                    "MEASUREMENT CACHE HIT entity=%s quantity=%s source=%s value=%r "
                    "unit=%r speech=%r",
                    reading.entity_id,
                    reading.measurement.quantity,
                    reading.measurement.source,
                    reading.measurement.value,
                    reading.measurement.unit,
                    speech,
                )
                if self._step_observer is not None:
                    self._step_observer(step)
                responses.append(speech)
                continue
            try:
                # Short-circuit simple state queries using the HA WebSocket cache
                if step.call.intent_name in ("HassGetState", "HassClimateGetTemperature"):
                    try:
                        # Only short-circuit for explicit measurement queries (temperature, humidity, pressure).
                        # Avoid short-circuiting broad "weather" or forecast requests which require richer
                        # attribute inspection or natural-language summarisation.
                        qtext = (request.text or "").lower()
                        if (
                            any(
                                keyword in qtext
                                for keyword in (
                                    "weather",
                                    "forecast",
                                    "today",
                                    "tomorrow",
                                    "conditions",
                                )
                            )
                            and "temperature" not in qtext
                            and "temp" not in qtext
                        ):
                            result = await self._executor.execute(step.call)
                        else:
                            ha_ws = runtime.ha_ws
                            if ha_ws is not None and ha_ws.ready.is_set() and step.entity_ids:
                                # Prefer the first resolved entity target
                                entity_id = step.entity_ids[0]
                                state = ha_ws.states.get(entity_id)
                                if isinstance(state, dict):
                                    attributes = state.get("attributes") or {}
                                    unit = (
                                        attributes.get("unit_of_measurement")
                                        or attributes.get("temperature_unit")
                                        or ""
                                    )
                                    value = state.get("state")
                                    domain = entity_id.split(".", 1)[0] if "." in entity_id else ""
                                    # If the entity is a simple measurement, return concise value. Otherwise defer.
                                    if domain in ("sensor",) or "temperature" in (
                                        attributes.get("device_class") or ""
                                    ):
                                        speech = f"{value}{(' ' + unit) if unit else ''}"
                                        result = type(
                                            "R", (), {"speech": speech, "response": state}
                                        )()
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
            if self._step_observer is not None:
                self._step_observer(step)

        return " ".join(responses) or self._default_response


__all__ = ["DeterministicIntentEngine"]
