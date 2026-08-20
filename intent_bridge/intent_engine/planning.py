"""Composable deterministic-planner policies."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

from intent_bridge.core.voice import RouteDeclined
from intent_bridge.intent_engine.models import CatalogSnapshot, IntentPlan
from intent_bridge.intent_engine.outcomes import (
    AmbiguousTarget,
    PlanningOutcome,
    Resolved,
    UnsupportedOperation,
)
from intent_bridge.intent_engine.ports import IntentPlanner

LOGGER = logging.getLogger(__name__)


class IntentPlannerChain:
    """Use the first planner that can express a request deterministically.

    Planners may decline only before producing a plan. A clarification response
    is a handled result and therefore stops the chain just like executable steps.
    """

    def __init__(self, planners: Sequence[IntentPlanner]) -> None:
        self._planners = tuple(planners)
        if not self._planners:
            raise ValueError("At least one intent planner is required")

    def plan(
        self,
        text: str,
        catalog: CatalogSnapshot,
        origin_context: Mapping[str, object] | None = None,
    ) -> IntentPlan:
        outcome = self.resolve(text, catalog, origin_context)
        if isinstance(outcome, Resolved):
            return outcome.plan
        if isinstance(outcome, AmbiguousTarget):
            return IntentPlan(
                response="I found more than one possible target. Please be more specific.",
                ambiguity_candidate_entity_ids=outcome.candidates,
                ambiguity_missing_constraint=outcome.missing_constraint,
                ambiguity_call=outcome.call,
            )
        reason = getattr(outcome, "reason", None)
        raise RouteDeclined(str(reason or outcome))

    def resolve(
        self,
        text: str,
        catalog: CatalogSnapshot,
        origin_context: Mapping[str, object] | None = None,
    ) -> PlanningOutcome:
        """Try planners while stopping only for a resolved or genuinely ambiguous result."""

        failures: list[PlanningOutcome] = []
        for planner in self._planners:
            planner_name = type(planner).__name__
            resolver = getattr(planner, "resolve", None)
            if callable(resolver):
                outcome = resolver(text, catalog, origin_context)
                LOGGER.info(
                    "PLANNER CHAIN decision planner=%s outcome=%s detail=%r",
                    planner_name,
                    type(outcome).__name__,
                    outcome,
                )
                if isinstance(outcome, (Resolved, AmbiguousTarget)):
                    return outcome
                failures.append(outcome)
                continue
            try:
                result = planner.plan(text, catalog, origin_context)
            except RouteDeclined as exc:
                LOGGER.info(
                    "PLANNER CHAIN declined planner=%s reason=%s",
                    planner_name,
                    exc,
                )
                failures.append(UnsupportedOperation(str(exc)))
                continue
            if result.steps or result.response is not None:
                LOGGER.info(
                    "PLANNER CHAIN selected planner=%s steps=%d response=%r",
                    planner_name,
                    len(result.steps),
                    result.response,
                )
                return Resolved(result)
            LOGGER.info("PLANNER CHAIN empty planner=%s", planner_name)
        detail = "; ".join(
            failure.reason
            for failure in failures
            if isinstance(failure, UnsupportedOperation) and failure.reason
        )
        return UnsupportedOperation(detail or "No deterministic planner matched the request")


__all__ = ["IntentPlannerChain"]
