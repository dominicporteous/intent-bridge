"""Composable deterministic-planner policies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from intent_bridge.core.voice import RouteDeclined
from intent_bridge.intent_engine.models import CatalogSnapshot, IntentPlan
from intent_bridge.intent_engine.ports import IntentPlanner


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
        declines: list[RouteDeclined] = []
        for planner in self._planners:
            try:
                result = planner.plan(text, catalog, origin_context)
            except RouteDeclined as exc:
                declines.append(exc)
                continue
            if result.steps or result.response is not None:
                return result
        detail = "; ".join(str(exc) for exc in declines if str(exc))
        raise RouteDeclined(detail or "No deterministic planner matched the request")


__all__ = ["IntentPlannerChain"]
