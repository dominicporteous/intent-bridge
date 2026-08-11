from __future__ import annotations

from dataclasses import dataclass

import pytest

from intent_bridge.core.voice import RouteDeclined
from intent_bridge.intent_engine.models import CatalogSnapshot, IntentPlan
from intent_bridge.intent_engine.outcomes import (
    AmbiguousTarget,
    CapabilityMismatch,
    IncompleteCompound,
    NoTarget,
    Resolved,
    UnsupportedOperation,
)
from intent_bridge.intent_engine.planning import IntentPlannerChain


@dataclass
class _Planner:
    result: IntentPlan | Exception
    calls: int = 0

    def plan(self, text, catalog, origin_context=None):
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def test_chain_continues_only_after_an_explicit_decline():
    first = _Planner(RouteDeclined("not mine"))
    second = _Planner(IntentPlan(response="Which one?"))
    third = _Planner(IntentPlan(response="must not run"))

    result = IntentPlannerChain((first, second, third)).plan("words", CatalogSnapshot())

    assert result.response == "Which one?"
    assert (first.calls, second.calls, third.calls) == (1, 1, 0)


def test_chain_rejects_empty_configuration_and_reports_total_decline():
    with pytest.raises(ValueError, match="At least one"):
        IntentPlannerChain(())

    chain = IntentPlannerChain(
        (
            _Planner(RouteDeclined("first reason")),
            _Planner(IntentPlan()),
        )
    )
    with pytest.raises(RouteDeclined, match="first reason"):
        chain.plan("words", CatalogSnapshot())


@dataclass
class _TypedPlanner:
    outcome: object
    calls: int = 0

    def resolve(self, text, catalog, origin_context=None):
        self.calls += 1
        return self.outcome


@pytest.mark.parametrize(
    "failure",
    [
        UnsupportedOperation("unknown predicate"),
        IncompleteCompound(("set the mystery",)),
        CapabilityMismatch("brightness", ("media_player.tv",)),
        NoTarget("light"),
    ],
)
def test_typed_non_ambiguity_failures_continue_to_next_planner(failure):
    resolved = Resolved(IntentPlan(response="handled by next strategy"))
    first = _TypedPlanner(failure)
    second = _TypedPlanner(resolved)

    outcome = IntentPlannerChain((first, second)).resolve("words", CatalogSnapshot())

    assert outcome == resolved
    assert (first.calls, second.calls) == (1, 1)


def test_typed_ambiguity_is_the_only_failure_that_stops_the_chain():
    ambiguous = AmbiguousTarget(("light.one", "light.two"), "device name")
    first = _TypedPlanner(ambiguous)
    second = _TypedPlanner(Resolved(IntentPlan(response="must not run")))

    outcome = IntentPlannerChain((first, second)).resolve("words", CatalogSnapshot())

    assert outcome == ambiguous
    assert second.calls == 0
