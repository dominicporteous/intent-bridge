from __future__ import annotations

from dataclasses import dataclass

import pytest

from intent_bridge.core.voice import RouteDeclined
from intent_bridge.intent_engine.models import CatalogSnapshot, IntentPlan
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
