from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from intent_bridge.core.voice import RouteDeclined, VoiceRequest
from intent_bridge.intent_engine.engine import DeterministicIntentEngine
from intent_bridge.intent_engine.models import (
    CatalogArea,
    CatalogEntity,
    CatalogSnapshot,
    ExecutionResult,
    IntentMatch,
    IntentPlan,
    OhfIntentCall,
    PlannedIntent,
    SlotValue,
)


@dataclass
class _Recognizer:
    matches: tuple[IntentMatch, ...]

    def recognize(self, text, catalog, origin_context=None):
        return self.matches


@dataclass
class _CatalogProvider:
    catalog: CatalogSnapshot

    def snapshot(self):
        return self.catalog


@dataclass
class _RecordingExecutor:
    calls: list[OhfIntentCall] = field(default_factory=list)

    async def execute(self, call: OhfIntentCall) -> ExecutionResult:
        self.calls.append(call)
        return ExecutionResult(speech=f"Handled {call.intent_name}.")


def _request(text: str = "turn the kitchen lights off") -> VoiceRequest:
    return VoiceRequest(text=text, conversation_key="planning-test")


def _catalog() -> CatalogSnapshot:
    return CatalogSnapshot(
        entities=(
            CatalogEntity(
                entity_id="light.kitchen_ceiling",
                name="Kitchen Ceiling",
                aliases=(),
                domain="light",
                area_id="kitchen",
            ),
            CatalogEntity(
                entity_id="light.kitchen_counter",
                name="Kitchen Counter",
                aliases=(),
                domain="light",
                area_id="kitchen",
            ),
        ),
        areas=(CatalogArea(area_id="kitchen", name="Kitchen"),),
    )


def _area_match() -> IntentMatch:
    return IntentMatch(
        intent_name="HassTurnOff",
        slots={
            "area": SlotValue(
                value="Kitchen",
                text="kitchen",
                metadata={"area_id": "kitchen"},
            ),
            "domain": SlotValue(value="light", text="lights"),
        },
    )


def test_plan_exposes_operation_call_and_resolved_entities_without_execution():
    executor = _RecordingExecutor()
    engine = DeterministicIntentEngine(
        _Recognizer((_area_match(),)),
        _CatalogProvider(_catalog()),
        executor,
    )

    plan = engine.plan(_request())

    assert executor.calls == []
    assert plan.response is None
    assert len(plan.steps) == 1
    step = plan.steps[0]
    assert step.operation == "HassTurnOff"
    assert step.call == OhfIntentCall(
        intent_name="HassTurnOff",
        data={"area": "Kitchen", "domain": "light"},
    )
    assert step.entity_ids == (
        "light.kitchen_ceiling",
        "light.kitchen_counter",
    )


def test_plan_preserves_declines_and_represents_clarification_without_steps():
    executor = _RecordingExecutor()
    no_match_engine = DeterministicIntentEngine(
        _Recognizer(()),
        _CatalogProvider(_catalog()),
        executor,
    )
    with pytest.raises(RouteDeclined):
        no_match_engine.plan(_request("do something unsupported"))

    relative_match = IntentMatch(
        intent_name="HassTurnOn",
        slots={"domain": SlotValue(value="light", text="lights")},
    )
    clarification_engine = DeterministicIntentEngine(
        _Recognizer((relative_match,)),
        _CatalogProvider(_catalog()),
        executor,
    )
    plan = clarification_engine.plan(_request("turn on the lights"))

    assert plan.steps == ()
    assert plan.response == "Which area should I use?"
    assert executor.calls == []


@pytest.mark.asyncio
async def test_handle_executes_a_multi_step_plan_in_order(monkeypatch):
    executor = _RecordingExecutor()
    engine = DeterministicIntentEngine(
        _Recognizer(()),
        _CatalogProvider(CatalogSnapshot()),
        executor,
    )
    planned_calls = (
        OhfIntentCall("HassTurnOff", {"name": "Kitchen Ceiling"}),
        OhfIntentCall("HassTurnOn", {"name": "Evening"}),
    )
    plan = IntentPlan(
        steps=tuple(
            PlannedIntent(call=call, entity_ids=(entity_id,))
            for call, entity_id in zip(
                planned_calls,
                ("light.kitchen_ceiling", "scene.evening"),
                strict=True,
            )
        )
    )
    monkeypatch.setattr(engine, "plan", lambda request: plan)

    speech = await engine.handle(_request("turn off the light and start evening"))

    assert executor.calls == list(planned_calls)
    assert speech == "Handled HassTurnOff. Handled HassTurnOn."
