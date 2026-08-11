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
from intent_bridge.intent_engine.natural_language import (
    NaturalLanguageIntentPlanner,
    split_compound_request,
)
from intent_bridge.intent_engine.supplemental import SupplementalIntentPlanner


@dataclass
class _Recognizer:
    matches: tuple[IntentMatch, ...]

    def recognize(self, text, catalog, origin_context=None):
        return self.matches


@dataclass
class _TvClauseRecognizer:
    texts: list[str] = field(default_factory=list)

    def recognize(self, text, catalog, origin_context=None):
        self.texts.append(text)
        if text == "check the state of the lg tv":
            return (
                IntentMatch(
                    "HassGetState",
                    {
                        "name": SlotValue(
                            "LG TV",
                            "lg tv",
                            {"entity_id": "media_player.lg_tv"},
                        )
                    },
                ),
            )
        return ()


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


@dataclass
class _ClausePlanner:
    texts: list[str] = field(default_factory=list)

    def plan(self, text, catalog, origin_context=None):
        self.texts.append(text)
        if "ceiling" in text:
            return IntentPlan(
                steps=(
                    PlannedIntent(
                        OhfIntentCall("HassTurnOff", {"name": "Kitchen Ceiling"}),
                        ("light.kitchen_ceiling",),
                    ),
                )
            )
        if "counter" in text:
            return IntentPlan(
                steps=(
                    PlannedIntent(
                        OhfIntentCall("HassTurnOn", {"name": "Kitchen Counter"}),
                        ("light.kitchen_counter",),
                    ),
                )
            )
        raise RouteDeclined("unsupported clause")


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


def _compound_catalog() -> CatalogSnapshot:
    return CatalogSnapshot(
        entities=(
            CatalogEntity("timer.oven", "Oven Timer", (), "timer"),
            CatalogEntity("media_player.lg_tv", "LG TV", (), "media_player", "living"),
            CatalogEntity("light.kitchen", "Kitchen Light", (), "light", "kitchen"),
            CatalogEntity(
                "light.kitchen_under_cabinet",
                "Under Cabinet Lights",
                (),
                "light",
                "kitchen",
            ),
            CatalogEntity("light.floor_lamp", "Floor Lamp", (), "light", "living"),
            CatalogEntity(
                "cover.window_blinds",
                "Window Blinds",
                (),
                "cover",
                "living",
            ),
            CatalogEntity("lock.front_door", "Front Door", (), "lock", "entry"),
            CatalogEntity("script.goodnight", "Goodnight", (), "script"),
        ),
        areas=(
            CatalogArea("entry", "Entry"),
            CatalogArea("kitchen", "Kitchen"),
            CatalogArea("living", "Living Room"),
        ),
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


def test_compounds_are_decomposed_before_a_preferred_planner_can_claim_them():
    planner = _ClausePlanner()
    engine = DeterministicIntentEngine(
        _Recognizer(()),
        _CatalogProvider(_catalog()),
        _RecordingExecutor(),
        preferred_planner=planner,
    )

    plan = engine.plan(
        _request("turn off the kitchen ceiling and turn on the kitchen counter")
    )

    assert planner.texts == (
        ["turn off the kitchen ceiling", "turn on the kitchen counter"]
    )
    assert [step.operation for step in plan.steps] == ["HassTurnOff", "HassTurnOn"]


def test_structural_split_preserves_target_coordination_and_orders_dependencies():
    assert split_compound_request("turn on the fan and mirror light") == (
        "turn on the fan and mirror light",
    )
    assert split_compound_request("lock the front door after pausing the LG TV") == (
        "pause the lg tv",
        "lock the front door",
    )
    assert split_compound_request(
        "Turn off the under cabinet lights while you are at it "
        "and launch the Goodnight script"
    ) == ("turn off the under cabinet lights", "launch the goodnight script")
    assert split_compound_request(
        "Please turn on the floor lamp while you're at it "
        "and get the state of the window blinds"
    ) == ("turn on the floor lamp", "get the state of the window blinds")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Start a 15 minute oven timer and also check the state of the LG TV",
            [
                ("HassStartTimer", ("timer.oven",)),
                ("HassGetState", ("media_player.lg_tv",)),
            ],
        ),
        (
            "Start a 15 minute oven timer and turn on the kitchen light",
            [
                ("HassStartTimer", ("timer.oven",)),
                ("HassTurnOn", ("light.kitchen",)),
            ],
        ),
        (
            "Pause the LG TV and lock the front door",
            [
                ("HassMediaPause", ("media_player.lg_tv",)),
                ("HassTurnOn", ("lock.front_door",)),
            ],
        ),
        (
            "Turn on the kitchen light while checking the state of the LG TV",
            [
                ("HassTurnOn", ("light.kitchen",)),
                ("HassGetState", ("media_player.lg_tv",)),
            ],
        ),
        (
            "Lock the front door after pausing the LG TV",
            [
                ("HassMediaPause", ("media_player.lg_tv",)),
                ("HassTurnOn", ("lock.front_door",)),
            ],
        ),
        (
            "Turn on the kitchen light and check its state",
            [
                ("HassTurnOn", ("light.kitchen",)),
                ("HassGetState", ("light.kitchen",)),
            ],
        ),
        (
            "Turn off the under cabinet lights while you are at it "
            "and launch the Goodnight script",
            [
                ("HassTurnOff", ("light.kitchen_under_cabinet",)),
                ("HassTurnOn", ("script.goodnight",)),
            ],
        ),
        (
            "Please turn on the floor lamp while you're at it "
            "and get the state of the window blinds",
            [
                ("HassTurnOn", ("light.floor_lamp",)),
                ("HassGetState", ("cover.window_blinds",)),
            ],
        ),
    ],
)
def test_mixed_compounds_are_fully_planned_per_structural_clause(text, expected):
    engine = DeterministicIntentEngine(
        _Recognizer(()),
        _CatalogProvider(_compound_catalog()),
        _RecordingExecutor(),
        preferred_planner=SupplementalIntentPlanner(),
        fallback_planner=NaturalLanguageIntentPlanner(),
    )

    plan = engine.plan(_request(text))

    assert plan.response is None
    assert [(step.operation, step.entity_ids) for step in plan.steps] == expected


def test_structural_automation_precedes_property_and_compound_fast_paths():
    engine = DeterministicIntentEngine(
        _Recognizer(()),
        _CatalogProvider(_compound_catalog()),
        _RecordingExecutor(),
        preferred_planner=SupplementalIntentPlanner(),
        fallback_planner=NaturalLanguageIntentPlanner(),
    )

    plan = engine.plan(
        _request("Create an automation to dim the floor lamp to 20 percent at sunset")
    )

    assert len(plan.steps) == 1
    assert plan.steps[0].operation == "IntentBridgeCreateAutomation"
    assert plan.steps[0].call.data["definition"] == {
        "trigger": {"platform": "sun", "event": "sunset"},
        "actions": [
            {
                "action": "light.turn_on",
                "entity_id": "light.floor_lamp",
                "brightness_pct": 20,
            }
        ],
    }


def test_supplemental_and_recognizer_each_receive_only_their_timer_tv_clause():
    recognizer = _TvClauseRecognizer()
    engine = DeterministicIntentEngine(
        recognizer,
        _CatalogProvider(_compound_catalog()),
        _RecordingExecutor(),
        preferred_planner=SupplementalIntentPlanner(),
        fallback_planner=NaturalLanguageIntentPlanner(),
    )

    plan = engine.plan(
        _request("Start a 15 minute oven timer and also check the state of the LG TV")
    )

    assert recognizer.texts == ["check the state of the lg tv"]
    assert [(step.operation, step.entity_ids) for step in plan.steps] == [
        ("HassStartTimer", ("timer.oven",)),
        ("HassGetState", ("media_player.lg_tv",)),
    ]


def test_compound_plan_is_rejected_atomically_when_any_clause_is_unhandled():
    engine = DeterministicIntentEngine(
        _Recognizer(()),
        _CatalogProvider(_catalog()),
        _RecordingExecutor(),
        preferred_planner=_ClausePlanner(),
    )

    with pytest.raises(RouteDeclined, match="No deterministic intent matched"):
        engine.plan(_request("turn off the kitchen ceiling and launch the unsupported widget"))


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
