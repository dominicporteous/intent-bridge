from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from intent_bridge.core.voice import RouteDeclined
from intent_bridge.intent_engine.models import (
    CatalogArea,
    CatalogEntity,
    CatalogSnapshot,
    IntentPlan,
    OhfIntentCall,
    PlannedIntent,
    semantic_effect_for_call,
)
from intent_bridge.intent_engine.natural_language import NaturalLanguageIntentPlanner
from intent_bridge.intent_engine.supplemental import (
    DialogueState,
    PendingClarification,
    PlanningSession,
    ReferentCardinality,
    SupplementalIntentPlanner,
    _canonical_item,
    _context_items,
    _first_number,
    _number,
    _shopping_items,
    _time_trigger,
)


def _entity(entity_id: str, name: str, area_id: str | None = None) -> CatalogEntity:
    return CatalogEntity(
        entity_id=entity_id,
        name=name,
        aliases=(),
        domain=entity_id.split(".", 1)[0],
        area_id=area_id,
    )


@pytest.fixture
def catalog() -> CatalogSnapshot:
    return CatalogSnapshot(
        entities=(
            _entity("todo.chores", "Chores"),
            _entity("timer.oven", "Oven Timer"),
            _entity("timer.laundry", "Laundry Timer"),
            _entity("light.kitchen_ceiling", "Ceiling Light", "kitchen"),
            _entity("light.kitchen_counter", "Counter Light", "kitchen"),
            _entity("cover.living_blinds", "Living Room Blinds", "living"),
            _entity("cover.garage_door", "Garage Door", "garage"),
            _entity("switch.coffee_maker", "Coffee Maker", "kitchen"),
            _entity("switch.dishwasher", "Dishwasher", "kitchen"),
            _entity("switch.rangehood", "Rangehood", "kitchen"),
            _entity("switch.washing_machine", "Washing Machine", "laundry"),
            _entity("switch.dryer", "Dryer", "laundry"),
            _entity("climate.thermostat", "Thermostat", "living"),
            _entity("fan.ceiling", "Ceiling Fan", "living"),
            _entity("lock.front_door", "Front Door", "living"),
            _entity("media_player.tv", "Living Room TV", "living"),
            _entity("light.floor_lamp", "Floor Lamp", "living"),
        ),
        areas=(
            CatalogArea("kitchen", "Kitchen"),
            CatalogArea("living", "Living Room"),
            CatalogArea("laundry", "Laundry"),
            CatalogArea("garage", "Garage"),
        ),
    )


def test_shopping_and_todo_list_wording_produce_canonical_calls(catalog):
    planner = SupplementalIntentPlanner()

    added = planner.plan("Please put some eggs on my shopping list", catalog)
    completed = planner.plan("Cross olive oil off the shopping list if you could", catalog)
    removed = planner.plan(
        "Please take clean bathroom off the Chores list for me",
        catalog,
        {"list_items": {"Chores": ["Clean the bathroom", "Do the laundry"]}},
    )

    assert added.steps[0].call == OhfIntentCall("HassShoppingListAddItem", {"item": "eggs"})
    assert completed.steps[0].call == OhfIntentCall(
        "HassShoppingListCompleteItem", {"item": "olive oil"}
    )
    assert removed.steps[0].call == OhfIntentCall(
        "HassListRemoveItem",
        {"name": "Chores", "item": "Clean the bathroom"},
    )
    assert removed.steps[0].entity_ids == ("todo.chores",)


def test_list_query_and_unknown_operation(catalog):
    planner = SupplementalIntentPlanner()

    query = planner.plan("Read back the most recent shopping list items", catalog)

    assert query.steps[0].operation == "HassShoppingListLastItems"
    with pytest.raises(RouteDeclined, match="operation"):
        planner.plan("Tell me about the shopping list", catalog)


def test_timer_planning_uses_named_and_active_timer_context(catalog):
    planner = SupplementalIntentPlanner()

    started = planner.plan("Start the oven timer for 2 minutes 30 seconds", catalog)
    increased = planner.plan(
        "Give that timer five more minutes",
        catalog,
        {"active_timer_ids": ["timer.laundry"]},
    )
    paused = planner.plan("Freeze the laundry timer", catalog)

    started_call = OhfIntentCall(
        "HassStartTimer", {"name": "Oven", "minutes": 2, "seconds": 30}
    )
    assert started.steps[0] == PlannedIntent(
        call=started_call,
        entity_ids=("timer.oven",),
        effect=semantic_effect_for_call(started_call),
    )
    assert increased.steps[0].call == OhfIntentCall(
        "HassIncreaseTimer", {"name": "Laundry", "minutes": 5}
    )
    assert paused.steps[0].operation == "HassPauseTimer"
    assert paused.steps[0].entity_ids == ("timer.laundry",)


def test_anonymous_timer_cancel_all_and_status(catalog):
    planner = SupplementalIntentPlanner()

    anonymous = planner.plan("Create a 45 second timer", catalog)
    cancelled = planner.plan("Clear every timer you have going", catalog)
    status = planner.plan(
        "How much time is left on that timer?",
        catalog,
        {"active_timer_ids": ["timer.oven"]},
    )

    assert anonymous.steps[0].entity_ids == ("timer.abstract",)
    assert anonymous.steps[0].call.data == {"seconds": 45}
    assert cancelled.steps[0].operation == "HassCancelAllTimers"
    assert cancelled.steps[0].entity_ids == ("timer.laundry", "timer.oven")
    assert status.steps[0].call == OhfIntentCall("HassTimerStatus", {"name": "Oven"})


@pytest.mark.parametrize(
    ("words", "expected_trigger", "expected_action"),
    [
        (
            "Create an automation at 7am to open the living room blinds and turn on the coffee maker",
            {"platform": "time", "at": "07:00:00"},
            {"action": "cover.open_cover", "entity_id": "cover.living_blinds"},
        ),
        (
            "Create an automation to turn on the dryer when the washing machine finishes",
            {"platform": "state", "entity_id": "switch.washing_machine", "to": "off"},
            {"action": "switch.turn_on", "entity_id": "switch.dryer"},
        ),
        (
            "Create a rule to turn on the ceiling fan when the thermostat goes above 24 degrees",
            {
                "platform": "numeric_state",
                "entity_id": "climate.thermostat",
                "above": 24,
            },
            {"action": "fan.turn_on", "entity_id": "fan.ceiling"},
        ),
        (
            "Create an automation to dim the floor lamp to 20 percent at sunset",
            {"platform": "sun", "event": "sunset"},
            {
                "action": "light.turn_on",
                "entity_id": "light.floor_lamp",
                "brightness_pct": 20,
            },
        ),
    ],
)
def test_automation_planning_is_declarative(
    catalog,
    words,
    expected_trigger,
    expected_action,
):
    plan = SupplementalIntentPlanner().plan(words, catalog)
    definition = plan.steps[0].call.data["definition"]

    assert plan.steps[0].operation == "IntentBridgeCreateAutomation"
    assert definition["trigger"] == expected_trigger
    assert expected_action in definition["actions"]


@dataclass
class _QueuedPlanner:
    plans: list[IntentPlan]

    def plan(self, text, catalog, origin_context=None):
        return self.plans.pop(0)


def test_dialogue_session_retains_referents_for_relative_adjustment(catalog):
    query = IntentPlan(
        steps=(
            PlannedIntent(
                OhfIntentCall("HassGetState", {"area": "Kitchen", "domain": "light"}),
                ("light.kitchen_ceiling", "light.kitchen_counter"),
            ),
        )
    )
    session = PlanningSession(_QueuedPlanner([query]))

    first = session.plan_turn("What is the brightness of the kitchen lights?", catalog)
    second = session.plan_turn("Set them to thirty three percent", catalog)

    assert first.state.referent_entity_ids == (
        "light.kitchen_ceiling",
        "light.kitchen_counter",
    )
    assert second.plan.steps[0].call == OhfIntentCall(
        "HassLightSet",
        {"area": "Kitchen", "domain": "light", "brightness": 33},
    )
    assert second.state.pending is None


@pytest.mark.parametrize(
    "words",
    [
        "Eighty two",
        "Set to 82",
        "Please adjust to eighty two.",
        "Change to 82 percent if you don't mind",
        "Quickly set to 82",
        "Set brightness to 82",
        "Flip to 82",
        "Flick on 82",
    ],
)
def test_dialogue_session_accepts_safe_bare_numeric_follow_up(catalog, words):
    state = DialogueState(
        referent_entity_ids=("light.kitchen_ceiling", "light.kitchen_counter"),
        referent_data={"area": "Kitchen", "domain": "light"},
        referent_intent_name="HassGetState",
    )
    session = PlanningSession(_QueuedPlanner([]), state)

    step = session.plan(words, catalog).steps[0]

    assert step.operation == "HassLightSet"
    assert step.call.data["brightness"] == 82
    assert step.entity_ids == ("light.kitchen_ceiling", "light.kitchen_counter")


def test_ambiguous_property_query_retains_group_for_numeric_follow_up(catalog):
    session = PlanningSession(
        _QueuedPlanner([IntentPlan(response="I found more than one possible target.")])
    )

    first = session.plan_turn("Kitchen light brightness level?", catalog)
    second = session.plan_turn("Make it 27 if you don't mind.", catalog)

    assert first.plan.steps == ()
    assert first.state.pending is not None
    assert second.plan.response is None
    assert second.plan.steps[0].operation == "HassLightSet"
    assert second.plan.steps[0].call.data == {"brightness": 27}
    assert second.plan.steps[0].entity_ids == (
        "light.kitchen_ceiling",
        "light.kitchen_counter",
    )
    assert second.state.pending is None


def test_ambiguous_property_query_rejects_unrelated_numeric_reply(catalog):
    session = PlanningSession(NaturalLanguageIntentPlanner())
    session.plan_turn("Kitchen light brightness level?", catalog)

    followup = session.plan_turn("I have 27 apples", catalog)

    assert followup.plan.steps == ()
    assert followup.plan.response == "I found more than one possible target. Please be more specific."
    assert followup.state.pending is not None


def test_dialogue_qualifier_repeats_prior_multi_entity_intent(catalog):
    lamps = (
        _entity("light.left_bedside", "Left Bedside Lamp", "living"),
        _entity("light.right_bedside", "Right Bedside Lamp", "living"),
    )
    lamp_catalog = replace(catalog, entities=(*catalog.entities, *lamps))
    initial = IntentPlan(
        steps=(
            PlannedIntent(
                OhfIntentCall("HassTurnOff", {"area": "Living Room", "domain": "light"}),
                tuple(lamp.entity_id for lamp in lamps),
            ),
        )
    )
    session = PlanningSession(_QueuedPlanner([initial]))

    first = session.plan_turn("Turn off the bedside lamps", lamp_catalog)
    second = session.plan_turn("The left one", lamp_catalog)

    assert first.state.referent_intent_name == "HassTurnOff"
    qualifier_call = OhfIntentCall("HassTurnOff", {"name": "Left Bedside Lamp"})
    assert second.plan.steps[0] == PlannedIntent(
        qualifier_call,
        ("light.left_bedside",),
        effect=semantic_effect_for_call(qualifier_call),
    )


def test_lamp_wording_creates_and_resolves_pending_clarification(catalog):
    lamps = (
        _entity("light.left_bedside", "Left Bedside Lamp", "living"),
        _entity("light.right_bedside", "Right Bedside Lamp", "living"),
    )
    lamp_catalog = replace(catalog, entities=(*catalog.entities, *lamps))
    session = PlanningSession(
        _QueuedPlanner([IntentPlan(response="Which bedside lamp?")])
    )

    first = session.plan_turn("Turn off the bedside lamp", lamp_catalog)
    second = session.plan_turn("Left bedside", lamp_catalog)

    assert first.state.pending is not None
    assert second.plan.steps[0].operation == "HassTurnOff"
    assert second.plan.steps[0].entity_ids == ("light.left_bedside",)


def test_dialogue_session_exposes_and_resolves_clarification(catalog):
    clarification = IntentPlan(response="I found more than one possible target.")
    session = PlanningSession(_QueuedPlanner([clarification]))

    first = session.plan_turn("Turn on the kitchen light", catalog)
    second = session.plan_turn("The ceiling one", catalog)

    assert first.state.pending is not None
    assert set(first.state.pending.candidate_entity_ids) == {
        "light.kitchen_ceiling",
        "light.kitchen_counter",
    }
    assert second.plan.steps[0].call == OhfIntentCall(
        "HassTurnOn", {"name": "Ceiling Light"}
    )
    assert second.plan.steps[0].entity_ids == ("light.kitchen_ceiling",)


def test_pending_clarification_retains_complete_property_operation(catalog):
    session = PlanningSession(
        _QueuedPlanner(
            [IntentPlan(response="I found more than one possible target.")]
        )
    )

    first = session.plan_turn("Set the kitchen light brightness to 27", catalog)

    assert first.plan.steps == ()
    pending = first.state.pending
    assert pending is not None
    assert pending.original_predicate == "HassLightSet"
    assert pending.data == {"brightness": 27}
    assert pending.property == "brightness"
    assert pending.value == 27
    assert pending.requested_effect is not None
    assert pending.requested_effect.explicit_power_transition is False
    assert pending.target_constraints == {"domain": "light", "area_id": "kitchen"}
    assert pending.intended_cardinality == ReferentCardinality.SINGULAR
    assert pending.required_constraint == "single target"
    assert pending.source_clause == "Set the kitchen light brightness to 27"
    assert first.state.unresolved is not None
    assert first.state.unresolved.original_frame is pending.original_frame
    assert first.state.unresolved.candidate_targets == pending.candidate_entity_ids


def test_invalid_qualifier_keeps_transaction_then_valid_qualifier_executes_once(catalog):
    session = PlanningSession(
        _QueuedPlanner(
            [IntentPlan(response="I found more than one possible target.")]
        )
    )
    first = session.plan_turn("Set the kitchen light brightness to 27", catalog)

    invalid = session.plan_turn("The kitchen one", catalog)
    valid = session.plan_turn("The ceiling one", catalog)

    assert invalid.plan.steps == ()
    assert invalid.state is first.state
    assert valid.plan.response is None
    assert len(valid.plan.steps) == 1
    step = valid.plan.steps[0]
    assert step.operation == "HassLightSet"
    assert step.call.data == {"brightness": 27, "name": "Ceiling Light"}
    assert step.entity_ids == ("light.kitchen_ceiling",)
    assert step.effect == first.state.pending.requested_effect
    assert valid.state.pending is None
    assert valid.state.unresolved is None


def test_complete_command_replaces_stale_pending_clarification(catalog):
    office_light = _entity("light.office", "Office Light", "office")
    expanded_catalog = replace(
        catalog,
        entities=(*catalog.entities, office_light),
        areas=(*catalog.areas, CatalogArea("office", "Office")),
    )
    replacement_call = OhfIntentCall("HassTurnOff", {"name": "Office Light"})
    replacement = IntentPlan(
        steps=(
            PlannedIntent(
                replacement_call,
                ("light.office",),
                effect=semantic_effect_for_call(replacement_call),
            ),
        )
    )
    session = PlanningSession(
        _QueuedPlanner(
            [
                IntentPlan(response="I found more than one possible target."),
                replacement,
            ]
        )
    )

    first = session.plan_turn("Turn on the kitchen light", expanded_catalog)
    second = session.plan_turn("Can you turn the office light off", expanded_catalog)

    assert first.state.pending is not None
    assert second.plan == replacement
    assert second.state.pending is None
    assert second.state.unresolved is None


def test_planning_session_accepts_explicit_state_and_can_reset(catalog):
    state = DialogueState(
        referent_entity_ids=("cover.living_blinds",),
        referent_data={"name": "Living Room Blinds"},
    )
    session = PlanningSession(_QueuedPlanner([]), state)

    plan = session.plan("Close them", catalog)
    session.reset()

    assert plan.steps[0].operation == "HassTurnOff"
    assert session.state == DialogueState()


def test_number_and_context_helpers_reject_malformed_values():
    assert _number("one-point-five") is None
    assert _number("1.5") == 1.5
    assert _first_number("there is no number here") is None
    assert _context_items(None, "Chores") == ()
    assert _context_items({"list_items": ["one", ""]}, "Chores") == ("one",)
    assert _context_items({"list_items": "not-a-sequence-of-items"}, "Chores") == ()
    assert _shopping_items(None) == ()
    assert _shopping_items({"shopping_list_items": "eggs"}) == ()
    assert _canonical_item("", ("something",)) == ""
    assert _canonical_item("clean", ("clean room", "clean kitchen")) == "clean"


def test_todo_add_complete_and_ambiguous_list(catalog):
    planner = SupplementalIntentPlanner()
    added = planner.plan("Add wash the dishes to Chores", catalog)
    completed = planner.plan(
        "Mark laundry done",
        catalog,
        {"list_items": {"Chores": ["Do the laundry"]}},
    )
    two_lists = replace(
        catalog,
        entities=(*catalog.entities, _entity("todo.errands", "Errands")),
    )
    ambiguous = planner.plan("Add call the dentist to the list", two_lists)

    assert added.steps[0].operation == "HassListAddItem"
    assert completed.steps[0].call.data["item"] == "Do the laundry"
    assert ambiguous.response == "Which list should I use?"


def test_list_dispatch_requires_list_evidence(catalog):
    planner = SupplementalIntentPlanner()

    with pytest.raises(RouteDeclined, match="supplemental"):
        planner.plan("Put the living room blinds at 76", catalog)


@pytest.mark.parametrize(
    ("words", "context", "operation", "item"),
    [
        (
            "check orange juice off the shopping list",
            {"shopping_list_items": ["orange juice", "olive oil"]},
            "HassShoppingListCompleteItem",
            "orange juice",
        ),
        (
            "Check off olive oil on the list",
            {"shopping_list_items": ["olive oil", "orange juice"]},
            "HassShoppingListCompleteItem",
            "olive oil",
        ),
        (
            "put bread on the done list",
            {"shopping_list_items": ["bread", "butter"]},
            "HassShoppingListCompleteItem",
            "bread",
        ),
        (
            "stick bread on the completed list",
            {"shopping_list_items": ["bread", "butter"]},
            "HassShoppingListCompleteItem",
            "bread",
        ),
        (
            "add take out the trash to the Chores list",
            {},
            "HassListAddItem",
            "Take out the trash",
        ),
        (
            "Put tidy the bedroom off the Chores list",
            {"list_items": {"Chores": ["tidy the bedroom"]}},
            "HassListRemoveItem",
            "Tidy the bedroom",
        ),
    ],
)
def test_list_operation_is_derived_from_the_command_not_item_words(
    catalog, words, context, operation, item
):
    step = SupplementalIntentPlanner().plan(words, catalog, context).steps[0]

    assert step.operation == operation
    assert step.call.data["item"] == item


@pytest.mark.parametrize(
    ("words", "expected"),
    [
        ("If possible, put wash the dishes on the Chores list", "Wash the dishes"),
        ("Put vacuuming the living room on Chores", "Vacuum the living room"),
        ("Please put taking out the trash on Chores", "Take out the trash"),
    ],
)
def test_todo_items_are_cleaned_and_canonicalized(catalog, words, expected):
    known = ["wash the dishes", "vacuum the living room", "take out the trash"]
    step = SupplementalIntentPlanner().plan(
        words,
        catalog,
        {"list_items": {"Chores": known}},
    ).steps[0]

    assert step.operation == "HassListAddItem"
    assert step.call.data["item"] == expected


@pytest.mark.parametrize(
    ("words", "expected"),
    [
        ("Reduce the oven timer by 30 seconds", "HassDecreaseTimer"),
        ("Cancel the oven timer", "HassCancelTimer"),
        ("Start the oven timer again", "HassUnpauseTimer"),
    ],
)
def test_more_timer_operations(catalog, words, expected):
    assert SupplementalIntentPlanner().plan(words, catalog).steps[0].operation == expected


def test_timer_rejects_missing_duration_and_ignores_invalid_active_context(catalog):
    planner = SupplementalIntentPlanner()

    with pytest.raises(RouteDeclined, match="duration"):
        planner.plan("Adjust the timer", catalog, {"active_timer_ids": "timer.oven"})


def test_time_trigger_handles_midnight_evening_and_invalid_hours():
    assert _time_trigger("at 12am") == {"platform": "time", "at": "00:00:00"}
    assert _time_trigger("at eight in the evening") == {
        "platform": "time",
        "at": "20:00:00",
    }
    assert _time_trigger("at 25pm") is None
    assert _time_trigger("sometime tomorrow") is None


def test_automation_supports_climate_global_light_and_lock_actions(catalog):
    planner = SupplementalIntentPlanner()
    climate = planner.plan(
        "Create an automation to set the thermostat to 19 at 10pm",
        catalog,
    )
    global_lights = planner.plan(
        "Create a rule to lock the front door and turn off all lights at 11pm",
        catalog,
    )

    assert climate.steps[0].call.data["definition"]["actions"] == [
        {
            "action": "climate.set_temperature",
            "entity_id": "climate.thermostat",
            "temperature": 19,
        }
    ]
    assert global_lights.steps[0].call.data["definition"]["actions"] == [
        {"action": "lock.lock", "entity_id": "lock.front_door"},
        {"action": "light.turn_off", "domain": "light"},
    ]


def test_automation_does_not_expand_weak_area_overlap_to_unrelated_devices(catalog):
    plan = SupplementalIntentPlanner().plan(
        "Create an automation to turn on the counter light when the ceiling light turns on",
        catalog,
    )
    definition = plan.steps[0].call.data["definition"]

    assert definition["trigger"] == {
        "platform": "state",
        "entity_id": "light.kitchen_ceiling",
        "to": "on",
    }
    assert definition["actions"] == [
        {"action": "light.turn_on", "entity_id": "light.kitchen_counter"}
    ]


def test_automation_bench_light_does_not_target_other_kitchen_devices(catalog):
    bench_catalog = replace(
        catalog,
        entities=tuple(
            replace(entity, name="Kitchen Bench Light", aliases=("Bench Light",))
            if entity.entity_id == "light.kitchen_counter"
            else entity
            for entity in catalog.entities
        ),
    )

    definition = SupplementalIntentPlanner().plan(
        "Create an automation to turn on the kitchen bench light when the kitchen ceiling light turns on",
        bench_catalog,
    ).steps[0].call.data["definition"]

    assert definition["actions"] == [
        {"action": "light.turn_on", "entity_id": "light.kitchen_counter"}
    ]
    assert definition["trigger"] == {
        "platform": "state",
        "entity_id": "light.kitchen_ceiling",
        "to": "on",
    }


@pytest.mark.parametrize(
    ("entity_id", "data", "words", "expected"),
    [
        ("cover.living_blinds", {"name": "Living Room Blinds"}, "Set them to 40", "HassSetPosition"),
        ("climate.thermostat", {"name": "Thermostat"}, "Set it to 20", "HassClimateSetTemperature"),
        ("fan.ceiling", {"name": "Ceiling Fan"}, "Set it to 50", "HassFanSetSpeed"),
        ("media_player.tv", {"name": "Living Room TV"}, "Set it to 30", "HassSetVolume"),
        ("lock.front_door", {"name": "Front Door"}, "Unlock it", "HassTurnOff"),
        ("lock.front_door", {"name": "Front Door"}, "Lock it", "HassTurnOn"),
        ("light.floor_lamp", {"name": "Floor Lamp"}, "Deactivate it", "HassTurnOff"),
        ("light.floor_lamp", {"name": "Floor Lamp"}, "Enable it", "HassTurnOn"),
    ],
)
def test_relative_dialogue_operations(catalog, entity_id, data, words, expected):
    session = PlanningSession(
        _QueuedPlanner([]),
        DialogueState(referent_entity_ids=(entity_id,), referent_data=data),
    )

    assert session.plan(words, catalog).steps[0].operation == expected


@pytest.mark.parametrize(
    ("entity_id", "words", "operation", "slot", "value"),
    [
        ("cover.living_blinds", "Quickly move to 40", "HassSetPosition", "position", 40),
        ("climate.thermostat", "Set temperature to 20", "HassClimateSetTemperature", "temperature", 20),
        ("fan.ceiling", "Set level to 50", "HassFanSetSpeed", "percentage", 50),
        ("media_player.tv", "Change the volume_level to 30", "HassSetVolume", "volume_level", 30),
    ],
)
def test_bare_numeric_follow_up_infers_attribute_from_referent_domain(
    catalog, entity_id, words, operation, slot, value
):
    session = PlanningSession(
        _QueuedPlanner([]),
        DialogueState(referent_entity_ids=(entity_id,)),
    )

    step = session.plan(words, catalog).steps[0]

    assert step.operation == operation
    assert step.call.data[slot] == value


@pytest.mark.parametrize(
    ("entity_id", "words", "expected"),
    [
        ("light.floor_lamp", "Ignite them", "HassTurnOn"),
        ("switch.coffee_maker", "Fire it up", "HassTurnOn"),
        ("media_player.tv", "Power it down", "HassTurnOff"),
        ("lock.front_door", "Secure it", "HassTurnOn"),
        ("lock.front_door", "Release it", "HassTurnOff"),
        ("lock.front_door", "Flick the lock on", "HassTurnOn"),
    ],
)
def test_relative_dialogue_understands_common_action_paraphrases(
    catalog, entity_id, words, expected
):
    session = PlanningSession(
        _QueuedPlanner([]),
        DialogueState(referent_entity_ids=(entity_id,)),
    )

    assert session.plan(words, catalog).steps[0].operation == expected


@pytest.mark.parametrize(
    ("entity_id", "state", "words", "expected"),
    [
        ("light.floor_lamp", "off", "Hit the switch", "HassTurnOn"),
        ("lock.front_door", "locked", "Do it", "HassTurnOff"),
    ],
)
def test_relative_toggle_uses_the_referent_current_state(
    catalog, entity_id, state, words, expected
):
    stateful_catalog = replace(
        catalog,
        entities=tuple(
            replace(entity, state=state) if entity.entity_id == entity_id else entity
            for entity in catalog.entities
        ),
    )
    session = PlanningSession(
        _QueuedPlanner([]),
        DialogueState(referent_entity_ids=(entity_id,)),
    )

    assert session.plan(words, stateful_catalog).steps[0].operation == expected


def test_unresolved_pending_clarification_is_retained(catalog):
    pending = PendingClarification(
        "HassTurnOn",
        ("light.kitchen_ceiling", "light.kitchen_counter"),
    )
    state = DialogueState(pending=pending)
    session = PlanningSession(_QueuedPlanner([]), state)

    result = session.plan_turn("That one", catalog)

    assert result.plan.response == "Which one did you mean?"
    assert result.state is state


def test_non_relative_reply_uses_delegate_and_preserves_state(catalog):
    state = DialogueState(referent_entity_ids=("light.floor_lamp",))
    response = IntentPlan(response="No change requested.")
    session = PlanningSession(_QueuedPlanner([response]), state)

    result = session.plan_turn("Thanks", catalog)

    assert result.plan is response
    assert result.state is state


def test_multistep_dialogue_state_retains_clause_operations_without_union(catalog):
    plan = IntentPlan(
        steps=(
            PlannedIntent(
                OhfIntentCall("HassGetState", {"name": "Ceiling Light"}),
                ("light.kitchen_ceiling",),
            ),
            PlannedIntent(
                OhfIntentCall("HassGetState", {"name": "Counter Light"}),
                ("light.kitchen_counter",),
            ),
        )
    )
    turn = PlanningSession(_QueuedPlanner([plan])).plan_turn(
        "Check the ceiling light and then the counter light", catalog
    )

    assert turn.state.focus is not None
    assert turn.state.focus.entity_set == ("light.kitchen_counter",)
    assert turn.state.focus.cardinality is ReferentCardinality.SINGULAR
    assert turn.state.focus.selected_member == "light.kitchen_counter"
    assert tuple(turn.state.prior_operations) == ("frame-1", "frame-2")
    assert turn.state.prior_operations["frame-1"].resolved_targets == (
        "light.kitchen_ceiling",
    )
    assert turn.state.prior_operations["frame-2"].resolved_targets == (
        "light.kitchen_counter",
    )


def test_coordinated_multistep_query_focuses_complete_target_set(catalog):
    plan = IntentPlan(
        steps=(
            PlannedIntent(
                OhfIntentCall("HassGetState", {"name": "Floor Lamp"}),
                ("light.floor_lamp",),
            ),
            PlannedIntent(
                OhfIntentCall("HassGetState", {"area": "Kitchen", "domain": "light"}),
                ("light.kitchen_ceiling", "light.kitchen_counter"),
            ),
        )
    )
    session = PlanningSession(_QueuedPlanner([plan]))

    first = session.plan_turn("Check the living and kitchen lights", catalog)
    second = session.plan_turn("Turn them off", catalog)

    assert first.state.focus is not None
    assert first.state.focus.entity_set == (
        "light.floor_lamp",
        "light.kitchen_ceiling",
        "light.kitchen_counter",
    )
    assert first.state.clause_referents["them"].frame_id is None
    assert second.plan.steps[0].entity_ids == first.state.focus.entity_set


def test_one_entity_with_plural_name_has_singular_cardinality(catalog):
    plural_entity = _entity("light.ceiling_bank", "Ceiling Lights", "living")
    plural_catalog = replace(catalog, entities=(*catalog.entities, plural_entity))
    query = IntentPlan(
        steps=(
            PlannedIntent(
                OhfIntentCall("HassGetState", {"name": "Ceiling Lights"}),
                (plural_entity.entity_id,),
            ),
        )
    )
    session = PlanningSession(_QueuedPlanner([query]))

    first = session.plan_turn("What is the state of the ceiling lights?", plural_catalog)
    second = session.plan_turn("Turn it on", plural_catalog)

    assert first.state.focus is not None
    assert first.state.focus.cardinality is ReferentCardinality.SINGULAR
    assert first.state.focus.selected_member == plural_entity.entity_id
    assert second.plan.steps[0].entity_ids == (plural_entity.entity_id,)


def test_singular_property_pronoun_distributes_over_group_focus(catalog):
    query = IntentPlan(
        steps=(
            PlannedIntent(
                OhfIntentCall("HassGetState", {"area": "Kitchen", "domain": "light"}),
                ("light.kitchen_ceiling", "light.kitchen_counter"),
            ),
        )
    )
    session = PlanningSession(_QueuedPlanner([query]))
    session.plan_turn("What is the brightness of the kitchen lights?", catalog)

    followup = session.plan_turn("Set it to 27", catalog)

    assert followup.plan.response is None
    assert followup.plan.steps[0].operation == "HassLightSet"
    assert followup.plan.steps[0].entity_ids == (
        "light.kitchen_ceiling",
        "light.kitchen_counter",
    )


def test_singular_pronoun_does_not_expand_group_focus(catalog):
    query = IntentPlan(
        steps=(
            PlannedIntent(
                OhfIntentCall("HassGetState", {"area": "Kitchen", "domain": "light"}),
                ("light.kitchen_ceiling", "light.kitchen_counter"),
            ),
        )
    )
    session = PlanningSession(_QueuedPlanner([query]))
    first = session.plan_turn("What is the state of the kitchen lights?", catalog)
    second = session.plan_turn("Turn it on", catalog)

    assert first.state.focus is not None
    assert first.state.focus.cardinality is ReferentCardinality.GROUP
    assert second.plan.steps == ()
    assert second.plan.response == "Which one did you mean?"
    assert second.state.unresolved is not None
    assert second.state.unresolved.candidate_targets == (
        "light.kitchen_ceiling",
        "light.kitchen_counter",
    )
    assert second.state.unresolved.original_frame.predicate == "HassTurnOn"
    assert second.state.unresolved.required_constraint == "single target"


def test_repeating_targetless_question_does_not_create_an_ambiguous_referent(catalog):
    first_plan = IntentPlan(
        steps=(PlannedIntent(OhfIntentCall("HassGetCurrentTime", {})),)
    )
    second_plan = IntentPlan(
        steps=(PlannedIntent(OhfIntentCall("HassGetCurrentTime", {})),)
    )
    session = PlanningSession(_QueuedPlanner([first_plan, second_plan]))

    first = session.plan_turn("What time is it?", catalog)
    second = session.plan_turn("What time is it?", catalog)

    assert first.state.focus is None
    assert second.plan is second_plan
    assert second.plan.response is None


def test_property_focus_and_clause_referents_are_typed(catalog):
    query = IntentPlan(
        steps=(
            PlannedIntent(
                OhfIntentCall("HassGetState", {"area": "Kitchen", "domain": "light"}),
                ("light.kitchen_ceiling", "light.kitchen_counter"),
            ),
        )
    )
    state = PlanningSession(_QueuedPlanner([query])).plan_turn(
        "What is the brightness of the kitchen lights?", catalog
    ).state

    assert state.property_focus is not None
    assert state.property_focus.property == "brightness"
    assert state.property_focus.source_clause == "What is the brightness of the kitchen lights?"
    assert state.clause_referents["them"].target_set == state.focus.entity_set
    assert state.clause_referents["their"].frame_id == "frame-1"
    assert "it" not in state.clause_referents


def test_conditional_relative_command_retains_explicit_condition(catalog):
    query = IntentPlan(
        steps=(
            PlannedIntent(
                OhfIntentCall("HassGetState", {"name": "Floor Lamp"}),
                ("light.floor_lamp",),
            ),
        )
    )
    session = PlanningSession(_QueuedPlanner([query]))
    session.plan_turn("What is the floor lamp state?", catalog)
    followup = session.plan_turn("If it's off, turn it on.", catalog)

    assert followup.plan.steps[0].operation == "HassTurnOn"
    operation = followup.state.prior_operations["frame-2"]
    assert operation.condition is not None
    assert operation.condition.property == "activation"
    assert operation.condition.operator == "equals"
    assert operation.condition.value is False
    assert operation.condition.target_frame_id == "frame-1"
    assert operation.resolved_targets == ("light.floor_lamp",)
