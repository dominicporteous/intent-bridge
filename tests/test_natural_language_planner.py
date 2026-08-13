from __future__ import annotations

from dataclasses import dataclass

import pytest

from intent_bridge.core.voice import RouteDeclined, VoiceRequest
from intent_bridge.intent_engine.engine import DeterministicIntentEngine
from intent_bridge.intent_engine.models import (
    CatalogArea,
    CatalogEntity,
    CatalogFloor,
    CatalogSnapshot,
    ExecutionResult,
)
from intent_bridge.intent_engine.natural_language import (
    NaturalLanguageIntentPlanner,
    NaturalLanguageIntentRecognizer,
    split_compound_request,
)
from intent_bridge.intent_engine.outcomes import (
    AmbiguousTarget,
    CapabilityMismatch,
    IncompleteCompound,
    NoTarget,
    Resolved,
    UnsupportedOperation,
)


def _entity(
    entity_id: str,
    name: str,
    area_id: str,
    *,
    aliases: tuple[str, ...] = (),
) -> CatalogEntity:
    return CatalogEntity(
        entity_id=entity_id,
        name=name,
        aliases=aliases,
        domain=entity_id.split(".", 1)[0],
        area_id=area_id,
    )


@pytest.fixture
def catalog() -> CatalogSnapshot:
    return CatalogSnapshot(
        areas=(
            CatalogArea("kitchen", "Kitchen", floor_id="ground"),
            CatalogArea("living", "Living Room", aliases=("Lounge",), floor_id="ground"),
            CatalogArea("bedroom", "Bedroom", floor_id="upstairs"),
            CatalogArea("entry", "Entryway", floor_id="ground"),
            CatalogArea("backyard", "Backyard", floor_id="ground"),
            CatalogArea("bathroom", "Bathroom", floor_id="ground"),
        ),
        floors=(
            CatalogFloor("ground", "Ground Floor", aliases=("Downstairs",)),
            CatalogFloor("upstairs", "Upstairs"),
        ),
        entities=(
            _entity("light.kitchen_ceiling", "Kitchen Ceiling", "kitchen"),
            _entity(
                "light.kitchen_counter",
                "Counter Lights",
                "kitchen",
                aliases=("Bench Lights",),
            ),
            _entity("light.bedside", "Bedside Lamp", "bedroom"),
            _entity("light.living_lamp", "Living Room Lamp", "living"),
            _entity("fan.bedroom", "Bedroom Fan", "bedroom"),
            _entity("cover.living_blinds", "Living Room Blinds", "living"),
            _entity("lock.front_door", "Front Door", "entry"),
            _entity("media_player.living_tv", "TV", "living", aliases=("Telly",)),
            _entity("climate.living", "Main Thermostat", "living"),
            _entity("sensor.living_temperature", "Living Temperature Sensor", "living"),
            _entity("scene.movie_night", "Movie Night", "living"),
            _entity("script.leaving_home", "Leaving Home", "entry"),
            _entity(
                "switch.bathroom_fan",
                "Bathroom Fan",
                "bathroom",
                aliases=("Bathroom Exhaust Fan",),
            ),
            _entity("light.bathroom_mirror", "Mirror Light", "bathroom"),
            _entity("switch.bathroom_wall", "Wall Switch", "bathroom"),
            _entity("switch.coffee_maker", "Coffee Maker", "kitchen"),
            _entity("light.backyard", "Backyard Lights", "backyard"),
        ),
    )


def test_plural_area_target_resolves_group(catalog):
    plan = NaturalLanguageIntentPlanner().plan("please turn the kitchen lights off", catalog)

    assert len(plan.steps) == 1
    assert plan.steps[0].operation == "HassTurnOff"
    assert plan.steps[0].call.data == {"domain": "light", "area": "Kitchen"}
    assert plan.steps[0].entity_ids == (
        "light.kitchen_ceiling",
        "light.kitchen_counter",
    )


def test_singular_area_target_requires_clarification_when_multiple_entities_exist(catalog):
    plan = NaturalLanguageIntentPlanner().plan("please turn the kitchen light off", catalog)

    assert plan.steps == ()
    assert plan.response == "I found more than one possible target. Please be more specific."


def test_named_entity_is_more_specific_than_area(catalog):
    plan = NaturalLanguageIntentPlanner().plan(
        "Could you switch the kitchen ceiling light on?", catalog
    )

    assert plan.steps[0].call.data == {
        "name": "Kitchen Ceiling",
        "entity_id": "light.kitchen_ceiling",
    }
    assert plan.steps[0].entity_ids == ("light.kitchen_ceiling",)


def test_shared_power_predicate_distributes_over_cross_domain_targets(catalog):
    plan = NaturalLanguageIntentPlanner().plan(
        "turn on the bathroom fan and the living room lamp", catalog
    )

    assert {step.entity_ids for step in plan.steps} == {
        ("switch.bathroom_fan",),
        ("light.living_lamp",),
    }
    assert {step.operation for step in plan.steps} == {"HassTurnOn"}


def test_shared_query_predicate_distributes_over_cross_domain_targets(catalog):
    plan = NaturalLanguageIntentPlanner().plan(
        "what is the state of the bathroom fan and the living room lamp", catalog
    )

    assert {step.entity_ids for step in plan.steps} == {
        ("switch.bathroom_fan",),
        ("light.living_lamp",),
    }
    assert {step.operation for step in plan.steps} == {"HassGetState"}


@pytest.mark.parametrize(
    ("text", "operation"),
    [
        ("turn on the bathroom exhaust fan and mirror light", "HassTurnOn"),
        ("what is the status of the mirror light and exhaust fan", "HassGetState"),
    ],
)
def test_shared_predicate_distributes_over_area_scoped_elliptical_names(
    catalog, text, operation
):
    plan = NaturalLanguageIntentPlanner().plan(text, catalog)

    assert [step.operation for step in plan.steps] == [operation, operation]
    assert {step.entity_ids for step in plan.steps} == {
        ("switch.bathroom_fan",),
        ("light.bathroom_mirror",),
    }


def test_homogeneous_coordinated_names_are_independent_targets(catalog):
    plan = NaturalLanguageIntentPlanner().plan(
        "turn on the kitchen ceiling and counter lights", catalog
    )

    assert [step.operation for step in plan.steps] == ["HassTurnOn", "HassTurnOn"]
    assert {step.entity_ids for step in plan.steps} == {
        ("light.kitchen_ceiling",),
        ("light.kitchen_counter",),
    }


@pytest.mark.parametrize(
    ("text", "entity_ids"),
    [
        (
            "turn on the bedroom fan and bathroom wall switch",
            {("fan.bedroom",), ("switch.bathroom_wall",)},
        ),
        (
            "turn on the bedside lamp and living room tv",
            {("light.bedside",), ("media_player.living_tv",)},
        ),
    ],
)
def test_power_coordination_is_not_device_type_specific(catalog, text, entity_ids):
    plan = NaturalLanguageIntentPlanner().plan(text, catalog)

    assert {step.entity_ids for step in plan.steps} == entity_ids
    assert {step.operation for step in plan.steps} == {"HassTurnOn"}


@pytest.mark.parametrize("verb", ["turn on", "turn off", "deactivate"])
def test_polymorphic_predicate_does_not_inherit_fan_domain_into_named_switch(catalog, verb):
    plan = NaturalLanguageIntentPlanner().plan(
        f"{verb} the bathroom exhaust fan and coffee maker", catalog
    )

    assert {step.entity_ids for step in plan.steps} == {
        ("switch.bathroom_fan",),
        ("switch.coffee_maker",),
    }


def test_coordinated_target_uses_sibling_terms_to_resolve_an_elliptical_name(catalog):
    local_catalog = CatalogSnapshot(
        areas=(CatalogArea("kitchen", "Kitchen"), CatalogArea("bathroom", "Bathroom")),
        entities=(
            _entity("light.ceiling", "Ceiling Lights", "kitchen"),
            _entity("fan.ceiling", "Ceiling Fan", "kitchen"),
            _entity("fan.exhaust", "Exhaust Fan", "bathroom"),
        ),
    )
    plan = NaturalLanguageIntentPlanner().plan(
        "turn on the ceiling lights and fan", local_catalog
    )

    assert {step.entity_ids for step in plan.steps} == {
        ("light.ceiling",),
        ("fan.ceiling",),
    }


@pytest.mark.parametrize(
    ("text", "entity_ids"),
    [
        (
            "turn the mirror light and fan on",
            {("light.bathroom_mirror",), ("switch.bathroom_fan",)},
        ),
        (
            "turn on the coffee maker and media player",
            {("switch.coffee_maker",), ("media_player.living_tv",)},
        ),
    ],
)
def test_bounded_generic_coordination_uses_unique_or_sibling_target(catalog, text, entity_ids):
    plan = NaturalLanguageIntentPlanner().plan(text, catalog)

    assert plan.response is None
    assert {step.entity_ids for step in plan.steps} == entity_ids


def test_unscoped_generic_fan_coordination_remains_ambiguous(catalog):
    plan = NaturalLanguageIntentPlanner().plan(
        "turn on the fan and coffee maker", catalog
    )

    assert plan.steps == ()
    assert plan.response == "I found more than one possible target. Please be more specific."


def test_generic_area_member_resolves_the_complete_area_group():
    local_catalog = CatalogSnapshot(
        areas=(CatalogArea("kitchen", "Kitchen"), CatalogArea("bathroom", "Bathroom")),
        entities=(
            _entity("fan.exhaust", "Exhaust Fan", "bathroom"),
            _entity("light.ceiling", "Ceiling Lights", "kitchen"),
            _entity("light.under_cabinet", "Under Cabinet Lights", "kitchen"),
        ),
    )

    plan = NaturalLanguageIntentPlanner().plan(
        "tell me the state of the bathroom exhaust fan and kitchen lights", local_catalog
    )

    assert {step.entity_ids for step in plan.steps} == {
        ("fan.exhaust",),
        ("light.ceiling", "light.under_cabinet"),
    }


def test_shared_property_update_distributes_when_every_member_is_capable(catalog):
    plan = NaturalLanguageIntentPlanner().plan(
        "set the kitchen ceiling and counter lights brightness to 20 percent", catalog
    )

    assert {step.entity_ids for step in plan.steps} == {
        ("light.kitchen_ceiling",),
        ("light.kitchen_counter",),
    }
    assert {step.operation for step in plan.steps} == {"HassLightSet"}
    assert {step.call.data["brightness"] for step in plan.steps} == {20}


def test_shared_property_update_rejects_an_incompatible_member(catalog):
    plan = NaturalLanguageIntentPlanner().plan(
        "set the bedside lamp and living room tv brightness to 20 percent", catalog
    )

    assert plan.steps == ()
    assert plan.response == "I found more than one possible target. Please be more specific."


@pytest.mark.parametrize(
    ("text", "operation", "entity_id"),
    [
        ("open the living room blinds", "HassTurnOn", "cover.living_blinds"),
        ("shut the lounge blinds", "HassTurnOff", "cover.living_blinds"),
        ("secure the front door", "HassTurnOn", "lock.front_door"),
        ("unlock the front door", "HassTurnOff", "lock.front_door"),
        ("open the entry locks", "HassTurnOff", "lock.front_door"),
        ("close the entry locks", "HassTurnOn", "lock.front_door"),
        ("flip the entry locks to unlocked", "HassTurnOff", "lock.front_door"),
        ("flick the entry deadbolts to locked", "HassTurnOn", "lock.front_door"),
        ("turn off the front door lock", "HassTurnOff", "lock.front_door"),
        ("switch on the front door lock", "HassTurnOn", "lock.front_door"),
        ("power off the front door lock", "HassTurnOff", "lock.front_door"),
        ("activate movie night", "HassTurnOn", "scene.movie_night"),
        ("run the leaving home script", "HassTurnOn", "script.leaving_home"),
    ],
)
def test_common_action_synonyms(catalog, text, operation, entity_id):
    step = NaturalLanguageIntentPlanner().plan(text, catalog).steps[0]

    assert step.operation == operation
    assert step.entity_ids == (entity_id,)


def test_motion_wording_prefers_binary_sensor_capability_over_lock_name():
    catalog = CatalogSnapshot(
        entities=(
            _entity("lock.front_door", "Front Door", "entry"),
            _entity("binary_sensor.front_door_motion", "Front Door Motion", "entry"),
        ),
        areas=(CatalogArea("entry", "Entryway"),),
    )

    step = NaturalLanguageIntentPlanner().plan("check motion at the front door", catalog).steps[0]

    assert step.operation == "HassGetState"
    assert step.entity_ids == ("binary_sensor.front_door_motion",)


@pytest.mark.parametrize(
    ("text", "operation", "attribute", "value", "entity_id"),
    [
        (
            "dim the kitchen ceiling light to twenty percent",
            "HassLightSet",
            "brightness",
            20,
            "light.kitchen_ceiling",
        ),
        (
            "set the bedroom fan speed to fifty percent",
            "HassFanSetSpeed",
            "percentage",
            50,
            "fan.bedroom",
        ),
        (
            "set the living room blinds to 35%",
            "HassSetPosition",
            "position",
            35,
            "cover.living_blinds",
        ),
        (
            "set the main thermostat temperature to 21 degrees",
            "HassClimateSetTemperature",
            "temperature",
            21,
            "climate.living",
        ),
        (
            "set the living room tv volume to seventy percent",
            "HassSetVolume",
            "volume_level",
            70,
            "media_player.living_tv",
        ),
    ],
)
def test_numeric_operations(catalog, text, operation, attribute, value, entity_id):
    step = NaturalLanguageIntentPlanner().plan(text, catalog).steps[0]

    assert step.operation == operation
    assert step.call.data[attribute] == value
    assert step.entity_ids == (entity_id,)
    assert step.effect is not None
    assert step.effect.property == attribute
    assert step.effect.value == value
    assert step.effect.operator == "set"
    assert step.effect.explicit_power_transition is False


@pytest.mark.parametrize(
    "text",
    [
        "Living room lights 27",
        "Turn living room lights to brightness level 27",
    ],
)
def test_area_brightness_shorthand_is_a_property_only_effect(catalog, text):
    plan = NaturalLanguageIntentPlanner().plan(text, catalog)

    assert plan.response is None
    assert len(plan.steps) == 1
    step = plan.steps[0]
    assert step.operation == "HassLightSet"
    assert step.entity_ids == ("light.living_lamp",)
    assert step.call.data["brightness"] == 27
    assert step.requested_effect is step.effect
    assert step.requested_effect.property == "brightness"
    assert step.requested_effect.value == 27
    assert step.requested_effect.explicit_power_transition is False


def test_explicit_power_command_is_marked_as_a_power_transition(catalog):
    step = NaturalLanguageIntentPlanner().plan(
        "turn the living room lights on", catalog
    ).steps[0]

    assert step.requested_effect.property == "power"
    assert step.requested_effect.value is True
    assert step.requested_effect.explicit_power_transition is True


@pytest.mark.parametrize(
    "text",
    [
        "set the living room climate to 17 degrees",
        "flick the living room temp down to 17",
        "make the living room temperature 17 degrees",
    ],
)
def test_climate_domain_aliases(catalog, text):
    step = NaturalLanguageIntentPlanner().plan(text, catalog).steps[0]

    assert step.operation == "HassClimateSetTemperature"
    assert step.entity_ids == ("climate.living",)
    assert step.call.data["temperature"] == 17


def test_light_color_and_color_temperature(catalog):
    planner = NaturalLanguageIntentPlanner()

    color = planner.plan("make the bedside lamp warm white", catalog).steps[0]
    temperature = planner.plan(
        "set the bedside lamp color temperature to 3000 kelvin", catalog
    ).steps[0]

    assert color.call.data == {"name": "Bedside Lamp", "color": "warm white"}
    assert temperature.call.data == {"name": "Bedside Lamp", "temperature": 3000}


@pytest.mark.parametrize(
    ("text", "operation", "data"),
    [
        ("pause the living room tv", "HassMediaPause", {"name": "TV"}),
        ("resume the lounge telly", "HassMediaUnpause", {"name": "TV"}),
        ("mute the living room tv", "HassMediaPlayerMute", {"name": "TV"}),
        ("unmute the living room tv", "HassMediaPlayerUnmute", {"name": "TV"}),
        ("skip on the living room tv", "HassMediaNext", {"name": "TV"}),
        ("play the previous track on the living room tv", "HassMediaPrevious", {"name": "TV"}),
        (
            "turn the living room tv volume down",
            "HassSetVolumeRelative",
            {"name": "TV", "volume_step": "down"},
        ),
    ],
)
def test_media_controls(catalog, text, operation, data):
    step = NaturalLanguageIntentPlanner().plan(text, catalog).steps[0]

    assert step.operation == operation
    assert step.call.data == data


def test_bare_play_uses_the_unique_origin_area_player(catalog):
    plan = NaturalLanguageIntentPlanner().plan(
        "play",
        catalog,
        {"area_id": "living", "area_name": "Living Room"},
    )

    assert len(plan.steps) == 1
    assert plan.steps[0].operation == "HassMediaUnpause"
    assert plan.steps[0].entity_ids == ("media_player.living_tv",)
    assert plan.steps[0].call.data == {"name": "TV"}
    assert plan.steps[0].effect is not None
    assert (
        plan.steps[0].effect.property,
        plan.steps[0].effect.value,
    ) == ("playback", "playing")


def test_bare_play_prefers_the_exact_area_player_and_does_not_expand_the_group():
    local_catalog = CatalogSnapshot(
        areas=(CatalogArea("office", "Office"),),
        entities=(
            _entity("media_player.office", "Office", "office"),
            _entity("media_player.office_dot_speaker", "Office Dot Speaker", "office"),
            _entity("media_player.office_speakerstick_2", "Office Speakerstick 2", "office"),
        ),
    )

    plan = NaturalLanguageIntentPlanner().plan(
        "play",
        local_catalog,
        {"area_id": "office", "area_name": "Office"},
    )

    assert len(plan.steps) == 1
    assert plan.steps[0].entity_ids == ("media_player.office",)
    assert plan.steps[0].call.data == {
        "name": "Office",
        "entity_id": "media_player.office",
    }


def test_bare_play_clarifies_when_origin_area_has_no_default_player():
    local_catalog = CatalogSnapshot(
        areas=(CatalogArea("office", "Office"),),
        entities=(
            _entity("media_player.office_dot", "Dot Speaker", "office"),
            _entity("media_player.office_tv", "Television", "office"),
        ),
    )

    plan = NaturalLanguageIntentPlanner().plan(
        "play",
        local_catalog,
        {"area_id": "office", "area_name": "Office"},
    )

    assert plan.steps == ()
    assert plan.response == "I found more than one possible target. Please be more specific."


def test_bare_play_control_does_not_capture_media_search_requests(catalog):
    outcome = NaturalLanguageIntentPlanner().resolve(
        "play Taylor Swift",
        catalog,
        {"area_id": "living", "area_name": "Living Room"},
    )

    assert isinstance(outcome, UnsupportedOperation)


def test_queries_preserve_target_and_requested_state(catalog):
    planner = NaturalLanguageIntentPlanner()

    state = planner.plan("tell me whether the kitchen ceiling light is on", catalog).steps[0]
    temperature = planner.plan("what is the main thermostat temperature", catalog).steps[0]

    assert state.operation == "HassGetState"
    assert state.call.data == {"name": "Kitchen Ceiling", "state": "on"}
    assert temperature.operation == "HassClimateGetTemperature"
    assert temperature.entity_ids == ("climate.living",)


def test_compound_request_produces_ordered_steps_and_does_not_split_area_list(catalog):
    planner = NaturalLanguageIntentPlanner()
    compounds = split_compound_request(
        "Turn off the kitchen ceiling light and then activate movie night"
    )
    area_list = split_compound_request("turn off the kitchen and living room lights")

    plan = planner.plan("Turn off the kitchen ceiling light and then activate movie night", catalog)

    assert compounds == ("turn off the kitchen ceiling light", "activate movie night")
    assert area_list == ("turn off the kitchen and living room lights",)
    assert [step.operation for step in plan.steps] == ["HassTurnOff", "HassTurnOn"]
    assert [step.entity_ids for step in plan.steps] == [
        ("light.kitchen_ceiling",),
        ("scene.movie_night",),
    ]


@pytest.mark.parametrize(
    "text",
    [
        "turn the kitchen ceiling light off and while you're at it tell me whether the bedside lamp is on",
        "tell me whether the bedside lamp is on after turning the kitchen ceiling light off",
    ],
)
def test_compound_connectors_separate_action_from_query(catalog, text):
    plan = NaturalLanguageIntentPlanner().plan(text, catalog)

    assert {step.operation for step in plan.steps} == {"HassTurnOff", "HassGetState"}
    assert {entity_id for step in plan.steps for entity_id in step.entity_ids} == {
        "light.kitchen_ceiling",
        "light.bedside",
    }


def test_pronoun_in_later_clause_reuses_previous_targets(catalog):
    plan = NaturalLanguageIntentPlanner().plan(
        "turn the kitchen lights off and then turn them on", catalog
    )

    assert [step.operation for step in plan.steps] == ["HassTurnOff", "HassTurnOn"]
    assert plan.steps[0].entity_ids == plan.steps[1].entity_ids


def test_origin_and_dialogue_context_resolve_implicit_targets(catalog):
    planner = NaturalLanguageIntentPlanner()

    local = planner.plan("turn on the lights", catalog, {"area_id": "kitchen"})
    dialogue = planner.plan("turn it off", catalog, {"last_entity_ids": ["fan.bedroom"]})

    assert local.steps[0].entity_ids == (
        "light.kitchen_ceiling",
        "light.kitchen_counter",
    )
    assert dialogue.steps[0].entity_ids == ("fan.bedroom",)


def test_contextual_numeric_follow_up_infers_attribute_from_prior_domain(catalog):
    step = (
        NaturalLanguageIntentPlanner()
        .plan(
            "adjust it to thirty three",
            catalog,
            {"last_entity_ids": ["light.kitchen_counter"]},
        )
        .steps[0]
    )

    assert step.operation == "HassLightSet"
    assert step.call.data == {"name": "Counter Lights", "brightness": 33}


def test_floor_and_multiple_area_groups(catalog):
    planner = NaturalLanguageIntentPlanner()

    floor = planner.plan("turn off all the lights upstairs", catalog)
    areas = planner.plan("turn off the kitchen and living room lights", catalog)

    assert floor.steps[0].call.data == {"domain": "light", "floor": "Upstairs"}
    assert floor.steps[0].entity_ids == ("light.bedside",)
    assert [step.call.data["area"] for step in areas.steps] == ["Kitchen", "Living Room"]


def test_fuzzy_name_typo_and_compact_area_name(catalog):
    planner = NaturalLanguageIntentPlanner()

    typo = planner.plan("turn on the kithcen ceiling", catalog)
    compact = planner.plan("close the livingroom blinds", catalog)

    assert typo.steps[0].entity_ids == ("light.kitchen_ceiling",)
    assert compact.steps[0].entity_ids == ("cover.living_blinds",)


@pytest.mark.parametrize(
    ("text", "operation", "entity_ids"),
    [
        (
            "flip the kitchen lights off if possible",
            "HassTurnOff",
            ("light.kitchen_ceiling", "light.kitchen_counter"),
        ),
        (
            "illuminate the kitchen",
            "HassTurnOn",
            ("light.kitchen_ceiling", "light.kitchen_counter"),
        ),
        ("spin up the bedroom fan", "HassTurnOn", ("fan.bedroom",)),
        ("close the living room windows", "HassTurnOff", ("cover.living_blinds",)),
    ],
)
def test_additional_natural_action_wording(catalog, text, operation, entity_ids):
    step = NaturalLanguageIntentPlanner().plan(text, catalog).steps[0]

    assert step.operation == operation
    assert step.entity_ids == entity_ids


@pytest.mark.parametrize(
    ("text", "operation"),
    [
        ("would you mind opening the living room curtains", "HassTurnOn"),
        ("would you mind closing the living room blinds", "HassTurnOff"),
    ],
)
def test_imperative_gerund_cover_wording(catalog, text, operation):
    step = NaturalLanguageIntentPlanner().plan(text, catalog).steps[0]

    assert step.operation == operation
    assert step.entity_ids == ("cover.living_blinds",)


def test_common_area_wording_and_unmarked_brightness(catalog):
    planner = NaturalLanguageIntentPlanner()

    outside = planner.plan("switch the lights outside on", catalog)
    brightness = planner.plan("kitchen lights brightness 35", catalog)
    black = planner.plan("make the bedroom light black", catalog)

    assert outside.steps[0].call.data["area"] == "Backyard"
    assert brightness.steps[0].call.data["brightness"] == 35
    assert black.steps[0].call.data["color"] == "black"


def test_multi_area_collective_and_distinctive_qualifier_resolution(catalog):
    planner = NaturalLanguageIntentPlanner()

    collective = planner.plan("turn off the kitchen and living room illumination", catalog)
    qualified = planner.plan("switch on the kitchen lights the bench ones", catalog)

    assert {step.entity_ids for step in collective.steps} == {
        ("light.kitchen_ceiling", "light.kitchen_counter"),
        ("light.living_lamp",),
    }
    assert qualified.steps[0].entity_ids == ("light.kitchen_counter",)


def test_temperature_sensor_query_is_not_misclassified_as_thermostat_query(catalog):
    step = (
        NaturalLanguageIntentPlanner()
        .plan("what is the living temperature sensor state", catalog)
        .steps[0]
    )

    assert step.operation == "HassGetState"
    assert step.entity_ids == ("sensor.living_temperature",)


def test_area_temperature_prefers_unique_settable_climate_for_follow_up(catalog):
    step = NaturalLanguageIntentPlanner().plan(
        "can you tell me the living room temperature", catalog
    ).steps[0]

    assert step.operation == "HassClimateGetTemperature"
    assert step.entity_ids == ("climate.living",)


def test_bare_area_domain_phrase_is_an_elliptical_query(catalog):
    step = NaturalLanguageIntentPlanner().plan("living room climate", catalog).steps[0]

    assert step.operation == "HassClimateGetTemperature"
    assert step.entity_ids == ("climate.living",)


@pytest.mark.parametrize(
    "text",
    [
        "whats the temperature",
        "whats the temprature outside",
        "what is the outdoor temp",
    ],
)
def test_unqualified_temperature_queries_are_weather_queries(catalog, text):
    step = NaturalLanguageIntentPlanner().plan(text, catalog).steps[0]

    assert step.operation == "HassGetWeather"
    assert step.call.data == {}


def test_single_generic_vacuum_starts_without_confirmation():
    catalog = CatalogSnapshot(
        entities=(
            _entity(
                "vacuum.valetudo_heftyconstanteagle",
                "E20 Robot Vacuum Robot",
                "living",
            ),
        ),
        areas=(CatalogArea("living", "Living Room"),),
    )

    step = NaturalLanguageIntentPlanner().plan("turn the vacuum on", catalog).steps[0]

    assert step.operation == "HassVacuumStart"
    assert step.entity_ids == ("vacuum.valetudo_heftyconstanteagle",)


def test_multiple_generic_vacuums_request_clarification():
    catalog = CatalogSnapshot(
        entities=(
            _entity("vacuum.downstairs", "Downstairs Cleaner", "living"),
            _entity("vacuum.upstairs", "Upstairs Cleaner", "bedroom"),
        ),
    )

    plan = NaturalLanguageIntentPlanner().plan("turn the vacuum on", catalog)

    assert plan.steps == ()
    assert "more than one" in (plan.response or "")


@pytest.mark.parametrize(
    "text",
    [
        "open the living room shades",
        "would you mind closing the living room windows",
        "pull the living room curtains down",
    ],
)
def test_window_covering_words_resolve_to_topology_compatible_cover(catalog, text):
    step = NaturalLanguageIntentPlanner().plan(text, catalog).steps[0]

    assert step.entity_ids == ("cover.living_blinds",)


def test_cover_subtype_never_broadens_windows_to_a_garage_door():
    catalog = CatalogSnapshot(
        entities=(_entity("cover.garage_door", "Garage Door", "garage"),),
        areas=(CatalogArea("garage", "Garage"),),
    )

    plan = NaturalLanguageIntentPlanner().plan("open the garage windows", catalog)

    assert plan.steps == ()
    assert plan.response is not None


@pytest.mark.parametrize(
    "text",
    [
        "kitchen ceiling",
        "kitchen ceiling brightness please",
        "what is the kitchen ceiling brightness",
    ],
)
def test_unique_elliptical_and_attribute_only_requests_are_read_only(catalog, text):
    step = NaturalLanguageIntentPlanner().plan(text, catalog).steps[0]

    assert step.operation == "HassGetState"
    assert step.entity_ids == ("light.kitchen_ceiling",)


def test_elliptical_query_without_unique_topology_evidence_does_not_execute(catalog):
    plan = NaturalLanguageIntentPlanner().plan("brightness", catalog)

    assert plan.steps == ()
    assert plan.response is not None


def test_ambiguous_or_invalid_requests_never_widen_scope(catalog):
    planner = NaturalLanguageIntentPlanner()

    assert planner.plan("turn on the lights", catalog).steps == ()
    with pytest.raises(RouteDeclined):
        planner.plan("set the kitchen lights to 150 percent", catalog)
    with pytest.raises(RouteDeclined):
        planner.plan("write me a poem", catalog)


def test_recognizer_adapter_exposes_slots_and_resolved_metadata(catalog):
    match = NaturalLanguageIntentRecognizer().recognize(
        "turn off the kitchen ceiling light", catalog
    )[0]

    assert match.intent_name == "HassTurnOff"
    assert match.slots["name"].value == "Kitchen Ceiling"
    assert match.metadata["entity_ids"] == ("light.kitchen_ceiling",)


@dataclass
class _NoMatches:
    def recognize(self, text, catalog, origin_context=None):
        return ()


@dataclass
class _Catalog:
    value: CatalogSnapshot

    def snapshot(self):
        return self.value


@dataclass
class _Executor:
    async def execute(self, call):
        return ExecutionResult(speech="done")


def test_engine_can_use_natural_planner_as_opt_in_fallback(catalog):
    engine = DeterministicIntentEngine(
        _NoMatches(),
        _Catalog(catalog),
        _Executor(),
        fallback_planner=NaturalLanguageIntentPlanner(),
    )

    plan = engine.plan(VoiceRequest("open the living room blinds", "fallback"))

    assert plan.steps[0].operation == "HassTurnOn"
    assert plan.steps[0].entity_ids == ("cover.living_blinds",)


def test_natural_planner_exposes_typed_resolution_outcomes(catalog):
    planner = NaturalLanguageIntentPlanner()

    assert isinstance(planner.resolve("turn the living room tv on", catalog), Resolved)
    assert isinstance(planner.resolve("turn the kitchen light off", catalog), AmbiguousTarget)
    assert isinstance(planner.resolve("perform a barrel roll", catalog), UnsupportedOperation)
    assert isinstance(
        planner.resolve("turn the living room tv on and then set mystery", catalog),
        IncompleteCompound,
    )
    assert isinstance(
        planner.resolve(
            "set the bedside lamp and living room tv brightness to 20 percent", catalog
        ),
        CapabilityMismatch,
    )
    assert isinstance(planner.resolve("turn the office lights on", catalog), NoTarget)


def test_natural_area_light_excludes_indicator_but_explicit_ring_remains_targetable():
    catalog = CatalogSnapshot(
        entities=(
            CatalogEntity("light.office", "Office Light", (), "light", "office"),
            CatalogEntity(
                "light.voice_ring",
                "Voice LED Ring",
                (),
                "light",
                "office",
                is_indicator=True,
            ),
        ),
        areas=(CatalogArea("office", "Office"),),
    )
    planner = NaturalLanguageIntentPlanner()

    room = planner.plan("turn the office light off", catalog)
    ring = planner.plan("turn the voice led ring on", catalog)

    assert room.steps[0].entity_ids == ("light.office",)
    assert room.steps[0].call.data == {"name": "Office Light"}
    assert ring.steps[0].entity_ids == ("light.voice_ring",)
