"""Focused tests for benchmark-side planning result translation."""

from __future__ import annotations

import pytest

from benchmark.loader import load_corpus
from benchmark.models import BenchmarkRequest, Operation
from benchmark.production import (
    _catalog,
    _planning_context,
    _step_operations,
    expand_static_invocation,
)
from benchmark.runner import compare_operations
from intent_bridge.intent_engine.models import OhfIntentCall, PlannedIntent


@pytest.fixture(scope="module")
def corpus():
    return load_corpus()


def _step(intent: str, data=None, *entity_ids: str) -> PlannedIntent:
    return PlannedIntent(
        call=OhfIntentCall(intent, data or {}),
        entity_ids=tuple(entity_ids),
    )


def test_setup_only_timer_is_available_to_the_planner(corpus):
    home = next(home for home in corpus.homes if home.home_id == "studio")
    setup = (
        Operation(
            kind="setup",
            entity_ids=("timer.abstract",),
            payload={"start_hours": 1},
        ),
    )
    request = BenchmarkRequest(turns=("pause the timer",), home=home, setup=setup)

    catalog = _catalog(request)
    timer = next(entity for entity in catalog.entities if entity.entity_id == "timer.abstract")

    assert timer.name == "Abstract Timer"
    assert timer.domain == "timer"
    assert _planning_context(request)["active_timer_ids"] == ("timer.abstract",)


def test_timer_delta_is_translated_to_the_absolute_expected_duration(corpus):
    home = next(home for home in corpus.homes if home.home_id == "studio")
    setup = (
        Operation(
            kind="setup",
            entity_ids=("timer.abstract",),
            payload={"start_hours": 1},
        ),
    )

    operations = _step_operations(
        _step("HassIncreaseTimer", {"minutes": 5, "name": "Abstract"}, "timer.abstract"),
        home,
        setup,
    )

    assert operations == (
        Operation(
            kind="timer",
            entity_ids=("timer.abstract",),
            payload={"hours": 1, "minutes": 5},
        ),
    )


def test_timer_decrease_borrows_across_minutes(corpus):
    home = next(home for home in corpus.homes if home.home_id == "studio")
    setup = (Operation(kind="setup", entity_ids=("timer.oven",), payload={"minutes": 45}),)

    operations = _step_operations(
        _step("HassDecreaseTimer", {"seconds": 30}, "timer.oven"),
        home,
        setup,
    )

    assert operations[0].payload == {"minutes": 44, "seconds": 30}


def test_timer_translation_resolves_runtime_targets_without_fixture_identity(corpus):
    home = next(home for home in corpus.homes if home.home_id == "studio")

    named = _step_operations(
        _step("HassPauseTimer", {"name": "Oven"}),
        home,
    )
    new_unnamed = _step_operations(
        _step("HassStartTimer", {"seconds": 45.0}),
        home,
    )
    cancel_all = _step_operations(_step("HassCancelAllTimers"), home)

    assert named[0].entity_ids == ("timer.oven",)
    assert new_unnamed[0].entity_ids == ("timer.abstract",)
    assert (
        new_unnamed[0].semantic_key()
        == Operation(
            kind="timer",
            entity_ids=("timer.abstract",),
            payload={"seconds": 45},
        ).semantic_key()
    )
    assert cancel_all[0].entity_ids == ("timer.laundry", "timer.oven")


@pytest.mark.parametrize(
    ("intent", "payload", "entity_id"),
    [
        ("HassLightSet", {"brightness": 27}, "light.living_room_lamp"),
        ("HassSetPosition", {"position": 44}, "cover.living_room_blinds"),
        ("HassFanSetSpeed", {"percentage": 68}, "fan.living_room_ceiling_fan"),
        (
            "HassClimateSetTemperature",
            {"temperature": 21},
            "climate.living_room_thermostat",
        ),
        ("HassSetVolume", {"volume_level": 63}, "media_player.main_lg_tv"),
    ],
)
def test_property_setters_record_no_implicit_power_transition(
    corpus, intent, payload, entity_id
):
    home = next(home for home in corpus.homes if home.home_id == "studio")

    operations = _step_operations(_step(intent, payload, entity_id), home)

    assert operations == (
        Operation(kind="action", entity_ids=(entity_id,), state=None, payload=payload),
    )


def test_list_and_shopping_query_fields_use_gold_schema(corpus):
    home = next(home for home in corpus.homes if home.home_id == "studio")

    removed = _step_operations(
        _step(
            "HassListRemoveItem",
            {"name": "Chores", "item": "Clean the bathroom"},
            "todo.chores",
        ),
        home,
    )
    latest = _step_operations(_step("HassShoppingListLastItems"), home)

    assert removed == (
        Operation(
            kind="todo_list",
            payload={
                "list_name": "Chores",
                "item": "Clean the bathroom",
                "removed": True,
            },
        ),
    )
    assert latest == (Operation(kind="query", intent="HassShoppingListLastItems"),)


def test_official_automation_name_translates_definition(corpus):
    home = next(home for home in corpus.homes if home.home_id == "studio")
    definition = {
        "trigger": {"platform": "sun", "event": "sunset"},
        "actions": [{"action": "light.turn_on", "entity_id": "light.living_room_lamp"}],
    }

    operations = _step_operations(
        _step("HassCreateAutomation", {"definition": definition}),
        home,
    )

    assert operations == (Operation(kind="automation", payload={"definition": definition}),)


def test_scene_invocation_and_gold_expand_to_the_same_device_effects(corpus):
    scenario = next(
        scenario
        for scenario in corpus.scenarios
        if scenario.source == "family_home_basic_single_storey/devices/scenes.yaml"
        and scenario.name == "movie_night"
    )

    actual = _step_operations(
        _step("HassTurnOn", {"domain": "scene"}, "scene.movie_night"),
        scenario.home,
    )

    assert {operation.semantic_key() for operation in actual} == {
        operation.semantic_key() for operation in scenario.expected
    }
    assert all(
        not entity_id.startswith("scene.")
        for operation in scenario.expected
        for entity_id in operation.entity_ids
    )


def test_static_invocation_interpreter_expands_nested_automation_definitions():
    from benchmark.models import Home, HomeEntity

    home = Home(
        home_id="nested",
        name="Nested",
        difficulty="basic",
        floors=(),
        areas=(),
        entities=(
            HomeEntity("script.routine", "Routine", "script"),
            HomeEntity("scene.evening", "Evening", "scene"),
            HomeEntity("light.one", "One", "light"),
            HomeEntity("light.two", "Two", "light"),
        ),
        metadata={
            "scripts": [
                {
                    "id": "routine",
                    "actions": [
                        {
                            "action": "light.turn_off",
                            "target": {"entity_id": ["light.one", "light.two"]},
                        },
                        {
                            "action": "scene.turn_on",
                            "target": {"entity_id": "scene.evening"},
                        },
                    ],
                }
            ],
            "scenes": [
                {
                    "id": "evening",
                    "entities": {"light.one": {"state": "on", "brightness": 35}},
                }
            ],
        },
    )

    assert expand_static_invocation(home, "script.routine") == (
        Operation(kind="action", entity_ids=("light.one", "light.two"), state="off"),
        Operation(
            kind="action",
            entity_ids=("light.one",),
            state="on",
            payload={"brightness": 35},
        ),
    )
    assert expand_static_invocation(home, "script.undefined") == (
        Operation(
            kind="action",
            entity_ids=("script.undefined",),
            payload={"invoked": True},
        ),
    )


def test_quoted_mute_values_are_normalized_to_booleans(corpus):
    scenario = next(
        scenario
        for scenario in corpus.scenarios
        if scenario.source == "studio/devices/media_player_main_lg_tv.yaml"
        and scenario.name == "media_player_main_lg_tv_mute"
    )

    assert scenario.expected[0].payload == {"is_volume_muted": True}


def test_service_grouping_does_not_change_observable_entity_effects():
    expected = (
        Operation(kind="action", entity_ids=("light.a",), state="off"),
        Operation(kind="action", entity_ids=("light.b",), state="off"),
    )
    actual = (Operation(kind="action", entity_ids=("light.a", "light.b"), state="off"),)

    assert compare_operations(expected, actual) == ((), ())


def test_list_identity_is_case_and_leading_article_insensitive():
    expected = (
        Operation(
            kind="shopping_list",
            payload={"item": "Orange Juice", "complete": True},
        ),
    )
    actual = (
        Operation(
            kind="shopping_list",
            payload={"item": "the orange juice", "complete": True},
        ),
    )

    assert compare_operations(expected, actual) == ((), ())
