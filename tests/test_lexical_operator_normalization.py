from __future__ import annotations

import pytest

from intent_bridge.core.voice import RouteDeclined, VoiceRequest
from intent_bridge.intent_engine.engine import DeterministicIntentEngine
from intent_bridge.intent_engine.models import (
    CatalogArea,
    CatalogEntity,
    CatalogSnapshot,
    ExecutionResult,
    IntentPlan,
    OhfIntentCall,
    PlannedIntent,
)
from intent_bridge.intent_engine.natural_language import (
    NaturalLanguageIntentPlanner,
    normalize_lexical_operator,
)


def _entity(entity_id: str, name: str, area: str, *aliases: str) -> CatalogEntity:
    return CatalogEntity(
        entity_id=entity_id,
        name=name,
        aliases=aliases,
        domain=entity_id.split(".", 1)[0],
        area_id=area,
    )


@pytest.fixture
def catalog() -> CatalogSnapshot:
    return CatalogSnapshot(
        areas=(
            CatalogArea("living", "Living Room"),
            CatalogArea("kitchen", "Kitchen"),
            CatalogArea("bedroom", "Bedroom"),
        ),
        entities=(
            _entity("cover.living_blinds", "Living Room Blinds", "living"),
            _entity("media_player.lg_tv", "LG TV", "living", "TV"),
            _entity("light.kitchen", "Kitchen Light", "kitchen"),
            _entity("fan.bedroom", "Bedroom Fan", "bedroom"),
            _entity("switch.wall", "Wall Switch", "living"),
        ),
    )


@pytest.mark.parametrize(
    ("text", "property_name", "value", "intent_name", "domain"),
    [
        ("Flip the living room blinds down", "position", "closed", "HassTurnOff", "cover"),
        ("Push the living room blinds down", "position", "closed", "HassTurnOff", "cover"),
        ("Move the living room blinds down", "position", "closed", "HassTurnOff", "cover"),
        ("Turn off the sound", "muted", True, "HassMediaPlayerMute", "media_player"),
        ("Silence the LG TV", "muted", True, "HassMediaPlayerMute", "media_player"),
        ("Restore audio", "muted", False, "HassMediaPlayerUnmute", "media_player"),
        ("Resume playback on the LG TV", "playback", "playing", "HassMediaUnpause", "media_player"),
        ("Restart the TV", "playback", "playing", "HassMediaUnpause", "media_player"),
        ("Switch on the LG TV", "power", True, "HassTurnOn", "media_player"),
        (
            "Could you switch on the LG TV please",
            "power",
            True,
            "HassTurnOn",
            "media_player",
        ),
        ("Set the LG TV to 27", "volume_level", 27, "HassSetVolume", "media_player"),
    ],
)
def test_surface_predicates_normalize_before_target_resolution(
    text, property_name, value, intent_name, domain
):
    operator = normalize_lexical_operator(text)

    assert operator is not None
    assert (operator.property, operator.operator, operator.value) == (
        property_name,
        "set",
        value,
    )
    assert (operator.intent_name, operator.domain) == (intent_name, domain)


@pytest.mark.parametrize(
    ("text", "intent_name", "entity_id"),
    [
        ("Flip the living room blinds down", "HassTurnOff", "cover.living_blinds"),
        ("Turn off the sound on the LG TV", "HassMediaPlayerMute", "media_player.lg_tv"),
        ("Restore audio on the LG TV", "HassMediaPlayerUnmute", "media_player.lg_tv"),
        ("Restart the TV", "HassMediaUnpause", "media_player.lg_tv"),
        ("Switch on the LG TV", "HassTurnOn", "media_player.lg_tv"),
        ("Switch on the wall switch", "HassTurnOn", "switch.wall"),
    ],
)
def test_normalized_operator_is_applied_to_compatible_resolved_target(
    catalog, text, intent_name, entity_id
):
    step = NaturalLanguageIntentPlanner().plan(text, catalog).steps[0]

    assert step.operation == intent_name
    assert step.entity_ids == (entity_id,)


@pytest.mark.parametrize(
    ("text", "property_name", "value"),
    [
        ("Flip the living room blinds down", "position", "closed"),
        ("Turn off the sound on the LG TV", "muted", True),
        ("Restore audio on the LG TV", "muted", False),
        ("Restart the TV", "playback", "playing"),
        ("Switch on the LG TV", "power", True),
    ],
)
def test_plan_retains_requested_semantics_not_transport_semantics(
    catalog, text, property_name, value
):
    effect = NaturalLanguageIntentPlanner().plan(text, catalog).steps[0].effect

    assert effect is not None
    assert (effect.property, effect.operator, effect.value) == (property_name, "set", value)
    assert effect.explicit_power_transition is (property_name == "power")


@pytest.mark.parametrize(
    ("text", "intent_name", "slot", "value"),
    [
        ("Set the kitchen light to 27", "HassLightSet", "brightness", 27),
        ("Move the living room blinds to 27", "HassSetPosition", "position", 27),
        ("Set the bedroom fan to 27", "HassFanSetSpeed", "percentage", 27),
        ("Set the LG TV to 27", "HassSetVolume", "volume_level", 27),
    ],
)
def test_bare_numeric_setter_infers_property_from_target_capability(
    catalog, text, intent_name, slot, value
):
    step = NaturalLanguageIntentPlanner().plan(text, catalog).steps[0]

    assert step.operation == intent_name
    assert step.call.data[slot] == value
    assert step.effect is not None
    assert step.effect.explicit_power_transition is False


def test_numeric_pronoun_infers_property_from_prior_target_domain(catalog):
    step = NaturalLanguageIntentPlanner().plan(
        "set it to 27", catalog, {"last_entity_ids": ["media_player.lg_tv"]}
    ).steps[0]

    assert step.operation == "HassSetVolume"
    assert step.call.data["volume_level"] == 27


def test_query_wording_is_not_rewritten_as_an_imperative():
    assert normalize_lexical_operator("is the sound off") is None


def test_capability_mismatch_never_executes_an_unrelated_target(catalog):
    plan = NaturalLanguageIntentPlanner().plan("silence the kitchen light", catalog)

    assert plan.steps == ()


def test_restart_non_media_target_is_not_normalized_as_playback(catalog):
    with pytest.raises(RouteDeclined):
        NaturalLanguageIntentPlanner().plan("restart the bedroom fan", catalog)


@pytest.mark.parametrize("text", ["unpause the timer", "resume the kitchen timer"])
def test_timer_resume_wording_is_left_for_the_timer_planner(text):
    assert normalize_lexical_operator(text) is None


class _CatalogProvider:
    def __init__(self, catalog):
        self._catalog = catalog

    def snapshot(self):
        return self._catalog


class _NoMatches:
    def recognize(self, text, catalog, origin_context=None):
        return ()


class _Executor:
    async def execute(self, call):
        return ExecutionResult(speech="done")


class _WrongQueryPlanner:
    def plan(self, text, catalog, origin_context=None):
        return IntentPlan(
            steps=(
                PlannedIntent(
                    OhfIntentCall("HassGetState", {"name": "Living Room Blinds"}),
                    ("cover.living_blinds",),
                ),
            )
        )


def test_normalized_operator_precedes_a_broad_preferred_query_planner(catalog):
    engine = DeterministicIntentEngine(
        _NoMatches(),
        _CatalogProvider(catalog),
        _Executor(),
        preferred_planner=_WrongQueryPlanner(),
        fallback_planner=NaturalLanguageIntentPlanner(),
    )

    step = engine.plan(VoiceRequest("Flip the living room blinds down", "test")).steps[0]

    assert step.operation == "HassTurnOff"
    assert step.effect is not None
    assert (step.effect.property, step.effect.value) == ("position", "closed")


@pytest.mark.parametrize(
    ("text", "domain", "expected_ids"),
    [
        (
            "Switch on the bathroom and kitchen lights please",
            "light",
            {"light.bathroom_mirror", "light.kitchen_ceiling"},
        ),
        (
            "Switch on the bathroom and living room fans",
            "fan",
            {"fan.bathroom_exhaust", "fan.living_ceiling"},
        ),
    ],
)
def test_imperative_switch_preserves_coordinated_target_domain(
    text, domain, expected_ids
):
    grouped_catalog = CatalogSnapshot(
        areas=(
            CatalogArea("bathroom", "Bathroom"),
            CatalogArea("kitchen", "Kitchen"),
            CatalogArea("living", "Living Room"),
        ),
        entities=(
            _entity("light.bathroom_mirror", "Mirror Light", "bathroom"),
            _entity("light.kitchen_ceiling", "Ceiling Light", "kitchen"),
            _entity("fan.bathroom_exhaust", "Exhaust Fan", "bathroom"),
            _entity("fan.living_ceiling", "Ceiling Fan", "living"),
        ),
    )
    operator = normalize_lexical_operator(text)
    assert operator is not None
    assert operator.domain == domain

    engine = DeterministicIntentEngine(
        _NoMatches(),
        _CatalogProvider(grouped_catalog),
        _Executor(),
        preferred_planner=_WrongQueryPlanner(),
        fallback_planner=NaturalLanguageIntentPlanner(),
    )
    plan = engine.plan(VoiceRequest(text, "test"))

    assert plan.response is None
    assert {entity_id for step in plan.steps for entity_id in step.entity_ids} == expected_ids
    assert {step.operation for step in plan.steps} == {"HassTurnOn"}
