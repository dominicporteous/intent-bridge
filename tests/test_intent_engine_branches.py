from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

import intent_bridge.intent_engine.engine as engine_module
import intent_bridge.intent_engine.recognizer as recognizer_module
import intent_bridge.intent_engine.resolution as resolution_module
from intent_bridge.core.voice import RouteDeclined, VoiceRequest
from intent_bridge.intent_engine.engine import DeterministicIntentEngine
from intent_bridge.intent_engine.grammar import LoadedIntentGrammar
from intent_bridge.intent_engine.models import (
    CatalogArea,
    CatalogEntity,
    CatalogFloor,
    CatalogSnapshot,
    ExecutionResult,
    IntentMatch,
    IntentPlan,
    OhfIntentCall,
    PlannedIntent,
    SlotValue,
)
from intent_bridge.intent_engine.outcomes import AmbiguousTarget, IncompleteCompound
from intent_bridge.intent_engine.recognizer import HassilIntentRecognizer


def _slot(value, *, metadata=None, text: str | None = None) -> SlotValue:
    return SlotValue(
        value=value,
        text=str(value) if text is None else text,
        metadata={} if metadata is None else metadata,
    )


def _match(intent_name: str, **slots: SlotValue) -> IntentMatch:
    return IntentMatch(intent_name=intent_name, slots=slots)


def _catalog() -> CatalogSnapshot:
    return CatalogSnapshot(
        entities=(
            CatalogEntity(
                entity_id="light.kitchen",
                name="Kitchen Light",
                aliases=("Cooking Lamp",),
                domain="light",
                area_id="kitchen",
                device_class="light",
            ),
            CatalogEntity(
                entity_id="switch.kitchen",
                name="Kitchen Switch",
                aliases=(),
                domain="switch",
                area_id="kitchen",
                device_class="outlet",
            ),
            CatalogEntity(
                entity_id="light.office",
                name="Office Light",
                aliases=(),
                domain="light",
                area_id="office",
            ),
            CatalogEntity(
                entity_id="binary_sensor.hall_motion",
                name="Hall Motion",
                aliases=(),
                domain="binary_sensor",
                area_id="hall",
                device_class="motion",
            ),
            CatalogEntity(
                entity_id="sensor.orphan",
                name="Orphan Sensor",
                aliases=(),
                domain="sensor",
            ),
        ),
        areas=(
            CatalogArea(
                area_id="kitchen",
                name="Kitchen",
                aliases=("Galley",),
                floor_id="ground",
            ),
            CatalogArea(area_id="office", name="Office", floor_id="ground"),
            CatalogArea(area_id="hall", name="Hall", floor_id="upper"),
            CatalogArea(area_id="shared_a", name="Shared", floor_id="upper"),
            CatalogArea(area_id="shared_b", name="Shared", floor_id="upper"),
        ),
        floors=(
            CatalogFloor(floor_id="ground", name="Ground Floor", aliases=("Downstairs",)),
            CatalogFloor(floor_id="upper", name="Upper Floor", aliases=("Upstairs",)),
            CatalogFloor(floor_id="duplicate_a", name="Duplicate"),
            CatalogFloor(floor_id="duplicate_b", name="Duplicate"),
        ),
    )


def test_engine_origin_and_context_helpers_cover_explicit_and_relative_targets():
    assert engine_module._origin_area(None) is None
    assert engine_module._origin_area({"area_name": 12, "area_id": " kitchen "}) == "kitchen"
    assert engine_module._origin_area({"area_name": " ", "area_id": None}) is None

    relative = _match("HassTurnOn", domain=_slot("light"))
    explicit = _match(
        "HassTurnOn",
        domain=_slot("light"),
        area=_slot("Kitchen"),
    )
    unrelated = _match("HassNevermind", domain=_slot("light"))

    assert engine_module._requires_origin_area(relative) is True
    assert engine_module._requires_origin_area(explicit) is False
    assert engine_module._requires_origin_area(unrelated) is False
    assert engine_module._call_for_match(relative, {"area_id": " kitchen "}).data == {
        "domain": "light",
        "area": "kitchen",
    }
    assert engine_module._call_for_match(relative, {"area_name": 5}).data == {"domain": "light"}
    assert engine_module._call_for_match(explicit, None).data["area"] == "Kitchen"


@dataclass
class _Recognizer:
    matches: tuple[IntentMatch, ...]
    calls: int = 0

    def recognize(self, text, catalog, origin_context=None):
        self.calls += 1
        return self.matches


@dataclass
class _CatalogProvider:
    catalog: CatalogSnapshot

    def snapshot(self):
        return self.catalog


@dataclass
class _Executor:
    result: ExecutionResult | None = None
    error: Exception | None = None

    async def execute(self, call: OhfIntentCall) -> ExecutionResult:
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


@dataclass
class _Planner:
    result: IntentPlan | None = None
    decline: bool = False
    calls: int = 0

    def plan(self, text, catalog, origin_context=None):
        self.calls += 1
        if self.decline:
            raise RouteDeclined("not handled")
        assert self.result is not None
        return self.result


@dataclass
class _OutcomePlanner:
    outcome: object
    calls: int = 0

    def resolve(self, text, catalog, origin_context=None):
        self.calls += 1
        return self.outcome


def _single_step_plan() -> IntentPlan:
    return IntentPlan(
        steps=(
            PlannedIntent(
                OhfIntentCall("HassTurnOn", {"name": "Kitchen Light"}),
                ("light.kitchen",),
            ),
        )
    )


def test_engine_preferred_planner_precedes_recognition_and_decline_continues_to_ohf():
    request = VoiceRequest("turn on the kitchen light", "preferred")
    handled = _Planner(_single_step_plan())
    unused_recognizer = _Recognizer(())
    handled_engine = DeterministicIntentEngine(
        unused_recognizer,
        _CatalogProvider(_catalog()),
        _Executor(),
        preferred_planner=handled,
    )

    assert handled_engine.plan(request) == handled.result
    assert handled.calls == 1
    assert unused_recognizer.calls == 0

    decline = _Planner(decline=True)
    official = _Recognizer(
        (
            _match(
                "HassTurnOn",
                name=_slot("Kitchen Light", metadata={"entity_id": "light.kitchen"}),
            ),
        )
    )
    continuing_engine = DeterministicIntentEngine(
        official,
        _CatalogProvider(_catalog()),
        _Executor(),
        preferred_planner=decline,
    )

    assert continuing_engine.plan(request).steps[0].entity_ids == ("light.kitchen",)
    assert decline.calls == 1
    assert official.calls == 1


def test_engine_routes_incomplete_structure_onward_but_stops_for_target_ambiguity():
    fallback = _Planner(_single_step_plan())
    incomplete = _OutcomePlanner(IncompleteCompound(("set mystery",)))
    engine = DeterministicIntentEngine(
        _Recognizer(()),
        _CatalogProvider(_catalog()),
        _Executor(),
        preferred_planner=incomplete,
        fallback_planner=fallback,
    )

    assert engine.plan(VoiceRequest("compound", "typed-failure")) == fallback.result
    assert fallback.calls == 1

    fallback.calls = 0
    ambiguous = _OutcomePlanner(
        AmbiguousTarget(("light.kitchen", "light.counter"), "device name")
    )
    ambiguous_engine = DeterministicIntentEngine(
        _Recognizer(()),
        _CatalogProvider(_catalog()),
        _Executor(),
        preferred_planner=ambiguous,
        fallback_planner=fallback,
    )

    plan = ambiguous_engine.plan(VoiceRequest("the light", "typed-ambiguity"))
    assert "specific" in (plan.response or "").casefold()
    assert fallback.calls == 0


def test_engine_filters_unresolved_entity_candidates_and_uses_fallback_for_ambiguity():
    catalog = _catalog()
    request = VoiceRequest("turn on the kitchen light", "resolution-filter")
    unresolved = _match(
        "HassTurnOn",
        area=_slot("Missing"),
        domain=_slot("light"),
    )
    resolved = _match(
        "HassTurnOn",
        name=_slot("Kitchen Light", metadata={"entity_id": "light.kitchen"}),
    )

    mixed_engine = DeterministicIntentEngine(
        _Recognizer((unresolved, resolved)),
        _CatalogProvider(catalog),
        _Executor(),
    )
    assert mixed_engine.plan(request).steps[0].entity_ids == ("light.kitchen",)

    fallback = _Planner(_single_step_plan())
    unresolved_engine = DeterministicIntentEngine(
        _Recognizer((unresolved,)),
        _CatalogProvider(catalog),
        _Executor(),
        fallback_planner=fallback,
    )
    assert unresolved_engine.plan(request) == fallback.result

    ambiguous_engine = DeterministicIntentEngine(
        _Recognizer((resolved, _match("HassTurnOff", name=_slot("Kitchen Light")))),
        _CatalogProvider(catalog),
        _Executor(),
        fallback_planner=fallback,
    )
    assert ambiguous_engine.plan(request) == fallback.result
    assert fallback.calls == 2

    declining_fallback = _Planner(decline=True)
    retained_ambiguity = DeterministicIntentEngine(
        _Recognizer((resolved, _match("HassTurnOff", name=_slot("Kitchen Light")))),
        _CatalogProvider(catalog),
        _Executor(),
        fallback_planner=declining_fallback,
    ).plan(request)
    assert retained_ambiguity.response is not None
    assert retained_ambiguity.steps == ()

    with pytest.raises(RouteDeclined, match="catalog target"):
        DeterministicIntentEngine(
            _Recognizer((unresolved,)),
            _CatalogProvider(catalog),
            _Executor(),
        ).plan(request)


@pytest.mark.parametrize("intent_name", ["HassStartTimer", "HassShoppingListAddItem"])
def test_engine_keeps_valid_targetless_official_intents(intent_name):
    engine = DeterministicIntentEngine(
        _Recognizer((_match(intent_name),)),
        _CatalogProvider(CatalogSnapshot()),
        _Executor(),
    )

    plan = engine.plan(VoiceRequest("targetless", "targetless"))

    assert plan.steps[0].operation == intent_name
    assert plan.steps[0].entity_ids == ()


@pytest.mark.asyncio
async def test_engine_preserves_executor_decline_and_uses_default_for_blank_speech():
    candidate = _match("HassNevermind")
    request = VoiceRequest(text="never mind", conversation_key="branches")
    decline = RouteDeclined("handler unavailable")
    declining_engine = DeterministicIntentEngine(
        _Recognizer((candidate,)),
        _CatalogProvider(CatalogSnapshot()),
        _Executor(error=decline),
    )

    with pytest.raises(RouteDeclined) as raised:
        await declining_engine.handle(request)
    assert raised.value is decline

    defaulting_engine = DeterministicIntentEngine(
        _Recognizer((candidate,)),
        _CatalogProvider(CatalogSnapshot()),
        _Executor(result=ExecutionResult(speech=" \t ")),
        default_response="All done.",
    )
    assert await defaulting_engine.handle(request) == "All done."


def test_recognizer_alias_and_context_helpers_handle_sparse_topology():
    aliases = recognizer_module._deduplicated_aliases(
        " Kitchen-Lamp ",
        "kitchen lamp",
        "",
        "!!!",
        None,  # type: ignore[arg-type]
    )
    assert aliases == ("Kitchen-Lamp",)

    catalog = _catalog()
    assert recognizer_module._area_context(catalog, None) is None
    assert recognizer_module._area_context(catalog, "missing") is None
    assert recognizer_module._area_context(catalog, "kitchen") == {
        "value": "Kitchen",
        "text": "Kitchen",
        "metadata": {"area_id": "kitchen"},
    }

    runtime_lists = recognizer_module._runtime_slot_lists(catalog)
    assert set(runtime_lists) == {"name", "area", "floor"}

    assert recognizer_module._intent_context(catalog, None) == {}
    assert (
        recognizer_module._intent_context(catalog, {"area_id": "kitchen"})["area"]["value"]
        == "Kitchen"
    )
    assert recognizer_module._intent_context(catalog, {"area_id": 7, "area_name": "Kitchen"})[
        "area"
    ]["metadata"] == {"area_id": "kitchen"}
    assert recognizer_module._intent_context(catalog, {"area_name": "Shared"}) == {}
    assert (
        recognizer_module._intent_context(catalog, {"area_id": "unknown", "area_name": "Kitchen"})
        == {}
    )


def _result(intent_name: str, value: str, *, metadata=None):
    return SimpleNamespace(
        intent=SimpleNamespace(name=intent_name),
        entities={
            "name": SimpleNamespace(value=value, text=value, metadata=metadata),
        },
        response="default",
        context={},
        intent_metadata=None,
    )


def test_recognizer_deduplicates_results_and_honours_candidate_limit(monkeypatch):
    first = _result("HassTurnOn", "Kitchen Light")
    second = _result("HassTurnOff", "Kitchen Light", metadata={"entity_id": "light.kitchen"})
    emitted = [first, first, second]
    calls = []

    def fake_recognize_all(*args, **kwargs):
        calls.append((args, kwargs))
        return iter(emitted)

    monkeypatch.setattr(recognizer_module, "recognize_all", fake_recognize_all)
    grammar = LoadedIntentGrammar(
        intents=object(),
        language="en",
        custom_files=(),
        provenance=(),
    )

    recognizer = HassilIntentRecognizer(grammar, max_candidates=8)
    matches = recognizer.recognize("lights", CatalogSnapshot(), {"area_name": "Missing"})
    assert [match.intent_name for match in matches] == ["HassTurnOn", "HassTurnOff"]
    assert matches[0].slots["name"].metadata == {}
    assert calls[-1][1]["language"] == "en"

    limited = HassilIntentRecognizer(grammar, max_candidates=0)
    emitted[:] = [first, second]
    assert [match.intent_name for match in limited.recognize("lights", CatalogSnapshot())] == [
        "HassTurnOn"
    ]


def test_resolution_lookup_helpers_cover_metadata_aliases_and_ambiguity():
    catalog = _catalog()
    no_slots = _match("HassTurnOn")
    assert resolution_module._area_id(no_slots, catalog) is None
    assert resolution_module._floor_id(no_slots, catalog) is None
    assert resolution_module._matching_named_entities(no_slots, catalog) == []

    assert (
        resolution_module._area_id(
            _match("HassTurnOn", area=_slot("ignored", metadata={"area_id": "kitchen"})),
            catalog,
        )
        == "kitchen"
    )
    assert (
        resolution_module._area_id(_match("HassTurnOn", area=_slot("Galley")), catalog) == "kitchen"
    )
    assert resolution_module._area_id(_match("HassTurnOn", area=_slot("Shared")), catalog) is None

    assert (
        resolution_module._floor_id(
            _match("HassTurnOn", floor=_slot("ignored", metadata={"floor_id": "ground"})),
            catalog,
        )
        == "ground"
    )
    assert (
        resolution_module._floor_id(_match("HassTurnOn", floor=_slot("Downstairs")), catalog)
        == "ground"
    )
    assert (
        resolution_module._floor_id(_match("HassTurnOn", floor=_slot("Duplicate")), catalog) is None
    )

    metadata_match = _match(
        "HassTurnOn", name=_slot("ignored", metadata={"entity_id": "light.kitchen"})
    )
    assert [
        entity.entity_id
        for entity in resolution_module._matching_named_entities(metadata_match, catalog)
    ] == ["light.kitchen"]
    missing_metadata_match = _match(
        "HassTurnOn", name=_slot("ignored", metadata={"entity_id": "light.missing"})
    )
    assert resolution_module._matching_named_entities(missing_metadata_match, catalog) == []
    alias_match = _match("HassTurnOn", name=_slot("Cooking Lamp"))
    assert [
        entity.entity_id
        for entity in resolution_module._matching_named_entities(alias_match, catalog)
    ] == ["light.kitchen"]


def test_resolution_does_not_trust_runtime_metadata_for_duplicate_display_names():
    catalog = CatalogSnapshot(
        entities=(
            CatalogEntity("cover.garage_left", "Garage Door", (), "cover", "garage"),
            CatalogEntity("cover.garage_right", "Garage Door", (), "cover", "garage"),
        ),
        areas=(CatalogArea("garage", "Garage"),),
    )
    match = _match(
        "HassTurnOn",
        name=_slot("Garage Door", metadata={"entity_id": "cover.garage_left"}),
    )

    named = resolution_module._matching_named_entities(match, catalog)
    resolved = resolution_module.resolve_candidate(match, catalog)

    assert {entity.entity_id for entity in named} == {
        "cover.garage_left",
        "cover.garage_right",
    }
    assert resolved.entity_ids == frozenset()


def test_resolution_builds_equivalent_keys_for_all_target_shapes():
    catalog = _catalog()

    named = resolution_module.resolve_candidate(
        _match(
            "HassLightSet",
            name=_slot("Cooking Lamp"),
            brightness=_slot({"levels": [20, 30]}),
        ),
        catalog,
    )
    assert named.entity_ids == frozenset({"light.kitchen"})
    assert named.specificity == 3
    assert named.semantic_key[2] == (("brightness", (("levels", (20, 30)),)),)

    area = resolution_module.resolve_candidate(
        _match(
            "HassTurnOff",
            area=_slot("Kitchen", metadata={"area_id": "kitchen"}),
            domain=_slot("LIGHT"),
        ),
        catalog,
    )
    assert area.entity_ids == frozenset({"light.kitchen"})
    assert area.specificity == 2

    floor = resolution_module.resolve_candidate(
        _match("HassTurnOff", floor=_slot("Downstairs"), domain=_slot("light")),
        catalog,
    )
    assert floor.entity_ids == frozenset({"light.kitchen", "light.office"})
    assert floor.specificity == 1

    from_context = resolution_module.resolve_candidate(
        IntentMatch(intent_name="HassTurnOn", context={"domain": "light"}),
        catalog,
        {"area_id": "kitchen"},
    )
    assert from_context.entity_ids == frozenset({"light.kitchen"})
    assert from_context.specificity == 2

    from_context_name = resolution_module.resolve_candidate(
        _match("HassTurnOn", domain=_slot("light")),
        catalog,
        {"area_name": "Office"},
    )
    assert from_context_name.entity_ids == frozenset({"light.office"})

    by_device_class = resolution_module.resolve_candidate(
        _match("HassGetState", device_class=_slot("motion")),
        catalog,
    )
    assert by_device_class.entity_ids == frozenset({"binary_sensor.hall_motion"})
    assert by_device_class.specificity == 0

    untargeted = resolution_module.resolve_candidate(_match("HassNevermind"), catalog)
    assert untargeted.entity_ids == frozenset()
    assert untargeted.semantic_key == ("HassNevermind", (), ())

    unresolved = resolution_module.resolve_candidate(
        _match("HassTurnOn", area=_slot("Nowhere"), domain=_slot("light")),
        catalog,
    )
    assert unresolved.entity_ids == frozenset()
    assert unresolved.semantic_key[1] == (("area", "Nowhere"), ("domain", "light"))

    unresolved_name = resolution_module.resolve_candidate(
        _match("HassTurnOn", name=_slot("Missing Lamp"), domain=_slot("light")),
        catalog,
    )
    unresolved_floor = resolution_module.resolve_candidate(
        _match("HassTurnOn", floor=_slot("Missing Floor"), domain=_slot("light")),
        catalog,
    )
    unresolved_origin = resolution_module.resolve_candidate(
        _match("HassTurnOn", domain=_slot("light")),
        catalog,
        {"area_name": "Missing Area"},
    )
    assert unresolved_name.entity_ids == frozenset()
    assert unresolved_floor.entity_ids == frozenset()
    assert unresolved_origin.entity_ids == frozenset()


def test_resolution_promotes_a_unique_surface_target_over_a_domain_group():
    catalog = CatalogSnapshot(
        entities=(
            CatalogEntity("light.living_light", "Living Room Light", (), "light", "living"),
            CatalogEntity("light.living_lamp", "Living Room Lamp", (), "light", "living"),
        ),
        areas=(CatalogArea("living", "Living Room"),),
    )
    domain_match = _match("HassTurnOff", domain=_slot("light"))

    lamp = resolution_module.resolve_candidate(
        domain_match,
        catalog,
        {"area_id": "living"},
        text="Turn the lamp off",
    )
    generic = resolution_module.resolve_candidate(
        domain_match,
        catalog,
        {"area_id": "living"},
        text="Turn the lights off",
    )

    assert lamp.entity_ids == frozenset({"light.living_lamp"})
    assert lamp.match.slots["name"].value == "Living Room Lamp"
    assert lamp.match.slots["entity_id"].value == "light.living_lamp"
    assert lamp.specificity == 3
    assert generic.entity_ids == frozenset({"light.living_light", "light.living_lamp"})
    assert "name" not in generic.match.slots


def test_freeze_handles_mapping_sequences_sets_and_scalars():
    frozen = resolution_module._freeze({"b": {2, 1}, "a": ["x", ("y",)]})
    assert frozen[0] == ("a", ("x", ("y",)))
    assert frozen[1][0] == "b"
    assert set(frozen[1][1]) == {1, 2}
    assert resolution_module._freeze("plain") == "plain"


def test_resolution_infers_known_cover_class_from_name_without_broadening_unknowns():
    catalog = CatalogSnapshot(
        entities=(
            CatalogEntity(
                entity_id="cover.living_blinds",
                name="Living Room Blinds",
                aliases=("Lounge Drapes",),
                domain="cover",
                area_id="living",
            ),
            CatalogEntity(
                entity_id="cover.living_skylight",
                name="Living Room Skylight",
                aliases=(),
                domain="cover",
                area_id="living",
            ),
        ),
        areas=(CatalogArea("living", "Living Room"),),
    )

    blind = resolution_module.resolve_candidate(
        _match(
            "HassTurnOn",
            area=_slot("Living Room"),
            domain=_slot("cover"),
            device_class=_slot("blind"),
        ),
        catalog,
    )
    curtain_alias = resolution_module.resolve_candidate(
        _match(
            "HassTurnOn",
            area=_slot("Living Room"),
            domain=_slot("cover"),
            device_class=_slot("curtain"),
        ),
        catalog,
    )
    unknown = resolution_module.resolve_candidate(
        _match(
            "HassTurnOn",
            area=_slot("Living Room"),
            domain=_slot("cover"),
            device_class=_slot("shade"),
        ),
        catalog,
    )

    assert blind.entity_ids == frozenset({"cover.living_blinds"})
    assert curtain_alias.entity_ids == frozenset({"cover.living_blinds"})
    assert unknown.entity_ids == frozenset()


def test_resolution_infers_domains_from_coherent_official_intent_families():
    catalog = CatalogSnapshot(
        entities=(
            CatalogEntity("fan.office", "Office Fan", (), "fan", "office"),
            CatalogEntity("light.office", "Office Light", (), "light", "office"),
            CatalogEntity("climate.office", "Office Thermostat", (), "climate", "office"),
            CatalogEntity("media_player.office", "Office Speaker", (), "media_player", "office"),
            CatalogEntity("vacuum.downstairs", "Downstairs Vacuum", (), "vacuum", "office"),
        ),
        areas=(CatalogArea("office", "Office"),),
    )

    expected = {
        "HassFanSetSpeed": "fan.office",
        "HassLightSet": "light.office",
        "HassClimateSetTemperature": "climate.office",
        "HassSetVolume": "media_player.office",
        "HassVacuumCleanArea": "vacuum.downstairs",
    }
    for intent_name, entity_id in expected.items():
        resolved = resolution_module.resolve_candidate(
            _match(intent_name, area=_slot("Office")),
            catalog,
        )
        assert resolved.entity_ids == frozenset({entity_id})

    explicit_context = resolution_module.resolve_candidate(
        IntentMatch(intent_name="HassFanSetSpeed", context={"domain": "light"}),
        catalog,
        {"area_id": "office"},
    )
    explicit_slot = resolution_module.resolve_candidate(
        _match("HassFanSetSpeed", area=_slot("Office"), domain=_slot("climate")),
        catalog,
    )
    assert explicit_context.entity_ids == frozenset({"light.office"})
    assert explicit_slot.entity_ids == frozenset({"climate.office"})


def test_area_light_resolution_excludes_indicators_but_explicit_names_retain_them():
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

    area = resolution_module.resolve_candidate(
        _match("HassTurnOff", area=_slot("Office"), domain=_slot("light")),
        catalog,
    )
    named = resolution_module.resolve_candidate(
        _match("HassTurnOn", name=_slot("Voice LED Ring")),
        catalog,
    )

    assert area.entity_ids == frozenset({"light.office"})
    assert named.entity_ids == frozenset({"light.voice_ring"})
