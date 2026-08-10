from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from intent_bridge.core.voice import RouteDeclined, RouteExecutionError, VoiceRequest
from intent_bridge.intent_engine.engine import DeterministicIntentEngine
from intent_bridge.intent_engine.grammar import load_intent_grammar
from intent_bridge.intent_engine.models import (
    CatalogArea,
    CatalogEntity,
    CatalogSnapshot,
    ExecutionResult,
    IntentMatch,
    OhfIntentCall,
    SlotValue,
)
from intent_bridge.intent_engine.recognizer import HassilIntentRecognizer
from intent_bridge.intent_engine.route import DeterministicVoiceRoute


def request(text: str, *, area_name: str | None = None) -> VoiceRequest:
    origin = {"area_name": area_name} if area_name else None
    return VoiceRequest(text=text, conversation_key="test", origin_context=origin)


def match(
    intent_name: str,
    **slots: str | tuple[str, dict[str, object]],
) -> IntentMatch:
    values = {}
    for name, raw_value in slots.items():
        if isinstance(raw_value, tuple):
            value, metadata = raw_value
        else:
            value, metadata = raw_value, {}
        values[name] = SlotValue(value=value, text=value, metadata=metadata)
    return IntentMatch(intent_name=intent_name, slots=values)


@dataclass
class StaticRecognizer:
    matches: tuple[IntentMatch, ...]

    def recognize(self, text, catalog, origin_context=None):
        return self.matches


@dataclass
class StaticCatalog:
    value: CatalogSnapshot

    def snapshot(self):
        return self.value


@dataclass
class RecordingExecutor:
    result: ExecutionResult = field(default_factory=lambda: ExecutionResult(speech="Done."))
    error: Exception | None = None
    calls: list[OhfIntentCall] = field(default_factory=list)

    async def execute(self, call: OhfIntentCall) -> ExecutionResult:
        self.calls.append(call)
        if self.error:
            raise self.error
        return self.result


def living_room_catalog(*, two_lights: bool = False) -> CatalogSnapshot:
    entities = [
        CatalogEntity(
            entity_id="light.living_room",
            name="Living Room Light",
            aliases=("Living Room Lights",),
            domain="light",
            area_id="living_room",
        )
    ]
    if two_lights:
        entities.append(
            CatalogEntity(
                entity_id="light.living_room_lamp",
                name="Living Room Lamp",
                aliases=(),
                domain="light",
                area_id="living_room",
            )
        )
    return CatalogSnapshot(
        entities=tuple(entities),
        areas=(CatalogArea(area_id="living_room", name="Living Room"),),
    )


@pytest.mark.asyncio
async def test_no_match_explicitly_declines_before_execution():
    executor = RecordingExecutor()
    engine = DeterministicIntentEngine(
        StaticRecognizer(()), StaticCatalog(CatalogSnapshot()), executor
    )

    with pytest.raises(RouteDeclined, match="No deterministic intent matched"):
        await engine.handle(request("tell me a joke"))
    assert executor.calls == []


@pytest.mark.asyncio
async def test_name_and_area_candidates_with_same_effective_target_execute_once():
    candidates = (
        match(
            "HassTurnOff",
            name=("Living Room Light", {"entity_id": "light.living_room"}),
        ),
        match(
            "HassTurnOff",
            area=("Living Room", {"area_id": "living_room"}),
            domain="light",
        ),
    )
    executor = RecordingExecutor()
    engine = DeterministicIntentEngine(
        StaticRecognizer(candidates), StaticCatalog(living_room_catalog()), executor
    )

    assert await engine.handle(request("turn the living room lights off")) == "Done."
    assert len(executor.calls) == 1
    assert executor.calls[0].intent_name == "HassTurnOff"


@pytest.mark.asyncio
async def test_incompatible_effective_targets_are_clarified_without_execution():
    candidates = (
        match(
            "HassTurnOff",
            name=("Living Room Light", {"entity_id": "light.living_room"}),
        ),
        match(
            "HassTurnOff",
            area=("Living Room", {"area_id": "living_room"}),
            domain="light",
        ),
    )
    executor = RecordingExecutor()
    engine = DeterministicIntentEngine(
        StaticRecognizer(candidates),
        StaticCatalog(living_room_catalog(two_lights=True)),
        executor,
    )

    speech = await engine.handle(request("turn the living room lights off"))
    assert "more than one possible target" in speech
    assert executor.calls == []


@pytest.mark.asyncio
async def test_origin_area_is_added_to_context_relative_domain_intent():
    executor = RecordingExecutor()
    engine = DeterministicIntentEngine(
        StaticRecognizer((match("HassTurnOn", domain="light"),)),
        StaticCatalog(living_room_catalog()),
        executor,
    )

    await engine.handle(request("turn on the lights", area_name="Living Room"))
    assert executor.calls[0].data == {"domain": "light", "area": "Living Room"}


@pytest.mark.asyncio
async def test_missing_origin_for_context_relative_target_returns_clarification():
    executor = RecordingExecutor()
    engine = DeterministicIntentEngine(
        StaticRecognizer((match("HassTurnOn", domain="light"),)),
        StaticCatalog(living_room_catalog()),
        executor,
    )

    assert await engine.handle(request("turn on the lights")) == "Which area should I use?"
    assert executor.calls == []


@pytest.mark.asyncio
async def test_execution_failure_is_marked_non_fallback():
    executor = RecordingExecutor(error=RuntimeError("HA unavailable"))
    engine = DeterministicIntentEngine(
        StaticRecognizer(
            (
                match(
                    "HassTurnOn",
                    name=("Living Room Light", {"entity_id": "light.living_room"}),
                ),
            )
        ),
        StaticCatalog(living_room_catalog()),
        executor,
    )

    with pytest.raises(RouteExecutionError, match="HA unavailable"):
        await engine.handle(request("turn the living room light on"))


@pytest.mark.asyncio
async def test_voice_route_delegates_full_request_to_engine():
    class Engine:
        seen = None

        async def handle(self, voice_request):
            self.seen = voice_request
            return "Handled."

    engine = Engine()
    route = DeterministicVoiceRoute(engine)
    voice_request = request("turn it off", area_name="Living Room")

    assert route.name == "ohf-hassil"
    assert await route.handle(voice_request) == "Handled."
    assert engine.seen is voice_request


def test_real_ohf_recognizer_uses_dynamic_names_areas_and_plural_grammar(tmp_path):
    grammar = load_intent_grammar(language="en", custom_sentences_path=tmp_path / "missing")
    recognizer = HassilIntentRecognizer(grammar)
    catalog = living_room_catalog()

    singular = recognizer.recognize("turn the living room light off", catalog)
    plural = recognizer.recognize("turn the living room lights off", catalog)

    assert {candidate.intent_name for candidate in singular} == {"HassTurnOff"}
    assert {candidate.intent_name for candidate in plural} == {"HassTurnOff"}
    assert any(candidate.slots.get("area") for candidate in singular)
    assert any(candidate.slots.get("area") for candidate in plural)
