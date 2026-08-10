from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from intent_bridge.core.voice import RouteDeclined, VoiceRequest
from intent_bridge.intent_engine.engine import DeterministicIntentEngine
from intent_bridge.intent_engine.measurement import MeasurementIntentPlanner
from intent_bridge.intent_engine.models import (
    CatalogArea,
    CatalogEntity,
    CatalogMeasurement,
    CatalogSnapshot,
    ExecutionResult,
    IntentPlan,
    OhfIntentCall,
)
from intent_bridge.intent_engine.natural_language import NaturalLanguageIntentPlanner


def _measurement(
    quantity: str,
    value: str,
    unit: str,
    *,
    source: str = "state",
) -> CatalogMeasurement:
    return CatalogMeasurement(quantity, value, unit, source)


@pytest.fixture
def bedroom_measurements() -> CatalogSnapshot:
    return CatalogSnapshot(
        areas=(CatalogArea("bedroom", "Bedroom"),),
        entities=(
            CatalogEntity(
                "sensor.bedroom_climate_temperature",
                "Bedroom Climate Temperature",
                (),
                "sensor",
                "bedroom",
                "temperature",
                "19.1",
                (_measurement("temperature", "19.1", "°C"),),
            ),
            CatalogEntity(
                "sensor.bedroom_climate_humidity",
                "Bedroom Climate Humidity",
                (),
                "sensor",
                "bedroom",
                "humidity",
                "64.2",
                (_measurement("humidity", "64.2", "%"),),
            ),
            CatalogEntity(
                "climate.bedroom_aircon",
                "Bedroom Aircon",
                (),
                "climate",
                "bedroom",
                state="off",
                measurements=(
                    _measurement(
                        "temperature",
                        "20.0",
                        "°C",
                        source="current_temperature",
                    ),
                ),
            ),
        ),
    )


def test_area_temperature_prefers_direct_quantity_entity_over_embedded_attribute(
    bedroom_measurements,
):
    step = (
        MeasurementIntentPlanner()
        .plan(
            "whats the temperature in bedroom",
            bedroom_measurements,
        )
        .steps[0]
    )

    assert step.operation == "HassGetMeasurement"
    assert step.entity_ids == ("sensor.bedroom_climate_temperature",)
    assert step.reading is not None
    assert step.reading.measurement.value == "19.1"


def test_explicit_entity_evidence_can_select_cross_domain_attribute_reading(
    bedroom_measurements,
):
    step = (
        MeasurementIntentPlanner()
        .plan(
            "what is the bedroom aircon temperature",
            bedroom_measurements,
        )
        .steps[0]
    )

    assert step.entity_ids == ("climate.bedroom_aircon",)
    assert step.reading is not None
    assert step.reading.measurement.source == "current_temperature"


def test_multiple_requested_quantities_produce_ordered_direct_readings(
    bedroom_measurements,
):
    plan = MeasurementIntentPlanner().plan(
        "what are the temperature and humidity in bedroom",
        bedroom_measurements,
    )

    assert [step.reading.measurement.quantity for step in plan.steps if step.reading] == [
        "humidity",
        "temperature",
    ]
    assert [step.entity_ids for step in plan.steps] == [
        ("sensor.bedroom_climate_humidity",),
        ("sensor.bedroom_climate_temperature",),
    ]


def test_shared_measurement_query_resolves_each_named_sensor_independently():
    catalog = CatalogSnapshot(
        entities=(
            CatalogEntity(
                "sensor.indoor_temperature",
                "Indoor Sensor",
                (),
                "sensor",
                measurements=(_measurement("temperature", "20", "Â°C"),),
            ),
            CatalogEntity(
                "sensor.outdoor_temperature",
                "Outdoor Sensor",
                (),
                "sensor",
                measurements=(_measurement("temperature", "12", "Â°C"),),
            ),
        )
    )

    plan = MeasurementIntentPlanner().plan(
        "what are the temperature readings of the indoor sensor and outdoor sensor",
        catalog,
    )

    assert [step.entity_ids for step in plan.steps] == [
        ("sensor.indoor_temperature",),
        ("sensor.outdoor_temperature",),
    ]
    assert {step.effect.property for step in plan.steps if step.effect} == {"temperature"}


def test_shared_measurement_query_resolves_multiple_areas_independently():
    catalog = CatalogSnapshot(
        areas=(CatalogArea("kitchen", "Kitchen"), CatalogArea("bedroom", "Bedroom")),
        entities=(
            CatalogEntity(
                "sensor.kitchen_temperature",
                "Kitchen Temperature",
                (),
                "sensor",
                "kitchen",
                measurements=(_measurement("temperature", "21", "Â°C"),),
            ),
            CatalogEntity(
                "sensor.bedroom_temperature",
                "Bedroom Temperature",
                (),
                "sensor",
                "bedroom",
                measurements=(_measurement("temperature", "19", "Â°C"),),
            ),
        ),
    )

    plan = MeasurementIntentPlanner().plan(
        "what is the temperature in the kitchen and bedroom", catalog
    )

    assert [step.entity_ids for step in plan.steps] == [
        ("sensor.kitchen_temperature",),
        ("sensor.bedroom_temperature",),
    ]


@pytest.mark.parametrize(
    ("text", "entity_id", "quantity"),
    [
        ("how humid is the bedroom", "sensor.bedroom_climate_humidity", "humidity"),
        ("what is the front door battery level", "sensor.front_door_battery", "battery"),
        ("what is the washing machine power", "sensor.washer_power", "power"),
    ],
)
def test_measurement_resolution_is_quantity_based_across_domains(
    bedroom_measurements,
    text,
    entity_id,
    quantity,
):
    extra_entities = (
        CatalogEntity(
            "sensor.front_door_battery",
            "Front Door Battery",
            (),
            "sensor",
            "hall",
            "battery",
            "87",
            (_measurement("battery", "87", "%"),),
        ),
        CatalogEntity(
            "sensor.washer_power",
            "Washing Machine Power",
            (),
            "sensor",
            "utility",
            "power",
            "412",
            (_measurement("power", "412", "W"),),
        ),
    )
    catalog = CatalogSnapshot(
        entities=(*bedroom_measurements.entities, *extra_entities),
        areas=bedroom_measurements.areas,
    )

    step = MeasurementIntentPlanner().plan(text, catalog).steps[0]

    assert step.entity_ids == (entity_id,)
    assert step.reading is not None
    assert step.reading.measurement.quantity == quantity


def test_equal_measurement_candidates_remain_ambiguous():
    catalog = CatalogSnapshot(
        areas=(CatalogArea("bedroom", "Bedroom"),),
        entities=tuple(
            CatalogEntity(
                f"sensor.bedroom_temperature_{index}",
                f"Bedroom Temperature {index}",
                (),
                "sensor",
                "bedroom",
                "temperature",
                str(19 + index),
                (_measurement("temperature", str(19 + index), "°C"),),
            )
            for index in (1, 2)
        ),
    )

    plan = MeasurementIntentPlanner().plan("what is the bedroom temperature", catalog)

    assert plan.steps == ()
    assert "more than one possible target" in (plan.response or "")


def test_generic_area_reading_prefers_primary_entity_over_supporting_diagnostic(caplog):
    catalog = CatalogSnapshot(
        areas=(CatalogArea("guest_room", "Guest Room"),),
        entities=(
            CatalogEntity(
                "sensor.guestroom_climate_temperature",
                "Guestroom Climate Temperature",
                (),
                "sensor",
                "guest_room",
                "temperature",
                "20.5",
                (_measurement("temperature", "20.5", "°C"),),
            ),
            CatalogEntity(
                "sensor.guestroom_light_device_temperature",
                "Guestroom Light Temperature",
                (),
                "sensor",
                "guest_room",
                "temperature",
                "22",
                (_measurement("temperature", "22", "°C"),),
                "1",
            ),
        ),
    )

    with caplog.at_level("INFO", logger="intent_bridge.intent_engine.measurement"):
        generic = (
            MeasurementIntentPlanner()
            .plan(
                "whats the temperature in guestroom",
                catalog,
            )
            .steps[0]
        )
        explicit = (
            MeasurementIntentPlanner()
            .plan(
                "what is the guestroom light temperature",
                catalog,
            )
            .steps[0]
        )

    assert generic.entity_ids == ("sensor.guestroom_climate_temperature",)
    assert explicit.entity_ids == ("sensor.guestroom_light_device_temperature",)
    assert "MEASUREMENT RESOLUTION candidate" in caplog.text
    assert "secondary_category=1:-40" in caplog.text
    assert "MEASUREMENT RESOLUTION selected" in caplog.text
    assert "areas=('guest_room',)" in caplog.text


def test_unscoped_multiple_readings_decline_and_do_not_use_generic_alias_as_identity(caplog):
    catalog = CatalogSnapshot(
        entities=(
            CatalogEntity(
                "sensor.first_temperature",
                "Temperature",
                (),
                "sensor",
                measurements=(_measurement("temperature", "20", "°C"),),
            ),
            CatalogEntity(
                "sensor.second_temperature",
                "Temperature Sensor",
                (),
                "sensor",
                measurements=(_measurement("temperature", "21", "°C"),),
            ),
        ),
    )

    with caplog.at_level("INFO", logger="intent_bridge.intent_engine.measurement"):
        with pytest.raises(RouteDeclined, match="without target or topology evidence"):
            MeasurementIntentPlanner().plan("whats the temperature", catalog)

    assert "declined_unscoped" in caplog.text
    assert "allowing fallback" in caplog.text
    assert "exact_label" not in caplog.text


def test_measurement_planner_declines_mutating_requests(bedroom_measurements):
    with pytest.raises(RouteDeclined, match="read-only"):
        MeasurementIntentPlanner().plan(
            "set the bedroom temperature to 21",
            bedroom_measurements,
        )


def test_polite_modal_prefix_is_still_a_measurement_query(bedroom_measurements):
    plan = MeasurementIntentPlanner().plan(
        "Can you tell me the bedroom temperature?",
        bedroom_measurements,
    )

    assert plan.steps[0].entity_ids == ("sensor.bedroom_climate_temperature",)


@dataclass
class _Recognizer:
    calls: int = 0

    def recognize(self, text, catalog, origin_context=None):
        self.calls += 1
        return ()


@dataclass
class _CatalogProvider:
    catalog: CatalogSnapshot

    def snapshot(self):
        return self.catalog


@dataclass
class _Executor:
    calls: list[OhfIntentCall] = field(default_factory=list)

    async def execute(self, call: OhfIntentCall) -> ExecutionResult:
        self.calls.append(call)
        return ExecutionResult(speech="provider should not be called")


@dataclass
class _FallbackPlanner:
    calls: int = 0

    def plan(self, text, catalog, origin_context=None):
        self.calls += 1
        return IntentPlan(response="fallback handled the unscoped query")


@pytest.mark.asyncio
async def test_engine_answers_resolved_measurement_without_provider_reresolution(
    bedroom_measurements,
    caplog,
):
    recognizer = _Recognizer()
    executor = _Executor()
    engine = DeterministicIntentEngine(
        recognizer,
        _CatalogProvider(bedroom_measurements),
        executor,
        preferred_planner=MeasurementIntentPlanner(),
    )

    with caplog.at_level("INFO", logger="intent_bridge.intent_engine.engine"):
        response = await engine.handle(
            VoiceRequest("whats the temperature in bedroom", "measurement-test")
        )

    assert response == "Bedroom Climate Temperature is 19.1 °C"
    assert recognizer.calls == 0
    assert executor.calls == []
    assert "MEASUREMENT CACHE HIT" in caplog.text


@pytest.mark.asyncio
async def test_unscoped_measurement_ambiguity_reaches_engine_fallback():
    catalog = CatalogSnapshot(
        entities=tuple(
            CatalogEntity(
                f"sensor.temperature_{index}",
                "Temperature",
                (),
                "sensor",
                measurements=(_measurement("temperature", str(20 + index), "°C"),),
            )
            for index in (1, 2)
        )
    )
    recognizer = _Recognizer()
    fallback = _FallbackPlanner()
    executor = _Executor()
    engine = DeterministicIntentEngine(
        recognizer,
        _CatalogProvider(catalog),
        executor,
        preferred_planner=MeasurementIntentPlanner(),
        fallback_planner=fallback,
    )

    response = await engine.handle(
        VoiceRequest("whats the temperature", "measurement-fallback-test")
    )

    assert response == "fallback handled the unscoped query"
    assert recognizer.calls == 1
    assert fallback.calls == 1
    assert executor.calls == []


def test_unscoped_temperature_query_can_be_reinterpreted_as_weather():
    catalog = CatalogSnapshot(
        entities=(
            CatalogEntity(
                "sensor.bedroom_temperature",
                "Bedroom Temperature",
                (),
                "sensor",
                measurements=(_measurement("temperature", "20", "°C"),),
            ),
            CatalogEntity(
                "sensor.office_temperature",
                "Office Temperature",
                (),
                "sensor",
                measurements=(_measurement("temperature", "21", "°C"),),
            ),
            CatalogEntity("weather.home", "Home Weather", (), "weather"),
        )
    )

    step = NaturalLanguageIntentPlanner().plan("whats the temperature", catalog).steps[0]

    assert step.operation == "HassGetWeather"
    assert step.entity_ids == ("weather.home",)


def test_origin_area_scopes_an_unqualified_measurement_query():
    catalog = CatalogSnapshot(
        areas=(CatalogArea("bedroom", "Bedroom"), CatalogArea("office", "Office")),
        entities=(
            CatalogEntity(
                "sensor.bedroom_humidity",
                "Bedroom Humidity",
                (),
                "sensor",
                "bedroom",
                measurements=(_measurement("humidity", "50", "%"),),
            ),
            CatalogEntity(
                "sensor.office_humidity",
                "Office Humidity",
                (),
                "sensor",
                "office",
                measurements=(_measurement("humidity", "42", "%"),),
            ),
        ),
    )

    step = (
        MeasurementIntentPlanner()
        .plan(
            "what is the humidity",
            catalog,
            {"area_name": "Office"},
        )
        .steps[0]
    )

    assert step.entity_ids == ("sensor.office_humidity",)
