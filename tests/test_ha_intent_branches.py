from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

import intent_bridge.home_assistant.intent_executor as executor_module
from intent_bridge.core.voice import RouteDeclined
from intent_bridge.home_assistant.intent_catalog import (
    HomeAssistantCatalogProvider,
    _clean_aliases,
    snapshot_from_client,
)
from intent_bridge.home_assistant.intent_executor import (
    HomeAssistantIntentExecutor,
    _error_detail,
    _speech_from_response,
)
from intent_bridge.intent_engine.models import CatalogMeasurement, CatalogSnapshot, OhfIntentCall


def test_clean_aliases_rejects_invalid_canonical_and_duplicate_values():
    aliases = _clean_aliases(
        [
            None,
            4,
            "",
            "!!!",
            " Kitchen ",
            "kitchen",
            " Cooking-Zone ",
            "cooking zone",
        ],
        "Kitchen",
    )
    assert aliases == ("Cooking-Zone",)


def test_catalog_snapshot_tolerates_sparse_and_malformed_cache_entries():
    class Client:
        areas = {
            "bad": [],
            "back_yard": {
                "name": " ",
                "alias": ["Garden", "garden", None],
                "floor_id": 0,
            },
            "kitchen": {"name": "Kitchen", "aliases": "invalid", "floor_id": ""},
        }
        floors = {
            "bad": [],
            "ground_floor": {"name": None, "aliases": "invalid"},
            "upper": {"name": "Upstairs", "aliases": ["Top", "upstairs"]},
        }
        states = {
            "invalid": {},
            "switch.invalid": [],
            "light.": {"state": None, "attributes": []},
            "sensor.temperature": {
                "state": 0,
                "attributes": {
                    "friendly_name": " Room Temperature ",
                    "aliases": "invalid",
                    "device_class": "",
                },
            },
            "switch.desk": {
                "state": "off",
                "attributes": {
                    "aliases": ["Table Switch", None],
                    "device_class": "outlet",
                },
            },
            "binary_sensor.hall_motion": {
                "state": "on",
                "attributes": {"friendly_name": "", "device_class": "motion"},
            },
        }
        entity_registry = {
            "sensor.temperature": {
                "en": "Temperature Sensor",
                "al": ["Thermometer"],
                "ai": "kitchen",
            },
            "switch.desk": {
                "en": "Desk Switch",
                "aliases": "invalid",
            },
            "binary_sensor.hall_motion": {"en": ""},
        }
        devices = {}

    snapshot = snapshot_from_client(Client())

    assert [area.area_id for area in snapshot.areas] == ["back_yard", "kitchen"]
    assert snapshot.areas[0].name == "Back Yard"
    assert snapshot.areas[0].aliases == ("Garden",)
    assert snapshot.areas[0].floor_id == "0"
    assert snapshot.areas[1].aliases == ()
    assert snapshot.areas[1].floor_id is None

    assert [floor.floor_id for floor in snapshot.floors] == ["ground_floor", "upper"]
    assert snapshot.floors[0].name == "Ground Floor"
    assert snapshot.floors[0].aliases == ()
    assert snapshot.floors[1].aliases == ("Top", "upper")

    entities = {entity.entity_id: entity for entity in snapshot.entities}
    assert set(entities) == {
        "light.",
        "sensor.temperature",
        "switch.desk",
        "binary_sensor.hall_motion",
    }
    assert entities["light."].name == "light."
    assert entities["light."].state is None
    assert entities["sensor.temperature"].name == "Room Temperature"
    assert entities["sensor.temperature"].aliases == (
        "temperature",
        "Temperature Sensor",
        "Thermometer",
    )
    assert entities["sensor.temperature"].area_id == "kitchen"
    assert entities["sensor.temperature"].device_class is None
    assert entities["sensor.temperature"].state == "0"
    assert entities["switch.desk"].name == "Desk Switch"
    assert entities["switch.desk"].aliases == ("desk", "Table Switch")
    assert entities["switch.desk"].device_class == "outlet"
    assert entities["binary_sensor.hall_motion"].name == "hall motion"


@pytest.mark.parametrize("floors", [None, []])
def test_catalog_snapshot_accepts_missing_or_non_mapping_floor_cache(floors):
    client = SimpleNamespace(states={}, entity_registry={}, devices={}, areas={})
    if floors is not None:
        client.floors = floors

    assert snapshot_from_client(client).floors == ()


def test_catalog_snapshot_extracts_domain_neutral_measurements():
    client = SimpleNamespace(
        states={
            "sensor.bedroom_temperature": {
                "state": "19.1",
                "attributes": {
                    "friendly_name": "Bedroom Temperature",
                    "device_class": "temperature",
                    "unit_of_measurement": "°C",
                },
            },
            "climate.bedroom_aircon": {
                "state": "off",
                "attributes": {
                    "friendly_name": "Bedroom Aircon",
                    "current_temperature": 20.0,
                    "temperature": 22.0,
                    "temperature_unit": "°C",
                },
            },
            "humidifier.bedroom": {
                "state": "on",
                "attributes": {
                    "friendly_name": "Bedroom Humidifier",
                    "current_humidity": 48,
                },
            },
            "sensor.unavailable_power": {
                "state": "unavailable",
                "attributes": {
                    "device_class": "power",
                    "unit_of_measurement": "W",
                },
            },
        },
        entity_registry={"sensor.bedroom_temperature": {"ec": 1}},
        devices={},
        areas={},
    )

    entities = {entity.entity_id: entity for entity in snapshot_from_client(client).entities}

    assert entities["sensor.bedroom_temperature"].measurements == (
        CatalogMeasurement("temperature", "19.1", "°C"),
    )
    assert entities["sensor.bedroom_temperature"].entity_category == "1"
    assert entities["climate.bedroom_aircon"].measurements == (
        CatalogMeasurement("temperature", "20.0", "°C", "current_temperature"),
    )
    assert entities["humidifier.bedroom"].measurements == (
        CatalogMeasurement("humidity", "48", "%", "current_humidity"),
    )
    assert entities["sensor.unavailable_power"].measurements == ()


def test_catalog_infers_temperature_for_numeric_sensor_without_device_class():
    client = SimpleNamespace(
        states={
            "sensor.studio_temperature": {
                "state": "20.5",
                "attributes": {
                    "friendly_name": "Studio Temperature Sensor",
                    "unit_of_measurement": "°C",
                },
            }
        },
        entity_registry={},
        devices={},
        areas={},
    )

    entity = snapshot_from_client(client).entities[0]

    assert entity.measurements == (
        CatalogMeasurement("temperature", "20.5", "°C", "inferred_state"),
    )


def test_catalog_classifies_voice_satellite_led_without_hiding_normal_lights():
    client = SimpleNamespace(
        states={
            "light.home_assistant_voice_0aaa03_led_ring": {
                "state": "off",
                "attributes": {"friendly_name": "Home Assistant Voice LED Ring"},
            },
            "light.office_light": {
                "state": "on",
                "attributes": {"friendly_name": "Office Light"},
            },
        },
        entity_registry={
            "light.home_assistant_voice_0aaa03_led_ring": {
                "di": "voice-ring",
                "ai": "office",
            },
            "light.office_light": {"di": "office-light", "ai": "office"},
        },
        devices={
            "voice-ring": {"name": "Home Assistant Voice LED Ring"},
            "office-light": {"name": "Office Ceiling Light"},
        },
        areas={"office": {"name": "Office"}},
    )

    entities = {entity.entity_id: entity for entity in snapshot_from_client(client).entities}

    assert entities["light.home_assistant_voice_0aaa03_led_ring"].is_indicator is True
    assert entities["light.office_light"].is_indicator is False


def test_catalog_provider_returns_empty_or_fresh_snapshot():
    unavailable = HomeAssistantCatalogProvider(lambda: None)
    assert unavailable.snapshot() == CatalogSnapshot()

    client = SimpleNamespace(
        states={},
        entity_registry={},
        devices={},
        areas={"office": {"name": "Office"}},
        floors={},
    )
    provider = HomeAssistantCatalogProvider(lambda: client)
    assert provider.snapshot().areas[0].name == "Office"


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (httpx.Response(500, text=" plain failure "), "plain failure"),
        (httpx.Response(502, content=b""), "HTTP 502"),
        (httpx.Response(400, json=[]), "HTTP 400"),
        (httpx.Response(400, json={"message": " direct "}), "direct"),
        (httpx.Response(400, json={"message": "", "error": " bad "}), "bad"),
        (
            httpx.Response(400, json={"error": {"message": " nested failure "}}),
            "nested failure",
        ),
        (
            httpx.Response(409, json={"detail": {"message": ""}}),
            "HTTP 409",
        ),
    ],
)
def test_error_detail_handles_text_structured_and_uninformative_bodies(response, expected):
    assert _error_detail(response) == expected


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ({"speech": {"plain": {"speech": " Spoken "}}}, "Spoken"),
        ({"response": {"speech": {"plain": {"speech": " Nested "}}}}, "Nested"),
        ({"speech": "invalid"}, ""),
        ({"speech": {"plain": "invalid"}}, ""),
        ({"speech": {"plain": {"speech": 5}}}, ""),
        ({"response": "invalid"}, ""),
        ({}, ""),
    ],
)
def test_speech_extraction_handles_nested_and_malformed_response_shapes(body, expected):
    assert _speech_from_response(body) == expected


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (
            {
                "entity_id": "climate.living_room",
                "state": "heat",
                "attributes": {
                    "friendly_name": "Living Room Thermostat",
                    "hvac_action": "heating",
                    "current_temperature": 20,
                    "temperature": 22,
                    "unit_of_measurement": "°C",
                },
            },
            "Living Room Thermostat is heating target 22 °C current 20 °C",
        ),
        (
            {
                "entity_id": "media_player.lounge",
                "state": "playing",
                "attributes": {
                    "friendly_name": "Lounge Speaker",
                    "media_title": "Jazz Playlist",
                    "media_artist": "Various Artists",
                    "volume_level": 0.58,
                },
            },
            "Lounge Speaker is playing Jazz Playlist by Various Artists",
        ),
        (
            {
                "entity_id": "light.kitchen",
                "state": "on",
                "attributes": {
                    "friendly_name": "Kitchen Light",
                    "brightness": 178,
                },
            },
            "Kitchen Light is on at 178%",
        ),
    ],
)
def test_summarise_intent_entity_state_supports_common_ha_domains(state, expected):
    assert executor_module._summarise_intent_entity_state(state) == expected


@pytest.mark.asyncio
async def test_intent_executor_declines_when_credentials_are_incomplete():
    for base_url, access_token in (("", "secret"), ("http://ha", " ")):
        executor = HomeAssistantIntentExecutor(base_url, access_token)
        with pytest.raises(RouteDeclined, match="not configured"):
            await executor.execute(OhfIntentCall("HassTurnOn", {}))


@pytest.mark.asyncio
async def test_intent_executor_owns_temporary_client_when_not_injected(monkeypatch):
    response = httpx.Response(
        200,
        json={"response": {"speech": {"plain": {"speech": " Handled "}}}},
    )

    class ManagedClient:
        entered = False
        exited = False
        post_arguments = None

        async def __aenter__(self):
            self.entered = True
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            self.exited = True

        async def post(self, url, **kwargs):
            self.post_arguments = (url, kwargs)
            return response

    managed = ManagedClient()
    monkeypatch.setattr(executor_module.httpx, "AsyncClient", lambda: managed)
    executor = HomeAssistantIntentExecutor("http://ha/", "secret", timeout=4.5)

    result = await executor.execute(OhfIntentCall("HassTurnOn", {"name": "Lamp"}))

    assert result.speech == "Handled"
    assert managed.entered is True
    assert managed.exited is True
    assert managed.post_arguments == (
        "http://ha/api/intent/handle",
        {
            "headers": {
                "Authorization": "Bearer secret",
                "Content-Type": "application/json",
            },
            "json": {"name": "HassTurnOn", "data": {"name": "Lamp"}},
            "timeout": 4.5,
        },
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx.Response(200, text="not-json"), "non-JSON"),
        (httpx.Response(200, json=["unexpected"]), "invalid intent response"),
    ],
)
async def test_intent_executor_rejects_invalid_success_bodies(response, message):
    async def handle(_request: httpx.Request) -> httpx.Response:
        return response

    transport = httpx.MockTransport(handle)
    async with httpx.AsyncClient(transport=transport) as client:
        executor = HomeAssistantIntentExecutor("http://ha", "secret", client=client)
        with pytest.raises(RuntimeError, match=message):
            await executor.execute(OhfIntentCall("HassTurnOn", {}))
