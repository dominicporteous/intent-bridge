from __future__ import annotations

import json

import httpx
import pytest

import intent_bridge.home_assistant.intent_executor as executor_module
from intent_bridge.config import settings
from intent_bridge.home_assistant.intent_catalog import snapshot_from_client
from intent_bridge.home_assistant.intent_executor import HomeAssistantIntentExecutor
from intent_bridge.intent_engine.models import IntentMatch, OhfIntentCall, SlotValue
from intent_bridge.intent_engine.resolution import resolve_candidate
from intent_bridge.runtime.dependencies import runtime
from intent_bridge.runtime.execution import _reset_voice_tool_run_state


def test_catalog_snapshot_normalizes_state_and_registry_aliases():
    class Client:
        states = {
            "light.kitchen_ceiling": {
                "entity_id": "light.kitchen_ceiling",
                "state": "on",
                "attributes": {
                    "friendly_name": "Kitchen Ceiling Lights",
                    "device_class": "light",
                },
            }
        }
        entity_registry = {
            "light.kitchen_ceiling": {
                "ei": "light.kitchen_ceiling",
                "en": "Ceiling Light",
                "ai": "kitchen",
                "di": "kitchen-lamp",
            }
        }
        devices = {"kitchen-lamp": {"name_by_user": "Kitchen Lamp"}}
        areas = {"kitchen": {"area_id": "kitchen", "name": "Kitchen"}}
        floors = {}

    snapshot = snapshot_from_client(Client())

    assert snapshot.areas[0].name == "Kitchen"
    entity = snapshot.entities[0]
    assert entity.entity_id == "light.kitchen_ceiling"
    assert entity.name == "Kitchen Ceiling Lights"
    assert "Ceiling Light" in entity.aliases
    assert "Kitchen Lamp" in entity.aliases
    assert "kitchen ceiling" in entity.aliases
    assert entity.area_id == "kitchen"

    resolved = resolve_candidate(
        IntentMatch(
            intent_name="HassTurnOff",
            slots={"domain": SlotValue(value="light", text="light")},
        ),
        snapshot,
        {"area_id": "kitchen"},
        text="Turn the kitchen lamp off",
    )
    assert resolved.entity_ids == frozenset({"light.kitchen_ceiling"})
    assert resolved.match.slots["entity_id"].value == "light.kitchen_ceiling"


def test_catalog_selection_excludes_disabled_and_unavailable_entities():
    class Client:
        states = {
            "light.kitchen_light": {
                "state": "on",
                "attributes": {"friendly_name": "Kitchen Light"},
            },
            "light.disabled_lamp": {
                "state": "off",
                "attributes": {"friendly_name": "Kitchen Lamp"},
            },
            "light.disabled_pendant": {
                "state": "on",
                "attributes": {"friendly_name": "Kitchen Pendant"},
            },
            "light.unavailable_lamp": {
                "state": "unavailable",
                "attributes": {"friendly_name": "Unavailable Kitchen Lamp"},
            },
            "switch.kitchen_lamp": {
                "state": "on",
                "attributes": {"friendly_name": "Kitchen Lamp"},
            },
        }
        entity_registry = {
            "light.kitchen_light": {"ei": "light.kitchen_light", "ai": "kitchen"},
            # ``db`` is the compact disabled marker returned by HA's display registry.
            "light.disabled_lamp": {
                "ei": "light.disabled_lamp",
                "ai": "kitchen",
                "db": "user",
            },
            "light.disabled_pendant": {
                "ei": "light.disabled_pendant",
                "ai": "kitchen",
                "di": "disabled-device",
            },
            "light.unavailable_lamp": {"ei": "light.unavailable_lamp", "ai": "kitchen"},
            "switch.kitchen_lamp": {"ei": "switch.kitchen_lamp", "ai": "kitchen"},
        }
        devices = {"disabled-device": {"disabled_by": "user"}}
        areas = {"kitchen": {"area_id": "kitchen", "name": "Kitchen"}}
        floors = {}

    snapshot = snapshot_from_client(Client())
    entities = {entity.entity_id: entity for entity in snapshot.entities}

    assert entities["light.disabled_lamp"].is_enabled is False
    assert entities["light.disabled_pendant"].is_enabled is False
    assert entities["light.unavailable_lamp"].is_available is False
    assert {entity.entity_id for entity in snapshot.selectable_entities} == {
        "light.kitchen_light",
        "switch.kitchen_lamp",
    }

    light_match = IntentMatch(
        intent_name="HassTurnOff",
        slots={"domain": SlotValue(value="light", text="light")},
    )
    named = resolve_candidate(
        light_match,
        snapshot,
        {"area_id": "kitchen"},
        text="Turn the kitchen lamp off",
    )
    generic = resolve_candidate(
        light_match,
        snapshot,
        {"area_id": "kitchen"},
        text="Turn the kitchen lights off",
    )
    disabled_name = resolve_candidate(
        IntentMatch(
            intent_name="HassTurnOff",
            slots={"name": SlotValue(value="Kitchen Pendant", text="Kitchen Pendant")},
        ),
        snapshot,
    )

    assert named.entity_ids == frozenset({"switch.kitchen_lamp"})
    assert generic.entity_ids == frozenset({"light.kitchen_light"})
    assert disabled_name.entity_ids == frozenset()


@pytest.mark.asyncio
async def test_intent_executor_posts_official_call_and_extracts_speech():
    seen = {}

    def handle(request: httpx.Request) -> httpx.Response:
        seen["authorization"] = request.headers.get("Authorization")
        seen["json"] = __import__("json").loads(request.content)
        return httpx.Response(
            200,
            json={"speech": {"plain": {"speech": "Turned off the kitchen lights"}}},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        executor = HomeAssistantIntentExecutor("http://homeassistant:8123", "secret", client=client)
        result = await executor.execute(
            OhfIntentCall(
                intent_name="HassTurnOff",
                data={"area": "Kitchen", "domain": "light"},
            )
        )

    assert seen == {
        "authorization": "Bearer secret",
        "json": {
            "name": "HassTurnOff",
            "data": {"area": "Kitchen", "domain": "light"},
        },
    }
    assert result.speech == "Turned off the kitchen lights"


@pytest.mark.asyncio
async def test_intent_executor_uses_persistent_websocket_before_http():
    seen = []

    class Ready:
        def is_set(self) -> bool:
            return True

    class WebSocket:
        ready = Ready()

        async def process_conversation(self, text, *, language, device_id, timeout):
            seen.append((text, language, device_id, timeout))
            return {
                "success": True,
                "result": {
                    "conversation_id": "ha-conversation",
                    "response": {
                        "response_type": "action_done",
                        "speech": {"plain": {"speech": "Turned off the office lights"}},
                        "data": {"success": [], "failed": []},
                    },
                },
            }

    def handle(request: httpx.Request) -> httpx.Response:
        pytest.fail(f"HTTP should not be used while the WebSocket is ready: {request.url}")

    _reset_voice_tool_run_state(
        "turn off the office lights",
        {"device_id": "assist_satellite.office"},
        allow_conversation_websocket=True,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        executor = HomeAssistantIntentExecutor(
            "http://ha",
            "secret",
            timeout=4.5,
            client=client,
            websocket_provider=lambda: WebSocket(),
        )
        result = await executor.execute(OhfIntentCall("HassTurnOff", {"name": "Office lights"}))

    assert result.speech == "Turned off the office lights"
    assert seen == [
        (
            "turn off the office lights",
            settings.deterministic.language,
            "assist_satellite.office",
            4.5,
        )
    ]


@pytest.mark.asyncio
async def test_intent_executor_falls_back_to_http_when_websocket_rejects_request():
    class Ready:
        def is_set(self) -> bool:
            return True

    class WebSocket:
        ready = Ready()

        async def process_conversation(self, *_args, **_kwargs):
            return {"success": False, "error": {"code": "unknown_command"}}

    seen = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(
            200,
            json={"speech": {"plain": {"speech": "Turned on the lamp"}}},
        )

    _reset_voice_tool_run_state("turn on the lamp", allow_conversation_websocket=True)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        executor = HomeAssistantIntentExecutor(
            "http://ha",
            "secret",
            client=client,
            websocket_provider=lambda: WebSocket(),
        )
        result = await executor.execute(OhfIntentCall("HassTurnOn", {"name": "Lamp"}))

    assert result.speech == "Turned on the lamp"
    assert seen == ["/api/intent/handle"]


@pytest.mark.asyncio
async def test_intent_executor_reports_home_assistant_error_without_claiming_success():
    def handle(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"message": "Intent could not be handled"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        executor = HomeAssistantIntentExecutor("http://ha", "secret", client=client)
        with pytest.raises(RuntimeError, match="Intent could not be handled"):
            await executor.execute(OhfIntentCall("HassTurnOn", {"name": "Lamp"}))


@pytest.mark.asyncio
async def test_intent_executor_prefers_websocket_state_for_summary(monkeypatch):
    seen = []

    class FakeReady:
        def is_set(self) -> bool:
            return True

    class FakeHaWs:
        ready = FakeReady()
        states = {
            "weather.forecast_home": {
                "entity_id": "weather.forecast_home",
                "state": "sunny",
                "attributes": {
                    "friendly_name": "Forecast Home",
                    "temperature": 22,
                    "unit_of_measurement": "°C",
                },
            }
        }

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path.endswith("/api/intent/handle"):
            return httpx.Response(
                200,
                json={
                    "speech": {},
                    "response_type": "query_answer",
                    "data": {
                        "targets": [],
                        "success": [
                            {
                                "name": "Forecast Home",
                                "type": "entity",
                                "id": "weather.forecast_home",
                                "entity_id": "weather.forecast_home",
                            }
                        ],
                        "failed": [],
                    },
                },
            )
        pytest.fail(f"Unexpected HTTP request to {request.url.path}")

    monkeypatch.setattr(runtime, "ha_ws", FakeHaWs())
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        executor = HomeAssistantIntentExecutor("http://ha", "secret", client=client)
        result = await executor.execute(OhfIntentCall("HassGetWeather", {}))

    assert result.speech == "Forecast Home is sunny and 22 °C"
    assert seen == ["/api/intent/handle"]


@pytest.mark.asyncio
async def test_action_done_does_not_speak_stale_websocket_state(monkeypatch):
    class FakeReady:
        def is_set(self) -> bool:
            return True

    class FakeHaWs:
        ready = FakeReady()
        states = {
            "vacuum.robot": {
                "entity_id": "vacuum.robot",
                "state": "docked",
                "attributes": {"friendly_name": "Robot Vacuum"},
            }
        }

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/api/intent/handle")
        return httpx.Response(
            200,
            json={
                "speech": {},
                "response_type": "action_done",
                "data": {
                    "targets": [],
                    "success": [
                        {
                            "name": "Robot Vacuum",
                            "type": "entity",
                            "id": "vacuum.robot",
                        }
                    ],
                    "failed": [],
                },
            },
        )

    monkeypatch.setattr(runtime, "ha_ws", FakeHaWs())
    monkeypatch.setattr(settings.api, "action_confirmation", "")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        executor = HomeAssistantIntentExecutor("http://ha", "secret", client=client)
        result = await executor.execute(OhfIntentCall("HassVacuumStart", {}))

    assert result.speech == ""


@pytest.mark.asyncio
async def test_duplicate_name_match_failure_retries_exact_entity_over_websocket(monkeypatch):
    seen = []

    class FakeHaWs:
        async def command(self, payload, *, timeout):
            seen.append((payload, timeout))
            return {"success": True, "result": {}}

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/api/intent/handle")
        return httpx.Response(
            200,
            json={
                "speech": {
                    "plain": {
                        "speech": (
                            "<MatchFailedError result=MatchTargetsResult(is_match=False, "
                            "no_match_reason=<MatchFailedReason.DUPLICATE_NAME: 11>)>"
                        )
                    }
                },
                "response_type": "action_done",
                "data": {"targets": [], "success": [], "failed": []},
            },
        )

    monkeypatch.setattr(runtime, "ha_ws", FakeHaWs())
    monkeypatch.setattr(settings.api, "action_confirmation", "")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        executor = HomeAssistantIntentExecutor("http://ha", "secret", client=client)
        result = await executor.execute(
            OhfIntentCall(
                "HassTurnOn",
                {
                    "name": "livingroom_light",
                    "entity_id": "light.livingroom_light",
                },
            )
        )

    assert result.speech == ""
    assert seen[0][0] == {
        "type": "call_service",
        "domain": "light",
        "service": "turn_on",
        "service_data": {},
        "target": {"entity_id": "light.livingroom_light"},
        "return_response": False,
    }


@pytest.mark.asyncio
async def test_intent_executor_delegates_to_advanced_agent_for_weather_sensor_with_numeric_state(monkeypatch):
    class FakeReady:
        def is_set(self) -> bool:
            return True

    class FakeHaWs:
        ready = FakeReady()
        states = {
            "sensor.home_assistant_weather": {
                "entity_id": "sensor.home_assistant_weather",
                "state": 1,
                "attributes": {
                    "friendly_name": "Home Assistant Weather",
                    "unit_of_measurement": "°C",
                },
            }
        }

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/api/intent/handle"):
            return httpx.Response(
                200,
                json={
                    "speech": {},
                    "response_type": "query_answer",
                    "data": {
                        "targets": [],
                        "success": [
                            {
                                "name": "Home Assistant Weather",
                                "type": "entity",
                                "id": "sensor.home_assistant_weather",
                                "entity_id": "sensor.home_assistant_weather",
                            }
                        ],
                        "failed": [],
                    },
                },
            )
        pytest.fail(f"Unexpected HTTP request to {request.url.path}")

    async def fake_advanced(request: str) -> str:
        return json.dumps({"success": True, "result": "Weather entity summary"})

    monkeypatch.setattr(runtime, "ha_ws", FakeHaWs())
    monkeypatch.setattr(runtime, "advanced_agent", object())
    monkeypatch.setattr(executor_module, "ha_advanced", fake_advanced)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        executor = HomeAssistantIntentExecutor("http://ha", "secret", client=client)
        result = await executor.execute(OhfIntentCall("HassGetWeather", {}))

    assert result.speech == "Weather entity summary"


@pytest.mark.asyncio
async def test_intent_executor_returns_error_when_advanced_agent_unavailable(monkeypatch):
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "speech": {},
                "response": {},
                "data": {"test": "payload"},
            },
        )

    monkeypatch.setattr(runtime, "advanced_agent", None)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        executor = HomeAssistantIntentExecutor("http://ha", "secret", client=client)
        result = await executor.execute(OhfIntentCall("HassGetState", {}))

    assert result.speech == "Sorry, I was not able to resolve that one."


@pytest.mark.asyncio
async def test_intent_executor_uses_entity_state_when_response_has_entities_but_no_speech():
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/api/intent/handle"):
            return httpx.Response(
                200,
                json={
                    "speech": {},
                    "response_type": "query_answer",
                    "data": {
                        "targets": [],
                        "success": [
                            {
                                "name": "Forecast Home",
                                "type": "entity",
                                "id": "weather.forecast_home",
                                "entity_id": "weather.forecast_home",
                            }
                        ],
                        "failed": [],
                    },
                },
            )
        if request.url.path.endswith("/api/states/weather.forecast_home"):
            return httpx.Response(
                200,
                json={
                    "entity_id": "weather.forecast_home",
                    "state": "sunny",
                    "attributes": {
                        "friendly_name": "Forecast Home",
                        "temperature": 23,
                        "unit_of_measurement": "°C",
                    },
                },
            )
        return httpx.Response(404, json={"message": "not found"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        executor = HomeAssistantIntentExecutor("http://ha", "secret", client=client)
        result = await executor.execute(OhfIntentCall("HassGetWeather", {}))

    assert result.speech == "Forecast Home is sunny and 23 °C"
