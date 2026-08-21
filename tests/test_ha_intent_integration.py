from __future__ import annotations

import json
from types import SimpleNamespace

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

_AUDITED_AREA_INDEPENDENT_CONVERSATION_INTENTS = frozenset(
    {
        "HassBroadcast",
        "HassGetCurrentDate",
        "HassGetCurrentTime",
        "HassGetWeather",
        "HassNevermind",
        "HassRespond",
        "HassShoppingListAddItem",
        "HassShoppingListCompleteItem",
        "HassShoppingListLastItems",
    }
)

_AUDITED_TIMER_INTENTS = frozenset(
    {
        "HassStartTimer",
        "HassCancelAllTimers",
        "HassIncreaseTimer",
        "HassDecreaseTimer",
        "HassCancelTimer",
        "HassPauseTimer",
        "HassUnpauseTimer",
        "HassTimerStatus",
    }
)


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


def test_catalog_snapshot_derives_power_capabilities_from_live_services():
    class Client:
        states = {
            "switch.lamp": {"state": "on", "attributes": {}},
            "number.lamp_countdown": {"state": "10", "attributes": {}},
        }
        entity_registry = {
            "switch.lamp": {"ei": "switch.lamp"},
            "number.lamp_countdown": {"ei": "number.lamp_countdown"},
        }
        devices = {}
        areas = {}
        floors = {}
        services = {
            "switch": {"turn_on": {}, "turn_off": {}},
            "number": {"set_value": {}},
        }

    entities = {entity.entity_id: entity for entity in snapshot_from_client(Client()).entities}

    assert entities["switch.lamp"].supported_intents == frozenset(
        {"HassTurnOn", "HassTurnOff"}
    )
    assert entities["number.lamp_countdown"].supported_intents == frozenset()


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


def test_area_independent_conversation_intents_match_the_audited_core_allow_list():
    assert executor_module._AREA_INDEPENDENT_CONVERSATION_INTENTS == (
        _AUDITED_AREA_INDEPENDENT_CONVERSATION_INTENTS
    )


def test_timer_intents_match_the_audited_core_allow_list():
    assert executor_module._TIMER_INTENTS == _AUDITED_TIMER_INTENTS


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("call", "speech_slots", "expected"),
    [
        (
            OhfIntentCall("HassGetCurrentTime", {}),
            {"time": "14:01:22.158666"},
            "The time is 2:01 PM.",
        ),
        (
            OhfIntentCall("HassGetCurrentDate", {}),
            {"date": "2026-08-20"},
            "Today is Thursday, 20 August 2026.",
        ),
        (
            OhfIntentCall("HassCancelAllTimers", {}),
            {"canceled": 2, "area": "Kitchen"},
            "Cancelled 2 timers in Kitchen.",
        ),
        (
            OhfIntentCall("HassTimerStatus", {}),
            {
                "timers": [
                    {
                        "name": "Pasta",
                        "is_active": True,
                        "rounded_hours_left": 0,
                        "rounded_minutes_left": 5,
                        "rounded_seconds_left": 0,
                    }
                ]
            },
            "Pasta has 5 minutes remaining.",
        ),
        (
            OhfIntentCall("HassShoppingListCompleteItem", {"item": "milk"}),
            {"completed_items": [{"name": "Milk"}]},
            "Completed Milk on the shopping list.",
        ),
        (
            OhfIntentCall("HassMediaSearchAndPlay", {"search_query": "Space Oddity"}),
            {"media": {"title": "Space Oddity"}},
            "Playing Space Oddity.",
        ),
    ],
)
async def test_intent_executor_renders_known_speech_slots_without_llm(
    monkeypatch, call, speech_slots, expected
):
    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/api/intent/handle")
        return httpx.Response(
            200,
            json={
                "speech": {},
                "response_type": "action_done",
                "speech_slots": speech_slots,
                "data": {"targets": [], "success": [], "failed": []},
            },
        )

    async def no_llm(*_args, **_kwargs):
        pytest.fail("Known Home Assistant speech_slots must not invoke an LLM")

    monkeypatch.setattr(executor_module, "Runner", SimpleNamespace(run=no_llm))
    monkeypatch.setattr(runtime, "fallback_agent", object())
    monkeypatch.setattr(runtime, "advanced_agent", None)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        executor = HomeAssistantIntentExecutor("http://ha", "secret", client=client)
        result = await executor.execute(call)

    assert result.speech == expected


@pytest.mark.asyncio
async def test_intent_executor_recovers_deterministically_from_malformed_known_speech_slots(
    monkeypatch, caplog
):
    def handle(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "speech": {},
                "response_type": "action_done",
                "speech_slots": {"time": "not-a-time"},
                "data": {"targets": [], "success": [], "failed": []},
            },
        )

    async def no_llm(*_args, **_kwargs):
        pytest.fail("Malformed built-in speech_slots must use deterministic recovery")

    monkeypatch.setattr(executor_module, "Runner", SimpleNamespace(run=no_llm))
    monkeypatch.setattr(runtime, "fallback_agent", object())
    monkeypatch.setattr(runtime, "advanced_agent", None)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        executor = HomeAssistantIntentExecutor("http://ha", "secret", client=client)
        result = await executor.execute(OhfIntentCall("HassGetCurrentTime", {}))

    assert result.speech == "I couldn't read the current time."
    assert "speech_slots rendered with deterministic recovery" in caplog.text


@pytest.mark.asyncio
async def test_intent_executor_logs_unknown_speech_slot_policy_before_llm_fallback(
    monkeypatch, caplog
):
    def handle(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "speech": {},
                "response_type": "action_done",
                "speech_slots": {"custom_result": "value"},
                "data": {"targets": [], "success": [], "failed": []},
            },
        )

    async def llm_fallback(_agent, request: str):
        assert "CustomIntent" in request
        return SimpleNamespace(final_output="Custom response")

    monkeypatch.setattr(executor_module, "Runner", SimpleNamespace(run=llm_fallback))
    monkeypatch.setattr(runtime, "fallback_agent", object())
    monkeypatch.setattr(runtime, "advanced_agent", None)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        executor = HomeAssistantIntentExecutor("http://ha", "secret", client=client)
        result = await executor.execute(OhfIntentCall("CustomIntent", {}))

    assert result.speech == "Custom response"
    assert "policy=state_summary_then_llm_fallback" in caplog.text


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
@pytest.mark.parametrize(
    ("intent_name", "data"),
    [
        ("HassStartTimer", {"minutes": 5}),
        ("HassCancelAllTimers", {}),
        ("HassCancelTimer", {"name": "Tea"}),
        ("HassIncreaseTimer", {"name": "Tea", "minutes": 1}),
        ("HassDecreaseTimer", {"name": "Tea", "minutes": 1}),
        ("HassPauseTimer", {"name": "Tea"}),
        ("HassUnpauseTimer", {"name": "Tea"}),
        ("HassTimerStatus", {}),
    ],
)
async def test_intent_executor_keeps_timer_device_id_off_unsupported_http_dispatch(
    intent_name, data
):
    seen: list[dict[str, object]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert "device_id" not in payload
        seen.append(payload)
        return httpx.Response(
            200,
            json={"speech": {"plain": {"speech": "Timer command completed."}}},
        )

    _reset_voice_tool_run_state(
        "Create a five minute timer",
        {"device_id": "kitchen-voice-device"},
        # This models an exact named-intent dispatch, including one from a
        # compound command that cannot safely replay the whole utterance.
        allow_conversation_websocket=False,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        executor = HomeAssistantIntentExecutor("http://ha", "secret", client=client)
        result = await executor.execute(OhfIntentCall(intent_name, data))

    assert result.speech == "Timer command completed."
    assert seen == [
        {"name": intent_name, "data": data}
    ]

@pytest.mark.asyncio
async def test_intent_executor_targets_timer_calling_device_over_websocket():
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
                    "response": {
                        "response_type": "action_done",
                        "speech": {"plain": {"speech": "Timer started."}},
                        "data": {"success": [], "failed": []},
                    }
                },
            }

    def handle(request: httpx.Request) -> httpx.Response:
        pytest.fail(f"Timer with a calling device must not use HTTP: {request.url}")

    _reset_voice_tool_run_state(
        "Set a 60 second timer",
        {"device_id": "office-voice"},
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
        result = await executor.execute(OhfIntentCall("HassStartTimer", {"seconds": 60}))

    assert result.speech == "Timer started."
    assert seen == [
        ("Set a 60 second timer", settings.deterministic.language, "office-voice", 4.5)
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("satellite_count", "expected_device_id"),
    [(1, "office-voice"), (2, None)],
)
async def test_intent_executor_uses_sole_origin_area_satellite_for_timers(
    satellite_count, expected_device_id
):
    states = {"assist_satellite.office": {"state": "idle", "attributes": {}}}
    registry = {"assist_satellite.office": {"di": "office-voice", "ai": "office"}}
    devices = {"office-voice": {"name": "Office Voice", "area_id": "office"}}
    if satellite_count == 2:
        states["assist_satellite.office_tablet"] = {"state": "idle", "attributes": {}}
        registry["assist_satellite.office_tablet"] = {"di": "office-tablet", "ai": "office"}
        devices["office-tablet"] = {"name": "Office Tablet", "area_id": "office"}

    async def refresh_registries() -> None:
        return None

    ha_ws = SimpleNamespace(
        states=states,
        entity_registry=registry,
        devices=devices,
        refresh_registries=refresh_registries,
        resolve_device_origin=lambda **_kwargs: {
            "device_id": None,
            "area_id": "office",
            "area_name": "Office",
        },
        resolve_area_reference=lambda **_kwargs: ("office", "Office"),
    )
    seen: list[dict[str, object]] = []
    websocket_seen = []

    class Ready:
        def is_set(self) -> bool:
            return True

    class WebSocket:
        ready = Ready()

        async def process_conversation(self, text, *, language, device_id, timeout):
            websocket_seen.append((text, language, device_id, timeout))
            return {
                "success": True,
                "result": {
                    "response": {
                        "response_type": "action_done",
                        "speech": {"plain": {"speech": "Timer command completed."}},
                        "data": {"success": [], "failed": []},
                    }
                },
            }


    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"speech": {"plain": {"speech": "Timer command completed."}}},
        )

    _reset_voice_tool_run_state(
        "Set a 60 second timer",
        {"area_id": "office", "area_name": "Office", "source": "ha_system_prompt"},
        allow_conversation_websocket=True,
    )
    with runtime.override(ha_ws=ha_ws):
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            executor = HomeAssistantIntentExecutor(
                "http://ha",
                "secret",
                client=client,
                timeout=4.5,
                websocket_provider=lambda: WebSocket(),
            )
            result = await executor.execute(OhfIntentCall("HassStartTimer", {"seconds": 60}))

    assert result.speech == "Timer command completed."
    if expected_device_id is None:
        assert seen == [{"name": "HassStartTimer", "data": {"seconds": 60}}]
        assert websocket_seen == []
    else:
        assert seen == []
        assert websocket_seen == [
            ("Set a 60 second timer", settings.deterministic.language, expected_device_id, 4.5)
        ]



@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("intent_name", "data", "text"),
    [
        ("HassBroadcast", {"message": "Dinner is ready"}, "Broadcast that dinner is ready"),
        ("HassGetCurrentDate", {}, "What is the date?"),
        ("HassGetCurrentTime", {}, "What time is it?"),
        ("HassGetWeather", {}, "What is the weather?"),
        ("HassNevermind", {}, "Never mind"),
        ("HassRespond", {"response": "You're welcome"}, "Thank you"),
        ("HassShoppingListAddItem", {"item": "Milk"}, "Add milk to the shopping list"),
        (
            "HassShoppingListCompleteItem",
            {"item": "Milk"},
            "Mark milk as complete on the shopping list",
        ),
        ("HassShoppingListLastItems", {}, "What is on the shopping list?"),
    ],
)
async def test_intent_executor_uses_conversation_websocket_for_area_independent_intents(
    intent_name, data, text
):
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
                    "response": {
                        "response_type": "action_done",
                        "speech": {"plain": {"speech": "Native Home Assistant response."}},
                        "data": {"success": [], "failed": []},
                    }
                },
            }

    def handle(request: httpx.Request) -> httpx.Response:
        pytest.fail(f"HTTP should not be used for an area-independent intent: {request.url}")

    _reset_voice_tool_run_state(
        text,
        {"area_id": "bedroom", "area_name": "Bedroom"},
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
        result = await executor.execute(OhfIntentCall(intent_name, data))

    assert result.speech == "Native Home Assistant response."
    assert seen == [(text, settings.deterministic.language, None, 4.5)]


@pytest.mark.asyncio
async def test_intent_executor_keeps_area_sensitive_intents_on_http_without_device_id():
    class Ready:
        def is_set(self) -> bool:
            return True

    class WebSocket:
        ready = Ready()

        async def process_conversation(self, *_args, **_kwargs):
            pytest.fail("Area-sensitive commands must preserve bridge target resolution")

    seen = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(
            200,
            json={"speech": {"plain": {"speech": "Turned off the bedroom lights."}}},
        )

    _reset_voice_tool_run_state(
        "Turn off the lights",
        {"area_id": "bedroom", "area_name": "Bedroom"},
        allow_conversation_websocket=True,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        executor = HomeAssistantIntentExecutor(
            "http://ha",
            "secret",
            client=client,
            websocket_provider=lambda: WebSocket(),
        )
        result = await executor.execute(
            OhfIntentCall("HassTurnOff", {"area": "Bedroom", "domain": "light"})
        )

    assert result.speech == "Turned off the bedroom lights."
    assert seen == ["/api/intent/handle"]


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
async def test_trigger_automation_intent_runs_the_resolved_automation(monkeypatch):
    seen = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        assert request.url.path == "/api/services/automation/trigger"
        assert request.headers["Authorization"] == "Bearer secret"
        assert json.loads(request.content) == {"entity_id": "automation.goodnight"}
        return httpx.Response(200, json=[])

    monkeypatch.setattr(settings.api, "action_confirmation", "Done.")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        executor = HomeAssistantIntentExecutor("http://ha", "secret", client=client)
        result = await executor.execute(
            OhfIntentCall(
                "IntentBridgeTriggerAutomation",
                {"entity_id": "automation.goodnight", "name": "Goodnight"},
            )
        )

    assert len(seen) == 1
    assert result.speech == "Done."
    assert result.response == {
        "entity_id": "automation.goodnight",
        "service": "automation.trigger",
    }


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
