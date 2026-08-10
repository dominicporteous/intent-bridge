import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from intent_bridge.config import settings
from intent_bridge.home_assistant import client as ha_client
from intent_bridge.runtime.dependencies import runtime


@pytest.fixture
def client():
    value = ha_client.HomeAssistantWebSocket("https://ha.test/base/", "token")
    value.states = {
        "light.office_main": {
            "state": "on",
            "attributes": {"friendly_name": "Office Light"},
        },
        "light.office_ring_led": {
            "state": "off",
            "attributes": {"friendly_name": "Office Voice LED"},
        },
        "media_player.office_speaker": {
            "state": "idle",
            "attributes": {"friendly_name": "Office Speaker", "device_class": "speaker"},
        },
        "media_player.office_tv": {
            "state": "off",
            "attributes": {"friendly_name": "Office TV", "device_class": "tv"},
        },
        "sensor.temperature": {
            "state": "21",
            "attributes": {"friendly_name": "Room Temperature", "unit_of_measurement": "°C"},
        },
    }
    value.entity_registry = {
        "light.office_main": {"ei": "light.office_main", "di": "dev-light", "ai": "office"},
        "light.office_ring_led": {"ei": "light.office_ring_led", "di": "dev-voice", "ai": "office"},
        "media_player.office_speaker": {
            "ei": "media_player.office_speaker",
            "di": "dev-speaker",
            "ai": "office",
        },
        "media_player.office_tv": {"ei": "media_player.office_tv", "di": "dev-tv", "ai": "office"},
    }
    value.devices = {
        "dev-light": {"id": "dev-light", "name": "Ceiling fixture", "area_id": "office"},
        "dev-voice": {"id": "dev-voice", "name": "Voice PE", "area_id": "office"},
        "dev-speaker": {"id": "dev-speaker", "name_by_user": "Office Speaker", "area_id": "office"},
        "dev-tv": {"id": "dev-tv", "name": "Office TV", "area_id": "office"},
        "inherited": {"id": "inherited", "name": "Satellite"},
    }
    value.entity_registry["assist_satellite.office"] = {
        "ei": "assist_satellite.office",
        "di": "inherited",
        "ai": "office",
    }
    value.areas = {
        "office": {"area_id": "office", "name": "Office"},
        "kitchen": {"area_id": "kitchen", "name": "Kitchen"},
    }
    return value


def test_initial_state_and_url():
    client = ha_client.HomeAssistantWebSocket("http://ha.test", "secret")
    assert client.base_url == "http://ha.test"
    assert client.ws_url == "ws://ha.test/api/websocket"
    assert not client.ready.is_set()
    assert client.states == {}


def test_require_success():
    assert ha_client.HomeAssistantWebSocket._require_success(
        {"success": True, "result": {"ok": 1}}, "demo"
    ) == {"ok": 1}
    with pytest.raises(RuntimeError, match="demo failed"):
        ha_client.HomeAssistantWebSocket._require_success(
            {"success": False, "error": "bad"}, "demo"
        )


def test_handle_events_updates_cache_and_invalidates_metadata(client):
    client._handle_event({"event": "bad"})
    client._handle_event({"event": {"event_type": "state_changed", "data": {}}})
    client._handle_event(
        {
            "event": {
                "event_type": "state_changed",
                "data": {"entity_id": "switch.new", "new_state": {"state": "on"}},
            }
        }
    )
    assert client.states["switch.new"]["state"] == "on"
    assert client.state_event_count == 1
    client._handle_event(
        {"event": {"event_type": "state_changed", "data": {"entity_id": "switch.new"}}}
    )
    assert "switch.new" not in client.states
    client._services_loaded_at = client._registries_loaded_at = 10
    client._handle_event({"event": {"event_type": "service_registered", "data": {}}})
    client._handle_event({"event": {"event_type": "entity_registry_updated", "data": {}}})
    assert client._services_loaded_at == client._registries_loaded_at == 0


@pytest.mark.asyncio
async def test_reader_loop_routes_results_events_and_ignores_bad_json(client):
    future = asyncio.get_running_loop().create_future()
    client._pending[7] = future

    class Frames:
        def __aiter__(self):
            self.items = iter(
                [
                    "not-json",
                    json.dumps({"type": "result", "id": 7, "success": True}),
                    json.dumps(
                        {
                            "type": "event",
                            "event": {
                                "event_type": "state_changed",
                                "data": {
                                    "entity_id": "switch.reader",
                                    "new_state": {"state": "on"},
                                },
                            },
                        }
                    ),
                ]
            )
            return self

        async def __anext__(self):
            try:
                return next(self.items)
            except StopIteration:
                raise StopAsyncIteration from None

    await client._reader_loop(Frames())
    assert future.result()["success"] is True
    assert client.states["switch.reader"]["state"] == "on"


def test_fail_pending_sets_exception(client):
    loop = asyncio.new_event_loop()
    try:
        pending = loop.create_future()
        done = loop.create_future()
        done.set_result("done")
        client._pending = {1: pending, 2: done}
        client._fail_pending(RuntimeError("closed"))
        assert isinstance(pending.exception(), RuntimeError)
        assert client._pending == {}
    finally:
        loop.close()


@pytest.mark.asyncio
async def test_send_current_sends_correlated_message(client):
    sent = []

    class Socket:
        async def send(self, raw):
            sent.append(json.loads(raw))
            request_id = sent[-1]["id"]
            asyncio.get_running_loop().call_soon(
                client._pending[request_id].set_result,
                {"id": request_id, "success": True, "result": "ok"},
            )

    client._ws = Socket()
    assert await client._send_current({"type": "ping"}) == {
        "id": 1,
        "success": True,
        "result": "ok",
    }
    assert sent == [{"id": 1, "type": "ping"}]


@pytest.mark.asyncio
async def test_send_current_requires_socket(client):
    with pytest.raises(ConnectionError, match="not connected"):
        await client._send_current({"type": "ping"})


@pytest.mark.asyncio
async def test_command_retries_one_disconnect(client, monkeypatch):
    client.ready.set()
    send = AsyncMock(side_effect=[ConnectionError("drop"), {"success": True}])
    monkeypatch.setattr(client, "_send_current", send)

    async def immediately_ready():
        client.ready.set()

    original_sleep = asyncio.sleep

    async def sleep(delay):
        await immediately_ready()
        await original_sleep(0)

    monkeypatch.setattr(ha_client.asyncio, "sleep", sleep)
    assert await client.command({"type": "x"}) == {"success": True}
    assert send.await_count == 2


@pytest.mark.asyncio
async def test_refresh_services_and_registries(client, monkeypatch):
    client.command = AsyncMock(
        side_effect=[
            {"success": True, "result": {"light": {"turn_on": {}}}},
            {"success": True, "result": {"entities": [{"ei": "light.x"}]}},
            {"success": True, "result": [{"id": "device"}]},
            {"success": True, "result": [{"area_id": "room", "name": "Room"}]},
        ]
    )
    await client.refresh_services(force=True)
    await client.refresh_registries(force=True)
    assert "light" in client.services
    assert client.entity_registry == {"light.x": {"ei": "light.x"}}
    assert "device" in client.devices
    assert client.areas["room"]["name"] == "Room"


@pytest.mark.asyncio
async def test_initialise_connection_caches(client):
    replies = [
        {"success": True, "result": [{"entity_id": "light.one"}, {"bad": True}]},
        {"success": True, "result": {"light": {}}},
        {"success": True, "result": {"entities": []}},
        {"success": True, "result": []},
        {"success": True, "result": []},
        *[{"success": True, "result": None} for _ in range(6)],
    ]
    client._send_current = AsyncMock(side_effect=replies)
    await client._initialise_connection_caches()
    assert list(client.states) == ["light.one"]
    assert client.services == {"light": {}}
    assert client._send_current.await_count == 11


def test_entity_context_and_area_resolution(client):
    context = client._entity_context("light.office_main", client.states["light.office_main"])
    assert context["device_name"] == "Ceiling fixture"
    assert context["area_name"] == "Office"
    assert client.resolve_area_reference(area_id="office") == ("office", "Office")
    assert client.resolve_area_reference(area_name=" office ") == ("office", "Office")
    assert client.resolve_area_reference(area_name="missing") == (None, None)


def test_search_ranks_room_lights_and_speakers(client):
    lights = client.search_cached_states(
        "light", domain_filter="light", area_filter=None, preferred_area_filter="office", limit=10
    )
    assert lights[0]["entity_id"] == "light.office_main"
    assert "indicator_light_penalty" not in lights[0]["match_reasons"]
    speakers = client.search_cached_states(
        "media player",
        domain_filter="media_player",
        area_filter="office",
        preferred_area_filter="office",
        limit=10,
    )
    assert speakers[0]["entity_id"] == "media_player.office_speaker"
    assert (
        client.search_cached_states(
            "missing", domain_filter="light", area_filter="kitchen", limit=5
        )
        == []
    )


def test_device_origin_area_mentions_and_entities(client):
    assert client.resolve_device_origin(device_name="office speaker") == {
        "device_id": "dev-speaker",
        "device_name": "Office Speaker",
        "area_id": "office",
        "area_name": "Office",
    }
    inherited = client.resolve_device_origin(device_id="inherited")
    assert inherited["area_id"] == "office"
    assert (
        client.resolve_device_origin(device_id="unknown", area_id="future")["area_id"] == "future"
    )
    assert client.area_mentioned_in_text("turn on the office lights") == ("office", "Office")
    assert client.area_mentioned_in_text("office and kitchen") is None
    assert client.entities_in_area("light", "office") == [
        "light.office_main",
        "light.office_ring_led",
    ]


@pytest.mark.asyncio
async def test_wait_for_expected_state(client):
    assert await client.wait_for_expected_state("light.office_main", "on", 0.1) == "on"
    assert await client.wait_for_expected_state("light.office_main", "off", 0) is None


@pytest.mark.asyncio
async def test_require_global_client(client, monkeypatch):
    monkeypatch.setattr(runtime, "ha_ws", None)
    with pytest.raises(RuntimeError, match="unavailable"):
        await ha_client._require_ha_ws()
    client.ready.set()
    monkeypatch.setattr(runtime, "ha_ws", client)
    assert await ha_client._require_ha_ws() is client


@pytest.mark.asyncio
async def test_connection_authentication_success_and_failures(client, monkeypatch):
    class Socket:
        def __init__(self, frames):
            self.frames = iter(frames)
            self.sent = []

        async def recv(self):
            return next(self.frames)

        async def send(self, value):
            self.sent.append(json.loads(value))

    class Connection:
        def __init__(self, socket):
            self.socket = socket

        async def __aenter__(self):
            return self.socket

        async def __aexit__(self, *_args):
            return False

    socket = Socket([json.dumps({"type": "auth_required"}), json.dumps({"type": "auth_ok"})])
    monkeypatch.setattr(
        ha_client.websockets, "connect", lambda *_args, **_kwargs: Connection(socket)
    )
    monkeypatch.setattr(client, "_initialise_connection_caches", AsyncMock())
    monkeypatch.setattr(client, "_reader_loop", AsyncMock())
    await client._run_connection()
    assert socket.sent == [{"type": "auth", "access_token": "token"}]
    assert client.connected_at is not None and not client.ready.is_set()

    client.connected_at = 1
    socket = Socket([json.dumps({"type": "auth_required"}), json.dumps({"type": "auth_ok"})])
    monkeypatch.setattr(
        ha_client.websockets, "connect", lambda *_args, **_kwargs: Connection(socket)
    )
    await client._run_connection()
    assert client.reconnect_count == 1

    for frames, message in [
        ([json.dumps({"type": "wrong"})], "Unexpected"),
        (
            [json.dumps({"type": "auth_required"}), json.dumps({"type": "auth_invalid"})],
            "authentication failed",
        ),
    ]:
        socket = Socket(frames)
        monkeypatch.setattr(
            ha_client.websockets,
            "connect",
            lambda *_args, socket=socket, **_kwargs: Connection(socket),
        )
        with pytest.raises(RuntimeError, match=message):
            await client._run_connection()


@pytest.mark.asyncio
async def test_client_start_stop_and_timeout(client, monkeypatch):
    async def supervisor():
        client.ready.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(client, "_supervisor", supervisor)
    await client.start()
    existing = client._supervisor_task
    await client.start()
    assert client._supervisor_task is existing

    socket = SimpleNamespace(close=AsyncMock(side_effect=RuntimeError("already closed")))
    client._ws = socket
    await client.stop()
    assert socket.close.await_count == 1

    timed_out = ha_client.HomeAssistantWebSocket("http://ha", "token")
    monkeypatch.setattr(settings.home_assistant.websocket, "connect_timeout_seconds", 0)

    async def never_ready():
        await asyncio.Event().wait()

    monkeypatch.setattr(timed_out, "_supervisor", never_ready)
    with pytest.raises(RuntimeError, match="Timed out"):
        await timed_out.start()
    await timed_out.stop()


@pytest.mark.asyncio
async def test_supervisor_records_error_and_stops(client, monkeypatch):
    calls = 0

    async def connection():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("offline")
        client._stopping.set()

    monkeypatch.setattr(client, "_run_connection", connection)
    monkeypatch.setattr(settings.home_assistant.websocket, "reconnect_delay_seconds", 0)
    await client._supervisor()
    assert calls == 2 and client.last_error == "offline"
