import asyncio
import json
import time
from unittest.mock import AsyncMock

import pytest

from intent_bridge.home_assistant.client import HomeAssistantWebSocket


@pytest.fixture
def client() -> HomeAssistantWebSocket:
    return HomeAssistantWebSocket("http://ha.test", "token")


def test_event_cache_handles_malformed_and_all_invalidation_variants(client):
    client._services_loaded_at = client._registries_loaded_at = 10
    client._handle_event({"event": {"event_type": "state_changed", "data": "bad"}})
    client._handle_event({"event": {"event_type": "state_changed", "data": {"entity_id": 4}}})
    client._handle_event({"event": {"event_type": "service_removed", "data": {}}})
    client._handle_event({"event": {"event_type": "device_registry_updated", "data": {}}})
    client._handle_event({"event": {"event_type": "area_registry_updated", "data": {}}})

    assert client.state_event_count == 0
    assert client._services_loaded_at == 0
    assert client._registries_loaded_at == 0


@pytest.mark.asyncio
async def test_reader_ignores_or_discards_unusable_results(client):
    completed = asyncio.get_running_loop().create_future()
    completed.set_result({"old": True})
    client._pending[1] = completed

    class Frames:
        def __aiter__(self):
            self._frames = iter(
                [
                    json.dumps({"type": "result", "id": 1, "success": True}),
                    json.dumps({"type": "result", "id": 999, "success": True}),
                    json.dumps({"type": "result", "id": "not-an-int"}),
                    json.dumps({"type": "unrelated"}),
                ]
            )
            return self

        async def __anext__(self):
            try:
                return next(self._frames)
            except StopIteration:
                raise StopAsyncIteration from None

    await client._reader_loop(Frames())

    assert completed.result() == {"old": True}
    assert client._pending == {}


@pytest.mark.asyncio
async def test_send_current_cleans_pending_after_send_error_and_timeout(client):
    client._ws = type("Socket", (), {"send": AsyncMock(side_effect=OSError("closed"))})()
    with pytest.raises(OSError, match="closed"):
        await client._send_current({"type": "ping"})
    assert client._pending == {}

    client._ws = type("Socket", (), {"send": AsyncMock()})()
    with pytest.raises(TimeoutError):
        await client._send_current({"type": "ping"}, timeout=0.001)
    assert client._pending == {}


@pytest.mark.asyncio
async def test_command_reraises_after_second_disconnect(client, monkeypatch):
    client.ready.set()
    send = AsyncMock(side_effect=ConnectionError("still offline"))
    monkeypatch.setattr(client, "_send_current", send)

    async def restore_ready(_delay):
        client.ready.set()

    monkeypatch.setattr(asyncio, "sleep", restore_ready)

    with pytest.raises(ConnectionError, match="still offline"):
        await client.command({"type": "ping"})
    assert send.await_count == 2


@pytest.mark.asyncio
async def test_initial_cache_rejects_non_list_states(client):
    client._send_current = AsyncMock(return_value={"success": True, "result": {"light.one": {}}})

    with pytest.raises(RuntimeError, match="unexpected payload"):
        await client._initialise_connection_caches()


@pytest.mark.asyncio
async def test_initial_cache_tolerates_optional_metadata_and_subscription_failures(client):
    client._send_current = AsyncMock(
        side_effect=[
            {"success": True, "result": []},
            {"success": True, "result": []},
            {"success": True, "result": []},
            {"success": True, "result": {}},
            {"success": True, "result": {}},
            *[{"success": False, "error": {"code": "unsupported"}} for _ in range(6)],
        ]
    )

    await client._initialise_connection_caches()

    assert client.services == {}
    assert client.entity_registry == {}
    assert client.devices == {}
    assert client.areas == {}


@pytest.mark.asyncio
async def test_refresh_caches_short_circuit_and_ignore_wrong_service_shape(client):
    client.services = {"existing": {}}
    client._services_loaded_at = time.monotonic()
    client.command = AsyncMock()
    await client.refresh_services()
    assert client.command.await_count == 0

    client.command = AsyncMock(return_value={"success": True, "result": []})
    await client.refresh_services(force=True)
    assert client.services == {"existing": {}}

    client._registries_loaded_at = time.monotonic()
    client.command = AsyncMock()
    await client.refresh_registries()
    assert client.command.await_count == 0


@pytest.mark.asyncio
async def test_refresh_cache_rechecks_freshness_after_waiting_for_lock(client):
    client._services_loaded_at = 0
    await client._service_refresh_lock.acquire()
    task = asyncio.create_task(client.refresh_services())
    await asyncio.sleep(0)
    client._services_loaded_at = time.monotonic()
    client._service_refresh_lock.release()
    await task

    client._registries_loaded_at = 0
    await client._registry_refresh_lock.acquire()
    task = asyncio.create_task(client.refresh_registries())
    await asyncio.sleep(0)
    client._registries_loaded_at = time.monotonic()
    client._registry_refresh_lock.release()
    await task


@pytest.mark.asyncio
async def test_registry_refresh_contains_each_optional_endpoint_failure(client):
    sender = AsyncMock(
        side_effect=[RuntimeError("entities"), RuntimeError("devices"), RuntimeError("areas")]
    )

    await client._refresh_registries(sender)

    assert sender.await_count == 3
    assert client._registries_loaded_at > 0


def test_search_exercises_exact_contains_tokens_fuzzy_ids_and_empty_query(client):
    client.states = {
        "light.office_ceiling": {
            "state": "on",
            "attributes": {"friendly_name": "Office Ceiling"},
        },
        "light.hallway": {
            "state": "off",
            "attributes": {"friendly_name": "Hallway Lamp"},
        },
        "sensor.bad_attributes": {"state": "unknown", "attributes": "invalid"},
    }

    exact = client.search_cached_states(
        "office ceiling", domain_filter="light", area_filter=None, limit=5
    )
    contains = client.search_cached_states(
        "ceiling", domain_filter="light", area_filter=None, limit=5
    )
    tokens = client.search_cached_states(
        "office missing", domain_filter="light", area_filter=None, limit=5
    )
    fuzzy = client.search_cached_states(
        "halway lmp", domain_filter="light", area_filter=None, limit=5
    )
    dotted = client.search_cached_states(
        "light.office ceiling", domain_filter="light", area_filter=None, limit=5
    )
    everything = client.search_cached_states("", domain_filter=None, area_filter=None, limit=10)

    assert "exact_name" in exact[0]["match_reasons"]
    assert "name_contains_query" in contains[0]["match_reasons"]
    assert "token_match" in tokens[0]["match_reasons"]
    assert "fuzzy_match" in fuzzy[0]["match_reasons"]
    assert dotted[0]["entity_id"] == "light.office_ceiling"
    assert {item["entity_id"] for item in everything} == set(client.states)


def test_search_applies_area_origin_media_and_indicator_ranking(client):
    client.states = {
        "light.office_main": {
            "state": "on",
            "attributes": {"friendly_name": "Office Main"},
        },
        "light.office_status": {
            "state": "on",
            "attributes": {"friendly_name": "Voice Status LED"},
        },
        "media_player.speaker": {
            "state": "idle",
            "attributes": {"friendly_name": "Speaker", "device_class": "speaker"},
        },
        "media_player.tv": {
            "state": "off",
            "attributes": {"friendly_name": "TV", "device_class": "tv"},
        },
    }
    client.entity_registry = {
        entity_id: {"ei": entity_id, "ai": "office"} for entity_id in client.states
    }
    client.areas = {"office": {"area_id": "office", "name": "Office"}}

    lights = client.search_cached_states(
        "light",
        domain_filter="light",
        area_filter="office",
        preferred_area_filter="office",
        limit=10,
    )
    media = client.search_cached_states(
        "media player",
        domain_filter="media_player",
        area_filter=None,
        preferred_area_filter="office",
        limit=10,
    )

    assert lights[0]["entity_id"] == "light.office_main"
    assert any("indicator_light_penalty" in item["match_reasons"] for item in lights)
    assert media[0]["entity_id"] == "media_player.speaker"
    assert "speaker_preference" in media[0]["match_reasons"]
    assert "tv_penalty" in media[1]["match_reasons"]
    assert (
        client.search_cached_states("light", domain_filter="light", area_filter="kitchen", limit=10)
        == []
    )


@pytest.mark.asyncio
async def test_wait_for_expected_state_times_out_after_polling(client, monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())

    assert await client.wait_for_expected_state("light.missing", "on", 0.001) is None
