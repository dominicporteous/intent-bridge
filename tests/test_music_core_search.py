import asyncio
from enum import Enum
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from intent_bridge.config import settings
from intent_bridge.music_assistant import client as music
from intent_bridge.music_assistant import search as music_search
from intent_bridge.runtime.dependencies import runtime
from intent_bridge.runtime.execution import voice_tool_run_state


class State(Enum):
    PLAYING = "playing"


class Collection:
    def __init__(self, values):
        self.values = list(values)

    def __iter__(self):
        return iter(self.values)

    def get(self, key):
        return next(
            (
                item
                for item in self.values
                if key in {getattr(item, "player_id", None), getattr(item, "queue_id", None)}
            ),
            None,
        )


def test_player_and_queue_summaries():
    current = SimpleNamespace(queue_item_id="i", name="Track", media_item=None, uri="x")
    queue = SimpleNamespace(
        queue_id="queue",
        display_name="Office",
        state=State.PLAYING,
        current_index=1,
        elapsed_time=2,
        shuffle_enabled=True,
        repeat_mode="all",
        current_item=current,
    )
    queues = Collection([queue])
    client = SimpleNamespace(player_queues=queues)
    player = SimpleNamespace(
        player_id="office",
        name="Office",
        available=True,
        powered=True,
        state=State.PLAYING,
        volume_level=50,
        volume_muted=False,
        active_source="queue",
        synced_to=None,
        group_childs=["child"],
    )
    summary = music._ma_player_summary(client, player)
    assert summary["queue_id"] == "queue" and summary["group_childs"] == ["child"]
    assert music._ma_queue_summary(client, "queue")["current_item"]["queue_item_id"] == "i"
    assert music._ma_queue_summary(client, "missing") == {"queue_id": "missing", "cached": False}


@pytest.mark.asyncio
async def test_native_manager_state_tracking_and_serialization(monkeypatch):
    manager = music.NativeMusicAssistant("ws://ma", "token")
    connection = SimpleNamespace(connected=True)
    manager.client = SimpleNamespace(connection=connection)
    manager.ready.set()
    assert manager.connected
    assert await manager.wait_ready() is manager.client
    assert (
        await manager.run_serialized("demo", lambda client: asyncio.sleep(0, result="ok")) == "ok"
    )
    assert manager.begin_queue_playback_generation("q") == 1
    assert manager.begin_queue_playback_generation("q") == 2
    assert manager.is_current_queue_playback_generation("q", 2)
    manager.remember_radio_seed("artist", "track")
    assert manager.last_radio_seed("artist") == "track"

    task = asyncio.create_task(asyncio.sleep(10))
    record = manager.register_inflight_playback("q", "f", task, "label")
    assert manager.get_inflight_playback("q", "f") is record
    assert manager.inflight_playback_count == 1
    manager.clear_inflight_playback("q", "f", task)
    assert manager.get_inflight_playback("q", "f") is None
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    background = asyncio.create_task(asyncio.sleep(0))
    manager.track_background_task(background, label="done")
    await background
    await asyncio.sleep(0)
    assert background not in manager._background_tasks
    await manager.stop()
    assert manager.client is None and not manager.ready.is_set()


@pytest.mark.asyncio
async def test_native_manager_start_unavailable_timeout_and_wait_errors(monkeypatch):
    manager = music.NativeMusicAssistant("ws://ma", "token")
    monkeypatch.setattr(music, "MusicAssistantClient", None)
    monkeypatch.setattr(music, "MUSIC_ASSISTANT_CLIENT_IMPORT_ERROR", "missing")
    monkeypatch.setattr(settings.music_assistant, "command_timeout_seconds", 0.01)
    assert not await manager.start()
    assert manager.last_error == "missing"
    with pytest.raises(RuntimeError, match="not connected"):
        await manager.wait_ready()
    monkeypatch.setattr(runtime, "music_assistant", None)
    with pytest.raises(RuntimeError, match="unavailable"):
        music._ma_native_manager()
    monkeypatch.setattr(runtime, "music_assistant", manager)
    assert music._ma_native_manager() is manager


@pytest.mark.asyncio
async def test_native_manager_start_stop_and_timeout(monkeypatch):
    monkeypatch.setattr(music, "MusicAssistantClient", object)
    manager = music.NativeMusicAssistant("ws://ma", "token")

    async def connected_supervisor():
        manager.client = SimpleNamespace(connection=SimpleNamespace(connected=True))
        manager.ready.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(manager, "_supervisor", connected_supervisor)
    assert await manager.start() is True
    existing = manager._supervisor_task
    assert await manager.start() is True
    assert manager._supervisor_task is existing

    background = asyncio.create_task(asyncio.Event().wait())
    manager.track_background_task(background, label="cancelled")
    await manager.stop()
    assert background.cancelled()

    timed_out = music.NativeMusicAssistant("ws://ma", "token")
    monkeypatch.setattr(settings.music_assistant, "connect_timeout_seconds", 0)

    async def never_ready():
        await asyncio.Event().wait()

    monkeypatch.setattr(timed_out, "_supervisor", never_ready)
    assert await timed_out.start() is False
    await timed_out.stop()


@pytest.mark.asyncio
async def test_native_supervisor_success_and_disconnect_cleanup(monkeypatch):
    manager = music.NativeMusicAssistant("ws://ma", "token")

    class FakeClient:
        def __init__(self, *_args, **_kwargs):
            self.connection = SimpleNamespace(connected=True)
            self.server_info = SimpleNamespace(server_version="1", schema_version="2")
            self.players = SimpleNamespace(players=[1])
            self.player_queues = SimpleNamespace(player_queues=[1])
            self.release = asyncio.Event()
            self.disconnect = AsyncMock()

        async def start_listening(self, ready):
            ready.set()
            await self.release.wait()

    created = []

    def factory(*args, **kwargs):
        client = FakeClient(*args, **kwargs)
        created.append(client)
        return client

    monkeypatch.setattr(music, "MusicAssistantClient", factory)
    task = asyncio.create_task(manager._supervisor())
    await manager.ready.wait()
    assert manager.connection_count == 1 and manager.connected
    manager._stopping.set()
    created[0].release.set()
    await task
    assert created[0].disconnect.await_count == 1
    assert manager.client is None and not manager.ready.is_set()


@pytest.mark.asyncio
async def test_native_manager_completed_inflight_and_background_errors():
    manager = music.NativeMusicAssistant("ws://ma", "token")
    done = asyncio.create_task(asyncio.sleep(0))
    manager.register_inflight_playback("q", "f", done, "done")
    await done
    assert manager.get_inflight_playback("q", "f") is None

    other = asyncio.create_task(asyncio.sleep(0))
    current = asyncio.create_task(asyncio.sleep(0))
    manager.register_inflight_playback("q", "other", current, "current")
    manager.clear_inflight_playback("q", "other", other)
    assert manager.get_inflight_playback("q", "other") is not None
    manager.remember_radio_seed("", "ignored")
    assert manager.last_radio_seed("") is None

    async def fail():
        raise ValueError("background failed")

    failed = asyncio.create_task(fail())
    manager.track_background_task(failed, label="failure")
    await asyncio.gather(failed, other, current, return_exceptions=True)
    await asyncio.sleep(0)
    assert failed not in manager._background_tasks


def test_media_type_mapping_aliases_and_errors(monkeypatch):
    assert (
        music_search._ma_media_types(["songs", "artist", "songs"])[0]
        == music_search.MediaType.TRACK
    )
    assert len(music_search._ma_default_play_query_media_types()) == 4
    assert len(music_search._ma_media_types(None)) == 4
    with pytest.raises(ValueError, match="Unsupported"):
        music_search._ma_media_types(["video"])
    monkeypatch.setattr(music_search, "MediaType", None)
    with pytest.raises(RuntimeError, match="unavailable"):
        music_search._ma_media_type_lookup()


@pytest.mark.asyncio
async def test_search_compatible_direct_and_per_type_fallback():
    item = SimpleNamespace(name="A", uri="artist://a", media_type="artist")
    direct = SimpleNamespace(artists=[item])
    client = SimpleNamespace(music=SimpleNamespace(search=AsyncMock(return_value=direct)))
    assert (
        await music_search._ma_search_compatible(
            client, query="a", media_types=[music_search.MediaType.ARTIST], limit=5
        )
        is direct
    )

    calls = 0

    async def fallback_search(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1 or kwargs["media_types"] == [music_search.MediaType.ALBUM]:
            raise RuntimeError("NotImplementedError")
        return SimpleNamespace(
            artists=[item, item],
            albums=[],
            tracks=[],
            playlists=[],
            radio=[],
            radios=[],
            podcasts=[],
            audiobooks=[],
        )

    client.music.search = fallback_search
    merged = await music_search._ma_search_compatible(
        client,
        query="a",
        media_types=[music_search.MediaType.ARTIST, music_search.MediaType.ALBUM],
        limit=5,
    )
    assert merged.artists == [item]

    calls = 0

    async def all_unsupported(**kwargs):
        raise RuntimeError("NotImplementedError")

    client.music.search = all_unsupported
    with pytest.raises(RuntimeError, match="not implemented"):
        await music_search._ma_search_compatible(
            client, query="a", media_types=[music_search.MediaType.ARTIST], limit=5
        )


def test_select_search_item_biases_explicit_media_and_library():
    artist = SimpleNamespace(name="Muse", uri="library://artist/1", media_type="artist")
    track = SimpleNamespace(name="Muse", uri="provider://track/1", media_type="track")
    results = SimpleNamespace(
        artists=[artist],
        tracks=[track],
        albums=[],
        playlists=[],
        radio=[],
        podcasts=[],
        audiobooks=[],
    )
    selected, score = music_search._ma_select_search_item(results, "Muse artist")
    assert selected is artist and score > 0
    assert music_search._ma_select_search_item(results, "") == (None, 0)


def test_resolve_player_mapping_exact_origin_single_and_failure(monkeypatch):
    office = SimpleNamespace(player_id="office", name="Office Speaker", available=True)
    kitchen = SimpleNamespace(player_id="kitchen", name="Kitchen", available=False)
    players = Collection([office, kitchen])
    client = SimpleNamespace(players=players)
    monkeypatch.setattr(music_search, "_parse_music_area_player_map", lambda: {"study": "office"})
    assert music_search._ma_resolve_player(client, area="study")[1] == "configured_area_map"
    assert music_search._ma_resolve_player(client, player_id="office")[1] == "explicit_player_id"
    assert music_search._ma_resolve_player(client, area="Office")[1] == "name_match"
    voice_tool_run_state.origin_area_name = "Office"
    assert music_search._ma_resolve_player(client)[0] is office
    voice_tool_run_state.origin_area_name = None
    assert (
        music_search._ma_resolve_player(SimpleNamespace(players=Collection([office])))[1]
        == "single_available_player"
    )
    kitchen.available = True
    with pytest.raises(ValueError, match="required"):
        music_search._ma_resolve_player(client)
    with pytest.raises(ValueError, match="No Music Assistant"):
        music_search._ma_resolve_player(client, area="zzzzzz")
    with pytest.raises(RuntimeError, match="no players"):
        music_search._ma_resolve_player(SimpleNamespace(players=Collection([])), area="x")


@pytest.mark.asyncio
async def test_resolve_queue_and_post_verification(monkeypatch):
    queue = SimpleNamespace(queue_id="active", display_name="A", current_item=None)
    queues = Collection([queue])
    queues.get_active_queue = AsyncMock(return_value=queue)
    queues.get_queue_items = AsyncMock(
        return_value=[SimpleNamespace(queue_item_id="i", name="X", media_item=None, uri="u")]
    )
    player = SimpleNamespace(player_id="p", active_source="active")
    client = SimpleNamespace(player_queues=queues, players=Collection([player]))
    assert await music_search._ma_resolve_queue_id(client, "p") == "active"
    monkeypatch.setattr(settings.music_assistant, "post_action_settle_seconds", 0)
    verified = await music_search._ma_post_action_verification(client, "active")
    assert verified["first_items"][0]["queue_item_id"] == "i"
    queues.get_queue_items.side_effect = RuntimeError("inspect failed")
    assert "verification_error" in await music_search._ma_post_action_verification(client, "active")
