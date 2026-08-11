import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from intent_bridge.config import settings
from intent_bridge.music_assistant import playback as music_playback


class Manager:
    def __init__(self):
        self.inflight = {}
        self.background = []
        self.generations = {}
        self.seeds = {}

    def get_inflight_playback(self, queue, fingerprint):
        return self.inflight.get((queue, fingerprint))

    def register_inflight_playback(self, queue, fingerprint, task, label):
        record = SimpleNamespace(task=task, label=label, started_at=0)
        self.inflight[(queue, fingerprint)] = record
        return record

    def clear_inflight_playback(self, queue, fingerprint, task):
        current = self.inflight.get((queue, fingerprint))
        if current is not None and current.task is task:
            self.inflight.pop((queue, fingerprint), None)

    def begin_queue_playback_generation(self, queue):
        self.generations[queue] = self.generations.get(queue, 0) + 1
        return self.generations[queue]

    def is_current_queue_playback_generation(self, queue, generation):
        return self.generations.get(queue) == generation

    def last_radio_seed(self, artist):
        return self.seeds.get(artist)

    def remember_radio_seed(self, artist, seed):
        self.seeds[artist] = seed

    def track_background_task(self, task, *, label):
        self.background.append(task)

    @property
    def inflight_playback_count(self):
        return len(self.inflight)


@pytest.mark.asyncio
async def test_dispatch_play_immediate_duplicate_timeout_and_error(monkeypatch):
    manager = Manager()
    queues = SimpleNamespace(play_media=AsyncMock(return_value=None))
    client = SimpleNamespace(player_queues=queues)
    result = await music_playback._ma_dispatch_play_media(
        manager, client, queue_id="q", media="track://1", option="play", radio_mode=False, label="x"
    )
    assert result.command_acknowledged and not result.still_processing

    fingerprint = '{"media":["track://1"],"option":"play","queue_id":"q","radio_mode":false}'
    manager.inflight[("q", fingerprint)] = SimpleNamespace(label="old", started_at=0, task=None)
    duplicate = await music_playback._ma_dispatch_play_media(
        manager, client, queue_id="q", media="track://1", option="play", radio_mode=False, label="x"
    )
    assert duplicate.duplicate_suppressed
    manager.inflight.clear()

    gate = asyncio.Event()

    async def slow_play(**kwargs):
        await gate.wait()

    queues.play_media = AsyncMock(side_effect=slow_play)
    monkeypatch.setattr(settings.music_assistant, "play_ack_timeout_seconds", 0.001)
    monkeypatch.setattr(
        music_playback.assistant_feedback, "begin", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(music_playback.assistant_feedback, "complete", AsyncMock())
    processing = await music_playback._ma_dispatch_play_media(
        manager,
        client,
        queue_id="q2",
        media=["a", "b"],
        option="replace",
        radio_mode=True,
        label="slow",
        origin_context={},
    )
    assert processing.still_processing and processing.command_dispatched
    gate.set()
    await asyncio.gather(*manager.background, return_exceptions=True)

    queues.play_media = AsyncMock(side_effect=RuntimeError("rejected"))
    with pytest.raises(RuntimeError, match="rejected"):
        await music_playback._ma_dispatch_play_media(
            manager,
            client,
            queue_id="q3",
            media="bad",
            option="play",
            radio_mode=False,
            label="bad",
        )


def test_provider_item_id_and_queue_markers():
    assert music_playback._ma_media_provider_and_item_id(
        SimpleNamespace(provider="", item_id="", uri="spotify://artist/abc/def")
    ) == ("spotify", "abc/def")
    assert music_playback._ma_media_provider_and_item_id(
        SimpleNamespace(provider="library", item_id="1", uri="")
    ) == ("library", "1")
    queues = SimpleNamespace(get=lambda key: None)
    client = SimpleNamespace(player_queues=queues)
    assert music_playback._ma_queue_playback_marker(client, "q") == ("", "", "", "")
    queue = SimpleNamespace(state="idle", current_item=None)
    queues.get = lambda key: queue
    assert music_playback._ma_queue_playback_marker(client, "q")[0] == "idle"
    media = SimpleNamespace(uri="track://1", name="Song")
    queue.state = "playing"
    queue.current_item = SimpleNamespace(queue_item_id="i", media_item=media, uri="", name="")
    assert music_playback._ma_queue_playback_marker(client, "q") == (
        "playing",
        "i",
        "track://1",
        "song",
    )


@pytest.mark.parametrize(
    ("marker", "baseline", "uri", "name", "expected"),
    [
        (("idle", "", "", ""), ("idle", "", "", ""), None, None, False),
        (("playing", "i", "u", "n"), ("idle", "", "", ""), None, None, True),
        (("playing", "new", "u", "n"), ("playing", "old", "x", "x"), "u", None, True),
        (("playing", "new", "u", "name"), ("playing", "old", "x", "x"), None, "Name", True),
        (
            ("playing", "same", "same", "same"),
            ("playing", "same", "same", "same"),
            None,
            None,
            False,
        ),
    ],
)
def test_first_audio_matching(marker, baseline, uri, name, expected):
    assert (
        music_playback._ma_first_audio_matches(
            marker, baseline=baseline, expected_uri=uri, expected_name=name
        )
        is expected
    )


@pytest.mark.asyncio
async def test_wait_for_first_audio_success_timeout_and_superseded(monkeypatch):
    queue = SimpleNamespace(
        state="playing",
        current_item=SimpleNamespace(queue_item_id="new", uri="u", media_item=None, name="N"),
    )
    client = SimpleNamespace(player_queues=SimpleNamespace(get=lambda key: queue))
    found = await music_playback._ma_wait_for_first_audio(
        client,
        queue_id="q",
        baseline=("idle", "", "", ""),
        expected_uri="u",
        expected_name="N",
        timeout=0.01,
    )
    assert found[0]
    manager = SimpleNamespace(is_current_queue_playback_generation=lambda q, g: False)
    superseded = await music_playback._ma_wait_for_first_audio(
        client,
        queue_id="q",
        baseline=found[2],
        expected_uri=None,
        expected_name=None,
        timeout=0.01,
        manager=manager,
        generation=1,
    )
    assert not superseded[0]
    queue.state = "idle"
    monkeypatch.setattr(settings.music_assistant, "first_audio_poll_seconds", 0.001)
    timed = await music_playback._ma_wait_for_first_audio(
        client,
        queue_id="q",
        baseline=("idle", "", "", ""),
        expected_uri=None,
        expected_name=None,
        timeout=0.002,
    )
    assert not timed[0]


@pytest.mark.asyncio
async def test_fast_artist_radio_immediate_and_validation(monkeypatch):
    manager = Manager()
    artist = SimpleNamespace(
        name="Artist",
        uri="spotify://artist/a",
        provider="spotify",
        item_id="a",
        media_type="artist",
    )
    track = SimpleNamespace(
        name="Song", uri="spotify://track/t", provider="spotify", item_id="t", media_type="track"
    )
    queue = SimpleNamespace(state="idle", current_item=None)

    async def play_media(**kwargs):
        queue.state = "playing"
        queue.current_item = SimpleNamespace(
            queue_item_id="new", uri=track.uri, media_item=track, name=track.name
        )

    queues = SimpleNamespace(
        get=lambda key: queue,
        play_media=play_media,
        dont_stop_the_music=AsyncMock(),
    )
    client = SimpleNamespace(
        music=SimpleNamespace(get_artist_tracks=AsyncMock(return_value=[track])),
        player_queues=queues,
    )
    monkeypatch.setattr(settings.music_assistant, "radio_seed_strategy", "first")
    monkeypatch.setattr(settings.music_assistant, "play_ack_timeout_seconds", 0.05)
    result = await music_playback._ma_dispatch_fast_artist_radio(
        manager, client, queue_id="q", artist=artist, origin_context=None, label="radio"
    )
    assert result.command_dispatched and result.seed["name"] == "Song"
    await asyncio.gather(*manager.background, return_exceptions=True)
    queues.dont_stop_the_music.assert_awaited_once()

    bad = SimpleNamespace(name="Bad", uri="", provider="", item_id="")
    with pytest.raises(ValueError, match="missing"):
        await music_playback._ma_dispatch_fast_artist_radio(
            manager, client, queue_id="q", artist=bad, origin_context=None, label="bad"
        )
