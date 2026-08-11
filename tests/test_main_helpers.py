from enum import Enum
from types import SimpleNamespace

import pytest

from intent_bridge.agents import results as tool_results
from intent_bridge.config import settings
from intent_bridge.core.tool_output import (
    serialise_tool_output,
    tool_output_failed,
    tool_output_mapping,
)
from intent_bridge.indicators import controller as indicators
from intent_bridge.music_assistant import client as music_assistant
from intent_bridge.music_assistant import playback as music_playback
from intent_bridge.music_assistant import search as music_search


class Kind(Enum):
    TRACK = "track"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, ""),
        ({"ok": True}, '{"ok": true}'),
        ([1, 2], "[1, 2]"),
        ("plain", "plain"),
    ],
)
def test_serialise_tool_output(value, expected):
    assert serialise_tool_output(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("", None),
        ('{"success": true}', {"success": True}),
        ("[]", None),
        ({"text": '{"answer": 1}'}, {"answer": 1}),
        (["bad", {"success": False}], {"success": False}),
    ],
)
def test_tool_output_mapping(value, expected):
    assert tool_output_mapping(value) == expected


@pytest.mark.parametrize(
    ("value", "failed"),
    [
        ({"success": False}, True),
        ({"success": True}, False),
        ({"error": "boom"}, True),
        ("request timed out", True),
        ("all good", False),
    ],
)
def test_tool_output_failed(value, failed):
    assert tool_output_failed(value) is failed


def test_music_media_summaries():
    artist = SimpleNamespace(name="Artist")
    album = SimpleNamespace(name="Album")
    item = SimpleNamespace(
        name="Track",
        uri="provider://track/1",
        media_type=Kind.TRACK,
        provider="provider",
        item_id="1",
        version="live",
        artists=[artist],
        album=album,
    )
    assert music_assistant._ma_enum_value(Kind.TRACK) == "track"
    assert music_assistant._ma_name(item) == "Track"
    assert music_assistant._ma_uri(item) == "provider://track/1"
    assert music_assistant._ma_media_type(item) == "track"
    assert music_assistant._ma_media_summary(item) == {
        "name": "Track",
        "uri": "provider://track/1",
        "media_type": "track",
        "provider": "provider",
        "item_id": "1",
        "version": "live",
        "artists": ["Artist"],
        "album": "Album",
    }
    queue_item = SimpleNamespace(queue_item_id="q1", name="Queued", media_item=item)
    assert music_assistant._ma_queue_item_summary(queue_item)["media"]["name"] == "Track"
    assert music_assistant._ma_queue_item_summary(
        SimpleNamespace(item_id="q2", name="Raw", media_item=None, uri="x://y")
    ) == {"queue_item_id": "q2", "name": "Raw", "uri": "x://y"}


def test_music_search_groups_and_payload():
    results = SimpleNamespace(
        artists=[SimpleNamespace(name="A", uri="a")],
        albums=[],
        tracks=[SimpleNamespace(name="T", uri="t")],
        playlists=None,
        radios=[SimpleNamespace(name="R", uri="r")],
        podcasts=[],
        audiobooks=[],
    )
    groups = music_search._ma_search_groups(results)
    assert [group[0] for group in groups] == ["artists", "tracks", "radio"]
    assert music_search._ma_search_payload(results, 1)["radio"][0]["media_type"] == "radio"


def test_radio_candidates_filter_duplicates_unavailable_and_limit(monkeypatch):
    monkeypatch.setattr(settings.music_assistant, "radio_seed_top_n", 2)
    tracks = [
        SimpleNamespace(uri="one", available=True),
        SimpleNamespace(uri="one", available=True),
        SimpleNamespace(uri="two", available=False),
        SimpleNamespace(uri="three", available=True),
        SimpleNamespace(uri="four", available=True),
    ]
    assert [item.uri for item in music_playback._ma_radio_seed_candidates(tracks)] == [
        "one",
        "three",
    ]


@pytest.mark.parametrize("strategy", ["first", "random", "weighted"])
def test_choose_radio_seed_strategies(monkeypatch, strategy):
    monkeypatch.setattr(settings.music_assistant, "radio_seed_strategy", strategy)
    monkeypatch.setattr(settings.music_assistant, "radio_seed_top_n", 8)
    monkeypatch.setattr(music_playback.random, "choice", lambda values: values[0])
    monkeypatch.setattr(music_playback.random, "choices", lambda values, weights, k: [values[0]])
    remembered = []
    manager = SimpleNamespace(
        last_radio_seed=lambda uri: "one",
        remember_radio_seed=lambda artist, track: remembered.append((artist, track)),
    )
    tracks = [
        SimpleNamespace(name="One", uri="one"),
        SimpleNamespace(name="Two", uri="two"),
    ]
    selected, rank, candidates = music_playback._ma_choose_radio_seed(
        manager, artist_uri="artist", tracks=tracks
    )
    assert selected.uri == "two"
    assert rank == 2
    assert candidates[0]["name"] == "One"
    assert remembered == [("artist", "two")]


def test_choose_radio_seed_requires_candidates():
    manager = SimpleNamespace(last_radio_seed=lambda _: None, remember_radio_seed=lambda *_: None)
    with pytest.raises(ValueError, match="no available top tracks"):
        music_playback._ma_choose_radio_seed(manager, artist_uri="artist", tracks=[])


@pytest.mark.parametrize(
    ("target", "available", "minimum"),
    [("Kitchen", True, 1000), ("Kit", True, 850), ("speaker", True, 700), ("Kitchen", False, 600)],
)
def test_player_match_score(target, available, minimum):
    player = SimpleNamespace(player_id="kitchen-speaker", name="Kitchen", available=available)
    assert music_search._ma_player_match_score(player, target) >= minimum
    assert music_search._ma_player_match_score(player, "") == 0


@pytest.mark.parametrize(
    ("tool", "output", "expected"),
    [
        ("ma_play_query", {"message": "Starting Jazz."}, "Starting Jazz."),
        ("ma_play_media", "added as next", "Added next."),
        ("ma_volume", "Volume set to 42%", "Volume set to 42 percent."),
        ("ma_volume", "muted", "Muted."),
        ("ma_playback", "paused", "Paused."),
        ("ma_transfer_queue", "ok", "Playback moved."),
        ("ma_group", "removed from group", "Speakers ungrouped."),
        ("ma_queue", "cleared", settings.api.action_confirmation),
    ],
)
def test_music_terminal_speech(tool, output, expected):
    assert tool_results._music_assistant_terminal_speech(tool, output) == expected


def test_music_failure_and_write_detection():
    assert tool_results._music_assistant_output_failed("")
    assert tool_results._music_assistant_output_failed("player not found")
    assert not tool_results._music_assistant_output_failed('{"success": true}')
    assert tool_results._is_music_assistant_write_result("ma_volume", "ok")
    assert tool_results._is_music_assistant_write_result("ma_queue", {"changed": True})
    assert tool_results._is_music_assistant_write_result("ma_queue", "shuffle enabled")
    assert not tool_results._is_music_assistant_write_result("ma_search", "ok")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("green", [0, 255, 0]),
        ("#102030", [16, 32, 48]),
        ("1, 2, 255", [1, 2, 255]),
        ("current", None),
        ("#broken", None),
        ("300,0,0", None),
    ],
)
def test_indicator_colour_parsing(monkeypatch, value, expected):
    monkeypatch.setattr(settings.assistant, "led_color", value)
    assert indicators._configured_indicator_rgb() == expected


def test_indicator_effect_and_light_helpers(monkeypatch):
    monkeypatch.setattr(settings.assistant, "led_effect", "pulse")
    attrs = {"effect_list": ["None", "Slow Pulse"], "supported_color_modes": ["rgb"]}
    assert indicators._find_configured_native_effect(attrs) == "Slow Pulse"
    assert indicators._configured_effect_wants_software_pulse()
    assert indicators._find_neutral_native_effect(attrs) == "None"
    assert indicators._light_supports_colour(attrs)
    assert not indicators._light_supports_colour({})


def test_snapshot_restore_light_data_clamps_and_preserves_visual_state():
    snapshot = indicators.AssistantLedSnapshot(
        entity_id="light.status",
        domain="light",
        state="on",
        attributes={
            "brightness": 999,
            "effect": "pulse",
            "color_mode": "rgb",
            "rgb_color": (1, 2, 3),
        },
    )
    assert indicators._snapshot_restore_light_data(snapshot) == {
        "brightness": 255,
        "effect": "pulse",
        "rgb_color": [1, 2, 3],
    }
