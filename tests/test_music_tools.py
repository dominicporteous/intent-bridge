import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from agents.tool_context import ToolContext

from intent_bridge.music_assistant import tools as music_tools
from intent_bridge.music_assistant.client import MusicPlayDispatchResult


async def invoke(tool, **arguments):
    encoded = json.dumps(arguments)
    context = ToolContext(None, tool_name=tool.name, tool_call_id="test", tool_arguments=encoded)
    return json.loads(await tool.on_invoke_tool(context, encoded))


class Collection:
    def __init__(self, values=()):
        self.values = list(values)

    def __iter__(self):
        return iter(self.values)

    def get(self, key):
        return next(
            (
                x
                for x in self.values
                if key in {getattr(x, "player_id", None), getattr(x, "queue_id", None)}
            ),
            None,
        )


@pytest.fixture
def client():
    player = SimpleNamespace(
        player_id="office",
        name="Office",
        available=True,
        powered=True,
        state="idle",
        volume_level=20,
        volume_muted=False,
        active_source="office",
        synced_to=None,
    )
    queue = SimpleNamespace(
        queue_id="office",
        display_name="Office",
        state="idle",
        current_index=0,
        elapsed_time=0,
        shuffle_enabled=False,
        repeat_mode="off",
        current_item=None,
    )
    players = Collection([player])
    for name in (
        "volume_set",
        "group_volume",
        "volume_up",
        "group_volume_up",
        "volume_down",
        "group_volume_down",
        "volume_mute",
        "group_many",
        "ungroup_many",
    ):
        setattr(players, name, AsyncMock())
    queues = Collection([queue])
    queues.get_active_queue = AsyncMock(return_value=queue)
    queues.get_queue_items = AsyncMock(return_value=[])
    for name in (
        "play",
        "seek",
        "pause",
        "stop",
        "play_pause",
        "next",
        "previous",
        "shuffle",
        "repeat",
        "clear",
        "move_up",
        "move_down",
        "move_next",
        "delete_item",
        "transfer",
    ):
        setattr(queues, name, AsyncMock())
    return SimpleNamespace(
        players=players,
        player_queues=queues,
        music=SimpleNamespace(
            browse=AsyncMock(
                return_value=[SimpleNamespace(name="Song", uri="track://1", media_type="track")]
            ),
            search=AsyncMock(
                return_value=SimpleNamespace(
                    artists=[],
                    albums=[],
                    tracks=[SimpleNamespace(name="Song", uri="track://1", media_type="track")],
                    playlists=[],
                    radio=[],
                    podcasts=[],
                    audiobooks=[],
                )
            ),
        ),
    )


@pytest.fixture
def manager(client, monkeypatch):
    value = SimpleNamespace(wait_ready=AsyncMock(return_value=client))

    async def run_serialized(name, operation):
        return await operation(client)

    value.run_serialized = run_serialized
    monkeypatch.setattr(music_tools, "_ma_native_manager", lambda: value)
    monkeypatch.setattr(
        music_tools, "_ma_resolve_queue_id", AsyncMock(side_effect=lambda client, value: value)
    )
    monkeypatch.setattr(
        music_tools, "_ma_post_action_verification", AsyncMock(return_value={"cached": True})
    )
    return value


@pytest.mark.asyncio
async def test_list_search_and_browse(client, manager, monkeypatch):
    assert (await invoke(music_tools.ma_list_players))["players"][0]["player_id"] == "office"
    assert not (await invoke(music_tools.ma_search, query="", media_types=None, limit=10))[
        "success"
    ]
    search = await invoke(music_tools.ma_search, query="Song", media_types=["track"], limit=100)
    assert search["results"]["tracks"][0]["name"] == "Song"
    browse = await invoke(music_tools.ma_browse, path=None, limit=10, offset=-1)
    assert browse["items"][0]["uri"] == "track://1"
    monkeypatch.setattr(
        music_tools, "_ma_native_manager", lambda: (_ for _ in ()).throw(RuntimeError("offline"))
    )
    assert not (await invoke(music_tools.ma_list_players))["success"]
    assert not (await invoke(music_tools.ma_search, query="x", media_types=None, limit=1))[
        "success"
    ]
    assert not (await invoke(music_tools.ma_browse, path="x", limit=1, offset=0))["success"]


@pytest.mark.asyncio
async def test_play_query_and_play_media(client, manager, monkeypatch):
    monkeypatch.setattr(
        music_tools,
        "_ma_resolve_player",
        lambda client, area, player_id: (client.players.get("office"), "exact"),
    )
    monkeypatch.setattr(music_tools, "_ma_resolve_queue_id", AsyncMock(return_value="office"))
    results = SimpleNamespace(
        artists=[],
        albums=[],
        tracks=[SimpleNamespace(name="Song", uri="track://1", media_type="track")],
        playlists=[],
        radio=[],
        podcasts=[],
        audiobooks=[],
    )
    monkeypatch.setattr(music_tools, "_ma_search_compatible", AsyncMock(return_value=results))
    dispatch = MusicPlayDispatchResult(True, True, False, False)
    monkeypatch.setattr(music_tools, "_ma_dispatch_play_media", AsyncMock(return_value=dispatch))
    played = await invoke(
        music_tools.ma_play_query, query="Song", area="Office", player_id=None, radio_mode=False
    )
    assert played["success"] and played["message"] == "Playing."
    assert not (
        await invoke(
            music_tools.ma_play_query, query="", area=None, player_id=None, radio_mode=False
        )
    )["success"]
    assert not (
        await invoke(
            music_tools.ma_play_media, queue_id="", media="x", option="play", radio_mode=False
        )
    )["success"]
    media = await invoke(
        music_tools.ma_play_media,
        queue_id="office",
        media="track://1",
        option="next",
        radio_mode=False,
    )
    assert media["message"] == "Added as next."


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("play", "Playing."),
        ("pause", "Paused."),
        ("stop", "Stopped."),
        ("toggle", "Playback toggled."),
        ("next", "Skipped to next."),
        ("previous", "Previous."),
    ],
)
async def test_playback_commands(client, manager, command, expected):
    result = await invoke(
        music_tools.ma_playback,
        queue_id="office",
        command=command,
        seek_seconds=5 if command == "play" else None,
    )
    assert result["message"] == expected


@pytest.mark.asyncio
async def test_volume_validation_and_operations(client, manager):
    assert not (
        await invoke(
            music_tools.ma_volume,
            player_id="office",
            level=None,
            adjust=None,
            mute=None,
            group=False,
        )
    )["success"]
    assert not (
        await invoke(
            music_tools.ma_volume,
            player_id="office",
            level=None,
            adjust=None,
            mute=True,
            group=True,
        )
    )["success"]
    assert (
        await invoke(
            music_tools.ma_volume,
            player_id="office",
            level=120,
            adjust=None,
            mute=None,
            group=False,
        )
    )["message"] == "Volume set to 100%."
    assert (
        await invoke(
            music_tools.ma_volume,
            player_id="office",
            level=None,
            adjust="up",
            mute=None,
            group=True,
        )
    )["message"] == "Volume increased."
    assert (
        await invoke(
            music_tools.ma_volume,
            player_id="office",
            level=None,
            adjust="down",
            mute=None,
            group=False,
        )
    )["message"] == "Volume decreased."
    assert (
        await invoke(
            music_tools.ma_volume,
            player_id="office",
            level=None,
            adjust=None,
            mute=False,
            group=False,
        )
    )["message"] == "Unmuted."
    assert not (
        await invoke(
            music_tools.ma_volume, player_id="missing", level=1, adjust=None, mute=None, group=False
        )
    )["success"]


@pytest.mark.asyncio
async def test_group_queue_items_and_transfer(client, manager):
    assert not (
        await invoke(music_tools.ma_group, action="join", player_ids=[], target_player_id="office")
    )["success"]
    assert not (
        await invoke(
            music_tools.ma_group, action="join", player_ids=["child"], target_player_id=None
        )
    )["success"]
    assert (
        await invoke(
            music_tools.ma_group,
            action="join",
            player_ids=["office", "child"],
            target_player_id="office",
        )
    )["message"] == "Speakers grouped."
    assert (
        await invoke(
            music_tools.ma_group, action="leave", player_ids=["child"], target_player_id=None
        )
    )["success"]
    queue = await invoke(
        music_tools.ma_queue,
        queue_id="office",
        get_items=True,
        shuffle=True,
        repeat="all",
        clear=True,
    )
    assert queue["changed"] and len(queue["changes_applied"]) == 3
    for action in ("move_up", "move_down", "move_next", "remove"):
        assert (
            await invoke(
                music_tools.ma_queue_item, queue_id="office", item_id="item", action=action
            )
        )["success"]
    transfer = await invoke(
        music_tools.ma_transfer_queue, source_queue_id="office", target_queue_id="kitchen"
    )
    assert transfer["message"] == "Playback moved."
