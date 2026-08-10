"""Pure Music Assistant configuration and agent policy."""

from intent_bridge.config import settings

MUSIC_ASSISTANT_ALWAYS_WRITE_TOOLS = frozenset(
    {
        "ma_play_query",
        "ma_play_media",
        "ma_volume",
        "ma_playback",
        "ma_group",
        "ma_queue_item",
        "ma_transfer_queue",
    }
)


def parse_music_area_player_map(raw: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item or "=" not in item:
            continue
        area, player_id = item.split("=", 1)
        area = area.strip()
        player_id = player_id.strip()
        if area and player_id:
            mapping[area.casefold()] = player_id
    return mapping


def music_area_player_map() -> dict[str, str]:
    return parse_music_area_player_map(settings.music_assistant.area_player_map)


def music_assistant_agent_instructions(
    mapping: dict[str, str] | None = None,
) -> str:
    mapping = music_area_player_map() if mapping is None else mapping
    mapping_text = ""
    if mapping:
        friendly = ", ".join(f"{area} -> {player_id}" for area, player_id in mapping.items())
        mapping_text = (
            "\nConfigured authoritative area-to-player mapping: "
            f"{friendly}. Use it unless the user explicitly names another player."
        )

    return (
        """
MUSIC ASSISTANT AUTHORITY

Native Music Assistant WebSocket tools are connected and are authoritative for
music and speaker playback operations. Do not use Home Assistant media_player
services for Music Assistant music playback.

For ordinary requests of the form "play <music> in/on <room/player>", prefer
ma_play_query. It resolves the player, searches Music Assistant and starts the
selected result in ONE tool call. Pass the explicitly named room in area. If the
request has no explicit room, pass the trusted voice-origin area when available.
Do not call ma_search or ma_list_players before ma_play_query unless the request
actually requires browsing or disambiguation. For an artist-radio request such as
"play Foo Fighters radio", pass query="Foo Fighters radio" and radio_mode=true;
the bridge optimizes time-to-first-audio and leaves continuation to Music Assistant.

Use the lower-level native tools when needed:
- ma_list_players: inspect the event-updated in-memory player topology.
- ma_search / ma_browse: discover media URIs.
- ma_play_media: play a known URI on a known player/queue.
- ma_playback: play/pause/stop/next/previous/toggle.
- ma_volume: set/adjust/mute Music Assistant players.
- ma_group: join/leave speaker sync groups.
- ma_queue / ma_queue_item / ma_transfer_queue: queue operations.

An explicitly named room/player always overrides voice-origin context. Do not
call ha_search merely to discover a Music Assistant player.

After any successful Music Assistant write/dispatch, stop. Do not verify it with
another model-driven write and do not repeat it. The proxy treats the native
result as terminal and returns the spoken confirmation directly.
""".strip()
        + mapping_text
    )


__all__ = [
    "MUSIC_ASSISTANT_ALWAYS_WRITE_TOOLS",
    "music_area_player_map",
    "music_assistant_agent_instructions",
    "parse_music_area_player_map",
]
