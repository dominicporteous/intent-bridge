"""Music Assistant search, matching, and player resolution."""

import asyncio
import difflib
import re
from typing import Any

try:
    from music_assistant_models.enums import MediaType
except Exception:  # pragma: no cover - deployment dependency guard
    MediaType = None

from intent_bridge.config import log, settings
from intent_bridge.core.text import normalize_search_text as _normalise_search_text
from intent_bridge.music_assistant.client import (
    _ma_media_summary,
    _ma_media_type,
    _ma_name,
    _ma_queue_item_summary,
    _ma_queue_summary,
    _ma_uri,
)
from intent_bridge.music_assistant.policy import (
    music_area_player_map as _parse_music_area_player_map,
)
from intent_bridge.runtime.execution import (
    voice_tool_run_state,
)


def _ma_media_type_lookup() -> dict[str, Any]:
    if MediaType is None:
        raise RuntimeError("music-assistant-models is unavailable")
    return {
        "artist": MediaType.ARTIST,
        "album": MediaType.ALBUM,
        "track": MediaType.TRACK,
        "playlist": MediaType.PLAYLIST,
        "radio": MediaType.RADIO,
        "podcast": MediaType.PODCAST,
        "audiobook": MediaType.AUDIOBOOK,
    }


def _ma_default_play_query_media_types() -> list[Any]:
    lookup = _ma_media_type_lookup()
    return [
        lookup["artist"],
        lookup["album"],
        lookup["track"],
        lookup["playlist"],
    ]


def _ma_media_types(media_types: list[str] | None) -> list[Any]:
    lookup = _ma_media_type_lookup()
    if not media_types:
        return _ma_default_play_query_media_types()
    result = []
    aliases = {
        "song": "track",
        "songs": "track",
        "tracks": "track",
        "artists": "artist",
        "albums": "album",
        "playlists": "playlist",
        "radio_station": "radio",
        "radio_stations": "radio",
        "podcasts": "podcast",
        "audiobooks": "audiobook",
    }
    for raw in media_types:
        value = str(raw or "").strip().casefold().replace(" ", "_")
        value = aliases.get(value, value)
        enum_value = lookup.get(value)
        if enum_value is None:
            raise ValueError(
                "Unsupported Music Assistant media type: "
                f"{raw!r}; allowed values are {', '.join(sorted(lookup))}"
            )
        if enum_value not in result:
            result.append(enum_value)
    return result


def _ma_search_groups(results: Any) -> list[tuple[str, str, list[Any]]]:
    groups: list[tuple[str, str, list[Any]]] = []
    specs = (
        ("artists", "artist", ("artists",)),
        ("albums", "album", ("albums",)),
        ("tracks", "track", ("tracks",)),
        ("playlists", "playlist", ("playlists",)),
        ("radio", "radio", ("radio", "radios")),
        ("podcasts", "podcast", ("podcasts",)),
        ("audiobooks", "audiobook", ("audiobooks",)),
    )
    for label, media_type, attrs in specs:
        values = None
        for attr in attrs:
            values = getattr(results, attr, None)
            if values is not None:
                break
        if values:
            groups.append((label, media_type, list(values)))
    return groups


def _ma_search_payload(results: Any, limit: int) -> dict[str, list[dict[str, Any]]]:
    payload: dict[str, list[dict[str, Any]]] = {}
    for label, media_type, items in _ma_search_groups(results):
        payload[label] = [_ma_media_summary(item, media_type) for item in items[:limit]]
    return payload


def _ma_media_type_label(media_type: Any) -> str:
    value = getattr(media_type, "value", media_type)
    return str(value or "").strip().casefold()


async def _ma_search_compatible(
    client: Any,
    *,
    query: str,
    media_types: list[Any],
    limit: int,
) -> Any:
    try:
        return await client.music.search(
            search_query=query,
            media_types=media_types,
            limit=limit,
        )
    except Exception as exc:
        if "NotImplementedError" not in str(exc):
            raise
        log.warning(
            "MA NATIVE SEARCH compatibility fallback query=%r media_types=%s error=%s",
            query,
            [_ma_media_type_label(item) for item in media_types],
            exc,
        )

    class _MergedSearchResults:
        pass

    merged = _MergedSearchResults()
    # Match all attribute names understood by _ma_search_groups().
    for attr in (
        "artists",
        "albums",
        "tracks",
        "playlists",
        "radio",
        "radios",
        "podcasts",
        "audiobooks",
    ):
        setattr(merged, attr, [])

    successes = 0
    skipped: list[str] = []
    for media_type in media_types:
        label = _ma_media_type_label(media_type)
        try:
            result = await client.music.search(
                search_query=query,
                media_types=[media_type],
                limit=limit,
            )
        except Exception as exc:
            if "NotImplementedError" in str(exc):
                skipped.append(label)
                log.info(
                    "MA NATIVE SEARCH media type unsupported on server query=%r media_type=%s",
                    query,
                    label,
                )
                continue
            raise

        successes += 1
        for attr in (
            "artists",
            "albums",
            "tracks",
            "playlists",
            "radio",
            "radios",
            "podcasts",
            "audiobooks",
        ):
            values = getattr(result, attr, None)
            if values:
                target = getattr(merged, attr)
                seen = {(_ma_uri(existing), _ma_name(existing)) for existing in target}
                for item in values:
                    key = (_ma_uri(item), _ma_name(item))
                    if key not in seen:
                        target.append(item)
                        seen.add(key)

    if not successes:
        raise RuntimeError(
            "Music Assistant search is not implemented for the requested media types"
        )

    if skipped:
        log.info(
            "MA NATIVE SEARCH compatibility result query=%r skipped_media_types=%s",
            query,
            skipped,
        )
    return merged


def _ma_select_search_item(results: Any, query: str) -> tuple[Any, float] | tuple[None, float]:
    q = _normalise_search_text(query)
    if not q:
        return None, 0.0

    type_bias = {
        "artist": 25.0,
        "track": 20.0,
        "playlist": 10.0,
        "album": 5.0,
        "radio": 0.0,
        "podcast": -5.0,
        "audiobook": -5.0,
    }
    explicit_hint: str | None = None
    q_words = set(q.split())
    for word, media_type in (
        ("artist", "artist"),
        ("album", "album"),
        ("playlist", "playlist"),
        ("song", "track"),
        ("track", "track"),
        ("radio", "radio"),
    ):
        if word in q_words:
            explicit_hint = media_type
            break

    best_item = None
    best_score = float("-inf")
    for _, fallback_type, items in _ma_search_groups(results):
        for index, item in enumerate(items):
            name = _normalise_search_text(_ma_name(item))
            if not name:
                continue
            media_type = _ma_media_type(item, fallback_type) or fallback_type
            similarity = difflib.SequenceMatcher(None, q, name).ratio() * 100.0
            score = similarity + type_bias.get(media_type, 0.0)
            if name == q:
                score += 100.0
            elif q in name or name in q:
                score += 30.0
            if explicit_hint:
                score += 100.0 if media_type == explicit_hint else -20.0
            uri = _ma_uri(item)
            if uri.startswith("library://"):
                score += 4.0
            score -= index * 0.05
            if score > best_score:
                best_item = item
                best_score = score
    return best_item, best_score


def _ma_player_match_score(player: Any, target: str) -> float:
    target_norm = _normalise_search_text(target)
    player_id = _normalise_search_text(getattr(player, "player_id", ""))
    name = _normalise_search_text(getattr(player, "name", ""))
    if not target_norm:
        return 0.0
    if target_norm in {player_id, name}:
        score = 1000.0
    elif name.startswith(target_norm) or player_id.startswith(target_norm):
        score = 850.0
    elif re.search(rf"\\b{re.escape(target_norm)}\\b", name):
        score = 800.0
    elif target_norm in name or target_norm in player_id:
        score = 700.0
    else:
        score = (
            max(
                difflib.SequenceMatcher(None, target_norm, name).ratio(),
                difflib.SequenceMatcher(None, target_norm, player_id).ratio(),
            )
            * 500.0
        )
    if not bool(getattr(player, "available", True)):
        score -= 400.0
    return score


def _ma_resolve_player(
    client: Any, *, area: str | None = None, player_id: str | None = None
) -> tuple[Any, str]:
    players = list(client.players)
    if not players:
        raise RuntimeError("Music Assistant has no players")

    mapping = _parse_music_area_player_map()
    if area:
        mapped = mapping.get(str(area).strip().casefold())
        if mapped:
            player = client.players.get(mapped)
            if player is not None:
                return player, "configured_area_map"
            log.warning(
                "MA NATIVE configured area mapping points to missing player area=%r player_id=%r",
                area,
                mapped,
            )

    if player_id:
        exact = client.players.get(str(player_id))
        if exact is not None:
            return exact, "explicit_player_id"

    target = str(player_id or area or "").strip()
    if not target:
        origin = str(voice_tool_run_state.origin_area_name or "").strip()
        if origin:
            target = origin
    if not target:
        available = [p for p in players if bool(getattr(p, "available", True))]
        if len(available) == 1:
            return available[0], "single_available_player"
        raise ValueError("A Music Assistant room/player is required")

    ranked = sorted(
        ((_ma_player_match_score(player, target), player) for player in players),
        key=lambda item: item[0],
        reverse=True,
    )
    score, player = ranked[0]
    if score < 275.0:
        raise ValueError(f"No Music Assistant player matches {target!r}")
    return player, "name_match"


async def _ma_resolve_queue_id(client: Any, player_id: str) -> str:
    active = await client.player_queues.get_active_queue(player_id)
    if active is not None and getattr(active, "queue_id", None):
        return str(active.queue_id)
    player = client.players.get(player_id)
    if player is not None:
        active_source = str(getattr(player, "active_source", "") or "")
        if active_source and client.player_queues.get(active_source) is not None:
            return active_source
    return player_id


async def _ma_post_action_verification(client: Any, queue_id: str) -> dict[str, Any]:
    if settings.music_assistant.post_action_settle_seconds:
        await asyncio.sleep(settings.music_assistant.post_action_settle_seconds)
    snapshot = _ma_queue_summary(client, queue_id)
    try:
        items = await client.player_queues.get_queue_items(queue_id, limit=5, offset=0)
        snapshot["first_items"] = [_ma_queue_item_summary(item) for item in items]
    except Exception as exc:
        # Never turn a confirmed state-changing server ACK into a failure merely
        # because post-action inspection failed. That could cause a replay.
        snapshot["verification_error"] = f"{type(exc).__name__}: {exc}"
    return snapshot
