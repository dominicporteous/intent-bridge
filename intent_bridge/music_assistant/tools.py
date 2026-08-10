"""Agent-facing Music Assistant tools."""

import re
from typing import Any, Literal

from agents import (
    function_tool,
)

try:
    from music_assistant_models.enums import QueueOption, RepeatMode
except Exception:  # pragma: no cover - deployment dependency guard
    QueueOption = None
    RepeatMode = None

from intent_bridge.config import log, settings
from intent_bridge.indicators.controller import (
    _voice_origin_snapshot,
)
from intent_bridge.music_assistant.client import (
    _ma_media_summary,
    _ma_native_manager,
    _ma_player_summary,
    _ma_queue_item_summary,
    _ma_queue_summary,
    _ma_uri,
)
from intent_bridge.music_assistant.playback import (
    _ma_dispatch_fast_artist_radio,
    _ma_dispatch_play_media,
)
from intent_bridge.music_assistant.search import (
    _ma_default_play_query_media_types,
    _ma_media_type_label,
    _ma_media_types,
    _ma_post_action_verification,
    _ma_resolve_player,
    _ma_resolve_queue_id,
    _ma_search_compatible,
    _ma_search_payload,
    _ma_select_search_item,
)
from intent_bridge.runtime.execution import (
    _json_tool_result,
)


@function_tool
async def ma_list_players() -> str:
    """List Music Assistant players from the live event-updated WebSocket state."""
    try:
        manager = _ma_native_manager()
        client = await manager.wait_ready()
        players = [_ma_player_summary(client, player) for player in client.players]
        return _json_tool_result(
            {
                "success": True,
                "transport": "native_websocket",
                "players": players,
            }
        )
    except Exception as exc:
        log.warning("MA NATIVE ma_list_players failed: %s", exc)
        return _json_tool_result({"success": False, "error": str(exc)})


@function_tool
async def ma_search(
    query: str,
    media_types: list[
        Literal["artist", "album", "track", "playlist", "radio", "podcast", "audiobook"]
    ]
    | None = None,
    limit: int = 10,
) -> str:
    """Search Music Assistant directly over its persistent WebSocket client."""
    query = str(query or "").strip()
    if not query:
        return _json_tool_result({"success": False, "error": "query is required"})
    limit = max(1, min(50, int(limit or settings.music_assistant.search_default_limit)))
    try:
        selected_types = _ma_media_types(media_types)
        manager = _ma_native_manager()
        log.info(
            "MA NATIVE SEARCH request query=%r requested_media_types=%r enum_media_types=%s limit=%s",
            query,
            media_types,
            [_ma_media_type_label(item) for item in selected_types],
            limit,
        )

        async def operation(client: Any):
            return await _ma_search_compatible(
                client,
                query=query,
                media_types=selected_types,
                limit=limit,
            )

        results = await manager.run_serialized("ma_search", operation)
        return _json_tool_result(
            {
                "success": True,
                "query": query,
                "results": _ma_search_payload(results, limit),
            }
        )
    except Exception as exc:
        log.warning("MA NATIVE ma_search failed query=%r: %s", query, exc)
        return _json_tool_result({"success": False, "error": str(exc)})


@function_tool
async def ma_browse(
    path: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> str:
    """Browse Music Assistant directly over its persistent WebSocket client."""
    path = str(path).strip() if path is not None else None
    limit = max(1, min(100, int(limit)))
    offset = max(0, int(offset))
    try:
        manager = _ma_native_manager()

        async def operation(client: Any):
            return await client.music.browse(path=path)

        items = await manager.run_serialized("ma_browse", operation)
        selected = list(items)[offset : offset + limit]
        return _json_tool_result(
            {
                "success": True,
                "path": path,
                "offset": offset,
                "limit": limit,
                "items": [_ma_media_summary(item) for item in selected],
            }
        )
    except Exception as exc:
        log.warning("MA NATIVE ma_browse failed path=%r: %s", path, exc)
        return _json_tool_result({"success": False, "error": str(exc)})


@function_tool
async def ma_play_query(
    query: str,
    area: str | None = None,
    player_id: str | None = None,
    radio_mode: bool = False,
) -> str:
    """Search and optimistically start music on a room/player in one native call.

    Prefer this for normal voice requests like "play Taylor Swift in the office".
    The proxy resolves player/queue/media deterministically. Artist-radio requests
    use v6.9 fast-start: one MA top-track seed is played first, then MA Don't Stop
    Music Assistant owns radio continuation. Long preparation never holds the voice turn.
    """
    query = str(query or "").strip()
    if not query:
        return _json_tool_result({"success": False, "error": "query is required"})
    try:
        selected_types = _ma_default_play_query_media_types()
        manager = _ma_native_manager()
        # "X radio" is an instruction to enable radio mode, not part of the
        # artist name. Removing the trailing word improves deterministic search.
        search_query = query
        if radio_mode:
            stripped = re.sub(r"\s+radio\s*$", "", query, flags=re.IGNORECASE).strip()
            if stripped:
                search_query = stripped
        origin_context = _voice_origin_snapshot()
        log.info(
            "MA NATIVE PLAY_QUERY search_types=%s query=%r search_query=%r area=%r player_id=%r radio_mode=%s",
            [_ma_media_type_label(item) for item in selected_types],
            query,
            search_query,
            area,
            player_id,
            radio_mode,
        )

        async def operation(client: Any):
            player, match_reason = _ma_resolve_player(client, area=area, player_id=player_id)
            resolved_player_id = str(getattr(player, "player_id", "") or "")
            queue_id = await _ma_resolve_queue_id(client, resolved_player_id)
            results = await _ma_search_compatible(
                client,
                query=search_query,
                media_types=selected_types,
                limit=settings.music_assistant.search_default_limit,
            )
            selected, score = _ma_select_search_item(results, search_query)
            if selected is None or not _ma_uri(selected):
                raise ValueError(f"No Music Assistant result found for {search_query!r}")
            media_uri = _ma_uri(selected)
            selected_summary = _ma_media_summary(selected)
            log.info(
                "MA NATIVE PLAY_QUERY query=%r area=%r requested_player_id=%r "
                "resolved_player_id=%r queue_id=%r media=%r media_type=%r score=%.2f radio_mode=%s",
                query,
                area,
                player_id,
                resolved_player_id,
                queue_id,
                media_uri,
                selected_summary.get("media_type"),
                score,
                radio_mode,
            )
            selected_media_type = str(selected_summary.get("media_type") or "").casefold()

            # v6.9: artist radio always uses the same fast-start contract on all
            # MA schema versions. Start one MA-ranked seed track immediately,
            # then enable MA Don't Stop The Music once that seed is PLAYING.
            if radio_mode and selected_media_type == "artist":
                fast_radio = await _ma_dispatch_fast_artist_radio(
                    manager,
                    client,
                    queue_id=queue_id,
                    artist=selected,
                    origin_context=origin_context,
                    label=f"ma_play_query:{queue_id}:{media_uri}:fast_artist_radio",
                )
                verification = _ma_queue_summary(client, queue_id)
                message = (
                    f"Playing {search_query} radio."
                    if fast_radio.first_audio_observed
                    else f"Starting {search_query} radio."
                )
                return {
                    "success": True,
                    "message": message,
                    "command_acknowledged": fast_radio.command_acknowledged,
                    "command_dispatched": fast_radio.command_dispatched,
                    "still_processing": fast_radio.still_processing,
                    "duplicate_suppressed": fast_radio.duplicate_suppressed,
                    "first_audio_observed": fast_radio.first_audio_observed,
                    "first_audio_seconds": fast_radio.first_audio_seconds,
                    "query": query,
                    "search_query": search_query,
                    "area": area,
                    "player_id": resolved_player_id,
                    "player_name": str(getattr(player, "name", "") or resolved_player_id),
                    "player_match": match_reason,
                    "queue_id": queue_id,
                    "selected_media": selected_summary,
                    "radio_seed": fast_radio.seed,
                    "radio_seed_rank": fast_radio.seed_rank,
                    "radio_seed_pool": fast_radio.seed_pool,
                    "radio_seed_strategy": settings.music_assistant.radio_seed_strategy,
                    "option": "replace",
                    "radio_mode": True,
                    "radio_fast_start": True,
                    "radio_continuation": "music_assistant_dont_stop_the_music",
                    "queue_after": verification,
                }

            assert QueueOption is not None
            dispatch = await _ma_dispatch_play_media(
                manager,
                client,
                queue_id=queue_id,
                media=media_uri,
                option=QueueOption.PLAY,
                radio_mode=bool(radio_mode),
                label=f"ma_play_query:{queue_id}:{media_uri}:radio={bool(radio_mode)}",
                origin_context=origin_context,
            )
            verification = (
                await _ma_post_action_verification(client, queue_id)
                if dispatch.command_acknowledged
                else _ma_queue_summary(client, queue_id)
            )
            if dispatch.command_acknowledged:
                message = "Playing."
            elif radio_mode:
                message = f"Starting {search_query} radio."
            else:
                message = f"Starting {search_query}."
            return {
                "success": True,
                "message": message,
                "command_acknowledged": dispatch.command_acknowledged,
                "command_dispatched": dispatch.command_dispatched,
                "still_processing": dispatch.still_processing,
                "duplicate_suppressed": dispatch.duplicate_suppressed,
                "query": query,
                "search_query": search_query,
                "area": area,
                "player_id": resolved_player_id,
                "player_name": str(getattr(player, "name", "") or resolved_player_id),
                "player_match": match_reason,
                "queue_id": queue_id,
                "selected_media": selected_summary,
                "option": "play",
                "radio_mode": bool(radio_mode),
                "radio_fast_start": False,
                "queue_after": verification,
            }

        payload = await manager.run_serialized("ma_play_query", operation)
        return _json_tool_result(payload)
    except Exception as exc:
        log.exception("MA NATIVE ma_play_query failed query=%r area=%r", query, area)
        return _json_tool_result({"success": False, "error": str(exc)})


@function_tool
async def ma_play_media(
    queue_id: str,
    media: str | list[str],
    option: Literal["play", "replace", "next", "add"] = "play",
    radio_mode: bool = False,
) -> str:
    """Play a known Music Assistant media URI on a player/queue."""
    queue_id = str(queue_id or "").strip()
    if not queue_id:
        return _json_tool_result({"success": False, "error": "queue_id is required"})
    try:
        manager = _ma_native_manager()

        async def operation(client: Any):
            resolved_queue = await _ma_resolve_queue_id(client, queue_id)
            assert QueueOption is not None
            option_value = QueueOption(str(option))
            log.info(
                "MA NATIVE PLAY_MEDIA requested_queue=%r resolved_queue=%r media=%r option=%s radio_mode=%s",
                queue_id,
                resolved_queue,
                media,
                option,
                radio_mode,
            )
            dispatch = await _ma_dispatch_play_media(
                manager,
                client,
                queue_id=resolved_queue,
                media=media,
                option=option_value,
                radio_mode=bool(radio_mode),
                label=f"ma_play_media:{resolved_queue}:{option}",
                origin_context=_voice_origin_snapshot(),
            )
            verification = (
                await _ma_post_action_verification(client, resolved_queue)
                if dispatch.command_acknowledged
                else _ma_queue_summary(client, resolved_queue)
            )
            return {
                "success": True,
                "message": (
                    "Added as next."
                    if option == "next"
                    else "Added to queue."
                    if option == "add"
                    else "Playing."
                    if dispatch.command_acknowledged
                    else "Starting playback."
                ),
                "command_acknowledged": dispatch.command_acknowledged,
                "command_dispatched": dispatch.command_dispatched,
                "still_processing": dispatch.still_processing,
                "duplicate_suppressed": dispatch.duplicate_suppressed,
                "queue_id": resolved_queue,
                "media": media,
                "option": option,
                "radio_mode": bool(radio_mode),
                "queue_after": verification,
            }

        return _json_tool_result(await manager.run_serialized("ma_play_media", operation))
    except Exception as exc:
        log.exception("MA NATIVE ma_play_media failed queue_id=%r", queue_id)
        return _json_tool_result({"success": False, "error": str(exc)})


@function_tool
async def ma_playback(
    queue_id: str,
    command: Literal["play", "pause", "stop", "toggle", "next", "previous"],
    seek_seconds: int | None = None,
) -> str:
    """Control Music Assistant queue playback directly over WebSocket."""
    queue_id = str(queue_id or "").strip()
    try:
        manager = _ma_native_manager()

        async def operation(client: Any):
            resolved_queue = await _ma_resolve_queue_id(client, queue_id)
            queues = client.player_queues
            log.info(
                "MA NATIVE PLAYBACK requested_queue=%r resolved_queue=%r command=%s seek=%r",
                queue_id,
                resolved_queue,
                command,
                seek_seconds,
            )
            if command == "play":
                await queues.play(resolved_queue)
                if seek_seconds is not None:
                    await queues.seek(resolved_queue, max(0, int(seek_seconds)))
                message = "Playing."
            elif command == "pause":
                await queues.pause(resolved_queue)
                message = "Paused."
            elif command == "stop":
                await queues.stop(resolved_queue)
                message = "Stopped."
            elif command == "toggle":
                await queues.play_pause(resolved_queue)
                message = "Playback toggled."
            elif command == "next":
                await queues.next(resolved_queue)
                message = "Skipped to next."
            elif command == "previous":
                await queues.previous(resolved_queue)
                message = "Previous."
            else:
                raise ValueError(f"Unsupported playback command: {command}")
            return {
                "success": True,
                "message": message,
                "command_acknowledged": True,
                "queue_id": resolved_queue,
                "command": command,
            }

        return _json_tool_result(await manager.run_serialized("ma_playback", operation))
    except Exception as exc:
        log.exception("MA NATIVE ma_playback failed queue_id=%r command=%r", queue_id, command)
        return _json_tool_result({"success": False, "error": str(exc)})


@function_tool
async def ma_volume(
    player_id: str,
    level: int | None = None,
    adjust: Literal["up", "down"] | None = None,
    mute: bool | None = None,
    group: bool = False,
) -> str:
    """Set, adjust, mute or unmute a Music Assistant player."""
    supplied = sum(value is not None for value in (level, adjust, mute))
    if supplied != 1:
        return _json_tool_result(
            {"success": False, "error": "Provide exactly one of level, adjust or mute"}
        )
    if mute is not None and group:
        return _json_tool_result(
            {"success": False, "error": "group=true is not supported for mute"}
        )
    try:
        manager = _ma_native_manager()

        async def operation(client: Any):
            players = client.players
            if client.players.get(player_id) is None:
                raise ValueError(f"Music Assistant player not found: {player_id}")
            if level is not None:
                value = max(0, min(100, int(level)))
                if group:
                    await players.group_volume(player_id, value)
                else:
                    await players.volume_set(player_id, value)
                message = f"Volume set to {value}%."
            elif adjust == "up":
                if group:
                    await players.group_volume_up(player_id)
                else:
                    await players.volume_up(player_id)
                message = "Volume increased."
            elif adjust == "down":
                if group:
                    await players.group_volume_down(player_id)
                else:
                    await players.volume_down(player_id)
                message = "Volume decreased."
            else:
                await players.volume_mute(player_id, bool(mute))
                message = "Muted." if mute else "Unmuted."
            log.info(
                "MA NATIVE VOLUME player_id=%r level=%r adjust=%r mute=%r group=%s",
                player_id,
                level,
                adjust,
                mute,
                group,
            )
            return {
                "success": True,
                "message": message,
                "command_acknowledged": True,
                "player_id": player_id,
            }

        return _json_tool_result(await manager.run_serialized("ma_volume", operation))
    except Exception as exc:
        log.exception("MA NATIVE ma_volume failed player_id=%r", player_id)
        return _json_tool_result({"success": False, "error": str(exc)})


@function_tool
async def ma_group(
    action: Literal["join", "leave"],
    player_ids: list[str],
    target_player_id: str | None = None,
) -> str:
    """Join Music Assistant players to a leader or remove them from groups."""
    ids = [str(value).strip() for value in player_ids if str(value).strip()]
    if not ids:
        return _json_tool_result({"success": False, "error": "player_ids is required"})
    if action == "join" and not target_player_id:
        return _json_tool_result(
            {"success": False, "error": "target_player_id is required for join"}
        )
    try:
        manager = _ma_native_manager()

        async def operation(client: Any):
            if action == "join":
                target = str(target_player_id)
                children = [item for item in ids if item != target]
                if not children:
                    raise ValueError("No child players supplied to join")
                log.info("MA NATIVE GROUP join target=%r children=%r", target, children)
                await client.players.group_many(target, children)
                message = "Speakers grouped."
            else:
                log.info("MA NATIVE GROUP leave players=%r", ids)
                await client.players.ungroup_many(ids)
                message = "Speakers removed from group."
            return {
                "success": True,
                "message": message,
                "command_acknowledged": True,
                "action": action,
                "player_ids": ids,
                "target_player_id": target_player_id,
            }

        return _json_tool_result(await manager.run_serialized("ma_group", operation))
    except Exception as exc:
        log.exception("MA NATIVE ma_group failed action=%r", action)
        return _json_tool_result({"success": False, "error": str(exc)})


@function_tool
async def ma_queue(
    queue_id: str,
    get_items: bool = True,
    shuffle: bool | None = None,
    repeat: Literal["off", "one", "all"] | None = None,
    clear: bool = False,
) -> str:
    """Read queue state and optionally change shuffle/repeat/clear settings."""
    queue_id = str(queue_id or "").strip()
    try:
        manager = _ma_native_manager()

        async def operation(client: Any):
            resolved_queue = await _ma_resolve_queue_id(client, queue_id)
            changes: list[str] = []
            if shuffle is not None:
                await client.player_queues.shuffle(resolved_queue, bool(shuffle))
                changes.append(f"shuffle {'enabled' if shuffle else 'disabled'}")
            if repeat is not None:
                assert RepeatMode is not None
                await client.player_queues.repeat(resolved_queue, RepeatMode(str(repeat)))
                changes.append(f"repeat set to {repeat}")
            if clear:
                await client.player_queues.clear(resolved_queue)
                changes.append("queue cleared")
            payload: dict[str, Any] = {
                "success": True,
                "changed": bool(changes),
                "changes_applied": changes,
                "queue": _ma_queue_summary(client, resolved_queue),
            }
            if get_items:
                items = await client.player_queues.get_queue_items(
                    resolved_queue, limit=50, offset=0
                )
                payload["items"] = [_ma_queue_item_summary(item) for item in items]
            return payload

        return _json_tool_result(await manager.run_serialized("ma_queue", operation))
    except Exception as exc:
        log.exception("MA NATIVE ma_queue failed queue_id=%r", queue_id)
        return _json_tool_result({"success": False, "error": str(exc)})


@function_tool
async def ma_queue_item(
    queue_id: str,
    item_id: str,
    action: Literal["move_up", "move_down", "move_next", "remove"],
) -> str:
    """Move or remove an individual Music Assistant queue item."""
    try:
        manager = _ma_native_manager()

        async def operation(client: Any):
            resolved_queue = await _ma_resolve_queue_id(client, queue_id)
            queues = client.player_queues
            log.info(
                "MA NATIVE QUEUE_ITEM queue_id=%r item_id=%r action=%s",
                resolved_queue,
                item_id,
                action,
            )
            if action == "move_up":
                await queues.move_up(resolved_queue, item_id)
            elif action == "move_down":
                await queues.move_down(resolved_queue, item_id)
            elif action == "move_next":
                await queues.move_next(resolved_queue, item_id)
            elif action == "remove":
                await queues.delete_item(resolved_queue, item_id)
            return {
                "success": True,
                "message": "Queue updated.",
                "command_acknowledged": True,
                "queue_id": resolved_queue,
                "item_id": item_id,
                "action": action,
            }

        return _json_tool_result(await manager.run_serialized("ma_queue_item", operation))
    except Exception as exc:
        log.exception("MA NATIVE ma_queue_item failed queue_id=%r", queue_id)
        return _json_tool_result({"success": False, "error": str(exc)})


@function_tool
async def ma_transfer_queue(source_queue_id: str, target_queue_id: str) -> str:
    """Transfer Music Assistant playback from one queue/player to another."""
    try:
        manager = _ma_native_manager()

        async def operation(client: Any):
            source = await _ma_resolve_queue_id(client, source_queue_id)
            target = await _ma_resolve_queue_id(client, target_queue_id)
            log.info("MA NATIVE TRANSFER source=%r target=%r", source, target)
            await client.player_queues.transfer(source, target)
            return {
                "success": True,
                "message": "Playback moved.",
                "command_acknowledged": True,
                "source_queue_id": source,
                "target_queue_id": target,
            }

        return _json_tool_result(await manager.run_serialized("ma_transfer_queue", operation))
    except Exception as exc:
        log.exception(
            "MA NATIVE ma_transfer_queue failed source=%r target=%r",
            source_queue_id,
            target_queue_id,
        )
        return _json_tool_result({"success": False, "error": str(exc)})
