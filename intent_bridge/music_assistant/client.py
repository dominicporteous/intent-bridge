"""Native Music Assistant connection and value models."""

import asyncio
import time
from dataclasses import dataclass
from typing import Any

try:
    from music_assistant_client import MusicAssistantClient

    MUSIC_ASSISTANT_CLIENT_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - deployment dependency guard
    MusicAssistantClient = None
    MUSIC_ASSISTANT_CLIENT_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

from intent_bridge.config import log, settings
from intent_bridge.runtime.dependencies import runtime


def _ma_enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _ma_name(value: Any) -> str:
    return str(getattr(value, "name", "") or "").strip()


def _ma_uri(value: Any) -> str:
    return str(getattr(value, "uri", "") or "").strip()


def _ma_media_type(value: Any, fallback: str | None = None) -> str | None:
    media_type = getattr(value, "media_type", None)
    if media_type is None:
        return fallback
    return str(_ma_enum_value(media_type))


def _ma_media_summary(item: Any, fallback_type: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": _ma_name(item),
        "uri": _ma_uri(item),
        "media_type": _ma_media_type(item, fallback_type),
    }
    for attr in ("provider", "item_id", "version"):
        value = getattr(item, attr, None)
        if value not in (None, ""):
            payload[attr] = _ma_enum_value(value)

    artists = getattr(item, "artists", None)
    if artists:
        payload["artists"] = [_ma_name(artist) for artist in artists if _ma_name(artist)]
    album = getattr(item, "album", None)
    if album is not None and _ma_name(album):
        payload["album"] = _ma_name(album)
    return payload


def _ma_queue_item_summary(item: Any) -> dict[str, Any]:
    media_item = getattr(item, "media_item", None)
    payload = {
        "queue_item_id": str(
            getattr(item, "queue_item_id", None) or getattr(item, "item_id", None) or ""
        ),
        "name": str(getattr(item, "name", "") or ""),
    }
    if media_item is not None:
        payload["media"] = _ma_media_summary(media_item)
    else:
        uri = str(getattr(item, "uri", "") or "")
        if uri:
            payload["uri"] = uri
    return payload


def _ma_player_summary(client: Any, player: Any) -> dict[str, Any]:
    player_id = str(getattr(player, "player_id", "") or "")
    active_source = str(getattr(player, "active_source", "") or "")
    local_queue = client.player_queues.get(active_source or player_id)
    queue_id = str(getattr(local_queue, "queue_id", "") or "") if local_queue else ""
    payload: dict[str, Any] = {
        "player_id": player_id,
        "name": str(getattr(player, "name", "") or player_id),
        "available": bool(getattr(player, "available", True)),
        "powered": getattr(player, "powered", None),
        "state": _ma_enum_value(getattr(player, "state", None)),
        "volume_level": getattr(player, "volume_level", None),
        "volume_muted": getattr(player, "volume_muted", None),
        "active_source": active_source or None,
        "queue_id": queue_id or active_source or player_id,
        "synced_to": getattr(player, "synced_to", None),
    }
    group_childs = getattr(player, "group_childs", None)
    if group_childs:
        payload["group_childs"] = list(group_childs)
    return payload


def _ma_queue_summary(client: Any, queue_id: str) -> dict[str, Any]:
    queue = client.player_queues.get(queue_id)
    if queue is None:
        return {"queue_id": queue_id, "cached": False}
    current_item = getattr(queue, "current_item", None)
    return {
        "queue_id": str(getattr(queue, "queue_id", queue_id)),
        "display_name": str(
            getattr(queue, "display_name", None) or getattr(queue, "name", None) or ""
        ),
        "state": _ma_enum_value(getattr(queue, "state", None)),
        "current_index": getattr(queue, "current_index", None),
        "elapsed_time": getattr(queue, "elapsed_time", None),
        "shuffle_enabled": getattr(queue, "shuffle_enabled", None),
        "repeat_mode": _ma_enum_value(getattr(queue, "repeat_mode", None)),
        "current_item": (
            _ma_queue_item_summary(current_item) if current_item is not None else None
        ),
        "cached": True,
    }


@dataclass
class InFlightMusicPlayback:
    queue_id: str
    fingerprint: str
    task: asyncio.Task[Any]
    label: str
    started_at: float


@dataclass(frozen=True)
class MusicPlayDispatchResult:
    command_acknowledged: bool
    command_dispatched: bool
    still_processing: bool
    duplicate_suppressed: bool


class NativeMusicAssistant:
    """Persistent official Music Assistant WebSocket client with reconnects.

    The official client owns the receive loop and maintains players/queues from
    server events. State-changing calls are deliberately NOT auto-retried: once a
    command has been sent, an ambiguous disconnect must not cause duplicate media
    actions. The next user request can execute after the supervisor reconnects.
    """

    def __init__(self, server_url: str, token: str) -> None:
        self.server_url = server_url
        self.token = token
        self.client: Any | None = None
        self.ready = asyncio.Event()
        self._stopping = asyncio.Event()
        self._supervisor_task: asyncio.Task | None = None
        self._tool_lock = asyncio.Lock()
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._inflight_playbacks: dict[tuple[str, str], InFlightMusicPlayback] = {}
        # Voice UX state only: last chosen artist-radio seed avoids immediate
        # repetition, while queue generations stop superseded background radio
        # lifecycles from enabling DSTM for a newer playback request.
        self._last_radio_seed_by_artist: dict[str, str] = {}
        self._queue_playback_generation: dict[str, int] = {}
        self.last_error: str | None = None
        self.last_connected_at: float | None = None
        self.connection_count = 0
        self.reconnect_count = 0

    @property
    def connected(self) -> bool:
        client = self.client
        return bool(
            client is not None
            and self.ready.is_set()
            and getattr(getattr(client, "connection", None), "connected", False)
        )

    async def start(self) -> bool:
        if MusicAssistantClient is None:
            self.last_error = (
                MUSIC_ASSISTANT_CLIENT_IMPORT_ERROR or "music-assistant-client unavailable"
            )
            log.error("Music Assistant native client unavailable: %s", self.last_error)
            return False
        if self._supervisor_task is None or self._supervisor_task.done():
            self._stopping.clear()
            self._supervisor_task = asyncio.create_task(
                self._supervisor(), name="music-assistant-native-supervisor"
            )
        try:
            await asyncio.wait_for(
                self.ready.wait(), timeout=settings.music_assistant.connect_timeout_seconds
            )
            return True
        except TimeoutError:
            log.warning(
                "Music Assistant native initial connection not ready after %.1fs; "
                "supervisor will keep retrying",
                settings.music_assistant.connect_timeout_seconds,
            )
            return False

    async def stop(self) -> None:
        self._stopping.set()
        task = self._supervisor_task
        self._supervisor_task = None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        background = list(self._background_tasks)
        self._background_tasks.clear()
        self._inflight_playbacks.clear()
        self._last_radio_seed_by_artist.clear()
        self._queue_playback_generation.clear()
        for pending in background:
            pending.cancel()
        if background:
            await asyncio.gather(*background, return_exceptions=True)
        self.ready.clear()
        self.client = None

    @property
    def inflight_playback_count(self) -> int:
        return sum(1 for item in self._inflight_playbacks.values() if not item.task.done())

    def get_inflight_playback(
        self, queue_id: str, fingerprint: str
    ) -> InFlightMusicPlayback | None:
        key = (queue_id, fingerprint)
        record = self._inflight_playbacks.get(key)
        if record is not None and record.task.done():
            self._inflight_playbacks.pop(key, None)
            return None
        return record

    def register_inflight_playback(
        self,
        queue_id: str,
        fingerprint: str,
        task: asyncio.Task[Any],
        label: str,
    ) -> InFlightMusicPlayback:
        record = InFlightMusicPlayback(
            queue_id=queue_id,
            fingerprint=fingerprint,
            task=task,
            label=label,
            started_at=time.time(),
        )
        self._inflight_playbacks[(queue_id, fingerprint)] = record
        return record

    def clear_inflight_playback(
        self, queue_id: str, fingerprint: str, task: asyncio.Task[Any]
    ) -> None:
        key = (queue_id, fingerprint)
        current = self._inflight_playbacks.get(key)
        if current is not None and current.task is task:
            self._inflight_playbacks.pop(key, None)

    def begin_queue_playback_generation(self, queue_id: str) -> int:
        generation = self._queue_playback_generation.get(queue_id, 0) + 1
        self._queue_playback_generation[queue_id] = generation
        return generation

    def is_current_queue_playback_generation(self, queue_id: str, generation: int) -> bool:
        return self._queue_playback_generation.get(queue_id, 0) == generation

    def last_radio_seed(self, artist_uri: str) -> str | None:
        return self._last_radio_seed_by_artist.get(artist_uri)

    def remember_radio_seed(self, artist_uri: str, seed_uri: str) -> None:
        if artist_uri and seed_uri:
            self._last_radio_seed_by_artist[artist_uri] = seed_uri

    def track_background_task(self, task: asyncio.Task[Any], *, label: str) -> None:
        """Keep a long-running MA command alive and log its eventual result."""
        self._background_tasks.add(task)

        def _done(completed: asyncio.Task[Any]) -> None:
            self._background_tasks.discard(completed)
            if completed.cancelled():
                log.info("MA NATIVE ASYNC COMMAND CANCELLED label=%s", label)
                return
            try:
                completed.result()
            except Exception as exc:
                log.error(
                    "MA NATIVE ASYNC COMMAND FAILED label=%s error=%s: %s",
                    label,
                    type(exc).__name__,
                    exc,
                )
            else:
                log.info("MA NATIVE ASYNC COMMAND COMPLETE label=%s", label)

        task.add_done_callback(_done)

    async def _supervisor(self) -> None:
        first_connection = True
        while not self._stopping.is_set():
            client: Any | None = None
            listener_task: asyncio.Task | None = None
            ready_wait_task: asyncio.Task | None = None
            try:
                assert MusicAssistantClient is not None
                client = MusicAssistantClient(
                    self.server_url,
                    None,
                    token=self.token,
                )
                self.client = client
                initial_state_ready = asyncio.Event()
                listener_task = asyncio.create_task(
                    client.start_listening(initial_state_ready),
                    name="music-assistant-native-listener",
                )
                ready_wait_task = asyncio.create_task(initial_state_ready.wait())

                done, _ = await asyncio.wait(
                    {listener_task, ready_wait_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if listener_task in done:
                    # Propagate connection/auth/schema errors immediately.
                    await listener_task
                    raise RuntimeError(
                        "Music Assistant listener exited before initial state was ready"
                    )

                await ready_wait_task
                self.ready.set()
                self.last_error = None
                self.last_connected_at = time.time()
                self.connection_count += 1
                if not first_connection:
                    self.reconnect_count += 1
                first_connection = False

                info = getattr(client, "server_info", None)
                log.info(
                    "Music Assistant native WebSocket ready url=%r version=%r schema=%r "
                    "players=%s queues=%s",
                    self.server_url,
                    getattr(info, "server_version", None),
                    getattr(info, "schema_version", None),
                    len(getattr(client.players, "players", []) or []),
                    len(getattr(client.player_queues, "player_queues", []) or []),
                )

                # Blocks until the websocket is closed or stop() cancels us.
                await listener_task
                if not self._stopping.is_set():
                    raise RuntimeError("Music Assistant WebSocket listener disconnected")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                log.warning("Music Assistant native connection lost: %s", self.last_error)
            finally:
                self.ready.clear()
                if ready_wait_task is not None and not ready_wait_task.done():
                    ready_wait_task.cancel()
                    await asyncio.gather(ready_wait_task, return_exceptions=True)
                if listener_task is not None and not listener_task.done():
                    listener_task.cancel()
                    await asyncio.gather(listener_task, return_exceptions=True)
                if client is not None:
                    try:
                        await client.disconnect()
                    except Exception:
                        pass
                if self.client is client:
                    self.client = None

            if not self._stopping.is_set():
                try:
                    await asyncio.wait_for(
                        self._stopping.wait(),
                        timeout=settings.music_assistant.reconnect_delay_seconds,
                    )
                except TimeoutError:
                    pass

    async def wait_ready(self) -> Any:
        if not self.connected:
            try:
                await asyncio.wait_for(
                    self.ready.wait(),
                    timeout=settings.music_assistant.command_timeout_seconds,
                )
            except TimeoutError as exc:
                raise RuntimeError("Music Assistant is not connected") from exc
        client = self.client
        if client is None or not self.connected:
            raise RuntimeError("Music Assistant is not connected")
        return client

    async def run_serialized(self, tool_name: str, operation) -> Any:
        # Wait outside the lock so one disconnected request does not block all
        # callers from observing a reconnect.
        await self.wait_ready()
        async with self._tool_lock:
            client = await self.wait_ready()
            log.info("MA NATIVE CALL tool=%s", tool_name)
            return await asyncio.wait_for(
                operation(client),
                timeout=settings.music_assistant.command_timeout_seconds,
            )


def _ma_native_manager() -> NativeMusicAssistant:
    manager = runtime.music_assistant
    if manager is None:
        raise RuntimeError("Music Assistant native client is unavailable")
    return manager
