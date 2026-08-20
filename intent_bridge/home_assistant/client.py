"""Persistent Home Assistant WebSocket transport."""

import asyncio
import difflib
import json
import time
from collections.abc import Callable
from typing import Any

import websockets

from intent_bridge.config import log, settings
from intent_bridge.core.text import normalize_search_text as _normalise_search_text
from intent_bridge.home_assistant import catalog as ha_catalog
from intent_bridge.indicators.topology import is_indicator_control
from intent_bridge.runtime.dependencies import runtime
from intent_bridge.runtime.execution import (
    _ha_websocket_url,
)


class HomeAssistantWebSocket:
    def __init__(self, base_url: str, token: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.ws_url = _ha_websocket_url(base_url)

        self.ready = asyncio.Event()
        self._stopping = asyncio.Event()
        self._ws = None
        self._supervisor_task: asyncio.Task | None = None
        self._reader_task: asyncio.Task | None = None
        self._send_lock = asyncio.Lock()
        self._request_id = 0
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}

        self.states: dict[str, dict[str, Any]] = {}
        self.config: dict[str, Any] = {}
        self.services: dict[str, Any] = {}
        self.entity_registry: dict[str, dict[str, Any]] = {}
        self.devices: dict[str, dict[str, Any]] = {}
        self.areas: dict[str, dict[str, Any]] = {}

        self._services_loaded_at = 0.0
        self._registries_loaded_at = 0.0
        self._service_refresh_lock = asyncio.Lock()
        self._registry_refresh_lock = asyncio.Lock()
        self._catalog_cache_listeners: set[Callable[[str], None]] = set()

        self.connected_at: float | None = None
        self.reconnect_count = 0
        self.last_error: str | None = None
        self.state_event_count = 0

    async def start(self) -> None:
        if self._supervisor_task is not None:
            return
        self._stopping.clear()
        self._supervisor_task = asyncio.create_task(self._supervisor(), name="ha-ws-supervisor")
        try:
            await asyncio.wait_for(
                self.ready.wait(), timeout=settings.home_assistant.websocket.connect_timeout_seconds
            )
        except TimeoutError as exc:
            raise RuntimeError(
                f"Timed out connecting to Home Assistant WebSocket {self.ws_url}"
            ) from exc

    async def stop(self) -> None:
        self._stopping.set()
        self.ready.clear()

        ws = self._ws
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                pass

        task = self._supervisor_task
        self._supervisor_task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                log.exception("HA WebSocket supervisor shutdown error")

        self._fail_pending(RuntimeError("Home Assistant WebSocket stopped"))

    async def _supervisor(self) -> None:
        while not self._stopping.is_set():
            try:
                await self._run_connection()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.last_error = str(exc)
                self.ready.clear()
                log.warning("HA WebSocket disconnected/error: %s", exc)

            if self._stopping.is_set():
                break

            await asyncio.sleep(settings.home_assistant.websocket.reconnect_delay_seconds)

    async def _run_connection(self) -> None:
        log.info("Connecting Home Assistant WebSocket %s", self.ws_url)

        async with websockets.connect(
            self.ws_url,
            open_timeout=settings.home_assistant.websocket.connect_timeout_seconds,
            ping_interval=20,
            ping_timeout=20,
            close_timeout=5,
            max_size=32 * 1024 * 1024,
        ) as ws:
            auth_required = json.loads(await ws.recv())
            if auth_required.get("type") != "auth_required":
                raise RuntimeError(f"Unexpected HA WebSocket auth message: {auth_required}")

            await ws.send(json.dumps({"type": "auth", "access_token": self.token}))
            auth_reply = json.loads(await ws.recv())
            if auth_reply.get("type") != "auth_ok":
                raise RuntimeError(f"Home Assistant WebSocket authentication failed: {auth_reply}")

            self._ws = ws
            if self.connected_at is not None:
                self.reconnect_count += 1
            self.connected_at = time.time()
            self.last_error = None
            self._reader_task = asyncio.create_task(self._reader_loop(ws), name="ha-ws-reader")

            try:
                await self._initialise_connection_caches()
                self._notify_catalog_cache_change("connection")
                self.ready.set()
                log.info(
                    "HA WebSocket ready states=%d service_domains=%d entities=%d",
                    len(self.states),
                    len(self.services),
                    len(self.entity_registry),
                )
                await self._reader_task
            finally:
                self.ready.clear()
                self._ws = None
                self._fail_pending(ConnectionError("Home Assistant WebSocket connection closed"))

    async def _reader_loop(self, ws) -> None:
        async for raw in ws:
            try:
                message = json.loads(raw)
            except Exception:
                log.debug("Ignoring non-JSON HA WebSocket frame")
                continue

            message_type = message.get("type")
            message_id = message.get("id")

            if message_type == "result" and isinstance(message_id, int):
                future = self._pending.pop(message_id, None)
                if future is not None and not future.done():
                    future.set_result(message)
                continue

            if message_type == "event":
                self._handle_event(message)
                continue

    def _handle_event(self, message: dict[str, Any]) -> None:
        event = message.get("event")
        if not isinstance(event, dict):
            return

        event_type = event.get("event_type")
        data = event.get("data")
        if not isinstance(data, dict):
            data = {}

        if event_type == "state_changed":
            entity_id = data.get("entity_id")
            if not isinstance(entity_id, str):
                return
            new_state = data.get("new_state")
            if isinstance(new_state, dict):
                self.states[entity_id] = new_state
            else:
                self.states.pop(entity_id, None)
            self.state_event_count += 1
            self._notify_catalog_cache_change("state")
            return

        if event_type in {"service_registered", "service_removed"}:
            self._services_loaded_at = 0.0
            return

        if event_type in {
            "entity_registry_updated",
            "device_registry_updated",
            "area_registry_updated",
        }:
            self._registries_loaded_at = 0.0
            self._notify_catalog_cache_change("registry")

    def add_catalog_cache_listener(self, listener: Callable[[str], None]) -> None:
        """Register a cheap synchronous callback for live catalog source changes."""

        self._catalog_cache_listeners.add(listener)

    def remove_catalog_cache_listener(self, listener: Callable[[str], None]) -> None:
        self._catalog_cache_listeners.discard(listener)

    def _notify_catalog_cache_change(self, change: str) -> None:
        """Notify observers without allowing one observer to break the reader loop."""

        for listener in tuple(self._catalog_cache_listeners):
            try:
                listener(change)
            except Exception:
                log.exception("HA catalog cache listener failed change=%s", change)

    def _fail_pending(self, exc: Exception) -> None:
        for request_id, future in list(self._pending.items()):
            if not future.done():
                future.set_exception(exc)
            self._pending.pop(request_id, None)

    async def _send_current(
        self,
        payload: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        ws = self._ws
        if ws is None:
            raise ConnectionError("Home Assistant WebSocket is not connected")

        loop = asyncio.get_running_loop()
        async with self._send_lock:
            self._request_id += 1
            request_id = self._request_id
            message = {"id": request_id, **payload}
            future: asyncio.Future[dict[str, Any]] = loop.create_future()
            self._pending[request_id] = future
            try:
                await ws.send(json.dumps(message))
            except Exception:
                self._pending.pop(request_id, None)
                raise

        try:
            return await asyncio.wait_for(
                future,
                timeout=timeout or settings.home_assistant.websocket.command_timeout_seconds,
            )
        except Exception:
            self._pending.pop(request_id, None)
            raise

    async def command(
        self,
        payload: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        # Wait for an established/reconnected socket. One retry handles the
        # narrow race where the connection drops between ready.wait and send.
        for attempt in range(2):
            await asyncio.wait_for(
                self.ready.wait(),
                timeout=timeout or settings.home_assistant.websocket.command_timeout_seconds,
            )
            try:
                return await self._send_current(payload, timeout=timeout)
            except (ConnectionError, websockets.ConnectionClosed):
                self.ready.clear()
                if attempt == 1:
                    raise
                await asyncio.sleep(0)
        raise RuntimeError("Home Assistant WebSocket command failed")

    async def process_conversation(
        self,
        text: str,
        *,
        language: str | None = None,
        device_id: str | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Process one voice utterance over the established HA WebSocket.

        The WebSocket API does not expose REST's named ``intent/handle``
        endpoint. ``conversation/process`` is its supported command for
        turning a spoken utterance into an intent response.
        """

        payload: dict[str, Any] = {
            "type": "conversation/process",
            "text": text,
        }
        if language:
            payload["language"] = language
        if device_id:
            payload["device_id"] = device_id
        return await self.command(payload, timeout=timeout)

    @staticmethod
    def _require_success(message: dict[str, Any], operation: str) -> Any:
        if message.get("success") is True:
            return message.get("result")
        error = message.get("error")
        raise RuntimeError(f"HA WebSocket {operation} failed: {error}")

    async def _initialise_connection_caches(self) -> None:
        try:
            config_reply = await self._send_current({"type": "get_config"})
            config = self._require_success(config_reply, "get_config")
            if isinstance(config, dict):
                self.config = config
        except Exception as exc:
            log.debug("HA instance configuration unavailable: %s", exc)

        states_reply = await self._send_current({"type": "get_states"})
        states = self._require_success(states_reply, "get_states")
        if not isinstance(states, list):
            raise RuntimeError("HA get_states returned an unexpected payload")
        self.states = {
            item["entity_id"]: item
            for item in states
            if isinstance(item, dict) and isinstance(item.get("entity_id"), str)
        }

        services_reply = await self._send_current({"type": "get_services"})
        services = self._require_success(services_reply, "get_services")
        self.services = services if isinstance(services, dict) else {}
        self._services_loaded_at = time.monotonic()

        await self._refresh_registries_current()

        # Subscriptions are cheap and keep normal state reads network-free.
        for event_type in (
            "state_changed",
            "service_registered",
            "service_removed",
            "entity_registry_updated",
            "device_registry_updated",
            "area_registry_updated",
        ):
            reply = await self._send_current({"type": "subscribe_events", "event_type": event_type})
            if reply.get("success") is not True:
                log.debug(
                    "HA event subscription unavailable event_type=%s error=%s",
                    event_type,
                    reply.get("error"),
                )

    async def refresh_services(self, *, force: bool = False) -> None:
        if not force and (
            time.monotonic() - self._services_loaded_at
            < settings.home_assistant.websocket.service_cache_ttl_seconds
        ):
            return
        async with self._service_refresh_lock:
            if not force and (
                time.monotonic() - self._services_loaded_at
                < settings.home_assistant.websocket.service_cache_ttl_seconds
            ):
                return
            reply = await self.command({"type": "get_services"})
            services = self._require_success(reply, "get_services")
            if isinstance(services, dict):
                self.services = services
                self._services_loaded_at = time.monotonic()

    async def refresh_registries(self, *, force: bool = False) -> None:
        if not force and (
            time.monotonic() - self._registries_loaded_at
            < settings.home_assistant.websocket.registry_cache_ttl_seconds
        ):
            return
        async with self._registry_refresh_lock:
            if not force and (
                time.monotonic() - self._registries_loaded_at
                < settings.home_assistant.websocket.registry_cache_ttl_seconds
            ):
                return
            await self._refresh_registries_via_command()

    async def _refresh_registries_current(self) -> None:
        async def send(payload: dict[str, Any]) -> dict[str, Any]:
            return await self._send_current(payload)

        await self._refresh_registries(send)

    async def _refresh_registries_via_command(self) -> None:
        async def send(payload: dict[str, Any]) -> dict[str, Any]:
            return await self.command(payload)

        await self._refresh_registries(send)

    async def _refresh_registries(self, sender) -> None:
        # entity_registry/list_for_display is documented and compact. The area
        # and device list commands are long-standing frontend WebSocket APIs; if
        # either is unavailable we simply search without that decoration.
        try:
            reply = await sender({"type": "config/entity_registry/list_for_display"})
            result = self._require_success(reply, "config/entity_registry/list_for_display")
            entries = result.get("entities", []) if isinstance(result, dict) else []
            self.entity_registry = {
                item["ei"]: item
                for item in entries
                if isinstance(item, dict) and isinstance(item.get("ei"), str)
            }
        except Exception as exc:
            log.debug("Entity display registry cache unavailable: %s", exc)

        try:
            reply = await sender({"type": "config/device_registry/list"})
            result = self._require_success(reply, "config/device_registry/list")
            if isinstance(result, list):
                self.devices = {
                    item["id"]: item
                    for item in result
                    if isinstance(item, dict) and isinstance(item.get("id"), str)
                }
        except Exception as exc:
            log.debug("Device registry cache unavailable: %s", exc)

        try:
            reply = await sender({"type": "config/area_registry/list"})
            result = self._require_success(reply, "config/area_registry/list")
            if isinstance(result, list):
                self.areas = {
                    item["area_id"]: item
                    for item in result
                    if isinstance(item, dict) and isinstance(item.get("area_id"), str)
                }
        except Exception as exc:
            log.debug("Area registry cache unavailable: %s", exc)

        self._registries_loaded_at = time.monotonic()

    def _entity_context(self, entity_id: str, state: dict[str, Any]) -> dict[str, Any]:
        return ha_catalog.entity_context(self, entity_id, state)

    def search_cached_states(
        self,
        query: str,
        *,
        domain_filter: str | None,
        area_filter: str | None,
        preferred_area_filter: str | None = None,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Search cached states.

        area_filter is a hard user-requested area restriction.
        preferred_area_filter is a soft voice-origin ranking preference.
        """
        query_norm = _normalise_search_text(query)
        query_tokens = set(query_norm.split())
        hard_area_norm = _normalise_search_text(area_filter) if area_filter else ""
        preferred_area_norm = (
            _normalise_search_text(preferred_area_filter) if preferred_area_filter else ""
        )
        domain_norm = (domain_filter or "").strip().casefold()
        domain_words = domain_norm.replace("_", " ")

        generic_domain_query = bool(
            domain_norm
            and query_norm
            and query_norm
            in {
                domain_words,
                domain_words.rstrip("s"),
                f"{domain_words} device",
                f"{domain_words} devices",
            }
        )
        # Semantic queries should enumerate the provider domain even when its
        # entity name is simply "Home" or "Forecast Home". Without this,
        # searching weather for "outside temperature" incorrectly returns no
        # candidates because those words need not occur in the entity label.
        if domain_norm == "weather" and query_tokens & {
            "weather",
            "forecast",
            "temperature",
            "temp",
            "outside",
            "outdoor",
            "outdoors",
        }:
            generic_domain_query = True

        scored: list[tuple[float, dict[str, Any]]] = []

        for entity_id, state in list(self.states.items()):
            domain = entity_id.split(".", 1)[0].casefold()
            if domain_norm and domain != domain_norm:
                continue

            ctx = self._entity_context(entity_id, state)
            local_entity_name = entity_id.split(".", 1)[-1]

            # IMPORTANT: do not normally include the full entity_id here.
            # Otherwise query="light" matches every "light.*" entity merely
            # because of its domain prefix.
            identity_parts = [
                _normalise_search_text(local_entity_name),
                _normalise_search_text(ctx.get("friendly_name")),
                _normalise_search_text(ctx.get("registry_name")),
            ]
            identity_parts = [p for p in identity_parts if p]

            context_parts = [
                _normalise_search_text(ctx.get("device_name")),
                _normalise_search_text(ctx.get("area_name")),
            ]
            context_parts = [p for p in context_parts if p]

            searchable_parts = identity_parts + context_parts
            if "." in query_norm:
                searchable_parts.append(_normalise_search_text(entity_id))

            searchable = " ".join(searchable_parts)
            entity_area = _normalise_search_text(ctx.get("area_name"))
            entity_area_id = _normalise_search_text(ctx.get("area_id"))

            # Explicit/user requested area remains a hard restriction.
            if hard_area_norm:
                if hard_area_norm not in {entity_area, entity_area_id} and (
                    hard_area_norm not in entity_area
                ):
                    continue

            score = 0.0
            reasons: list[str] = []

            if query_norm and query_norm in set(searchable_parts):
                score = 100.0
                reasons.append("exact_name")
            elif query_norm and query_norm in searchable:
                score = 75.0
                reasons.append("name_contains_query")
            elif query_tokens:
                combined_tokens = set(searchable.split())
                matched = len(query_tokens & combined_tokens)
                if matched:
                    score = 45.0 + (matched / len(query_tokens)) * 30.0
                    reasons.append("token_match")

            # Generic domain requests still consider all entities of that domain,
            # but only with a modest baseline score.
            if generic_domain_query:
                score = max(score, 35.0)
                reasons.append("generic_domain_candidate")

            if score < 45.0 and query_norm and searchable_parts:
                best_ratio = max(
                    (
                        difflib.SequenceMatcher(None, query_norm, candidate).ratio()
                        for candidate in searchable_parts
                        if candidate
                    ),
                    default=0.0,
                )
                if best_ratio >= 0.58:
                    score = max(score, best_ratio * 60.0)
                    reasons.append("fuzzy_match")

            if not query_norm:
                score = 1.0

            # Soft room preference. Actual HA area membership helps, but an
            # entity explicitly named after the room helps more. Thus
            # office_light can win even if it has no HA area assignment.
            if preferred_area_norm:
                actual_area_match = preferred_area_norm in {entity_area, entity_area_id} or (
                    entity_area and preferred_area_norm in entity_area
                )
                room_name_in_identity = any(preferred_area_norm in part for part in identity_parts)

                if actual_area_match:
                    score += 30.0
                    reasons.append("voice_origin_area")
                if room_name_in_identity:
                    score += 45.0
                    reasons.append("room_name_in_entity")

            attributes = state.get("attributes", {})
            if not isinstance(attributes, dict):
                attributes = {}

            device_class = _normalise_search_text(attributes.get("device_class"))

            # "Turn the light on" means room illumination, not a ring/status LED.
            # Only apply this to a generic light query; explicit "ring LED" or
            # "indicator light" requests remain available.
            if (
                settings.home_assistant.penalize_indicator_lights
                and domain == "light"
                and generic_domain_query
            ):
                explicit_indicator_words = {
                    "led",
                    "ring",
                    "indicator",
                    "status",
                    "notification",
                    "backlight",
                }
                if not (query_tokens & explicit_indicator_words):
                    if is_indicator_control(self, entity_id):
                        score -= 90.0
                        reasons.append("indicator_light_penalty")

            # Helpful for generic local media playback too.
            if domain == "media_player" and preferred_area_norm and generic_domain_query:
                if device_class == "speaker":
                    score += 20.0
                    reasons.append("speaker_preference")
                elif device_class == "tv":
                    score -= 10.0
                    reasons.append("tv_penalty")

            if score <= 0:
                continue

            scored.append(
                (
                    score,
                    {
                        "entity_id": entity_id,
                        "name": ctx.get("friendly_name") or ctx.get("registry_name") or entity_id,
                        "domain": domain,
                        "state": state.get("state"),
                        "area": ctx.get("area_name"),
                        "area_id": ctx.get("area_id"),
                        "device": ctx.get("device_name"),
                        "device_class": attributes.get("device_class"),
                        "unit": attributes.get("unit_of_measurement"),
                        "match_score": round(score, 2),
                        "match_reasons": reasons,
                    },
                )
            )

        scored.sort(key=lambda item: (-item[0], item[1]["entity_id"]))
        return [record for _, record in scored[:limit]]

    def resolve_area_reference(
        self,
        *,
        area_id: str | None = None,
        area_name: str | None = None,
    ) -> tuple[str | None, str | None]:
        return ha_catalog.resolve_area_reference(self, area_id=area_id, area_name=area_name)

    def resolve_device_origin(
        self,
        *,
        device_id: str | None = None,
        device_name: str | None = None,
        area_id: str | None = None,
        area_name: str | None = None,
    ) -> dict[str, Any]:
        return ha_catalog.resolve_device_origin(
            self,
            device_id=device_id,
            device_name=device_name,
            area_id=area_id,
            area_name=area_name,
        )

    def area_mentioned_in_text(self, text: str) -> tuple[str, str] | None:
        return ha_catalog.area_mentioned_in_text(self, text)

    def entities_in_area(self, domain: str, area_id: str) -> list[str]:
        return ha_catalog.entities_in_area(self, domain, area_id)

    async def wait_for_expected_state(
        self,
        entity_id: str,
        expected_state: str,
        timeout: float,
    ) -> str | None:
        if timeout <= 0:
            return None
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = self.states.get(entity_id)
            if isinstance(state, dict) and state.get("state") == expected_state:
                return expected_state
            await asyncio.sleep(0.04)
        return None


async def _require_ha_ws() -> HomeAssistantWebSocket:
    if runtime.ha_ws is None:
        raise RuntimeError("Home Assistant WebSocket fast path is unavailable")
    await asyncio.wait_for(
        runtime.ha_ws.ready.wait(),
        timeout=settings.home_assistant.websocket.command_timeout_seconds,
    )
    return runtime.ha_ws
