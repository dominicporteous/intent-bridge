"""Indicator discovery and asynchronous session control."""

import asyncio
from dataclasses import dataclass
from typing import Any

from intent_bridge.config import log, settings
from intent_bridge.core.text import normalize_search_text as _normalise_search_text
from intent_bridge.home_assistant.client import (
    _require_ha_ws,
)
from intent_bridge.indicators.policy import (
    effect_wants_software_pulse,
    find_native_effect,
    find_neutral_effect,
    light_supports_colour,
    parse_indicator_rgb,
    snapshot_restore_light_data,
)
from intent_bridge.indicators.topology import (
    connected_satellite_device_scope as _connected_satellite_device_scope,
)
from intent_bridge.indicators.topology import (
    indicator_relation_bonus as _indicator_relation_bonus,
)
from intent_bridge.indicators.topology import (
    indicator_score as _indicator_score,
)
from intent_bridge.runtime.execution import voice_tool_run_state


@dataclass
class AssistantLedTarget:
    entity_id: str
    domain: str
    satellite_entity_id: str
    device_id: str
    device_name: str | None
    area_id: str | None
    area_name: str | None
    match_reason: str


@dataclass
class AssistantLedSnapshot:
    entity_id: str
    domain: str
    state: str
    attributes: dict[str, Any]


@dataclass
class AssistantLedSession:
    target: AssistantLedTarget
    snapshot: AssistantLedSnapshot
    refcount: int = 1
    pulse_task: asyncio.Task[Any] | None = None
    native_effect: str | None = None


@dataclass(frozen=True)
class AssistantLedHandle:
    entity_id: str


def _voice_origin_snapshot() -> dict[str, Any]:
    """Capture request-scoped voice origin before background work outlives it."""
    return {
        "device_id": voice_tool_run_state.origin_device_id,
        "device_name": voice_tool_run_state.origin_device_name,
        "area_id": voice_tool_run_state.origin_area_id,
        "area_name": voice_tool_run_state.origin_area_name,
        "source": voice_tool_run_state.origin_source,
    }


async def _ha_internal_service_call(
    domain: str,
    service: str,
    entity_id: str,
    data: dict[str, Any] | None = None,
) -> None:
    """Call a tightly-scoped HA service without mutating LLM tool scratch state."""
    client = await _require_ha_ws()
    payload: dict[str, Any] = {
        "type": "call_service",
        "domain": domain,
        "service": service,
        "service_data": dict(data or {}),
        "target": {"entity_id": entity_id},
        "return_response": False,
    }
    reply = await client.command(
        payload, timeout=settings.home_assistant.websocket.command_timeout_seconds
    )
    if reply.get("success") is not True:
        error = reply.get("error")
        raise RuntimeError(f"HA {domain}.{service} failed for {entity_id}: {error}")


def _configured_indicator_rgb() -> list[int] | None:
    return parse_indicator_rgb(settings.assistant.led_color)


def _find_configured_native_effect(attributes: dict[str, Any]) -> str | None:
    return find_native_effect(attributes, settings.assistant.led_effect)


def _configured_effect_wants_software_pulse() -> bool:
    return effect_wants_software_pulse(settings.assistant.led_effect)


def _find_neutral_native_effect(attributes: dict[str, Any]) -> str | None:
    return find_neutral_effect(attributes)


def _light_supports_colour(attributes: dict[str, Any]) -> bool:
    return light_supports_colour(attributes)


def _snapshot_restore_light_data(snapshot: AssistantLedSnapshot) -> dict[str, Any]:
    return snapshot_restore_light_data(snapshot.attributes)


async def _resolve_satellite_indicator(
    origin_context: dict[str, Any] | None,
) -> AssistantLedTarget | None:
    """Resolve an indicator only through a real assist_satellite device relation."""
    if not settings.assistant.led_enabled or not origin_context:
        return None
    client = await _require_ha_ws()
    await client.refresh_registries()

    origin_device_id = str(origin_context.get("device_id") or "").strip() or None
    origin_device_name = str(origin_context.get("device_name") or "").strip() or None
    origin_area_id = str(origin_context.get("area_id") or "").strip() or None
    origin_area_name = str(origin_context.get("area_name") or "").strip() or None

    # Resolve any supplied device name/id first, but only trust it as a satellite
    # anchor if that HA device actually owns an assist_satellite.* entity.
    resolved_origin = client.resolve_device_origin(
        device_id=origin_device_id,
        device_name=origin_device_name,
        area_id=origin_area_id,
        area_name=origin_area_name,
    )
    candidate_device_id = resolved_origin.get("device_id")
    satellite_entity_id: str | None = None
    match_reason: str | None = None

    if isinstance(candidate_device_id, str) and candidate_device_id:
        device_satellites = [
            entity_id
            for entity_id in client.states
            if entity_id.startswith("assist_satellite.")
            and client.entity_registry.get(entity_id, {}).get("di") == candidate_device_id
        ]
        if len(device_satellites) == 1:
            satellite_entity_id = device_satellites[0]
            match_reason = "origin_device_assist_satellite"

    # HA's OpenAI conversation caller often gives us only an area through its
    # trusted system prompt. In that case require exactly one assist_satellite in
    # the area before traversing to its owning device.
    if satellite_entity_id is None:
        resolved_area_id, resolved_area_name = client.resolve_area_reference(
            area_id=origin_area_id or resolved_origin.get("area_id"),
            area_name=origin_area_name or resolved_origin.get("area_name"),
        )
        area_satellites: list[str] = []
        for entity_id, state in client.states.items():
            if not entity_id.startswith("assist_satellite."):
                continue
            ctx = client._entity_context(entity_id, state)
            if resolved_area_id and ctx.get("area_id") == resolved_area_id:
                area_satellites.append(entity_id)
            elif (
                not resolved_area_id
                and resolved_area_name
                and _normalise_search_text(ctx.get("area_name"))
                == _normalise_search_text(resolved_area_name)
            ):
                area_satellites.append(entity_id)
        if len(area_satellites) != 1:
            log.info(
                "ASSISTANT LED skipped: area satellite resolution ambiguous area=%r matches=%s",
                resolved_area_name or origin_area_name or origin_area_id,
                area_satellites,
            )
            return None
        satellite_entity_id = area_satellites[0]
        match_reason = "sole_assist_satellite_in_origin_area"
        candidate_device_id = client.entity_registry.get(satellite_entity_id, {}).get("di")

    if not isinstance(candidate_device_id, str) or not candidate_device_id:
        log.info(
            "ASSISTANT LED skipped: assist_satellite has no registry device satellite=%s",
            satellite_entity_id,
        )
        return None

    # Resolve indicator controls across the Assist satellite's bounded HA device
    # topology. A number of integrations model one physical voice satellite as
    # connected endpoint devices (assistant/microphone/playback/ring), so an LED
    # can legitimately live on a sibling device rather than the assist_satellite's
    # own device entry.
    device_scope = _connected_satellite_device_scope(client, candidate_device_id, max_depth=2)
    if len(device_scope) > 1:
        scope_log = []
        for scoped_device_id, (depth, relation) in sorted(
            device_scope.items(), key=lambda item: (item[1][0], item[0])
        ):
            scoped_device = client.devices.get(scoped_device_id, {})
            scoped_name = None
            if isinstance(scoped_device, dict):
                scoped_name = scoped_device.get("name_by_user") or scoped_device.get("name")
            scope_log.append((scoped_name or scoped_device_id, relation, depth))
        log.info(
            "ASSISTANT LED connected-device scope satellite=%s anchor_device=%s devices=%s",
            satellite_entity_id,
            candidate_device_id,
            scope_log,
        )

    # Tuple: final_score, entity_id, reasons, owning_device_id, relation, depth.
    sibling_controls: list[tuple[float, str, list[str], str, str, int]] = []
    for entity_id, _state in client.states.items():
        domain = entity_id.split(".", 1)[0].casefold()
        if domain not in settings.assistant.led_domains:
            continue
        registry = client.entity_registry.get(entity_id, {})
        owning_device_id = registry.get("di") if isinstance(registry, dict) else None
        if not isinstance(owning_device_id, str) or owning_device_id not in device_scope:
            continue
        depth, relation = device_scope[owning_device_id]
        score, reasons = _indicator_score(client, entity_id)
        if score > 0:
            relation_bonus = _indicator_relation_bonus(depth, relation)
            sibling_controls.append(
                (
                    score + relation_bonus,
                    entity_id,
                    [*reasons, relation],
                    owning_device_id,
                    relation,
                    depth,
                )
            )

    # Generic domain-only controls (for example remote_adb or bluetooth_proxy
    # switches) should not prevent traversal to a clearly named connected ring
    # endpoint. If no indicator-like name exists anywhere in the topology, retain
    # the old conservative sole-control fallback but only when the entire bounded
    # topology exposes exactly one light/switch control.
    if not sibling_controls:
        sole_controls: list[tuple[str, str, int, str]] = []
        for entity_id in client.states:
            if entity_id.split(".", 1)[0].casefold() not in settings.assistant.led_domains:
                continue
            registry = client.entity_registry.get(entity_id, {})
            owning_device_id = registry.get("di") if isinstance(registry, dict) else None
            if not isinstance(owning_device_id, str) or owning_device_id not in device_scope:
                continue
            depth, relation = device_scope[owning_device_id]
            sole_controls.append((entity_id, owning_device_id, depth, relation))
        if len(sole_controls) == 1:
            entity_id, owning_device_id, depth, relation = sole_controls[0]
            sibling_controls.append(
                (
                    10.0 + _indicator_relation_bonus(depth, relation),
                    entity_id,
                    ["sole_connected_light_control", relation],
                    owning_device_id,
                    relation,
                    depth,
                )
            )

    if not sibling_controls:
        log.info(
            "ASSISTANT LED skipped: no indicator control in connected satellite topology "
            "satellite=%s device=%s",
            satellite_entity_id,
            candidate_device_id,
        )
        return None

    sibling_controls.sort(key=lambda item: (-item[0], item[5], item[1]))
    top_score, entity_id, reasons, indicator_device_id, relation, depth = sibling_controls[0]
    if len(sibling_controls) > 1 and sibling_controls[1][0] == top_score:
        log.info(
            "ASSISTANT LED skipped: connected indicator tie satellite=%s candidates=%s",
            satellite_entity_id,
            [
                (item[1], item[0], item[4], item[3])
                for item in sibling_controls
                if item[0] == top_score
            ],
        )
        return None

    if indicator_device_id != candidate_device_id:
        indicator_device = client.devices.get(indicator_device_id, {})
        indicator_device_name = None
        if isinstance(indicator_device, dict):
            indicator_device_name = indicator_device.get("name_by_user") or indicator_device.get(
                "name"
            )
        log.info(
            "ASSISTANT LED selected connected control satellite=%s "
            "satellite_device=%s indicator_device=%s indicator_device_name=%r "
            "relation=%s depth=%d entity=%s score=%.1f",
            satellite_entity_id,
            candidate_device_id,
            indicator_device_id,
            indicator_device_name,
            relation,
            depth,
            entity_id,
            top_score,
        )

    state = client.states.get(satellite_entity_id, {})
    sat_ctx = client._entity_context(satellite_entity_id, state)
    indicator_device = client.devices.get(indicator_device_id, {})
    device_name = None
    if isinstance(indicator_device, dict):
        device_name = indicator_device.get("name_by_user") or indicator_device.get("name")

    return AssistantLedTarget(
        entity_id=entity_id,
        domain=entity_id.split(".", 1)[0].casefold(),
        satellite_entity_id=satellite_entity_id,
        device_id=indicator_device_id,
        device_name=device_name,
        area_id=sat_ctx.get("area_id"),
        area_name=sat_ctx.get("area_name"),
        match_reason=(f"{match_reason}:{relation}:depth{depth}:{'+'.join(reasons)}"),
    )


class AssistantLeds:
    """Reference-counted LED feedback for any asynchronous assistant adapter."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._sessions: dict[str, AssistantLedSession] = {}
        self.last_error: str | None = None
        self.last_target: dict[str, Any] | None = None

    @property
    def active_count(self) -> int:
        return len(self._sessions)

    async def begin(self, origin_context: dict[str, Any] | None) -> AssistantLedHandle | None:
        try:
            target = await _resolve_satellite_indicator(origin_context)
            if target is None:
                return None
            async with self._lock:
                existing = self._sessions.get(target.entity_id)
                if existing is not None:
                    existing.refcount += 1
                    log.info(
                        "ASSISTANT LED reuse entity=%s refcount=%d",
                        target.entity_id,
                        existing.refcount,
                    )
                    return AssistantLedHandle(target.entity_id)

                client = await _require_ha_ws()
                state = client.states.get(target.entity_id)
                if not isinstance(state, dict):
                    return None
                attrs = state.get("attributes")
                if not isinstance(attrs, dict):
                    attrs = {}
                snapshot = AssistantLedSnapshot(
                    entity_id=target.entity_id,
                    domain=target.domain,
                    state=str(state.get("state") or "unknown"),
                    attributes=dict(attrs),
                )
                session = AssistantLedSession(target=target, snapshot=snapshot)
                self._sessions[target.entity_id] = session

            try:
                await self._activate(session)
            except Exception:
                async with self._lock:
                    self._sessions.pop(target.entity_id, None)
                raise

            self.last_error = None
            self.last_target = {
                "entity_id": target.entity_id,
                "satellite_entity_id": target.satellite_entity_id,
                "device_id": target.device_id,
                "device_name": target.device_name,
                "area_name": target.area_name,
                "match_reason": target.match_reason,
                "native_effect": session.native_effect,
            }
            return AssistantLedHandle(target.entity_id)
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            log.warning("ASSISTANT LED start failed: %s", self.last_error)
            return None

    async def _activate(self, session: AssistantLedSession) -> None:
        target = session.target
        attrs = session.snapshot.attributes
        if target.domain == "light":
            data: dict[str, Any] = {}
            configured_rgb = _configured_indicator_rgb()
            if configured_rgb is not None and _light_supports_colour(attrs):
                # Home Assistant translates rgb_color to another supported colour
                # mode when required, so RGB is a safe common input.
                data["rgb_color"] = configured_rgb
            configured_effect = _find_configured_native_effect(attrs)
            if configured_effect:
                data["effect"] = configured_effect
                session.native_effect = configured_effect
            await _ha_internal_service_call("light", "turn_on", target.entity_id, data)
            if (
                not configured_effect
                and settings.assistant.led_software_pulse_enabled
                and _configured_effect_wants_software_pulse()
            ):
                session.pulse_task = asyncio.create_task(
                    self._software_pulse_light(session),
                    name=f"assistant-led-pulse-{target.entity_id}",
                )
        elif target.domain == "switch":
            await _ha_internal_service_call("switch", "turn_on", target.entity_id)
            if (
                settings.assistant.led_software_pulse_enabled
                and _configured_effect_wants_software_pulse()
            ):
                session.pulse_task = asyncio.create_task(
                    self._software_pulse_switch(session),
                    name=f"assistant-led-pulse-{target.entity_id}",
                )
        log.info(
            "ASSISTANT LED active entity=%s satellite=%s device=%s area=%r "
            "color=%r effect_requested=%r native_effect=%r software_pulse=%s "
            "previous_state=%s previous_effect=%r",
            target.entity_id,
            target.satellite_entity_id,
            target.device_id,
            target.area_name,
            settings.assistant.led_color,
            settings.assistant.led_effect,
            session.native_effect,
            session.pulse_task is not None,
            session.snapshot.state,
            session.snapshot.attributes.get("effect"),
        )

    async def _software_pulse_light(self, session: AssistantLedSession) -> None:
        entity_id = session.target.entity_id
        attrs = session.snapshot.attributes
        on_data: dict[str, Any] = {}
        configured_rgb = _configured_indicator_rgb()
        if configured_rgb is not None and _light_supports_colour(attrs):
            on_data["rgb_color"] = configured_rgb
        try:
            while True:
                await asyncio.sleep(settings.assistant.led_pulse_interval_seconds)
                await _ha_internal_service_call("light", "turn_off", entity_id)
                await asyncio.sleep(settings.assistant.led_pulse_interval_seconds)
                await _ha_internal_service_call("light", "turn_on", entity_id, on_data)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning(
                "ASSISTANT LED software light pulse stopped entity=%s error=%s", entity_id, exc
            )

    async def _software_pulse_switch(self, session: AssistantLedSession) -> None:
        entity_id = session.target.entity_id
        try:
            while True:
                await asyncio.sleep(settings.assistant.led_pulse_interval_seconds)
                await _ha_internal_service_call("switch", "turn_off", entity_id)
                await asyncio.sleep(settings.assistant.led_pulse_interval_seconds)
                await _ha_internal_service_call("switch", "turn_on", entity_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning(
                "ASSISTANT LED software switch pulse stopped entity=%s error=%s", entity_id, exc
            )

    async def end(self, handle: AssistantLedHandle | None) -> None:
        if handle is None:
            return
        session: AssistantLedSession | None = None
        async with self._lock:
            current = self._sessions.get(handle.entity_id)
            if current is None:
                return
            current.refcount -= 1
            if current.refcount > 0:
                log.info(
                    "ASSISTANT LED retain entity=%s refcount=%d",
                    handle.entity_id,
                    current.refcount,
                )
                return
            session = self._sessions.pop(handle.entity_id, None)
        if session is not None:
            await self._restore(session)

    async def _restore(self, session: AssistantLedSession) -> None:
        pulse_task = session.pulse_task
        if pulse_task is not None and not pulse_task.done():
            pulse_task.cancel()
            await asyncio.gather(pulse_task, return_exceptions=True)

        snap = session.snapshot
        try:
            if snap.domain == "light":
                restore_data = _snapshot_restore_light_data(snap)
                # If the light previously had no effect and v6.8.2 temporarily
                # enabled one, explicitly select an advertised neutral effect when
                # possible so the configured activity effect does not become the
                # light's next-on/default effect. Existing non-empty effects are
                # already restored verbatim by _snapshot_restore_light_data().
                previous_effect = snap.attributes.get("effect")
                if (
                    session.native_effect
                    and not (isinstance(previous_effect, str) and previous_effect.strip())
                    and (neutral_effect := _find_neutral_native_effect(snap.attributes))
                ):
                    restore_data["effect"] = neutral_effect
                if snap.state == "on":
                    await _ha_internal_service_call(
                        "light", "turn_on", snap.entity_id, restore_data
                    )
                else:
                    # Restore colour/effect as well as off state when possible.
                    # This may briefly turn the indicator on, but prevents our
                    # temporary green/pulse settings becoming its next-on state.
                    if restore_data:
                        await _ha_internal_service_call(
                            "light", "turn_on", snap.entity_id, restore_data
                        )
                    await _ha_internal_service_call("light", "turn_off", snap.entity_id)
            elif snap.domain == "switch":
                await _ha_internal_service_call(
                    "switch",
                    "turn_on" if snap.state == "on" else "turn_off",
                    snap.entity_id,
                )
            log.info(
                "ASSISTANT LED restored entity=%s state=%s colour_mode=%r effect=%r",
                snap.entity_id,
                snap.state,
                snap.attributes.get("color_mode"),
                snap.attributes.get("effect"),
            )
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            log.warning("ASSISTANT LED restore failed entity=%s error=%s", snap.entity_id, exc)

    async def stop_all(self) -> None:
        async with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            session.refcount = 0
            await self._restore(session)


assistant_leds = AssistantLeds()
