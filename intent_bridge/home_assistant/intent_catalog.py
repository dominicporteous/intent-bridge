"""Adapt Home Assistant's live caches to the intent engine catalog model."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from intent_bridge.core.text import normalize_search_text
from intent_bridge.home_assistant.catalog import entity_context
from intent_bridge.home_assistant.intent_services import power_intents_supported_by_domain
from intent_bridge.indicators.topology import is_indicator_control
from intent_bridge.intent_engine.models import (
    CatalogArea,
    CatalogEntity,
    CatalogFloor,
    CatalogMeasurement,
    CatalogSnapshot,
)

_UNAVAILABLE_STATES = frozenset({"", "none", "unknown", "unavailable"})
_TEMPERATURE_UNITS = frozenset(
    {"c", "celsius", "f", "fahrenheit", "k", "kelvin"}
)
_TEMPERATURE_NAME_HINTS = frozenset({"temp", "temperature", "thermometer"})
LOGGER = logging.getLogger(__name__)


def _measurement_quantity(value: object) -> str:
    return normalize_search_text(value).replace(" ", "_")


def _measurement_unit(attributes: dict[str, Any], quantity: str) -> str | None:
    if "temperature" in quantity:
        value = attributes.get("temperature_unit") or attributes.get("unit_of_measurement")
    else:
        value = attributes.get("unit_of_measurement")
    if isinstance(value, str) and value.strip():
        return value.strip()
    if quantity in {"battery", "humidity", "position"}:
        return "%"
    return None


def _is_temperature_unit(value: object) -> bool:
    normalized = normalize_search_text(value).replace(" ", "")
    return normalized in _TEMPERATURE_UNITS


def _has_temperature_name(*values: object) -> bool:
    return any(
        _TEMPERATURE_NAME_HINTS.intersection(normalize_search_text(value).split())
        for value in values
    )


def _inferred_sensor_temperature(
    state: dict[str, Any],
    attributes: dict[str, Any],
    entity_id: str,
    entity_name: str,
) -> CatalogMeasurement | None:
    """Expose only strongly identified numeric sensor temperatures.

    Some integrations omit ``device_class`` while still publishing a numeric
    temperature state. Require a sensor domain, temperature-bearing identity,
    and a recognized unit before promoting the state to a reading.
    """

    if entity_id.split(".", 1)[0] != "sensor":
        return None
    if not _has_temperature_name(entity_id, entity_name):
        return None
    unit = _measurement_unit(attributes, "temperature")
    if not _is_temperature_unit(unit):
        return None
    raw_state = state.get("state")
    state_text = str(raw_state).strip() if raw_state is not None else ""
    if state_text.casefold() in _UNAVAILABLE_STATES:
        return None
    try:
        float(state_text)
    except ValueError:
        return None
    return CatalogMeasurement("temperature", state_text, unit, source="inferred_state")


def _measurements(
    state: dict[str, Any],
    attributes: dict[str, Any],
    *,
    entity_id: str,
    entity_name: str,
) -> tuple[CatalogMeasurement, ...]:
    """Extract readings using HA's semantic metadata rather than entity domains."""

    readings: dict[str, CatalogMeasurement] = {}
    raw_state = state.get("state")
    state_text = str(raw_state).strip() if raw_state is not None else ""
    device_class = _measurement_quantity(attributes.get("device_class"))
    if device_class and state_text.casefold() not in _UNAVAILABLE_STATES:
        try:
            float(state_text)
        except ValueError:
            pass
        else:
            readings[device_class] = CatalogMeasurement(
                quantity=device_class,
                value=state_text,
                unit=_measurement_unit(attributes, device_class),
            )

    for key, raw_value in attributes.items():
        if (
            not isinstance(key, str)
            or isinstance(raw_value, bool)
            or not isinstance(raw_value, (int, float))
        ):
            continue
        normalized_key = _measurement_quantity(key)
        if normalized_key.startswith("current_"):
            quantity = normalized_key.removeprefix("current_")
        elif normalized_key.endswith("_level"):
            quantity = normalized_key.removesuffix("_level")
        else:
            continue
        if not quantity or quantity in readings:
            continue
        readings[quantity] = CatalogMeasurement(
            quantity=quantity,
            value=str(raw_value),
            unit=_measurement_unit(attributes, quantity),
            source=key,
        )

    if "temperature" not in readings:
        inferred_temperature = _inferred_sensor_temperature(
            state, attributes, entity_id, entity_name
        )
        if inferred_temperature is not None:
            readings["temperature"] = inferred_temperature

    return tuple(readings[key] for key in sorted(readings))


class CachedHomeAssistant(Protocol):
    states: dict[str, dict[str, Any]]
    entity_registry: dict[str, dict[str, Any]]
    devices: dict[str, dict[str, Any]]
    areas: dict[str, dict[str, Any]]


class CatalogCacheChangeSource(CachedHomeAssistant, Protocol):
    """Live HA cache with synchronous change notifications for catalog publication."""

    def add_catalog_cache_listener(self, listener: Callable[[str], None]) -> None: ...

    def remove_catalog_cache_listener(self, listener: Callable[[str], None]) -> None: ...

    async def refresh_registries(self, *, force: bool = False) -> None: ...


@dataclass(slots=True)
class _CatalogSourceCopy:
    """A stable shallow copy of HA cache maps for an off-loop full rebuild.

    The WebSocket client replaces state and registry entries rather than mutating
    their nested mappings. Copying the top-level cache dictionaries on the event
    loop therefore gives a coherent source set to the worker without holding a
    lock or iterating a map that the reader task may change.
    """

    states: dict[str, dict[str, Any]]
    entity_registry: dict[str, dict[str, Any]]
    devices: dict[str, dict[str, Any]]
    areas: dict[str, dict[str, Any]]
    floors: dict[str, dict[str, Any]]
    services: dict[str, Any] | None


def _copy_catalog_source(client: CachedHomeAssistant) -> _CatalogSourceCopy:
    """Capture every catalog input; never patch a previously published catalog."""

    raw_floors = getattr(client, "floors", {})
    raw_services = getattr(client, "services", None)
    return _CatalogSourceCopy(
        states=dict(client.states),
        entity_registry=dict(client.entity_registry),
        devices=dict(client.devices),
        areas=dict(client.areas),
        floors=dict(raw_floors) if isinstance(raw_floors, dict) else {},
        services=dict(raw_services) if isinstance(raw_services, dict) else None,
    )


def _clean_aliases(values: list[object], canonical: str) -> tuple[str, ...]:
    canonical_normal = normalize_search_text(canonical)
    aliases: dict[str, str] = {}
    for value in values:
        if not isinstance(value, str) or not value.strip():
            continue
        cleaned = value.strip()
        normal = normalize_search_text(cleaned)
        if not normal or normal == canonical_normal:
            continue
        aliases.setdefault(normal, cleaned)
    return tuple(aliases[key] for key in sorted(aliases))


def _registry_entry_is_disabled(entry: object) -> bool:
    """Read the standard and compact registry markers used by HA WebSocket APIs."""

    if not isinstance(entry, dict):
        return False
    for key in ("disabled_by", "db", "disabled"):
        value = entry.get(key)
        if isinstance(value, str):
            if normalize_search_text(value) not in {"", "none", "false", "0"}:
                return True
        elif value:
            return True
    return False


def _state_is_available(state: object) -> bool:
    return normalize_search_text(state) not in _UNAVAILABLE_STATES


def snapshot_from_client(client: CachedHomeAssistant) -> CatalogSnapshot:
    raw_services = getattr(client, "services", None)
    services = raw_services if isinstance(raw_services, dict) else None
    areas: list[CatalogArea] = []
    for area_id, raw_area in sorted(client.areas.items()):
        if not isinstance(raw_area, dict):
            continue
        name = raw_area.get("name")
        if not isinstance(name, str) or not name.strip():
            name = area_id.replace("_", " ").title()
        aliases = raw_area.get("aliases") or raw_area.get("alias") or []
        if not isinstance(aliases, list):
            aliases = []
        areas.append(
            CatalogArea(
                area_id=area_id,
                name=name.strip(),
                aliases=_clean_aliases([*aliases, area_id.replace("_", " ")], name),
                floor_id=(
                    str(raw_area["floor_id"])
                    if raw_area.get("floor_id") not in (None, "")
                    else None
                ),
            )
        )

    raw_floors = getattr(client, "floors", {})
    floors: list[CatalogFloor] = []
    if isinstance(raw_floors, dict):
        for floor_id, raw_floor in sorted(raw_floors.items()):
            if not isinstance(raw_floor, dict):
                continue
            name = raw_floor.get("name")
            if not isinstance(name, str) or not name.strip():
                name = floor_id.replace("_", " ").title()
            aliases = raw_floor.get("aliases") or []
            if not isinstance(aliases, list):
                aliases = []
            floors.append(
                CatalogFloor(
                    floor_id=floor_id,
                    name=name.strip(),
                    aliases=_clean_aliases([*aliases, floor_id.replace("_", " ")], name),
                )
            )

    entities: list[CatalogEntity] = []
    for entity_id, state in sorted(client.states.items()):
        if not isinstance(state, dict) or "." not in entity_id:
            continue
        attributes = state.get("attributes")
        if not isinstance(attributes, dict):
            attributes = {}
        context = entity_context(client, entity_id, state)
        friendly_name = context.get("friendly_name")
        registry_name = context.get("registry_name")
        local_name = entity_id.split(".", 1)[1].replace("_", " ")
        canonical_name = next(
            (
                value.strip()
                for value in (friendly_name, registry_name, local_name)
                if isinstance(value, str) and value.strip()
            ),
            entity_id,
        )
        registry = client.entity_registry.get(entity_id, {})
        device_id = registry.get("di") if isinstance(registry, dict) else None
        device = client.devices.get(device_id, {}) if isinstance(device_id, str) else {}
        registry_aliases = registry.get("aliases") or registry.get("al") or []
        if not isinstance(registry_aliases, list):
            registry_aliases = []
        attribute_aliases = attributes.get("aliases") or []
        if not isinstance(attribute_aliases, list):
            attribute_aliases = []

        entities.append(
            CatalogEntity(
                entity_id=entity_id,
                name=canonical_name,
                aliases=_clean_aliases(
                    [
                        registry_name,
                        context.get("device_name"),
                        local_name,
                        *registry_aliases,
                        *attribute_aliases,
                    ],
                    canonical_name,
                ),
                domain=entity_id.split(".", 1)[0],
                area_id=(str(context["area_id"]) if context.get("area_id") else None),
                device_class=(
                    str(attributes["device_class"])
                    if attributes.get("device_class") not in (None, "")
                    else None
                ),
                state=(str(state["state"]) if state.get("state") is not None else None),
                measurements=_measurements(
                    state,
                    attributes,
                    entity_id=entity_id,
                    entity_name=canonical_name,
                ),
                supported_intents=power_intents_supported_by_domain(
                    entity_id.split(".", 1)[0],
                    services,
                ),
                entity_category=(
                    str(registry["ec"]) if registry.get("ec") not in (None, "") else None
                ),
                is_indicator=is_indicator_control(client, entity_id),
                is_enabled=not (
                    _registry_entry_is_disabled(registry) or _registry_entry_is_disabled(device)
                ),
                is_available=_state_is_available(state.get("state")),
            )
        )

    return CatalogSnapshot(
        entities=tuple(entities),
        areas=tuple(areas),
        floors=tuple(floors),
    )


class HomeAssistantCatalogPublisher:
    """Publish complete immutable catalog snapshots without request-path rebuilds.

    Change events never mutate an existing ``CatalogSnapshot``. They only wake
    this publisher, which captures *all* of the WebSocket cache maps and builds
    a replacement snapshot. That deliberately trades a bounded background
    rebuild for immunity to incremental-update drift.
    """

    def __init__(
        self,
        client: CatalogCacheChangeSource,
        *,
        refresh_seconds: float = 60.0,
        event_debounce_seconds: float = 0.5,
        minimum_refresh_seconds: float = 1.0,
    ) -> None:
        if refresh_seconds <= 0:
            raise ValueError("refresh_seconds must be greater than zero")
        if event_debounce_seconds < 0:
            raise ValueError("event_debounce_seconds must not be negative")
        if minimum_refresh_seconds < 0:
            raise ValueError("minimum_refresh_seconds must not be negative")

        self.client = client
        self._refresh_seconds = refresh_seconds
        self._event_debounce_seconds = event_debounce_seconds
        self._minimum_refresh_seconds = minimum_refresh_seconds
        self._changed = asyncio.Event()
        self._registry_refresh_needed = False
        self._task: asyncio.Task[None] | None = None
        self._snapshot: CatalogSnapshot | None = None
        self._published_at: float | None = None
        self._last_refresh_started_at = 0.0
        self._generation = 0
        self._build_failures = 0
        self._stopping = False

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def age_seconds(self) -> float | None:
        if self._published_at is None:
            return None
        return max(0.0, time.monotonic() - self._published_at)

    @property
    def build_failures(self) -> int:
        return self._build_failures

    def snapshot(self) -> CatalogSnapshot | None:
        """Return the last whole snapshot; readers never observe a partial build."""

        return self._snapshot

    async def start(self) -> None:
        """Publish an initial snapshot, then retain it in a coalescing worker."""

        if self._task is not None:
            return
        self._stopping = False
        self.client.add_catalog_cache_listener(self._on_cache_change)
        await self._publish("startup")
        self._task = asyncio.create_task(
            self._run(),
            name="ha-catalog-publisher",
        )

    async def stop(self) -> None:
        """Stop publication and detach from the HA client before it is closed."""

        self._stopping = True
        self.client.remove_catalog_cache_listener(self._on_cache_change)
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    def _on_cache_change(self, change: str) -> None:
        """Schedule a replacement snapshot; this must remain reader-loop cheap."""

        if self._stopping:
            return
        if change == "registry":
            # A registry event only includes an ID/action. Fetch the complete
            # registry before the next complete catalog build so renames,
            # reassignments, removals, and area/device moves cannot drift.
            self._registry_refresh_needed = True
        self._changed.set()

    async def _run(self) -> None:
        while True:
            event_triggered = False
            try:
                await asyncio.wait_for(self._changed.wait(), timeout=self._refresh_seconds)
                event_triggered = True
            except TimeoutError:
                pass

            self._changed.clear()
            if event_triggered and self._event_debounce_seconds:
                await asyncio.sleep(self._event_debounce_seconds)
                # All arrivals during the debounce interval are included by
                # this rebuild. A later arrival will set the event again and
                # cause exactly one follow-up full rebuild.
                self._changed.clear()

            remaining = self._minimum_refresh_seconds - (
                time.monotonic() - self._last_refresh_started_at
            )
            if remaining > 0:
                await asyncio.sleep(remaining)

            refresh_registry = self._registry_refresh_needed
            self._registry_refresh_needed = False
            if refresh_registry:
                try:
                    await self.client.refresh_registries(force=True)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    # The last good registry cache remains usable. The next
                    # periodic/event rebuild retries from a full source copy.
                    LOGGER.exception("HA catalog registry refresh failed before rebuild")

            await self._publish("event" if event_triggered else "periodic")

    async def _publish(self, reason: str) -> None:
        self._last_refresh_started_at = time.monotonic()
        try:
            source = _copy_catalog_source(self.client)
            # ``snapshot_from_client`` performs CPU-bound normalisation for
            # every entity. Keep it off the request path and away from the
            # WebSocket reader; ``source`` is a stable complete input set.
            snapshot = await asyncio.to_thread(snapshot_from_client, source)
        except asyncio.CancelledError:
            raise
        except Exception:
            self._build_failures += 1
            LOGGER.exception("HA catalog snapshot rebuild failed reason=%s", reason)
            return

        self._snapshot = snapshot
        self._published_at = time.monotonic()
        self._generation += 1
        LOGGER.debug(
            "HA catalog snapshot published generation=%d reason=%s entities=%d areas=%d floors=%d",
            self._generation,
            reason,
            len(snapshot.entities),
            len(snapshot.areas),
            len(snapshot.floors),
        )


class HomeAssistantCatalogProvider:
    """Serve a background snapshot when its source matches the active HA cache."""

    def __init__(self, client_provider, *, publisher_provider: Callable[[], Any | None] | None = None) -> None:
        self._client_provider = client_provider
        self._publisher_provider = publisher_provider

    def snapshot(self) -> CatalogSnapshot:
        client = self._client_provider()
        if client is None:
            return CatalogSnapshot()
        if self._publisher_provider is not None:
            publisher = self._publisher_provider()
            if getattr(publisher, "client", None) is client:
                snapshot = publisher.snapshot()
                if snapshot is not None:
                    return snapshot
        return snapshot_from_client(client)


__all__ = [
    "CachedHomeAssistant",
    "HomeAssistantCatalogProvider",
    "HomeAssistantCatalogPublisher",
    "snapshot_from_client",
]
