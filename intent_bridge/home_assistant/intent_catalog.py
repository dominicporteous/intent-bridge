"""Adapt Home Assistant's live caches to the intent engine catalog model."""

from __future__ import annotations

from typing import Any, Protocol

from intent_bridge.core.text import normalize_search_text
from intent_bridge.home_assistant.catalog import entity_context
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


def snapshot_from_client(client: CachedHomeAssistant) -> CatalogSnapshot:
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
                entity_category=(
                    str(registry["ec"]) if registry.get("ec") not in (None, "") else None
                ),
                is_indicator=is_indicator_control(client, entity_id),
            )
        )

    return CatalogSnapshot(
        entities=tuple(entities),
        areas=tuple(areas),
        floors=tuple(floors),
    )


class HomeAssistantCatalogProvider:
    """Take a fresh immutable snapshot from a reconnecting cache on each request."""

    def __init__(self, client_provider) -> None:
        self._client_provider = client_provider

    def snapshot(self) -> CatalogSnapshot:
        client = self._client_provider()
        if client is None:
            return CatalogSnapshot()
        return snapshot_from_client(client)


__all__ = ["CachedHomeAssistant", "HomeAssistantCatalogProvider", "snapshot_from_client"]
