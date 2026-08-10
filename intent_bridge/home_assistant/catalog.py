"""Pure queries over Home Assistant's cached entity and registry catalog."""

from typing import Any, Protocol

from intent_bridge.core.text import normalize_search_text as _normalise_search_text


class HomeAssistantCatalog(Protocol):
    states: dict[str, dict[str, Any]]
    entity_registry: dict[str, dict[str, Any]]
    devices: dict[str, dict[str, Any]]
    areas: dict[str, dict[str, Any]]


def entity_context(
    catalog: HomeAssistantCatalog,
    entity_id: str,
    state: dict[str, Any],
) -> dict[str, Any]:
    registry = catalog.entity_registry.get(entity_id, {})
    attributes = state.get("attributes") if isinstance(state, dict) else {}
    if not isinstance(attributes, dict):
        attributes = {}

    device_id = registry.get("di")
    device = catalog.devices.get(device_id, {}) if isinstance(device_id, str) else {}
    area_id = registry.get("ai")
    if not area_id and isinstance(device, dict):
        area_id = device.get("area_id")
    area = catalog.areas.get(area_id, {}) if isinstance(area_id, str) else {}

    return {
        "friendly_name": attributes.get("friendly_name"),
        "registry_name": registry.get("en"),
        "device_id": device_id,
        "device_name": (
            device.get("name_by_user") or device.get("name") if isinstance(device, dict) else None
        ),
        "area_id": area_id,
        "area_name": area.get("name") if isinstance(area, dict) else None,
    }


def resolve_area_reference(
    catalog: HomeAssistantCatalog,
    *,
    area_id: str | None = None,
    area_name: str | None = None,
) -> tuple[str | None, str | None]:
    if isinstance(area_id, str) and area_id.strip():
        area_id = area_id.strip()
        area = catalog.areas.get(area_id)
        if isinstance(area, dict):
            return area_id, area.get("name") or area_name

    if isinstance(area_name, str) and area_name.strip():
        wanted = _normalise_search_text(area_name)
        matches = [
            (candidate_id, area.get("name"))
            for candidate_id, area in catalog.areas.items()
            if isinstance(area, dict) and _normalise_search_text(area.get("name")) == wanted
        ]
        if len(matches) == 1:
            return matches[0]
    return None, None


def resolve_device_origin(
    catalog: HomeAssistantCatalog,
    *,
    device_id: str | None = None,
    device_name: str | None = None,
    area_id: str | None = None,
    area_name: str | None = None,
) -> dict[str, Any]:
    device: dict[str, Any] | None = None
    resolved_device_id: str | None = None

    if isinstance(device_id, str) and device_id.strip():
        candidate = catalog.devices.get(device_id.strip())
        if isinstance(candidate, dict):
            resolved_device_id = device_id.strip()
            device = candidate

    if device is None and isinstance(device_name, str) and device_name.strip():
        wanted = _normalise_search_text(device_name)
        matches = [
            (candidate_id, candidate)
            for candidate_id, candidate in catalog.devices.items()
            if isinstance(candidate, dict)
            and _normalise_search_text(candidate.get("name_by_user") or candidate.get("name"))
            == wanted
        ]
        if len(matches) == 1:
            resolved_device_id, device = matches[0]

    resolved_device_name = device_name
    if isinstance(device, dict):
        resolved_device_name = device.get("name_by_user") or device.get("name") or device_name
        if not area_id:
            area_id = device.get("area_id")

    if resolved_device_id and not area_id:
        entity_area_ids = {
            entry.get("ai")
            for entry in catalog.entity_registry.values()
            if isinstance(entry, dict)
            and entry.get("di") == resolved_device_id
            and isinstance(entry.get("ai"), str)
            and entry.get("ai")
        }
        if len(entity_area_ids) == 1:
            area_id = next(iter(entity_area_ids))

    resolved_area_id, resolved_area_name = resolve_area_reference(
        catalog, area_id=area_id, area_name=area_name
    )
    if not resolved_area_id and isinstance(area_id, str) and area_id.strip():
        resolved_area_id = area_id.strip()
        resolved_area_name = area_name

    return {
        "device_id": resolved_device_id
        or (device_id.strip() if isinstance(device_id, str) and device_id.strip() else None),
        "device_name": resolved_device_name,
        "area_id": resolved_area_id,
        "area_name": resolved_area_name,
    }


def area_mentioned_in_text(catalog: HomeAssistantCatalog, text: str) -> tuple[str, str] | None:
    haystack = f" {_normalise_search_text(text)} "
    matches: list[tuple[str, str]] = []
    for candidate_id, area in catalog.areas.items():
        if not isinstance(area, dict):
            continue
        name = area.get("name")
        normalised = _normalise_search_text(name)
        if normalised and f" {normalised} " in haystack:
            matches.append((candidate_id, str(name)))
    return matches[0] if len(matches) == 1 else None


def entities_in_area(catalog: HomeAssistantCatalog, domain: str, area_id: str) -> list[str]:
    matches = [
        entity_id
        for entity_id, state in catalog.states.items()
        if entity_id.split(".", 1)[0].casefold() == domain.casefold()
        and entity_context(catalog, entity_id, state).get("area_id") == area_id
    ]
    return sorted(matches)


__all__ = [
    "HomeAssistantCatalog",
    "area_mentioned_in_text",
    "entities_in_area",
    "entity_context",
    "resolve_area_reference",
    "resolve_device_origin",
]
