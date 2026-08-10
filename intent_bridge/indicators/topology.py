"""Pure topology and scoring policy for satellite indicator discovery."""

from typing import Any, Protocol

from intent_bridge.core.text import normalize_search_text as _normalise_search_text


class IndicatorCatalog(Protocol):
    states: dict[str, dict[str, Any]]
    entity_registry: dict[str, dict[str, Any]]
    devices: dict[str, dict[str, Any]]


def indicator_identity_text(catalog: IndicatorCatalog, entity_id: str) -> str:
    state = catalog.states.get(entity_id, {})
    attrs = state.get("attributes", {}) if isinstance(state, dict) else {}
    if not isinstance(attrs, dict):
        attrs = {}
    registry = catalog.entity_registry.get(entity_id, {})
    device_id = registry.get("di") if isinstance(registry, dict) else None
    device = catalog.devices.get(device_id, {}) if isinstance(device_id, str) else {}
    device_name = None
    if isinstance(device, dict):
        device_name = device.get("name_by_user") or device.get("name")
    parts = [
        entity_id.split(".", 1)[-1],
        attrs.get("friendly_name"),
        registry.get("en") if isinstance(registry, dict) else None,
        device_name,
    ]
    return " ".join(_normalise_search_text(part) for part in parts if part)


def device_via_device_id(catalog: IndicatorCatalog, device_id: str) -> str | None:
    device = catalog.devices.get(device_id, {})
    if not isinstance(device, dict):
        return None
    for key in ("via_device_id", "via_device"):
        value = device.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def connected_satellite_device_scope(
    catalog: IndicatorCatalog,
    anchor_device_id: str,
    *,
    max_depth: int = 2,
) -> dict[str, tuple[int, str]]:
    adjacency: dict[str, set[str]] = {device_id: set() for device_id in catalog.devices}
    for child_id in list(catalog.devices):
        parent_id = device_via_device_id(catalog, child_id)
        if not parent_id or parent_id not in catalog.devices:
            continue
        adjacency.setdefault(child_id, set()).add(parent_id)
        adjacency.setdefault(parent_id, set()).add(child_id)

    distances: dict[str, int] = {anchor_device_id: 0}
    frontier = [anchor_device_id]
    while frontier:
        current = frontier.pop(0)
        depth = distances[current]
        if depth >= max_depth:
            continue
        for neighbour in sorted(adjacency.get(current, set())):
            if neighbour in distances:
                continue
            distances[neighbour] = depth + 1
            frontier.append(neighbour)

    anchor_parent = device_via_device_id(catalog, anchor_device_id)
    result: dict[str, tuple[int, str]] = {}
    for device_id, depth in distances.items():
        if depth == 0:
            relation = "same_device"
        elif device_id == anchor_parent:
            relation = "parent_device"
        elif device_via_device_id(catalog, device_id) == anchor_device_id:
            relation = "child_device"
        elif (
            anchor_parent
            and device_id != anchor_device_id
            and device_via_device_id(catalog, device_id) == anchor_parent
        ):
            relation = "sibling_device"
        else:
            relation = f"connected_depth_{depth}"
        result[device_id] = (depth, relation)
    return result


def indicator_relation_bonus(depth: int, relation: str) -> float:
    bonuses = {
        "same_device": 30.0,
        "child_device": 20.0,
        "sibling_device": 18.0,
        "parent_device": 10.0,
    }
    return bonuses.get(relation, max(0.0, 10.0 - (depth * 2.0)))


def indicator_score(catalog: IndicatorCatalog, entity_id: str) -> tuple[float, list[str]]:
    identity = indicator_identity_text(catalog, entity_id)
    domain = entity_id.split(".", 1)[0].casefold()
    score = 0.0
    reasons: list[str] = []
    for word, points in (
        ("led", 120.0),
        ("ring", 105.0),
        ("indicator", 95.0),
        ("notification", 70.0),
        ("status", 55.0),
        ("pixel", 50.0),
    ):
        if word in identity.split() or word in identity:
            score += points
            reasons.append(word)
    if domain == "light":
        score += 12.0
        reasons.append("light_domain")
    elif domain == "switch":
        score += 4.0
        reasons.append("switch_domain")
    return score, reasons


__all__ = [
    "IndicatorCatalog",
    "connected_satellite_device_scope",
    "device_via_device_id",
    "indicator_identity_text",
    "indicator_relation_bonus",
    "indicator_score",
]
