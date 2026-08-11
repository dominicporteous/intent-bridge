"""Topology-aware equivalence and target resolution for intent candidates."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from intent_bridge.core.text import normalize_search_text
from intent_bridge.intent_engine.models import CatalogEntity, CatalogSnapshot, IntentMatch

_TARGET_SLOT_NAMES = frozenset({"name", "area", "floor", "domain", "device_class"})
_INTENT_DEFAULT_DOMAINS = {
    "HassClimateGetTemperature": "climate",
    "HassClimateSetTemperature": "climate",
    "HassFanSetSpeed": "fan",
    "HassLightSet": "light",
    "HassMediaNext": "media_player",
    "HassMediaPause": "media_player",
    "HassMediaPlayerMute": "media_player",
    "HassMediaPlayerUnmute": "media_player",
    "HassMediaPrevious": "media_player",
    "HassMediaSearchAndPlay": "media_player",
    "HassMediaUnpause": "media_player",
    "HassLawnMowerDock": "lawn_mower",
    "HassLawnMowerStartMowing": "lawn_mower",
    "HassSetPosition": "cover",
    "HassSetVolume": "media_player",
    "HassSetVolumeRelative": "media_player",
    "HassVacuumCleanArea": "vacuum",
    "HassVacuumReturnToBase": "vacuum",
    "HassVacuumStart": "vacuum",
}


def _normal(value: object) -> str:
    return normalize_search_text(str(value or ""))


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _freeze(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class ResolvedCandidate:
    match: IntentMatch
    entity_ids: frozenset[str]
    semantic_key: tuple[Any, ...]
    specificity: int


def _area_id(match: IntentMatch, catalog: CatalogSnapshot) -> str | None:
    area_slot = match.slots.get("area")
    if area_slot is None:
        return None
    metadata_id = area_slot.metadata.get("area_id")
    if isinstance(metadata_id, str) and metadata_id:
        return metadata_id
    wanted = _normal(area_slot.value)
    matches = [
        area.area_id
        for area in catalog.areas
        if wanted in {_normal(area.name), *(_normal(alias) for alias in area.aliases)}
    ]
    return matches[0] if len(matches) == 1 else None


def _floor_id(match: IntentMatch, catalog: CatalogSnapshot) -> str | None:
    floor_slot = match.slots.get("floor")
    if floor_slot is None:
        return None
    metadata_id = floor_slot.metadata.get("floor_id")
    if isinstance(metadata_id, str) and metadata_id:
        return metadata_id
    wanted = _normal(floor_slot.value)
    matches = [
        floor.floor_id
        for floor in catalog.floors
        if wanted in {_normal(floor.name), *(_normal(alias) for alias in floor.aliases)}
    ]
    return matches[0] if len(matches) == 1 else None


def _matching_named_entities(match: IntentMatch, catalog: CatalogSnapshot) -> list[CatalogEntity]:
    name_slot = match.slots.get("name")
    if name_slot is None:
        return []
    wanted = _normal(name_slot.value)
    catalog_matches = [
        entity
        for entity in catalog.entities
        if wanted
        in {
            _normal(entity.name),
            _normal(entity.entity_id),
            *(_normal(alias) for alias in entity.aliases),
        }
    ]
    if catalog_matches:
        return catalog_matches
    metadata_id = name_slot.metadata.get("entity_id")
    if isinstance(metadata_id, str) and metadata_id:
        return [entity for entity in catalog.entities if entity.entity_id == metadata_id]
    return []


def _matches_device_class(entity: CatalogEntity, wanted: str) -> bool:
    """Match an explicit device class without requiring registry metadata.

    Some Home Assistant integrations do not expose ``device_class`` on every
    entity.  A cover named "Living Room Blinds" is still a safe match for the
    spoken class ``blind``; it is not safe to broaden that request to every
    cover in the area.
    """

    actual = _normal(entity.device_class)
    if actual:
        return actual == wanted
    forms = {
        "awning": ("awning", "awnings"),
        "blind": ("blind", "blinds"),
        "curtain": ("curtain", "curtains", "drape", "drapes"),
        "damper": ("damper", "dampers"),
        "door": ("door", "doors"),
        "garage": ("garage door", "garage doors"),
        "gate": ("gate", "gates"),
        "shade": ("shade", "shades"),
        "shutter": ("shutter", "shutters"),
        "window": ("window", "windows"),
    }.get(wanted, (wanted,))
    labels = (_normal(entity.name), *(_normal(alias) for alias in entity.aliases))
    return any(
        re.search(rf"(?<!\w){re.escape(form)}(?!\w)", label) for form in forms for label in labels
    )


def resolve_candidate(
    match: IntentMatch,
    catalog: CatalogSnapshot,
    origin_context: Mapping[str, object] | None = None,
) -> ResolvedCandidate:
    """Resolve a parser match and construct a provider-neutral equivalence key."""

    domain_slot = match.slots.get("domain")
    if domain_slot:
        domain = _normal(domain_slot.value)
    elif context_domain := _normal(match.context.get("domain")):
        domain = context_domain
    else:
        domain = _normal(_INTENT_DEFAULT_DOMAINS.get(match.intent_name, ""))
    device_class_slot = match.slots.get("device_class")
    device_class = _normal(device_class_slot.value) if device_class_slot else ""

    named_entities = _matching_named_entities(match, catalog)
    area_id = _area_id(match, catalog)
    floor_id = _floor_id(match, catalog)
    has_explicit_name = "name" in match.slots
    has_explicit_area = "area" in match.slots
    has_explicit_floor = "floor" in match.slots
    has_explicit_target = has_explicit_name or has_explicit_area or has_explicit_floor
    unresolved_origin_area = False

    if not area_id and not named_entities and not has_explicit_target and domain and origin_context:
        origin_area_id = origin_context.get("area_id")
        if isinstance(origin_area_id, str) and origin_area_id:
            area_id = origin_area_id
        else:
            origin_area_name = origin_context.get("area_name")
            wanted = _normal(origin_area_name)
            area_matches = [area.area_id for area in catalog.areas if _normal(area.name) == wanted]
            if len(area_matches) == 1:
                area_id = area_matches[0]
            else:
                unresolved_origin_area = True

    if len(named_entities) == 1:
        entities = named_entities
        specificity = 3
    else:
        entities = list(catalog.entities)
        specificity = 0
        if has_explicit_name:
            entities = []
        elif area_id:
            entities = [entity for entity in entities if entity.area_id == area_id]
            specificity = 2
        elif has_explicit_area:
            entities = []
        elif floor_id:
            area_ids = {area.area_id for area in catalog.areas if area.floor_id == floor_id}
            entities = [entity for entity in entities if entity.area_id in area_ids]
            specificity = 1
        elif has_explicit_floor:
            entities = []
        elif unresolved_origin_area:
            entities = []
        elif not domain and not device_class:
            entities = []

        if domain:
            entities = [entity for entity in entities if _normal(entity.domain) == domain]
            if domain == "light":
                entities = [entity for entity in entities if not entity.is_indicator]
        if device_class:
            entities = [
                entity for entity in entities if _matches_device_class(entity, device_class)
            ]

    entity_ids = frozenset(entity.entity_id for entity in entities)
    action_slots = tuple(
        sorted(
            (name, _freeze(slot.value))
            for name, slot in match.slots.items()
            if name not in _TARGET_SLOT_NAMES
        )
    )
    if entity_ids:
        target_key: Any = tuple(sorted(entity_ids))
    else:
        target_key = tuple(
            sorted(
                (name, _freeze(slot.value))
                for name, slot in match.slots.items()
                if name in _TARGET_SLOT_NAMES
            )
        )

    return ResolvedCandidate(
        match=match,
        entity_ids=entity_ids,
        semantic_key=(match.intent_name, target_key, action_slots),
        specificity=specificity,
    )


__all__ = ["ResolvedCandidate", "resolve_candidate"]
