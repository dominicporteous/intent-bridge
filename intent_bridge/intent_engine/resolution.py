"""Topology-aware equivalence and target resolution for intent candidates."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from intent_bridge.core.text import normalize_search_text
from intent_bridge.intent_engine.models import (
    CatalogEntity,
    CatalogSnapshot,
    IntentMatch,
    SlotValue,
)

_TARGET_SLOT_NAMES = frozenset(
    {"name", "area", "floor", "domain", "device_class", "entity_id"}
)
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


def _surface_target_phrases(
    entity: CatalogEntity,
    catalog: CatalogSnapshot,
) -> frozenset[str]:
    """Return full and area-relative catalog labels for spoken target matching."""

    labels = {
        entity.name,
        *entity.aliases,
        entity.entity_id.split(".", 1)[-1].replace("_", " "),
    }
    area = next((area for area in catalog.areas if area.area_id == entity.area_id), None)
    area_words: set[str] = set()
    if area is not None:
        for area_label in (area.name, area.area_id, *area.aliases):
            normalized_area = _normal(area_label)
            area_words.update(normalized_area.split())
            area_words.add(normalized_area.replace(" ", ""))

    phrases: set[str] = set()
    for label in labels:
        normalized_label = _normal(label)
        if not normalized_label:
            continue
        phrases.add(normalized_label)
        relative_label = " ".join(
            word for word in normalized_label.split() if word not in area_words
        )
        if relative_label:
            phrases.add(relative_label)
    return frozenset(phrases)


def _contains_surface_phrase(text: str, phrase: str) -> bool:
    return bool(re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text))


def _has_plural_domain_target(text: str, domain: str) -> bool:
    """Keep plural domain words as group requests, not abbreviated names."""

    plural_forms = {
        "cover": ("blinds", "covers", "curtains", "shades", "shutters"),
        "fan": ("fans",),
        "light": ("lights", "lamps"),
        "lock": ("locks", "deadbolts"),
        "media_player": ("media players", "speakers", "televisions", "tvs"),
        "switch": ("switches",),
    }.get(domain, ())
    return any(_contains_surface_phrase(text, form) for form in plural_forms)


def _surface_target_domains(intent_name: str, domain: str) -> frozenset[str]:
    """Return domains that can satisfy an exact spoken target for an intent."""


    if intent_name in {"HassTurnOn", "HassTurnOff"} and domain == "light":
        return frozenset({"light", "switch"})
    return frozenset({domain})


def _surface_named_entity(
    text: str | None,
    catalog: CatalogSnapshot,
    domains: frozenset[str],
    area_ids: frozenset[str] | None,
) -> CatalogEntity | None:
    """Find one catalog entity named by the surface target phrase.

    Packaged OHF grammar intentionally treats terms such as ``lamp`` as the
    ``light`` domain.  A unique label within the requested/origin area is more
    specific than that broad domain target, but ties remain unresolved so the
    normal clarification path can handle them safely.
    """

    normalized_text = _normal(text)
    if not normalized_text:
        return None
    candidates = [
        entity
        for entity in catalog.entities
        if _normal(entity.domain) in domains
        and (area_ids is None or entity.area_id in area_ids)
    ]
    matched: list[tuple[int, CatalogEntity]] = []
    for entity in candidates:
        phrase_lengths = [
            len(phrase.split())
            for phrase in _surface_target_phrases(entity, catalog)
            if _contains_surface_phrase(normalized_text, phrase)
        ]
        if phrase_lengths:
            matched.append((max(phrase_lengths), entity))
    if not matched:
        return None
    best_length = max(length for length, _ in matched)
    winners = [entity for length, entity in matched if length == best_length]
    return winners[0] if len(winners) == 1 else None


def _with_surface_name(match: IntentMatch, entity: CatalogEntity) -> IntentMatch:
    """Replace a broad parser target with the catalog's exact name and ID."""

    slots = {
        name: slot
        for name, slot in match.slots.items()
        if name not in _TARGET_SLOT_NAMES
    }
    slots.update(
        {
            "name": SlotValue(
                value=entity.name,
                text=entity.name,
                metadata={"entity_id": entity.entity_id},
            ),
            # The HA intent API resolves by name; retaining the catalog ID
            # lets the executor retry an exact service call if HA reports a
            # duplicate display name.
            "entity_id": SlotValue(value=entity.entity_id, text=entity.entity_id),
        }
    )
    return replace(match, slots=slots)


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
    *,
    text: str | None = None,
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

    surface_area_ids = (
        frozenset({area_id})
        if area_id
        else frozenset(area.area_id for area in catalog.areas if area.floor_id == floor_id)
        if floor_id
        else None
    )
    has_unresolved_explicit_topology = (has_explicit_area and not area_id) or (
        has_explicit_floor and not floor_id
    )
    if not named_entities and not has_explicit_name and domain and not has_unresolved_explicit_topology:
        normalized_text = _normal(text)
        if (
            not _has_plural_domain_target(normalized_text, domain)
            and (
                surface_entity := _surface_named_entity(
                    text,
                    catalog,
                    _surface_target_domains(match.intent_name, domain),
                    surface_area_ids,
                )
            )
        ):
            match = _with_surface_name(match, surface_entity)
            named_entities = [surface_entity]
            has_explicit_name = True

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
