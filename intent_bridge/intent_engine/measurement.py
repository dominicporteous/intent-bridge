"""Capability-first deterministic planning for read-only measurements."""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence

from intent_bridge.core.text import normalize_search_text
from intent_bridge.core.voice import RouteDeclined
from intent_bridge.intent_engine.models import (
    CatalogArea,
    CatalogEntity,
    CatalogFloor,
    CatalogMeasurement,
    CatalogSnapshot,
    IntentPlan,
    OhfIntentCall,
    PlannedIntent,
    PlannedReading,
    SemanticEffect,
)

_QUERY_RE = re.compile(
    r"^(?:(?:can|could|would|will) you\s+)?(?:what(?:s| is| are)?|"
    r"how(?: much| high| low| hot| cold| warm| cool| humid)?|"
    r"tell me|give me|check|get|read|report|show me)\b|\b(?:current|reading|level)\b"
)
_MUTATION_RE = re.compile(r"\b(?:set|change|adjust|make|raise|lower|increase|decrease|turn)\b")
_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "at",
        "check",
        "current",
        "for",
        "get",
        "give",
        "high",
        "how",
        "in",
        "is",
        "level",
        "low",
        "me",
        "of",
        "please",
        "read",
        "reading",
        "report",
        "show",
        "tell",
        "the",
        "what",
        "whats",
    }
)
_GENERIC_IDENTITY_WORDS = frozenset({"level", "measurement", "reading", "sensor", "value"})
_QUANTITY_ALIASES: Mapping[str, tuple[str, ...]] = {
    "battery": ("battery level",),
    "humidity": ("humid",),
    "temperature": ("temp", "hot", "cold", "warm", "cool"),
}

LOGGER = logging.getLogger(__name__)


def _normal(value: object) -> str:
    return normalize_search_text(value).replace("_", " ")


def _contains_phrase(text: str, phrase: str) -> bool:
    return bool(phrase and re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text))


def _contains_topology_phrase(text: str, phrase: str) -> bool:
    if _contains_phrase(text, phrase):
        return True
    compact_phrase = phrase.replace(" ", "")
    return len(compact_phrase) >= 7 and compact_phrase in text.replace(" ", "")


def _labels(entity: CatalogEntity) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            _normal(value)
            for value in (
                entity.name,
                entity.entity_id,
                entity.entity_id.split(".", 1)[-1],
                *entity.aliases,
            )
            if _normal(value)
        )
    )


def _area_labels(area: CatalogArea) -> tuple[str, ...]:
    return tuple(_normal(value) for value in (area.name, area.area_id, *area.aliases))


def _floor_labels(floor: CatalogFloor) -> tuple[str, ...]:
    return tuple(_normal(value) for value in (floor.name, floor.floor_id, *floor.aliases))


def _mentioned_areas(text: str, catalog: CatalogSnapshot) -> tuple[CatalogArea, ...]:
    return tuple(
        area
        for area in catalog.areas
        if any(_contains_topology_phrase(text, label) for label in _area_labels(area))
    )


def _mentioned_floors(text: str, catalog: CatalogSnapshot) -> tuple[CatalogFloor, ...]:
    return tuple(
        floor
        for floor in catalog.floors
        if any(_contains_topology_phrase(text, label) for label in _floor_labels(floor))
    )


def _origin_area_id(
    context: Mapping[str, object] | None,
    catalog: CatalogSnapshot,
) -> str | None:
    if not context:
        return None
    area_id = context.get("area_id")
    if isinstance(area_id, str) and any(area.area_id == area_id for area in catalog.areas):
        return area_id
    area_name = _normal(context.get("area_name"))
    matches = [
        area.area_id
        for area in catalog.areas
        if area_name
        and any(
            area_name == label or area_name.replace(" ", "") == label.replace(" ", "")
            for label in _area_labels(area)
        )
    ]
    return matches[0] if len(matches) == 1 else None


def _quantity_phrases(quantity: str) -> tuple[str, ...]:
    canonical = _normal(quantity)
    phrases = [canonical, *_QUANTITY_ALIASES.get(quantity, ())]
    if canonical.endswith("s"):
        phrases.append(canonical[:-1])
    return tuple(dict.fromkeys(phrases))


def _requested_quantities(text: str, catalog: CatalogSnapshot) -> tuple[str, ...]:
    quantities = sorted(
        {
            measurement.quantity
            for entity in catalog.selectable_entities
            for measurement in entity.measurements
        }
    )
    return tuple(
        quantity
        for quantity in quantities
        if any(_contains_phrase(text, phrase) for phrase in _quantity_phrases(quantity))
    )


def _specific_request_tokens(
    text: str,
    quantity: str,
    areas: Sequence[CatalogArea],
    floors: Sequence[CatalogFloor],
) -> frozenset[str]:
    ignored = set(_STOP_WORDS)
    for phrase in _quantity_phrases(quantity):
        ignored.update(phrase.split())
    for area in areas:
        for label in _area_labels(area):
            ignored.update(label.split())
            ignored.add(label.replace(" ", ""))
    for floor in floors:
        for label in _floor_labels(floor):
            ignored.update(label.split())
            ignored.add(label.replace(" ", ""))
    return frozenset(token for token in text.split() if token not in ignored)


def _identity_labels(entity: CatalogEntity, quantity: str) -> tuple[str, ...]:
    quantity_words = {word for phrase in _quantity_phrases(quantity) for word in phrase.split()}
    generic_words = quantity_words | _GENERIC_IDENTITY_WORDS
    return tuple(label for label in _labels(entity) if set(label.split()) - generic_words)


def _candidate_score(
    text: str,
    quantity: str,
    entity: CatalogEntity,
    measurement: CatalogMeasurement,
    specific_tokens: frozenset[str],
) -> tuple[int, tuple[str, ...]]:
    labels = _labels(entity)
    identity_labels = _identity_labels(entity, quantity)
    label_tokens = {token for label in labels for token in label.split()}
    evidence: list[str] = []
    specific_matches = specific_tokens & label_tokens
    score = 100 * len(specific_matches)
    if specific_matches:
        evidence.append(f"specific_tokens={','.join(sorted(specific_matches))}:+{score}")
    if any(_contains_phrase(text, label) for label in identity_labels):
        score += 250
        evidence.append("exact_label:+250")
    if any(
        _contains_phrase(label, phrase)
        for label in labels
        for phrase in _quantity_phrases(quantity)
    ):
        evidence.append("quantity_in_label:required")
    if measurement.source == "state":
        score += 20
        evidence.append("direct_state:+20")
    elif measurement.source == "inferred_state":
        score -= 30
        evidence.append("inferred_state:-30")
    if entity.entity_category is not None:
        score -= 40
        evidence.append(f"secondary_category={entity.entity_category}:-40")
    return score, tuple(evidence)


def _coordinated_target_members(
    text: str,
    catalog: CatalogSnapshot,
) -> tuple[str, ...]:
    """Return sensor/area conjuncts when each side has independent target evidence."""

    parts = tuple(
        part.strip(" ,")
        for part in re.split(r"\s*(?:,\s*(?:and\s+)?|\b(?:and|plus|as well as)\b)\s*", text)
        if part.strip(" ,")
    )
    if len(parts) < 2:
        return ()

    def has_evidence(part: str) -> bool:
        if _mentioned_areas(part, catalog) or _mentioned_floors(part, catalog):
            return True
        return any(
            _contains_phrase(part, label)
            for entity in catalog.selectable_entities
            for label in _labels(entity)
            if set(label.split()) - _GENERIC_IDENTITY_WORDS
        )

    return parts if all(has_evidence(part) for part in parts) else ()


class MeasurementIntentPlanner:
    """Resolve measurement queries by quantity and topology, independent of domain."""

    def __init__(
        self,
        *,
        ambiguity_response: str = (
            "I found more than one possible target. Please be more specific."
        ),
    ) -> None:
        self._ambiguity_response = ambiguity_response

    def plan(
        self,
        text: str,
        catalog: CatalogSnapshot,
        origin_context: Mapping[str, object] | None = None,
    ) -> IntentPlan:
        normal = _normal(text)
        if not normal or _MUTATION_RE.search(normal) or not _QUERY_RE.search(normal):
            raise RouteDeclined("Not a read-only measurement query")

        quantities = _requested_quantities(normal, catalog)
        if not quantities:
            raise RouteDeclined("No catalog measurement quantity was requested")

        areas = _mentioned_areas(normal, catalog)
        floors = _mentioned_floors(normal, catalog)
        area_ids = {area.area_id for area in areas}
        if not area_ids and floors:
            floor_ids = {floor.floor_id for floor in floors}
            area_ids = {area.area_id for area in catalog.areas if area.floor_id in floor_ids}
        if not area_ids and not areas and not floors:
            if origin_area_id := _origin_area_id(origin_context, catalog):
                area_ids = {origin_area_id}

        LOGGER.info(
            "MEASUREMENT RESOLUTION start text=%r quantities=%s areas=%s floors=%s "
            "effective_area_ids=%s catalog_entities=%d",
            text,
            quantities,
            tuple(area.area_id for area in areas),
            tuple(floor.floor_id for floor in floors),
            tuple(sorted(area_ids)),
            len(catalog.entities),
        )

        steps: list[PlannedIntent] = []
        for quantity in quantities:
            coordinated_members = _coordinated_target_members(normal, catalog)
            if coordinated_members:
                selected: list[tuple[CatalogEntity, CatalogMeasurement]] = []
                for member in coordinated_members:
                    member_areas = _mentioned_areas(member, catalog)
                    member_floors = _mentioned_floors(member, catalog)
                    if not member_areas and len(areas) == 1:
                        member_areas = areas
                    if not member_floors and not member_areas and len(floors) == 1:
                        member_floors = floors
                    member_area_ids = {area.area_id for area in member_areas}
                    if member_floors:
                        member_floor_ids = {floor.floor_id for floor in member_floors}
                        member_area_ids.update(
                            area.area_id
                            for area in catalog.areas
                            if area.floor_id in member_floor_ids
                        )
                    member_candidates = [
                        (entity, measurement)
                        for entity in catalog.selectable_entities
                        if not member_area_ids or entity.area_id in member_area_ids
                        for measurement in entity.measurements
                        if measurement.quantity == quantity
                    ]
                    if not member_candidates:
                        raise RouteDeclined(
                            f"No {quantity} reading matched coordinated target {member!r}"
                        )
                    member_tokens = _specific_request_tokens(
                        member,
                        quantity,
                        member_areas,
                        member_floors,
                    )
                    scored = [
                        (
                            _candidate_score(
                                member,
                                quantity,
                                entity,
                                measurement,
                                member_tokens,
                            )[0],
                            entity,
                            measurement,
                        )
                        for entity, measurement in member_candidates
                    ]
                    highest = max(score for score, _, _ in scored)
                    winners = [
                        (entity, measurement)
                        for score, entity, measurement in scored
                        if score == highest
                    ]
                    if len(winners) != 1:
                        return IntentPlan(response=self._ambiguity_response)
                    selected.extend(winners)

                seen_entities: set[str] = set()
                for entity, measurement in selected:
                    if entity.entity_id in seen_entities:
                        continue
                    seen_entities.add(entity.entity_id)
                    steps.append(
                        PlannedIntent(
                            call=OhfIntentCall(
                                "HassGetMeasurement",
                                {"entity_id": entity.entity_id, "quantity": quantity},
                            ),
                            entity_ids=(entity.entity_id,),
                            reading=PlannedReading(entity.entity_id, entity.name, measurement),
                            effect=SemanticEffect("query", quantity, "read", measurement.value),
                        )
                    )
                continue

            candidates = [
                (entity, measurement)
                for entity in catalog.selectable_entities
                if not area_ids or entity.area_id in area_ids
                for measurement in entity.measurements
                if measurement.quantity == quantity
            ]
            if not candidates:
                LOGGER.info(
                    "MEASUREMENT RESOLUTION no_candidates quantity=%s area_ids=%s",
                    quantity,
                    tuple(sorted(area_ids)),
                )
                raise RouteDeclined(f"No {quantity} reading matched the requested topology")

            specific_tokens = _specific_request_tokens(
                normal,
                quantity,
                areas,
                floors,
            )
            has_request_scope = bool(area_ids or areas or floors or specific_tokens)
            if len(candidates) > 1 and not has_request_scope:
                LOGGER.info(
                    "MEASUREMENT RESOLUTION declined_unscoped quantity=%s "
                    "candidate_count=%d; allowing fallback",
                    quantity,
                    len(candidates),
                )
                raise RouteDeclined(
                    f"Multiple {quantity} readings exist without target or topology evidence"
                )
            scored = []
            for entity, measurement in candidates:
                score, evidence = _candidate_score(
                    normal,
                    quantity,
                    entity,
                    measurement,
                    specific_tokens,
                )
                scored.append((score, entity, measurement, evidence))
                LOGGER.info(
                    "MEASUREMENT RESOLUTION candidate quantity=%s entity=%s name=%r "
                    "area_id=%r category=%r source=%s value=%r unit=%r score=%d evidence=%s",
                    quantity,
                    entity.entity_id,
                    entity.name,
                    entity.area_id,
                    entity.entity_category,
                    measurement.source,
                    measurement.value,
                    measurement.unit,
                    score,
                    evidence,
                )
            highest = max(score for score, _, _, _ in scored)
            winners = [
                (entity, measurement)
                for score, entity, measurement, _ in scored
                if score == highest
            ]
            if len(winners) != 1:
                LOGGER.info(
                    "MEASUREMENT RESOLUTION ambiguous quantity=%s top_score=%d winners=%s",
                    quantity,
                    highest,
                    tuple(entity.entity_id for entity, _ in winners),
                )
                return IntentPlan(response=self._ambiguity_response)

            entity, measurement = winners[0]
            LOGGER.info(
                "MEASUREMENT RESOLUTION selected quantity=%s entity=%s source=%s "
                "value=%r unit=%r score=%d",
                quantity,
                entity.entity_id,
                measurement.source,
                measurement.value,
                measurement.unit,
                highest,
            )
            steps.append(
                PlannedIntent(
                    call=OhfIntentCall(
                        "HassGetMeasurement",
                        {"entity_id": entity.entity_id, "quantity": quantity},
                    ),
                    entity_ids=(entity.entity_id,),
                    reading=PlannedReading(entity.entity_id, entity.name, measurement),
                    effect=SemanticEffect("query", quantity, "read", measurement.value),
                )
            )

        return IntentPlan(steps=tuple(steps))


__all__ = ["MeasurementIntentPlanner"]
