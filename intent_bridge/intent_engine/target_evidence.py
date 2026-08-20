"""Catalog-backed target evidence shared by deterministic planners."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from intent_bridge.core.text import normalize_search_text
from intent_bridge.intent_engine.models import CatalogArea, CatalogEntity, CatalogSnapshot

_LABEL_TARGET_INTENTS = frozenset({"HassGetState", "HassTurnOn", "HassTurnOff"})
_POWER_INTENTS = frozenset({"HassTurnOn", "HassTurnOff"})
_NON_PRIMARY_ENTITY_CATEGORIES = frozenset({"config", "diagnostic"})


def _normal(value: object) -> str:
    return normalize_search_text(value)


def _contains_phrase(text: str, phrase: str) -> bool:
    return bool(phrase and re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text))


def _area_words(area: CatalogArea) -> frozenset[str]:
    labels = (area.name, area.area_id, *area.aliases)
    words = {word for label in labels for word in _normal(label).split()}
    name = _normal(area.name)
    if name:
        words.add(name.replace(" ", ""))
    if name.endswith(" room"):
        words.update(name.removesuffix(" room").split())
    elif name:
        words.update((name + " room").split())
    return frozenset(words)


def _relative_labels(
    entity: CatalogEntity,
    catalog: CatalogSnapshot,
    labels: Iterable[str],
) -> frozenset[str]:
    area = next((item for item in catalog.areas if item.area_id == entity.area_id), None)
    area_words = _area_words(area) if area is not None else frozenset()
    return frozenset(
        " ".join(word for word in _normal(label).split() if word not in area_words)
        for label in labels
        if label
    ) - {""}


def _entity_primary_relative_labels(
    entity: CatalogEntity,
    catalog: CatalogSnapshot,
) -> frozenset[str]:
    return _relative_labels(entity, catalog, (entity.name, entity.entity_id.split(".", 1)[-1]))


def _entity_alias_relative_labels(
    entity: CatalogEntity,
    catalog: CatalogSnapshot,
) -> frozenset[str]:
    return _relative_labels(entity, catalog, entity.aliases)


def _surface_labels(
    entity: CatalogEntity,
    catalog: CatalogSnapshot,
    labels: Iterable[str],
) -> frozenset[str]:
    area = next((item for item in catalog.areas if item.area_id == entity.area_id), None)
    area_labels = (
        {_normal(label) for label in (area.name, area.area_id, *area.aliases) if _normal(label)}
        if area is not None
        else set()
    )
    return frozenset(
        _normal(label)
        for label in labels
        if _normal(label) and _normal(label) not in area_labels
    )


def _entity_primary_surface_labels(
    entity: CatalogEntity,
    catalog: CatalogSnapshot,
) -> frozenset[str]:
    return _surface_labels(entity, catalog, (entity.name, entity.entity_id.split(".", 1)[-1]))


def _entity_alias_surface_labels(
    entity: CatalogEntity,
    catalog: CatalogSnapshot,
) -> frozenset[str]:
    return _surface_labels(entity, catalog, entity.aliases)


def _is_intent_capable(entity: CatalogEntity, intent_name: str) -> bool:
    """Keep implicit power labels on actionable, primary controls only."""

    if intent_name not in _POWER_INTENTS:
        return True
    if entity.entity_category and entity.entity_category.casefold() in _NON_PRIMARY_ENTITY_CATEGORIES:
        return False
    return entity.supported_intents is None or intent_name in entity.supported_intents


def _eligible_entities(
    catalog: CatalogSnapshot,
    *,
    intent_name: str,
    area_ids: frozenset[str],
    ignored_domains: frozenset[str],
) -> tuple[CatalogEntity, ...]:
    return tuple(
        entity
        for entity in catalog.selectable_entities
        if entity.domain.casefold() not in ignored_domains
        and (not area_ids or entity.area_id in area_ids)
        and _is_intent_capable(entity, intent_name)
    )


@dataclass(frozen=True, slots=True)
class CatalogTargetEvidence:
    """The catalog entities supported by one piece of target evidence."""

    entities: tuple[CatalogEntity, ...] = ()
    source: str = "none"
    label: str = ""

    @property
    def candidate_entity_ids(self) -> tuple[str, ...]:
        return tuple(entity.entity_id for entity in self.entities)

    @property
    def is_unique(self) -> bool:
        return len(self.entities) == 1

    @property
    def is_ambiguous(self) -> bool:
        return len(self.entities) > 1


def _label_phrase_evidence(
    text: str,
    candidates: Iterable[CatalogEntity],
    catalog: CatalogSnapshot,
    label_for_entity: Callable[[CatalogEntity, CatalogSnapshot], frozenset[str]],
    source: str,
    *,
    ignore_plural_labels: bool = False,
    reject_distinct_winner_labels: bool = False,
) -> CatalogTargetEvidence:
    """Match catalog labels *within* a command without parsing its residue.

    The command is action language wrapped around a target, so removing a
    maintained list of verbs, articles, and punctuation is both incomplete and
    brittle.  Instead, labels from the scoped catalog are the evidence: select
    the longest whole label phrase that appears in the normalized utterance.
    """

    matches: list[tuple[int, str, CatalogEntity]] = []
    for entity in candidates:
        for label in label_for_entity(entity, catalog):
            if ignore_plural_labels and label.endswith("s"):
                continue
            if _contains_phrase(text, label):
                matches.append((len(label.split()), label, entity))
    if not matches:
        return CatalogTargetEvidence()
    longest = max(length for length, _, _ in matches)
    winner_ids: set[str] = set()
    winners: list[CatalogEntity] = []
    for length, _, entity in matches:
        if length == longest and entity.entity_id not in winner_ids:
            winner_ids.add(entity.entity_id)
            winners.append(entity)
    matched_labels = sorted(
        {label for length, label, _ in matches if length == longest}
    )
    if reject_distinct_winner_labels and len(matched_labels) > 1:
        # Several labels in one utterance are a coordinated request, not an
        # ambiguity over one target. Let the compound planner split it.
        return CatalogTargetEvidence()
    return CatalogTargetEvidence(tuple(winners), source, matched_labels[0])


def _most_specific_label_evidence(
    primary: CatalogTargetEvidence,
    aliases: CatalogTargetEvidence,
) -> CatalogTargetEvidence:
    """Choose the longer catalog label; primary labels only break ties."""

    if not primary.entities:
        return aliases
    if not aliases.entities:
        return primary
    primary_length = len(primary.label.split())
    alias_length = len(aliases.label.split())
    if primary_length > alias_length:
        return primary
    if alias_length > primary_length:
        return aliases
    if primary.label != aliases.label:
        # The same utterance has two equally specific but different labels.
        # Treat it as coordination and allow the compound resolver to split it.
        return CatalogTargetEvidence()
    return primary


def power_target_evidence(
    text: str | None,
    catalog: CatalogSnapshot,
    *,
    intent_name: str,
    area_ids: Iterable[str] = (),
    ignored_entity_domains: Iterable[str] = (),
) -> CatalogTargetEvidence:
    """Find a singular target from catalog labels, independent of domain.

    A term such as ``lamp`` is evidence about a target label, not proof that
    the backing implementation is a ``light`` entity. Entity area prefixes are
    removed before comparison, so ``Bedroom Lamp`` and ``switch.bedroom_lamp``
    both expose the label ``lamp``. The label is matched inside the utterance;
    action words and punctuation are not treated as part of it.
    """

    if intent_name not in _LABEL_TARGET_INTENTS:
        return CatalogTargetEvidence()
    normal = _normal(text)
    if not normal:
        return CatalogTargetEvidence()
    selected_area_ids = frozenset(area_ids)
    ignored = frozenset(domain.casefold() for domain in ignored_entity_domains)
    candidates = _eligible_entities(
        catalog,
        intent_name=intent_name,
        area_ids=selected_area_ids,
        ignored_domains=ignored,
    )
    primary = _label_phrase_evidence(
        normal,
        candidates,
        catalog,
        _entity_primary_relative_labels,
        "primary_relative_label",
        ignore_plural_labels=True,
        reject_distinct_winner_labels=True,
    )
    aliases = _label_phrase_evidence(
        normal,
        candidates,
        catalog,
        _entity_alias_relative_labels,
        "alias_relative_label",
        ignore_plural_labels=True,
        reject_distinct_winner_labels=True,
    )
    return _most_specific_label_evidence(primary, aliases)


def surface_target_evidence(
    text: str | None,
    catalog: CatalogSnapshot,
    *,
    intent_name: str,
    area_ids: Iterable[str] = (),
    ignored_entity_domains: Iterable[str] = (),
) -> CatalogTargetEvidence:
    """Return exact surface-name evidence, then singular relative-label evidence."""

    normal = _normal(text)
    if not normal:
        return CatalogTargetEvidence()
    selected_area_ids = frozenset(area_ids)
    ignored = frozenset(domain.casefold() for domain in ignored_entity_domains)
    candidates = _eligible_entities(
        catalog,
        intent_name=intent_name,
        area_ids=selected_area_ids,
        ignored_domains=ignored,
    )
    for source, label_for_entity in (
        ("primary_surface_name", _entity_primary_surface_labels),
        ("alias_surface_name", _entity_alias_surface_labels),
    ):
        evidence = _label_phrase_evidence(
            normal,
            candidates,
            catalog,
            label_for_entity,
            source,
        )
        if evidence.entities:
            return evidence
    return power_target_evidence(
        text,
        catalog,
        intent_name=intent_name,
        area_ids=selected_area_ids,
        ignored_entity_domains=ignored,
    )


__all__ = ["CatalogTargetEvidence", "power_target_evidence", "surface_target_evidence"]
