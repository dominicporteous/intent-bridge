"""HassIL adapter that injects a live Home Assistant topology snapshot."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from hassil import TextSlotList, recognize_all

from intent_bridge.core.text import normalize_search_text
from intent_bridge.intent_engine.grammar import LoadedIntentGrammar
from intent_bridge.intent_engine.models import CatalogSnapshot, IntentMatch, SlotValue


def _deduplicated_aliases(*values: str) -> tuple[str, ...]:
    aliases: dict[str, str] = {}
    for value in values:
        if not isinstance(value, str) or not value.strip():
            continue
        cleaned = value.strip()
        normal = normalize_search_text(cleaned)
        if normal:
            aliases.setdefault(normal, cleaned)
    return tuple(aliases[key] for key in sorted(aliases))


def _area_context(catalog: CatalogSnapshot, area_id: str | None) -> dict[str, Any] | None:
    if not area_id:
        return None
    area = next((candidate for candidate in catalog.areas if candidate.area_id == area_id), None)
    if area is None:
        return None
    return {
        "value": area.name,
        "text": area.name,
        "metadata": {"area_id": area.area_id},
    }


def _runtime_slot_lists(catalog: CatalogSnapshot) -> dict[str, TextSlotList]:
    name_values: list[tuple[str, Any, dict[str, Any], dict[str, Any]]] = []
    for entity in catalog.entities:
        context: dict[str, Any] = {"domain": entity.domain}
        area_context = _area_context(catalog, entity.area_id)
        if area_context is not None:
            context["area"] = area_context
        if entity.device_class:
            context["device_class"] = entity.device_class
        metadata = {
            "entity_id": entity.entity_id,
            "area_id": entity.area_id,
            "domain": entity.domain,
        }
        for alias in _deduplicated_aliases(entity.name, *entity.aliases):
            name_values.append((alias, entity.name, context, metadata))

    area_values = [
        (
            alias,
            area.name,
            {},
            {"area_id": area.area_id, "floor_id": area.floor_id},
        )
        for area in catalog.areas
        for alias in _deduplicated_aliases(area.name, *area.aliases)
    ]
    floor_values = [
        (alias, floor.name, {}, {"floor_id": floor.floor_id})
        for floor in catalog.floors
        for alias in _deduplicated_aliases(floor.name, *floor.aliases)
    ]

    return {
        "name": TextSlotList.from_tuples(name_values, allow_template=False, name="name"),
        "area": TextSlotList.from_tuples(area_values, allow_template=False, name="area"),
        "floor": TextSlotList.from_tuples(floor_values, allow_template=False, name="floor"),
    }


def _intent_context(
    catalog: CatalogSnapshot,
    origin_context: Mapping[str, object] | None,
) -> dict[str, Any]:
    if not origin_context:
        return {}
    area_id = origin_context.get("area_id")
    if not isinstance(area_id, str) or not area_id:
        wanted = normalize_search_text(str(origin_context.get("area_name") or ""))
        matches = [
            area.area_id for area in catalog.areas if normalize_search_text(area.name) == wanted
        ]
        area_id = matches[0] if len(matches) == 1 else None
    area = _area_context(catalog, area_id)
    return {"area": area} if area is not None else {}


class HassilIntentRecognizer:
    def __init__(self, grammar: LoadedIntentGrammar, *, max_candidates: int = 128) -> None:
        self._grammar = grammar
        self._max_candidates = max(1, max_candidates)

    def recognize(
        self,
        text: str,
        catalog: CatalogSnapshot,
        origin_context: Mapping[str, object] | None = None,
    ) -> tuple[IntentMatch, ...]:
        matches: list[IntentMatch] = []
        seen: set[tuple[Any, ...]] = set()
        results = recognize_all(
            text,
            self._grammar.intents,
            slot_lists=_runtime_slot_lists(catalog),
            intent_context=_intent_context(catalog, origin_context),
            language=self._grammar.language,
        )
        for result in results:
            slots = {
                name: SlotValue(
                    value=entity.value,
                    text=entity.text,
                    metadata=dict(entity.metadata or {}),
                )
                for name, entity in result.entities.items()
            }
            identity = (
                result.intent.name,
                tuple(
                    sorted(
                        (
                            name,
                            repr(slot.value),
                            repr(sorted(slot.metadata.items())),
                        )
                        for name, slot in slots.items()
                    )
                ),
                result.response,
            )
            if identity in seen:
                continue
            seen.add(identity)
            matches.append(
                IntentMatch(
                    intent_name=result.intent.name,
                    slots=slots,
                    response_key=result.response,
                    context=dict(result.context),
                    metadata=dict(result.intent_metadata or {}),
                )
            )
            if len(matches) >= self._max_candidates:
                break
        return tuple(matches)


__all__ = ["HassilIntentRecognizer"]
