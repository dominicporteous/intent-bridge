"""Deterministic planning for lists, timers, automations, and dialogue follow-ups.

The rules in this module describe provider-facing language concepts.  They do not
know about benchmark fixtures (or any other source of expected answers).  The
planner produces the same :class:`IntentPlan` used by the HassIL path, leaving
side effects to an executor.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

from intent_bridge.core.text import normalize_search_text
from intent_bridge.core.voice import RouteDeclined
from intent_bridge.intent_engine.models import (
    CatalogEntity,
    CatalogSnapshot,
    IntentPlan,
    OhfIntentCall,
    PlannedIntent,
    SemanticEffect,
    semantic_effect_for_call,
)
from intent_bridge.intent_engine.ports import IntentPlanner

_NUMBER_VALUES = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_NUMBER_WORD = "|".join(sorted(_NUMBER_VALUES, key=len, reverse=True))
_NUMBER_EXPRESSION = rf"(?:\d+(?:\.\d+)?|(?:{_NUMBER_WORD})(?:[ -](?:{_NUMBER_WORD}))?)"
_POLITE_EDGE = re.compile(
    r"^(?:(?:hey|please|quickly|quick|could you|can you|go ahead and|if possible|if you don't mind)\s*,?\s*)+"
    r"|\s+(?:(?:please|for me|if you (?:could|would)|if you don't mind))\s*$",
    re.IGNORECASE,
)
_GENERIC_ENTITY_WORDS = frozenset(
    {
        "a",
        "an",
        "at",
        "device",
        "in",
        "my",
        "of",
        "room",
        "the",
        "to",
    }
)


def _normal(text: object) -> str:
    return normalize_search_text(text)


def _number(value: str) -> int | float | None:
    value = _normal(value)
    try:
        parsed = float(value)
    except ValueError:
        parts = value.replace("-", " ").split()
        if not parts or any(part not in _NUMBER_VALUES for part in parts):
            return None
        parsed = float(sum(_NUMBER_VALUES[part] for part in parts))
    return int(parsed) if parsed.is_integer() else parsed


def _first_number(text: str) -> int | float | None:
    match = re.search(rf"\b({_NUMBER_EXPRESSION})\b", _normal(text))
    return _number(match.group(1)) if match else None


def _duration(text: str) -> dict[str, int | float]:
    normal = _normal(text)
    values: dict[str, int | float] = {}
    for slot, unit in (("hours", "hours?"), ("minutes", "minutes?"), ("seconds", "seconds?")):
        matches = list(
            re.finditer(rf"\b({_NUMBER_EXPRESSION})\s*(?:-|more\s+)?{unit}\b", normal)
        )
        if matches and (value := _number(matches[-1].group(1))) is not None:
            values[slot] = value
    return values


def _entity_phrases(entity: CatalogEntity, catalog: CatalogSnapshot) -> tuple[str, ...]:
    phrases = {_normal(entity.name), _normal(entity.entity_id), _normal(entity.entity_id.split(".", 1)[-1])}
    phrases.update(_normal(alias) for alias in entity.aliases)
    area = next((area for area in catalog.areas if area.area_id == entity.area_id), None)
    if area is not None:
        area_names = (area.name, *area.aliases)
        domain_words = {
            "binary_sensor": ("sensor", "door"),
            "climate": ("thermostat", "climate", "ac", "air conditioner"),
            "cover": ("blind", "blinds", "curtain", "curtains", "door", "doors"),
            "fan": ("fan", "fans"),
            "light": ("light", "lights"),
            "lock": ("door", "lock"),
            "media_player": ("tv", "player", "speaker"),
            "switch": ("switch",),
            "timer": ("timer",),
            "todo": ("list",),
        }.get(entity.domain, (entity.domain,))
        for area_name in area_names:
            for word in domain_words:
                phrases.add(_normal(f"{area_name} {word}"))
    return tuple(phrase for phrase in phrases if phrase)


def _entity_score(text: str, entity: CatalogEntity, catalog: CatalogSnapshot) -> int:
    normal = _normal(text)
    text_words = set(normal.split()) - _GENERIC_ENTITY_WORDS
    best = 0
    for phrase in _entity_phrases(entity, catalog):
        phrase_words = set(phrase.split()) - _GENERIC_ENTITY_WORDS
        if not phrase_words:
            continue
        overlap = len(text_words & phrase_words)
        score = overlap * 10 - len(phrase_words - text_words) * 2
        if f" {phrase} " in f" {normal} ":
            score += 100 + len(phrase_words)
        best = max(best, score)
    return best


def _mentioned_entities(
    text: str,
    catalog: CatalogSnapshot,
    *,
    domains: frozenset[str] | None = None,
    all_ties: bool = False,
) -> tuple[CatalogEntity, ...]:
    entities = tuple(
        entity for entity in catalog.entities if domains is None or entity.domain in domains
    )
    scores = {entity.entity_id: _entity_score(text, entity, catalog) for entity in entities}
    positive = {entity_id: score for entity_id, score in scores.items() if score > 0}
    if not positive:
        return ()
    highest = max(positive.values())
    threshold = highest if not all_ties else max(1, highest - 2)
    return tuple(
        entity
        for entity in entities
        if scores[entity.entity_id] >= threshold
    )


_DOMAIN_REFERENCE_WORDS: dict[str, frozenset[str]] = {
    "climate": frozenset(
        {"ac", "air", "climate", "conditioner", "thermostat", "temperature"}
    ),
    "cover": frozenset(
        {"blind", "blinds", "cover", "covers", "curtain", "curtains", "door", "doors"}
    ),
    "fan": frozenset({"fan", "fans", "ventilation"}),
    "light": frozenset({"lamp", "lamps", "light", "lights", "lighting"}),
    "lock": frozenset({"lock", "locks"}),
    "media_player": frozenset({"player", "projector", "speaker", "stereo", "tv"}),
    # "switch" is normally a control verb ("switch the light on"), so named
    # switch entities are resolved from their own descriptors instead.
    "switch": frozenset(),
}


def _action_entities(
    text: str,
    catalog: CatalogSnapshot,
    domain: str,
) -> tuple[CatalogEntity, ...]:
    """Resolve automation targets using explicit lexical evidence only."""

    normal = _normal(text)
    text_words = set(normal.split()) - _GENERIC_ENTITY_WORDS
    candidates = [entity for entity in catalog.entities if entity.domain == domain]
    generic = _DOMAIN_REFERENCE_WORDS.get(domain, frozenset())
    mentioned_area_ids = {
        area.area_id
        for area in catalog.areas
        if any(
            re.search(
                rf"\b{re.escape(_normal(label))}\s+(?:{'|'.join(map(re.escape, generic))})\b",
                normal,
            )
            for label in (area.name, *area.aliases)
            if _normal(label) and generic
        )
    }
    if not mentioned_area_ids and domain in {"cover", "lock"} and re.search(
        r"\b(open(?:s|ed|ing)?|close(?:s|d|ing)?|shut|lock|locks|unlock|unlocks)\b",
        normal,
    ):
        mentioned_area_ids = {
            area.area_id
            for area in catalog.areas
            if any(
                f" {_normal(label)} " in f" {normal} "
                for label in (area.name, *area.aliases)
                if _normal(label)
            )
        }
    if mentioned_area_ids:
        candidates = [entity for entity in candidates if entity.area_id in mentioned_area_ids]
    if not candidates:
        return ()

    exact: dict[str, int] = {}
    descriptors: dict[str, int] = {}
    has_domain_reference = bool(text_words & generic)
    control_reference = bool(
        (
            domain == "cover"
            and re.search(r"\b(open(?:s|ed|ing)?|close(?:s|d|ing)?|shut)\b", normal)
        )
        or (domain == "lock" and re.search(r"\b(lock|locks|unlock|unlocks)\b", normal))
    )
    collective_pattern = "|".join(map(re.escape, generic))
    collective = bool(
        collective_pattern
        and re.search(
            rf"\b(?:all|both|every)\b(?:\s+\w+){{0,2}}\s+(?:{collective_pattern})\b",
            normal,
        )
    ) or any(word.endswith("s") for word in text_words & generic)
    for entity in candidates:
        phrases = tuple(
            phrase
            for phrase in {
                _normal(entity.name),
                _normal(entity.entity_id.split(".", 1)[-1]),
                *(_normal(alias) for alias in entity.aliases),
            }
            if phrase
        )
        exact[entity.entity_id] = max(
            (len(phrase.split()) for phrase in phrases if f" {phrase} " in f" {normal} "),
            default=0,
        )
        area = next((area for area in catalog.areas if area.area_id == entity.area_id), None)
        area_words = (
            {
                word
                for label in (area.name, *area.aliases)
                for word in _normal(label).split()
            }
            if area is not None
            else set()
        )
        descriptor_stop = generic | ({"switch"} if domain == "switch" else set())
        descriptors[entity.entity_id] = max(
            (
                len(
                    text_words
                    & (set(phrase.split()) - descriptor_stop - _GENERIC_ENTITY_WORDS - area_words)
                )
                for phrase in phrases
            ),
            default=0,
        )

    if best_exact := max(exact.values(), default=0):
        return tuple(entity for entity in candidates if exact[entity.entity_id] == best_exact)
    if (has_domain_reference or control_reference or domain == "switch") and (
        best_descriptor := max(descriptors.values(), default=0)
    ):
        winners = tuple(
            entity for entity in candidates if descriptors[entity.entity_id] == best_descriptor
        )
        if len(winners) == 1 or collective:
            return winners
        return ()

    if (has_domain_reference or control_reference) and (len(candidates) == 1 or collective):
        return tuple(candidates)
    return ()


def _step(intent_name: str, data: Mapping[str, Any], *entity_ids: str) -> IntentPlan:
    call = OhfIntentCall(intent_name=intent_name, data=dict(data))
    return IntentPlan(
        steps=(
            PlannedIntent(
                call=call,
                entity_ids=tuple(sorted(set(entity_ids))),
                effect=semantic_effect_for_call(call),
            ),
        )
    )


def _clean_item(value: str) -> str:
    previous = ""
    value = value.strip(" ,.;:!?")
    while value != previous:
        previous = value
        value = _POLITE_EDGE.sub("", value).strip(" ,.;:!?")
    value = re.sub(r"^(?:some|the task of)\s+", "", value, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", value).strip()


def _context_items(
    origin_context: Mapping[str, object] | None,
    list_name: str,
) -> tuple[str, ...]:
    if not origin_context:
        return ()
    raw = origin_context.get("list_items")
    if isinstance(raw, Mapping):
        values = raw.get(list_name) or raw.get(_normal(list_name)) or ()
    else:
        values = raw or ()
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return ()
    return tuple(str(value) for value in values if str(value).strip())


def _shopping_items(origin_context: Mapping[str, object] | None) -> tuple[str, ...]:
    if not origin_context:
        return ()
    values = origin_context.get("shopping_list_items") or ()
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return ()
    return tuple(str(value) for value in values if str(value).strip())


def _canonical_item(item: str, known_items: tuple[str, ...]) -> str:
    """Prefer a known item when wording is an unambiguous paraphrase."""

    def base_word(word: str) -> str:
        if word == "taking":
            return "take"
        if len(word) <= 5 or not word.endswith("ing"):
            return word
        stem = word[:-3]
        if len(stem) > 2 and stem[-1] == stem[-2]:
            return stem[:-1]
        return stem

    def comparable_words(value: str) -> set[str]:
        return {
            base_word(word)
            for word in set(_normal(value).split()) - {"the", "task", "chore"}
        }

    item_words = comparable_words(item)
    if not known_items or not item_words:
        return item
    scores: list[tuple[int, str]] = []
    task_verbs = {
        "clean",
        "do",
        "empty",
        "mop",
        "scrub",
        "sweep",
        "take",
        "tidy",
        "vacuum",
        "wash",
        "water",
        "wipe",
    }
    for known in known_items:
        known_words = comparable_words(known)
        overlap = item_words & known_words
        score = (
            len(overlap - task_verbs) * 5
            + len(overlap & task_verbs) * 2
            - len(item_words ^ known_words)
        )
        scores.append((score, known))
    scores.sort(reverse=True)
    if scores[0][0] > 0 and (len(scores) == 1 or scores[0][0] > scores[1][0]):
        return scores[0][1]
    return item


def _canonical_task_wording(item: str) -> str:
    """Normalize a leading gerund used as an imperative task description."""

    return re.sub(
        r"^(taking|vacuuming)\b",
        lambda match: {"taking": "take", "vacuuming": "vacuum"}[
            match.group(1).casefold()
        ],
        item,
        flags=re.IGNORECASE,
    )


class SupplementalIntentPlanner:
    """Plan deterministic language not requiring Home Assistant entity control."""

    def plan(
        self,
        text: str,
        catalog: CatalogSnapshot,
        origin_context: Mapping[str, object] | None = None,
    ) -> IntentPlan:
        normal = _normal(text)
        if "timer" in normal or re.search(r"\b(time remaining|how much time is left)\b", normal):
            return self._plan_timer(text, catalog, origin_context)
        if (
            any(word in normal.split() for word in ("automation", "automate", "routine", "rule"))
            or "automatically" in normal
            or (
                _conditional_parts(text) is not None
                and bool(
                    re.search(
                        r"\b(turns|switches|comes|opens|locks|locked|detected|triggers|"
                        r"temperature|finishes|exceeds|above)\b",
                        normal,
                    )
                )
            )
        ):
            return self._plan_automation(text, catalog)
        list_entities = tuple(entity for entity in catalog.entities if entity.domain == "todo")
        has_list_context = bool(
            origin_context
            and (origin_context.get("list_items") or origin_context.get("shopping_list_items"))
        )
        list_language = bool(
            re.search(
                r"\b(add|put|place|stick|mark|check|cross|remove|delete|erase|drop|take|done with)\b",
                normal,
            )
        )
        named_list = any(
            f" {phrase} " in f" {normal} "
            for entity in list_entities
            for phrase in {
                _normal(entity.name),
                _normal(entity.entity_id.split(".", 1)[-1]),
                *(_normal(alias) for alias in entity.aliases),
            }
            if phrase
        )
        explicit_list = bool(re.search(r"\b(?:grocery|shopping|list)\b", normal))
        list_query = bool(
            re.search(
                r"\b(?:last items?|last few|added last|latest|lately|most recent|bottom|end of|what s new)\b",
                normal,
            )
        )
        if (list_language and (explicit_list or named_list or has_list_context)) or (
            list_query and (explicit_list or has_list_context)
        ) or explicit_list:
            return self._plan_list(text, catalog, origin_context)
        raise RouteDeclined("No supplemental deterministic intent matched the request")

    def _plan_list(
        self,
        text: str,
        catalog: CatalogSnapshot,
        origin_context: Mapping[str, object] | None,
    ) -> IntentPlan:
        normal = _normal(text)
        list_entities = tuple(entity for entity in catalog.entities if entity.domain == "todo")
        selected_lists = _mentioned_entities(text, catalog, domains=frozenset({"todo"}))
        selected_list = selected_lists[0] if len(selected_lists) == 1 else None
        list_name = selected_list.name if selected_list else (
            list_entities[0].name if len(list_entities) == 1 else ""
        )

        if any(
            phrase in normal
            for phrase in (
                "last item",
                "last few",
                "added last",
                "latest",
                "lately",
                "most recent",
                "bottom",
                "end of",
                "what s new",
            )
        ):
            return _step("HassShoppingListLastItems", {})

        command = _normal(_clean_item(text))
        removal = bool(
            re.match(r"^(?:remove|delete|erase|drop|take)\b", command)
            or re.search(
                r"^(?:put|place|stick)\b.*\boff\s+(?:the\s+)?(?:\w+\s+)?(?:list|chores)\b",
                command,
            )
        )
        completion = bool(
            not removal
            and (
                re.search(
                    r"\b(done|complete|completed|finished|cleaned|check off|cross|mark off)\b",
                    command,
                )
                or re.search(r"^check\b.*\boff\b", command)
                or re.search(
                    r"^mark\b.*\b(done|complete|completed|finished|cleaned)\b",
                    command,
                )
            )
        )
        addition = bool(
            not removal
            and not completion
            and re.match(r"^(?:add|put|place|stick)\b", command)
        )
        explicit_shopping = "shopping" in normal or "grocery" in normal
        known_shopping = _shopping_items(origin_context)
        known_todo = tuple(
            item
            for entity in list_entities
            for item in _context_items(origin_context, entity.name)
        )
        is_shopping = explicit_shopping or (
            selected_list is None
            and bool(known_shopping)
            and _canonical_item(text, known_shopping) in known_shopping
        )
        if selected_list is None and known_todo:
            matched_todo = _canonical_item(text, known_todo)
            if not known_shopping or matched_todo in known_todo:
                is_shopping = False
        elif selected_list is None and not known_todo:
            is_shopping = True
        if (
            selected_list is None
            and len(list_entities) > 1
            and not explicit_shopping
            and not known_shopping
            and not known_todo
        ):
            is_shopping = False
        if not (removal or completion or addition or (explicit_shopping and " to " in normal)):
            raise RouteDeclined("The list operation was not understood")

        item = _clean_item(text)
        item = re.sub(
            r"^(?:done with|check off|check|mark off|cross|remove|delete|erase|drop|take|mark|put|place|stick|add)\s+",
            "",
            item,
            flags=re.IGNORECASE,
        )
        if removal:
            item = re.split(
                r"\s+(?:off|from|out of)\s+(?:the\s+)?(?:shopping\s+|grocery\s+|chores\s+)?(?:list|chores)\b",
                item,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0]
        elif completion:
            item = re.split(
                r"\s+(?:as\s+)?(?:done|complete|completed|finished|cleaned|off)\b"
                r"|\s+on\s+(?:the\s+)?(?:done|completed)\s+list\b"
                r"|\s+to\s+(?:the\s+)?completed items\b"
                r"|\s+(?:on|from)\s+(?:the\s+)?(?:shopping\s+|grocery\s+|chores\s+)?(?:list|chores)\b",
                item,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0]
        else:
            item = re.split(
                r"\s+(?:to|onto|on|in)\s+(?:my\s+|the\s+)?"
                r"(?:shopping\s+|grocery\s+|chores\s+)?(?:list|chores)\b",
                item,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0]
        item = _clean_item(item)
        if not item:
            raise RouteDeclined("The list item was empty")
        if is_shopping:
            intent = "HassShoppingListCompleteItem" if completion or removal else "HassShoppingListAddItem"
            return _step(intent, {"item": item.casefold()})
        if not list_name:
            return IntentPlan(response="Which list should I use?")
        item = _canonical_task_wording(item)
        item = _canonical_item(item, _context_items(origin_context, list_name))
        item = item[:1].upper() + item[1:]
        if removal:
            intent = "HassListRemoveItem"
        elif completion:
            intent = "HassListCompleteItem"
        else:
            intent = "HassListAddItem"
        entity_ids = (selected_list.entity_id,) if selected_list else ()
        return _step(intent, {"name": list_name, "item": item}, *entity_ids)

    def _timer_entities(
        self,
        text: str,
        catalog: CatalogSnapshot,
        origin_context: Mapping[str, object] | None,
    ) -> tuple[CatalogEntity, ...]:
        timers = tuple(entity for entity in catalog.entities if entity.domain == "timer")
        normal = _normal(text)
        explicit = tuple(
            entity
            for entity in timers
            if any(
                identifier
                and f" {identifier} " in f" {normal} "
                for identifier in {
                    _normal(entity.name).removesuffix(" timer"),
                    _normal(entity.entity_id.split(".", 1)[-1]),
                    *(_normal(alias).removesuffix(" timer") for alias in entity.aliases),
                }
            )
        )
        if explicit:
            return explicit
        if origin_context:
            raw = origin_context.get("active_timer_ids")
            if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
                active = {str(value) for value in raw}
                selected = tuple(entity for entity in timers if entity.entity_id in active)
                if selected:
                    return selected
        return ()

    def _plan_timer(
        self,
        text: str,
        catalog: CatalogSnapshot,
        origin_context: Mapping[str, object] | None,
    ) -> IntentPlan:
        normal = _normal(text)
        timers = self._timer_entities(text, catalog, origin_context)
        timer_ids = tuple(entity.entity_id for entity in timers)
        timer_name = timers[0].name.removesuffix(" Timer") if len(timers) == 1 else None
        data: dict[str, Any] = {"name": timer_name} if timer_name else {}

        if re.search(r"\b(all|every)\b.*\btimers?\b", normal):
            all_ids = tuple(entity.entity_id for entity in catalog.entities if entity.domain == "timer")
            return _step("HassCancelAllTimers", {}, *all_ids)
        if re.search(
            r"\b(how (?:long|much time)|time remaining|status|check|where (?:are we|we re))\b",
            normal,
        ):
            return _step("HassTimerStatus", data, *timer_ids)
        if re.search(r"\b(resume|unpause|continue|restart|pick up)\b", normal) or re.search(
            r"\bstart\b.*\bagain\b", normal
        ):
            return _step("HassUnpauseTimer", data, *timer_ids)
        if re.search(r"\b(pause|hold|freeze|halt)\b", normal):
            return _step("HassPauseTimer", data, *timer_ids)
        if re.search(r"\b(cancel|stop|clear|delete|turn off)\b", normal):
            return _step("HassCancelTimer", data, *timer_ids)

        values = _duration(text)
        if not values:
            raise RouteDeclined("The timer duration was not understood")
        data.update(values)
        if re.search(r"\b(add|extend|increase|bump|tack|give|extra|forward)\b", normal):
            intent = "HassIncreaseTimer"
        elif re.search(r"\b(subtract|cut|reduce|shorten|remove|knock|take)\b", normal):
            intent = "HassDecreaseTimer"
        elif (
            re.search(r"\b(start|set|begin|create)\b", normal)
            or normal.startswith("timer ")
            or bool(values)
        ):
            intent = "HassStartTimer"
            if not timer_ids:
                timer_ids = ("timer.abstract",)
        else:
            raise RouteDeclined("The timer operation was not understood")
        return _step(intent, data, *timer_ids)

    def _plan_automation(self, text: str, catalog: CatalogSnapshot) -> IntentPlan:
        trigger, action_text = _automation_trigger(text, catalog)
        actions, entity_ids = _automation_actions(action_text, catalog)
        if trigger is None or not actions:
            raise RouteDeclined("The automation could not be expressed deterministically")
        definition = {"trigger": trigger, "actions": actions}
        return _step("IntentBridgeCreateAutomation", {"definition": definition}, *entity_ids)


def _time_trigger(text: str) -> dict[str, Any] | None:
    normal = _normal(text)
    matches = list(
        re.finditer(
            rf"\bat\s+({_NUMBER_EXPRESSION})(?:\s*o clock)?(?:\s*(am|pm)|\s+in (?:the )?(morning|evening|night))?\b",
            normal,
        )
    )
    if not matches:
        return None
    match = matches[-1]
    value = _number(match.group(1))
    if value is None or not float(value).is_integer() or not 0 <= int(value) <= 23:
        return None
    hour = int(value)
    period = match.group(2) or match.group(3)
    if period == "pm" or period in {"evening", "night"} or (
        not period and re.search(r"\b(night|nightly|evening)\b", normal)
    ):
        if hour < 12:
            hour += 12
    elif period == "am" and hour == 12:
        hour = 0
    return {"platform": "time", "at": f"{hour:02d}:00:00"}


def _conditional_parts(text: str) -> tuple[str, str] | None:
    normal = text.strip()
    leading = re.match(
        r"^(?:when|whenever|after|if|any time|every time)\s+(.+?)"
        r"(?:,|\s+i want\s+|\s+then\s+)(.+)$",
        normal,
        re.IGNORECASE,
    )
    if leading:
        return leading.group(1), leading.group(2)
    trailing = re.search(
        r"\b(?:when|whenever|after|if|any time|every time)\s+(.+)$",
        normal,
        re.IGNORECASE,
    )
    if trailing:
        return trailing.group(1), normal[: trailing.start()]
    return None


def _automation_trigger(
    text: str,
    catalog: CatalogSnapshot,
) -> tuple[dict[str, Any] | None, str]:
    normal = _normal(text)
    if "sunset" in normal or "sun sets" in normal or "sun goes down" in normal:
        return {"platform": "sun", "event": "sunset"}, text
    if trigger := _time_trigger(text):
        return trigger, text

    parts = _conditional_parts(text)
    if parts is None:
        return None, text
    condition, actions = parts
    condition_normal = _normal(condition)
    if re.search(r"\b(above|over|hotter than|exceeds|rises above)\b", condition_normal):
        value = _first_number(condition)
        climates = _mentioned_entities(condition, catalog, domains=frozenset({"climate"}))
        if not climates:
            climates = tuple(entity for entity in catalog.entities if entity.domain == "climate")
        if value is not None and len(climates) == 1:
            return {
                "platform": "numeric_state",
                "entity_id": climates[0].entity_id,
                "above": value,
            }, actions

    preferred_domains = frozenset({"binary_sensor"}) if re.search(
        r"\b(open(?:s|ed)?|motion|detects?|triggers?)\b", condition_normal
    ) else None
    entities = _mentioned_entities(condition, catalog, domains=preferred_domains)
    if not entities and preferred_domains:
        entities = _mentioned_entities(condition, catalog)
    if len(entities) != 1:
        return None, actions
    if re.search(r"\b(locked|locks?)\b", condition_normal):
        state = "locked"
    elif re.search(r"\b(off|finishes?|finished|done)\b", condition_normal):
        state = "off"
    else:
        state = "on"
    return {
        "platform": "state",
        "entity_id": entities[0].entity_id,
        "to": state,
    }, actions


def _automation_actions(
    text: str,
    catalog: CatalogSnapshot,
) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    normal = _normal(text)
    actions: list[dict[str, Any]] = []
    entity_ids: list[str] = []

    def add(action: str, *, entity: CatalogEntity | None = None, **payload: Any) -> None:
        item: dict[str, Any] = {"action": action}
        if entity is not None:
            item["entity_id"] = entity.entity_id
            entity_ids.append(entity.entity_id)
        item.update(payload)
        if item not in actions:
            actions.append(item)

    all_lights = bool(
        re.search(r"\b(all|every)\s+(?:the\s+)?lights?\b", normal)
        or re.search(r"\blights?\b.{0,20}\ball\b", normal)
    )
    if all_lights:
        state = "off" if re.search(r"\b(off|kill|kills|shut)\b", normal) else "on"
        add(f"light.turn_{state}", domain="light")

    adjustment_matches = list(
        re.finditer(
            rf"\b(?:to|above|over|than)\s+({_NUMBER_EXPRESSION})\b",
            normal,
        )
    )
    adjustment = _number(adjustment_matches[-1].group(1)) if adjustment_matches else None
    brightness = adjustment if re.search(r"\b(dim|dims|dimmed|brightness|percent)\b", normal) else None
    if brightness is not None:
        lights = _action_entities(text, catalog, "light")
        if len(lights) == 1:
            add("light.turn_on", entity=lights[0], brightness_pct=brightness)

    temperature_match = re.search(r"\b(?:temperature|thermostat|ac|air conditioner)\b", normal)
    if temperature_match and re.search(r"\b(set|sets|lower|lowers|drop|drops)\b", normal):
        temperature = adjustment
        climates = _action_entities(text, catalog, "climate")
        if not climates:
            climates = tuple(entity for entity in catalog.entities if entity.domain == "climate")
        if temperature is not None and len(climates) == 1:
            add("climate.set_temperature", entity=climates[0], temperature=temperature)

    for domain, open_action, close_action in (
        ("cover", "cover.open_cover", "cover.close_cover"),
        ("lock", "lock.unlock", "lock.lock"),
    ):
        entities = _action_entities(text, catalog, domain)
        if not entities:
            continue
        if domain == "cover":
            verb = close_action if re.search(r"\b(close|shut)\b", normal) else open_action
        else:
            verb = open_action if "unlock" in normal else close_action
        command_pattern = (
            r"\b(open(?:s|ed|ing)?|close(?:s|d|ing)?|shut)\b"
            if domain == "cover"
            else r"\b(lock|locks|unlock|unlocks)\b"
        )
        if re.search(command_pattern, normal):
            for entity in entities:
                add(verb, entity=entity)

    on_state = bool(re.search(r"\b(on|start|starts|activate|come on|comes on)\b", normal))
    off_state = bool(re.search(r"\b(off|kill|shut off|shuts off)\b", normal))
    if on_state or off_state:
        for domain in ("light", "switch", "fan", "media_player"):
            if domain == "light" and (brightness is not None or all_lights):
                continue
            entities = _action_entities(text, catalog, domain)
            for entity in entities:
                state = "off" if off_state and not on_state else "on"
                add(f"{domain}.turn_{state}", entity=entity)

    def position(item: Mapping[str, Any]) -> int:
        if entity_id := item.get("entity_id"):
            entity = next(
                (candidate for candidate in catalog.entities if candidate.entity_id == entity_id),
                None,
            )
            if entity is not None:
                positions = [
                    normal.find(phrase)
                    for phrase in (
                        _normal(entity.name),
                        _normal(entity.entity_id.split(".", 1)[-1]),
                        *(_normal(alias) for alias in entity.aliases),
                    )
                    if phrase and normal.find(phrase) >= 0
                ]
                if positions:
                    return min(positions)
                descriptor_positions = [
                    normal.find(word)
                    for word in set(_normal(entity.name).split())
                    - _DOMAIN_REFERENCE_WORDS.get(entity.domain, frozenset())
                    - _GENERIC_ENTITY_WORDS
                    if normal.find(word) >= 0
                ]
                if descriptor_positions:
                    return min(descriptor_positions)
        domain = str(item.get("domain") or str(item.get("action", "")).partition(".")[0])
        references = _DOMAIN_REFERENCE_WORDS.get(domain, frozenset())
        positions = [normal.find(word) for word in references if normal.find(word) >= 0]
        return min(positions) if positions else len(normal)

    actions.sort(key=position)
    return actions, tuple(sorted(set(entity_ids)))


@dataclass(frozen=True, slots=True)
class PendingClarification:
    """A requested operation waiting for one of several target entities."""

    intent_name: str
    candidate_entity_ids: tuple[str, ...]
    data: Mapping[str, Any] = field(default_factory=dict)
    response: str = "Which one did you mean?"


class ReferentCardinality(str, Enum):
    """The discourse meaning of an entity focus, independent of its size."""

    SINGULAR = "singular"
    GROUP = "group"


@dataclass(frozen=True, slots=True)
class EntityFocus:
    """The entity or collection currently at the centre of the dialogue."""

    entity_set: tuple[str, ...] = ()
    cardinality: ReferentCardinality = ReferentCardinality.SINGULAR
    selected_member: str | None = None


@dataclass(frozen=True, slots=True)
class PropertyFocus:
    """The property most recently discussed and the clause that established it."""

    property: str
    source_clause: str


@dataclass(frozen=True, slots=True)
class DiscourseCondition:
    """A condition guarding an operation in a conversational follow-up."""

    property: str
    operator: str
    value: Any
    target_frame_id: str | None = None
    source_clause: str = ""


@dataclass(frozen=True, slots=True)
class DiscourseOperationFrame:
    """One clause-level operation and the targets resolved for that clause."""

    frame_id: str
    predicate: str
    resolved_targets: tuple[str, ...]
    data: Mapping[str, Any] = field(default_factory=dict)
    effect: SemanticEffect | None = None
    source_clause: str = ""
    condition: DiscourseCondition | None = None


@dataclass(frozen=True, slots=True)
class UnresolvedDiscourseFrame:
    """An operation which cannot execute until a target constraint is supplied."""

    original_frame: DiscourseOperationFrame
    candidate_targets: tuple[str, ...]
    required_constraint: str


@dataclass(frozen=True, slots=True)
class ClauseReferent:
    """The frame and target set denoted by a local pronoun."""

    frame_id: str | None
    target_set: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DialogueState:
    """Typed discourse state retained by a caller between voice turns.

    The ``referent_*`` fields remain as a compatibility view for callers which
    predate discourse frames. New code should use ``focus``, ``property_focus``
    and ``prior_operations``.
    """

    referent_entity_ids: tuple[str, ...] = ()
    referent_data: Mapping[str, Any] = field(default_factory=dict)
    referent_intent_name: str | None = None
    referent_effect: SemanticEffect | None = None
    pending: PendingClarification | None = None
    focus: EntityFocus | None = None
    property_focus: PropertyFocus | None = None
    prior_operations: Mapping[str, DiscourseOperationFrame] = field(default_factory=dict)
    unresolved: UnresolvedDiscourseFrame | None = None
    clause_referents: Mapping[str, ClauseReferent] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Make legacy explicitly-constructed states immediately usable by the
        # typed resolver, and expose typed states through the old read API.
        if self.focus is None and self.referent_entity_ids:
            entity_set = tuple(dict.fromkeys(self.referent_entity_ids))
            object.__setattr__(
                self,
                "focus",
                EntityFocus(
                    entity_set=entity_set,
                    cardinality=(
                        ReferentCardinality.SINGULAR
                        if len(entity_set) == 1
                        else ReferentCardinality.GROUP
                    ),
                    selected_member=entity_set[0] if len(entity_set) == 1 else None,
                ),
            )
        elif self.focus is not None and not self.referent_entity_ids:
            object.__setattr__(self, "referent_entity_ids", self.focus.entity_set)


@dataclass(frozen=True, slots=True)
class PlanningTurn:
    """A plan together with the next immutable dialogue state."""

    plan: IntentPlan
    state: DialogueState


def _relative_plan(
    text: str,
    catalog: CatalogSnapshot,
    state: DialogueState,
) -> IntentPlan | None:
    normal = _normal(text)
    compact = _normal(_clean_item(text))
    target_ids = state.focus.entity_set if state.focus is not None else state.referent_entity_ids
    if not target_ids:
        return None
    entities = tuple(
        entity for entity in catalog.entities if entity.entity_id in target_ids
    )
    domains = {entity.domain for entity in entities}
    data = dict(state.referent_data)
    number = _first_number(text)
    explicit_referent = bool(
        re.search(r"\b(it|its|them|their|theirs|they|that|those)\b", normal)
    )
    contextual_command = bool(
        re.fullmatch(
            r"(?:(?:flick|flip|turn)\s+(?:the\s+)?locks?\s+"
            r"(?:on|off|open|closed?)|(?:hit|toggle)\s+(?:the\s+)?switch)"
            r"(?:\s+(?:now|right now))?",
            compact,
        )
    )
    bare_numeric = bool(
        number is not None
        and len(domains) == 1
        and re.fullmatch(
            rf"(?:(?:set|change|adjust|make|flip|flick|move|turn|bump)"
            rf"(?:\s+(?:it|them))?(?:\s+the)?"
            rf"(?:\s+(?:brightness|position|temperature|volume(?:[ _]+level)?|speed|level))?"
            rf"(?:\s+(?:up|on))?(?:\s+to)?\s+)?"
            rf"(?:about\s+|around\s+|roughly\s+)?"
            rf"{_NUMBER_EXPRESSION}(?:\s+(?:percent|degrees?))?",
            compact,
        )
    )
    if not explicit_referent and not contextual_command and not bare_numeric:
        return None
    if number is not None:
        property_operation = {
            "brightness": ("HassLightSet", "brightness", "light"),
            "percentage": ("HassFanSetSpeed", "percentage", "fan"),
            "position": ("HassSetPosition", "position", "cover"),
            "temperature": ("HassClimateSetTemperature", "temperature", "climate"),
            "volume_level": ("HassSetVolume", "volume_level", "media_player"),
        }.get(state.referent_effect.property if state.referent_effect else "")
        if property_operation is not None and domains == {property_operation[2]}:
            intent, slot, _ = property_operation
        elif domains == {"light"}:
            intent, slot = "HassLightSet", "brightness"
        elif domains == {"cover"}:
            intent, slot = "HassSetPosition", "position"
        elif domains == {"climate"}:
            intent, slot = "HassClimateSetTemperature", "temperature"
        elif domains == {"fan"}:
            intent, slot = "HassFanSetSpeed", "percentage"
        elif domains == {"media_player"}:
            intent, slot = "HassSetVolume", "volume_level"
        else:
            return None
        data[slot] = number
    elif re.search(r"\b(open|up)\b", normal) and domains <= {"cover", "lock"}:
        intent = "HassTurnOn" if domains == {"cover"} else "HassTurnOff"
    elif re.search(r"\b(close|closed|shut)\b", normal) and domains == {"cover"}:
        intent = "HassTurnOff"
    elif re.search(r"\bunlock(?:ed)?\b", normal) and domains == {"lock"}:
        intent = "HassTurnOff"
    elif re.search(r"\b(?:lock(?:ed)?|secure)\b", normal) and domains == {"lock"}:
        intent = "HassTurnOn"
    elif re.search(
        r"\b(off|deactivate|disable|kill|release)\b|\b(?:power|spin)\b.*\bdown\b",
        normal,
    ):
        intent = "HassTurnOff"
    elif re.search(
        r"\b(on|activate|enable|power|ignite|illuminate)\b|"
        r"\b(?:fire|light|spin)\b.*\bup\b",
        normal,
    ):
        intent = "HassTurnOn"
    elif re.search(r"\b(?:do it|hit(?: the)? switch|toggle)\b", normal):
        enabled_states = {
            "cover": "open",
            "lock": "locked",
            "media_player": "on",
        }
        states = {
            entity.state
            for entity in entities
            if entity.state is not None
        }
        if len(domains) != 1 or len(states) != 1:
            return None
        domain = next(iter(domains))
        intent = (
            "HassTurnOff"
            if next(iter(states)) == enabled_states.get(domain, "on")
            else "HassTurnOn"
        )
    else:
        return None
    data.pop("state", None)
    return _step(intent, data, *target_ids)


def _referent_qualifier_plan(
    text: str,
    catalog: CatalogSnapshot,
    state: DialogueState,
) -> IntentPlan | None:
    """Narrow a prior multi-entity operation without changing its semantics."""

    if len(state.referent_entity_ids) < 2 or not state.referent_intent_name:
        return None
    candidates = tuple(
        entity for entity in catalog.entities if entity.entity_id in state.referent_entity_ids
    )
    scores = [(entity, _entity_score(text, entity, catalog)) for entity in candidates]
    if not scores:
        return None
    highest = max(score for _, score in scores)
    winners = tuple(entity for entity, score in scores if score == highest and score > 0)
    if len(winners) != 1:
        return None
    entity = winners[0]
    data = {
        key: value
        for key, value in state.referent_data.items()
        if key not in {"area", "domain", "floor", "name"}
    }
    data["name"] = entity.name
    return _step(state.referent_intent_name, data, entity.entity_id)


def _pending_from_request(
    text: str,
    catalog: CatalogSnapshot,
    response: str,
) -> PendingClarification | None:
    normal = _normal(text)
    if re.search(r"\b(on|activate|enable)\b", normal):
        intent = "HassTurnOn"
    elif re.search(r"\b(off|deactivate|disable)\b", normal):
        intent = "HassTurnOff"
    elif "open" in normal:
        intent = "HassTurnOn"
    elif re.search(r"\b(close|shut)\b", normal):
        intent = "HassTurnOff"
    else:
        return None
    if re.search(r"\b(?:lamps?|lights?)\b", normal):
        domains = frozenset({"light"})
    elif re.search(r"\b(blinds?|curtains?|covers?|garage doors?)\b", normal):
        domains = frozenset({"cover"})
    else:
        return None
    candidates = _mentioned_entities(text, catalog, domains=domains, all_ties=True)
    if len(candidates) < 2:
        return None
    return PendingClarification(
        intent_name=intent,
        candidate_entity_ids=tuple(entity.entity_id for entity in candidates),
        response=response,
    )


def _resolve_pending(
    text: str,
    catalog: CatalogSnapshot,
    pending: PendingClarification,
) -> IntentPlan | None:
    candidates = tuple(
        entity for entity in catalog.entities if entity.entity_id in pending.candidate_entity_ids
    )
    scores = [(entity, _entity_score(text, entity, catalog)) for entity in candidates]
    if not scores:
        return None
    highest = max(score for _, score in scores)
    winners = tuple(entity for entity, score in scores if score == highest and score > 0)
    if len(winners) != 1:
        return None
    entity = winners[0]
    data = {**pending.data, "name": entity.name}
    return _step(pending.intent_name, data, entity.entity_id)


def _singular_pronoun_for_group(text: str, state: DialogueState) -> bool:
    """Return whether a singular pronoun conflicts with the current group focus."""

    return bool(
        state.focus is not None
        and state.focus.cardinality == ReferentCardinality.GROUP
        and re.search(r"\b(it|its|this|that)\b", _normal(text))
    )


def _predicate_from_followup(text: str) -> tuple[str, SemanticEffect | None]:
    normal = _normal(text)
    if re.search(r"\b(off|deactivate|disable)\b", normal):
        return "HassTurnOff", SemanticEffect("command", "activation", "set", False)
    if re.search(r"\b(on|activate|enable)\b", normal):
        return "HassTurnOn", SemanticEffect("command", "activation", "set", True)
    return "unresolved", None


def _state_with_unresolved_singular(text: str, state: DialogueState) -> DialogueState:
    """Retain a singular follow-up transaction instead of applying it to a group."""

    predicate, effect = _predicate_from_followup(text)
    frame_id = f"frame-{len(state.prior_operations) + 1}"
    original = DiscourseOperationFrame(
        frame_id=frame_id,
        predicate=predicate,
        resolved_targets=(),
        effect=effect,
        source_clause=text.strip(),
    )
    return replace(
        state,
        unresolved=UnresolvedDiscourseFrame(
            original_frame=original,
            candidate_targets=state.focus.entity_set if state.focus else (),
            required_constraint="single target",
        ),
    )


class PlanningSession:
    """Apply an ``IntentPlanner`` while explicitly retaining dialogue state."""

    def __init__(self, planner: IntentPlanner, state: DialogueState | None = None) -> None:
        self._planner = planner
        self.state = state or DialogueState()

    def reset(self, state: DialogueState | None = None) -> None:
        self.state = state or DialogueState()

    def plan_turn(
        self,
        text: str,
        catalog: CatalogSnapshot,
        origin_context: Mapping[str, object] | None = None,
    ) -> PlanningTurn:
        if self.state.pending is not None:
            plan = _resolve_pending(text, catalog, self.state.pending)
            if plan is None:
                plan = IntentPlan(response=self.state.pending.response)
                return PlanningTurn(plan=plan, state=self.state)
            next_state = _state_from_plan(plan)
            self.state = next_state
            return PlanningTurn(plan=plan, state=next_state)

        if plan := _referent_qualifier_plan(text, catalog, self.state):
            next_state = _state_from_plan(plan, text, self.state)
            self.state = next_state
            return PlanningTurn(plan=plan, state=next_state)

        if _singular_pronoun_for_group(text, self.state):
            next_state = _state_with_unresolved_singular(text, self.state)
            plan = IntentPlan(response="Which one did you mean?")
            self.state = next_state
            return PlanningTurn(plan=plan, state=next_state)

        if plan := _relative_plan(text, catalog, self.state):
            next_state = _state_from_plan(plan, text, self.state)
            self.state = next_state
            return PlanningTurn(plan=plan, state=next_state)

        plan = self._planner.plan(text, catalog, origin_context)
        if plan.steps:
            next_state = _state_from_plan(plan, text, self.state)
        elif plan.response and (pending := _pending_from_request(text, catalog, plan.response)):
            next_state = DialogueState(pending=pending)
        else:
            next_state = self.state
        self.state = next_state
        return PlanningTurn(plan=plan, state=next_state)

    def plan(
        self,
        text: str,
        catalog: CatalogSnapshot,
        origin_context: Mapping[str, object] | None = None,
    ) -> IntentPlan:
        """IntentPlanner-compatible convenience method returning only the plan."""

        return self.plan_turn(text, catalog, origin_context).plan


def _state_from_plan(plan: IntentPlan) -> DialogueState:
    entity_ids = tuple(
        sorted({entity_id for step in plan.steps for entity_id in step.entity_ids})
    )
    data: dict[str, Any] = {}
    intent_name: str | None = None
    effect: SemanticEffect | None = None
    if len(plan.steps) == 1:
        intent_name = plan.steps[0].operation
        effect = plan.steps[0].effect or semantic_effect_for_call(plan.steps[0].call)
        call_data = plan.steps[0].call.data
        data.update(call_data)
    return DialogueState(
        referent_entity_ids=entity_ids,
        referent_data=data,
        referent_intent_name=intent_name,
        referent_effect=effect,
    )


__all__ = [
    "DialogueState",
    "PendingClarification",
    "PlanningSession",
    "PlanningTurn",
    "SupplementalIntentPlanner",
]
