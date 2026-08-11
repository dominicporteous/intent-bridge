"""Topology-driven natural-language planning beyond the packaged HassIL grammar.

This module deliberately knows nothing about benchmark files or expected results.  It
matches commands only against the utterance, the current Home Assistant catalogue and
optional conversation/origin context.  HassIL remains the primary recognizer; this is a
conservative fallback for common, naturally phrased commands and compound requests.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from intent_bridge.config import settings
from intent_bridge.core.text import normalize_search_text
from intent_bridge.core.voice import RouteDeclined
from intent_bridge.intent_engine.measurement import MeasurementIntentPlanner
from intent_bridge.intent_engine.models import (
    CatalogArea,
    CatalogEntity,
    CatalogFloor,
    CatalogSnapshot,
    IntentMatch,
    IntentPlan,
    OhfIntentCall,
    PlannedIntent,
    SemanticEffect,
    SlotValue,
    semantic_effect_for_call,
)
from intent_bridge.intent_engine.outcomes import (
    AmbiguousTarget,
    CapabilityMismatch,
    IncompleteCompound,
    NoTarget,
    PlanningOutcome,
    Resolved,
    UnsupportedOperation,
)

_DOMAIN_ALIASES: tuple[tuple[str, str], ...] = (
    ("robot vacuum cleaners", "vacuum"),
    ("robot vacuum cleaner", "vacuum"),
    ("robot vacuums", "vacuum"),
    ("robot vacuum", "vacuum"),
    ("vacuum cleaners", "vacuum"),
    ("vacuum cleaner", "vacuum"),
    ("vacuums", "vacuum"),
    ("vacuum", "vacuum"),
    ("weather", "weather"),
    ("temperature sensors", "sensor"),
    ("temperature sensor", "sensor"),
    ("media players", "media_player"),
    ("media player", "media_player"),
    ("garage doors", "cover"),
    ("garage door", "cover"),
    ("thermostats", "climate"),
    ("thermostat", "climate"),
    ("climate controls", "climate"),
    ("climate control", "climate"),
    ("climates", "climate"),
    ("climate", "climate"),
    ("temperatures", "climate"),
    ("temperature", "climate"),
    ("temp", "climate"),
    ("curtains", "cover"),
    ("curtain", "cover"),
    ("shutters", "cover"),
    ("shutter", "cover"),
    ("shades", "cover"),
    ("shade", "cover"),
    ("blinds", "cover"),
    ("blind", "cover"),
    ("covers", "cover"),
    ("cover", "cover"),
    ("speakers", "media_player"),
    ("speaker", "media_player"),
    ("televisions", "media_player"),
    ("television", "media_player"),
    ("lighting", "light"),
    ("windows", "cover"),
    ("window", "cover"),
    ("lights", "light"),
    ("light", "light"),
    ("lamps", "light"),
    ("lamp", "light"),
    ("fans", "fan"),
    ("fan", "fan"),
    ("switches", "switch"),
    ("switch", "switch"),
    ("locks", "lock"),
    ("lock", "lock"),
    ("scenes", "scene"),
    ("scene", "scene"),
    ("scripts", "script"),
    ("script", "script"),
    ("tvs", "media_player"),
    ("tv", "media_player"),
    ("sensors", "sensor"),
    ("sensor", "sensor"),
    ("illumination", "light"),
    ("deadbolts", "lock"),
    ("deadbolt", "lock"),
)
_DOMAIN_WORDS = frozenset(word for phrase, _ in _DOMAIN_ALIASES for word in phrase.split())
_COLORS = (
    "warm white",
    "cool white",
    "dark blue",
    "light blue",
    "dark green",
    "light green",
    "turquoise",
    "magenta",
    "lavender",
    "orange",
    "purple",
    "yellow",
    "white",
    "violet",
    "indigo",
    "green",
    "blue",
    "pink",
    "cyan",
    "teal",
    "black",
    "gold",
    "silver",
    "lime",
    "navy",
    "maroon",
    "red",
)
_SMALL_NUMBERS = {
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
}
_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_NUMBER_WORDS = frozenset((*_SMALL_NUMBERS, *_TENS, "hundred", "and"))
_QUERY_MARKERS = (
    "what is",
    "what s",
    "whats",
    "how is",
    "tell me",
    "let me know",
    "check",
    "report",
    "give me",
    "read",
    "confirm",
    "show me",
    "see if",
    "find out",
    "current position",
    "situation",
    "where",
    "status",
    "state",
    "doing",
    "whether",
)
_ACTION_START = (
    r"turn|turning|switch|switching|power|powering|flip|flipping|flick|flicking|"
    r"set|setting|change|changing|adjust|adjusting|dim|dimming|brighten|brightening|"
    r"illuminate|open|opening|close|closing|shut|"
    r"raise|raising|lower|lowering|lock|locking|unlock|unlocking|secure|securing|"
    r"activate|activating|deactivate|deactivating|run|running|start|starting|"
    r"stop|stopping|launch|launching|pause|pausing|resume|resuming|continue|"
    r"continuing|play|playing|mute|muting|unmute|unmuting|skip|skipping|"
    r"next|previous|check|checking|"
    r"tell|let|get|report|give|read|confirm|what|is|are|how"
)
_COMPOUND_RE = re.compile(
    rf"\b(?P<connector>and\s+then|and\s+also|and\s+oh\s+and|"
    rf"and\s+while\s+you\s+(?:are|re)\s+at\s+it|"
    rf"and\s+while(?!\s+you\s+(?:are|re)\s+at\s+it)|plus|then|after|"
    rf"as\s+well\s+as|while\s+you\s+(?:are|re)\s+at\s+it|"
    rf"while(?!\s+you\s+(?:are|re)\s+at\s+it)|and)\b\s+"
    rf"(?=(?:(?:please|also|then|oh|you|could\s+you)\s+)*(?:{_ACTION_START})\b)",
    re.IGNORECASE,
)
_POLITE_RE = re.compile(
    r"\b(?:hey|okay|ok|please|could you|would you|can you|will you|"
    r"if you don t mind|if you do not mind|if possible|if you could|for me|"
    r"while you re at it|while you are at it|right away|oh and|oh|quickly|quick)\b",
    re.IGNORECASE,
)
_PRONOUN_RE = re.compile(r"\b(?:it|its|them|their|theirs|they|those|these|that|there)\b")
_COVER_SUBTYPE_FORMS: Mapping[str, tuple[str, ...]] = {
    "window_covering": (
        "blind",
        "blinds",
        "curtain",
        "curtains",
        "drape",
        "drapes",
        "shade",
        "shades",
        "shutter",
        "shutters",
        "window",
        "windows",
    ),
    "garage_door": ("garage door", "garage doors"),
    "awning": ("awning", "awnings"),
    "damper": ("damper", "dampers"),
    "gate": ("gate", "gates"),
}


@dataclass(frozen=True, slots=True)
class _Operation:
    intent_name: str
    slots: Mapping[str, Any]
    domain: str | None = None
    requested_effect: SemanticEffect | None = None


@dataclass(frozen=True, slots=True)
class NormalizedOperator:
    """A domain-neutral predicate selected before target resolution.

    ``property`` and ``value`` describe what the speaker requested; ``intent_name``
    is only the OHF transport chosen for that predicate.  Keeping this boundary
    explicit prevents lexical verbs such as ``switch`` and nouns such as ``sound``
    from being mistaken for target domains.
    """

    property: str
    operator: str
    value: Any
    intent_name: str
    domain: str | None
    slots: Mapping[str, Any]

    def as_operation(self) -> _Operation:
        return _Operation(
            self.intent_name,
            self.slots,
            self.domain,
            SemanticEffect(
                "command",
                self.property,
                self.operator,
                self.value,
                explicit_power_transition=self.property == "power",
            ),
        )


@dataclass(frozen=True, slots=True)
class _Target:
    slots: Mapping[str, Any]
    entity_ids: tuple[str, ...]
    excluded_entity_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _TargetMember:
    """One independently resolvable member of a coordinated noun phrase."""

    text: str


@dataclass(frozen=True, slots=True)
class _TargetSet:
    """The semantic target of an operation, before catalog resolution."""

    relation: str
    members: tuple[_TargetMember, ...]


class _AmbiguousTarget(Exception):
    pass


class _CapabilityMismatch(Exception):
    pass


def _normal(value: object) -> str:
    if isinstance(value, str):
        value = value.replace("%", " percent ").replace("°", " degrees ")
    normalized = normalize_search_text(value).replace(".", " ")
    # Correct a few frequent speech-to-text spellings before classifying the
    # request. Fuzzy entity matching happens too late to infer query semantics.
    return re.sub(r"\b(?:temprature|temperture|tempature)\b", "temperature", normalized)


_GENERIC_ENTITY_MATCH_TOKENS = frozenset(
    {
        "on",
        "off",
        "turn",
        "switch",
        "power",
        "activate",
        "deactivate",
        "enable",
        "disable",
        "run",
        "start",
        "stop",
        "open",
        "close",
        "lock",
        "unlock",
        "play",
        "pause",
        "resume",
        "mute",
        "unmute",
        "set",
        "change",
        "adjust",
        "dim",
        "brighten",
        "raise",
        "lower",
        "flick",
        "flip",
    }
)

DEFAULT_IGNORED_ENTITY_DOMAINS = frozenset({"update"})

_ON_OFF_ACTION_RE = re.compile(
    r"\b(?:turn|switch|power|flip|flick|enable|disable|activate|deactivate|start|stop|on|off)\b"
)


def _contains_phrase(text: str, phrase: str) -> bool:
    phrase = _normal(phrase)
    if not phrase:
        return False
    if re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text):
        return True
    compact_phrase = phrase.replace(" ", "")
    return len(compact_phrase) >= 7 and compact_phrase in text.replace(" ", "")


def _is_generic_entity_phrase(phrase: str) -> bool:
    return phrase in _GENERIC_ENTITY_MATCH_TOKENS


def _is_on_off_action(text: str) -> bool:
    return bool(_ON_OFF_ACTION_RE.search(text))


def _is_power_control_entity(entity: CatalogEntity) -> bool:
    if entity.domain != "switch":
        return False
    normalized_name = _normal(entity.name)
    normalized_id_tail = _normal(entity.entity_id.split(".", 1)[1])
    return "power" in normalized_name or "power" in normalized_id_tail


def _prefer_power_switch_hits(
    text: str,
    hits: list[tuple[CatalogEntity, str]],
) -> list[tuple[CatalogEntity, str]]:
    if not _is_on_off_action(text):
        return hits
    power_hits = [hit for hit in hits if _is_power_control_entity(hit[0])]
    return power_hits or hits


def _clean_clause(text: str) -> str:
    text = _POLITE_RE.sub(" ", text)
    text = re.sub(r"\b(?:also|then|just|kindly)\b", " ", text)
    text = re.sub(r"^(?:you\s+)?turning\b", "turn", text)
    text = re.sub(r"^(?:you\s+)?setting\b", "set", text)
    text = re.sub(r"^(?:you\s+)?checking\b", "check", text)
    text = re.sub(r"^(?:you\s+)?closing\b", "close", text)
    text = re.sub(r"^(?:you\s+)?opening\b", "open", text)
    gerunds = {
        "activating": "activate",
        "adjusting": "adjust",
        "changing": "change",
        "checking": "check",
        "continuing": "continue",
        "deactivating": "deactivate",
        "dimming": "dim",
        "launching": "launch",
        "locking": "lock",
        "lowering": "lower",
        "muting": "mute",
        "pausing": "pause",
        "playing": "play",
        "powering": "power",
        "raising": "raise",
        "resuming": "resume",
        "running": "run",
        "securing": "secure",
        "skipping": "skip",
        "starting": "start",
        "stopping": "stop",
        "switching": "switch",
        "unlocking": "unlock",
        "unmuting": "unmute",
    }
    for gerund, imperative in gerunds.items():
        text = re.sub(rf"^(?:you\s+)?{gerund}\b", imperative, text)
    return re.sub(r"\s+", " ", text).strip()


def split_compound_request(text: str) -> tuple[str, ...]:
    """Return independently recognizable clauses in dependency order.

    This intentionally leaves target lists such as ``bedroom and kitchen lights``
    intact because ``kitchen`` is not an operation word. For ``A after B``, B is
    returned first so execution order reflects the dependency expressed by the
    utterance rather than its surface word order.
    """

    normalized = _normal(text)
    matches = tuple(_COMPOUND_RE.finditer(normalized))
    if not matches:
        clause = _clean_clause(normalized)
        return (clause,) if clause else ()

    raw_parts: list[str] = []
    connectors: list[str] = []
    start = 0
    for match in matches:
        raw_parts.append(normalized[start : match.start()])
        connectors.append(match.group("connector").casefold())
        start = match.end()
    raw_parts.append(normalized[start:])

    ordered = [_clean_clause(raw_parts[0])]
    for connector, raw_part in zip(connectors, raw_parts[1:], strict=True):
        clause = _clean_clause(raw_part)
        if not clause:
            continue
        if connector == "after":
            ordered.insert(0, clause)
        else:
            ordered.append(clause)
    return tuple(clause for clause in ordered if clause)


def _words_to_number(words: Sequence[str]) -> int | None:
    current = 0
    found = False
    for word in words:
        if word == "and":
            continue
        if word.isdigit():
            current += int(word)
            found = True
        elif word in _SMALL_NUMBERS:
            current += _SMALL_NUMBERS[word]
            found = True
        elif word in _TENS:
            current += _TENS[word]
            found = True
        elif word == "hundred" and found:
            current = max(current, 1) * 100
        else:
            return None
    return current if found else None


def _number_before_marker(text: str, markers: Iterable[str]) -> int | None:
    tokens = text.split()
    marker_set = set(markers)
    for index in range(len(tokens) - 1, -1, -1):
        token = tokens[index].rstrip("%")
        has_inline_percent = tokens[index].endswith("%") and token.isdigit()
        if token not in marker_set and not has_inline_percent:
            continue
        if has_inline_percent:
            return int(token)
        start = index - 1
        while start >= 0 and (tokens[start] in _NUMBER_WORDS or tokens[start].isdigit()):
            start -= 1
        value = _words_to_number(tokens[start + 1 : index])
        if value is not None:
            return value
    match = re.search(r"\bto\s+(\d{1,5})\b", text)
    if match:
        return int(match.group(1))
    match = re.search(
        rf"\bto\s+((?:{'|'.join(sorted(_NUMBER_WORDS - {'and'}))})(?:\s+(?:and\s+)?"
        rf"(?:{'|'.join(sorted(_NUMBER_WORDS - {'and'}))}))*)\b",
        text,
    )
    return _words_to_number(match.group(1).split()) if match else None


def _number_after_keyword(text: str, *keywords: str) -> int | None:
    tokens = text.split()
    for keyword in keywords:
        keyword_tokens = keyword.split()
        for index in range(len(tokens) - len(keyword_tokens) + 1):
            if tokens[index : index + len(keyword_tokens)] != keyword_tokens:
                continue
            start = index + len(keyword_tokens)
            while start < len(tokens) and tokens[start] in {"to", "at", "is"}:
                start += 1
            end = start
            while end < len(tokens) and (tokens[end] in _NUMBER_WORDS or tokens[end].isdigit()):
                end += 1
            value = _words_to_number(tokens[start:end])
            if value is not None:
                return value
    return None


def _number_before_keyword(text: str, *keywords: str) -> int | None:
    tokens = text.split()
    for keyword in keywords:
        keyword_tokens = keyword.split()
        for index in range(len(tokens) - len(keyword_tokens) + 1):
            if tokens[index : index + len(keyword_tokens)] != keyword_tokens:
                continue
            start = index - 1
            while start >= 0 and (tokens[start] in _NUMBER_WORDS or tokens[start].isdigit()):
                start -= 1
            value = _words_to_number(tokens[start + 1 : index])
            if value is not None:
                return value
    return None


def _number_after_relation(text: str) -> int | None:
    tokens = text.split()
    for index, token in enumerate(tokens[:-1]):
        if token not in {"to", "at", "around"}:
            continue
        end = index + 1
        while end < len(tokens) and (tokens[end] in _NUMBER_WORDS or tokens[end].isdigit()):
            end += 1
        value = _words_to_number(tokens[index + 1 : end])
        if value is not None:
            return value
    return None


def _trailing_number(text: str) -> int | None:
    """Return a numeric value at the end of an otherwise resolved setter."""

    tokens = text.rstrip(" .?!").split()
    start = len(tokens)
    while start > 0 and (tokens[start - 1] in _NUMBER_WORDS or tokens[start - 1].isdigit()):
        start -= 1
    return _words_to_number(tokens[start:]) if start < len(tokens) else None


def _explicit_domain(text: str) -> str | None:
    # Explicit automation-container nouns own the target even when their name
    # contains another capability word (for example, ``lock up script``).
    if re.search(r"\bscripts?\b", text):
        return "script"
    if re.search(r"\bscenes?\b", text):
        return "scene"
    imperative_switch = bool(re.match(r"^(?:please\s+)?switch\s+(?:on|off)\b", text))
    # A terminal control noun is stronger than a phrase embedded within it:
    # "garage door locks" means locks, not every garage-door cover.
    for pattern, domain in (
        (r"\b(?:locks?|deadbolts?)\b", "lock"),
        (r"\b(?:fans?)\b", "fan"),
        (r"\b(?:lights?|lamps?|lighting|illumination)\b", "light"),
        (r"\b(?:switches?)\b", "switch"),
    ):
        if domain == "switch" and imperative_switch and not re.search(r"\bswitches\b", text):
            continue
        if re.search(pattern, text):
            return domain
    for phrase, domain in _DOMAIN_ALIASES:
        if domain == "switch" and imperative_switch and not re.search(r"\bswitches\b", text):
            continue
        if _contains_phrase(text, phrase):
            return domain
    return None


def _cover_subtype(text: str) -> str | None:
    for subtype, forms in _COVER_SUBTYPE_FORMS.items():
        if any(_contains_phrase(text, form) for form in forms):
            return subtype
    return None


def _semantic_domain(text: str) -> str | None:
    if re.search(r"\b(?:weather|outside|outdoors|outdoor)\b", text) and re.search(
        r"\b(?:temperature|temp|forecast|weather|hot|cold|warm)\b", text
    ):
        return "weather"
    if re.search(r"\b(?:color|colour|brightness|bright|dim)\b", text):
        return "light"
    if re.search(r"\bfan speed\b", text):
        return "fan"
    if re.search(r"\bposition\b", text) or _cover_subtype(text):
        return "cover"
    if re.search(r"\b(?:volume|sound|mute)\b", text):
        return "media_player"
    if re.search(r"\btemperature sensors?\b", text):
        return "sensor"
    if re.search(r"\b(?:temperature|temp|climate|warm)\b", text):
        return "climate"
    return None


def _matches_cover_subtype(entity: CatalogEntity, subtype: str | None) -> bool:
    if subtype is None:
        return True
    forms = _COVER_SUBTYPE_FORMS[subtype]
    labels = (
        _normal(entity.device_class),
        _normal(entity.name),
        _normal(entity.entity_id.split(".", 1)[-1]),
        *(_normal(alias) for alias in entity.aliases),
    )
    return any(_contains_phrase(label, form) for label in labels for form in forms)


def _desired_query_state(text: str) -> str | None:
    for spoken, state in (
        ("unlocked", "unlocked"),
        ("locked", "locked"),
        ("closed", "closed"),
        ("close", "closed"),
        ("open", "open"),
        ("running", "on"),
        ("playing", "playing"),
        ("paused", "paused"),
        ("off", "off"),
        ("on", "on"),
    ):
        if _contains_phrase(text, spoken):
            return state
    return None


def normalize_lexical_operator(
    text: str,
    contextual_domain: str | None = None,
) -> NormalizedOperator | None:
    """Normalize imperative surface wording without resolving a target.

    The patterns here intentionally require imperative syntax.  For example,
    ``is the sound off`` remains a query, while ``turn off the sound`` becomes a
    mute operation.  Applicability is subsequently enforced by resolving only
    entities compatible with ``domain``.
    """

    text = _clean_clause(_normal(text))
    explicit_domain = _explicit_domain(text)
    domain = _semantic_domain(text) or explicit_domain or contextual_domain

    if re.match(r"^(?:flip|flick|push|move|pull|roll)\b", text) and re.search(
        r"\b(?:blinds?|curtains?|shades?|shutters?|covers?|windows?)\b.*\bdown\b",
        text,
    ):
        return NormalizedOperator(
            "position", "set", "closed", "HassTurnOff", "cover", {}
        )
    if re.match(r"^(?:flip|flick|push|move|pull|roll)\b", text) and re.search(
        r"\b(?:blinds?|curtains?|shades?|shutters?|covers?|windows?)\b.*\bup\b",
        text,
    ):
        return NormalizedOperator(
            "position", "set", "open", "HassTurnOn", "cover", {}
        )

    if re.match(
        r"^(?:turn|switch)\s+off\s+(?:the\s+)?(?:sound|audio|volume)\b|"
        r"^(?:silence|mute)\b",
        text,
    ):
        return NormalizedOperator(
            "muted", "set", True, "HassMediaPlayerMute", "media_player", {}
        )
    if re.match(
        r"^(?:restore|enable)\s+(?:the\s+)?(?:sound|audio)\b|^unmute\b",
        text,
    ):
        return NormalizedOperator(
            "muted", "set", False, "HassMediaPlayerUnmute", "media_player", {}
        )

    if (
        re.match(r"^(?:resume|unpause|continue)\b", text)
        and (domain == "media_player" or re.search(r"\bplayback\b", text))
    ) or (
        re.match(r"^restart\b", text)
        and (domain == "media_player" or re.search(r"\bplayback\b", text))
    ):
        return NormalizedOperator(
            "playback", "set", "playing", "HassMediaUnpause", "media_player", {}
        )

    # At the start of an imperative, "switch" is a verb.  Preserve a domain
    # supplied by the target phrase (for coordinated groups such as
    # ``bathroom and kitchen lights``), while never inferring the switch domain
    # merely from the verb itself.  Power remains polymorphic for explicitly
    # named heterogeneous members.
    switch_power = re.match(r"^switch\s+(?P<state>on|off)\b", text)
    if switch_power:
        is_on = switch_power.group("state") == "on"
        return NormalizedOperator(
            "power",
            "set",
            is_on,
            "HassTurnOn" if is_on else "HassTurnOff",
            domain,
            {},
        )

    # Bare ``X to 27`` setters inherit the single conventional numeric property
    # exposed by X's domain (or by the prior discourse property's domain).
    numeric_value = _number_after_relation(text)
    if numeric_value is not None and 0 <= numeric_value <= 100:
        numeric_operators = {
            "light": ("brightness", "HassLightSet", "brightness"),
            "cover": ("position", "HassSetPosition", "position"),
            "fan": ("percentage", "HassFanSetSpeed", "percentage"),
            "climate": ("temperature", "HassClimateSetTemperature", "temperature"),
            "media_player": ("volume_level", "HassSetVolume", "volume_level"),
        }
        if domain in numeric_operators:
            property_name, intent_name, slot = numeric_operators[domain]
            return NormalizedOperator(
                property_name,
                "set",
                numeric_value,
                intent_name,
                domain,
                {slot: numeric_value},
            )
    return None


def _classify(text: str, contextual_domain: str | None = None) -> _Operation | None:
    domain = _semantic_domain(text) or _explicit_domain(text) or contextual_domain
    normalized_operator = normalize_lexical_operator(text, contextual_domain)
    if normalized_operator is not None:
        return normalized_operator.as_operation()
    paired_state_query = bool(
        re.search(
            r"\b(?:on\s+or\s+off|off\s+or\s+on|open\s+or\s+closed|"
            r"closed\s+or\s+open|locked\s+or\s+unlocked|unlocked\s+or\s+locked|"
            r"up\s+or\s+down|down\s+or\s+up)\b",
            text,
        )
    )
    is_query = (
        text.startswith(("is ", "are ", "what ", "whats ", "how "))
        or any(marker in text for marker in _QUERY_MARKERS)
        or paired_state_query
        or text.startswith("get ")
    )
    if is_query:
        generic_temperature_query = bool(
            re.search(r"\b(?:temperature|temp)\b", text)
            and contextual_domain != "climate"
            and not re.search(
                r"\b(?:thermostat|climate|sensor)\b|"
                r"\btemperature\s+(?:in|at|of|for)\b",
                text,
            )
        )
        if (
            re.search(r"\b(?:weather|outside|outdoors|outdoor)\b", text)
            or generic_temperature_query
        ):
            return _Operation("HassGetWeather", {}, "weather")
        if re.search(r"\b(?:brightness|bright)\b", text):
            return _Operation("HassGetState", {}, "light")
        if re.search(r"\b(?:temperature|temp|warm)\b", text) and domain in {
            None,
            "climate",
        }:
            return _Operation("HassClimateGetTemperature", {}, "climate")
        slots: dict[str, Any] = {}
        if not paired_state_query and (state := _desired_query_state(text)):
            slots["state"] = state
        if domain is None and re.search(r"\b(?:locked|unlocked)\b", text):
            domain = "lock"
        return _Operation("HassGetState", slots, domain)

    if (
        re.search(r"\b(?:brightness|bright)\b", text)
        and not re.search(r"\b(?:set|change|adjust|make|dim|turn|flick|flip)\b", text)
        and _number_after_keyword(text, "brightness") is None
        and _number_before_keyword(text, "brightness") is None
    ):
        return _Operation("HassGetState", {}, "light")
    if re.search(r"\b(?:temperature|temp)\b", text) and not re.search(
        r"\b(?:set|change|adjust|make|raise|lower|turn|flick|flip)\b", text
    ):
        return (
            _Operation("HassGetState", {}, "sensor")
            if domain == "sensor"
            else _Operation("HassClimateGetTemperature", {}, "climate")
        )

    if "color temperature" in text or "colour temperature" in text:
        temperature = _number_before_marker(text, ("kelvin", "k"))
        if temperature is not None:
            return _Operation("HassLightSet", {"temperature": temperature}, "light")

    percentage = _number_before_marker(text, ("percent", "%"))
    if percentage is None:
        percentage = _number_after_keyword(
            text,
            "brightness",
            "level",
            "fan speed",
            "speed",
            "position",
            "volume",
        )
    if percentage is None:
        percentage = _number_before_keyword(text, "brightness", "speed", "position", "volume")
    if percentage is None:
        percentage = _number_after_relation(text)
    if percentage is None and domain in {
        "climate",
        "cover",
        "fan",
        "light",
        "media_player",
    }:
        percentage = _trailing_number(text)
    if percentage is not None and not 0 <= percentage <= 100:
        return None
    if "volume" in text and percentage is not None:
        return _Operation("HassSetVolume", {"volume_level": percentage}, "media_player")
    if (domain == "fan" or "fan speed" in text) and percentage is not None:
        return _Operation("HassFanSetSpeed", {"percentage": percentage}, "fan")
    if domain == "cover" and percentage is not None:
        return _Operation("HassSetPosition", {"position": percentage}, "cover")
    if percentage is not None and (domain == "light" or "brightness" in text or "dim" in text):
        return _Operation("HassLightSet", {"brightness": percentage}, "light")

    if "temperature" in text or re.search(r"\btemp\b", text) or domain == "climate":
        temperature = _number_before_marker(text, ("degrees", "degree", "c", "f"))
        if temperature is None:
            temperature = percentage
        if temperature is not None:
            return _Operation(
                "HassClimateSetTemperature",
                {"temperature": temperature},
                "climate",
            )

    for color in _COLORS:
        if color.startswith("light ") and not re.search(
            rf"\b(?:to|color|colour)\s+{re.escape(color)}\b", text
        ):
            continue
        if _contains_phrase(text, color) and (
            domain == "light"
            or "color" in text
            or "colour" in text
            or re.search(r"\b(?:set|change|make|flick|flip|turn)\b", text)
        ):
            if re.search(rf"\boff\b.*\b{re.escape(color)}\b", text):
                continue
            return _Operation("HassLightSet", {"color": color}, "light")

    if "volume" in text and re.search(r"\b(?:up|increase|louder)\b", text):
        return _Operation("HassSetVolumeRelative", {"volume_step": "up"}, "media_player")
    if "volume" in text and re.search(r"\b(?:down|decrease|quieter)\b", text):
        return _Operation("HassSetVolumeRelative", {"volume_step": "down"}, "media_player")
    if re.search(r"\bunmute\b|\b(?:restore|enable)\s+(?:the\s+)?(?:audio|sound)\b", text):
        return _Operation("HassMediaPlayerUnmute", {}, "media_player")
    if domain == "media_player" and re.search(
        r"\bmute\s+off\b|\b(?:sound|volume)\b.*\b(?:back\s+on|on\s+again)\b",
        text,
    ):
        return _Operation("HassMediaPlayerUnmute", {}, "media_player")
    if domain == "media_player" and re.search(
        r"(?:\b(?:sound|volume)\b.*\boff\b|\boff\b.*\b(?:sound|volume)\b)",
        text,
    ):
        return _Operation("HassMediaPlayerMute", {}, "media_player")
    if re.search(r"\bmute\b", text):
        return _Operation("HassMediaPlayerMute", {}, "media_player")
    if re.search(r"\b(?:next|skip)\b", text):
        return _Operation("HassMediaNext", {}, "media_player")
    if re.search(r"\b(?:previous|last track|go back)\b", text):
        return _Operation("HassMediaPrevious", {}, "media_player")
    if re.search(r"\b(?:resume|unpause|continue)\b", text) or (
        domain == "media_player"
        and re.search(r"\b(?:back\s+on|start\s+again|play\s+again|restart)\b", text)
    ):
        return _Operation("HassMediaUnpause", {}, "media_player")
    if re.search(r"\b(?:pause|hold)\b", text) or (
        domain == "media_player" and re.search(r"\bstop\b", text)
    ):
        return _Operation("HassMediaPause", {}, "media_player")
    if re.search(r"\b(?:quiet|silence)\b", text):
        return _Operation("HassMediaPlayerMute", {}, "media_player")
    if re.search(r"\bplay\b", text) and domain == "media_player":
        return _Operation("HassMediaUnpause", {}, "media_player")

    if domain == "lock" and re.search(r"\b(?:open|opened|opening)\b", text):
        return _Operation("HassTurnOff", {}, "lock")
    if domain == "lock" and re.search(r"\b(?:close|closed|closing|shut)\b", text):
        return _Operation("HassTurnOn", {}, "lock")
    if domain not in {"scene", "script"} and re.search(r"\b(?:unlock|unlocked)\b", text):
        return _Operation("HassTurnOff", {}, "lock")
    if domain not in {"scene", "script"} and re.search(r"\b(?:lock|locked|secure)\b", text):
        return _Operation("HassTurnOn", {}, "lock")
    if domain == "vacuum" and re.search(
        r"\b(?:turn|switch|power|activate|run|start)\b.*\bon\b|\b(?:run|start)\b",
        text,
    ):
        return _Operation("HassVacuumStart", {}, "vacuum")
    if domain == "vacuum" and re.search(r"\b(?:dock|return)(?:\s+to\s+(?:base|dock))?\b", text):
        return _Operation("HassVacuumReturnToBase", {}, "vacuum")
    if re.search(r"\b(?:turn|switch|power|flip|flick)\b.*\boff\b|\boff\b$", text):
        return _Operation("HassTurnOff", {}, domain)
    if re.search(r"\b(?:turn|switch|power|flip|flick)\b.*\bon\b|\bon\b$", text):
        return _Operation("HassTurnOn", {}, domain)
    if domain and re.search(r"\boff\b", text):
        return _Operation("HassTurnOff", {}, domain)
    if domain and re.search(r"\bon\b", text):
        return _Operation("HassTurnOn", {}, domain)
    if re.search(r"\b(?:enable|enabled|power up)\b", text):
        return _Operation("HassTurnOn", {}, domain)
    if re.search(r"\b(?:disable|disabled|power down)\b", text):
        return _Operation("HassTurnOff", {}, domain)
    if re.search(r"\b(?:close|closed|closing|shut|lower)\b", text):
        if domain == "lock":
            return _Operation("HassTurnOn", {}, domain)
        return _Operation("HassTurnOff", {}, "cover" if domain == "cover" else domain)
    if re.search(r"\b(?:open|opened|opening|raise|lift)\b", text):
        if domain == "lock":
            return _Operation("HassTurnOff", {}, domain)
        return _Operation("HassTurnOn", {}, "cover" if domain == "cover" else domain)
    if re.search(r"\b(?:activate|run|start|launch)\b", text):
        return _Operation("HassTurnOn", {}, domain)
    if re.search(r"\b(?:engage)\b", text):
        return _Operation("HassTurnOn", {}, domain)
    if re.search(r"\billuminate\b", text):
        return _Operation("HassTurnOn", {}, "light")
    if re.search(r"\blight up\b", text):
        return _Operation("HassTurnOn", {}, "light")
    if re.search(r"\bspin up\b", text):
        return _Operation("HassTurnOn", {}, domain or "fan")
    if re.search(r"\bspin down\b", text):
        return _Operation("HassTurnOff", {}, domain)
    if domain == "cover" and re.search(r"\b(?:pull|roll)\b.*\bup\b", text):
        return _Operation("HassTurnOn", {}, domain or "cover")
    if domain == "cover" and re.search(r"\b(?:pull|roll)\b.*\bdown\b|\bdraw\b.*\bclosed\b", text):
        return _Operation("HassTurnOff", {}, domain or "cover")
    if domain == "cover" and re.search(r"\b(?:flip|flick|move|push)\b.*\bdown\b", text):
        return _Operation("HassTurnOff", {}, "cover")
    if domain == "cover" and re.search(r"\b(?:flip|flick|move|push)\b.*\bup\b", text):
        return _Operation("HassTurnOn", {}, "cover")
    if re.search(r"\b(?:deactivate|stop|kill)\b", text):
        return _Operation("HassTurnOff", {}, domain)
    if re.search(r"\b(?:disengage|release)\b", text):
        return _Operation("HassTurnOff", {}, domain)
    if domain in {"scene", "script"} and re.search(r"\b(?:set|execute|trigger)\b", text):
        return _Operation("HassTurnOn", {}, domain)
    return None


def _domain_compatible(entity: CatalogEntity, domain: str | None) -> bool:
    if domain is None or entity.domain == domain:
        return True
    if domain == "fan" and entity.domain == "switch":
        return "fan" in _normal(entity.name)
    if domain == "sensor" and entity.domain == "binary_sensor":
        return True
    return False


def _area_phrases(area: CatalogArea) -> tuple[str, ...]:
    phrases = {area.name, area.area_id, *area.aliases}
    name = _normal(area.name)
    if name.endswith(" room"):
        phrases.add(name.removesuffix(" room"))
    else:
        phrases.add(f"{name} room")
    common_aliases = {
        "backyard": ("back yard", "outside", "outdoors", "out back", "yard"),
        "entryway": (
            "entry",
            "entrance",
            "foyer",
            "front hall",
            "by the door",
            "near the door",
            "doorway",
        ),
        "entry": (
            "entryway",
            "entrance",
            "foyer",
            "front hall",
            "by the door",
            "near the door",
            "doorway",
        ),
        "bathroom": ("washroom", "restroom", "loo"),
        "laundry": ("laundry room", "utility room"),
        "living room": ("lounge",),
    }
    phrases.update(common_aliases.get(name, ()))
    return tuple(phrases)


def _floor_phrases(floor: CatalogFloor) -> tuple[str, ...]:
    return (floor.name, floor.floor_id, *floor.aliases)


def _mentioned_areas(text: str, catalog: CatalogSnapshot) -> tuple[CatalogArea, ...]:
    return tuple(
        area
        for area in catalog.areas
        if any(_contains_phrase(text, phrase) for phrase in _area_phrases(area))
    )


def _mentioned_floors(text: str, catalog: CatalogSnapshot) -> tuple[CatalogFloor, ...]:
    return tuple(
        floor
        for floor in catalog.floors
        if any(_contains_phrase(text, phrase) for phrase in _floor_phrases(floor))
    )


def _entity_phrases(entity: CatalogEntity, catalog: CatalogSnapshot) -> tuple[str, ...]:
    phrases = {entity.name, entity.entity_id, entity.entity_id.split(".", 1)[-1], *entity.aliases}
    area = next((area for area in catalog.areas if area.area_id == entity.area_id), None)
    if area is not None:
        phrases.add(f"{area.name} {entity.name}")
    return tuple(sorted((_normal(phrase) for phrase in phrases if phrase), key=len, reverse=True))


def _distinctive_tokens(entity: CatalogEntity, catalog: CatalogSnapshot) -> frozenset[str]:
    tokens = {
        token for phrase in (entity.name, *entity.aliases) for token in _normal(phrase).split()
    }
    area = next((area for area in catalog.areas if area.area_id == entity.area_id), None)
    if area is not None:
        for phrase in _area_phrases(area):
            tokens.difference_update(_normal(phrase).split())
    tokens.difference_update(
        {
            "the",
            "room",
            "lights",
            "lighting",
            "illumination",
            "fan",
            "fans",
            "cover",
            "covers",
            "window",
            "windows",
            "light",
            "lamp",
            "lock",
            "locks",
            "switch",
            "switches",
            "sensor",
            "sensors",
            *entity.domain.split("_"),
        }
    )
    return frozenset(
        token for token in tokens if len(token) >= 2 and token not in _GENERIC_ENTITY_MATCH_TOKENS
    )


def _has_distinctive_mention(
    text: str,
    entity: CatalogEntity,
    catalog: CatalogSnapshot,
) -> bool:
    return any(_contains_phrase(text, token) for token in _distinctive_tokens(entity, catalog))


def _fuzzy_phrase(text: str, phrase: str) -> bool:
    phrase_words = phrase.split()
    if len("".join(phrase_words)) < 7:
        return False
    words = text.split()
    for size in range(max(1, len(phrase_words) - 1), len(phrase_words) + 2):
        for index in range(len(words) - size + 1):
            candidate = " ".join(words[index : index + size])
            if SequenceMatcher(None, phrase, candidate).ratio() >= 0.9:
                return True
    return False


LOGGER = logging.getLogger(__name__)


def _describe_entity(entity: CatalogEntity) -> str:
    return f"{entity.entity_id} ({entity.name})"


def _format_targets(targets: Sequence[_Target]) -> str:
    return "; ".join(
        f"{', '.join(target.entity_ids)} [{', '.join(f'{name}={value}' for name, value in target.slots.items())}]"
        for target in targets
    )


def _named_entities(
    text: str,
    catalog: CatalogSnapshot,
    domain: str | None,
    areas: Sequence[CatalogArea],
    ignored_entity_domains: frozenset[str],
) -> tuple[tuple[CatalogEntity, bool], ...]:
    area_ids = {area.area_id for area in areas}
    candidates: list[tuple[CatalogEntity, tuple[str, ...]]] = []
    for entity in catalog.entities:
        if entity.domain.casefold() in ignored_entity_domains:
            continue
        if not _domain_compatible(entity, domain):
            continue
        if area_ids and entity.area_id not in area_ids:
            continue
        phrases = _entity_phrases(entity, catalog)
        candidates.append((entity, phrases))

    hits = [
        (entity, phrase)
        for entity, phrases in candidates
        if (
            phrase := next(
                (
                    item
                    for item in phrases
                    if not _is_generic_entity_phrase(item) and _contains_phrase(text, item)
                ),
                None,
            )
        )
        is not None
    ]
    if not hits:
        hits = [
            (entity, token)
            for entity, _ in candidates
            if (
                token := next(
                    (
                        item
                        for item in sorted(
                            _distinctive_tokens(entity, catalog), key=len, reverse=True
                        )
                        if _contains_phrase(text, item)
                    ),
                    None,
                )
            )
            is not None
        ]
    if not hits:
        hits = [
            (entity, phrase)
            for entity, phrases in candidates
            if (phrase := next((item for item in phrases if _fuzzy_phrase(text, item)), None))
            is not None
        ]
    if not hits:
        return ()

    hits = _prefer_power_switch_hits(text, hits)

    by_phrase: dict[str, list[CatalogEntity]] = {}
    for entity, phrase in hits:
        by_phrase.setdefault(phrase, []).append(entity)
    ambiguous_phrases = [
        (phrase, entities) for phrase, entities in by_phrase.items() if len(entities) > 1
    ]
    if ambiguous_phrases:
        LOGGER.info(
            "Ambiguous target phrases %s match multiple devices: %s",
            ", ".join(repr(phrase) for phrase, _ in ambiguous_phrases),
            "; ".join(
                f"{phrase} -> [{', '.join(_describe_entity(entity) for entity in entities)}]"
                for phrase, entities in ambiguous_phrases
            ),
        )
        # Instead of raising, return the ambiguous hits so they can be processed
        # by the caller to determine if they can be resolved or if clarification is needed.
        return tuple((entity, True) for _, entities in ambiguous_phrases for entity in entities)

    # A shorter name embedded in a longer matched name is not a second target.
    selected: list[tuple[CatalogEntity, str]] = []
    for entity, phrase in sorted(hits, key=lambda item: len(item[1]), reverse=True):
        if any(phrase in existing_phrase for _, existing_phrase in selected):
            continue
        selected.append((entity, phrase))
    return tuple((entity, False) for entity, _ in selected)


def _target_evidence(
    text: str,
    catalog: CatalogSnapshot,
    ignored_entity_domains: frozenset[str],
) -> bool:
    """Return whether a conjunct contains an independently meaningful target."""

    if _explicit_domain(text) or _mentioned_areas(text, catalog) or _mentioned_floors(text, catalog):
        return True
    return bool(
        _named_entities(text, catalog, None, (), ignored_entity_domains)
    )


def _target_set(
    text: str,
    catalog: CatalogSnapshot,
    ignored_entity_domains: frozenset[str],
) -> _TargetSet:
    """Parse target coordination without mistaking ordinary ``and`` for a target list."""

    parts = tuple(
        part.strip(" ,")
        for part in re.split(r"\s*(?:,\s*(?:and\s+)?|\b(?:and|plus|as well as)\b)\s*", text)
        if part.strip(" ,")
    )
    if len(parts) >= 2 and all(
        _target_evidence(part, catalog, ignored_entity_domains) for part in parts
    ):
        return _TargetSet("conjunction", tuple(_TargetMember(part) for part in parts))
    return _TargetSet("single", (_TargetMember(text),))


def _context_entity_ids(
    context: Mapping[str, object] | None,
    catalog: CatalogSnapshot,
) -> tuple[str, ...]:
    if not context:
        return ()
    known = {entity.entity_id for entity in catalog.entities}
    for key in ("target_entity_ids", "entity_ids", "last_entity_ids"):
        value = context.get(key)
        values = (value,) if isinstance(value, str) else value
        if isinstance(values, Sequence):
            matched = tuple(item for item in values if isinstance(item, str) and item in known)
            if matched:
                return matched
    return ()


def _contextual_domain(
    text: str,
    context: Mapping[str, object] | None,
    catalog: CatalogSnapshot,
    previous_targets: Sequence[_Target],
) -> str | None:
    is_contextual = bool(
        _PRONOUN_RE.search(text)
        or re.match(r"^(?:set|adjust|change|flick|flip)\b(?:\s+(?:it|them))?\s+to\b", text)
    )
    if not is_contextual:
        return None
    entity_ids = tuple(
        dict.fromkeys(entity_id for target in previous_targets for entity_id in target.entity_ids)
    ) or _context_entity_ids(context, catalog)
    by_id = {entity.entity_id: entity for entity in catalog.entities}
    domains = {by_id[entity_id].domain for entity_id in entity_ids if entity_id in by_id}
    return next(iter(domains)) if len(domains) == 1 else None


def _named_domain_hint(
    text: str,
    catalog: CatalogSnapshot,
    ignored_entity_domains: frozenset[str],
) -> str | None:
    # Area-scoped temperature wording without an explicit sensor noun refers
    # to the area's temperature-control capability when that target is unique.
    # This keeps ``living room temperature`` usable for a later ``set it``
    # follow-up while preserving explicit ``temperature sensor`` queries.
    if re.search(r"\b(?:temperature|temp)\b", text) and not re.search(
        r"\bsensors?\b", text
    ):
        area_ids = {area.area_id for area in _mentioned_areas(text, catalog)}
        climates = tuple(
            entity
            for entity in catalog.entities
            if entity.domain == "climate" and entity.area_id in area_ids
        )
        if len(climates) == 1:
            return "climate"
    try:
        entities = _named_entities(
            text,
            catalog,
            None,
            _mentioned_areas(text, catalog),
            ignored_entity_domains,
        )
    except _AmbiguousTarget:
        return None
    domains = {entity.domain for entity, _ in entities}
    if len(domains) == 1:
        return next(iter(domains))
    if _is_on_off_action(text) and "switch" in domains:
        return "switch"
    return None


def _elliptical_query(
    text: str,
    catalog: CatalogSnapshot,
    ignored_entity_domains: frozenset[str],
) -> _Operation | None:
    """Treat a unique topology label with no operation as a read-only request."""

    try:
        entities = _named_entities(
            text,
            catalog,
            None,
            _mentioned_areas(text, catalog),
            ignored_entity_domains,
        )
    except _AmbiguousTarget:
        return None
    if len(entities) == 1:
        entity, ambiguous = entities[0]
        if not ambiguous and _has_distinctive_mention(text, entity, catalog):
            return _Operation("HassGetState", {}, entity.domain)

    # A bare area-plus-domain phrase can still be an unambiguous elliptical
    # query even when it contains no device name (``living room climate``).
    # Resolve it only when topology leaves exactly one compatible entity.
    domain = _explicit_domain(text) or _semantic_domain(text)
    areas = _mentioned_areas(text, catalog)
    floors = _mentioned_floors(text, catalog)
    if domain and (areas or floors):
        area_ids = {area.area_id for area in areas}
        if floors:
            floor_ids = {floor.floor_id for floor in floors}
            area_ids.update(area.area_id for area in catalog.areas if area.floor_id in floor_ids)
        compatible = tuple(
            entity
            for entity in catalog.entities
            if entity.domain.casefold() not in ignored_entity_domains
            and _domain_compatible(entity, domain)
            and entity.area_id in area_ids
        )
        if len(compatible) == 1:
            intent = (
                "HassClimateGetTemperature"
                if domain == "climate" and re.search(r"\b(?:climate|temperature|temp)\b", text)
                else "HassGetState"
            )
            return _Operation(intent, {}, domain)
    return None


def _origin_area(
    context: Mapping[str, object] | None,
    catalog: CatalogSnapshot,
) -> CatalogArea | None:
    if not context:
        return None
    area_id = context.get("area_id")
    if isinstance(area_id, str):
        area = next((item for item in catalog.areas if item.area_id == area_id), None)
        if area is not None:
            return area
    area_name = _normal(context.get("area_name"))
    matches = [area for area in catalog.areas if _normal(area.name) == area_name]
    return matches[0] if len(matches) == 1 else None


def _is_raw_entity_name(entity: CatalogEntity) -> bool:
    name = entity.name.strip()
    if not name:
        return False
    tail = entity.entity_id.split(".", 1)[-1]
    return _normal(name) == _normal(tail)


def _target_for_entity(entity: CatalogEntity) -> _Target:
    slots: dict[str, Any] = {"name": entity.name}
    if _is_raw_entity_name(entity):
        slots["entity_id"] = entity.entity_id
    return _Target(slots, (entity.entity_id,))


def _group_target(
    *,
    catalog: CatalogSnapshot,
    domain: str,
    area: CatalogArea | None = None,
    floor: CatalogFloor | None = None,
    cover_subtype: str | None = None,
) -> _Target | None:
    entities = [entity for entity in catalog.entities if _domain_compatible(entity, domain)]
    if domain == "cover":
        entities = [entity for entity in entities if _matches_cover_subtype(entity, cover_subtype)]
    slots: dict[str, Any] = {"domain": domain}
    if area is not None:
        entities = [entity for entity in entities if entity.area_id == area.area_id]
        slots["area"] = area.name
    elif floor is not None:
        area_ids = {item.area_id for item in catalog.areas if item.floor_id == floor.floor_id}
        entities = [entity for entity in entities if entity.area_id in area_ids]
        slots["floor"] = floor.name
    excluded_entities: list[CatalogEntity] = []
    if domain == "light":
        excluded_entities = [entity for entity in entities if entity.is_indicator]
        entities = [entity for entity in entities if not entity.is_indicator]
    return (
        _Target(
            slots,
            tuple(sorted(entity.entity_id for entity in entities)),
            tuple(sorted(entity.entity_id for entity in excluded_entities)),
        )
        if entities
        else None
    )


def singular_generic_target(text: str, domain: str) -> bool:
    forms = {
        "cover": ("blind", "curtain", "cover", "shade", "shutter"),
        "fan": ("fan",),
        "light": ("light", "lamp"),
        "lock": ("lock", "deadbolt"),
        "media_player": ("television", "tv", "speaker", "media player"),
        "sensor": ("sensor",),
        "switch": ("switch",),
    }.get(domain, ())
    if re.search(r"\b(?:all|both|every|each|whole|entire)\b", text):
        return False
    return any(
        re.search(rf"\b{re.escape(form)}\b", text)
        and not re.search(rf"\b{re.escape(form)}s\b", text)
        for form in forms
    )


_DOMAIN_POLYMORPHIC_INTENTS = frozenset({"HassGetState", "HassTurnOff", "HassTurnOn"})


def _resolve_target_member(
    member: _TargetMember,
    operation: _Operation,
    catalog: CatalogSnapshot,
    shared_areas: Sequence[CatalogArea],
    shared_floors: Sequence[CatalogFloor],
    ignored_entity_domains: frozenset[str],
) -> tuple[_Target, ...]:
    """Resolve one conjunct without inheriting another conjunct's domain."""

    text = member.text
    member_areas = _mentioned_areas(text, catalog)
    member_floors = _mentioned_floors(text, catalog)

    # Entity nouns constrain a member more strongly than the shared property's
    # words (``TV brightness`` still names a media player target).
    mentioned_domain = _explicit_domain(text) or _semantic_domain(text)
    polymorphic = operation.intent_name in _DOMAIN_POLYMORPHIC_INTENTS
    if not polymorphic and operation.domain and mentioned_domain:
        # A shared property setter is valid only when every target exposes the
        # capability represented by the operation's domain.
        if mentioned_domain != operation.domain:
            raise _CapabilityMismatch(
                f"{mentioned_domain} does not support {operation.intent_name}"
            )
    domain = (
        mentioned_domain
        if polymorphic and mentioned_domain is not None
        else operation.domain or mentioned_domain
    )
    cover_subtype = _cover_subtype(text) if domain == "cover" else None

    # A polymorphic power/state predicate must not use the domain inferred
    # from a different conjunct to filter an explicitly named member. Resolve
    # unrestricted names first; retain the operation domain only as the
    # fallback for generic area/floor phrases such as ``kitchen lights``.
    name_domain = None if polymorphic and mentioned_domain is None else domain
    named = _named_entities(
        text,
        catalog,
        name_domain,
        member_areas,
        ignored_entity_domains,
    )
    if any(ambiguous for _, ambiguous in named):
        raise _AmbiguousTarget
    if named:
        return tuple(_target_for_entity(entity) for entity, _ in named)

    # A single area/floor named elsewhere in the conjunction may modify an
    # elliptical member (``bathroom exhaust fan and mirror light``). Apply it
    # only after exact unscoped name resolution, so it cannot override a name
    # such as ``bedside lamp`` merely because the other member names an area.
    if not member_areas and len(shared_areas) == 1:
        member_areas = tuple(shared_areas)
    if not member_floors and not member_areas and len(shared_floors) == 1:
        member_floors = tuple(shared_floors)
    if member_areas or member_floors:
        named = _named_entities(
            text,
            catalog,
            name_domain,
            member_areas,
            ignored_entity_domains,
        )
        if any(ambiguous for _, ambiguous in named):
            raise _AmbiguousTarget
        if named:
            return tuple(_target_for_entity(entity) for entity, _ in named)

    if domain is None:
        return ()
    if member_areas:
        return tuple(
            target
            for area in member_areas
            if (
                target := _group_target(
                    catalog=catalog,
                    domain=domain,
                    area=area,
                    cover_subtype=cover_subtype,
                )
            )
            is not None
        )
    if member_floors:
        return tuple(
            target
            for floor in member_floors
            if (
                target := _group_target(
                    catalog=catalog,
                    domain=domain,
                    floor=floor,
                    cover_subtype=cover_subtype,
                )
            )
            is not None
        )
    return ()


def _resolve_target_set(
    target_set: _TargetSet,
    operation: _Operation,
    catalog: CatalogSnapshot,
    ignored_entity_domains: frozenset[str],
) -> tuple[_Target, ...]:
    """Resolve every conjunction member, rejecting partial target sets."""

    if target_set.relation != "conjunction":
        return ()
    full_text = " and ".join(member.text for member in target_set.members)
    shared_areas = _mentioned_areas(full_text, catalog)
    shared_floors = _mentioned_floors(full_text, catalog)
    targets: list[_Target] = []
    for member in target_set.members:
        resolved = _resolve_target_member(
            member,
            operation,
            catalog,
            shared_areas,
            shared_floors,
            ignored_entity_domains,
        )
        if not resolved:
            raise _AmbiguousTarget
        targets.extend(resolved)
    unique: list[_Target] = []
    seen: set[tuple[str, ...]] = set()
    for target in targets:
        if target.entity_ids not in seen:
            unique.append(target)
            seen.add(target.entity_ids)
    return tuple(unique)


def _resolve_targets(
    text: str,
    operation: _Operation,
    catalog: CatalogSnapshot,
    context: Mapping[str, object] | None,
    previous_targets: tuple[_Target, ...],
    ignored_entity_domains: frozenset[str],
) -> tuple[_Target, ...]:
    # HassGetWeather is deliberately location-level. Supplying an arbitrary
    # weather entity name can make HA reject an otherwise valid query, while
    # asking the user to choose among weather providers is rarely useful.
    if operation.intent_name == "HassGetWeather":
        weather_ids = tuple(
            entity.entity_id for entity in catalog.entities if entity.domain == "weather"
        )
        return (_Target({}, weather_ids),)

    areas = _mentioned_areas(text, catalog)
    floors = _mentioned_floors(text, catalog)
    domain = operation.domain or _explicit_domain(text)
    cover_subtype = _cover_subtype(text) if domain == "cover" else None
    LOGGER.info(
        "Resolving targets: text=%r domain=%r areas=%s floors=%s cover_subtype=%r previous_targets=%s",
        text,
        domain,
        [area.area_id for area in areas],
        [floor.floor_id for floor in floors],
        cover_subtype,
        _format_targets(previous_targets),
    )
    # Coordination is represented before catalog matching. Each member is
    # resolved independently, then the shared predicate is distributed over
    # the complete set. This prevents a domain inferred from the first noun
    # phrase from filtering out later heterogeneous members.
    target_set = _target_set(text, catalog, ignored_entity_domains)
    coordinated_targets = _resolve_target_set(
        target_set,
        operation,
        catalog,
        ignored_entity_domains,
    )
    if coordinated_targets:
        LOGGER.info("Coordinated target set resolved: %s", _format_targets(coordinated_targets))
        return coordinated_targets

    entity_hits = _named_entities(
        text, catalog, domain, areas, ignored_entity_domains
    )
    if entity_hits:
        unambiguous_entities = tuple(entity for entity, ambiguous in entity_hits if not ambiguous)
        ambiguous_entities = tuple(entity for entity, ambiguous in entity_hits if ambiguous)

        if ambiguous_entities and not (areas and domain):
            # Partial-name ambiguity is genuine only when topology cannot
            # resolve the phrase as an explicit area/floor group.  For example,
            # ``kitchen switches`` may partially overlap the alias ``switch
            # coffee maker`` while still denoting the one complete switch group
            # in that area.
            LOGGER.info(
                "TARGET RESOLUTION ambiguous reason=shared_name_without_topology "
                "text=%r domain=%r candidates=%s",
                text,
                domain,
                tuple(_describe_entity(entity) for entity in ambiguous_entities),
            )
            raise _AmbiguousTarget

        if unambiguous_entities:
            LOGGER.info(
                "Named entity candidate matches: %s",
                ", ".join(_describe_entity(entity) for entity in unambiguous_entities),
            )
            if areas and domain:
                distinct_entities = tuple(
                    entity
                    for entity in unambiguous_entities
                    if _has_distinctive_mention(text, entity, catalog)
                )
                if distinct_entities:
                    targets = [_target_for_entity(entity) for entity in distinct_entities]
                    covered_area_ids = {entity.area_id for entity in distinct_entities}
                    targets.extend(
                        target
                        for area in areas
                        if area.area_id not in covered_area_ids
                        if (
                            target := _group_target(
                                catalog=catalog,
                                domain=domain,
                                area=area,
                                cover_subtype=cover_subtype,
                            )
                        )
                        is not None
                    )
                    LOGGER.info(
                        "Distinct entity targets and area groups resolved: %s",
                        _format_targets(targets),
                    )
                    return tuple(targets)
                LOGGER.info(
                    "Entities matched but no distinctive mention found in area context; treating area-level group targets as needed.",
                )
                # A label made only from an area and a collective domain word denotes
                # the whole group, not whichever entity happens to have that label.
            targets = tuple(_target_for_entity(entity) for entity in unambiguous_entities)
            LOGGER.info("Entity targets resolved: %s", _format_targets(targets))
            return targets

    if _PRONOUN_RE.search(text):
        if previous_targets:
            LOGGER.info(
                "Pronoun reference resolved to previous targets: %s",
                _format_targets(previous_targets),
            )
            return previous_targets
        referenced = _context_entity_ids(context, catalog)
        if referenced:
            by_id = {entity.entity_id: entity for entity in catalog.entities}
            targets = tuple(_target_for_entity(by_id[entity_id]) for entity_id in referenced)
            LOGGER.info(
                "Pronoun reference resolved to context entities: %s",
                _format_targets(targets),
            )
            return targets

    if domain is None:
        # This case is for when no entities are found and no domain is explicitly mentioned.
        # This is still an ambiguous situation, so we raise _AmbiguousTarget.
        LOGGER.info(
            "TARGET RESOLUTION ambiguous reason=no_domain_or_named_entity text=%r",
            text,
        )
        raise _AmbiguousTarget
    if areas:
        targets = tuple(
            target
            for area in areas
            if (
                target := _group_target(
                    catalog=catalog,
                    domain=domain,
                    area=area,
                    cover_subtype=cover_subtype,
                )
            )
            is not None
        )
        if (
            len(targets) == 1
            and len(targets[0].entity_ids) > 1
            and singular_generic_target(text, domain)
        ):
            LOGGER.info(
                "Singular generic target has multiple area candidates: %s",
                _format_targets(targets),
            )
            raise _AmbiguousTarget
        LOGGER.info("Area group targets resolved: %s", _format_targets(targets))
        return targets
    if floors:
        targets = tuple(
            target
            for floor in floors
            if (
                target := _group_target(
                    catalog=catalog,
                    domain=domain,
                    floor=floor,
                    cover_subtype=cover_subtype,
                )
            )
            is not None
        )
        LOGGER.info("Floor group targets resolved: %s", _format_targets(targets))
        return targets
    origin_area = _origin_area(context, catalog)
    if origin_area is not None:
        target = _group_target(
            catalog=catalog,
            domain=domain,
            area=origin_area,
            cover_subtype=cover_subtype,
        )
        if target is not None:
            LOGGER.info(
                "Origin area group target resolved: %s",
                _format_targets((target,)),
            )
            return (target,)
        LOGGER.info("Origin area %s has no matching group target", origin_area.area_id)
        return ()
    if re.search(r"\b(?:all|every|everywhere|whole house|entire house)\b", text):
        target = _group_target(catalog=catalog, domain=domain, cover_subtype=cover_subtype)
        if target is not None:
            LOGGER.info("Whole-house group target resolved: %s", _format_targets((target,)))
            return (target,)
        LOGGER.info("Whole-house target matched no entities for domain %r", domain)
        return ()
    compatible = tuple(
        entity
        for entity in catalog.entities
        if _domain_compatible(entity, domain)
        and (domain != "cover" or _matches_cover_subtype(entity, cover_subtype))
    )
    if len(compatible) == 1:
        target = _target_for_entity(compatible[0])
        LOGGER.info("Single compatible entity fallback resolved: %s", _format_targets((target,)))
        return (target,)
    if compatible:
        LOGGER.info(
            "Multiple compatible entities exist for domain %r; leaving targets ambiguous: %s",
            domain,
            ", ".join(_describe_entity(entity) for entity in compatible),
        )
    else:
        LOGGER.info("No compatible entities found for domain %r", domain)
    return ()


class NaturalLanguageIntentPlanner:
    """Create conservative OHF-compatible plans from flexible English commands."""

    def __init__(
        self,
        *,
        ambiguity_response: str = (
            "I found more than one possible target. Please be more specific."
        ),
        ignored_entity_domains: tuple[str, ...] | None = None,
    ) -> None:
        self._ambiguity_response = ambiguity_response
        self._measurement_planner = MeasurementIntentPlanner(ambiguity_response=ambiguity_response)
        self._ignored_entity_domains = (
            frozenset(
                domain.casefold()
                for domain in (
                    ignored_entity_domains
                    if ignored_entity_domains is not None
                    else settings.home_assistant.ignored_entity_domains
                )
                if domain
            )
            or DEFAULT_IGNORED_ENTITY_DOMAINS
        )

    def plan(
        self,
        text: str,
        catalog: CatalogSnapshot,
        origin_context: Mapping[str, object] | None = None,
        *,
        _typed: bool = False,
    ) -> IntentPlan:
        try:
            measurement_plan = self._measurement_planner.plan(
                text,
                catalog,
                origin_context,
            )
        except RouteDeclined:
            pass
        else:
            return measurement_plan

        clauses = split_compound_request(text)
        if not clauses:
            raise RouteDeclined("The request was empty")

        steps: list[PlannedIntent] = []
        previous_targets: tuple[_Target, ...] = ()
        for clause in clauses:
            contextual_domain = _contextual_domain(
                clause,
                origin_context,
                catalog,
                previous_targets,
            )
            operation = _classify(
                clause,
                contextual_domain
                or _named_domain_hint(
                    clause,
                    catalog,
                    self._ignored_entity_domains,
                ),
            )
            if operation is None:
                # An imperative predicate that is unsupported for its named
                # target is not an elliptical request to read that target.
                # Declining allows another strategy/fallback to handle it.
                if not re.match(r"^restart\b", clause):
                    operation = _elliptical_query(
                        clause,
                        catalog,
                        self._ignored_entity_domains,
                    )
            if operation is None:
                raise RouteDeclined(f"No deterministic operation matched: {clause}")
            try:
                targets = _resolve_targets(
                    clause,
                    operation,
                    catalog,
                    origin_context,
                    previous_targets,
                    self._ignored_entity_domains,
                )
            except _AmbiguousTarget:
                LOGGER.info(
                    "NATURAL PLAN ambiguous clause=%r intent=%s domain=%r",
                    clause,
                    operation.intent_name,
                    operation.domain,
                )
                return IntentPlan(response=self._ambiguity_response)
            except _CapabilityMismatch as exc:
                if _typed:
                    raise RouteDeclined(f"Capability mismatch: {exc}") from exc
                return IntentPlan(response=self._ambiguity_response)
            if not targets:
                LOGGER.info(
                    "NATURAL PLAN no_target clause=%r intent=%s domain=%r typed=%s",
                    clause,
                    operation.intent_name,
                    operation.domain,
                    _typed,
                )
                if _typed:
                    raise RouteDeclined(
                        f"No target resolved for deterministic operation: {clause}"
                    )
                return IntentPlan(response=self._ambiguity_response)

            for target in targets:
                if target.excluded_entity_ids:
                    by_id = {entity.entity_id: entity for entity in catalog.entities}
                    LOGGER.info(
                        "NATURAL PLAN materialized filtered light group clause=%r "
                        "entity_ids=%s excluded=%s",
                        clause,
                        target.entity_ids,
                        target.excluded_entity_ids,
                    )
                    for entity_id in target.entity_ids:
                        entity = by_id[entity_id]
                        data = {
                            "name": entity.name,
                            "entity_id": entity.entity_id,
                            **operation.slots,
                        }
                        if operation.intent_name in {
                            "HassGetState",
                            "HassClimateGetTemperature",
                        }:
                            data.pop("entity_id", None)
                        call = OhfIntentCall(operation.intent_name, data)
                        steps.append(
                            PlannedIntent(
                                call=call,
                                entity_ids=(entity_id,),
                                effect=operation.requested_effect
                                or semantic_effect_for_call(call),
                            )
                        )
                    continue
                data = {**target.slots, **operation.slots}
                # HA's query intents resolve by their supported name/domain
                # slots; entity_id is useful for exact action targets but is
                # not part of the official HassGetState query contract.
                if operation.intent_name in {"HassGetState", "HassClimateGetTemperature"}:
                    data.pop("entity_id", None)
                call = OhfIntentCall(operation.intent_name, data)
                steps.append(
                    PlannedIntent(
                        call=call,
                        entity_ids=target.entity_ids,
                        effect=operation.requested_effect or semantic_effect_for_call(call),
                    )
                )
            previous_targets = targets
        return IntentPlan(steps=tuple(steps))

    def resolve(
        self,
        text: str,
        catalog: CatalogSnapshot,
        origin_context: Mapping[str, object] | None = None,
    ) -> PlanningOutcome:
        """Return a typed result while keeping :meth:`plan` backward compatible."""

        try:
            plan = self.plan(text, catalog, origin_context, _typed=True)
        except RouteDeclined as exc:
            reason = str(exc)
            if reason.startswith("Capability mismatch:"):
                return CapabilityMismatch(reason.removeprefix("Capability mismatch:").strip())
            if reason.startswith("No target resolved"):
                return NoTarget()
            if reason.startswith("No deterministic operation matched:"):
                uncovered = reason.partition(":")[2].strip()
                clauses = split_compound_request(text)
                return (
                    IncompleteCompound((uncovered,))
                    if len(clauses) > 1
                    else UnsupportedOperation(reason)
                )
            return UnsupportedOperation(reason)
        if plan.response == self._ambiguity_response:
            return AmbiguousTarget()
        if plan.steps:
            return Resolved(plan)
        return UnsupportedOperation("Planner produced no executable result")


class NaturalLanguageIntentRecognizer:
    """IntentRecognizer-compatible view of the planner's parsed operations.

    The adapter is useful for diagnostics and single-clause composition.  Compound
    commands should use :class:`NaturalLanguageIntentPlanner` so ordered operations are
    not mistaken for alternative recognizer candidates.
    """

    def __init__(self, planner: NaturalLanguageIntentPlanner | None = None) -> None:
        self._planner = planner or NaturalLanguageIntentPlanner()

    def recognize(
        self,
        text: str,
        catalog: CatalogSnapshot,
        origin_context: Mapping[str, object] | None = None,
    ) -> tuple[IntentMatch, ...]:
        plan = self._planner.plan(text, catalog, origin_context)
        if plan.response is not None:
            return ()
        return tuple(
            IntentMatch(
                intent_name=step.call.intent_name,
                slots={
                    name: SlotValue(value=value, text=str(value))
                    for name, value in step.call.data.items()
                },
                metadata={"entity_ids": step.entity_ids},
            )
            for step in plan.steps
        )


__all__ = [
    "NaturalLanguageIntentPlanner",
    "NaturalLanguageIntentRecognizer",
    "NormalizedOperator",
    "normalize_lexical_operator",
    "singular_generic_target",
    "split_compound_request",
]
