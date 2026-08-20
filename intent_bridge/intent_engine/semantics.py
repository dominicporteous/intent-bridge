"""Shared deterministic language and capability vocabulary.

This module is the single home for domain words that are part of the product's
spoken-language ontology.  It intentionally does not contain entity-name
rules: those come from the live Home Assistant catalog.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

DOMAIN_ALIASES: tuple[tuple[str, str], ...] = (
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

DOMAIN_REFERENCE_WORDS: Mapping[str, frozenset[str]] = {
    "climate": frozenset({"ac", "air", "climate", "conditioner", "thermostat", "temperature"}),
    "cover": frozenset(
        {
            "blind",
            "blinds",
            "cover",
            "covers",
            "curtain",
            "curtains",
            "door",
            "doors",
            "garage door",
            "garage doors",
            "shade",
            "shades",
            "shutter",
            "shutters",
            "window",
            "windows",
        }
    ),
    "fan": frozenset({"fan", "fans", "ventilation"}),
    "light": frozenset({"lamp", "lamps", "light", "lights", "lighting", "illumination"}),
    "lock": frozenset({"deadbolt", "deadbolts", "lock", "locks"}),
    "media_player": frozenset(
        {"media player", "player", "projector", "speaker", "speakers", "stereo", "tv", "tvs"}
    ),
    # "switch" is normally an imperative verb. Entity labels still resolve
    # through catalog evidence rather than using this word as a target hint.
    "switch": frozenset(),
}

COVER_SUBTYPE_FORMS: Mapping[str, tuple[str, ...]] = {
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

COVER_DEVICE_CLASS_FORMS: Mapping[str, tuple[str, ...]] = {
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
}


def plural_domain_forms(domain: str) -> tuple[str, ...]:
    """Return plural references that retain group semantics."""

    return tuple(form for form in DOMAIN_REFERENCE_WORDS.get(domain, ()) if form.endswith("s"))


def referenced_domain(
    text: str,
    contains_phrase: Callable[[str, str], bool],
) -> str | None:
    """Return the first spoken semantic domain in conservative priority order."""

    for domain in ("light", "cover", "fan", "climate", "media_player", "lock", "switch"):
        if any(contains_phrase(text, form) for form in DOMAIN_REFERENCE_WORDS[domain]):
            return domain
    return None


__all__ = [
    "COVER_DEVICE_CLASS_FORMS",
    "COVER_SUBTYPE_FORMS",
    "DOMAIN_ALIASES",
    "DOMAIN_REFERENCE_WORDS",
    "plural_domain_forms",
    "referenced_domain",
]
