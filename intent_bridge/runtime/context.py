"""Formatting policies for trusted time and voice-origin context."""

from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo


def runtime_context(local_timezone: str) -> str:
    try:
        tz = ZoneInfo(local_timezone)
    except Exception:
        tz = UTC
    now = datetime.now(tz)
    zone_label = local_timezone if tz is not UTC else "UTC"
    return (
        "Trusted runtime context: the current local date and time is "
        f"{now.strftime('%A %d %B %Y at %H:%M')} in {zone_label}. "
        "Use this for relative dates and direct clock/date questions."
    )


def origin_runtime_context(origin_context: dict[str, Any] | None) -> str:
    if not origin_context:
        return "Voice origin context: not supplied by the caller."
    parts: list[str] = []
    if origin_context.get("device_name"):
        parts.append(f"device={origin_context['device_name']}")
    if origin_context.get("area_name"):
        parts.append(f"area={origin_context['area_name']}")
    elif origin_context.get("area_id"):
        parts.append(f"area_id={origin_context['area_id']}")
    if origin_context.get("floor_name"):
        parts.append(f"floor={origin_context['floor_name']}")
    if not parts:
        return "Voice origin context: caller supplied no resolvable room."
    return (
        "Trusted voice origin context: "
        + ", ".join(parts)
        + ". Use the area as a soft default only for requests that omit a location; "
        "an explicit user location always overrides it."
    )


def informational_runtime_context(
    local_timezone: str,
    locale: str,
    location: str,
    origin_context: dict[str, Any] | None = None,
    *,
    home_assistant_config: dict[str, Any] | None = None,
    timezone_explicit: bool = True,
    locale_explicit: bool = True,
    location_explicit: bool = True,
) -> str:
    """Build trusted, request-current grounding for the general assistant."""
    ha_config = home_assistant_config or {}
    ha_timezone = _string_value(ha_config.get("time_zone"))
    ha_language = _string_value(ha_config.get("language"))
    ha_country = _string_value(ha_config.get("country"))

    timezone_value = local_timezone.strip()
    if not timezone_explicit and ha_timezone:
        timezone_value = ha_timezone
    locale_value = locale.strip()
    if not locale_explicit:
        locale_value = _ha_locale(ha_language, ha_country) or locale_value
    location_value = location.strip()
    if not location_explicit and ha_country:
        location_value = f"country code {ha_country.upper()}"

    parts = [runtime_context(timezone_value)]
    if locale_value or location_value:
        details = []
        if locale_value:
            details.append(f"locale={locale_value}")
        if location_value:
            details.append(f"default geographic location={location_value}")
        parts.append(
            "Trusted user locale context: "
            + ", ".join(details)
            + ". Use the geographic location as the default for genuinely local or "
            "location-relative general questions unless the user names another place."
        )

    unit_system = ha_config.get("unit_system") or ha_config.get("units")
    local_preferences = []
    if isinstance(unit_system, dict):
        for key in ("temperature", "length", "mass", "volume", "wind_speed"):
            value = _string_value(unit_system.get(key))
            if value:
                local_preferences.append(f"{key}={value}")
    currency = _string_value(ha_config.get("currency"))
    if currency:
        local_preferences.append(f"currency={currency}")
    if local_preferences:
        parts.append(
            "Trusted Home Assistant regional preferences: "
            + ", ".join(local_preferences)
            + ". Use these units and currency when relevant."
        )

    origin_area = None
    if origin_context:
        origin_area = origin_context.get("area_name") or origin_context.get("area_id")
    if origin_area:
        parts.append(
            f"Voice-origin room: {origin_area}. This is a room or Home Assistant area, "
            "not a city, region, or country; never use it as geographic location."
        )
    return "\n".join(parts)


def _string_value(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _ha_locale(language: str, country: str) -> str:
    normalized_language = language.replace("_", "-").strip()
    if not normalized_language:
        return ""
    if "-" in normalized_language or not country:
        return normalized_language
    return f"{normalized_language}-{country.upper()}"
