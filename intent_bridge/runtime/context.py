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
