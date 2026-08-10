"""Pure visual-state policies for voice activity indicators."""

from typing import Any

NAMED_RGB: dict[str, tuple[int, int, int]] = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "orange": (255, 165, 0),
    "purple": (128, 0, 128),
    "magenta": (255, 0, 255),
    "pink": (255, 105, 180),
    "cyan": (0, 255, 255),
    "teal": (0, 128, 128),
    "lime": (0, 255, 0),
}


def parse_indicator_rgb(raw: str) -> list[int] | None:
    raw = raw.strip()
    if not raw or raw.casefold() in {"none", "off", "disabled", "current", "preserve"}:
        return None
    normalized = raw.casefold().replace(" ", "_").replace("-", "_")
    named = NAMED_RGB.get(normalized)
    if named is not None:
        return list(named)
    if raw.startswith("#") and len(raw) == 7:
        try:
            return [int(raw[1:3], 16), int(raw[3:5], 16), int(raw[5:7], 16)]
        except ValueError:
            return None
    parts = [part.strip() for part in raw.strip("[]() ").split(",")]
    if len(parts) == 3:
        try:
            values = [int(part) for part in parts]
        except ValueError:
            return None
        if all(0 <= value <= 255 for value in values):
            return values
    return None


def find_native_effect(attributes: dict[str, Any], requested: str) -> str | None:
    requested = requested.strip()
    if not requested or requested.casefold() in {"none", "off", "disabled", "current", "preserve"}:
        return None
    if requested.casefold() == "auto":
        requested = "pulse"
    effects = attributes.get("effect_list")
    if not isinstance(effects, (list, tuple)):
        return None
    names = [str(item) for item in effects if str(item).strip()]
    wanted = requested.casefold()
    exact = next((name for name in names if name.casefold() == wanted), None)
    if exact:
        return exact
    containing = [name for name in names if wanted in name.casefold()]
    return min(containing, key=len) if containing else None


def effect_wants_software_pulse(requested: str) -> bool:
    requested = requested.strip().casefold()
    return requested == "auto" or "pulse" in requested


def find_neutral_effect(attributes: dict[str, Any]) -> str | None:
    effects = attributes.get("effect_list")
    if not isinstance(effects, (list, tuple)):
        return None
    names = [str(item) for item in effects if str(item).strip()]
    for neutral in ("none", "off", "static", "solid"):
        exact = next((name for name in names if name.casefold() == neutral), None)
        if exact:
            return exact
    return None


def light_supports_colour(attributes: dict[str, Any]) -> bool:
    modes = attributes.get("supported_color_modes")
    if not isinstance(modes, (list, tuple, set)):
        return False
    colour_modes = {"hs", "rgb", "rgbw", "rgbww", "xy"}
    return any(str(mode).casefold() in colour_modes for mode in modes)


def snapshot_restore_light_data(attributes: dict[str, Any]) -> dict[str, Any]:
    data: dict[str, Any] = {}
    brightness = attributes.get("brightness")
    if isinstance(brightness, (int, float)):
        data["brightness"] = max(1, min(255, int(brightness)))

    effect = attributes.get("effect")
    if isinstance(effect, str) and effect.strip():
        data["effect"] = effect

    color_mode = str(attributes.get("color_mode") or "").casefold()
    mode_to_key = {
        "rgb": "rgb_color",
        "rgbw": "rgbw_color",
        "rgbww": "rgbww_color",
        "hs": "hs_color",
        "xy": "xy_color",
        "color_temp": "color_temp_kelvin",
    }
    candidate_keys = [
        mode_to_key.get(color_mode),
        "rgb_color",
        "rgbw_color",
        "rgbww_color",
        "hs_color",
        "xy_color",
        "color_temp_kelvin",
    ]
    seen: set[str] = set()
    for key in candidate_keys:
        if not key or key in seen:
            continue
        seen.add(key)
        value = attributes.get(key)
        if value is None:
            continue
        if key in {"rgb_color", "rgbw_color", "rgbww_color", "hs_color", "xy_color"}:
            if isinstance(value, (list, tuple)):
                data[key] = list(value)
                break
        elif key == "color_temp_kelvin" and isinstance(value, (int, float)):
            data[key] = int(value)
            break
    return data


__all__ = [
    "effect_wants_software_pulse",
    "find_native_effect",
    "find_neutral_effect",
    "light_supports_colour",
    "parse_indicator_rgb",
    "snapshot_restore_light_data",
]
