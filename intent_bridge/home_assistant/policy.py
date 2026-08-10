"""Pure Home Assistant URL and service-schema policies."""

import difflib
import re
from typing import Any
from urllib.parse import urlparse, urlunparse

EXPECTED_PRIMARY_STATES = {
    "turn_on": "on",
    "turn_off": "off",
    "open": "open",
    "close": "closed",
    "open_cover": "open",
    "close_cover": "closed",
    "lock": "locked",
    "unlock": "unlocked",
    "media_play": "playing",
    "media_pause": "paused",
    "media_stop": "idle",
}


def websocket_url(base_url: str) -> str:
    parsed = urlparse(base_url.strip())
    if parsed.scheme not in {"http", "https", "ws", "wss"}:
        raise ValueError("INTENT_BRIDGE_HA_BASE_URL must use http:// or https://")
    scheme = "wss" if parsed.scheme in {"https", "wss"} else "ws"
    return urlunparse(
        (scheme, parsed.netloc, f"{parsed.path.rstrip('/')}/api/websocket", "", "", "")
    )


def truncate_text(value: Any, limit: int = 220) -> str | None:
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    if not text:
        return None
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def selector_allowed_values(field: dict[str, Any]) -> list[Any]:
    selector = field.get("selector")
    select = selector.get("select") if isinstance(selector, dict) else None
    options = select.get("options") if isinstance(select, dict) else None
    if not isinstance(options, list):
        return []
    values = []
    for option in options:
        if isinstance(option, dict):
            if "value" in option:
                values.append(option["value"])
            elif "label" in option:
                values.append(option["label"])
        elif isinstance(option, (str, int, float, bool)):
            values.append(option)
    return values[:50]


def target_entity_domains(target: Any) -> list[str]:
    entity = target.get("entity") if isinstance(target, dict) else None
    domain = entity.get("domain") if isinstance(entity, dict) else None
    if isinstance(domain, str):
        return [domain]
    return [item for item in domain if isinstance(item, str)] if isinstance(domain, list) else []


def compact_service_definition(
    domain: str, service: str, definition: dict[str, Any]
) -> dict[str, Any]:
    fields = definition.get("fields")
    fields = fields if isinstance(fields, dict) else {}
    parameters = {}
    for name, raw in fields.items():
        field = raw if isinstance(raw, dict) else {}
        item = {"required": bool(field.get("required", False))}
        allowed = selector_allowed_values(field)
        if allowed:
            item["allowed"] = allowed
        for key in ("default", "example"):
            if key in field:
                item[key] = field.get(key)
        description = truncate_text(field.get("description"), 180)
        if description:
            item["description"] = description
        parameters[str(name)] = item
    target = definition.get("target")
    return {
        "domain": domain,
        "service": service,
        "name": definition.get("name"),
        "description": truncate_text(definition.get("description"), 240),
        "target": {
            "required": bool(target),
            "entity_domains": target_entity_domains(target),
            "accepts_entity_id": bool(target),
        },
        "parameters": parameters,
        "required_parameters": [key for key, value in parameters.items() if value["required"]],
        "returns_data": bool(definition.get("response")),
    }


def normalise_service_data(
    domain: str,
    service: str,
    definition: dict[str, Any] | None,
    supplied_data: dict[str, Any] | None,
    previous_data: dict[str, Any] | None,
    *,
    auto_repair: bool = True,
) -> tuple[dict[str, Any], list[str], dict[str, Any] | None]:
    data, repairs = dict(supplied_data or {}), []
    if not isinstance(definition, dict):
        return data, repairs, None
    fields = definition.get("fields")
    fields = fields if isinstance(fields, dict) else {}
    if isinstance(previous_data, dict):
        for key, value in previous_data.items():
            if key in fields and key not in data:
                data[key] = value
                repairs.append(f"preserved previous valid field '{key}'")
    required = [
        str(name)
        for name, field in fields.items()
        if isinstance(field, dict) and field.get("required") is True
    ]
    unknown = [str(key) for key in data if key not in fields]
    missing = [name for name in required if name not in data]
    if auto_repair and len(unknown) == len(missing) == 1:
        old, new = unknown[0], missing[0]
        field = fields.get(new)
        allowed = selector_allowed_values(field if isinstance(field, dict) else {})
        related = (
            old.casefold().endswith("_" + new.casefold())
            or new.casefold() in old.casefold()
            or difflib.SequenceMatcher(None, old.casefold(), new.casefold()).ratio() >= 0.55
        )
        if allowed and data.get(old) in allowed and related:
            data[new] = data.pop(old)
            repairs.append(
                f"renamed invalid field '{old}' to required field '{new}' because its value matches the allowed schema"
            )
    unknown = [str(key) for key in data if key not in fields]
    missing = [name for name in required if name not in data]
    invalid = {}
    for key, value in data.items():
        field = fields.get(key)
        allowed = selector_allowed_values(field) if isinstance(field, dict) else []
        if allowed and value not in allowed:
            invalid[key] = {"value": value, "allowed": allowed}
    if unknown or missing or invalid:
        return (
            data,
            repairs,
            {
                "unknown_parameters": unknown,
                "missing_required_parameters": missing,
                "invalid_values": invalid,
                "service_schema": compact_service_definition(domain, service, definition),
            },
        )
    return data, repairs, None
