"""Home Assistant service capabilities shared by cataloging and execution."""

from __future__ import annotations

from collections.abc import Mapping

_EXACT_TARGET_SERVICE_BY_INTENT = {
    "HassVacuumStart": "start",
    "HassVacuumReturnToBase": "return_to_base",
}
_POWER_INTENTS = frozenset({"HassTurnOn", "HassTurnOff"})


def exact_target_service(intent_name: str, entity_id: str) -> str | None:
    """Return the HA service used for an exact entity retry."""

    service = _EXACT_TARGET_SERVICE_BY_INTENT.get(intent_name)
    if service:
        return service
    domain = entity_id.split(".", 1)[0]
    if intent_name == "HassTurnOn":
        return {"cover": "open_cover", "lock": "lock", "vacuum": "start"}.get(
            domain, "turn_on"
        )
    if intent_name == "HassTurnOff":
        return {"cover": "close_cover", "lock": "unlock", "vacuum": "stop"}.get(
            domain, "turn_off"
        )
    return None


def power_intents_supported_by_domain(
    domain: str,
    services: Mapping[str, object] | None,
) -> frozenset[str] | None:
    """Return known power intents from HA's live service registry.

    ``None`` means the registry was unavailable while the catalog was built;
    callers must retain conservative compatibility behaviour in that case.
    An empty set means HA explicitly exposes neither power action for the
    domain, which is materially different from not knowing.
    """

    if not services:
        return None
    domain_services = services.get(domain)
    if not isinstance(domain_services, Mapping):
        return frozenset()
    return frozenset(
        intent_name
        for intent_name in _POWER_INTENTS
        if exact_target_service(intent_name, f"{domain}.placeholder") in domain_services
    )


__all__ = ["exact_target_service", "power_intents_supported_by_domain"]
