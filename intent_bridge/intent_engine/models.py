"""Provider-neutral values exchanged by deterministic intent components."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class SlotValue:
    """One recognized slot, retaining both canonical and spoken forms."""

    value: Any
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class IntentMatch:
    """Recognition result at the HassIL/OHF boundary."""

    intent_name: str
    slots: Mapping[str, SlotValue] = field(default_factory=dict)
    response_key: str | None = None
    context: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CatalogArea:
    area_id: str
    name: str
    aliases: tuple[str, ...] = ()
    floor_id: str | None = None


@dataclass(frozen=True, slots=True)
class CatalogFloor:
    floor_id: str
    name: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CatalogMeasurement:
    """One provider-neutral reading exposed by a catalog entity."""

    quantity: str
    value: str
    unit: str | None = None
    source: str = "state"


@dataclass(frozen=True, slots=True)
class CatalogEntity:
    entity_id: str
    name: str
    aliases: tuple[str, ...]
    domain: str
    area_id: str | None = None
    device_class: str | None = None
    state: str | None = None
    measurements: tuple[CatalogMeasurement, ...] = ()
    entity_category: str | None = None
    is_indicator: bool = False


@dataclass(frozen=True, slots=True)
class CatalogSnapshot:
    entities: tuple[CatalogEntity, ...] = ()
    areas: tuple[CatalogArea, ...] = ()
    floors: tuple[CatalogFloor, ...] = ()


@dataclass(frozen=True, slots=True)
class OhfIntentCall:
    """An official intent call ready for a compatible handler."""

    intent_name: str
    data: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class PlannedReading:
    """A resolved catalog reading that needs no provider-side target matching."""

    entity_id: str
    entity_name: str
    measurement: CatalogMeasurement


@dataclass(frozen=True, slots=True)
class SemanticEffect:
    """Provider-neutral meaning retained independently of an HA transport call."""

    speech_act: str
    property: str
    operator: str
    value: Any = None
    explicit_power_transition: bool = False


# ``RequestedEffect`` is the public vocabulary for command semantics.  Keep the
# older name as the concrete class so dialogue-state callers introduced before
# this name remain source and pickle compatible.
RequestedEffect = SemanticEffect


def semantic_effect_for_call(call: OhfIntentCall) -> SemanticEffect:
    """Describe an OHF call without importing Home Assistant service semantics."""

    intent = call.intent_name
    data = call.data
    if intent in {"HassGetState", "HassClimateGetTemperature"}:
        property_name = "temperature" if intent == "HassClimateGetTemperature" else "state"
        return SemanticEffect("query", property_name, "read")
    if intent in {"HassTurnOn", "HassTurnOff"}:
        return SemanticEffect(
            "command",
            "power",
            "set",
            intent == "HassTurnOn",
            explicit_power_transition=True,
        )
    property_by_intent = {
        "HassClimateSetTemperature": "temperature",
        "HassFanSetSpeed": "percentage",
        "HassLightSet": next(
            (
                key
                for key in ("brightness", "color", "color_temp", "temperature")
                if key in data
            ),
            "light_attribute",
        ),
        "HassSetPosition": "position",
        "HassSetVolume": "volume_level",
    }
    property_name = property_by_intent.get(intent, "operation")
    return SemanticEffect("command", property_name, "set", data.get(property_name))


@dataclass(frozen=True, slots=True)
class PlannedIntent:
    """One resolved, side-effect-free operation in an intent plan."""

    call: OhfIntentCall
    entity_ids: tuple[str, ...] = ()
    reading: PlannedReading | None = None
    effect: SemanticEffect | None = None

    @property
    def operation(self) -> str:
        """Return the canonical OHF operation name."""

        return self.call.intent_name

    @property
    def requested_effect(self) -> RequestedEffect | None:
        """The requested meaning, independent of the provider transport."""

        return self.effect


@dataclass(frozen=True, slots=True)
class IntentPlan:
    """A deterministic plan that can contain one or more ordered operations."""

    steps: tuple[PlannedIntent, ...] = ()
    response: str | None = None


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    speech: str = ""
    response: Mapping[str, Any] = field(default_factory=dict)


__all__ = [
    "CatalogArea",
    "CatalogEntity",
    "CatalogFloor",
    "CatalogMeasurement",
    "CatalogSnapshot",
    "ExecutionResult",
    "IntentMatch",
    "IntentPlan",
    "OhfIntentCall",
    "PlannedIntent",
    "PlannedReading",
    "RequestedEffect",
    "SemanticEffect",
    "SlotValue",
    "semantic_effect_for_call",
]
