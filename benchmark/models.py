"""Provider-neutral benchmark inputs and expected effects.

The matcher receives :class:`BenchmarkRequest`, which deliberately contains no
fixture identity or expected output.  This keeps the benchmark honest: an
implementation can only use the words, home topology, and initial state that a
real voice request would have available.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


def _json_value(value: Any) -> Any:
    """Return a stable, JSON-compatible representation of arbitrary YAML data."""

    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in sorted(value.items())}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def canonical_payload(value: Mapping[str, Any]) -> str:
    """Serialize a payload for deterministic comparison and collision checks."""

    return json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class HomeFloor:
    floor_id: str
    name: str
    level: int | None = None
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class HomeArea:
    area_id: str
    name: str
    floor_id: str | None = None
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class HomeEntity:
    entity_id: str
    name: str
    domain: str
    area_id: str | None = None
    state: str | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Home:
    """One isolated Home Assistant topology and its baseline entity state."""

    home_id: str
    name: str
    difficulty: str
    floors: tuple[HomeFloor, ...]
    areas: tuple[HomeArea, ...]
    entities: tuple[HomeEntity, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def entity(self, entity_id: str) -> HomeEntity | None:
        return next((entity for entity in self.entities if entity.entity_id == entity_id), None)

    def area(self, area_id: str) -> HomeArea | None:
        return next((area for area in self.areas if area.area_id == area_id), None)

    def entities_in(self, area_id: str, domain: str | None = None) -> tuple[HomeEntity, ...]:
        return tuple(
            entity
            for entity in self.entities
            if entity.area_id == area_id and (domain is None or entity.domain == domain)
        )


@dataclass(frozen=True, slots=True)
class Operation:
    """A canonical observable effect expected from an utterance or dialogue."""

    kind: str
    entity_ids: tuple[str, ...] = ()
    area_id: str | None = None
    domain: str | None = None
    state: str | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)
    intent: str | None = None
    # Used only while projecting a scenario's combined expectations onto an
    # individual wording. It is intentionally excluded from semantic equality.
    target_phrases: tuple[str, ...] = field(default=(), compare=False, repr=False)

    def _semantic_payload(self) -> Mapping[str, Any]:
        payload = dict(self.payload)
        if self.kind in {"shopping_list", "todo_list"}:
            item = payload.get("item")
            if isinstance(item, str):
                normal_item = " ".join(item.casefold().split())
                payload["item"] = re.sub(r"^(?:a|an|some|the)\s+", "", normal_item)
            list_name = payload.get("list_name")
            if isinstance(list_name, str):
                payload["list_name"] = " ".join(list_name.casefold().split())
        return payload

    def semantic_key(self) -> tuple[Any, ...]:
        target: tuple[Any, ...]
        if self.entity_ids:
            target = ("entities", *sorted(set(self.entity_ids)))
        else:
            target = ("area", self.area_id, self.domain)
        return (
            self.kind,
            target,
            self.state,
            self.intent,
            canonical_payload(self._semantic_payload()),
        )

    def atomic_semantic_keys(self) -> tuple[tuple[Any, ...], ...]:
        """Treat a multi-entity service call as its observable per-entity effects."""

        if len(self.entity_ids) <= 1:
            return (self.semantic_key(),)
        return tuple(
            Operation(
                kind=self.kind,
                entity_ids=(entity_id,),
                state=self.state,
                payload=self.payload,
                intent=self.intent,
            ).semantic_key()
            for entity_id in sorted(set(self.entity_ids))
        )


@dataclass(frozen=True, slots=True)
class BenchmarkRequest:
    """Safe matcher input with no answer key or fixture provenance."""

    turns: tuple[str, ...]
    home: Home
    setup: tuple[Operation, ...] = ()
    origin_context: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BenchmarkExample:
    """One independently executable gold example."""

    diagnostic_id: str
    request: BenchmarkRequest
    expected: tuple[Operation, ...]


@dataclass(frozen=True, slots=True)
class BenchmarkScenario:
    diagnostic_id: str
    name: str
    source: str
    home: Home
    setup: tuple[Operation, ...]
    expected: tuple[Operation, ...]
    examples: tuple[BenchmarkExample, ...]


@dataclass(frozen=True, slots=True)
class BenchmarkCorpus:
    root: str
    homes: tuple[Home, ...]
    scenarios: tuple[BenchmarkScenario, ...]

    @property
    def examples(self) -> tuple[BenchmarkExample, ...]:
        return tuple(example for scenario in self.scenarios for example in scenario.examples)


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    operations: tuple[Operation, ...]
    response: str = ""


class BenchmarkMatcher(Protocol):
    """Adapter boundary implemented by ``benchmark.production``."""

    async def match(self, request: BenchmarkRequest) -> BenchmarkResult: ...


__all__ = [
    "BenchmarkCorpus",
    "BenchmarkExample",
    "BenchmarkMatcher",
    "BenchmarkRequest",
    "BenchmarkResult",
    "BenchmarkScenario",
    "Home",
    "HomeArea",
    "HomeEntity",
    "HomeFloor",
    "Operation",
    "canonical_payload",
]
