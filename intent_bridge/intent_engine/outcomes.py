"""Typed results for deterministic planning and resolution."""

from __future__ import annotations

from dataclasses import dataclass

from intent_bridge.intent_engine.models import IntentPlan


@dataclass(frozen=True, slots=True)
class Resolved:
    plan: IntentPlan


@dataclass(frozen=True, slots=True)
class AmbiguousTarget:
    candidates: tuple[str, ...] = ()
    missing_constraint: str = "target"


@dataclass(frozen=True, slots=True)
class UnsupportedOperation:
    reason: str = "unsupported operation"


@dataclass(frozen=True, slots=True)
class IncompleteCompound:
    uncovered_clauses: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CapabilityMismatch:
    requested_capability: str
    candidate_targets: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class NoTarget:
    requested_domain: str | None = None


PlanningOutcome = (
    Resolved
    | AmbiguousTarget
    | UnsupportedOperation
    | IncompleteCompound
    | CapabilityMismatch
    | NoTarget
)


__all__ = [
    "AmbiguousTarget",
    "CapabilityMismatch",
    "IncompleteCompound",
    "NoTarget",
    "PlanningOutcome",
    "Resolved",
    "UnsupportedOperation",
]
