"""Narrow dependency-inversion ports for the deterministic engine."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from intent_bridge.intent_engine.models import (
    CatalogSnapshot,
    ExecutionResult,
    IntentMatch,
    IntentPlan,
    OhfIntentCall,
)


class IntentRecognizer(Protocol):
    def recognize(
        self,
        text: str,
        catalog: CatalogSnapshot,
        origin_context: Mapping[str, object] | None = None,
    ) -> tuple[IntentMatch, ...]: ...


class IntentPlanner(Protocol):
    """Build an ordered, side-effect-free plan directly from natural language."""

    def plan(
        self,
        text: str,
        catalog: CatalogSnapshot,
        origin_context: Mapping[str, object] | None = None,
    ) -> IntentPlan: ...


class CatalogProvider(Protocol):
    def snapshot(self) -> CatalogSnapshot: ...


class IntentExecutor(Protocol):
    async def execute(self, call: OhfIntentCall) -> ExecutionResult: ...


__all__ = ["CatalogProvider", "IntentExecutor", "IntentPlanner", "IntentRecognizer"]
