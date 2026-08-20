"""Voice-route adapter for the deterministic intent engine."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Protocol

from intent_bridge.core.voice import (
    RouteDeclinedWithFallback,
    VoiceRequest,
)
from intent_bridge.intent_engine.models import CatalogSnapshot, IntentPlan
from intent_bridge.intent_engine.supplemental import PlanningSession

LOGGER = logging.getLogger(__name__)


class IntentRequestHandler(Protocol):
    async def handle(self, request: VoiceRequest) -> str: ...


class ConversationalIntentHandler(IntentRequestHandler, Protocol):
    def plan(self, request: VoiceRequest) -> IntentPlan: ...

    def catalog_snapshot(self) -> CatalogSnapshot: ...

    async def execute_plan(self, request: VoiceRequest, plan: IntentPlan) -> str: ...


@dataclass(slots=True)
class _ConversationPlanner:
    engine: ConversationalIntentHandler
    request: VoiceRequest

    def plan(
        self,
        text: str,
        catalog: CatalogSnapshot,
        origin_context: dict[str, object] | None = None,
    ) -> IntentPlan:
        request = replace(self.request, text=text, origin_context=origin_context)
        plan_with_catalog = getattr(self.engine, "plan_with_catalog", None)
        if callable(plan_with_catalog):
            return plan_with_catalog(request, catalog)
        # Retain compatibility with injected test/extension engines that only
        # expose the original ``plan(request)`` protocol.
        return self.engine.plan(request)


@dataclass(frozen=True, slots=True)
class DeterministicVoiceRoute:
    engine: IntentRequestHandler
    name: str = "ohf-hassil"

    async def handle(self, request: VoiceRequest) -> str:
        return await self.engine.handle(request)


class ConversationalDeterministicVoiceRoute:
    """Retain deterministic dialogue state per production conversation."""

    name = "ohf-hassil"

    def __init__(
        self,
        engine: ConversationalIntentHandler,
        *,
        ambiguous_target_fallback_enabled: bool = False,
        ambiguity_response: str = (
            "I found more than one possible target. Please be more specific."
        ),
    ) -> None:
        self._engine = engine
        self._ambiguous_target_fallback_enabled = ambiguous_target_fallback_enabled
        self._ambiguity_response = ambiguity_response
        self._sessions: dict[str, tuple[PlanningSession, _ConversationPlanner]] = {}

    async def handle(self, request: VoiceRequest) -> str:
        stored = self._sessions.get(request.conversation_key)
        if stored is None:
            planner = _ConversationPlanner(self._engine, request)
            session = PlanningSession(planner)
            self._sessions[request.conversation_key] = (session, planner)
            LOGGER.info(
                "DETERMINISTIC ROUTE session_created conversation=%r text=%r",
                request.conversation_key,
                request.text,
            )
        else:
            session, planner = stored
            planner.request = request
            LOGGER.info(
                "DETERMINISTIC ROUTE session_reused conversation=%r text=%r pending=%s "
                "focus=%s",
                request.conversation_key,
                request.text,
                session.state.pending.candidate_entity_ids if session.state.pending else (),
                session.state.focus.entity_set if session.state.focus else (),
            )
        plan = session.plan(
            request.text,
            self._engine.catalog_snapshot(),
            request.origin_context,
        )
        LOGGER.info(
            "DETERMINISTIC ROUTE planned conversation=%r text=%r steps=%s response=%r",
            request.conversation_key,
            request.text,
            tuple(
                {
                    "intent": step.call.intent_name,
                    "entity_ids": step.entity_ids,
                    "data": dict(step.call.data),
                }
                for step in plan.steps
            ),
            plan.response,
        )
        if (
            self._ambiguous_target_fallback_enabled
            and plan.response == self._ambiguity_response
        ):
            pending = session.state.pending
            LOGGER.info(
                "DETERMINISTIC ROUTE deferring ambiguity to fallback conversation=%r "
                "text=%r candidates=%s",
                request.conversation_key,
                request.text,
                pending.candidate_entity_ids if pending else (),
            )
            raise RouteDeclinedWithFallback(
                plan.response,
                on_alternative_success=lambda: session.dismiss_pending(pending),
            )
        return await self._engine.execute_plan(request, plan)


__all__ = [
    "ConversationalDeterministicVoiceRoute",
    "DeterministicVoiceRoute",
    "IntentRequestHandler",
]
