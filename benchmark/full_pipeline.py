"""Benchmark adapter around the production voice pipeline.

The adapter replaces only Home Assistant's external I/O boundary. Grammar,
planning, dialogue state, route ordering, execution, and LLM fallback are all
constructed by :func:`intent_bridge.application.build_voice_pipeline`.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from benchmark.models import BenchmarkRequest, BenchmarkResult, Home, Operation
from benchmark.production import _service_effect, _step_operations, expand_static_invocation
from intent_bridge.agents.factory import make_fallback_agent
from intent_bridge.application import build_voice_pipeline
from intent_bridge.config import settings
from intent_bridge.core.voice import VoiceRequest
from intent_bridge.home_assistant.client import HomeAssistantWebSocket
from intent_bridge.intent_engine.grammar import load_intent_grammar
from intent_bridge.intent_engine.models import ExecutionResult, OhfIntentCall, PlannedIntent
from intent_bridge.intent_engine.recognizer import HassilIntentRecognizer
from intent_bridge.runtime.dependencies import runtime

FallbackHandler = Callable[[VoiceRequest], Awaitable[str]]


def _service_definitions(
    home: Home,
    extra_domains: tuple[str, ...] = (),
) -> dict[str, dict[str, dict[str, Any]]]:
    """Build fixture metadata in the shape exposed by Home Assistant."""

    def definition(domain: str, *fields: str, returns_data: bool = False) -> dict[str, Any]:
        value: dict[str, Any] = {
            "target": {"entity": {"domain": domain}},
            "fields": {field: {"required": False} for field in fields},
        }
        if returns_data:
            value["response"] = {"optional": True}
        return value

    templates = {
        "light": {
            "turn_on": definition(
                "light", "brightness", "brightness_pct", "color_name", "rgb_color", "kelvin"
            ),
            "turn_off": definition("light"),
        },
        "switch": {"turn_on": definition("switch"), "turn_off": definition("switch")},
        "fan": {
            "turn_on": definition("fan", "percentage"),
            "turn_off": definition("fan"),
            "set_percentage": definition("fan", "percentage"),
        },
        "cover": {
            "open_cover": definition("cover"),
            "close_cover": definition("cover"),
            "set_cover_position": definition("cover", "position"),
        },
        "lock": {"lock": definition("lock"), "unlock": definition("lock")},
        "climate": {"set_temperature": definition("climate", "temperature", "hvac_mode")},
        "media_player": {
            "media_play": definition("media_player"),
            "media_pause": definition("media_player"),
            "media_stop": definition("media_player"),
            "volume_set": definition("media_player", "volume_level"),
            "volume_mute": definition("media_player", "is_volume_muted"),
        },
        "vacuum": {
            "start": definition("vacuum"),
            "stop": definition("vacuum"),
            "return_to_base": definition("vacuum"),
        },
        "scene": {"turn_on": definition("scene")},
        "script": {"turn_on": definition("script")},
        "weather": {"get_forecasts": definition("weather", "type", returns_data=True)},
        "todo": {
            "add_item": definition("todo", "item"),
            "update_item": definition("todo", "item", "rename", "status"),
            "remove_item": definition("todo", "item"),
            "get_items": definition("todo", "status", returns_data=True),
        },
    }
    return {
        domain: templates.get(domain, {})
        for domain in {*extra_domains, *(entity.domain for entity in home.entities)}
    }


class BenchmarkHomeAssistant:
    """Fixture-backed implementation of the production HA cache/tool boundary."""

    def __init__(self, request: BenchmarkRequest) -> None:
        self.home = request.home
        self.setup = request.setup
        setup_states = {
            entity_id: operation.state
            for operation in request.setup
            if operation.state is not None
            for entity_id in operation.entity_ids
        }
        self.states = {
            entity.entity_id: {
                "entity_id": entity.entity_id,
                "state": setup_states.get(entity.entity_id, entity.state or "unknown"),
                "attributes": {"friendly_name": entity.name, **dict(entity.attributes)},
            }
            for entity in request.home.entities
        }
        for operation in request.setup:
            for entity_id in operation.entity_ids:
                state = self.states.setdefault(
                    entity_id,
                    {
                        "entity_id": entity_id,
                        "state": operation.state or "unknown",
                        "attributes": {
                            "friendly_name": entity_id.split(".", 1)[-1]
                            .replace("_", " ")
                            .title()
                        },
                    },
                )
                if operation.state is not None:
                    state["state"] = operation.state
                state["attributes"].update(operation.payload)

        shopping_items = [
            str(operation.payload["shopping_list_item"])
            for operation in request.setup
            if operation.payload.get("shopping_list_item")
        ]
        todo_items: dict[str, list[str]] = {}
        for operation in request.setup:
            item = operation.payload.get("todo_item")
            list_name = operation.payload.get("list_name")
            if item and list_name:
                todo_items.setdefault(str(list_name), []).append(str(item))
        if shopping_items:
            self.states["todo.shopping_list"] = {
                "entity_id": "todo.shopping_list",
                "state": str(len(shopping_items)),
                "attributes": {
                    "friendly_name": "Shopping List",
                    "items": tuple(shopping_items),
                },
            }
        for list_name, items in todo_items.items():
            object_id = "_".join(
                part for part in "".join(
                    char.casefold() if char.isalnum() else " " for char in list_name
                ).split() if part
            )
            self.states[f"todo.{object_id}"] = {
                "entity_id": f"todo.{object_id}",
                "state": str(len(items)),
                "attributes": {"friendly_name": list_name, "items": tuple(items)},
            }
        self.entity_registry = {
            entity.entity_id: {
                "ai": entity.area_id,
                "en": entity.name,
                "aliases": list(entity.aliases),
            }
            for entity in request.home.entities
        }
        for entity_id, state in self.states.items():
            self.entity_registry.setdefault(
                entity_id,
                {
                    "ai": None,
                    "en": state["attributes"].get("friendly_name"),
                    "aliases": [],
                },
            )
        self.devices: dict[str, dict[str, Any]] = {}
        self.areas = {
            area.area_id: {
                "name": area.name,
                "aliases": list(area.aliases),
                "floor_id": area.floor_id,
            }
            for area in request.home.areas
        }
        self.floors = {
            floor.floor_id: {
                "name": floor.name,
                "aliases": list(floor.aliases),
                "level": floor.level,
            }
            for floor in request.home.floors
        }
        self.services = _service_definitions(
            request.home,
            tuple(entity_id.split(".", 1)[0] for entity_id in self.states),
        )
        self.operations: list[Operation] = []
        self.ready = asyncio.Event()
        self.ready.set()

    async def refresh_services(self, *, force: bool = False) -> None:
        del force

    async def refresh_registries(self, *, force: bool = False) -> None:
        del force

    # Reuse the production cache discovery implementation. The benchmark owns
    # only the external I/O boundary; agent tools see the same search and area
    # behavior as the application.
    _entity_context = HomeAssistantWebSocket._entity_context
    search_cached_states = HomeAssistantWebSocket.search_cached_states
    resolve_area_reference = HomeAssistantWebSocket.resolve_area_reference
    resolve_device_origin = HomeAssistantWebSocket.resolve_device_origin
    area_mentioned_in_text = HomeAssistantWebSocket.area_mentioned_in_text
    entities_in_area = HomeAssistantWebSocket.entities_in_area

    async def wait_for_expected_state(
        self,
        entity_id: str,
        expected_state: str,
        timeout: float,
    ) -> str | None:
        del timeout
        state = self.states.get(entity_id)
        return expected_state if state and state.get("state") == expected_state else None

    def record_state_read(self, entity_id: str) -> None:
        self.operations.append(Operation(kind="query", entity_ids=(entity_id,)))

    def record_step(self, step: PlannedIntent) -> None:
        effects = _step_operations(step, self.home, self.setup)
        self.operations.extend(effects)
        self._apply_states(effects)

    def _apply_states(self, effects: tuple[Operation, ...]) -> None:
        for effect in effects:
            if effect.state is None:
                continue
            for entity_id in effect.entity_ids:
                if entity_id in self.states:
                    self.states[entity_id]["state"] = effect.state

    def _targets(self, domain: str, target: Mapping[str, Any]) -> tuple[str, ...]:
        raw_ids = target.get("entity_id")
        if isinstance(raw_ids, str):
            return tuple(item.strip() for item in raw_ids.split(",") if item.strip())
        if isinstance(raw_ids, list):
            return tuple(str(item) for item in raw_ids)
        area_id = target.get("area_id")
        if isinstance(area_id, str):
            return tuple(entity.entity_id for entity in self.home.entities_in(area_id, domain))
        candidates = tuple(
            entity.entity_id for entity in self.home.entities if entity.domain == domain
        )
        return candidates if len(candidates) == 1 else ()

    async def command(
        self,
        payload: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        del timeout
        if payload.get("type") != "call_service":
            return {"success": True, "result": {}}

        domain = str(payload.get("domain") or "")
        service = str(payload.get("service") or "")
        target = payload.get("target")
        target = target if isinstance(target, Mapping) else {}
        entity_ids = self._targets(domain, target)
        if not entity_ids:
            return {
                "success": False,
                "error": {"code": "benchmark_target", "message": "No unique target"},
            }

        data = payload.get("service_data")
        data = dict(data) if isinstance(data, Mapping) else {}
        if "brightness_pct" in data and "brightness" not in data:
            data["brightness"] = data.pop("brightness_pct")
        if "color_name" in data and "color" not in data:
            data["color"] = data.pop("color_name")
        volume = data.get("volume_level")
        if isinstance(volume, (int, float)) and 0 <= volume <= 1:
            data["volume_level"] = round(volume * 100)

        expanded = domain in {"scene", "script"} and service == "turn_on"
        effects = (
            tuple(
                effect
                for entity_id in entity_ids
                for effect in expand_static_invocation(self.home, entity_id)
            )
            if expanded
            else _service_effect(f"{domain}.{service}", entity_ids, data)
        )
        self.operations.extend(effects)
        self._apply_states(effects)

        response = (
            {entity_id: self.states.get(entity_id, {}) for entity_id in entity_ids}
            if payload.get("return_response")
            else {}
        )
        return {"success": True, "result": {"response": response}}


class _FixtureIntentExecutor:
    """Successful HA intent boundary; the production engine supplies resolved steps."""

    async def execute(self, call: OhfIntentCall) -> ExecutionResult:
        del call
        return ExecutionResult(speech=settings.api.action_confirmation)


class FullPipelineBenchmarkMatcher:
    """Run requests through the production pipeline against an isolated HA fixture."""

    def __init__(
        self,
        *,
        fallback_handler: FallbackHandler | None = None,
    ) -> None:
        self.last_routes: tuple[str, ...] = ()
        self._fallback_handler = fallback_handler
        if fallback_handler is None:
            configured = bool(
                settings.llm.enabled and settings.llm.base_url and settings.llm.model
            )
            self._fallback_agent = make_fallback_agent(False) if configured else None
        else:
            self._fallback_agent = None
        grammar = load_intent_grammar(
            language=settings.deterministic.language,
            custom_sentences_path=settings.deterministic.custom_sentences_path,
        )
        self._intent_recognizer = HassilIntentRecognizer(grammar)

    async def match(self, request: BenchmarkRequest) -> BenchmarkResult:
        fixture = BenchmarkHomeAssistant(request)
        pipeline = build_voice_pipeline(
            intent_executor=_FixtureIntentExecutor(),
            intent_recognizer=self._intent_recognizer,
            fallback_handler=self._fallback_handler,
            step_observer=fixture.record_step,
        )

        previous_ws = runtime.ha_ws
        previous_agent = runtime.fallback_agent
        routes: list[str] = []
        response_text = ""
        operations_by_turn: list[tuple[Operation, ...]] = []
        conversation_id = f"benchmark-{uuid.uuid4().hex}"
        messages: list[dict[str, str]] = []
        runtime.ha_ws = fixture
        if self._fallback_agent is not None:
            runtime.fallback_agent = self._fallback_agent
        try:
            for turn in request.turns:
                operation_start = len(fixture.operations)
                messages.append({"role": "user", "content": turn})
                voice_request = VoiceRequest(
                    text=turn,
                    conversation_key=conversation_id,
                    client_history=tuple(messages),
                    origin_context=dict(request.origin_context),
                )
                result = await pipeline.handle(voice_request)
                response_text = result.speech
                routes.append(result.route)
                messages.append({"role": "assistant", "content": response_text})
                operations_by_turn.append(tuple(fixture.operations[operation_start:]))
        finally:
            runtime.ha_ws = previous_ws
            runtime.fallback_agent = previous_agent

        self.last_routes = tuple(routes)
        return BenchmarkResult.from_turn_operations(
            operations_by_turn,
            response=response_text,
        )


__all__ = ["BenchmarkHomeAssistant", "FullPipelineBenchmarkMatcher"]
