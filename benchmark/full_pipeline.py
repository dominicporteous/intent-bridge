"""Opt-in benchmark adapter for the complete HTTP -> routes -> LLM/tool path.

Unlike :mod:`benchmark.production`, this adapter intentionally invokes the
OpenAI-compatible endpoint and may call a configured model.  Benchmark homes
are exposed through an in-memory Home Assistant WebSocket-compatible transport;
no command is sent to the user's real Home Assistant instance.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

import httpx

from benchmark.models import BenchmarkRequest, BenchmarkResult, Home, Operation
from benchmark.production import (
    ProductionBenchmarkMatcher,
    _catalog,
    _dedupe,
    _planning_context,
    _service_effect,
    _step_operations,
)
from intent_bridge.agents.factory import make_fallback_agent
from intent_bridge.agents.plugins import HOME_ASSISTANT_PLUGIN
from intent_bridge.application import create_app
from intent_bridge.config import settings
from intent_bridge.core.voice import (
    FunctionVoiceRoute,
    RouteDeclined,
    VoiceActionPipeline,
    VoiceRequest,
)
from intent_bridge.home_assistant.client import HomeAssistantWebSocket
from intent_bridge.intent_engine.models import CatalogSnapshot
from intent_bridge.intent_engine.supplemental import PlanningSession
from intent_bridge.llm import process_llm_fallback
from intent_bridge.runtime.dependencies import runtime

FallbackHandler = Callable[[VoiceRequest], Awaitable[str]]


def _service_definitions(home: Home) -> dict[str, dict[str, dict[str, Any]]]:
    def definition(domain: str, *fields: str, returns_data: bool = False) -> dict[str, Any]:
        value: dict[str, Any] = {
            "target": {"entity": {"domain": domain}},
            "fields": {field: {"required": False} for field in fields},
        }
        if returns_data:
            value["response"] = {"optional": True}
        return value

    templates: dict[str, dict[str, dict[str, Any]]] = {
        "light": {
            "turn_on": definition(
                "light", "brightness", "brightness_pct", "color_name", "rgb_color", "kelvin"
            ),
            "turn_off": definition("light"),
        },
        "switch": {
            "turn_on": definition("switch"),
            "turn_off": definition("switch"),
        },
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
        "lock": {
            "lock": definition("lock"),
            "unlock": definition("lock"),
        },
        "climate": {
            "set_temperature": definition("climate", "temperature", "hvac_mode"),
        },
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
        "weather": {
            "get_forecasts": definition("weather", "type", returns_data=True),
        },
    }
    domains = {entity.domain for entity in home.entities}
    return {domain: templates.get(domain, {}) for domain in domains}


class BenchmarkHomeAssistant(HomeAssistantWebSocket):
    """A fixture-backed HA cache that records service effects in memory."""

    def __init__(self, request: BenchmarkRequest) -> None:
        super().__init__("http://benchmark.invalid", "benchmark-only")
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
        self.entity_registry = {
            entity.entity_id: {
                "ai": entity.area_id,
                "en": entity.name,
                "aliases": list(entity.aliases),
            }
            for entity in request.home.entities
        }
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
        self.services = _service_definitions(request.home)
        self.operations: list[Operation] = []
        self.ready.set()
        self._services_loaded_at = time.monotonic()
        self._registries_loaded_at = time.monotonic()

    async def refresh_services(self, *, force: bool = False) -> None:
        del force

    async def refresh_registries(self, *, force: bool = False) -> None:
        del force

    def record_state_read(self, entity_id: str) -> None:
        self.operations.append(Operation(kind="query", entity_ids=(entity_id,)))

    def _targets(self, domain: str, target: Mapping[str, Any]) -> tuple[str, ...]:
        raw_ids = target.get("entity_id")
        if isinstance(raw_ids, str):
            return tuple(item.strip() for item in raw_ids.split(",") if item.strip())
        if isinstance(raw_ids, list):
            return tuple(str(item) for item in raw_ids)
        area_id = target.get("area_id")
        if isinstance(area_id, str):
            return tuple(
                entity.entity_id
                for entity in self.home.entities_in(area_id, domain)
            )
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
        self.operations.extend(_service_effect(f"{domain}.{service}", entity_ids, data))

        state_by_service = {
            "turn_on": "on",
            "turn_off": "off",
            "open_cover": "open",
            "close_cover": "closed",
            "lock": "locked",
            "unlock": "unlocked",
            "media_play": "playing",
            "media_pause": "paused",
            "start": "cleaning" if domain == "vacuum" else "on",
            "stop": "idle",
            "return_to_base": "returning",
        }
        if new_state := state_by_service.get(service):
            for entity_id in entity_ids:
                if entity_id in self.states:
                    self.states[entity_id]["state"] = new_state

        response: dict[str, Any] = {}
        if payload.get("return_response"):
            response = {
                entity_id: self.states.get(entity_id, {}) for entity_id in entity_ids
            }
        return {"success": True, "result": {"response": response}}


class _EndpointDeterministicRoute:
    name = "ohf-hassil"

    def __init__(
        self,
        request: BenchmarkRequest,
        operations: list[Operation],
        *,
        enabled: bool,
    ) -> None:
        self._request = request
        self._catalog: CatalogSnapshot = _catalog(request)
        self._context = _planning_context(request)
        self._operations = operations
        self._enabled = enabled
        self._session = PlanningSession(ProductionBenchmarkMatcher())

    async def handle(self, request: VoiceRequest) -> str:
        if not self._enabled:
            raise RouteDeclined("Deterministic route disabled by full benchmark")
        plan = self._session.plan(request.text, self._catalog, self._context)
        if plan.response is not None:
            return plan.response
        if not plan.steps:
            raise RouteDeclined("No deterministic intent matched the request")
        for step in plan.steps:
            self._operations.extend(
                _step_operations(step, self._request.home, self._request.setup)
            )
        self._context["last_entity_ids"] = tuple(
            dict.fromkeys(
                entity_id for step in plan.steps for entity_id in step.entity_ids
            )
        )
        return settings.api.action_confirmation


class FullPipelineBenchmarkMatcher:
    """Exercise the HTTP endpoint and production route order against fixtures.

    The adapter owns process-global runtime integration slots while ``match`` is
    active, so corpus execution must use concurrency=1.
    """

    def __init__(
        self,
        *,
        force_llm: bool = False,
        fallback_handler: FallbackHandler | None = None,
    ) -> None:
        self.force_llm = force_llm
        self.last_routes: tuple[str, ...] = ()
        self._fallback_handler = fallback_handler
        if fallback_handler is None:
            missing = []
            if not settings.llm.enabled:
                missing.append("INTENT_BRIDGE_LLM_ENABLED=true")
            if not settings.llm.base_url:
                missing.append("INTENT_BRIDGE_LLM_BASE_URL")
            if not settings.llm.model:
                missing.append("INTENT_BRIDGE_LLM_MODEL")
            if missing:
                raise RuntimeError(
                    "Full-pipeline LLM benchmark configuration is incomplete: "
                    + ", ".join(missing)
                )
            self._fallback_agent = make_fallback_agent(
                False,
                plugins=(HOME_ASSISTANT_PLUGIN,),
            )
        else:
            self._fallback_agent = None

    async def _live_fallback(self, request: VoiceRequest) -> str:
        return await process_llm_fallback(
            request.text,
            conversation_key=request.conversation_key,
            client_history=list(request.client_history),
            origin_context=request.origin_context,
        )

    async def match(self, request: BenchmarkRequest) -> BenchmarkResult:
        deterministic_operations: list[Operation] = []
        synthetic_ha = BenchmarkHomeAssistant(request)
        deterministic = _EndpointDeterministicRoute(
            request,
            deterministic_operations,
            enabled=not self.force_llm,
        )
        fallback = self._fallback_handler or self._live_fallback
        pipeline = VoiceActionPipeline(
            (
                deterministic,
                FunctionVoiceRoute("llm-ha-ws", fallback),
            )
        )
        app = create_app(pipeline=pipeline)

        previous_ws = runtime.ha_ws
        previous_agent = runtime.fallback_agent
        previous_advanced = runtime.advanced_agent
        runtime.ha_ws = synthetic_ha
        runtime.fallback_agent = self._fallback_agent
        runtime.advanced_agent = None
        routes: list[str] = []
        response_text = ""
        conversation_id = f"benchmark-{uuid.uuid4().hex}"
        messages: list[dict[str, str]] = []
        try:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://benchmark.local",
            ) as client:
                for turn_index, turn in enumerate(request.turns):
                    messages.append({"role": "user", "content": turn})
                    body: dict[str, Any] = {
                        "model": settings.api.model_name,
                        "messages": messages,
                        "conversation_id": conversation_id,
                        "reset_conversation": turn_index == 0,
                        **dict(request.origin_context),
                    }
                    response = await client.post("/v1/chat/completions", json=body)
                    response.raise_for_status()
                    payload = response.json()
                    response_text = str(payload["choices"][0]["message"]["content"])
                    routes.append(str(payload["home_intent_proxy"]["route"]))
                    messages.append({"role": "assistant", "content": response_text})
        finally:
            runtime.ha_ws = previous_ws
            runtime.fallback_agent = previous_agent
            runtime.advanced_agent = previous_advanced

        self.last_routes = tuple(routes)
        return BenchmarkResult(
            operations=_dedupe([*deterministic_operations, *synthetic_ha.operations]),
            response=response_text,
        )


__all__ = ["BenchmarkHomeAssistant", "FullPipelineBenchmarkMatcher"]
