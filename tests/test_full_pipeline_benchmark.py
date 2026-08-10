from __future__ import annotations

import json

from agents.tool_context import ToolContext

from benchmark.full_pipeline import FullPipelineBenchmarkMatcher
from benchmark.models import BenchmarkRequest, Home, HomeArea, HomeEntity, Operation
from intent_bridge.config import settings
from intent_bridge.home_assistant.tools import ha_call_service


def _home() -> Home:
    return Home(
        home_id="endpoint-test",
        name="Endpoint Test",
        difficulty="basic",
        floors=(),
        areas=(HomeArea("living", "Living Room"),),
        entities=(
            HomeEntity(
                "light.lamp",
                "Lamp",
                "light",
                area_id="living",
                state="off",
            ),
        ),
    )


async def test_full_pipeline_benchmark_enters_endpoint_and_uses_deterministic_route():
    async def fallback(_request):
        raise AssertionError("fallback should not run")

    matcher = FullPipelineBenchmarkMatcher(fallback_handler=fallback)
    result = await matcher.match(BenchmarkRequest(("turn the lamp on",), _home()))

    assert matcher.last_routes == ("ohf-hassil",)
    assert result.operations == (
        Operation(kind="action", entity_ids=("light.lamp",), state="on"),
    )


async def test_full_pipeline_benchmark_runs_fallback_tools_against_fixture_home():
    async def fallback(_request):
        arguments = json.dumps(
            {
                "domain": "light",
                "service": "turn_on",
                "entity_id": "light.lamp",
                "area_id": None,
                "data": {},
                "return_response": False,
            }
        )
        context = ToolContext(
            None,
            tool_name="ha_call_service",
            tool_call_id="benchmark-test",
            tool_arguments=arguments,
        )
        reply = json.loads(await ha_call_service.on_invoke_tool(context, arguments))
        assert reply["success"] is True
        return settings.api.action_confirmation

    matcher = FullPipelineBenchmarkMatcher(force_llm=True, fallback_handler=fallback)
    result = await matcher.match(BenchmarkRequest(("use the fallback",), _home()))

    assert matcher.last_routes == ("llm-ha-ws",)
    assert result.operations == (
        Operation(kind="action", entity_ids=("light.lamp",), state="on"),
    )
