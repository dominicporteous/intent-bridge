from __future__ import annotations

import json
from types import SimpleNamespace

from agents.tool_context import ToolContext

from benchmark.full_pipeline import FullPipelineBenchmarkMatcher
from benchmark.models import (
    BenchmarkRequest,
    BenchmarkResult,
    Home,
    HomeArea,
    HomeEntity,
    Operation,
)
from benchmark.runner import BenchmarkOptions, select_examples
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


def _two_light_home() -> Home:
    return Home(
        home_id="endpoint-dialogue-test",
        name="Endpoint Dialogue Test",
        difficulty="basic",
        floors=(),
        areas=(HomeArea("kitchen", "Kitchen"),),
        entities=(
            HomeEntity("light.ceiling", "Ceiling Light", "light", "kitchen", state="off"),
            HomeEntity("light.counter", "Counter Light", "light", "kitchen", state="off"),
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


async def test_full_pipeline_keeps_all_mutations_but_only_final_turn_reads():
    async def fallback(_request):
        raise AssertionError("fallback should not run")

    matcher = FullPipelineBenchmarkMatcher(fallback_handler=fallback)
    result = await matcher.match(
        BenchmarkRequest(
            ("what is the state of the ceiling light", "turn it on"),
            _two_light_home(),
        )
    )

    assert matcher.last_routes == ("ohf-hassil", "ohf-hassil")
    assert result.operations == (
        Operation(kind="action", entity_ids=("light.ceiling",), state="on"),
    )
    assert result.turn_operations == (
        (Operation(kind="query", entity_ids=("light.ceiling",)),),
        (Operation(kind="action", entity_ids=("light.ceiling",), state="on"),),
    )
    assert result.mutation_ledger == result.operations
    assert result.ignored_setup_observations == (
        Operation(kind="query", entity_ids=("light.ceiling",)),
    )


async def test_full_pipeline_does_not_mutate_before_clarification_resolves():
    async def fallback(_request):
        raise AssertionError("fallback should not run")

    matcher = FullPipelineBenchmarkMatcher(fallback_handler=fallback)
    result = await matcher.match(
        BenchmarkRequest(
            ("turn the kitchen light on", "the ceiling one"),
            _two_light_home(),
        )
    )

    assert matcher.last_routes == ("ohf-hassil", "ohf-hassil")
    assert result.operations == (
        Operation(kind="action", entity_ids=("light.ceiling",), state="on"),
    )


async def test_full_pipeline_preserves_eager_mutation_as_a_safety_failure():
    calls = 0

    async def fallback(_request):
        nonlocal calls
        entity_id = "light.counter" if calls == 0 else "light.ceiling"
        calls += 1
        arguments = json.dumps(
            {
                "domain": "light",
                "service": "turn_on",
                "entity_id": entity_id,
                "area_id": None,
                "data": {},
                "return_response": False,
            }
        )
        context = ToolContext(
            None,
            tool_name="ha_call_service",
            tool_call_id=f"eager-mutation-{calls}",
            tool_arguments=arguments,
        )
        await ha_call_service.on_invoke_tool(context, arguments)
        return settings.api.action_confirmation

    matcher = FullPipelineBenchmarkMatcher(force_llm=True, fallback_handler=fallback)
    result = await matcher.match(
        BenchmarkRequest(
            ("turn the kitchen light on", "the ceiling one"),
            _two_light_home(),
        )
    )

    eager = Operation(kind="action", entity_ids=("light.counter",), state="on")
    intended = Operation(kind="action", entity_ids=("light.ceiling",), state="on")
    assert result.turn_operations == ((eager,), (intended,))
    assert result.mutation_ledger == (eager, intended)
    assert result.operations == (eager, intended)


def test_temporal_result_does_not_dedupe_repeated_mutations():
    mutation = Operation(kind="action", entity_ids=("light.ceiling",), state="on")

    result = BenchmarkResult.from_turn_operations(((mutation,), (mutation,)))

    assert result.mutation_ledger == (mutation, mutation)
    assert result.operations == (mutation, mutation)


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


def test_benchmark_options_enable_full_pipeline_for_configured_llm():
    options = BenchmarkOptions.from_environment(
        {},
        llm_settings=SimpleNamespace(
            enabled=True,
            base_url="http://llm.test/v1",
            model="test-model",
        ),
    )

    assert options.use_full_pipeline is True
    assert options.force_llm is False
    assert options.limit is None


def test_benchmark_options_preserve_exhaustive_deterministic_default():
    options = BenchmarkOptions.from_environment(
        {"BENCHMARK_HOME": "first, second", "BENCHMARK_LIMIT": "2"},
        llm_settings=SimpleNamespace(enabled=False, base_url="", model=""),
    )

    assert options.homes == ("first", "second")
    assert options.use_full_pipeline is False
    assert options.limit == 2


def test_benchmark_options_force_llm_even_when_configuration_is_incomplete():
    options = BenchmarkOptions.from_environment(
        {"BENCHMARK_FORCE_LLM": "true"},
        llm_settings=SimpleNamespace(enabled=False, base_url="", model=""),
    )

    assert options.use_full_pipeline is True
    assert options.force_llm is True
    assert options.limit is None


def test_benchmark_selection_applies_filters_before_limit():
    examples = tuple(
        SimpleNamespace(
            diagnostic_id=f"home/{source}/{index}",
            request=SimpleNamespace(home=SimpleNamespace(home_id=home_id)),
        )
        for index, (home_id, source) in enumerate(
            (("first", "lights"), ("second", "lights"), ("first", "covers"))
        )
    )
    options = BenchmarkOptions(homes=("first",), sources=("lights",), limit=1)

    assert select_examples(examples, options) == (examples[0],)
