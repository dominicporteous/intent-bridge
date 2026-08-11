from __future__ import annotations

import json
from types import SimpleNamespace

from agents.tool_context import ToolContext

import benchmark.full_pipeline as full_pipeline_module
from benchmark.full_pipeline import (
    BenchmarkHomeAssistant,
    FullPipelineBenchmarkMatcher,
)
from benchmark.loader import load_corpus
from benchmark.models import (
    BenchmarkRequest,
    BenchmarkResult,
    Home,
    HomeArea,
    HomeEntity,
    Operation,
)
from benchmark.runner import (
    BenchmarkOptions,
    compare_operations,
    make_benchmark_matcher,
    select_examples,
)
from intent_bridge.config import settings
from intent_bridge.home_assistant.tools import ha_call_service, ha_get_state, ha_search


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


async def test_full_pipeline_benchmark_uses_production_deterministic_route():
    async def fallback(_request):
        raise AssertionError("fallback should not run")

    matcher = FullPipelineBenchmarkMatcher(fallback_handler=fallback)
    result = await matcher.match(BenchmarkRequest(("turn the lamp on",), _home()))

    assert matcher.last_routes == ("ohf-hassil",)
    assert result.operations == (
        Operation(kind="action", entity_ids=("light.lamp",), state="on"),
    )


async def test_full_pipeline_matcher_reuses_compiled_intent_recognizer(monkeypatch):
    grammar_loads = 0
    original_load = full_pipeline_module.load_intent_grammar

    def counting_load(**kwargs):
        nonlocal grammar_loads
        grammar_loads += 1
        return original_load(**kwargs)

    async def fallback(_request):
        raise AssertionError("fallback should not run")

    monkeypatch.setattr(full_pipeline_module, "load_intent_grammar", counting_load)
    matcher = FullPipelineBenchmarkMatcher(fallback_handler=fallback)

    await matcher.match(BenchmarkRequest(("turn the lamp on",), _home()))
    await matcher.match(BenchmarkRequest(("turn the lamp on",), _home()))

    assert grammar_loads == 1


async def test_fixture_ha_expands_script_effects_and_updates_downstream_state():
    home = Home(
        home_id="script-test",
        name="Script Test",
        difficulty="basic",
        floors=(),
        areas=(),
        entities=(
            HomeEntity("script.shutdown", "Shutdown", "script"),
            HomeEntity("light.one", "One", "light", state="on"),
            HomeEntity("light.two", "Two", "light", state="on"),
        ),
        metadata={
            "scripts": [
                {
                    "id": "shutdown",
                    "actions": [
                        {
                            "action": "light.turn_off",
                            "target": {"entity_id": ["light.one", "light.two"]},
                        }
                    ],
                }
            ]
        },
    )
    ha = BenchmarkHomeAssistant(BenchmarkRequest(("run shutdown",), home))

    result = await ha.command(
        {
            "type": "call_service",
            "domain": "script",
            "service": "turn_on",
            "target": {"entity_id": "script.shutdown"},
        }
    )

    assert result["success"] is True
    assert ha.operations == [
        Operation(
            kind="action",
            entity_ids=("light.one", "light.two"),
            state="off",
        )
    ]
    assert ha.states["light.one"]["state"] == "off"
    assert ha.states["light.two"]["state"] == "off"


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
    assert result.mutation_ledger == result.operations
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
    assert result.turn_operations[0] == ()
    assert result.operations == (
        Operation(kind="action", entity_ids=("light.ceiling",), state="on"),
    )
    assert result.mutation_ledger == result.operations


async def test_singular_studio_clarifications_are_transactional_end_to_end():
    async def fallback(_request):
        raise AssertionError("every clarification turn should remain deterministic")

    examples = tuple(
        example
        for example in load_corpus().examples
        if "/clarifications/" in example.diagnostic_id
        and example.request.home.home_id == "studio"
        and "kitchen lights" not in example.request.turns[0].casefold()
    )
    assert len(examples) == 9

    for example in examples:
        result = await FullPipelineBenchmarkMatcher(fallback_handler=fallback).match(
            example.request
        )
        missing, unexpected = compare_operations(example.expected, result.operations)
        assert not missing and not unexpected, example.diagnostic_id
        assert result.turn_operations[0] == (), example.diagnostic_id
        assert result.mutation_ledger == result.operations, example.diagnostic_id


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

    matcher = FullPipelineBenchmarkMatcher(fallback_handler=fallback)
    result = await matcher.match(
        BenchmarkRequest(
            ("perform the fallback operation", "perform it again"),
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
        search_arguments = json.dumps(
            {
                "query": "living room lamp",
                "domain_filter": "light",
                "area_filter": None,
                "limit": 10,
            }
        )
        search_context = ToolContext(
            None,
            tool_name="ha_search",
            tool_call_id="benchmark-search-test",
            tool_arguments=search_arguments,
        )
        search_reply = json.loads(
            await ha_search.on_invoke_tool(search_context, search_arguments)
        )
        assert search_reply["recommended_entity_id"] == "light.lamp"

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

    matcher = FullPipelineBenchmarkMatcher(fallback_handler=fallback)
    result = await matcher.match(BenchmarkRequest(("use the fallback",), _home()))

    assert matcher.last_routes == ("llm-ha-ws",)
    assert result.operations == (
        Operation(kind="action", entity_ids=("light.lamp",), state="on"),
    )


async def test_fallback_can_return_speech_with_observable_query_evidence():
    async def fallback(_request):
        await ha_get_state.on_invoke_tool(
            ToolContext(
                None,
                tool_name="ha_get_state",
                tool_call_id="query-evidence",
                tool_arguments='{"entity_id":"light.lamp","attribute_keys":null}',
            ),
            '{"entity_id":"light.lamp","attribute_keys":null}',
        )
        return "The living room lamp is off."

    matcher = FullPipelineBenchmarkMatcher(fallback_handler=fallback)
    result = await matcher.match(
        BenchmarkRequest(("use the fallback to inspect the lamp",), _home())
    )

    assert matcher.last_routes == ("llm-ha-ws",)
    assert result.response == "The living room lamp is off."
    assert result.operations == (Operation(kind="query", entity_ids=("light.lamp",)),)


def test_benchmark_options_are_independent_of_llm_configuration():
    options = BenchmarkOptions.from_environment({})

    assert options.limit is None
    assert isinstance(make_benchmark_matcher(options), FullPipelineBenchmarkMatcher)


def test_benchmark_options_preserve_exhaustive_default():
    options = BenchmarkOptions.from_environment(
        {"BENCHMARK_HOME": "first, second", "BENCHMARK_LIMIT": "2"}
    )

    assert options.homes == ("first", "second")
    assert options.limit == 2


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
