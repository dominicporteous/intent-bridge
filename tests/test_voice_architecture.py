import ast
import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import main
from intent_bridge import application
from intent_bridge.agents import factory as agent
from intent_bridge.agents.contracts import AgentToolPlugin
from intent_bridge.core.voice import (
    FunctionVoiceRoute,
    RouteExecutionError,
    VoiceActionPipeline,
    VoicePipelineError,
    VoiceRequest,
    VoiceResult,
)
from intent_bridge.runtime.execution import _reset_voice_tool_run_state, voice_tool_run_state


def request() -> VoiceRequest:
    return VoiceRequest("turn it on", "conversation")


@pytest.mark.asyncio
async def test_pipeline_stops_after_first_success():
    calls = []

    async def deterministic(_request):
        calls.append("deterministic")
        return "Done."

    async def fallback(_request):
        calls.append("fallback")
        return "Fallback."

    pipeline = VoiceActionPipeline(
        (
            FunctionVoiceRoute("deterministic", deterministic),
            FunctionVoiceRoute("fallback", fallback),
        )
    )
    result = await pipeline.handle(request())
    assert result.speech == "Done."
    assert result.route == "deterministic"
    assert result.failures == ()
    assert calls == ["deterministic"]


@pytest.mark.asyncio
async def test_pipeline_falls_back_and_reports_attempts():
    async def unavailable(_request):
        raise RuntimeError("not understood")

    async def fallback(_request):
        return "Handled."

    pipeline = VoiceActionPipeline(
        (
            FunctionVoiceRoute("deterministic", unavailable),
            FunctionVoiceRoute("llm", fallback),
        )
    )
    result = await pipeline.handle(request())
    assert result.route == "llm"
    assert result.failures[0].route == "deterministic"
    assert str(result.failures[0].error) == "not understood"


@pytest.mark.asyncio
async def test_pipeline_rejects_empty_results_and_reports_total_failure():
    async def empty(_request):
        return "  "

    pipeline = VoiceActionPipeline((FunctionVoiceRoute("empty", empty),))
    with pytest.raises(VoicePipelineError, match="empty response") as captured:
        await pipeline.handle(request())
    assert captured.value.failures[0].route == "empty"

    with pytest.raises(VoicePipelineError, match="No voice routes"):
        await VoiceActionPipeline(()).handle(request())


@pytest.mark.asyncio
async def test_pipeline_does_not_fall_through_after_execution_failure():
    fallback_called = False

    async def failed_after_execution(_request):
        raise RouteExecutionError("action may already have completed")

    async def fallback(_request):
        nonlocal fallback_called
        fallback_called = True
        return "Repeated action"

    pipeline = VoiceActionPipeline(
        (
            FunctionVoiceRoute("deterministic", failed_after_execution),
            FunctionVoiceRoute("llm", fallback),
        )
    )

    with pytest.raises(VoicePipelineError, match="may already have completed"):
        await pipeline.handle(request())
    assert fallback_called is False


def test_custom_agent_plugin_is_composed_without_agent_changes(monkeypatch):
    captured = {}
    custom_tool = object()
    plugin = AgentToolPlugin("custom-actions", (custom_tool,), "Use custom actions.")
    monkeypatch.setattr(agent, "Agent", lambda **kwargs: captured.update(kwargs) or kwargs)
    monkeypatch.setattr(agent, "_make_lemonade_model", lambda: "model")

    result = agent.make_fallback_agent(plugins=(plugin,))
    assert result["tools"] == [custom_tool]
    assert result["instructions"].endswith("Use custom actions.")


def test_application_factory_accepts_a_replacement_pipeline():
    class ReplacementPipeline:
        def __init__(self):
            self.requests = []

        async def handle(self, voice_request):
            self.requests.append(voice_request)
            return VoiceResult(speech="Injected.", route="replacement")

    replacement = ReplacementPipeline()
    created = application.create_app(replacement)
    assert created.state.voice_pipeline is replacement
    assert created.state.dependencies.voice_pipeline is replacement
    assert set(created.openapi()["paths"]) >= {
        "/v1/chat/completions",
        "/v1/models",
        "/health",
    }

    response = TestClient(created).post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "test request"}]},
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "Injected."
    assert response.json()["home_intent_proxy"]["route"] == "replacement"
    assert replacement.requests[0].text == "test request"

    with pytest.raises(ValueError, match="either pipeline or dependencies"):
        application.create_app(
            replacement,
            dependencies=application.ApplicationDependencies(replacement),
        )


def test_main_exposes_only_the_legacy_asgi_contract():
    assert main.__all__ == ["app"]
    assert main.app is application.app
    assert not hasattr(main, "asyncio")
    assert not hasattr(main, "MusicAssistantClient")


@pytest.mark.asyncio
async def test_voice_tool_state_is_isolated_between_concurrent_requests():
    ready = asyncio.Event()
    arrivals = 0
    arrival_lock = asyncio.Lock()

    async def run(area):
        nonlocal arrivals
        _reset_voice_tool_run_state("play music", {"area_name": area})
        async with arrival_lock:
            arrivals += 1
            if arrivals == 2:
                ready.set()
        await ready.wait()
        await asyncio.sleep(0)
        return voice_tool_run_state.origin_area_name

    assert await asyncio.gather(run("Office"), run("Kitchen")) == [
        "Office",
        "Kitchen",
    ]


def test_core_contracts_do_not_import_concrete_integrations():
    package = Path(application.__file__).parent
    prohibited = {
        "intent_bridge.home_assistant.advanced",
        "intent_bridge.home_assistant.tools",
        "intent_bridge.music_assistant.tools",
    }

    for module_name in (
        "core/voice.py",
        "agents/contracts.py",
        "core/tool_output.py",
    ):
        tree = ast.parse((package / module_name).read_text(encoding="utf-8"))
        imported = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        assert imported.isdisjoint(prohibited), module_name
