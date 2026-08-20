import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from agents.tool_context import ToolContext

from intent_bridge.agents import factory as agent
from intent_bridge.agents import results as tool_results
from intent_bridge.config import settings
from intent_bridge.home_assistant import advanced
from intent_bridge.runtime.dependencies import runtime
from intent_bridge.runtime.execution import recent_music_action_responses, voice_tool_run_state


def result(name, output):
    return SimpleNamespace(tool=SimpleNamespace(name=name), output=output)


def test_music_replay_cache(monkeypatch):
    recent_music_action_responses.clear()
    monkeypatch.setattr(settings.music_assistant, "replay_guard_seconds", 5)
    origin = {"area_id": "office"}
    tool_results._remember_music_action_response("c", " Play Song! ", origin, "Playing.")
    assert tool_results._get_recent_music_action_response("c", "play song", origin) == "Playing."
    recent_music_action_responses["expired"] = (time.monotonic() - 10, "old")
    assert tool_results._get_recent_music_action_response("other", "x", None) is None
    assert "expired" not in recent_music_action_responses
    monkeypatch.setattr(settings.music_assistant, "replay_guard_seconds", 0)
    tool_results._remember_music_action_response("c", "x", None, "x")
    assert tool_results._get_recent_music_action_response("c", "x", None) is None


def test_fast_result_handler_music_success_failure_and_reads(monkeypatch):
    monkeypatch.setattr(settings.music_assistant, "terminal_actions_enabled", True)
    success = tool_results.fast_tool_result_handler(
        None, [result("ma_volume", {"success": True, "message": "Muted."})]
    )
    assert success.is_final_output and success.final_output == "Muted."
    assert voice_tool_run_state.last_successful_music_action["tool"] == "ma_volume"
    multiple = tool_results.fast_tool_result_handler(
        None,
        [result("ma_volume", {"success": True}), result("ma_playback", {"success": True})],
    )
    assert multiple.final_output == settings.api.action_confirmation
    failed = tool_results.fast_tool_result_handler(None, [result("ma_volume", {"success": False})])
    assert not failed.is_final_output
    read = tool_results.fast_tool_result_handler(None, [result("ma_search", {"success": True})])
    assert not read.is_final_output


def test_fast_result_handler_home_assistant_paths():
    failed = tool_results.fast_tool_result_handler(
        None, [result("ha_call_service", {"success": False})]
    )
    assert not failed.is_final_output
    data = tool_results.fast_tool_result_handler(
        None,
        [
            result(
                "ha_call_service",
                {
                    "success": True,
                    "domain": "weather",
                    "service": "forecast",
                    "service_response": {"x": 1},
                },
            )
        ],
    )
    assert not data.is_final_output
    assert voice_tool_run_state.last_successful_data["service_response"] == {"x": 1}
    unknown = tool_results.fast_tool_result_handler(
        None, [result("ha_call_service", {"success": True, "service": "custom"})]
    )
    assert not unknown.is_final_output
    action = tool_results.fast_tool_result_handler(
        None, [result("ha_call_service", {"success": True, "service": "turn_on"})]
    )
    assert action.is_final_output and action.final_output == settings.api.action_confirmation
    assert voice_tool_run_state.last_successful_ha_action == {
        "calls": 1,
        "spoken": settings.api.action_confirmation,
    }
    verified = tool_results.fast_tool_result_handler(
        None,
        [result("ha_call_service", {"success": True, "service": "custom", "verified_state": "on"})],
    )
    assert verified.is_final_output


def test_advanced_configuration_helpers(monkeypatch):
    monkeypatch.setattr(
        settings.music_assistant, "area_player_map", "Office=player1, bad, Kitchen = player2"
    )
    assert advanced._parse_music_area_player_map() == {"office": "player1", "kitchen": "player2"}
    assert "office -> player1" in advanced._music_assistant_agent_instructions()
    monkeypatch.setattr(settings.music_assistant, "area_player_map", "")
    assert advanced._parse_music_area_player_map() == {}
    monkeypatch.setattr(settings.mcp, "client_session_timeout_seconds", 45)
    server = advanced.make_ha_mcp_server()
    assert server.name == "Home Assistant Advanced"
    assert server.client_session_timeout_seconds == 45


@pytest.mark.asyncio
async def test_advanced_tool_unavailable_success_empty_and_error(monkeypatch):
    encoded = json.dumps({"request": "diagnose"})
    context = ToolContext(
        None, tool_name="ha_advanced", tool_call_id="test", tool_arguments=encoded
    )

    async def invoke():
        return json.loads(await advanced.ha_advanced.on_invoke_tool(context, encoded))

    monkeypatch.setattr(runtime, "advanced_agent", None)
    assert not (await invoke())["success"]
    monkeypatch.setattr(runtime, "advanced_agent", object())
    monkeypatch.setattr(
        advanced.Runner, "run", AsyncMock(return_value=SimpleNamespace(final_output="Result"))
    )
    assert (await invoke())["result"] == "Result"
    advanced.Runner.run = AsyncMock(return_value=SimpleNamespace(final_output=""))
    assert not (await invoke())["success"]
    advanced.Runner.run = AsyncMock(side_effect=RuntimeError("failed"))
    assert not (await invoke())["success"]


def test_agent_factory_tool_sets(monkeypatch):
    created = {}

    def fake_agent(**kwargs):
        created.update(kwargs)
        return kwargs

    monkeypatch.setattr(agent, "Agent", fake_agent)
    monkeypatch.setattr(agent, "_make_lemonade_model", lambda: "model")
    monkeypatch.setattr(runtime, "advanced_agent", None)
    result = agent.make_fallback_agent(False)
    assert len(result["tools"]) == 4
    monkeypatch.setattr(runtime, "advanced_agent", object())
    result = agent.make_fallback_agent(True)
    assert len(result["tools"]) == 16
    assert "MUSIC ASSISTANT AUTHORITY" in result["instructions"]


def test_agent_factory_exposes_custom_mcp_servers(monkeypatch):
    created = {}
    monkeypatch.setattr(agent, "Agent", lambda **kwargs: created.update(kwargs) or kwargs)
    monkeypatch.setattr(agent, "_make_lemonade_model", lambda: "model")
    monkeypatch.setattr(runtime, "advanced_agent", None)
    server = object()

    agent.make_fallback_agent(
        False,
        mcp_servers=(server,),
        mcp_instructions="CUSTOM MCP TOOLS\n\n- Web Search MCP",
    )

    assert created["mcp_servers"] == [server]
    assert "Web Search MCP" in created["instructions"]


def test_informational_agent_has_mcp_but_no_household_tools(monkeypatch):
    created = {}
    monkeypatch.setattr(agent, "Agent", lambda **kwargs: created.update(kwargs) or kwargs)
    monkeypatch.setattr(agent, "_make_lemonade_model", lambda: "model")
    server = object()

    agent.make_informational_agent(
        mcp_servers=(server,),
        mcp_instructions="CUSTOM MCP TOOLS\n\n- Web Search MCP",
    )

    assert created["tools"] == []
    assert created["mcp_servers"] == [server]
    assert "MUST use a relevant web search" in created["instructions"]
    assert "application-supplied block as authoritative" in created["instructions"]
    assert "Web Search MCP" in created["instructions"]
