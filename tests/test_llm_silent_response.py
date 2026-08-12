from types import SimpleNamespace

import pytest

from intent_bridge import llm
from intent_bridge.config import settings
from intent_bridge.runtime.dependencies import runtime
from intent_bridge.runtime.execution import voice_tool_run_state


@pytest.mark.asyncio
async def test_successful_ha_action_can_return_silent_llm_response(monkeypatch):
    async def run(*_args, **_kwargs):
        voice_tool_run_state.last_successful_ha_action = {"calls": 1, "spoken": ""}
        return SimpleNamespace(final_output="", raw_responses=[])

    monkeypatch.setattr(settings.llm, "enabled", True)
    monkeypatch.setattr(settings.api, "action_confirmation", "")
    monkeypatch.setattr(runtime, "fallback_agent", object())
    monkeypatch.setattr(llm.Runner, "run", run)
    monkeypatch.setattr(llm, "seed_conversation_history", lambda *_args: _completed())

    response = await llm.process_llm_fallback(
        "turn it off",
        "conversation",
        client_history=[{"role": "user", "content": "turn it off"}],
    )

    assert response == ""


@pytest.mark.asyncio
async def test_empty_llm_response_without_successful_action_is_an_error(monkeypatch):
    async def run(*_args, **_kwargs):
        return SimpleNamespace(final_output="", raw_responses=[])

    monkeypatch.setattr(settings.llm, "enabled", True)
    monkeypatch.setattr(runtime, "fallback_agent", object())
    monkeypatch.setattr(llm.Runner, "run", run)
    monkeypatch.setattr(llm, "seed_conversation_history", lambda *_args: _completed())

    with pytest.raises(RuntimeError, match="returned no response"):
        await llm.process_llm_fallback(
            "say something",
            "conversation",
            client_history=[{"role": "user", "content": "say something"}],
        )


@pytest.mark.asyncio
async def test_informational_query_uses_isolated_agent_and_history(monkeypatch):
    informational_agent = object()
    captured = {}

    async def run(agent, agent_input, **kwargs):
        captured["agent"] = agent
        captured["input"] = agent_input
        captured.update(kwargs)
        return SimpleNamespace(final_output="The answer.", raw_responses=[object()])

    monkeypatch.setattr(settings.llm, "enabled", True)
    monkeypatch.setattr(settings.api, "timezone", "UTC")
    monkeypatch.setattr(settings.api, "locale", "en-GB")
    monkeypatch.setattr(settings.api, "location", "London, United Kingdom")
    monkeypatch.setattr(settings.api, "timezone_explicit", True)
    monkeypatch.setattr(settings.api, "locale_explicit", True)
    monkeypatch.setattr(settings.api, "location_explicit", True)
    monkeypatch.setattr(runtime, "informational_agent", informational_agent)
    monkeypatch.setattr(llm.Runner, "run", run)
    monkeypatch.setattr(llm, "seed_conversation_history", lambda *_args: _completed())

    response = await llm.process_informational_query(
        "Who is the president of France?",
        "conversation",
        client_history=[{"role": "user", "content": "Let's discuss France"}],
        origin_context={"area_name": "Office"},
    )

    assert response == "The answer."
    assert captured["agent"] is informational_agent
    assert captured["input"][0]["content"] == "Let's discuss France"
    assert "Latest user request: Who is the president of France?" in captured["input"][-1][
        "content"
    ]
    assert " in UTC" in captured["input"][-1]["content"]
    assert "locale=en-GB" in captured["input"][-1]["content"]
    assert "default geographic location=London, United Kingdom" in captured["input"][-1][
        "content"
    ]
    assert "Voice-origin room: Office" in captured["input"][-1]["content"]
    assert "not a city, region, or country" in captured["input"][-1]["content"]


@pytest.mark.asyncio
async def test_informational_query_requires_agent_and_nonempty_response(monkeypatch):
    monkeypatch.setattr(settings.llm, "enabled", True)
    monkeypatch.setattr(runtime, "informational_agent", None)
    with pytest.raises(RuntimeError, match="unavailable"):
        await llm.process_informational_query("Hello", "conversation")

    monkeypatch.setattr(runtime, "informational_agent", object())
    monkeypatch.setattr(
        llm.Runner,
        "run",
        lambda *_args, **_kwargs: _result(final_output=""),
    )
    monkeypatch.setattr(llm, "get_conversation_history", lambda *_args: _history())
    with pytest.raises(RuntimeError, match="returned no response"):
        await llm.process_informational_query("Hello", "conversation")


async def _completed():
    return None


async def _history():
    return []


async def _result(**kwargs):
    return SimpleNamespace(raw_responses=[], **kwargs)
