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


async def _completed():
    return None
