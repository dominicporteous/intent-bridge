import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from intent_bridge import application
from intent_bridge.config import settings
from intent_bridge.runtime.dependencies import RuntimeState, runtime


@pytest.mark.asyncio
async def test_lifespan_minimal_startup_shutdown(monkeypatch):
    monkeypatch.setattr(settings.llm, "enabled", False)
    monkeypatch.setattr(settings.music_assistant, "enabled", False)
    monkeypatch.setattr(application.voice_activity_indicators, "stop_all", AsyncMock())

    async with application.lifespan(application.app):
        pass

    application.voice_activity_indicators.stop_all.assert_awaited_once()


@pytest.mark.asyncio
async def test_lifespan_full_integrations(monkeypatch):
    monkeypatch.setattr(settings.llm, "enabled", True)
    monkeypatch.setattr(settings.home_assistant.websocket, "enabled", True)
    monkeypatch.setattr(settings.home_assistant, "base_url", "http://ha")
    monkeypatch.setattr(settings.home_assistant, "access_token", "token")
    monkeypatch.setattr(settings.music_assistant, "enabled", True)
    monkeypatch.setattr(settings.home_assistant.advanced, "enabled", True)
    monkeypatch.setattr(application, "validate_fallback_config", lambda: [])
    monkeypatch.setattr(application, "validate_music_assistant_config", lambda: [])
    monkeypatch.setattr(application, "MusicAssistantClient", object())

    ws = SimpleNamespace(
        start=AsyncMock(),
        stop=AsyncMock(),
        ready=asyncio.Event(),
        states={},
        services={},
        entity_registry={},
        devices={},
        areas={},
    )
    ws.ready.set()
    monkeypatch.setattr(application, "HomeAssistantWebSocket", lambda *args: ws)
    native = SimpleNamespace(start=AsyncMock(return_value=True), stop=AsyncMock(), connected=True)
    monkeypatch.setattr(application, "NativeMusicAssistant", lambda *args: native)
    monkeypatch.setattr(application, "make_ha_mcp_server", lambda: "server")
    active = SimpleNamespace(name="Home Assistant Advanced")

    class Manager:
        def __init__(self, *args, **kwargs):
            self.active_servers = [active]
            self.failed_servers = []
            self.errors = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

    monkeypatch.setattr(application, "MCPServerManager", Manager)
    monkeypatch.setattr(application, "make_advanced_agent", lambda servers: "advanced")
    monkeypatch.setattr(application, "make_fallback_agent", lambda music_enabled: "fallback")
    monkeypatch.setattr(application.voice_activity_indicators, "stop_all", AsyncMock())
    async with application.lifespan(application.app):
        assert runtime.ha_ws is ws
        assert runtime.music_assistant is native
        assert runtime.advanced_agent == "advanced"
        assert runtime.fallback_agent == "fallback"
    ws.stop.assert_awaited_once()
    native.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_lifespan_starts_ha_catalog_when_llm_is_disabled(monkeypatch):
    monkeypatch.setattr(settings.llm, "enabled", False)
    monkeypatch.setattr(settings.home_assistant.websocket, "enabled", True)
    monkeypatch.setattr(settings.home_assistant, "base_url", "http://ha")
    monkeypatch.setattr(settings.home_assistant, "access_token", "token")
    monkeypatch.setattr(settings.music_assistant, "enabled", False)
    ws = SimpleNamespace(start=AsyncMock(), stop=AsyncMock(), ready=asyncio.Event())
    monkeypatch.setattr(application, "HomeAssistantWebSocket", lambda *args: ws)
    monkeypatch.setattr(application.voice_activity_indicators, "stop_all", AsyncMock())

    async with application.lifespan(application.app):
        assert runtime.ha_ws is ws
        ws.start.assert_awaited_once()
    ws.stop.assert_awaited_once()



@pytest.mark.asyncio
async def test_lifespan_missing_fallback_config(monkeypatch):
    monkeypatch.setattr(settings.llm, "enabled", True)
    monkeypatch.setattr(application, "validate_fallback_config", lambda: ["TOKEN"])
    monkeypatch.setattr(settings.music_assistant, "enabled", True)
    monkeypatch.setattr(application, "validate_music_assistant_config", lambda: ["MA_TOKEN"])
    monkeypatch.setattr(application.voice_activity_indicators, "stop_all", AsyncMock())
    async with application.lifespan(application.app):
        assert runtime.fallback_agent is None


def test_runtime_state_has_one_owner_for_integrations():
    state = RuntimeState(ha_ws="ws", music_assistant="music", fallback_agent="agent")
    state.clear_integrations()
    assert state.ha_ws is None
    assert state.music_assistant is None
    assert state.fallback_agent is None
