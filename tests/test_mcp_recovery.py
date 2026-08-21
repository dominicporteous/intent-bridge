from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from anyio import ClosedResourceError

from intent_bridge.mcp_recovery import McpReconnectCoordinator, ReconnectingMcpServer


def _server(*, list_tools: AsyncMock, call_tool: AsyncMock | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        name="Web Search MCP",
        use_structured_content=False,
        tool_meta_resolver=None,
        custom_data_extractor=None,
        cached_tools=None,
        list_tools=list_tools,
        call_tool=call_tool or AsyncMock(),
    )


@pytest.mark.asyncio
async def test_closed_session_during_tool_discovery_reconnects_and_limits_flapping():
    now = 0.0
    expected_tools = [object()]
    server = _server(
        list_tools=AsyncMock(
            side_effect=[ClosedResourceError(), expected_tools, ClosedResourceError()]
        )
    )
    manager = SimpleNamespace(active_servers=[server], reconnect=AsyncMock())
    adapter = ReconnectingMcpServer(
        server,
        McpReconnectCoordinator(manager, clock=lambda: now),
    )

    assert await adapter.list_tools() == expected_tools
    assert await adapter.list_tools() == []
    manager.reconnect.assert_awaited_once_with(failed_only=False)
    assert server.list_tools.await_count == 3


@pytest.mark.asyncio
async def test_failed_recovery_uses_per_server_exponential_backoff():
    now = 0.0
    server = _server(list_tools=AsyncMock(side_effect=ClosedResourceError()))
    manager = SimpleNamespace(active_servers=[], reconnect=AsyncMock())
    adapter = ReconnectingMcpServer(
        server,
        McpReconnectCoordinator(
            manager,
            initial_backoff_seconds=1,
            max_backoff_seconds=60,
            clock=lambda: now,
        ),
    )

    assert await adapter.list_tools() == []
    assert await adapter.list_tools() == []
    assert manager.reconnect.await_count == 1
    assert server.list_tools.await_count == 1

    now = 1.0
    assert await adapter.list_tools() == []
    assert manager.reconnect.await_count == 2
    assert server.list_tools.await_count == 2

    now = 2.9
    assert await adapter.list_tools() == []
    assert manager.reconnect.await_count == 2
    assert server.list_tools.await_count == 2

    now = 3.0
    assert await adapter.list_tools() == []
    assert manager.reconnect.await_count == 3
    assert server.list_tools.await_count == 3


@pytest.mark.asyncio
async def test_closed_session_during_tool_call_reconnects_without_replaying_call():
    server = _server(
        list_tools=AsyncMock(),
        call_tool=AsyncMock(side_effect=ClosedResourceError()),
    )
    manager = SimpleNamespace(active_servers=[server], reconnect=AsyncMock())
    adapter = ReconnectingMcpServer(server, McpReconnectCoordinator(manager))

    with pytest.raises(ClosedResourceError):
        await adapter.call_tool("search", {"query": "Super Bowl 2002"})

    manager.reconnect.assert_awaited_once_with(failed_only=False)
    server.call_tool.assert_awaited_once()
