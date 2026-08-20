import json

import pytest
from agents.mcp import MCPServerSse, MCPServerStdio, MCPServerStreamableHttp

from intent_bridge.mcp_config import (
    McpConfigurationError,
    load_mcp_servers,
    mcp_agent_instructions,
)


def test_missing_mcp_config_is_optional(tmp_path):
    assert load_mcp_servers(tmp_path / "missing.json") == ()
    empty = tmp_path / "empty.json"
    empty.write_text("", encoding="utf-8")
    assert load_mcp_servers(empty) == ()


def test_loads_active_http_servers_and_skips_inactive_entries(tmp_path):
    path = tmp_path / "mcp.json"
    path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "web_search": {
                        "name": "Web Search MCP",
                        "type": "streamableHttp",
                        "description": "Search current web content",
                        "isActive": True,
                        "baseUrl": "http://localhost:3000/mcp",
                        "headers": {"Authorization": "Bearer test"},
                    },
                    "old": {"isActive": False},
                    "events": {
                        "type": "sse",
                        "baseUrl": "https://example.test/sse",
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    configured = load_mcp_servers(path, client_session_timeout_seconds=45)

    assert [item.key for item in configured] == ["web_search", "events"]
    assert isinstance(configured[0].server, MCPServerStreamableHttp)
    assert configured[0].server.params["url"] == "http://localhost:3000/mcp"
    assert configured[0].server.params["headers"] == {"Authorization": "Bearer test"}
    assert configured[0].server.client_session_timeout_seconds == 45
    assert isinstance(configured[1].server, MCPServerSse)
    assert configured[1].server.client_session_timeout_seconds == 45
    instructions = mcp_agent_instructions(configured)
    assert "Web Search MCP: Search current web content" in instructions


def test_loads_stdio_server(tmp_path):
    path = tmp_path / "mcp.json"
    path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "local": {
                        "type": "stdio",
                        "command": "python",
                        "args": ["-m", "example"],
                        "env": {"TOKEN": "test"},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    configured = load_mcp_servers(path, client_session_timeout_seconds=30)

    assert isinstance(configured[0].server, MCPServerStdio)
    assert configured[0].server.params.command == "python"
    assert configured[0].server.params.args == ["-m", "example"]
    assert configured[0].server.client_session_timeout_seconds == 30


@pytest.mark.parametrize(
    "document",
    [
        {},
        {"mcpServers": {"bad": {"type": "streamableHttp"}}},
        {"mcpServers": {"bad": {"type": "socket", "baseUrl": "http://localhost"}}},
        {"mcpServers": {"bad": {"type": "sse", "baseUrl": "relative"}}},
    ],
)
def test_rejects_invalid_active_servers(tmp_path, document):
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(McpConfigurationError):
        load_mcp_servers(path)
