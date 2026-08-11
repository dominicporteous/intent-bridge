"""Load user-defined MCP servers from the repository MCP configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from agents.mcp import MCPServerSse, MCPServerStdio, MCPServerStreamableHttp


class McpConfigurationError(ValueError):
    """Raised when an active custom MCP server has invalid configuration."""


@dataclass(frozen=True, slots=True)
class ConfiguredMcpServer:
    key: str
    description: str
    server: Any


def load_mcp_servers(path: Path) -> tuple[ConfiguredMcpServer, ...]:
    """Build active MCP transports from a Codex-style ``mcpServers`` object."""
    if not path.is_file():
        return ()

    try:
        raw_document = path.read_text(encoding="utf-8")
        if not raw_document.strip():
            return ()
        document = json.loads(raw_document)
    except (OSError, json.JSONDecodeError) as exc:
        raise McpConfigurationError(f"Could not read {path}: {exc}") from exc

    if not isinstance(document, dict) or not isinstance(document.get("mcpServers"), dict):
        raise McpConfigurationError(f"{path} must contain an mcpServers object")

    configured: list[ConfiguredMcpServer] = []
    names: set[str] = set()
    for key, raw in document["mcpServers"].items():
        if not isinstance(key, str) or not key.strip():
            raise McpConfigurationError("MCP server keys must be non-empty strings")
        if not isinstance(raw, dict):
            raise McpConfigurationError(f"MCP server {key!r} must be an object")
        if raw.get("isActive", True) is False:
            continue
        if not isinstance(raw.get("isActive", True), bool):
            raise McpConfigurationError(f"MCP server {key!r} isActive must be a boolean")

        name = _optional_string(raw, "name") or key
        if name in names or name == "Home Assistant Advanced":
            raise McpConfigurationError(f"MCP server name {name!r} must be unique")
        names.add(name)
        description = _optional_string(raw, "description") or ""
        transport = _required_string(raw, "type", key)
        normalized_type = transport.replace("-", "").replace("_", "").casefold()

        common = {
            "name": name,
            "cache_tools_list": _optional_boolean(raw, "cacheToolsList", True, key),
        }
        if normalized_type == "streamablehttp":
            params = _http_params(raw, key)
            server = MCPServerStreamableHttp(params=params, **common)
        elif normalized_type == "sse":
            params = _http_params(raw, key)
            server = MCPServerSse(params=params, **common)
        elif normalized_type == "stdio":
            params = _stdio_params(raw, key)
            server = MCPServerStdio(params=params, **common)
        else:
            raise McpConfigurationError(
                f"MCP server {key!r} has unsupported type {transport!r}; "
                "use streamableHttp, sse, or stdio"
            )
        configured.append(ConfiguredMcpServer(key, description, server))

    return tuple(configured)


def mcp_agent_instructions(servers: tuple[ConfiguredMcpServer, ...]) -> str:
    """Describe configured integrations without assuming their tool names."""
    lines = []
    for configured in servers:
        detail = f": {configured.description}" if configured.description else ""
        lines.append(f"- {configured.server.name}{detail}")
    if not lines:
        return ""
    return (
        "CUSTOM MCP TOOLS\n\n"
        "The following custom MCP integrations are available. Use their tools when "
        "they are relevant to the user's request:\n\n" + "\n".join(lines)
    )


def _http_params(raw: dict[str, Any], key: str) -> dict[str, Any]:
    url = _optional_string(raw, "baseUrl") or _optional_string(raw, "url")
    if not url:
        raise McpConfigurationError(f"MCP server {key!r} requires baseUrl")
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise McpConfigurationError(f"MCP server {key!r} baseUrl must be an absolute HTTP URL")

    params: dict[str, Any] = {"url": url}
    headers = raw.get("headers")
    if headers is not None:
        if not isinstance(headers, dict) or not all(
            isinstance(name, str) and isinstance(value, str) for name, value in headers.items()
        ):
            raise McpConfigurationError(f"MCP server {key!r} headers must map strings to strings")
        params["headers"] = headers
    for source, target in (
        ("timeoutSeconds", "timeout"),
        ("sseReadTimeoutSeconds", "sse_read_timeout"),
    ):
        if source in raw:
            params[target] = _positive_number(raw[source], source, key)
    if "terminateOnClose" in raw:
        params["terminate_on_close"] = _optional_boolean(
            raw, "terminateOnClose", True, key
        )
    return params


def _stdio_params(raw: dict[str, Any], key: str) -> dict[str, Any]:
    params: dict[str, Any] = {"command": _required_string(raw, "command", key)}
    args = raw.get("args", [])
    if not isinstance(args, list) or not all(isinstance(item, str) for item in args):
        raise McpConfigurationError(f"MCP server {key!r} args must be a string array")
    params["args"] = args
    env = raw.get("env")
    if env is not None:
        if not isinstance(env, dict) or not all(
            isinstance(name, str) and isinstance(value, str) for name, value in env.items()
        ):
            raise McpConfigurationError(f"MCP server {key!r} env must map strings to strings")
        params["env"] = env
    cwd = _optional_string(raw, "cwd")
    if cwd:
        params["cwd"] = cwd
    return params


def _required_string(raw: dict[str, Any], field: str, key: str) -> str:
    value = _optional_string(raw, field)
    if not value:
        raise McpConfigurationError(f"MCP server {key!r} requires {field}")
    return value


def _optional_string(raw: dict[str, Any], field: str) -> str | None:
    value = raw.get(field)
    if value is None:
        return None
    if not isinstance(value, str):
        raise McpConfigurationError(f"MCP field {field!r} must be a string")
    return value.strip() or None


def _optional_boolean(raw: dict[str, Any], field: str, default: bool, key: str) -> bool:
    value = raw.get(field, default)
    if not isinstance(value, bool):
        raise McpConfigurationError(f"MCP server {key!r} {field} must be a boolean")
    return value


def _positive_number(value: Any, field: str, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise McpConfigurationError(f"MCP server {key!r} {field} must be greater than zero")
    return float(value)


__all__ = [
    "ConfiguredMcpServer",
    "McpConfigurationError",
    "load_mcp_servers",
    "mcp_agent_instructions",
]
