"""Recover custom MCP sessions that were closed by their remote server."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from agents.mcp import MCPServer
from anyio import ClosedResourceError

from intent_bridge.config import log

_INITIAL_BACKOFF_SECONDS = 1.0
_MAX_BACKOFF_SECONDS = 60.0


def _contains_closed_resource_error(error: BaseException) -> bool:
    """Return whether ``error`` contains an AnyIO closed-stream failure."""
    if isinstance(error, ClosedResourceError):
        return True
    if isinstance(error, BaseExceptionGroup):
        return any(_contains_closed_resource_error(inner) for inner in error.exceptions)
    return False


@dataclass(slots=True)
class _McpBackoff:
    failures: int
    retry_after: float
    suppression_logged: bool = False
    recovered: bool = False


class McpReconnectCoordinator:
    """Serialize and rate-limit reconnects shared by custom server adapters."""

    def __init__(
        self,
        manager: Any,
        *,
        initial_backoff_seconds: float = _INITIAL_BACKOFF_SECONDS,
        max_backoff_seconds: float = _MAX_BACKOFF_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if initial_backoff_seconds <= 0:
            raise ValueError("initial_backoff_seconds must be greater than zero")
        if max_backoff_seconds < initial_backoff_seconds:
            raise ValueError("max_backoff_seconds must not be less than initial_backoff_seconds")

        self._manager = manager
        self._initial_backoff_seconds = initial_backoff_seconds
        self._max_backoff_seconds = max_backoff_seconds
        self._clock = clock
        self._lock = asyncio.Lock()
        self._backoffs: dict[str, _McpBackoff] = {}

    async def should_defer(self, server: MCPServer) -> bool:
        """Return whether a failed server remains in its reconnect cooldown."""
        async with self._lock:
            state = self._backoffs.get(server.name)
            if state is None:
                return False

            if state.recovered:
                if self._clock() >= state.retry_after:
                    self._backoffs.pop(server.name, None)
                    log.info("MCP TRANSPORT RECOVERED server=%s", server.name)
                return False

            return self._should_defer_locked(server)

    async def reconnect(self, server: MCPServer) -> bool:
        """Recreate managed transports unless this server is in backoff."""
        async with self._lock:
            if self._should_defer_locked(server):
                return False

            state = self._backoffs.get(server.name)
            failures = (state.failures if state is not None else 0) + 1
            delay = min(
                self._initial_backoff_seconds * (2 ** (failures - 1)),
                self._max_backoff_seconds,
            )
            self._backoffs[server.name] = _McpBackoff(
                failures=failures,
                retry_after=self._clock() + delay,
            )

            if failures == 1:
                log.warning(
                    "MCP TRANSPORT CLOSED server=%s; reconnecting managed MCP connections "
                    "backoff_seconds=%.1f",
                    server.name,
                    delay,
                )
            else:
                log.info(
                    "MCP TRANSPORT RETRY server=%s failures=%d backoff_seconds=%.1f",
                    server.name,
                    failures,
                    delay,
                )

            try:
                await self._manager.reconnect(failed_only=False)
            except asyncio.CancelledError:
                raise
            except BaseException as error:
                self._log_reconnect_failure(server, failures, error)
                return False

            active_servers = getattr(self._manager, "active_servers", ())
            if server in active_servers:
                log.info("MCP TRANSPORT RECONNECTED server=%s", server.name)
                return True

            if failures == 1:
                log.warning(
                    "MCP TRANSPORT UNAVAILABLE AFTER RECONNECT server=%s",
                    server.name,
                )
            else:
                log.info(
                    "MCP TRANSPORT STILL UNAVAILABLE server=%s failures=%d",
                    server.name,
                    failures,
                )
            return False

    async def mark_healthy(self, server: MCPServer) -> None:
        """Record a successful tool discovery without resetting a fresh cooldown."""
        async with self._lock:
            state = self._backoffs.get(server.name)
            if state is None:
                return

            if self._clock() >= state.retry_after:
                self._backoffs.pop(server.name, None)
                log.info("MCP TRANSPORT RECOVERED server=%s", server.name)
            else:
                state.recovered = True

    def _should_defer_locked(self, server: MCPServer) -> bool:
        state = self._backoffs.get(server.name)
        if state is None:
            return False

        remaining = state.retry_after - self._clock()
        if remaining <= 0:
            return False

        if not state.suppression_logged:
            state.suppression_logged = True
            log.info(
                "MCP TRANSPORT BACKOFF server=%s retry_in_seconds=%.1f failures=%d",
                server.name,
                remaining,
                state.failures,
            )
        return True

    def _log_reconnect_failure(
        self,
        server: MCPServer,
        failures: int,
        error: BaseException,
    ) -> None:
        if failures == 1:
            log.warning(
                "MCP TRANSPORT RECONNECT FAILED server=%s error_type=%s",
                server.name,
                type(error).__name__,
                exc_info=True,
            )
        else:
            log.info(
                "MCP TRANSPORT RECONNECT FAILED server=%s failures=%d error_type=%s",
                server.name,
                failures,
                type(error).__name__,
            )


class ReconnectingMcpServer(MCPServer):
    """Expose a managed MCP server while repairing a closed session on tool discovery.

    The Agents SDK fetches each server's tool list before an agent run. That operation
    is safe to retry after rebuilding a closed transport. Tool calls are not retried:
    a remote server may have completed a mutating operation before its connection closed.
    """

    def __init__(self, server: MCPServer, coordinator: McpReconnectCoordinator) -> None:
        super().__init__(
            use_structured_content=server.use_structured_content,
            require_approval=False,
        )
        self._server = server
        self._coordinator = coordinator
        self.tool_meta_resolver = server.tool_meta_resolver
        self.custom_data_extractor = server.custom_data_extractor

    @property
    def name(self) -> str:
        return self._server.name

    @property
    def cached_tools(self) -> Any:
        return self._server.cached_tools

    async def connect(self) -> None:
        await self._server.connect()

    async def cleanup(self) -> None:
        await self._server.cleanup()

    async def list_tools(self, *args: Any, **kwargs: Any) -> Any:
        if await self._coordinator.should_defer(self._server):
            return []

        try:
            tools = await self._server.list_tools(*args, **kwargs)
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            if not _contains_closed_resource_error(error):
                raise
            if not await self._coordinator.reconnect(self._server):
                return []

            try:
                tools = await self._server.list_tools(*args, **kwargs)
            except asyncio.CancelledError:
                raise
            except BaseException as retry_error:
                if _contains_closed_resource_error(retry_error):
                    return []
                raise

        await self._coordinator.mark_healthy(self._server)
        return tools

    async def call_tool(self, *args: Any, **kwargs: Any) -> Any:
        try:
            return await self._server.call_tool(*args, **kwargs)
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            if _contains_closed_resource_error(error):
                await self._coordinator.reconnect(self._server)
            raise

    async def list_prompts(self, *args: Any, **kwargs: Any) -> Any:
        return await self._server.list_prompts(*args, **kwargs)

    async def get_prompt(self, *args: Any, **kwargs: Any) -> Any:
        return await self._server.get_prompt(*args, **kwargs)

    async def list_resources(self, *args: Any, **kwargs: Any) -> Any:
        return await self._server.list_resources(*args, **kwargs)

    async def list_resource_templates(self, *args: Any, **kwargs: Any) -> Any:
        return await self._server.list_resource_templates(*args, **kwargs)

    async def read_resource(self, *args: Any, **kwargs: Any) -> Any:
        return await self._server.read_resource(*args, **kwargs)

    def _get_needs_approval_for_tool(self, *args: Any, **kwargs: Any) -> Any:
        return self._server._get_needs_approval_for_tool(*args, **kwargs)

    def _get_failure_error_function(self, *args: Any, **kwargs: Any) -> Any:
        return self._server._get_failure_error_function(*args, **kwargs)


__all__ = ["McpReconnectCoordinator", "ReconnectingMcpServer"]
