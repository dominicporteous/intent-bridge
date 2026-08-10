"""Single owner for mutable process runtime dependencies."""

import asyncio
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RuntimeState:
    """Resources created by the composition root and consumed by adapters."""

    event_loop: asyncio.AbstractEventLoop | None = None
    ha_ws: Any | None = None
    music_assistant: Any | None = None
    advanced_agent: Any | None = None
    fallback_agent: Any | None = None
    mcp_manager: Any | None = None
    fallback_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def clear_integrations(self) -> None:
        self.ha_ws = None
        self.music_assistant = None
        self.advanced_agent = None
        self.fallback_agent = None
        self.mcp_manager = None


runtime = RuntimeState()


__all__ = ["RuntimeState", "runtime"]
