"""Single owner for mutable process runtime dependencies."""

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

_runtime_overrides: ContextVar[dict[str, Any] | None] = ContextVar(
    "runtime_overrides",
    default=None,
)
_CONTEXT_OVERRIDABLE_FIELDS = frozenset({"ha_ws", "fallback_agent", "fallback_lock"})


@dataclass(slots=True)
class RuntimeState:
    """Resources created by the composition root and consumed by adapters."""

    event_loop: asyncio.AbstractEventLoop | None = None
    ha_ws: Any | None = None
    music_assistant: Any | None = None
    advanced_agent: Any | None = None
    informational_agent: Any | None = None
    fallback_agent: Any | None = None
    mcp_manager: Any | None = None
    fallback_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def __getattribute__(self, name: str) -> Any:
        if name in _CONTEXT_OVERRIDABLE_FIELDS:
            overrides = _runtime_overrides.get()
            if overrides is not None and name in overrides:
                return overrides[name]
        return object.__getattribute__(self, name)

    @contextmanager
    def override(self, **values: Any) -> Iterator[None]:
        """Temporarily bind request-local integrations for concurrent adapters."""

        unknown = set(values) - _CONTEXT_OVERRIDABLE_FIELDS
        if unknown:
            raise ValueError(f"unsupported runtime override(s): {sorted(unknown)!r}")
        current = _runtime_overrides.get()
        merged = dict(current or {})
        merged.update(values)
        token = _runtime_overrides.set(merged)
        try:
            yield
        finally:
            _runtime_overrides.reset(token)

    def clear_integrations(self) -> None:
        self.ha_ws = None
        self.music_assistant = None
        self.advanced_agent = None
        self.informational_agent = None
        self.fallback_agent = None
        self.mcp_manager = None


runtime = RuntimeState()


__all__ = ["RuntimeState", "runtime"]
