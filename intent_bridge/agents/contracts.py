"""Dependency-free contract for extending the fallback agent."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class AgentToolPlugin:
    """A named capability bundle contributed to the fallback agent."""

    name: str
    tools: tuple[Any, ...]
    instructions: str = ""


__all__ = ["AgentToolPlugin"]
