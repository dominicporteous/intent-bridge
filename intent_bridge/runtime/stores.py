"""In-process stores owned by the application lifecycle."""

import asyncio
from dataclasses import dataclass


@dataclass(slots=True)
class PendingRequest:
    response_future: asyncio.Future[str]
    intent_future: asyncio.Future[dict]
    original_text: str
    normalized_text: str
    request_id: str
    created_at: float


@dataclass(slots=True)
class ConversationMemory:
    messages: list[dict[str, str]]
    updated_at: float


pending_requests: dict[str, PendingRequest] = {}
conversation_memories: dict[str, ConversationMemory] = {}
conversation_history_lock = asyncio.Lock()
recent_music_action_responses: dict[str, tuple[float, str]] = {}


__all__ = [
    "ConversationMemory",
    "PendingRequest",
    "conversation_history_lock",
    "conversation_memories",
    "pending_requests",
    "recent_music_action_responses",
]
