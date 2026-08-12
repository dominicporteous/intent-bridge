"""Conservative routing gate for general information and conversation."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field

from intent_bridge.core.voice import RouteDeclined, VoiceRequest

_HOME_TERMS = re.compile(
    r"\b(?:"
    r"light|lights|lamp|switch|plug|socket|thermostat|heating|heater|radiator|"
    r"air\s*condition(?:er|ing)?|fan|blind|blinds|curtain|curtains|cover|lock|"
    r"door|garage|vacuum|speaker|music|song|album|playlist|volume|media|tv|"
    r"television|kettle|sensor|automation|script|scene|timer|alarm|camera|"
    r"device|entity|home\s+assistant"
    r")\b",
    re.IGNORECASE,
)
_HOME_ACTIONS = re.compile(
    r"\b(?:turn|switch|set|dim|brighten|open|close|lock|unlock|start|stop|"
    r"pause|resume|skip|queue|mute|unmute|play)\b",
    re.IGNORECASE,
)
_GENERAL_OPENING = re.compile(
    r"^(?:who(?:'s|\s+is|\s+was|\s+are|\s+were)?|"
    r"what(?:'s|\s+is|\s+was|\s+are|\s+were|\s+does|\s+do|\s+did)?|"
    r"when|where|why|how|which|is|are|was|were|do|does|did|"
    r"can you|could you)\b",
    re.IGNORECASE,
)
_GENERAL_COMMAND = re.compile(
    r"^(?:please\s+)?(?:tell me|explain|define|describe|summari[sz]e|translate|"
    r"spell|calculate|"
    r"convert|recommend|give me (?:a |an )?(?:fact|joke|story)|say something)\b",
    re.IGNORECASE,
)
_CONVERSATION = re.compile(
    r"^(?:hi|hello|hey|good (?:morning|afternoon|evening)|thanks|thank you|"
    r"how are you|what can you do|who are you)\b",
    re.IGNORECASE,
)
_FOLLOW_UP = re.compile(
    r"^(?:(?:and|also)\b|(?:what|how) about\b|"
    r"(?:why|why is that|tell me more|go on|continue|really)\??$)",
    re.IGNORECASE,
)


def is_informational_or_conversational(
    text: str,
    history: Sequence[dict[str, str]] = (),
) -> bool:
    """Return true only when a request has clear non-household intent."""
    normalized = " ".join(text.strip().split())
    if not normalized:
        return False
    if _HOME_TERMS.search(normalized) or _HOME_ACTIONS.search(normalized):
        return False
    if _FOLLOW_UP.search(normalized):
        previous_user = next(
            (
                str(message.get("content", ""))
                for message in reversed(history)
                if message.get("role") == "user" and message.get("content")
            ),
            "",
        )
        return bool(previous_user) and is_informational_or_conversational(previous_user)
    return bool(
        _GENERAL_OPENING.search(normalized)
        or _GENERAL_COMMAND.search(normalized)
        or _CONVERSATION.search(normalized)
    )


@dataclass(frozen=True, slots=True)
class InformationalVoiceRoute:
    """Delegate clearly general requests to an isolated LLM agent."""

    name: str
    handler: Callable[[VoiceRequest], Awaitable[str]] = field(repr=False)

    async def handle(self, request: VoiceRequest) -> str:
        if not is_informational_or_conversational(request.text, request.client_history):
            raise RouteDeclined("Not an informational or conversational request")
        response = await self.handler(request)
        if response is None:
            raise RuntimeError("informational route returned no response")
        return str(response).strip()


__all__ = [
    "InformationalVoiceRoute",
    "is_informational_or_conversational",
]
