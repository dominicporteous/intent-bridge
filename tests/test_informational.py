import pytest

from intent_bridge.core.voice import RouteDeclined, VoiceRequest
from intent_bridge.informational import (
    InformationalVoiceRoute,
    is_informational_or_conversational,
)


@pytest.mark.parametrize(
    "text",
    [
        "Who's the president?",
        "What is the capital of France?",
        "Why is the sky blue?",
        "Tell me a joke",
        "Hello",
        "Translate hello into French",
        "Is Pluto a planet?",
        "Could you explain gravity?",
    ],
)
def test_gate_accepts_clear_general_requests(text):
    assert is_informational_or_conversational(text)


@pytest.mark.parametrize(
    "text",
    [
        "Why is the bedroom light on?",
        "What is the thermostat set to?",
        "Play some music",
        "Turn it off",
        "List unavailable devices",
        "What's my calendar like next week?",
        "What is the weather tomorrow?",
        "Tell me about my appointments",
    ],
)
def test_gate_rejects_household_and_media_requests(text):
    assert not is_informational_or_conversational(text)


def test_gate_treats_unpunctuated_general_openings_consistently():
    assert is_informational_or_conversational("whats the capital of France")
    assert not is_informational_or_conversational("whats my calendar like next week")


def test_gate_uses_general_conversation_context_for_follow_up():
    history = [
        {"role": "user", "content": "Who is the president of France?"},
        {"role": "assistant", "content": "Emmanuel Macron."},
    ]
    assert is_informational_or_conversational("What about Germany?", history)
    assert not is_informational_or_conversational(
        "What about the bedroom?",
        [{"role": "user", "content": "Turn on the kitchen light"}],
    )


@pytest.mark.asyncio
async def test_informational_route_delegates_only_accepted_requests():
    calls = []

    async def handler(request):
        calls.append(request.text)
        return "Answer."

    route = InformationalVoiceRoute("informational-llm", handler)
    assert await route.handle(VoiceRequest("Who's the president?", "c")) == "Answer."
    with pytest.raises(RouteDeclined):
        await route.handle(VoiceRequest("Turn on the light", "c"))
    assert calls == ["Who's the president?"]


@pytest.mark.asyncio
async def test_informational_route_rejects_missing_handler_response():
    async def handler(_request):
        return None

    route = InformationalVoiceRoute("informational-llm", handler)
    with pytest.raises(RuntimeError, match="returned no response"):
        await route.handle(VoiceRequest("Hello", "c"))
