from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from intent_bridge import application
from intent_bridge.config import settings
from intent_bridge.core.voice import VoicePipelineError, VoiceResult
from intent_bridge.sounds import controller as sounds


@pytest.fixture
def client():
    states = {
        "assist_satellite.office": {"state": "idle", "attributes": {}},
        "media_player.office_speaker": {
            "state": "idle",
            "attributes": {
                "friendly_name": "Office satellite speaker",
                "device_class": "speaker",
            },
        },
        "media_player.office_tv": {
            "state": "off",
            "attributes": {"friendly_name": "Office TV", "device_class": "tv"},
        },
    }
    registry = {
        "assist_satellite.office": {"di": "sat", "ai": "office"},
        "media_player.office_speaker": {"di": "audio", "ai": "office"},
        "media_player.office_tv": {"di": "tv", "ai": "office"},
    }
    devices = {
        "hub": {"name": "Office voice hub"},
        "sat": {"name": "Assistant", "via_device_id": "hub", "area_id": "office"},
        "audio": {"name": "Audio", "via_device_id": "hub", "area_id": "office"},
        "tv": {"name": "TV", "via_device_id": "hub", "area_id": "office"},
    }

    def entity_context(entity_id, _state):
        entry = registry.get(entity_id, {})
        device = devices.get(entry.get("di"), {})
        area_id = entry.get("ai") or device.get("area_id")
        return {"area_id": area_id, "area_name": "Office" if area_id == "office" else None}

    return SimpleNamespace(
        states=states,
        entity_registry=registry,
        devices=devices,
        refresh_registries=AsyncMock(),
        resolve_device_origin=lambda **kwargs: {
            "device_id": kwargs.get("device_id"),
            "area_id": kwargs.get("area_id"),
            "area_name": kwargs.get("area_name"),
        },
        resolve_area_reference=lambda area_id=None, area_name=None: (
            area_id or "office",
            area_name or "Office",
        ),
        _entity_context=entity_context,
        command=AsyncMock(return_value={"success": True, "result": {}}),
    )


@pytest.mark.asyncio
async def test_resolves_satellite_speaker_and_sends_absolute_media_url(client, monkeypatch):
    monkeypatch.setattr(settings.assistant, "sounds_enabled", True)
    monkeypatch.setattr(settings.api, "base_url", "https://bridge.example/proxy")
    monkeypatch.setattr(sounds, "_require_ha_ws", AsyncMock(return_value=client))

    target = await sounds.resolve_assistant_sound_target(
        {"device_id": "sat", "area_id": "office"}
    )

    assert target.entity_id == "media_player.office_speaker"
    assert "sibling_device" in target.match_reason
    assert await sounds.AssistantSounds().play(target, "processing") is True
    payload = client.command.await_args.args[0]
    assert payload["target"] == {"entity_id": "media_player.office_speaker"}
    assert payload["service_data"] == {
        "media_content_id": "https://bridge.example/proxy/assistant/sounds/processing.mp3",
        "media_content_type": "audio/mpeg",
    }


@pytest.mark.asyncio
async def test_sound_resolution_is_optional_conservative_and_failure_safe(client, monkeypatch):
    monkeypatch.setattr(sounds, "_require_ha_ws", AsyncMock(return_value=client))
    monkeypatch.setattr(settings.assistant, "sounds_enabled", False)
    assert await sounds.resolve_assistant_sound_target({"device_id": "sat"}) is None

    monkeypatch.setattr(settings.assistant, "sounds_enabled", True)
    client.states["assist_satellite.second"] = {"state": "idle", "attributes": {}}
    client.entity_registry["assist_satellite.second"] = {"di": "second", "ai": "office"}
    client.devices["second"] = {"name": "Second", "area_id": "office"}
    assert await sounds.resolve_assistant_sound_target({"area_name": "Office"}) is None

    manager = sounds.AssistantSounds()
    client.command.return_value = {"success": False, "error": "unavailable"}
    target = sounds.AssistantSoundTarget("media_player.x", "assist_satellite.x", "x", "test")
    assert await manager.play(target, "error") is False
    assert "unavailable" in manager.last_error


def test_sound_files_are_served_and_unknown_names_are_hidden():
    created = application.create_app(SimpleNamespace(handle=AsyncMock()))
    client = TestClient(created)
    response = client.get("/assistant/sounds/success.mp3")
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/mpeg"
    assert response.content[:3] == b"ID3"
    assert client.get("/assistant/sounds/other.mp3").status_code == 404


def test_enabled_sounds_preserve_default_confirmation_without_success_sound(monkeypatch):
    class Pipeline:
        async def handle(self, _request):
            return VoiceResult(speech="Done.", route="test")

    handle = object()
    begin = AsyncMock(return_value=handle)
    complete = AsyncMock()
    monkeypatch.setattr(settings.assistant, "sounds_enabled", True)
    monkeypatch.setattr(settings.api, "action_confirmation", "Done.")
    monkeypatch.setattr(application.assistant_feedback, "begin", begin)
    monkeypatch.setattr(application.assistant_feedback, "complete", complete)

    response = TestClient(application.create_app(Pipeline())).post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "turn it on"}]},
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "Done."
    begin.assert_awaited_once_with(None, led=False, sounds=True)
    complete.assert_awaited_once_with(
        handle,
        success=True,
        play_terminal_sound=False,
    )


def test_enabled_sounds_play_success_for_blank_response(monkeypatch):
    class Pipeline:
        async def handle(self, _request):
            return VoiceResult(speech="   ", route="test")

    handle = object()
    complete = AsyncMock()
    monkeypatch.setattr(settings.assistant, "sounds_enabled", True)
    monkeypatch.setattr(
        application.assistant_feedback,
        "begin",
        AsyncMock(return_value=handle),
    )
    monkeypatch.setattr(application.assistant_feedback, "complete", complete)

    response = TestClient(application.create_app(Pipeline())).post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "turn it on"}]},
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == ""
    complete.assert_awaited_once_with(
        handle,
        success=True,
        play_terminal_sound=True,
    )


def test_enabled_sounds_preserve_non_default_text_without_success_sound(monkeypatch):
    class Pipeline:
        async def handle(self, _request):
            return VoiceResult(
                speech="Pulp Fiction was directed by Quentin Tarantino.",
                route="llm-ha-ws",
            )

    handle = object()
    complete = AsyncMock()
    monkeypatch.setattr(settings.assistant, "sounds_enabled", True)
    monkeypatch.setattr(settings.api, "action_confirmation", "Done.")
    monkeypatch.setattr(
        application.assistant_feedback,
        "begin",
        AsyncMock(return_value=handle),
    )
    monkeypatch.setattr(application.assistant_feedback, "complete", complete)

    response = TestClient(application.create_app(Pipeline())).post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "who directed Pulp Fiction?"}]},
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == (
        "Pulp Fiction was directed by Quentin Tarantino."
    )
    complete.assert_awaited_once_with(
        handle,
        success=True,
        play_terminal_sound=False,
    )


def test_failure_response_plays_error_sound(monkeypatch):
    class Pipeline:
        async def handle(self, _request):
            return VoiceResult(speech="Sorry.", route="voice-error-response")

    handle = object()
    complete = AsyncMock()
    monkeypatch.setattr(settings.assistant, "sounds_enabled", True)
    monkeypatch.setattr(
        application.assistant_feedback, "begin", AsyncMock(return_value=handle)
    )
    monkeypatch.setattr(application.assistant_feedback, "complete", complete)

    response = TestClient(application.create_app(Pipeline())).post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "fail"}]},
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == ""
    complete.assert_awaited_once_with(
        handle,
        success=False,
        play_terminal_sound=True,
    )


def test_pipeline_exception_plays_error_sound(monkeypatch):
    class Pipeline:
        async def handle(self, _request):
            raise VoicePipelineError(())

    handle = object()
    complete = AsyncMock()
    monkeypatch.setattr(settings.assistant, "sounds_enabled", True)
    monkeypatch.setattr(
        application.assistant_feedback, "begin", AsyncMock(return_value=handle)
    )
    monkeypatch.setattr(application.assistant_feedback, "complete", complete)

    response = TestClient(application.create_app(Pipeline())).post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "explode"}]},
    )
    assert response.status_code == 502
    complete.assert_awaited_once_with(handle, success=False)
