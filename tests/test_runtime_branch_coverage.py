from __future__ import annotations

from types import SimpleNamespace

import pytest

from intent_bridge.agents import results
from intent_bridge.config import settings
from intent_bridge.core import tool_output
from intent_bridge.runtime import execution


class _Dumpable:
    def __init__(self, value):
        self.value = value

    def model_dump(self):
        return self.value


class _BrokenDump:
    def model_dump(self):
        raise RuntimeError("cannot dump")

    def __str__(self):
        return "broken-dump"


def _result(name: str | None, output):
    tool = None if name is None else SimpleNamespace(name=name)
    return SimpleNamespace(tool=tool, output=output)


def test_tool_output_serialisation_handles_sdk_models_and_fallbacks():
    assert tool_output.serialise_tool_output(_Dumpable({"answer": 42})) == '{"answer": 42}'
    assert tool_output.serialise_tool_output(_BrokenDump()) == "broken-dump"

    circular = {}
    circular["self"] = circular
    assert tool_output.serialise_tool_output(circular) == "{'self': {...}}"


def test_tool_output_mapping_handles_sdk_models_nested_fallbacks_and_unknown_values():
    assert tool_output.tool_output_mapping(_Dumpable({"ok": True})) == {"ok": True}
    assert tool_output.tool_output_mapping(_BrokenDump()) is None
    original = {"text": "not json", "status": "kept"}
    assert tool_output.tool_output_mapping(original) is original
    assert tool_output.tool_output_mapping([None, "not json", 17]) is None
    assert tool_output.tool_output_mapping(17) is None


@pytest.mark.parametrize(
    "output",
    [
        {"detail": "Forbidden"},
        {"error": "boom"},
        "error calling tool upstream",
        "ToolError: broken",
        "unauthorized request",
        "invalid parameter x",
    ],
)
def test_tool_output_failure_fallback_markers(output):
    assert tool_output.tool_output_failed(output)


def test_tool_output_mapping_without_failure_is_not_failed():
    assert not tool_output.tool_output_failed({"success": None, "error": ""})


def test_voice_run_state_initialisation_proxy_and_reset():
    explicit_entities = {"light": "light.desk"}
    explicit = execution.VoiceToolRunState(last_entity_by_domain=explicit_entities)
    assert explicit.last_entity_by_domain is explicit_entities

    token = execution._voice_tool_run_state.set(None)
    try:
        created = execution.current_voice_tool_run_state()
        assert created.last_entity_by_domain == {}
        execution.voice_tool_run_state.last_area_id = "office"
        assert execution.voice_tool_run_state.last_area_id == "office"

        execution._reset_voice_tool_run_state(
            "  switch it on  ",
            {
                "device_id": "dev-1",
                "device_name": "Voice",
                "area_id": "kitchen",
                "area_name": "Kitchen",
                "floor_name": "Ground",
                "source": "satellite",
            },
        )
        reset = execution.current_voice_tool_run_state()
        assert reset.request_text == "switch it on"
        assert (
            reset.origin_device_id,
            reset.origin_device_name,
            reset.origin_area_id,
            reset.origin_area_name,
            reset.origin_floor_name,
            reset.origin_source,
        ) == ("dev-1", "Voice", "kitchen", "Kitchen", "Ground", "satellite")

        execution._reset_voice_tool_run_state("empty origin", None)
        assert execution.current_voice_tool_run_state().origin_device_id is None
    finally:
        execution._voice_tool_run_state.reset(token)


def test_runtime_compatibility_text_and_url_wrappers(monkeypatch):
    monkeypatch.setattr(settings.deterministic, "error_phrases", ())
    assert not execution.is_home_intent_error_response("anything")
    monkeypatch.setattr(settings.deterministic, "error_phrases", ("not found", "failed"))
    assert execution.is_home_intent_error_response("  Device NOT FOUND  ")
    assert not execution.is_home_intent_error_response("all good")

    assert execution.normalize_command("  Hello, WORLD! ") == "hello, world"
    assert execution._normalise_search_text("  Hello_World  ") == "hello world"
    assert execution._ha_websocket_url("https://ha.example/") == "wss://ha.example/api/websocket"
    assert execution._json_tool_result({"value": object()}).startswith('{"value": "<object object')


def test_compact_attributes_covers_supported_shapes_and_bounds():
    attributes = {
        "attribution": "discard",
        "short": "ok",
        "long": "x" * 510,
        "number": 12,
        "flag": True,
        "missing": None,
        "short_list": list(range(20)),
        "long_list": list(range(21)),
        "short_dict": {str(index): index for index in range(20)},
        "long_dict": {str(index): index for index in range(21)},
        "unsupported": {1, 2},
    }

    compact = execution._compact_attributes(attributes)

    assert "attribution" not in compact
    assert compact["short"] == "ok"
    assert compact["long"] == "x" * 497 + "..."
    assert compact["number"] == 12
    assert compact["flag"] is True
    assert compact["missing"] is None
    assert len(compact["short_list"]) == 20
    assert len(compact["short_dict"]) == 20
    assert "long_list" not in compact
    assert "long_dict" not in compact
    assert "unsupported" not in compact


@pytest.mark.parametrize(
    ("value", "limit", "expected"),
    [
        (None, 10, None),
        ("  \n\t ", 10, None),
        ("  two   words ", 20, "two words"),
        ("abcdefghij", 8, "abcde..."),
    ],
)
def test_truncate_text_branches(value, limit, expected):
    assert execution._truncate_text(value, limit) == expected


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ({}, []),
        ({"selector": "select"}, []),
        ({"selector": {"select": "bad"}}, []),
        ({"selector": {"select": {"options": "bad"}}}, []),
        (
            {
                "selector": {
                    "select": {
                        "options": [
                            {"value": "heat", "label": "Heat"},
                            {"label": "Cool"},
                            {"ignored": True},
                            "auto",
                            3,
                            None,
                        ]
                    }
                }
            },
            ["heat", "Cool", "auto", 3],
        ),
    ],
)
def test_selector_allowed_values_handles_schema_variants(field, expected):
    assert execution._selector_allowed_values(field) == expected


def test_selector_allowed_values_is_bounded():
    field = {"selector": {"select": {"options": list(range(75))}}}
    assert execution._selector_allowed_values(field) == list(range(50))


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        (None, []),
        ({"entity": "light"}, []),
        ({"entity": {"domain": "light"}}, ["light"]),
        ({"entity": {"domain": ["light", 2, "switch"]}}, ["light", "switch"]),
        ({"entity": {"domain": 12}}, []),
    ],
)
def test_target_entity_domains_handles_schema_variants(target, expected):
    assert execution._target_entity_domains(target) == expected


def test_compact_service_definition_handles_invalid_and_complete_fields():
    assert (
        execution._compact_service_definition("light", "turn_on", {"fields": []})["parameters"]
        == {}
    )

    result = execution._compact_service_definition(
        "climate",
        "set_mode",
        {
            "name": "Set mode",
            "description": "  Changes   the mode  ",
            "target": {"entity": {"domain": ["climate"]}},
            "response": {"optional": True},
            "fields": {
                "mode": {
                    "required": True,
                    "default": "auto",
                    "example": "heat",
                    "description": " Mode to use ",
                    "selector": {"select": {"options": [{"value": "heat"}, {"label": "Cool"}]}},
                },
                "bad_field": "not a definition",
            },
        },
    )

    assert result["description"] == "Changes the mode"
    assert result["target"] == {
        "required": True,
        "entity_domains": ["climate"],
        "accepts_entity_id": True,
    }
    assert result["parameters"]["mode"] == {
        "required": True,
        "allowed": ["heat", "Cool"],
        "default": "auto",
        "example": "heat",
        "description": "Mode to use",
    }
    assert result["parameters"]["bad_field"] == {"required": False}
    assert result["required_parameters"] == ["mode"]
    assert result["returns_data"] is True


def test_cached_service_lookup_and_single_entity_selection():
    client = SimpleNamespace(
        services={
            "light": {"turn_on": {"name": "On"}, "bad": []},
            "switch": [],
        },
        states={"LIGHT.one": {}, "switch.one": {}},
    )
    assert execution._get_cached_service_definition(client, "missing", "turn_on") is None
    assert execution._get_cached_service_definition(client, "switch", "turn_on") is None
    assert execution._get_cached_service_definition(client, "light", "bad") is None
    assert execution._get_cached_service_definition(client, "light", "turn_on") == {"name": "On"}
    assert execution._single_cached_entity_for_domain(client, "light") == "LIGHT.one"
    assert execution._single_cached_entity_for_domain(client, "cover") is None

    client.states["light.two"] = {}
    assert execution._single_cached_entity_for_domain(client, "LIGHT") is None


def test_service_data_normalisation_wrapper_forwards_policy(monkeypatch):
    calls = []

    def normalize(*args, **kwargs):
        calls.append((args, kwargs))
        return {"mode": "heat"}, [], None

    monkeypatch.setattr(execution.ha_domain, "normalise_service_data", normalize)
    monkeypatch.setattr(settings.home_assistant, "schema_auto_repair_enabled", False)
    value = execution._normalise_service_data_from_schema(
        "climate",
        "set_mode",
        {"fields": {}},
        {"mode": "heat"},
        {"mode": "cool"},
    )

    assert value == ({"mode": "heat"}, [], None)
    assert calls[0][0][:2] == ("climate", "set_mode")
    assert calls[0][1] == {"auto_repair": False}


@pytest.mark.parametrize(
    ("tool_name", "output", "expected"),
    [
        ("ma_play_query", {"message": "   "}, "Playing."),
        ("ma_play_query", "added to queue", "Added to the queue."),
        ("ma_play_media", "starting playback now", "Starting playback."),
        ("ma_play_media", "done", "Playing."),
        ("ma_volume", "unmuted", "Unmuted."),
        ("ma_volume", "increased", "Volume up."),
        ("ma_volume", "decreased", "Volume down."),
        ("ma_volume", "done", "Volume set."),
        ("ma_playback", "stopped", "Stopped."),
        ("ma_playback", "skipped to next", "Next."),
        ("ma_playback", "previous item", "Previous."),
        ("ma_playback", "playing", "Playing."),
        ("ma_playback", "done", settings.api.action_confirmation),
        ("ma_group", "grouped", "Speakers grouped."),
        ("ma_queue_item", "done", settings.api.action_confirmation),
        ("ma_custom", "done", settings.api.action_confirmation),
    ],
)
def test_remaining_music_terminal_speech_branches(tool_name, output, expected):
    assert results._music_assistant_terminal_speech(tool_name, output) == expected


@pytest.mark.parametrize(
    "output",
    [
        {"success": False},
        "error: broken",
        "failed to connect",
        "connector is closed",
        "queue not found: q1",
        "unknown command: x",
        "unknown action: x",
        "no changes made",
        "invalid player",
    ],
)
def test_music_assistant_failure_variants(output):
    assert results._music_assistant_output_failed(output)


@pytest.mark.parametrize(
    "output",
    [
        {"changed": False},
        "ordinary queue status",
    ],
)
def test_music_queue_reads_are_not_writes(output):
    assert not results._is_music_assistant_write_result("ma_queue", output)


def test_tool_result_name_tolerates_missing_tool_and_name():
    assert results._tool_result_name(_result(None, "ok")) == ""
    assert results._tool_result_name(SimpleNamespace(tool=SimpleNamespace(name=None))) == ""


def test_replay_cache_disabled_missing_expired_and_area_name_paths(monkeypatch):
    cache = {}
    monkeypatch.setattr(results, "recent_music_action_responses", cache)
    monkeypatch.setattr(settings.music_assistant, "replay_guard_seconds", 0)
    results._remember_music_action_response("conversation", "Play Jazz", None, "Playing")
    assert cache == {}
    assert results._get_recent_music_action_response("conversation", "Play Jazz", None) is None

    monkeypatch.setattr(settings.music_assistant, "replay_guard_seconds", 10)
    monkeypatch.setattr(results.time, "monotonic", lambda: 100.0)
    cache["expired"] = (50.0, "old")
    assert (
        results._get_recent_music_action_response(
            "conversation", "Play Jazz", {"area_name": " Lounge "}
        )
        is None
    )
    assert cache == {}
    assert (
        results._music_replay_cache_key(" key ", " Play Jazz ", {"area_name": " Lounge "})
        == "key|lounge|play jazz"
    )


def test_replay_cache_defensive_expiry_after_lookup(monkeypatch):
    key = results._music_replay_cache_key("conversation", "play jazz", None)

    class _ChangingCache(dict):
        def items(self):
            return [(key, (95.0, "fresh during cleanup"))]

        def get(self, requested, default=None):
            assert requested == key
            return (80.0, "expired during lookup")

    cache = _ChangingCache()
    monkeypatch.setattr(results, "recent_music_action_responses", cache)
    monkeypatch.setattr(settings.music_assistant, "replay_guard_seconds", 10)
    monkeypatch.setattr(results.time, "monotonic", lambda: 100.0)

    assert results._get_recent_music_action_response("conversation", "play jazz", None) is None


def test_fast_handler_with_terminal_music_disabled_returns_to_model(monkeypatch):
    monkeypatch.setattr(settings.music_assistant, "terminal_actions_enabled", False)
    outcome = results.fast_tool_result_handler(
        None,
        [_result("ma_volume", {"success": True, "message": "Muted."})],
    )
    assert not outcome.is_final_output
    assert outcome.final_output is None


def test_agent_spoken_response_wrapper_uses_configured_policy():
    assert results.sanitise_spoken_response(" **Done.** ") == "Done."
