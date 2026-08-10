from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from intent_bridge.config import settings
from intent_bridge.indicators import controller as indicators
from intent_bridge.indicators import topology as indicator_topology


@pytest.fixture
def client():
    states = {
        "assist_satellite.office": {
            "state": "idle",
            "attributes": {"friendly_name": "Office Assist"},
        },
        "light.ring": {
            "state": "off",
            "attributes": {
                "friendly_name": "Status Ring LED",
                "brightness": 50,
                "effect": None,
                "effect_list": ["None", "Pulse"],
                "supported_color_modes": ["rgb"],
                "color_mode": "rgb",
                "rgb_color": [1, 2, 3],
            },
        },
        "switch.unrelated": {"state": "off", "attributes": {"friendly_name": "Remote ADB"}},
    }
    registry = {
        "assist_satellite.office": {"di": "sat", "ai": "office", "en": "Office assist"},
        "light.ring": {"di": "ring", "ai": "office", "en": "Ring LED"},
        "switch.unrelated": {"di": "other"},
    }
    devices = {
        "hub": {"name": "Office Voice"},
        "sat": {"name": "Assistant", "via_device_id": "hub", "area_id": "office"},
        "ring": {"name": "Office Dot ring", "via_device_id": "hub", "area_id": "office"},
        "other": {"name": "Other"},
    }

    def entity_context(entity_id, state):
        entry = registry.get(entity_id, {})
        device = devices.get(entry.get("di"), {})
        return {
            "area_id": entry.get("ai") or device.get("area_id"),
            "area_name": "Office"
            if (entry.get("ai") or device.get("area_id")) == "office"
            else None,
        }

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


def test_identity_topology_score_and_relations(client):
    assert "status ring led" in indicator_topology.indicator_identity_text(client, "light.ring")
    assert indicator_topology.device_via_device_id(client, "sat") == "hub"
    client.devices["legacy"] = {"via_device": "hub"}
    assert indicator_topology.device_via_device_id(client, "legacy") == "hub"
    assert indicator_topology.device_via_device_id(client, "missing") is None
    scope = indicator_topology.connected_satellite_device_scope(client, "sat")
    assert scope["sat"] == (0, "same_device")
    assert scope["hub"] == (1, "parent_device")
    assert scope["ring"] == (2, "sibling_device")
    assert indicator_topology.indicator_relation_bonus(0, "same_device") == 30
    assert indicator_topology.indicator_relation_bonus(5, "unknown") == 0
    score, reasons = indicator_topology.indicator_score(client, "light.ring")
    assert score > 200 and "led" in reasons and "light_domain" in reasons
    switch_score, switch_reasons = indicator_topology.indicator_score(client, "switch.unrelated")
    assert switch_score >= 4 and "switch_domain" in switch_reasons


@pytest.mark.asyncio
async def test_internal_service_call_success_and_failure(client, monkeypatch):
    monkeypatch.setattr(indicators, "_require_ha_ws", AsyncMock(return_value=client))
    await indicators._ha_internal_service_call("light", "turn_on", "light.ring", {"brightness": 1})
    assert client.command.await_args.args[0]["target"] == {"entity_id": "light.ring"}
    client.command.return_value = {"success": False, "error": "bad"}
    with pytest.raises(RuntimeError, match="failed"):
        await indicators._ha_internal_service_call("light", "turn_on", "light.ring")


@pytest.mark.asyncio
async def test_resolve_indicator_by_device_and_area(client, monkeypatch):
    monkeypatch.setattr(indicators, "_require_ha_ws", AsyncMock(return_value=client))
    target = await indicators._resolve_satellite_indicator(
        {"device_id": "sat", "area_id": "office", "area_name": "Office"}
    )
    assert target.entity_id == "light.ring"
    assert target.satellite_entity_id == "assist_satellite.office"
    assert "sibling_device" in target.match_reason

    target = await indicators._resolve_satellite_indicator({"area_name": "Office"})
    assert target.entity_id == "light.ring"
    assert target.match_reason.startswith("sole_assist_satellite_in_origin_area")
    monkeypatch.setattr(settings.indicators, "music_playback_enabled", False)
    assert await indicators._resolve_satellite_indicator({"area_name": "Office"}) is None
    assert await indicators._resolve_satellite_indicator(None) is None


@pytest.mark.asyncio
async def test_resolve_indicator_ambiguity_missing_device_and_tie(client, monkeypatch):
    monkeypatch.setattr(indicators, "_require_ha_ws", AsyncMock(return_value=client))
    client.states["assist_satellite.second"] = {"state": "idle", "attributes": {}}
    client.entity_registry["assist_satellite.second"] = {"di": "second", "ai": "office"}
    client.devices["second"] = {"name": "Second", "area_id": "office"}
    assert await indicators._resolve_satellite_indicator({"area_name": "Office"}) is None

    client.states.pop("assist_satellite.second")
    client.entity_registry["assist_satellite.office"].pop("di")
    assert await indicators._resolve_satellite_indicator({"area_name": "Office"}) is None

    client.entity_registry["assist_satellite.office"]["di"] = "sat"
    client.states["light.ring_two"] = dict(client.states["light.ring"])
    client.entity_registry["light.ring_two"] = {"di": "ring", "ai": "office", "en": "Ring LED"}
    assert await indicators._resolve_satellite_indicator({"device_id": "sat"}) is None


@pytest.mark.asyncio
async def test_indicator_manager_lifecycle_light(client, monkeypatch):
    target = indicators.SatelliteIndicatorTarget(
        entity_id="light.ring",
        domain="light",
        satellite_entity_id="assist_satellite.office",
        device_id="ring",
        device_name="Ring",
        area_id="office",
        area_name="Office",
        match_reason="test",
    )
    monkeypatch.setattr(indicators, "_resolve_satellite_indicator", AsyncMock(return_value=target))
    monkeypatch.setattr(indicators, "_require_ha_ws", AsyncMock(return_value=client))
    service = AsyncMock()
    monkeypatch.setattr(indicators, "_ha_internal_service_call", service)
    monkeypatch.setattr(settings.indicators, "color", "green")
    monkeypatch.setattr(settings.indicators, "effect", "pulse")
    monkeypatch.setattr(settings.indicators, "software_pulse_enabled", False)
    manager = indicators.VoiceSatelliteActivityIndicators()
    first = await manager.begin({"device_id": "sat"})
    second = await manager.begin({"device_id": "sat"})
    assert first == second == indicators.SatelliteIndicatorHandle("light.ring")
    assert manager.active_count == 1
    assert manager.last_target["native_effect"] == "Pulse"
    await manager.end(first)
    assert manager.active_count == 1
    await manager.end(second)
    assert manager.active_count == 0
    assert service.await_count >= 2
    await manager.end(None)
    await manager.end(first)


@pytest.mark.asyncio
async def test_indicator_manager_switch_restore_stop_and_failure(client, monkeypatch):
    target = indicators.SatelliteIndicatorTarget(
        "switch.unrelated",
        "switch",
        "assist_satellite.office",
        "sat",
        "Sat",
        "office",
        "Office",
        "test",
    )
    monkeypatch.setattr(indicators, "_resolve_satellite_indicator", AsyncMock(return_value=target))
    monkeypatch.setattr(indicators, "_require_ha_ws", AsyncMock(return_value=client))
    monkeypatch.setattr(settings.indicators, "software_pulse_enabled", False)
    service = AsyncMock()
    monkeypatch.setattr(indicators, "_ha_internal_service_call", service)
    manager = indicators.VoiceSatelliteActivityIndicators()
    handle = await manager.begin({"device_id": "sat"})
    assert handle.entity_id == "switch.unrelated"
    await manager.stop_all()
    assert manager.active_count == 0

    monkeypatch.setattr(
        indicators, "_resolve_satellite_indicator", AsyncMock(side_effect=RuntimeError("bad"))
    )
    assert await manager.begin({"device_id": "sat"}) is None
    assert "bad" in manager.last_error


def test_snapshot_restore_variants():
    for mode, key, value in [
        ("rgbw", "rgbw_color", [1, 2, 3, 4]),
        ("rgbww", "rgbww_color", [1, 2, 3, 4, 5]),
        ("hs", "hs_color", [10, 20]),
        ("xy", "xy_color", [0.1, 0.2]),
        ("color_temp", "color_temp_kelvin", 3000),
    ]:
        snapshot = indicators.SatelliteIndicatorSnapshot(
            "light.x", "light", "on", {"color_mode": mode, key: value}
        )
        assert indicators._snapshot_restore_light_data(snapshot)[key] == value


def _session(domain="light", state="off", attributes=None, native_effect=None):
    target = indicators.SatelliteIndicatorTarget(
        f"{domain}.indicator",
        domain,
        "assist_satellite.office",
        "sat",
        "Sat",
        "office",
        "Office",
        "test",
    )
    snapshot = indicators.SatelliteIndicatorSnapshot(
        target.entity_id, domain, state, attributes or {}
    )
    return indicators.SatelliteIndicatorSession(target, snapshot, native_effect=native_effect)


@pytest.mark.asyncio
async def test_indicator_begin_missing_state_and_activation_failure(client, monkeypatch):
    missing = _session().target
    monkeypatch.setattr(indicators, "_resolve_satellite_indicator", AsyncMock(return_value=missing))
    monkeypatch.setattr(indicators, "_require_ha_ws", AsyncMock(return_value=client))
    manager = indicators.VoiceSatelliteActivityIndicators()
    assert await manager.begin({}) is None

    client.states[missing.entity_id] = {"state": "off", "attributes": None}
    monkeypatch.setattr(manager, "_activate", AsyncMock(side_effect=RuntimeError("activate")))
    assert await manager.begin({}) is None
    assert manager.active_count == 0 and "activate" in manager.last_error


@pytest.mark.asyncio
async def test_indicator_pulse_loops_and_restore_paths(monkeypatch):
    monkeypatch.setattr(settings.indicators, "pulse_interval_seconds", 0)
    calls = AsyncMock(side_effect=[None, RuntimeError("pulse stopped")])
    monkeypatch.setattr(indicators, "_ha_internal_service_call", calls)
    manager = indicators.VoiceSatelliteActivityIndicators()
    light = _session("light", attributes={"supported_color_modes": ["rgb"]})
    await manager._software_pulse_light(light)
    assert calls.await_count == 2

    calls.reset_mock()
    calls.side_effect = [None, RuntimeError("pulse stopped")]
    switch = _session("switch")
    await manager._software_pulse_switch(switch)
    assert calls.await_count == 2

    service = AsyncMock()
    monkeypatch.setattr(indicators, "_ha_internal_service_call", service)
    lit = _session(
        "light", "on", {"brightness": 20, "effect": "Sparkle", "effect_list": ["None", "Pulse"]}
    )
    await manager._restore(lit)
    assert service.await_args.args[1] == "turn_on"

    service.reset_mock()
    dark = _session(
        "light",
        "off",
        {"brightness": 20, "effect": None, "effect_list": ["None", "Pulse"]},
        "Pulse",
    )
    await manager._restore(dark)
    assert [call.args[1] for call in service.await_args_list] == ["turn_on", "turn_off"]

    service.reset_mock()
    await manager._restore(_session("switch", "on"))
    assert service.await_args.args[1] == "turn_on"
    service.side_effect = RuntimeError("restore")
    await manager._restore(_session("switch", "off"))
    assert "restore" in manager.last_error
