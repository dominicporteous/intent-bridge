import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from agents.tool_context import ToolContext

from intent_bridge.config import settings
from intent_bridge.home_assistant import tools as ha_adapter
from intent_bridge.runtime.execution import voice_tool_run_state


async def invoke(tool, **arguments):
    encoded = json.dumps(arguments)
    context = ToolContext(
        None,
        tool_name=tool.name,
        tool_call_id="test-call",
        tool_arguments=encoded,
    )
    return json.loads(await tool.on_invoke_tool(context, encoded))


@pytest.fixture(autouse=True)
def reset_state():
    voice_tool_run_state.origin_area_id = None
    voice_tool_run_state.origin_area_name = None
    voice_tool_run_state.last_area_id = None
    voice_tool_run_state.last_entity_by_domain.clear()
    voice_tool_run_state.last_service_call = None
    voice_tool_run_state.last_successful_data = None


@pytest.fixture
def client(monkeypatch):
    value = SimpleNamespace(
        states={
            "light.office": {
                "state": "off",
                "attributes": {"friendly_name": "Office", "brightness": 10, "large": "x" * 600},
                "last_changed": "then",
                "last_updated": "now",
            },
            "weather.home": {"state": "sunny", "attributes": {}},
        },
        services={
            "light": {
                "turn_on": {"name": "Turn on", "description": "Power on", "fields": {}},
            },
            "weather": {
                "get_forecasts": {
                    "name": "Forecast",
                    "description": "Get forecast data",
                    "target": {"entity": {"domain": "weather"}},
                    "fields": {
                        "type": {
                            "required": True,
                            "selector": {"select": {"options": ["daily", "hourly"]}},
                        }
                    },
                    "response": {"optional": True},
                }
            },
        },
        refresh_registries=AsyncMock(),
        refresh_services=AsyncMock(),
        area_mentioned_in_text=lambda query: ("office", "Office") if "office" in query else None,
        search_cached_states=lambda *args, **kwargs: [
            {
                "entity_id": "light.office",
                "domain": "light",
                "area_id": "office",
                "match_score": 100,
            }
        ],
        entities_in_area=lambda domain, area: [f"{domain}.office"],
        command=AsyncMock(return_value={"success": True, "result": {}}),
        wait_for_expected_state=AsyncMock(return_value="on"),
    )
    monkeypatch.setattr(ha_adapter, "_require_ha_ws", AsyncMock(return_value=value))
    return value


@pytest.mark.asyncio
async def test_search_explicit_query_and_origin_modes(client, monkeypatch):
    result = await invoke(
        ha_adapter.ha_search,
        query="office light",
        domain_filter="light",
        area_filter=None,
        limit=999,
    )
    assert result["recommended_entity_id"] == "light.office"
    assert result["area_context_source"] == "query"
    assert voice_tool_run_state.last_entity_by_domain["light"] == "light.office"

    client.area_mentioned_in_text = lambda query: None
    voice_tool_run_state.origin_area_name = "Office"
    monkeypatch.setattr(settings.voice_origin, "soft_ranking_enabled", True)
    result = await invoke(
        ha_adapter.ha_search, query="light", domain_filter="light", area_filter=None, limit=0
    )
    assert result["preferred_area"] == "Office"
    assert result["area_context_source"] == "voice_origin_soft"

    monkeypatch.setattr(settings.voice_origin, "soft_ranking_enabled", False)
    calls = []
    client.search_cached_states = lambda *args, **kwargs: (
        calls.append(kwargs)
        or ([] if len(calls) == 1 else [{"entity_id": "light.x", "domain": "light"}])
    )
    result = await invoke(
        ha_adapter.ha_search, query="light", domain_filter="light", area_filter=None, limit=3
    )
    assert result["area_context_source"] == "voice_origin_global_fallback"
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_search_and_get_state_failures(client, monkeypatch):
    monkeypatch.setattr(
        ha_adapter, "_require_ha_ws", AsyncMock(side_effect=RuntimeError("offline"))
    )
    assert not (
        await invoke(ha_adapter.ha_search, query="x", domain_filter=None, area_filter=None, limit=1)
    )["success"]
    assert not (await invoke(ha_adapter.ha_get_state, entity_id="x", attribute_keys=None))[
        "success"
    ]
    monkeypatch.setattr(ha_adapter, "_require_ha_ws", AsyncMock(return_value=client))
    missing = await invoke(ha_adapter.ha_get_state, entity_id="light.missing", attribute_keys=None)
    assert "not found" in missing["error"]


@pytest.mark.asyncio
async def test_get_state_compact_and_selected_attributes(client):
    compact = await invoke(ha_adapter.ha_get_state, entity_id="light.office", attribute_keys=None)
    assert compact["state"] == "off"
    assert compact["attributes"]["large"].endswith("...")
    selected = await invoke(
        ha_adapter.ha_get_state, entity_id="light.office", attribute_keys=["brightness", "missing"]
    )
    assert selected["attributes"] == {"brightness": 10}


@pytest.mark.asyncio
async def test_list_services_summary_full_filters_and_failure(client, monkeypatch):
    summary = await invoke(
        ha_adapter.ha_list_services, domain="light", query="power", detail_level="summary", limit=50
    )
    assert summary["count"] == 1
    assert summary["services"][0]["service"] == "turn_on"
    full = await invoke(
        ha_adapter.ha_list_services, domain="weather", query=None, detail_level="full", limit=1
    )
    assert full["services"][0]["required_parameters"] == ["type"]
    monkeypatch.setattr(
        ha_adapter, "_require_ha_ws", AsyncMock(side_effect=RuntimeError("offline"))
    )
    assert not (
        await invoke(
            ha_adapter.ha_list_services, domain=None, query=None, detail_level="summary", limit=5
        )
    )["success"]


@pytest.mark.asyncio
async def test_call_service_validation_repair_response_and_verification(client):
    missing = await invoke(
        ha_adapter.ha_call_service,
        domain="",
        service="",
        entity_id=None,
        area_id=None,
        data=None,
        return_response=False,
    )
    assert missing["error_type"] == "local_schema_validation"

    invalid = await invoke(
        ha_adapter.ha_call_service,
        domain="weather",
        service="get_forecasts",
        entity_id=None,
        area_id=None,
        data={"type": "weekly"},
        return_response=True,
    )
    assert invalid["invalid_values"]["type"]["allowed"] == ["daily", "hourly"]

    voice_tool_run_state.last_service_call = None
    client.command.return_value = {
        "success": True,
        "result": {"response": {"weather.home": {"forecast": [{"temperature": 20}]}}},
    }
    result = await invoke(
        ha_adapter.ha_call_service,
        domain="weather",
        service="get_forecasts",
        entity_id=None,
        area_id=None,
        data={"forecast_type": "daily"},
        return_response=True,
    )
    assert result["success"]
    assert result["entity_id"] == "weather.home"
    assert result["repairs_applied"]
    assert "service_response" in result

    result = await invoke(
        ha_adapter.ha_call_service,
        domain="light",
        service="turn_on",
        entity_id="light.office",
        area_id=None,
        data={},
        return_response=False,
    )
    assert result["verified_state"] == "on"
    assert client.command.await_args.args[0]["target"] == {"entity_id": "light.office"}


@pytest.mark.asyncio
async def test_call_service_target_reuse_multiple_targets_and_server_error(client):
    voice_tool_run_state.last_service_call = {
        "domain": "light",
        "service": "turn_on",
        "entity_id": "light.previous",
        "area_id": None,
        "data": {},
        "return_response": True,
    }
    result = await invoke(
        ha_adapter.ha_call_service,
        domain="light",
        service="turn_on",
        entity_id=None,
        area_id=None,
        data={},
        return_response=False,
    )
    assert result["entity_id"] == "light.previous"
    assert "preserved return_response" in result["repairs_applied"][0]

    client.command.return_value = {
        "success": False,
        "error": {"code": "bad", "message": "Rejected"},
    }
    failed = await invoke(
        ha_adapter.ha_call_service,
        domain="light",
        service="turn_on",
        entity_id="light.one, light.two",
        area_id="office",
        data={},
        return_response=False,
    )
    assert failed["error_code"] == "bad"
    assert failed["entity_id"] == "light.one,light.two"


@pytest.mark.asyncio
async def test_call_service_repairs_vacuum_turn_on_to_start(client):
    client.services["vacuum"] = {
        "start": {"name": "Start", "description": "Start cleaning", "fields": {}}
    }
    client.states["vacuum.robot"] = {"state": "docked", "attributes": {}}

    result = await invoke(
        ha_adapter.ha_call_service,
        domain="vacuum",
        service="turn_on",
        entity_id="vacuum.robot",
        area_id=None,
        data=None,
        return_response=False,
    )

    assert result["success"]
    assert result["service"] == "start"
    assert client.command.await_args.args[0]["service"] == "start"


@pytest.mark.asyncio
async def test_call_service_exception(client, monkeypatch):
    monkeypatch.setattr(
        ha_adapter, "_require_ha_ws", AsyncMock(side_effect=RuntimeError("offline"))
    )
    result = await invoke(
        ha_adapter.ha_call_service,
        domain="light",
        service="turn_on",
        entity_id=None,
        area_id=None,
        data=None,
        return_response=False,
    )
    assert result["success"] is False
    assert result["domain"] == "light"
    assert result["service"] == "turn_on"
    assert result["error"] == "offline"
