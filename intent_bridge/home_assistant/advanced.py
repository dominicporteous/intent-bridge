"""Optional advanced Home Assistant MCP specialist."""

import asyncio
import os

from agents import (
    Agent,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    Runner,
    function_tool,
)
from agents.mcp import MCPServerStdio

from intent_bridge.config import log, settings
from intent_bridge.core.tool_output import (
    serialise_tool_output,
    tool_output_failed,
    tool_output_mapping,
)
from intent_bridge.home_assistant.tools import (
    ADVANCED_INSTRUCTIONS,
)
from intent_bridge.music_assistant.policy import (
    music_assistant_agent_instructions,
    parse_music_area_player_map,
)
from intent_bridge.runtime.context import origin_runtime_context, runtime_context
from intent_bridge.runtime.dependencies import runtime
from intent_bridge.runtime.execution import (
    _json_tool_result,
    voice_tool_run_state,
)

# Private compatibility aliases for callers migrating from the former monolith.
_serialise_tool_output = serialise_tool_output
_tool_output_failed = tool_output_failed
_tool_output_mapping = tool_output_mapping


def _make_lemonade_model() -> OpenAIChatCompletionsModel:
    client = AsyncOpenAI(
        base_url=settings.llm.base_url,
        api_key=settings.llm.api_key,
        timeout=settings.llm.timeout_seconds,
    )
    return OpenAIChatCompletionsModel(
        model=settings.llm.model,
        openai_client=client,
    )


def make_ha_mcp_server() -> MCPServerStdio:
    child_env = dict(os.environ)
    child_env.update(
        {
            # ha-mcp owns these child-process variable names.
            "HOMEASSISTANT_URL": settings.home_assistant.base_url,
            "HOMEASSISTANT_TOKEN": settings.home_assistant.access_token,
            "ENABLE_TOOL_SEARCH": "true"
            if settings.home_assistant.advanced.tool_search_enabled
            else "false",
            "TOOL_SEARCH_MAX_RESULTS": str(
                settings.home_assistant.advanced.tool_search_max_results
            ),
            "PINNED_TOOLS": ",".join(settings.home_assistant.advanced.pinned_tools),
        }
    )
    return MCPServerStdio(
        name="Home Assistant Advanced",
        params={
            "command": settings.home_assistant.advanced.command,
            "args": list(settings.home_assistant.advanced.args),
            "env": child_env,
        },
        cache_tools_list=True,
        client_session_timeout_seconds=settings.mcp.client_session_timeout_seconds,
    )


def _parse_music_area_player_map() -> dict[str, str]:
    """Compatibility wrapper; mapping policy lives in ``music_policy``."""
    return parse_music_area_player_map(settings.music_assistant.area_player_map)


def _music_assistant_agent_instructions() -> str:
    """Compatibility wrapper; prompt policy lives in ``music_policy``."""
    return music_assistant_agent_instructions(_parse_music_area_player_map())


def make_advanced_agent(active_servers) -> Agent:
    return Agent(
        name="Home Assistant advanced specialist",
        model=_make_lemonade_model(),
        instructions=ADVANCED_INSTRUCTIONS,
        mcp_servers=list(active_servers),
    )


async def run_advanced_agent(request: str) -> str:
    """Run an advanced Home Assistant request through the agent harness."""
    if runtime.advanced_agent is None:
        raise RuntimeError("Advanced Home Assistant agent is unavailable.")

    result = await asyncio.wait_for(
        Runner.run(
            runtime.advanced_agent,
            f"{runtime_context(settings.api.timezone)}\n{
                origin_runtime_context(
                    {
                        'device_name': voice_tool_run_state.origin_device_name,
                        'area_name': voice_tool_run_state.origin_area_name,
                        'area_id': voice_tool_run_state.origin_area_id,
                    }
                )
            }\n\nAdvanced request: {request.strip()}",
            max_turns=settings.home_assistant.advanced.max_turns,
        ),
        timeout=settings.llm.timeout_seconds,
    )
    return str(result.final_output or "").strip()


@function_tool
async def ha_advanced(request: str) -> str:
    """Use full ha-mcp for advanced Home Assistant work.

    Use only for requests that truly need configuration, automation editing,
    history/traces, dashboards, helpers, integrations, system diagnostics or
    another capability unavailable from the fast state/service tools.

    Args:
        request: Concise self-contained description of the advanced operation.
    """
    if runtime.advanced_agent is None:
        return _json_tool_result(
            {
                "success": False,
                "error": "Advanced Home Assistant tools are unavailable.",
            }
        )

    try:
        output = await run_advanced_agent(request)
        return _json_tool_result(
            {
                "success": bool(output),
                "result": output or "No advanced result was returned.",
            }
        )
    except Exception as exc:
        log.exception("Advanced ha-mcp tool failed")
        return _json_tool_result({"success": False, "error": str(exc)})
