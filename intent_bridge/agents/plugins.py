"""Built-in tool plugins used by the LLM voice route."""

from intent_bridge.agents.contracts import AgentToolPlugin
from intent_bridge.home_assistant.advanced import ha_advanced
from intent_bridge.home_assistant.tools import (
    ha_call_service,
    ha_get_state,
    ha_list_services,
    ha_search,
)
from intent_bridge.music_assistant.tools import (
    ma_browse,
    ma_group,
    ma_list_players,
    ma_play_media,
    ma_play_query,
    ma_playback,
    ma_queue,
    ma_queue_item,
    ma_search,
    ma_transfer_queue,
    ma_volume,
)

HOME_ASSISTANT_PLUGIN = AgentToolPlugin(
    name="home-assistant",
    tools=(ha_search, ha_get_state, ha_list_services, ha_call_service),
)

HOME_ASSISTANT_ADVANCED_PLUGIN = AgentToolPlugin(
    name="home-assistant-advanced",
    tools=(ha_advanced,),
)


def music_assistant_plugin(instructions: str) -> AgentToolPlugin:
    return AgentToolPlugin(
        name="music-assistant",
        tools=(
            ma_play_query,
            ma_list_players,
            ma_search,
            ma_browse,
            ma_play_media,
            ma_playback,
            ma_volume,
            ma_group,
            ma_queue,
            ma_queue_item,
            ma_transfer_queue,
        ),
        instructions=instructions,
    )


__all__ = [
    "AgentToolPlugin",
    "HOME_ASSISTANT_ADVANCED_PLUGIN",
    "HOME_ASSISTANT_PLUGIN",
    "music_assistant_plugin",
]
