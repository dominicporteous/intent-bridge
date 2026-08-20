"""Extracted application layer; see README architecture map."""

import time
import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from agents.mcp import MCPServerManager
from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse

try:
    from music_assistant_client import MusicAssistantClient

    MUSIC_ASSISTANT_CLIENT_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - deployment dependency guard
    MusicAssistantClient = None
    MUSIC_ASSISTANT_CLIENT_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

from intent_bridge.agents.factory import (
    make_fallback_agent,
    make_informational_agent,
)
from intent_bridge.api.conversation import (
    _message_text,
    clear_conversation_history,
    extract_client_history,
    extract_voice_origin_context,
    get_conversation_history,
    get_conversation_key,
    remember_conversation_turn,
    seed_conversation_history,
)
from intent_bridge.assistant import assistant_feedback
from intent_bridge.config import log, settings
from intent_bridge.core.voice import (
    FunctionVoiceRoute,
    VoiceActionPipeline,
    VoicePipelineError,
    VoiceRequest,
    VoiceRequestHandler,
)
from intent_bridge.home_assistant.advanced import (
    make_advanced_agent,
    make_ha_mcp_server,
)
from intent_bridge.home_assistant.client import HomeAssistantWebSocket
from intent_bridge.home_assistant.intent_catalog import (
    HomeAssistantCatalogProvider,
    HomeAssistantCatalogPublisher,
)
from intent_bridge.home_assistant.intent_executor import HomeAssistantIntentExecutor
from intent_bridge.informational import InformationalVoiceRoute
from intent_bridge.intent_engine.engine import DeterministicIntentEngine
from intent_bridge.intent_engine.grammar import load_intent_grammar
from intent_bridge.intent_engine.measurement import MeasurementIntentPlanner
from intent_bridge.intent_engine.models import PlannedIntent
from intent_bridge.intent_engine.natural_language import NaturalLanguageIntentPlanner
from intent_bridge.intent_engine.planning import IntentPlannerChain
from intent_bridge.intent_engine.ports import IntentExecutor, IntentRecognizer
from intent_bridge.intent_engine.recognizer import HassilIntentRecognizer
from intent_bridge.intent_engine.route import ConversationalDeterministicVoiceRoute
from intent_bridge.intent_engine.supplemental import SupplementalIntentPlanner
from intent_bridge.llm import (
    process_informational_query,
    process_llm_fallback,
    validate_fallback_config,
    validate_music_assistant_config,
)
from intent_bridge.mcp_config import (
    ConfiguredMcpServer,
    load_mcp_servers,
    mcp_agent_instructions,
)
from intent_bridge.music_assistant.client import (
    NativeMusicAssistant,
)
from intent_bridge.music_assistant.intent_executor import MusicAssistantIntentExecutor
from intent_bridge.music_assistant.policy import music_area_player_map
from intent_bridge.runtime.dependencies import runtime
from intent_bridge.runtime.stores import (
    conversation_memories,
)
from intent_bridge.runtime.stores import (
    pending_requests as pending,
)
from intent_bridge.sounds.controller import SOUND_NAMES

# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    runtime.clear_integrations()

    missing = validate_fallback_config() if settings.llm.enabled else []

    ha_missing = []
    if not settings.home_assistant.base_url:
        ha_missing.append("INTENT_BRIDGE_HA_BASE_URL")
    if not settings.home_assistant.access_token:
        ha_missing.append("INTENT_BRIDGE_HA_ACCESS_TOKEN")

    if settings.home_assistant.websocket.enabled and not ha_missing:
        try:
            runtime.ha_ws = HomeAssistantWebSocket(
                settings.home_assistant.base_url, settings.home_assistant.access_token
            )
            await runtime.ha_ws.start()
            publisher = HomeAssistantCatalogPublisher(
                runtime.ha_ws,
                refresh_seconds=settings.home_assistant.websocket.catalog_refresh_seconds,
                event_debounce_seconds=(
                    settings.home_assistant.websocket.catalog_event_debounce_seconds
                ),
                minimum_refresh_seconds=(
                    settings.home_assistant.websocket.catalog_minimum_refresh_seconds
                ),
            )
            await publisher.start()
            runtime.ha_catalog_publisher = publisher
        except Exception:
            log.exception("Failed to initialise direct Home Assistant WebSocket")
    elif settings.home_assistant.websocket.enabled:
        log.warning(
            "Home Assistant deterministic catalog unavailable; missing configuration: %s",
            ", ".join(ha_missing),
        )

    # v6.8.2: direct official Music Assistant client. Its WebSocket receive loop
    # maintains players/queues in memory and the supervisor reconnects with a new
    # client object if the connection is lost.
    music_missing = validate_music_assistant_config()
    runtime.music_assistant = None
    if settings.llm.enabled and not missing and settings.music_assistant.enabled:
        if music_missing:
            log.warning(
                "Music Assistant native integration unavailable; missing configuration: %s",
                ", ".join(music_missing),
            )
        elif MusicAssistantClient is None:
            log.error(
                "Music Assistant native integration unavailable; install "
                "music-assistant-client==1.4.3 (%s)",
                MUSIC_ASSISTANT_CLIENT_IMPORT_ERROR,
            )
        else:
            runtime.music_assistant = NativeMusicAssistant(
                settings.music_assistant.base_url, settings.music_assistant.access_token
            )
            initial_ready = await runtime.music_assistant.start()
            log.info(
                "Music Assistant native transport configured url=%r ready=%s area_map=%s",
                settings.music_assistant.base_url,
                initial_ready,
                bool(music_area_player_map()),
            )

    # One lifecycle manager owns both the HA specialist and user-configured MCP
    # transports. Only custom servers are exposed directly to the fallback agent.
    manager_context = None
    configured_mcp: tuple[ConfiguredMcpServer, ...] = ()
    active_custom_mcp: tuple[ConfiguredMcpServer, ...] = ()
    ha_mcp_server = None
    mcp_servers = []
    if settings.llm.enabled and not missing:
        try:
            configured_mcp = load_mcp_servers(
                settings.mcp.config_path,
                client_session_timeout_seconds=settings.mcp.client_session_timeout_seconds,
            )
            mcp_servers.extend(item.server for item in configured_mcp)
            if configured_mcp:
                log.info(
                    "Loaded custom MCP configuration path=%s active_servers=%s",
                    settings.mcp.config_path,
                    [item.key for item in configured_mcp],
                )
        except Exception:
            configured_mcp = ()
            log.exception("Failed to load custom MCP configuration path=%s", settings.mcp.config_path)

    if settings.llm.enabled and not missing and settings.home_assistant.advanced.enabled:
        try:
            ha_mcp_server = make_ha_mcp_server()
            mcp_servers.append(ha_mcp_server)
            log.info(
                "Starting advanced HA MCP command=%r args=%r",
                settings.home_assistant.advanced.command,
                settings.home_assistant.advanced.args,
            )
        except Exception:
            ha_mcp_server = None
            log.exception("Failed to configure advanced ha-mcp server")

    if mcp_servers:
        try:
            runtime.mcp_manager = MCPServerManager(
                mcp_servers,
                connect_timeout_seconds=max(
                    settings.mcp.connect_timeout_seconds,
                    settings.home_assistant.advanced.connect_timeout_seconds
                    if ha_mcp_server is not None
                    else 0,
                ),
                cleanup_timeout_seconds=max(
                    settings.mcp.cleanup_timeout_seconds,
                    settings.home_assistant.advanced.cleanup_timeout_seconds
                    if ha_mcp_server is not None
                    else 0,
                ),
                strict=False,
                drop_failed_servers=True,
                connect_in_parallel=True,
            )
            manager_context = runtime.mcp_manager
            await manager_context.__aenter__()
            active_names = {server.name for server in runtime.mcp_manager.active_servers}
            active_custom_mcp = tuple(
                item for item in configured_mcp if item.server.name in active_names
            )
            active_ha = next(
                (
                    server
                    for server in runtime.mcp_manager.active_servers
                    if server.name == "Home Assistant Advanced"
                ),
                None,
            )
            if active_ha is not None:
                runtime.advanced_agent = make_advanced_agent([active_ha])
                log.info("Advanced ha-mcp specialist ready")
            elif ha_mcp_server is not None:
                log.warning(
                    "Advanced ha-mcp unavailable; MCP errors=%s", runtime.mcp_manager.errors
                )
            if configured_mcp and len(active_custom_mcp) != len(configured_mcp):
                log.warning(
                    "Some custom MCP servers are unavailable; active=%s errors=%s",
                    [item.key for item in active_custom_mcp],
                    runtime.mcp_manager.errors,
                )
        except Exception:
            runtime.advanced_agent = None
            active_custom_mcp = ()
            log.exception("Failed to initialise MCP servers")

    if settings.llm.enabled:
        if missing:
            log.warning(
                "LLM fallback unavailable; missing configuration: %s",
                ", ".join(missing),
            )
        else:
            try:
                music_tools_enabled = runtime.music_assistant is not None
                custom_mcp_servers = tuple(item.server for item in active_custom_mcp)
                custom_mcp_instructions = mcp_agent_instructions(active_custom_mcp)
                runtime.informational_agent = make_informational_agent(
                    mcp_servers=custom_mcp_servers,
                    mcp_instructions=custom_mcp_instructions,
                )
                runtime.fallback_agent = make_fallback_agent(
                    music_tools_enabled,
                    mcp_servers=custom_mcp_servers,
                    mcp_instructions=custom_mcp_instructions,
                )
                log.info(
                    "LLM fallback ready model=%s direct_ws=%s advanced=%s "
                    "music_assistant=%s custom_mcp=%s music_transport=native_websocket "
                    "native_ready=%s",
                    settings.llm.model,
                    bool(runtime.ha_ws),
                    runtime.advanced_agent is not None,
                    music_tools_enabled,
                    [item.key for item in active_custom_mcp],
                    bool(runtime.music_assistant and runtime.music_assistant.connected),
                )
            except Exception:
                runtime.informational_agent = None
                runtime.fallback_agent = None
                log.exception("Failed to initialise LLM agents")

    try:
        yield
    finally:
        log.info("Shutting down proxy")

        for _session_id, request in list(pending.items()):
            message = "Proxy is shutting down"
            if not request.intent_future.done():
                request.intent_future.set_exception(RuntimeError(message))
            if not request.response_future.done():
                request.response_future.set_exception(RuntimeError(message))
        pending.clear()

        runtime.fallback_agent = None
        runtime.informational_agent = None
        runtime.advanced_agent = None

        if runtime.music_assistant is not None:
            try:
                await runtime.music_assistant.stop()
            except Exception:
                log.exception("Error shutting down native Music Assistant client")
            runtime.music_assistant = None

        try:
            await assistant_feedback.stop_all()
        except Exception:
            log.exception("Error stopping assistant feedback")

        if runtime.ha_catalog_publisher is not None:
            publisher = runtime.ha_catalog_publisher
            runtime.ha_catalog_publisher = None
            try:
                await publisher.stop()
            except Exception:
                log.exception("Error stopping HA catalog publisher")

        if runtime.ha_ws is not None:
            try:
                await runtime.ha_ws.stop()
            except Exception:
                log.exception("Error shutting down HA WebSocket")
            runtime.ha_ws = None

        if manager_context is not None:
            try:
                await manager_context.__aexit__(None, None, None)
            except Exception:
                log.exception("Error shutting down MCP manager")
        runtime.mcp_manager = None


async def _llm_voice_route(request: VoiceRequest) -> str:
    return await process_llm_fallback(
        request.text,
        conversation_key=request.conversation_key,
        client_history=list(request.client_history),
        origin_context=request.origin_context,
    )


async def _informational_voice_route(request: VoiceRequest) -> str:
    return await process_informational_query(
        request.text,
        conversation_key=request.conversation_key,
        client_history=list(request.client_history),
        origin_context=request.origin_context,
    )


def build_voice_pipeline(
    *,
    intent_executor: IntentExecutor | None = None,
    intent_recognizer: IntentRecognizer | None = None,
    fallback_handler: Callable[[VoiceRequest], Awaitable[str]] | None = None,
    informational_handler: Callable[[VoiceRequest], Awaitable[str]] | None = None,
    step_observer: Callable[[PlannedIntent], None] | None = None,
    include_deterministic: bool = True,
) -> VoiceActionPipeline:
    """Compose routes in business priority order.

    New deterministic engines or fallback providers can be inserted here, or a
    completely different pipeline can be supplied through ``app.state``.
    """
    if intent_recognizer is None:
        grammar = load_intent_grammar(
            language=settings.deterministic.language,
            custom_sentences_path=settings.deterministic.custom_sentences_path,
        )
        log.info(
            "OHF/HassIL deterministic grammar ready language=%s custom_files=%d "
            "custom_sentences=%d",
            grammar.language,
            len(grammar.custom_files),
            grammar.custom_sentence_count,
        )
        intent_recognizer = HassilIntentRecognizer(grammar)
    home_assistant_intent_executor = intent_executor or HomeAssistantIntentExecutor(
        settings.home_assistant.base_url,
        settings.home_assistant.access_token,
        timeout=settings.home_assistant.websocket.command_timeout_seconds,
        websocket_provider=lambda: runtime.ha_ws,
    )
    deterministic_engine = DeterministicIntentEngine(
        intent_recognizer,
        HomeAssistantCatalogProvider(
            lambda: runtime.ha_ws,
            publisher_provider=lambda: runtime.ha_catalog_publisher,
        ),
        MusicAssistantIntentExecutor(
            home_assistant_intent_executor,
            native_playback_available=lambda: bool(
                settings.music_assistant.prefer_native_playback
                and runtime.music_assistant is not None
            ),
        ),
        preferred_planner=IntentPlannerChain(
            (MeasurementIntentPlanner(), SupplementalIntentPlanner())
        ),
        fallback_planner=NaturalLanguageIntentPlanner(),
        step_observer=step_observer,
        default_response=settings.api.action_confirmation,
    )
    routes = []
    if include_deterministic:
        routes.append(
            ConversationalDeterministicVoiceRoute(
                deterministic_engine,
                ambiguous_target_fallback_enabled=(
                    settings.llm.ambiguous_target_fallback_enabled
                ),
            )
        )
    routes.append(
        InformationalVoiceRoute(
            "informational-llm",
            informational_handler or _informational_voice_route,
        )
    )
    routes.append(FunctionVoiceRoute("llm-ha-ws", fallback_handler or _llm_voice_route))
    return VoiceActionPipeline(
        routes,
        failure_response=settings.api.voice_failure_response,
    )


# ---------------------------------------------------------------------------
# FastAPI / OpenAI-compatible API
# ---------------------------------------------------------------------------

api_router = APIRouter()
SOUNDS_DIRECTORY = Path(__file__).resolve().parent / "sounds"


@dataclass(frozen=True, slots=True)
class ApplicationDependencies:
    """Narrow set of application services consumed by HTTP routes."""

    voice_pipeline: VoiceRequestHandler


def create_app(
    pipeline: VoiceRequestHandler | None = None,
    *,
    dependencies: ApplicationDependencies | None = None,
) -> FastAPI:
    if pipeline is not None and dependencies is not None:
        raise ValueError("Supply either pipeline or dependencies, not both")
    dependencies = dependencies or ApplicationDependencies(
        voice_pipeline=pipeline or build_voice_pipeline()
    )
    application = FastAPI(
        title="Home Intent + HA WebSocket + Native Music Assistant OpenAI Proxy",
        version=settings.api.version,
        lifespan=lifespan,
    )
    application.state.dependencies = dependencies
    # Compatibility for deployments/tests that inspected this state attribute.
    application.state.voice_pipeline = dependencies.voice_pipeline
    application.include_router(api_router)
    return application


def extract_user_message(body: dict) -> str:
    for message in reversed(body.get("messages", [])):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        return _message_text(message)
    return ""


@api_router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    if body.get("stream"):
        raise HTTPException(status_code=400, detail="Streaming is not supported")

    text = extract_user_message(body)
    if not text:
        raise HTTPException(status_code=400, detail="No user message supplied")

    conversation_key = get_conversation_key(request, body)
    client_history = extract_client_history(body)
    origin_context = await extract_voice_origin_context(request, body)
    feedback_handle = await assistant_feedback.begin(
        origin_context,
        led=False,
        sounds=True,
    )

    if body.get("reset_conversation"):
        await clear_conversation_history(conversation_key)
        client_history = []
        log.info("CONVERSATION RESET key=%r", conversation_key)

    voice_request = VoiceRequest(
        text=text,
        conversation_key=conversation_key,
        client_history=tuple(client_history),
        origin_context=origin_context,
    )
    try:
        request_app = getattr(request, "app", None)
        dependencies = getattr(getattr(request_app, "state", None), "dependencies", None)
        active_pipeline = (
            dependencies.voice_pipeline
            if dependencies is not None
            else app.state.dependencies.voice_pipeline
        )
        result = await active_pipeline.handle(voice_request)
    except VoicePipelineError as exc:
        await assistant_feedback.complete(feedback_handle, success=False)
        log.exception("All voice action routes failed")
        raise HTTPException(
            status_code=502,
            detail=f"All voice action routes failed. {exc}",
        ) from exc
    except Exception:
        await assistant_feedback.complete(feedback_handle, success=False)
        raise

    is_error_response = result.route == "voice-error-response"
    use_success_sound = (
        settings.assistant.sounds_enabled
        and not is_error_response
        and not result.speech.strip()
    )
    use_error_sound = settings.assistant.sounds_enabled and is_error_response
    use_terminal_sound = use_success_sound or use_error_sound
    await assistant_feedback.complete(
        feedback_handle,
        success=not is_error_response,
        play_terminal_sound=use_terminal_sound,
    )

    for failure in result.failures:
        log.warning(
            "VOICE ROUTE FALLBACK route=%s text=%r reason=%s", failure.route, text, failure.error
        )
    speech = result.speech
    response_speech = "" if use_terminal_sound else speech
    route = result.route
    log.info("ROUTE COMPLETE route=%s text=%r response=%r", route, text, speech)

    if client_history:
        await seed_conversation_history(conversation_key, client_history)
    await remember_conversation_turn(conversation_key, text, speech)
    history_messages = len(await get_conversation_history(conversation_key))

    return {
        "id": "chatcmpl-homeintent-" + uuid.uuid4().hex,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": settings.api.model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": response_speech},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        "home_intent_proxy": {
            "route": route,
            "conversation_history_messages": history_messages,
            "voice_origin": {
                "device_name": (origin_context or {}).get("device_name"),
                "area_name": (origin_context or {}).get("area_name"),
                "area_id": (origin_context or {}).get("area_id"),
                "floor_name": (origin_context or {}).get("floor_name"),
                "source": (origin_context or {}).get("source"),
            },
        },
    }


@api_router.get("/assistant/sounds/{sound_name}.mp3", name="assistant_sound")
async def assistant_sound(sound_name: str):
    """Serve the fixed bundled sound set for Home Assistant media players."""
    if sound_name not in SOUND_NAMES:
        raise HTTPException(status_code=404, detail="Unknown assistant sound")
    return FileResponse(
        SOUNDS_DIRECTORY / f"{sound_name}.mp3",
        media_type="audio/mpeg",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@api_router.get("/v1/models")
async def models():
    return {
        "object": "list",
        "data": [
            {
                "id": settings.api.model_name,
                "object": "model",
                "created": 0,
                "owned_by": "home-intent",
            }
        ],
    }


@api_router.get("/health")
async def health():
    ws_ready = bool(runtime.ha_ws and runtime.ha_ws.ready.is_set())
    ha_instance_config = getattr(runtime.ha_ws, "config", {}) or {}
    catalog_publisher = runtime.ha_catalog_publisher

    ma_manager = runtime.music_assistant
    ma_client = ma_manager.client if ma_manager is not None else None
    ma_info = getattr(ma_client, "server_info", None) if ma_client is not None else None

    return {
        "status": "ok",
        "version": settings.api.version,
        "pending_requests": len(pending),
        "deterministic_timeout_seconds": settings.deterministic.timeout_seconds,
        "deterministic_response_grace_seconds": settings.deterministic.response_grace_seconds,
        "deterministic_minimum_confidence": settings.deterministic.minimum_confidence,
        "deterministic_default_response": settings.deterministic.default_response,
        "deterministic_error_phrases": list(settings.deterministic.error_phrases),
        "llm_enabled": settings.llm.enabled,
        "llm_ambiguous_target_fallback_enabled": (
            settings.llm.ambiguous_target_fallback_enabled
        ),
        "voice_failure_response": settings.api.voice_failure_response,
        "base_url": settings.api.base_url or None,
        "assistant_led_enabled": settings.assistant.led_enabled,
        "assistant_led_active": assistant_feedback.leds.active_count,
        "assistant_led_color": settings.assistant.led_color,
        "assistant_led_effect": settings.assistant.led_effect,
        "assistant_led_software_pulse": settings.assistant.led_software_pulse_enabled,
        "assistant_led_pulse_interval_seconds": settings.assistant.led_pulse_interval_seconds,
        "assistant_led_domains": list(settings.assistant.led_domains),
        "assistant_led_last_target": assistant_feedback.leds.last_target,
        "assistant_led_last_error": assistant_feedback.leds.last_error,
        "assistant_sounds_enabled": settings.assistant.sounds_enabled,
        "assistant_sounds_last_target": assistant_feedback.sounds.last_target,
        "assistant_sounds_last_error": assistant_feedback.sounds.last_error,
        "llm_fallback_ready": runtime.fallback_agent is not None,
        "informational_llm_ready": runtime.informational_agent is not None,
        "llm_model": settings.llm.model or None,
        "music_assistant_enabled": settings.music_assistant.enabled,
        "music_assistant_prefer_native_playback": (
            settings.music_assistant.prefer_native_playback
        ),
        "music_assistant_transport": "native_websocket",
        "music_assistant_client_package_available": MusicAssistantClient is not None,
        "music_assistant_client_import_error": MUSIC_ASSISTANT_CLIENT_IMPORT_ERROR,
        "music_assistant_ready": bool(ma_manager and ma_manager.connected),
        "music_assistant_url": settings.music_assistant.base_url or None,
        "music_assistant_server_version": getattr(ma_info, "server_version", None),
        "music_assistant_schema_version": getattr(ma_info, "schema_version", None),
        "music_assistant_players_cached": (
            len(getattr(ma_client.players, "players", []) or []) if ma_client is not None else 0
        ),
        "music_assistant_queues_cached": (
            len(getattr(ma_client.player_queues, "player_queues", []) or [])
            if ma_client is not None
            else 0
        ),
        "music_assistant_connection_count": (
            ma_manager.connection_count if ma_manager is not None else 0
        ),
        "music_assistant_reconnect_count": (
            ma_manager.reconnect_count if ma_manager is not None else 0
        ),
        "music_assistant_last_connected_at": (
            ma_manager.last_connected_at if ma_manager is not None else None
        ),
        "music_assistant_last_error": (ma_manager.last_error if ma_manager is not None else None),
        "music_assistant_area_player_map_configured": bool(music_area_player_map()),
        "music_assistant_terminal_actions": settings.music_assistant.terminal_actions_enabled,
        "music_assistant_replay_guard_seconds": settings.music_assistant.replay_guard_seconds,
        "music_assistant_connect_timeout_seconds": settings.music_assistant.connect_timeout_seconds,
        "music_assistant_command_timeout_seconds": settings.music_assistant.command_timeout_seconds,
        "music_assistant_play_ack_wait_seconds": settings.music_assistant.play_ack_timeout_seconds,
        "music_assistant_play_completion_timeout_seconds": settings.music_assistant.play_completion_timeout_seconds,
        "music_assistant_first_audio_timeout_seconds": settings.music_assistant.first_audio_timeout_seconds,
        "music_assistant_background_timeout_seconds": settings.music_assistant.background_timeout_seconds,
        "music_assistant_first_audio_poll_seconds": settings.music_assistant.first_audio_poll_seconds,
        "music_assistant_radio_seed_top_n": settings.music_assistant.radio_seed_top_n,
        "music_assistant_radio_seed_strategy": settings.music_assistant.radio_seed_strategy,
        "music_assistant_inflight_playbacks": (
            ma_manager.inflight_playback_count if ma_manager is not None else 0
        ),
        "music_assistant_post_action_settle_seconds": settings.music_assistant.post_action_settle_seconds,
        "llm_base_url": settings.llm.base_url if settings.llm.enabled else None,
        "ha_ws_enabled": settings.home_assistant.websocket.enabled,
        "ha_ws_ready": ws_ready,
        "ha_ws_url": runtime.ha_ws.ws_url if runtime.ha_ws is not None else None,
        "ha_ws_states_cached": len(runtime.ha_ws.states) if runtime.ha_ws is not None else 0,
        "ha_ws_service_domains_cached": len(runtime.ha_ws.services)
        if runtime.ha_ws is not None
        else 0,
        "ha_ws_entity_registry_cached": len(runtime.ha_ws.entity_registry)
        if runtime.ha_ws is not None
        else 0,
        "ha_ws_device_registry_cached": len(runtime.ha_ws.devices)
        if runtime.ha_ws is not None
        else 0,
        "ha_ws_area_registry_cached": len(runtime.ha_ws.areas) if runtime.ha_ws is not None else 0,
        "ha_catalog_snapshot_ready": bool(
            catalog_publisher is not None and catalog_publisher.snapshot() is not None
        ),
        "ha_catalog_snapshot_generation": (
            catalog_publisher.generation if catalog_publisher is not None else 0
        ),
        "ha_catalog_snapshot_age_seconds": (
            catalog_publisher.age_seconds if catalog_publisher is not None else None
        ),
        "ha_catalog_snapshot_build_failures": (
            catalog_publisher.build_failures if catalog_publisher is not None else 0
        ),
        "ha_catalog_refresh_seconds": settings.home_assistant.websocket.catalog_refresh_seconds,
        "ha_catalog_event_debounce_seconds": (
            settings.home_assistant.websocket.catalog_event_debounce_seconds
        ),
        "ha_catalog_minimum_refresh_seconds": (
            settings.home_assistant.websocket.catalog_minimum_refresh_seconds
        ),
        "ha_instance_config_ready": bool(ha_instance_config),
        "ha_instance_time_zone": ha_instance_config.get("time_zone"),
        "ha_instance_language": ha_instance_config.get("language"),
        "ha_instance_country": ha_instance_config.get("country"),
        "voice_origin_context_enabled": settings.voice_origin.enabled,
        "voice_origin_area_bias": settings.voice_origin.area_bias_enabled,
        "voice_origin_soft_area_ranking": settings.voice_origin.soft_ranking_enabled,
        "generic_light_indicator_penalty": settings.home_assistant.penalize_indicator_lights,
        "voice_origin_system_prompt_fallback": settings.voice_origin.system_prompt_fallback_enabled,
        "ha_ws_state_events_seen": runtime.ha_ws.state_event_count
        if runtime.ha_ws is not None
        else 0,
        "ha_ws_reconnect_count": runtime.ha_ws.reconnect_count if runtime.ha_ws is not None else 0,
        "ha_ws_last_error": runtime.ha_ws.last_error if runtime.ha_ws is not None else None,
        "ha_ws_connect_timeout_seconds": settings.home_assistant.websocket.connect_timeout_seconds,
        "ha_ws_command_timeout_seconds": settings.home_assistant.websocket.command_timeout_seconds,
        "ha_ws_service_cache_ttl_seconds": settings.home_assistant.websocket.service_cache_ttl_seconds,
        "ha_ws_registry_cache_ttl_seconds": settings.home_assistant.websocket.registry_cache_ttl_seconds,
        "ha_service_schema_auto_repair": settings.home_assistant.schema_auto_repair_enabled,
        "data_response_recovery_enabled": settings.llm.data_recovery_enabled,
        "data_response_recovery_max_chars": settings.llm.data_recovery_max_chars,
        "ha_advanced_enabled": settings.home_assistant.advanced.enabled,
        "ha_advanced_ready": runtime.advanced_agent is not None,
        "ha_advanced_command": settings.home_assistant.advanced.command
        if settings.home_assistant.advanced.enabled
        else None,
        "ha_advanced_args": settings.home_assistant.advanced.args
        if settings.home_assistant.advanced.enabled
        else None,
        "ha_advanced_tool_search": settings.home_assistant.advanced.tool_search_enabled,
        "ha_advanced_pinned_tools": settings.home_assistant.advanced.pinned_tools,
        "mcp_client_session_timeout_seconds": settings.mcp.client_session_timeout_seconds,
        "mcp_active_servers": (
            [server.name for server in runtime.mcp_manager.active_servers]
            if runtime.mcp_manager is not None
            else []
        ),
        "mcp_failed_servers": (
            [server.name for server in runtime.mcp_manager.failed_servers]
            if runtime.mcp_manager is not None
            else []
        ),
        "voice_action_confirmation": settings.api.action_confirmation,
        "voice_response_max_chars": settings.api.spoken_response_max_chars,
        "llm_max_turns": settings.llm.max_turns,
        "ha_advanced_max_turns": settings.home_assistant.advanced.max_turns,
        "conversation_enabled": settings.conversation.enabled,
        "conversation_history_turns": settings.conversation.history_turns,
        "conversation_history_ttl_seconds": settings.conversation.ttl_seconds,
        "conversation_history_active_sessions": len(conversation_memories),
        "conversation_history_max_sessions": settings.conversation.max_sessions,
        "local_timezone": settings.api.timezone,
        "locale": settings.api.locale,
        "location": settings.api.location or None,
        "timezone_override_configured": settings.api.timezone_explicit,
        "locale_override_configured": settings.api.locale_explicit,
        "location_override_configured": settings.api.location_explicit,
    }


app = create_app()
