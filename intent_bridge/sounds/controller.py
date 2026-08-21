"""Resolve a satellite-associated player and play hosted assistant sounds."""

from dataclasses import dataclass
from typing import Any

from intent_bridge.config import log, settings
from intent_bridge.core.text import normalize_search_text
from intent_bridge.home_assistant.catalog import unique_assist_satellite_device
from intent_bridge.home_assistant.client import _require_ha_ws
from intent_bridge.indicators.topology import (
    connected_satellite_device_scope,
    indicator_relation_bonus,
)

SOUND_NAMES = frozenset({"processing", "success", "error"})


@dataclass(frozen=True, slots=True)
class AssistantSoundTarget:
    entity_id: str
    satellite_entity_id: str
    device_id: str
    match_reason: str


def assistant_sound_url(sound_name: str) -> str:
    """Return the externally resolvable URL Home Assistant should fetch."""
    if sound_name not in SOUND_NAMES:
        raise ValueError(f"Unknown assistant sound: {sound_name}")
    return f"{settings.api.base_url}/assistant/sounds/{sound_name}.mp3"


def _entity_identity(client: Any, entity_id: str) -> str:
    state = client.states.get(entity_id, {})
    attributes = state.get("attributes", {}) if isinstance(state, dict) else {}
    if not isinstance(attributes, dict):
        attributes = {}
    registry = client.entity_registry.get(entity_id, {})
    device_id = registry.get("di") if isinstance(registry, dict) else None
    device = client.devices.get(device_id, {}) if isinstance(device_id, str) else {}
    parts = [
        entity_id.split(".", 1)[-1],
        attributes.get("friendly_name"),
        attributes.get("device_class"),
        registry.get("en") if isinstance(registry, dict) else None,
        device.get("name_by_user") if isinstance(device, dict) else None,
        device.get("name") if isinstance(device, dict) else None,
    ]
    return " ".join(normalize_search_text(part) for part in parts if part)


async def resolve_assistant_sound_target(
    origin_context: dict[str, Any] | None,
) -> AssistantSoundTarget | None:
    """Find one media player in the calling Assist satellite's device topology."""
    if not settings.assistant.sounds_enabled or not origin_context:
        return None

    client = await _require_ha_ws()
    await client.refresh_registries()
    origin_device_id = str(origin_context.get("device_id") or "").strip() or None
    origin_device_name = str(origin_context.get("device_name") or "").strip() or None
    origin_area_id = str(origin_context.get("area_id") or "").strip() or None
    origin_area_name = str(origin_context.get("area_name") or "").strip() or None
    resolved_origin = client.resolve_device_origin(
        device_id=origin_device_id,
        device_name=origin_device_name,
        area_id=origin_area_id,
        area_name=origin_area_name,
    )

    anchor_device_id = resolved_origin.get("device_id")
    satellite_entity_id: str | None = None
    match_reason = "origin_device_assist_satellite"
    if isinstance(anchor_device_id, str) and anchor_device_id:
        satellites = [
            entity_id
            for entity_id in client.states
            if entity_id.startswith("assist_satellite.")
            and client.entity_registry.get(entity_id, {}).get("di") == anchor_device_id
        ]
        if len(satellites) == 1:
            satellite_entity_id = satellites[0]

    if satellite_entity_id is None:
        satellite = unique_assist_satellite_device(
            client,
            area_id=origin_area_id or resolved_origin.get("area_id"),
        )
        if satellite is not None:
            satellite_entity_id = satellite.entity_id
            anchor_device_id = satellite.device_id
            match_reason = "sole_assist_satellite_in_origin_area"

    if satellite_entity_id is None:
        resolved_area_id, resolved_area_name = client.resolve_area_reference(
            area_id=origin_area_id or resolved_origin.get("area_id"),
            area_name=origin_area_name or resolved_origin.get("area_name"),
        )
        satellites = []
        for entity_id, state in client.states.items():
            if not entity_id.startswith("assist_satellite."):
                continue
            context = client._entity_context(entity_id, state)
            if resolved_area_id and context.get("area_id") == resolved_area_id:
                satellites.append(entity_id)
            elif (
                not resolved_area_id
                and resolved_area_name
                and normalize_search_text(context.get("area_name"))
                == normalize_search_text(resolved_area_name)
            ):
                satellites.append(entity_id)
        if len(satellites) != 1:
            log.info("ASSISTANT SOUND skipped: ambiguous satellite matches=%s", satellites)
            return None
        satellite_entity_id = satellites[0]
        match_reason = "sole_assist_satellite_in_origin_area"
        anchor_device_id = client.entity_registry.get(satellite_entity_id, {}).get("di")

    if not isinstance(anchor_device_id, str) or not anchor_device_id:
        log.info("ASSISTANT SOUND skipped: satellite has no registry device")
        return None

    device_scope = connected_satellite_device_scope(client, anchor_device_id, max_depth=2)
    candidates: list[tuple[float, int, str, str, str]] = []
    for entity_id, state in client.states.items():
        if not entity_id.startswith("media_player."):
            continue
        registry = client.entity_registry.get(entity_id, {})
        device_id = registry.get("di") if isinstance(registry, dict) else None
        if not isinstance(device_id, str) or device_id not in device_scope:
            continue
        depth, relation = device_scope[device_id]
        identity = _entity_identity(client, entity_id)
        tokens = set(identity.split())
        attributes = state.get("attributes", {}) if isinstance(state, dict) else {}
        device_class = normalize_search_text(
            attributes.get("device_class") if isinstance(attributes, dict) else None
        )
        score = 100.0 + indicator_relation_bonus(depth, relation)
        reasons = [relation]
        if device_class == "speaker":
            score += 60.0
            reasons.append("speaker_device_class")
        for token in ("speaker", "audio", "playback", "assistant", "satellite"):
            if token in tokens:
                score += 12.0
                reasons.append(token)
        if device_class == "tv" or tokens & {"tv", "television", "projector"}:
            score -= 50.0
            reasons.append("video_device_penalty")
        candidates.append((score, depth, entity_id, device_id, "+".join(reasons)))

    if not candidates:
        log.info(
            "ASSISTANT SOUND skipped: no media player for satellite=%s", satellite_entity_id
        )
        return None
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    top = candidates[0]
    if len(candidates) > 1 and candidates[1][0] == top[0] and candidates[1][1] == top[1]:
        log.info(
            "ASSISTANT SOUND skipped: media player tie candidates=%s",
            [(item[2], item[0]) for item in candidates if item[:2] == top[:2]],
        )
        return None
    return AssistantSoundTarget(
        entity_id=top[2],
        satellite_entity_id=satellite_entity_id,
        device_id=top[3],
        match_reason=f"{match_reason}:{top[4]}:depth{top[1]}",
    )


class AssistantSounds:
    def __init__(self) -> None:
        self.last_error: str | None = None
        self.last_target: dict[str, str] | None = None

    async def resolve(
        self, origin_context: dict[str, Any] | None
    ) -> AssistantSoundTarget | None:
        try:
            target = await resolve_assistant_sound_target(origin_context)
            if target is not None:
                self.last_target = {
                    "entity_id": target.entity_id,
                    "satellite_entity_id": target.satellite_entity_id,
                    "device_id": target.device_id,
                    "match_reason": target.match_reason,
                }
            self.last_error = None
            return target
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            log.warning("ASSISTANT SOUND target resolution failed: %s", self.last_error)
            return None

    async def play(self, target: AssistantSoundTarget | None, sound_name: str) -> bool:
        if target is None:
            return False
        try:
            client = await _require_ha_ws()
            url = assistant_sound_url(sound_name)
            reply = await client.command(
                {
                    "type": "call_service",
                    "domain": "media_player",
                    "service": "play_media",
                    "service_data": {
                        "media_content_id": url,
                        "media_content_type": "audio/mpeg",
                    },
                    "target": {"entity_id": target.entity_id},
                    "return_response": False,
                },
                timeout=settings.home_assistant.websocket.command_timeout_seconds,
            )
            if reply.get("success") is not True:
                raise RuntimeError(f"Home Assistant rejected media playback: {reply.get('error')}")
            self.last_error = None
            log.info(
                "ASSISTANT SOUND played event=%s entity=%s url=%s",
                sound_name,
                target.entity_id,
                url,
            )
            return True
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            log.warning("ASSISTANT SOUND playback failed event=%s: %s", sound_name, self.last_error)
            return False


assistant_sounds = AssistantSounds()


__all__ = [
    "AssistantSoundTarget",
    "AssistantSounds",
    "SOUND_NAMES",
    "assistant_sound_url",
    "assistant_sounds",
    "resolve_assistant_sound_target",
]
