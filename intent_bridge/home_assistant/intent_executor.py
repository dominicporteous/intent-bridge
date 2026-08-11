"""Execute recognized OHF calls through Home Assistant's official intent API."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any

import httpx
from agents import Runner

from intent_bridge.config import settings
from intent_bridge.core.voice import RouteDeclined
from intent_bridge.home_assistant.advanced import run_advanced_agent
from intent_bridge.intent_engine.models import ExecutionResult, OhfIntentCall
from intent_bridge.runtime.dependencies import runtime
from intent_bridge.runtime.execution import voice_tool_run_state

log = logging.getLogger(__name__)

# Kept as a module-level seam so deployments/tests can replace advanced HA
# resolution without patching the agent harness itself.
ha_advanced = run_advanced_agent

_EXACT_TARGET_SERVICE_BY_INTENT = {
    "HassVacuumStart": "start",
    "HassVacuumReturnToBase": "return_to_base",
}


def _contains_internal_match_failure(speech: str) -> bool:
    return any(
        marker in speech
        for marker in (
            "<MatchFailedError",
            "MatchTargetsResult(is_match=False",
            "MatchFailedReason.",
        )
    )


def _exact_target_service(call: OhfIntentCall, entity_id: str) -> str | None:
    service = _EXACT_TARGET_SERVICE_BY_INTENT.get(call.intent_name)
    if service:
        return service
    domain = entity_id.split(".", 1)[0]
    if call.intent_name == "HassTurnOn":
        return {"cover": "open_cover", "lock": "lock", "vacuum": "start"}.get(
            domain, "turn_on"
        )
    if call.intent_name == "HassTurnOff":
        return {"cover": "close_cover", "lock": "unlock", "vacuum": "stop"}.get(
            domain, "turn_off"
        )
    return None

def _error_detail(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text.strip() or f"HTTP {response.status_code}"
    if isinstance(body, Mapping):
        for key in ("message", "error", "detail"):
            value = body.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, Mapping):
                nested = value.get("message")
                if isinstance(nested, str) and nested.strip():
                    return nested.strip()
    return f"HTTP {response.status_code}"


def _speech_from_response(body: Mapping[str, Any]) -> str:
    speech = body.get("speech")
    if isinstance(speech, Mapping):
        plain = speech.get("plain")
        if isinstance(plain, Mapping):
            value = plain.get("speech")
            if isinstance(value, str):
                return value.strip()
    nested_response = body.get("response")
    if isinstance(nested_response, Mapping):
        return _speech_from_response(nested_response)
    return ""


def _entity_ids_from_intent_response(body: Mapping[str, Any]) -> tuple[str, ...]:
    data = body.get("data")
    if not isinstance(data, Mapping):
        return ()

    ids: list[str] = []
    for key in ("success", "targets"):
        items = data.get(key)
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, str):
                ids.append(item)
                continue
            if not isinstance(item, Mapping):
                continue
            entity_id = item.get("entity_id") or item.get("id")
            if isinstance(entity_id, str):
                ids.append(entity_id)

    return tuple(dict.fromkeys(ids))


def _summarise_intent_entity_state(state: Mapping[str, Any]) -> str:
    """Produce a concise spoken summary for a Home Assistant entity state."""
    entity_id = state.get("entity_id")
    if not isinstance(entity_id, str):
        return ""
    raw_state = state.get("state")
    if raw_state is None:
        return ""

    state_text = str(raw_state).strip()
    if not state_text:
        return ""

    attributes = state.get("attributes")
    if not isinstance(attributes, Mapping):
        attributes = {}

    friendly_name = attributes.get("friendly_name") or entity_id
    if not isinstance(friendly_name, str):
        friendly_name = entity_id

    domain = entity_id.split(".", 1)[0] if "." in entity_id else ""
    unit = attributes.get("unit_of_measurement") or attributes.get("temperature_unit") or ""
    if not isinstance(unit, str):
        unit = ""
    unit = unit.strip()

    def _format_value(value: Any) -> str | None:
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, (int, float)):
            return str(value)
        return None

    def _percentage(value: Any) -> str | None:
        if isinstance(value, (int, float)):
            try:
                if 0 < value <= 1:
                    return f"{int(round(value * 100))}%"
                return f"{int(round(value))}%"
            except Exception:
                return str(value)
        return None

    def _with_unit(value: Any, unit_text: str) -> str | None:
        formatted = _format_value(value)
        if not formatted:
            return None
        return f"{formatted} {unit_text}" if unit_text else formatted

    if domain == "weather":
        temperature = _with_unit(attributes.get("temperature"), unit)
        weather_summary = _format_value(attributes.get("condition")) or _format_value(attributes.get("forecast"))
        if weather_summary and temperature:
            return f"{friendly_name} is {weather_summary} and {temperature}"
        if temperature:
            return f"{friendly_name} is {state_text} and {temperature}"
        return f"{friendly_name} is {state_text}"

    if domain == "climate":
        hvac_action = _format_value(attributes.get("hvac_action"))
        hvac_mode = _format_value(attributes.get("hvac_mode"))
        current = _with_unit(attributes.get("current_temperature"), unit)
        target = _with_unit(
            attributes.get("temperature")
            or attributes.get("target_temp")
            or attributes.get("target_temperature"),
            unit,
        )
        summary_parts: list[str] = []
        if hvac_action:
            summary_parts.append(hvac_action)
        elif hvac_mode and hvac_mode != state_text:
            summary_parts.append(hvac_mode)
        if target:
            summary_parts.append(f"target {target}")
        if current:
            summary_parts.append(f"current {current}")
        if summary_parts:
            return f"{friendly_name} is {' '.join(summary_parts)}"
        if target:
            return f"{friendly_name} is {state_text} and set to {target}"
        return f"{friendly_name} is {state_text}"

    if domain == "media_player":
        title = _format_value(attributes.get("media_title"))
        artist = _format_value(attributes.get("media_artist"))
        volume = _percentage(attributes.get("volume_level"))
        if title and artist:
            verb = "playing" if state_text.casefold() == "playing" else f"{state_text} playing"
            return f"{friendly_name} is {verb} {title} by {artist}"
        if title:
            verb = "playing" if state_text.casefold() == "playing" else f"{state_text} playing"
            return f"{friendly_name} is {verb} {title}"
        if volume:
            return f"{friendly_name} is {state_text} at {volume} volume"
        return f"{friendly_name} is {state_text}"

    if domain == "light":
        brightness = _percentage(attributes.get("brightness"))
        if brightness and state_text.lower() in {"on", "off"}:
            return f"{friendly_name} is {state_text} at {brightness}"
        return f"{friendly_name} is {state_text}"

    if domain == "cover":
        position = _percentage(attributes.get("current_position") or attributes.get("position"))
        if position:
            return f"{friendly_name} is {state_text} at {position}"
        return f"{friendly_name} is {state_text}"

    if domain == "fan":
        fan_mode = _format_value(attributes.get("fan_mode"))
        percentage = _percentage(attributes.get("percentage"))
        if fan_mode:
            return f"{friendly_name} is {state_text} in {fan_mode} mode"
        if percentage:
            return f"{friendly_name} is {state_text} at {percentage}"
        return f"{friendly_name} is {state_text}"

    if domain == "vacuum":
        battery = _percentage(attributes.get("battery_level"))
        if battery:
            return f"{friendly_name} is {state_text} with {battery} battery"
        return f"{friendly_name} is {state_text}"

    if domain == "lock":
        return f"{friendly_name} is {state_text}"

    if domain == "water_heater":
        current = _with_unit(attributes.get("current_temperature"), unit)
        if current:
            return f"{friendly_name} is {state_text} at {current}"
        return f"{friendly_name} is {state_text}"

    if domain in {"humidifier", "dehumidifier"}:
        humidity = _percentage(attributes.get("current_humidity"))
        if humidity:
            return f"{friendly_name} is {state_text} at {humidity} humidity"
        return f"{friendly_name} is {state_text}"

    if domain == "sensor":
        weather_summary = _format_value(attributes.get("condition")) or _format_value(attributes.get("forecast"))
        if weather_summary:
            temperature = _with_unit(attributes.get("temperature"), unit)
            if temperature:
                return f"{friendly_name} is {weather_summary} and {temperature}"
            return f"{friendly_name} is {weather_summary}"
        if "weather" in friendly_name.casefold() or "weather" in entity_id.casefold():
            # A raw numeric integration status is not necessarily an outdoor
            # temperature. Let the HA-aware agent inspect the integration
            # instead of confidently speaking a misleading value.
            return ""
        if unit:
            return f"{friendly_name} is {state_text} {unit}"
        if state_text.casefold() not in {"unknown", "unavailable", "none"}:
            return f"{friendly_name} is {state_text}"
        return ""

    if domain in {"alarm_control_panel", "camera", "binary_sensor", "switch", "button", "scene"}:
        return f"{friendly_name} is {state_text}{(' ' + unit) if unit else ''}"

    extras: list[str] = []
    for attr, label in (
        ("temperature", "temperature"),
        ("current_temperature", "current temperature"),
        ("humidity", "humidity"),
        ("battery_level", "battery level"),
        ("brightness", "brightness"),
        ("position", "position"),
        ("volume_level", "volume level"),
    ):
        if attr in attributes:
            value = attributes[attr]
            if attr in {"brightness", "volume_level"}:
                extra = _percentage(value)
            elif attr.endswith("temperature"):
                extra = _with_unit(value, unit)
            else:
                extra = _format_value(value)
            if extra:
                extras.append(f"{label} {extra}")
    if extras:
        return f"{friendly_name} is {state_text} ({', '.join(extras)})"

    return f"{friendly_name} is {state_text}{(' ' + unit) if unit else ''}"


class HomeAssistantIntentExecutor:
    def __init__(
        self,
        base_url: str,
        access_token: str,
        *,
        timeout: float = 15.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._configured = bool(base_url.strip() and access_token.strip())
        self._base_url = base_url.rstrip("/")
        self._url = f"{base_url.rstrip('/')}/api/intent/handle"
        self._access_token = access_token
        self._timeout = timeout
        self._client = client

    async def execute(self, call: OhfIntentCall) -> ExecutionResult:
        if not self._configured:
            raise RouteDeclined("Home Assistant intent API is not configured")
        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json",
        }
        payload = {"name": call.intent_name, "data": dict(call.data)}

        log.info(
            "Executing Home Assistant intent intent=%s data=%s",
            call.intent_name,
            dict(call.data),
        )

        if self._client is not None:
            response = await self._client.post(
                self._url,
                headers=headers,
                json=payload,
                timeout=self._timeout,
            )
        else:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self._url,
                    headers=headers,
                    json=payload,
                    timeout=self._timeout,
                )

        if not response.is_success:
            raise RuntimeError(
                f"Home Assistant intent {call.intent_name} failed: {_error_detail(response)}"
            )

        try:
            body = response.json()
            log.info(
                "Home Assistant intent response intent=%s body=%s",
                call.intent_name,
                body,
            )
        except ValueError as exc:
            raise RuntimeError("Home Assistant returned a non-JSON intent response") from exc
        if not isinstance(body, Mapping):
            raise RuntimeError("Home Assistant returned an invalid intent response")

        speech = _speech_from_response(body)
        action_succeeded = False
        if speech and _contains_internal_match_failure(speech):
            entity_id = call.data.get("entity_id")
            entity_id = entity_id.strip() if isinstance(entity_id, str) else ""
            service = _exact_target_service(call, entity_id) if entity_id else None
            ha_ws = runtime.ha_ws
            if service and ha_ws is not None:
                domain = entity_id.split(".", 1)[0]
                log.warning(
                    "HA intent target match failed; retrying exact target over WebSocket "
                    "intent=%s domain=%s service=%s entity=%s",
                    call.intent_name,
                    domain,
                    service,
                    entity_id,
                )
                reply = await ha_ws.command(
                    {
                        "type": "call_service",
                        "domain": domain,
                        "service": service,
                        "service_data": {},
                        "target": {"entity_id": entity_id},
                        "return_response": False,
                    },
                    timeout=settings.home_assistant.websocket.command_timeout_seconds,
                )
                if reply.get("success") is True:
                    log.info(
                        "Exact-target WebSocket retry succeeded entity=%s",
                        entity_id,
                    )
                    action_succeeded = True
                    speech = settings.api.action_confirmation
                else:
                    log.warning(
                        "Exact-target WebSocket retry failed entity=%s error=%s",
                        entity_id,
                        reply.get("error"),
                    )
                    raise RouteDeclined("Home Assistant could not match or control the exact target")
            else:
                raise RouteDeclined("Home Assistant could not match the resolved target")
        if not speech and body.get("response_type") == "action_done":
            response_data = body.get("data")
            if isinstance(response_data, Mapping):
                succeeded = response_data.get("success")
                failed = response_data.get("failed")
                if isinstance(succeeded, list) and succeeded and not failed:
                    # State updates arrive asynchronously after HA acknowledges
                    # an action. Do not summarize the pre-action cache here.
                    action_succeeded = True
                    speech = settings.api.action_confirmation
        if not speech and not action_succeeded:
            entity_ids = _entity_ids_from_intent_response(body)
            user_text = voice_tool_run_state.request_text or call.intent_name
            cached_entity_found = False

            async def _summarise_state_body(state_body: Mapping[str, Any]) -> str:
                summary = _summarise_intent_entity_state(state_body)
                if summary:
                    return summary
                return ""

            async def _summary_from_websocket() -> str:
                nonlocal cached_entity_found
                ha_ws = runtime.ha_ws
                if ha_ws is None:
                    return ""
                ready = getattr(ha_ws, "ready", None)
                if not (callable(getattr(ready, "is_set", None)) and ready.is_set()):
                    return ""
                for entity_id in entity_ids:
                    state_body = getattr(ha_ws, "states", {}).get(entity_id)
                    if not isinstance(state_body, Mapping):
                        continue
                    cached_entity_found = True
                    summary = await _summarise_state_body(state_body)
                    if summary:
                        return summary
                return ""

            async def _summary_from_rest() -> str:
                for entity_id in entity_ids:
                    url = f"{self._base_url}/api/states/{entity_id}"
                    try:
                        if self._client is not None:
                            state_response = await self._client.get(
                                url,
                                headers=headers,
                                timeout=self._timeout,
                            )
                        else:
                            async with httpx.AsyncClient() as client:
                                state_response = await client.get(
                                    url,
                                    headers=headers,
                                    timeout=self._timeout,
                                )
                        if not state_response.is_success:
                            continue
                        state_body = state_response.json()
                    except (httpx.HTTPError, ValueError):
                        continue
                    if isinstance(state_body, Mapping):
                        summary = await _summarise_state_body(state_body)
                        if summary:
                            return summary
                return ""

            async def _delegate_to_voice_agents() -> str:
                # Try fallback agent first
                if runtime.fallback_agent is not None:
                    request = (
                        f"Resolve this Home Assistant intent result into a final spoken answer. "
                        f"The intent name is {call.intent_name}. "
                        f"The user request is: {user_text}. "
                        f"Entity IDs: {', '.join(entity_ids) if entity_ids else 'none'}. "
                        f"Home Assistant response: {json.dumps(body, ensure_ascii=False)}. "
                        f"Return only the spoken answer, no JSON, no tool trace, and no extra explanation."
                    )
                    try:
                        result = await Runner.run(runtime.fallback_agent, request)
                        if result.final_output:
                            return result.final_output.strip()
                    except Exception:
                        log.exception("Fallback agent failed to resolve intent, trying advanced agent")

                # Fallback to advanced agent
                if runtime.advanced_agent is None:
                    return ""

                request = (
                    f"Resolve this Home Assistant intent result into a final spoken answer. "
                    f"The intent name is {call.intent_name}. "
                    f"The user request is: {user_text}. "
                    f"Entity IDs: {', '.join(entity_ids) if entity_ids else 'none'}. "
                    f"Home Assistant response: {json.dumps(body, ensure_ascii=False)}. "
                    f"If you need more details, investigate using the advanced Home Assistant tools. "
                    f"Return only the spoken answer, no JSON, no tool trace, and no extra explanation."
                )
                result = await ha_advanced(request)
                if not isinstance(result, str):
                    return ""

                try:
                    tool_output = json.loads(result)
                except ValueError:
                    return result.strip()

                if isinstance(tool_output, dict) and tool_output.get("success"):
                    answer = tool_output.get("result")
                    if isinstance(answer, str) and answer.strip():
                        return answer.strip()
                return ""

            speech = await _summary_from_websocket()
            if not speech and not cached_entity_found:
                speech = await _summary_from_rest()
            if not speech:
                speech = await _delegate_to_voice_agents()

            if not speech:
                speech = "Sorry, I was not able to resolve that one."

        return ExecutionResult(speech=speech, response=dict(body))


__all__ = ["HomeAssistantIntentExecutor"]
