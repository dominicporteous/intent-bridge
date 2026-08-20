"""Execute recognized OHF calls through Home Assistant's official intent API."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping
from datetime import date, time
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

# ``/api/intent/handle`` returns these values for the caller to render.  The
# Home Assistant conversation agent normally applies the matching response
# template, but that does not happen on the named-intent endpoint used here.
# Keep the set deliberately narrow and aligned with the upstream Core
# handlers. Unknown custom intents retain the existing recovery path below.
_SPEECH_SLOT_INTENTS = frozenset(
    {
        "HassGetCurrentDate",
        "HassGetCurrentTime",
        "HassCancelAllTimers",
        "HassTimerStatus",
        "HassMediaSearchAndPlay",
        "HassShoppingListCompleteItem",
    }
)
_UNKNOWN_SPEECH_SLOTS_POLICY = "state_summary_then_llm_fallback"


def _speech_slots_from_response(body: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Return structured response-template values, if Home Assistant supplied them."""

    speech_slots = body.get("speech_slots")
    return speech_slots if isinstance(speech_slots, Mapping) else None


def _spoken_text(value: object, *, limit: int = 96) -> str:
    """Normalise a slot value before including it in a spoken response."""

    if not isinstance(value, str):
        return ""
    text = " ".join(value.split())
    return text[:limit].rstrip()


def _whole_number(value: object) -> int | None:
    """Parse a non-negative integer without accepting booleans or fractions."""

    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _join_spoken_items(items: list[str]) -> str:
    """Join a short list naturally and keep TTS responses bounded."""

    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def _format_duration(hours: int, minutes: int, seconds: int) -> str:
    parts: list[str] = []
    for value, unit in ((hours, "hour"), (minutes, "minute"), (seconds, "second")):
        if value:
            parts.append(f"{value} {unit}{'' if value == 1 else 's'}")
    return _join_spoken_items(parts) if parts else "0 seconds"


def _timer_duration(status: Mapping[str, Any]) -> str | None:
    """Format the lower-precision timer feedback supplied by Home Assistant."""

    rounded = tuple(
        _whole_number(status.get(key))
        for key in (
            "rounded_hours_left",
            "rounded_minutes_left",
            "rounded_seconds_left",
        )
    )
    if all(value is not None for value in rounded):
        hours, minutes, seconds = rounded
        assert hours is not None and minutes is not None and seconds is not None
        return _format_duration(hours, minutes, seconds)

    total_seconds = _whole_number(status.get("total_seconds_left"))
    if total_seconds is None:
        return None
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return _format_duration(hours, minutes, seconds)


def _render_timer_status(speech_slots: Mapping[str, Any]) -> tuple[str, str | None]:
    statuses = speech_slots.get("timers")
    if not isinstance(statuses, list):
        return "I couldn't read the timer status.", "timers was not a list"
    if not statuses:
        return "There are no active timers.", None

    summaries: list[str] = []
    malformed = False
    for status in statuses[:3]:
        if not isinstance(status, Mapping):
            malformed = True
            continue
        duration = _timer_duration(status)
        if duration is None:
            malformed = True
            continue
        name = _spoken_text(status.get("name"))
        subject = name or "The timer"
        if duration == "0 seconds":
            summaries.append(f"{subject} has finished.")
        elif status.get("is_active") is False:
            summaries.append(f"{subject} is paused with {duration} remaining.")
        else:
            summaries.append(f"{subject} has {duration} remaining.")

    if not summaries:
        return "I couldn't read the timer status.", "timer entries were malformed"
    if len(statuses) > 3:
        summaries.append(f"and {len(statuses) - 3} more")
    reason = "some timer entries were malformed" if malformed else None
    return " ".join(summaries), reason


def _render_shopping_completion(speech_slots: Mapping[str, Any]) -> tuple[str, str | None]:
    completed_items = speech_slots.get("completed_items")
    if not isinstance(completed_items, list):
        return "I couldn't read the shopping-list update.", "completed_items was not a list"
    if not completed_items:
        return "I couldn't find that item on the shopping list.", None

    item_names: list[str] = []
    for item in completed_items[:3]:
        if isinstance(item, Mapping):
            name = _spoken_text(item.get("name"))
        else:
            name = _spoken_text(item)
        if name:
            item_names.append(name)
    if not item_names:
        return "I completed the shopping-list item.", "completed item names were missing"
    if len(completed_items) > 3:
        item_names.append(f"{len(completed_items) - 3} more")
    return f"Completed {_join_spoken_items(item_names)} on the shopping list.", None


def _render_speech_slots(
    intent_name: str,
    speech_slots: Mapping[str, Any],
    call_data: Mapping[str, Any],
) -> tuple[str | None, str | None]:
    """Render the Core named-intent slot responses without involving an LLM.

    Returns ``(speech, reason)``. ``speech`` is ``None`` only when the intent
    has no registered deterministic renderer. A non-empty reason records a
    deterministic recovery from malformed data.
    """

    if intent_name not in _SPEECH_SLOT_INTENTS:
        return None, None

    if intent_name == "HassGetCurrentTime":
        value = _spoken_text(speech_slots.get("time"))
        try:
            current_time = time.fromisoformat(value)
        except ValueError:
            return "I couldn't read the current time.", "time was not ISO formatted"
        hour = current_time.hour % 12 or 12
        period = "AM" if current_time.hour < 12 else "PM"
        return f"The time is {hour}:{current_time.minute:02d} {period}.", None

    if intent_name == "HassGetCurrentDate":
        value = _spoken_text(speech_slots.get("date"))
        try:
            current_date = date.fromisoformat(value)
        except ValueError:
            return "I couldn't read the current date.", "date was not ISO formatted"
        weekday = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
        month = (
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        )
        return (
            f"Today is {weekday[current_date.weekday()]}, {current_date.day} "
            f"{month[current_date.month - 1]} {current_date.year}.",
            None,
        )

    if intent_name == "HassCancelAllTimers":
        canceled = _whole_number(speech_slots.get("canceled"))
        if canceled is None:
            return "I couldn't read how many timers were cancelled.", "canceled was not an integer"
        if canceled == 0:
            return "There were no timers to cancel.", None
        area = _spoken_text(speech_slots.get("area"), limit=64)
        suffix = f" in {area}" if area else ""
        return f"Cancelled {canceled} timer{'' if canceled == 1 else 's'}{suffix}.", None

    if intent_name == "HassTimerStatus":
        return _render_timer_status(speech_slots)

    if intent_name == "HassShoppingListCompleteItem":
        return _render_shopping_completion(speech_slots)

    # HassMediaSearchAndPlay returns a BrowseMedia dictionary. The title is
    # normally always present, but preserve an action confirmation if a media
    # integration provides an incomplete result.
    media = speech_slots.get("media")
    if not isinstance(media, Mapping):
        return "Playing the selected media.", "media was not a mapping"
    title = _spoken_text(media.get("title")) or _spoken_text(call_data.get("search_query"))
    if not title:
        return "Playing the selected media.", "media title was missing"
    return f"Playing {title}.", None


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


def _intent_response_body(body: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the common intent-result shape from REST or conversation WS."""

    nested_response = body.get("response")
    return nested_response if isinstance(nested_response, Mapping) else body


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
        websocket_provider: Callable[[], Any | None] | None = None,
    ) -> None:
        self._configured = bool(base_url.strip() and access_token.strip())
        self._base_url = base_url.rstrip("/")
        self._url = f"{base_url.rstrip('/')}/api/intent/handle"
        self._access_token = access_token
        self._timeout = timeout
        self._client = client
        self._websocket_provider = websocket_provider

    async def _execute_via_conversation_websocket(
        self,
        call: OhfIntentCall,
    ) -> Mapping[str, Any] | None:
        """Use HA's persistent conversation WebSocket when it is safe to do so.

        Home Assistant exposes named intent handling only through its REST API.
        Its WebSocket API accepts the source utterance via conversation/process.
        We use that path for a single, ordinary deterministic command, but keep
        the exact HTTP intent path for plans that cannot safely re-submit their
        full utterance (compound plans and exact-target materialisation).
        """

        if self._websocket_provider is None:
            return None
        if not call.intent_name.startswith("Hass") or call.data.get("entity_id"):
            return None

        state = voice_tool_run_state
        source_text = state.request_text.strip()
        if not state.allow_conversation_websocket or not source_text:
            return None

        # An area inferred by this bridge cannot be represented by HA's
        # conversation WebSocket unless the originating device is also known.
        # Falling back preserves the explicit area on the resolved intent.
        device_id = state.origin_device_id
        if (state.origin_area_id or state.origin_area_name) and not device_id:
            return None

        ha_ws = self._websocket_provider()
        ready = getattr(ha_ws, "ready", None)
        if not (callable(getattr(ready, "is_set", None)) and ready.is_set()):
            return None

        processor = getattr(ha_ws, "process_conversation", None)
        if not callable(processor):
            return None

        try:
            reply = await processor(
                source_text,
                language=settings.deterministic.language,
                device_id=device_id,
                timeout=self._timeout,
            )
        except Exception:
            # The command may have reached HA before a transport failure. Do
            # not retry through HTTP here: doing so could execute an action
            # twice. The normal WebSocket command retry handles a clean
            # reconnect before this point where possible.
            log.exception(
                "HA WebSocket conversation dispatch failed; not retrying HTTP "
                "to avoid a duplicate action intent=%s",
                call.intent_name,
            )
            raise

        if not isinstance(reply, Mapping):
            log.warning(
                "HA WebSocket conversation returned an invalid reply; falling back to HTTP "
                "intent=%s",
                call.intent_name,
            )
            return None
        if reply.get("success") is not True:
            log.warning(
                "HA WebSocket conversation was rejected; falling back to HTTP "
                "intent=%s error=%s",
                call.intent_name,
                reply.get("error"),
            )
            return None

        body = reply.get("result")
        if not isinstance(body, Mapping):
            log.warning(
                "HA WebSocket conversation returned no result; falling back to HTTP "
                "intent=%s",
                call.intent_name,
            )
            return None
        if _intent_response_body(body).get("response_type") == "error":
            log.info(
                "HA WebSocket conversation could not handle intent; falling back to HTTP "
                "intent=%s",
                call.intent_name,
            )
            return None

        log.info(
            "Executed Home Assistant intent over persistent WebSocket "
            "intent=%s data=%s",
            call.intent_name,
            dict(call.data),
        )
        return body

    async def _execute_via_http(
        self,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> Mapping[str, Any]:
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
                f"Home Assistant intent {payload['name']} failed: {_error_detail(response)}"
            )

        try:
            body = response.json()
        except ValueError as exc:
            raise RuntimeError("Home Assistant returned a non-JSON intent response") from exc
        if not isinstance(body, Mapping):
            raise RuntimeError("Home Assistant returned an invalid intent response")
        return body

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

        body = await self._execute_via_conversation_websocket(call)
        if body is None:
            body = await self._execute_via_http(headers, payload)
            log.info(
                "Home Assistant intent response transport=http intent=%s body=%s",
                call.intent_name,
                body,
            )
        else:
            log.info(
                "Home Assistant intent response transport=websocket intent=%s body=%s",
                call.intent_name,
                body,
            )
        body = _intent_response_body(body)

        speech = _speech_from_response(body)
        speech_slots = _speech_slots_from_response(body) if not speech else None
        unrendered_speech_slots = False
        if speech_slots:
            rendered_speech, recovery_reason = _render_speech_slots(
                call.intent_name,
                speech_slots,
                call.data,
            )
            if rendered_speech:
                speech = rendered_speech
                if recovery_reason:
                    log.warning(
                        "Home Assistant speech_slots rendered with deterministic recovery "
                        "intent=%s keys=%s reason=%s",
                        call.intent_name,
                        sorted(str(key) for key in speech_slots),
                        recovery_reason,
                    )
                else:
                    log.info(
                        "Home Assistant speech_slots rendered deterministically intent=%s keys=%s",
                        call.intent_name,
                        sorted(str(key) for key in speech_slots),
                    )
            else:
                unrendered_speech_slots = True
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
                if unrendered_speech_slots:
                    log.warning(
                        "Home Assistant speech_slots have no deterministic renderer "
                        "intent=%s keys=%s policy=%s",
                        call.intent_name,
                        sorted(str(key) for key in speech_slots or {}),
                        _UNKNOWN_SPEECH_SLOTS_POLICY,
                    )
                speech = await _delegate_to_voice_agents()

            if not speech:
                speech = "Sorry, I was not able to resolve that one."

        return ExecutionResult(speech=speech, response=dict(body))


__all__ = ["HomeAssistantIntentExecutor"]
