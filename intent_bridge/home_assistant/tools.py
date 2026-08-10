"""Agent-facing Home Assistant tools."""

from typing import Any, Literal

from agents import (
    function_tool,
)

from intent_bridge.config import log, settings
from intent_bridge.core.text import normalize_search_text as _normalise_search_text
from intent_bridge.home_assistant.client import (
    _require_ha_ws,
)
from intent_bridge.home_assistant.policy import EXPECTED_PRIMARY_STATES
from intent_bridge.runtime.execution import (
    _compact_attributes,
    _compact_service_definition,
    _get_cached_service_definition,
    _json_tool_result,
    _normalise_service_data_from_schema,
    _single_cached_entity_for_domain,
    _target_entity_domains,
    _truncate_text,
    voice_tool_run_state,
)

# ---------------------------------------------------------------------------
# Direct fast Home Assistant tools exposed to the normal voice agent
# ---------------------------------------------------------------------------


@function_tool
async def ha_search(
    query: str,
    domain_filter: str | None = None,
    area_filter: str | None = None,
    limit: int = settings.home_assistant.search_default_limit,
) -> str:
    """Search cached Home Assistant entities by name, entity ID, device or area.

    Args:
        query: Natural-language entity/device name to find, such as "office light".
        domain_filter: Optional exact entity domain such as light, switch, climate, weather.
        area_filter: Optional explicit Home Assistant area name such as Office or Kitchen.
        limit: Maximum matches to return.
    """
    try:
        client = await _require_ha_ws()
        await client.refresh_registries()
        bounded_limit = min(max(1, int(limit)), settings.home_assistant.search_max_limit)

        effective_area_filter = area_filter
        preferred_area_filter: str | None = None
        area_context_source = "explicit" if area_filter else None

        # Explicit room names override the calling satellite room.
        if not effective_area_filter:
            mentioned = client.area_mentioned_in_text(query)
            if mentioned is not None:
                _, mentioned_name = mentioned
                effective_area_filter = mentioned_name
                area_context_source = "query"

        # The voice-origin room is only a preference. Do not hide room-named
        # entities which lack an explicit HA area assignment.
        if (
            not effective_area_filter
            and settings.voice_origin.enabled
            and settings.voice_origin.area_bias_enabled
            and (voice_tool_run_state.origin_area_name or voice_tool_run_state.origin_area_id)
        ):
            origin_area = (
                voice_tool_run_state.origin_area_name or voice_tool_run_state.origin_area_id
            )
            if settings.voice_origin.soft_ranking_enabled:
                preferred_area_filter = origin_area
                area_context_source = "voice_origin_soft"
            else:
                effective_area_filter = origin_area
                area_context_source = "voice_origin"

        results = client.search_cached_states(
            query,
            domain_filter=domain_filter,
            area_filter=effective_area_filter,
            preferred_area_filter=preferred_area_filter,
            limit=bounded_limit,
        )

        # Backward-compatible fallback if hard origin mode is manually enabled.
        if not results and area_context_source == "voice_origin":
            results = client.search_cached_states(
                query,
                domain_filter=domain_filter,
                area_filter=None,
                preferred_area_filter=None,
                limit=bounded_limit,
            )
            area_context_source = "voice_origin_global_fallback"

        if results:
            top_preview = [
                {
                    "entity_id": item.get("entity_id"),
                    "score": item.get("match_score"),
                    "reasons": item.get("match_reasons"),
                }
                for item in results[:3]
            ]
            log.info(
                "HA SEARCH query=%r domain=%r hard_area=%r preferred_area=%r top=%s",
                query,
                domain_filter,
                effective_area_filter,
                preferred_area_filter,
                top_preview,
            )

        if len(results) == 1:
            entity_id = results[0].get("entity_id")
            domain = results[0].get("domain")
            if isinstance(entity_id, str) and isinstance(domain, str):
                voice_tool_run_state.last_entity_by_domain[domain] = entity_id
            area_id = results[0].get("area_id")
            if isinstance(area_id, str) and area_id:
                voice_tool_run_state.last_area_id = area_id

        return _json_tool_result(
            {
                "success": True,
                "query": query,
                "domain_filter": domain_filter,
                "area_filter": effective_area_filter,
                "preferred_area": preferred_area_filter,
                "area_context_source": area_context_source,
                "origin_area": voice_tool_run_state.origin_area_name,
                "count": len(results),
                "recommended_entity_id": (results[0].get("entity_id") if results else None),
                "results": results,
                "source": "local_state_cache",
            }
        )
    except Exception as exc:
        log.warning("DIRECT TOOL ha_search failed: %s", exc)
        return _json_tool_result({"success": False, "error": str(exc)})


@function_tool
async def ha_get_state(
    entity_id: str,
    attribute_keys: list[str] | None = None,
) -> str:
    """Read one Home Assistant entity from the local state cache.

    Args:
        entity_id: Exact Home Assistant entity ID previously found with ha_search.
        attribute_keys: Optional attribute names to return. Omit for compact useful attributes.
    """
    try:
        client = await _require_ha_ws()
        state = client.states.get(entity_id)
        if not isinstance(state, dict):
            return _json_tool_result(
                {
                    "success": False,
                    "error": f"Entity not found in current state cache: {entity_id}",
                    "suggestion": "Use ha_search to resolve the entity ID.",
                }
            )

        domain = entity_id.split(".", 1)[0]
        voice_tool_run_state.last_entity_by_domain[domain] = entity_id
        record_read = getattr(client, "record_state_read", None)
        if callable(record_read):
            record_read(entity_id)

        attributes = state.get("attributes", {})
        if not isinstance(attributes, dict):
            attributes = {}

        if attribute_keys is not None:
            selected_attributes = {
                key: attributes.get(key) for key in attribute_keys if key in attributes
            }
        else:
            selected_attributes = _compact_attributes(attributes)

        return _json_tool_result(
            {
                "success": True,
                "entity_id": entity_id,
                "state": state.get("state"),
                "attributes": selected_attributes,
                "last_changed": state.get("last_changed"),
                "last_updated": state.get("last_updated"),
                "source": "local_state_cache",
            }
        )
    except Exception as exc:
        log.warning("DIRECT TOOL ha_get_state failed: %s", exc)
        return _json_tool_result({"success": False, "error": str(exc)})


@function_tool
async def ha_list_services(
    domain: str | None = None,
    query: str | None = None,
    detail_level: Literal["summary", "full"] = "summary",
    limit: int = 50,
) -> str:
    """Discover actions from the cached Home Assistant service catalogue.

    Full detail returns a compact CALL schema rather than Home Assistant's
    UI-oriented raw schema. Exact keys under "parameters" are the keys that must
    be placed in ha_call_service(data=...).

    Args:
        domain: Optional exact service domain such as weather, light, calendar.
        query: Optional service-name/description search text.
        detail_level: summary or full. Full includes exact call parameters.
        limit: Maximum matching services to return.
    """
    try:
        client = await _require_ha_ws()
        await client.refresh_services()
        domain_norm = domain.strip().casefold() if isinstance(domain, str) else None
        query_norm = _normalise_search_text(query) if query else ""
        bounded_limit = min(max(1, int(limit)), 200)

        matches: list[dict[str, Any]] = []
        for service_domain, domain_services in client.services.items():
            if domain_norm and service_domain.casefold() != domain_norm:
                continue
            if not isinstance(domain_services, dict):
                continue

            for service_name, definition in domain_services.items():
                if not isinstance(definition, dict):
                    definition = {}
                searchable = _normalise_search_text(
                    f"{service_domain} {service_name} "
                    f"{definition.get('name', '')} {definition.get('description', '')}"
                )
                if query_norm and query_norm not in searchable:
                    continue

                if detail_level == "full":
                    record = _compact_service_definition(
                        service_domain,
                        service_name,
                        definition,
                    )
                else:
                    record = {
                        "domain": service_domain,
                        "service": service_name,
                        "name": definition.get("name"),
                        "description": _truncate_text(definition.get("description"), 220),
                    }

                matches.append(record)
                if len(matches) >= bounded_limit:
                    break
            if len(matches) >= bounded_limit:
                break

        return _json_tool_result(
            {
                "success": True,
                "domain": domain,
                "query": query,
                "detail_level": detail_level,
                "count": len(matches),
                "services": matches,
                "schema_note": (
                    "For full detail, use exact keys under parameters as data keys. "
                    "Do not derive parameter names from human-readable labels."
                    if detail_level == "full"
                    else None
                ),
                "source": "cached_websocket_service_catalogue",
            }
        )
    except Exception as exc:
        log.warning("DIRECT TOOL ha_list_services failed: %s", exc)
        return _json_tool_result({"success": False, "error": str(exc)})


# This tool accepts arbitrary Home Assistant service_data keys.
# OpenAI Agents SDK strict JSON schemas reject free-form dict objects,
# so this one tool must use a non-strict schema.
@function_tool(strict_mode=False)
async def ha_call_service(
    domain: str,
    service: str,
    entity_id: str | None = None,
    area_id: str | None = None,
    data: dict[str, Any] | None = None,
    return_response: bool = False,
) -> str:
    """Call a Home Assistant action over the persistent WebSocket.

    Args:
        domain: Exact service domain such as light, climate, weather, calendar.
        service: Exact service/action name such as turn_off or get_forecasts.
        entity_id: Optional exact target entity ID.
        area_id: Optional exact Home Assistant area ID for an area-wide action.
        data: Service-specific fields using EXACT keys from ha_list_services.
        return_response: True for actions that return information/data.
    """
    try:
        client = await _require_ha_ws()
        await client.refresh_services()

        domain = str(domain or "").strip()
        service = str(service or "").strip()
        if not domain or not service:
            return _json_tool_result(
                {
                    "success": False,
                    "error_type": "local_schema_validation",
                    "error": "domain and service are required",
                }
            )

        definition = _get_cached_service_definition(client, domain, service)

        # Natural speech often describes a vacuum like a switch, but HA's
        # vacuum domain uses start rather than turn_on. Repair that harmless
        # semantic alias locally so an agent cannot issue a known-bad service.
        if domain == "vacuum" and service == "turn_on" and definition is None:
            start_definition = _get_cached_service_definition(client, domain, "start")
            if start_definition is not None:
                log.info("HA SERVICE LOCAL REPAIR vacuum.turn_on -> vacuum.start")
                service = "start"
                definition = start_definition

        previous = voice_tool_run_state.last_service_call
        same_previous_service = (
            isinstance(previous, dict)
            and previous.get("domain") == domain
            and previous.get("service") == service
        )

        # Preserve a target already resolved in this SAME request while the model
        # repairs service_data. This prevents retries from accidentally dropping
        # weather.forecast_home, a light, calendar, etc.
        if not entity_id and not area_id and same_previous_service:
            previous_entity = previous.get("entity_id")
            previous_area = previous.get("area_id")
            if isinstance(previous_entity, str) and previous_entity:
                entity_id = previous_entity
                log.info(
                    "HA SERVICE RETRY preserved previous entity target=%s",
                    entity_id,
                )
            elif isinstance(previous_area, str) and previous_area:
                area_id = previous_area
                log.info(
                    "HA SERVICE RETRY preserved previous area target=%s",
                    area_id,
                )

        if (
            not entity_id
            and not area_id
            and settings.voice_origin.enabled
            and settings.voice_origin.area_bias_enabled
            and voice_tool_run_state.origin_area_id
        ):
            origin_candidates = client.entities_in_area(domain, voice_tool_run_state.origin_area_id)
            if len(origin_candidates) == 1:
                entity_id = origin_candidates[0]
                log.info(
                    "HA SERVICE TARGET auto-selected sole origin-area entity domain=%s area=%s entity=%s",
                    domain,
                    voice_tool_run_state.origin_area_name or voice_tool_run_state.origin_area_id,
                    entity_id,
                )

        if not entity_id and not area_id:
            remembered = voice_tool_run_state.last_entity_by_domain.get(domain)
            if isinstance(remembered, str) and remembered:
                entity_id = remembered
                log.info(
                    "HA SERVICE TARGET reused unambiguous run target domain=%s entity=%s",
                    domain,
                    entity_id,
                )

        # For read/data actions only, selecting the sole entity of the required
        # target domain is deterministic and safe. Never do this for writes.
        if not entity_id and not area_id and return_response and definition:
            target_domains = _target_entity_domains(definition.get("target"))
            candidate_domains = target_domains or [domain]
            candidates = [
                _single_cached_entity_for_domain(client, target_domain)
                for target_domain in candidate_domains
            ]
            candidates = [candidate for candidate in candidates if candidate]
            unique_candidates = list(dict.fromkeys(candidates))
            if len(unique_candidates) == 1:
                entity_id = unique_candidates[0]
                log.info(
                    "HA DATA SERVICE TARGET auto-selected sole entity=%s",
                    entity_id,
                )

        previous_data = (
            previous.get("data")
            if same_previous_service and isinstance(previous.get("data"), dict)
            else None
        )
        normalised_data, repairs, validation = _normalise_service_data_from_schema(
            domain,
            service,
            definition,
            data,
            previous_data,
        )

        # Preserve return_response=True on a repair retry. The default False in
        # a fresh model tool call otherwise makes it easy to lose returned data.
        if (
            same_previous_service
            and previous.get("return_response") is True
            and return_response is False
        ):
            return_response = True
            repairs.append("preserved return_response=True from previous retry")

        voice_tool_run_state.last_service_call = {
            "domain": domain,
            "service": service,
            "entity_id": entity_id,
            "area_id": area_id,
            "data": dict(normalised_data),
            "return_response": bool(return_response),
        }

        if repairs:
            log.info(
                "HA SERVICE LOCAL REPAIR domain=%s service=%s repairs=%s",
                domain,
                service,
                repairs,
            )

        if validation is not None:
            log.warning(
                "HA SERVICE LOCAL VALIDATION FAILED domain=%s service=%s validation=%s",
                domain,
                service,
                {
                    "unknown": validation.get("unknown_parameters"),
                    "missing": validation.get("missing_required_parameters"),
                    "invalid_values": validation.get("invalid_values"),
                },
            )
            return _json_tool_result(
                {
                    "success": False,
                    "domain": domain,
                    "service": service,
                    "entity_id": entity_id,
                    "area_id": area_id,
                    "error_type": "local_schema_validation",
                    "error": "Service arguments do not match the cached Home Assistant schema.",
                    **validation,
                    "repairs_applied": repairs,
                    "retry_instruction": (
                        "Change only the invalid service-data fields. Preserve the "
                        "domain, service, target, return_response, and every already-valid field."
                    ),
                }
            )

        payload: dict[str, Any] = {
            "type": "call_service",
            "domain": domain,
            "service": service,
            "service_data": normalised_data,
            "return_response": bool(return_response),
        }

        target: dict[str, Any] = {}
        if entity_id:
            if isinstance(entity_id, str):
                entity_ids = [item.strip() for item in entity_id.split(",") if item.strip()]
            elif isinstance(entity_id, list):
                entity_ids = [str(item).strip() for item in entity_id if str(item).strip()]
            else:
                entity_ids = [str(entity_id).strip()]
            if entity_ids:
                target["entity_id"] = entity_ids[0] if len(entity_ids) == 1 else entity_ids
                entity_id = ",".join(entity_ids)

        if area_id:
            target["area_id"] = area_id
        if target:
            payload["target"] = target

        reply = await client.command(
            payload, timeout=settings.home_assistant.websocket.command_timeout_seconds
        )
        if reply.get("success") is not True:
            error = reply.get("error")
            message = error.get("message") if isinstance(error, dict) else str(error)
            code = error.get("code") if isinstance(error, dict) else None
            log.warning(
                "HA WS SERVICE FAILED domain=%s service=%s entity=%s code=%s error=%s",
                domain,
                service,
                entity_id,
                code,
                message,
            )
            compact_schema = (
                _compact_service_definition(domain, service, definition)
                if isinstance(definition, dict)
                else None
            )
            return _json_tool_result(
                {
                    "success": False,
                    "domain": domain,
                    "service": service,
                    "entity_id": entity_id,
                    "area_id": area_id,
                    "data": normalised_data,
                    "return_response": bool(return_response),
                    "error": message or "Home Assistant service call failed",
                    "error_code": code,
                    "service_schema": compact_schema,
                    "retry_instruction": (
                        "Do not repeat this call unchanged. Preserve its valid target "
                        "and valid fields; change only what the error identifies."
                    ),
                }
            )

        result = reply.get("result")
        if not isinstance(result, dict):
            result = {}

        response: dict[str, Any] = {
            "success": True,
            "domain": domain,
            "service": service,
            "entity_id": entity_id,
            "area_id": area_id,
            "repairs_applied": repairs,
            "source": "home_assistant_websocket",
        }

        if return_response:
            response["service_response"] = result.get("response")
            voice_tool_run_state.last_successful_data = {
                "domain": domain,
                "service": service,
                "entity_id": entity_id,
                "area_id": area_id,
                "service_response": result.get("response"),
            }

        expected_state = EXPECTED_PRIMARY_STATES.get(service)
        if entity_id and expected_state and "," not in entity_id:
            verified = await client.wait_for_expected_state(
                entity_id,
                expected_state,
                settings.home_assistant.websocket.state_confirm_timeout_seconds,
            )
            if verified is not None:
                response["verified_state"] = verified

        if entity_id and "," not in entity_id:
            voice_tool_run_state.last_entity_by_domain[domain] = entity_id

        log.info(
            "HA WS SERVICE COMPLETE domain=%s service=%s entity=%s return_response=%s repairs=%d",
            domain,
            service,
            entity_id,
            return_response,
            len(repairs),
        )
        return _json_tool_result(response)
    except Exception as exc:
        log.warning(
            "DIRECT TOOL ha_call_service failed domain=%s service=%s: %s",
            domain,
            service,
            exc,
        )
        return _json_tool_result(
            {
                "success": False,
                "domain": domain,
                "service": service,
                "entity_id": entity_id,
                "area_id": area_id,
                "error": str(exc),
            }
        )


# ---------------------------------------------------------------------------
# Optional advanced ha-mcp specialist
# ---------------------------------------------------------------------------


ADVANCED_INSTRUCTIONS = """
You are the advanced Home Assistant specialist behind a spoken voice assistant.

The normal agent has already decided this request needs capabilities beyond its
fast cached state/search/service tools. Use the available ha-mcp tools to perform
or investigate the requested Home Assistant operation.

Use tool search when available rather than guessing obscure tool names. Never
invent entity IDs or claim success without tool confirmation. Minimize tool calls.
Avoid destructive administration unless the user's request clearly requires it.
Return a concise factual result to the parent agent. Do not add offers for more
help, troubleshooting advice, Markdown, URLs, or long explanations unless the
request itself explicitly asks for technical detail.

If you are asked to provide information (such as weather or sensor values) and the
tools provide raw state data, format it into a natural, conversational, human-friendly
sentence. Never simply state the raw state value (e.g., instead of "the sensor
is currently reporting a state of 1", report "it is raining" if the value 1 corresponds
to rain, or similar human-readable interpretations).
""".strip()
