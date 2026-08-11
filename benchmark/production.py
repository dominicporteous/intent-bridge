"""Translation helpers for observable production-pipeline effects."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from benchmark.models import (
    BenchmarkRequest,
    Home,
    Operation,
)
from intent_bridge.intent_engine.models import (
    CatalogArea,
    CatalogEntity,
    CatalogFloor,
    CatalogSnapshot,
    PlannedIntent,
)

_TARGET_KEYS = frozenset({"area", "domain", "floor", "name"})
_DURATION_KEYS = ("days", "hours", "minutes", "seconds", "milliseconds")


def _setup_by_entity(request: BenchmarkRequest) -> dict[str, Operation]:
    return {
        entity_id: operation for operation in request.setup for entity_id in operation.entity_ids
    }


def _planning_context(request: BenchmarkRequest) -> dict[str, object]:
    """Expose only runtime state that a real planner could obtain from HA."""

    context: dict[str, object] = dict(request.origin_context)
    active_timer_ids: list[str] = []
    shopping_items: list[str] = []
    list_items: dict[str, list[str]] = {}
    for operation in request.setup:
        active_timer_ids.extend(
            entity_id for entity_id in operation.entity_ids if entity_id.startswith("timer.")
        )
        if item := operation.payload.get("shopping_list_item"):
            shopping_items.append(str(item))
        if item := operation.payload.get("todo_item"):
            list_name = str(operation.payload.get("list_name") or "")
            if list_name:
                list_items.setdefault(list_name, []).append(str(item))
    if active_timer_ids:
        context["active_timer_ids"] = tuple(dict.fromkeys(active_timer_ids))
    if shopping_items:
        context["shopping_list_items"] = tuple(shopping_items)
    if list_items:
        context["list_items"] = {name: tuple(items) for name, items in list_items.items()}
    return context


def _catalog(request: BenchmarkRequest) -> CatalogSnapshot:
    setup = _setup_by_entity(request)
    entities: list[CatalogEntity] = []
    for entity in request.home.entities:
        mutation = setup.get(entity.entity_id)
        state = mutation.state if mutation and mutation.state is not None else entity.state
        entities.append(
            CatalogEntity(
                entity_id=entity.entity_id,
                name=entity.name,
                aliases=entity.aliases,
                domain=entity.domain,
                area_id=entity.area_id,
                device_class=(
                    str(entity.attributes["device_class"])
                    if entity.attributes.get("device_class") is not None
                    else None
                ),
                state=state,
            )
        )
    known_entity_ids = {entity.entity_id for entity in request.home.entities}
    for operation in request.setup:
        for entity_id in operation.entity_ids:
            if entity_id in known_entity_ids or "." not in entity_id:
                continue
            domain, object_id = entity_id.split(".", 1)
            name = object_id.replace("_", " ").title()
            if domain == "timer" and not name.casefold().endswith(" timer"):
                name = f"{name} Timer"
            entities.append(
                CatalogEntity(
                    entity_id=entity_id,
                    name=name,
                    aliases=(object_id.replace("_", " "),),
                    domain=domain,
                    state=operation.state,
                )
            )
            known_entity_ids.add(entity_id)
    return CatalogSnapshot(
        entities=tuple(entities),
        areas=tuple(
            CatalogArea(
                area_id=area.area_id,
                name=area.name,
                aliases=area.aliases,
                floor_id=area.floor_id,
            )
            for area in request.home.areas
        ),
        floors=tuple(
            CatalogFloor(
                floor_id=floor.floor_id,
                name=floor.name,
                aliases=floor.aliases,
            )
            for floor in request.home.floors
        ),
    )


def _dedupe(operations: list[Operation]) -> tuple[Operation, ...]:
    result: list[Operation] = []
    seen: set[tuple[Any, ...]] = set()
    for operation in operations:
        key = operation.semantic_key()
        if key not in seen:
            seen.add(key)
            result.append(operation)
    return tuple(result)


def _state_for_activation(intent_name: str, domain: str) -> str | None:
    enabled = intent_name == "HassTurnOn"
    if domain == "cover":
        return "open" if enabled else "closed"
    if domain == "lock":
        return "locked" if enabled else "unlocked"
    if domain in {"scene", "script"}:
        return None
    return "on" if enabled else "off"


def _raw_collection(home: Home, name: str) -> tuple[Mapping[str, Any], ...]:
    value = home.metadata.get(name)
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _state_operation(
    entity_ids: tuple[str, ...],
    state: str | None,
    *,
    payload: Mapping[str, Any] | None = None,
) -> Operation:
    return Operation(
        kind="action",
        entity_ids=tuple(sorted(entity_ids)),
        state=state,
        payload=dict(payload or {}),
    )


def _expand_scene(home: Home, entity_id: str) -> tuple[Operation, ...]:
    scene_id = entity_id.split(".", 1)[-1]
    raw = next(
        (item for item in _raw_collection(home, "scenes") if str(item.get("id")) == scene_id),
        None,
    )
    if raw is None or not isinstance(raw.get("entities"), Mapping):
        return ()
    operations: list[Operation] = []
    for target, value in raw["entities"].items():
        if isinstance(value, Mapping):
            state = value.get("state")
            payload = {key: item for key, item in value.items() if key != "state"}
        else:
            state = value
            payload = {}
        operations.append(
            _state_operation(
                (str(target),),
                str(state) if state is not None else None,
                payload=payload,
            )
        )
    return tuple(operations)


def _service_effect(
    action: str,
    entity_ids: tuple[str, ...],
    data: Mapping[str, Any],
) -> tuple[Operation, ...]:
    domain, _, service = action.partition(".")
    if service in {"turn_on", "turn_off"}:
        state = "on" if service == "turn_on" else "off"
    elif service in {"open_cover", "open"}:
        state = "open"
    elif service in {"close_cover", "close"}:
        state = "closed"
    elif service == "lock":
        state = "locked"
    elif service == "unlock":
        state = "unlocked"
    elif service in {"media_pause", "pause"}:
        state = "paused"
    elif service in {"media_play", "play"}:
        state = "playing"
    else:
        state = None
    if domain == "cover" and service == "turn_on":
        state = "open"
    elif domain == "cover" and service == "turn_off":
        state = "closed"
    elif domain == "lock" and service == "turn_on":
        state = "locked"
    elif domain == "lock" and service == "turn_off":
        state = "unlocked"
    return (_state_operation(entity_ids, state, payload=data),)


def _expand_script(home: Home, entity_id: str) -> tuple[Operation, ...]:
    script_id = entity_id.split(".", 1)[-1]
    raw = next(
        (item for item in _raw_collection(home, "scripts") if str(item.get("id")) == script_id),
        None,
    )
    if raw is None or not isinstance(raw.get("actions"), list):
        return ()
    operations: list[Operation] = []
    for action_item in raw["actions"]:
        if not isinstance(action_item, Mapping):
            continue
        target = action_item.get("target")
        raw_entity_ids = target.get("entity_id") if isinstance(target, Mapping) else None
        if raw_entity_ids is None:
            raw_entity_ids = action_item.get("entity_id")
        if isinstance(raw_entity_ids, str):
            entity_ids = (raw_entity_ids,)
        elif isinstance(raw_entity_ids, list):
            entity_ids = tuple(str(item) for item in raw_entity_ids)
        else:
            entity_ids = ()
        data = action_item.get("data")
        operations.extend(
            _service_effect(
                str(action_item.get("action") or action_item.get("service") or ""),
                entity_ids,
                dict(data) if isinstance(data, Mapping) else {},
            )
        )
    return tuple(operations)


def expand_static_invocation(
    home: Home,
    entity_id: str,
    *,
    _visited: frozenset[str] = frozenset(),
) -> tuple[Operation, ...]:
    """Return the statically configured effects of a scene or script invocation.

    Home Assistant performs these effects itself in production.  Fixture-backed
    execution has no HA automation engine, so the benchmark must interpret the
    same declarative definitions.  Unknown and cyclic invocations remain
    observable as invocation operations instead of silently disappearing.
    """

    domain = entity_id.partition(".")[0]
    invocation = _state_operation((entity_id,), None, payload={"invoked": True})
    if domain not in {"scene", "script"} or entity_id in _visited:
        return (invocation,)

    visited = _visited | {entity_id}
    if domain == "scene":
        effects = _expand_scene(home, entity_id)
    else:
        effects = _expand_script(home, entity_id)
    if not effects:
        return (invocation,)

    expanded: list[Operation] = []
    for effect in effects:
        nested_ids = tuple(
            nested_id
            for nested_id in effect.entity_ids
            if nested_id.partition(".")[0] in {"scene", "script"}
        )
        ordinary_ids = tuple(
            nested_id for nested_id in effect.entity_ids if nested_id not in nested_ids
        )
        if ordinary_ids:
            expanded.append(
                _state_operation(ordinary_ids, effect.state, payload=effect.payload)
            )
        for nested_id in nested_ids:
            expanded.extend(
                expand_static_invocation(home, nested_id, _visited=visited)
            )
    return tuple(expanded) or (invocation,)


def _resolve_domain(step: PlannedIntent, home: Home) -> str:
    if value := step.call.data.get("domain"):
        return str(value)
    domains = {
        entity.domain
        for entity_id in step.entity_ids
        if (entity := home.entity(entity_id)) is not None
    }
    return next(iter(domains)) if len(domains) == 1 else ""


def _duration_seconds(payload: Mapping[str, Any], *, setup: bool = False) -> float:
    multipliers = {
        "days": 86_400,
        "hours": 3_600,
        "minutes": 60,
        "seconds": 1,
        "milliseconds": 0.001,
    }
    total = 0.0
    for key, multiplier in multipliers.items():
        value = payload.get(key)
        if value is None and setup:
            value = payload.get(f"start_{key}")
        if isinstance(value, (int, float)):
            total += float(value) * multiplier
    return total


def _duration_payload(seconds: float) -> dict[str, int | float]:
    milliseconds = round(max(0.0, seconds) * 1000)
    days, milliseconds = divmod(milliseconds, 86_400_000)
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    whole_seconds, milliseconds = divmod(milliseconds, 1000)
    result: dict[str, int | float] = {}
    for key, value in (("days", days), ("hours", hours), ("minutes", minutes)):
        if value:
            result[key] = value
    if whole_seconds or milliseconds:
        result["seconds"] = whole_seconds + milliseconds / 1000 if milliseconds else whole_seconds
    if not result:
        result["seconds"] = 0
    return result


def _timer_payload(
    intent: str,
    data: Mapping[str, Any],
    entity_ids: tuple[str, ...],
    setup: tuple[Operation, ...],
) -> Mapping[str, Any]:
    delta = {key: data[key] for key in _DURATION_KEYS if key in data}
    if intent not in {"HassIncreaseTimer", "HassDecreaseTimer"}:
        return delta
    baselines = [
        operation for operation in setup if set(operation.entity_ids).intersection(entity_ids)
    ]
    if len(baselines) != 1:
        return delta
    baseline_seconds = _duration_seconds(baselines[0].payload, setup=True)
    delta_seconds = _duration_seconds(delta)
    if intent == "HassDecreaseTimer":
        delta_seconds *= -1
    return _duration_payload(baseline_seconds + delta_seconds)


def _timer_entity_ids(
    intent: str,
    data: Mapping[str, Any],
    entity_ids: tuple[str, ...],
    home: Home,
    setup: tuple[Operation, ...],
) -> tuple[str, ...]:
    if entity_ids:
        return entity_ids
    if intent == "HassCancelAllTimers":
        return tuple(
            entity.entity_id
            for entity in home.entities
            if entity.domain == "timer" and entity.entity_id != "timer.abstract"
        )
    name = str(data.get("name") or "").strip().casefold().removesuffix(" timer")
    if name:
        matches = tuple(
            entity.entity_id
            for entity in home.entities
            if entity.domain == "timer"
            and name
            in {
                entity.name.casefold().removesuffix(" timer"),
                entity.entity_id.split(".", 1)[-1].replace("_", " ").casefold(),
            }
        )
        if len(matches) == 1:
            return matches
    active = tuple(
        dict.fromkeys(
            entity_id
            for operation in setup
            for entity_id in operation.entity_ids
            if entity_id.startswith("timer.")
        )
    )
    if len(active) == 1:
        return active
    if intent == "HassStartTimer":
        return ("timer.abstract",)
    return ()


def _step_operations(
    step: PlannedIntent,
    home: Home,
    setup: tuple[Operation, ...] = (),
) -> tuple[Operation, ...]:
    intent = step.operation
    data = dict(step.call.data)
    entity_ids = tuple(sorted(step.entity_ids))
    domain = _resolve_domain(step, home)

    if intent in {"HassTurnOn", "HassTurnOff"}:
        if domain in {"scene", "script"} and intent == "HassTurnOn":
            return tuple(
                operation
                for entity_id in entity_ids
                for operation in expand_static_invocation(home, entity_id)
            )
        return (_state_operation(entity_ids, _state_for_activation(intent, domain)),)

    if intent in {"HassGetState", "HassClimateGetTemperature", "HassGetMeasurement"}:
        return (Operation(kind="query", entity_ids=entity_ids),)

    attribute_intents = {
        "HassLightSet": ("brightness", "color", "color_temp", "temperature"),
        "HassClimateSetTemperature": ("temperature",),
        "HassSetPosition": ("position",),
        "HassFanSetSpeed": ("percentage",),
        "HassSetVolume": ("volume_level",),
    }
    if intent in attribute_intents:
        payload = {key: data[key] for key in attribute_intents[intent] if key in data}
        return (_state_operation(entity_ids, None, payload=payload),)

    state_by_intent = {
        "HassMediaPause": "paused",
        "HassMediaUnpause": "playing",
    }
    if intent in state_by_intent:
        return (_state_operation(entity_ids, state_by_intent[intent]),)

    if intent in {"HassMediaPlayerMute", "HassMediaPlayerUnmute"}:
        return (
            _state_operation(
                entity_ids,
                None,
                payload={"is_volume_muted": intent == "HassMediaPlayerMute"},
            ),
        )

    list_kind_by_intent = {
        "HassShoppingListAddItem": "shopping_list",
        "HassShoppingListCompleteItem": "shopping_list",
        "HassListAddItem": "todo_list",
        "HassListCompleteItem": "todo_list",
        "HassListRemoveItem": "todo_list",
    }
    if intent in list_kind_by_intent:
        payload = {key: value for key, value in data.items() if key not in _TARGET_KEYS}
        if intent.startswith("HassList") and data.get("name"):
            payload["list_name"] = data["name"]
        if intent in {"HassShoppingListCompleteItem", "HassListCompleteItem"}:
            payload["complete"] = True
        elif intent in {"HassShoppingListAddItem", "HassListAddItem"}:
            payload["complete"] = False
        elif intent == "HassListRemoveItem":
            payload["removed"] = True
        return (Operation(kind=list_kind_by_intent[intent], payload=payload),)

    if intent == "HassShoppingListLastItems":
        return (Operation(kind="query", intent=intent),)

    timer_intents = {
        "HassStartTimer",
        "HassIncreaseTimer",
        "HassDecreaseTimer",
        "HassPauseTimer",
        "HassUnpauseTimer",
        "HassCancelTimer",
        "HassCancelAllTimers",
        "HassTimerStatus",
    }
    if intent in timer_intents:
        entity_ids = _timer_entity_ids(intent, data, entity_ids, home, setup)
        if intent == "HassCancelAllTimers":
            entity_ids = tuple(
                entity_id for entity_id in entity_ids if entity_id != "timer.abstract"
            )
        payload = _timer_payload(intent, data, entity_ids, setup)
        state = {
            "HassPauseTimer": "paused",
            "HassUnpauseTimer": "active",
            "HassCancelTimer": "idle",
            "HassCancelAllTimers": "idle",
        }.get(intent)
        kind = "query" if intent == "HassTimerStatus" else "timer"
        return (Operation(kind=kind, entity_ids=entity_ids, state=state, payload=payload),)

    if intent in {"HassCreateAutomation", "IntentBridgeCreateAutomation"}:
        definition = data.get("definition")
        return (
            Operation(
                kind="automation",
                payload={"definition": dict(definition) if isinstance(definition, Mapping) else {}},
            ),
        )
    return ()
