from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import pytest

from intent_bridge.home_assistant.intent_catalog import (
    HomeAssistantCatalogProvider,
    HomeAssistantCatalogPublisher,
)


class _CatalogSource:
    def __init__(self) -> None:
        self.states: dict[str, dict[str, Any]] = {
            "light.desk": {"state": "on", "attributes": {}},
            "sensor.temperature": {"state": "20", "attributes": {"device_class": "temperature"}},
        }
        self.entity_registry: dict[str, dict[str, Any]] = {
            "light.desk": {"en": "Desk Light", "di": "desk-device", "ai": "office"},
            "sensor.temperature": {"en": "Office Temperature", "ai": "office"},
        }
        self.devices: dict[str, dict[str, Any]] = {
            "desk-device": {"name": "Desk device", "area_id": "office"},
        }
        self.areas: dict[str, dict[str, Any]] = {"office": {"name": "Office"}}
        self.floors: dict[str, dict[str, Any]] = {}
        self._listeners: set[Callable[[str], None]] = set()
        self.registry_refreshes = 0
        self.next_registry: dict[str, dict[str, Any]] | None = None

    def add_catalog_cache_listener(self, listener: Callable[[str], None]) -> None:
        self._listeners.add(listener)

    def remove_catalog_cache_listener(self, listener: Callable[[str], None]) -> None:
        self._listeners.discard(listener)

    async def refresh_registries(self, *, force: bool = False) -> None:
        assert force is True
        self.registry_refreshes += 1
        if self.next_registry is not None:
            self.entity_registry = self.next_registry

    def emit(self, change: str) -> None:
        for listener in tuple(self._listeners):
            listener(change)


async def _wait_for_generation(publisher: HomeAssistantCatalogPublisher, generation: int) -> None:
    async def published() -> None:
        while publisher.generation < generation:
            await asyncio.sleep(0.005)

    await asyncio.wait_for(published(), timeout=1.0)


@pytest.mark.asyncio
async def test_catalog_publisher_replaces_the_entire_snapshot_after_coalesced_state_changes():
    source = _CatalogSource()
    publisher = HomeAssistantCatalogPublisher(
        source,
        refresh_seconds=3600,
        event_debounce_seconds=0,
        minimum_refresh_seconds=0,
    )
    await publisher.start()
    try:
        first = publisher.snapshot()
        assert first is not None
        assert {entity.entity_id for entity in first.entities} == {
            "light.desk",
            "sensor.temperature",
        }

        # These are deliberately several changes before the worker runs. The
        # published result must be a full fresh view, not a patched first view.
        source.states["light.desk"] = {"state": "off", "attributes": {}}
        source.states.pop("sensor.temperature")
        source.states["switch.fan"] = {"state": "on", "attributes": {}}
        source.entity_registry["switch.fan"] = {"en": "Desk Fan", "ai": "office"}
        source.emit("state")
        source.emit("state")
        source.emit("state")

        await _wait_for_generation(publisher, 2)
        refreshed = publisher.snapshot()
        assert refreshed is not None
        by_id = {entity.entity_id: entity for entity in refreshed.entities}
        assert set(by_id) == {"light.desk", "switch.fan"}
        assert by_id["light.desk"].state == "off"
        assert by_id["switch.fan"].name == "Desk Fan"

        provider = HomeAssistantCatalogProvider(
            lambda: source,
            publisher_provider=lambda: publisher,
        )
        assert provider.snapshot() is refreshed
    finally:
        await publisher.stop()

    generation = publisher.generation
    source.emit("state")
    await asyncio.sleep(0.02)
    assert publisher.generation == generation


@pytest.mark.asyncio
async def test_catalog_publisher_refreshes_registry_before_a_full_rebuild():
    source = _CatalogSource()
    publisher = HomeAssistantCatalogPublisher(
        source,
        refresh_seconds=3600,
        event_debounce_seconds=0,
        minimum_refresh_seconds=0,
    )
    await publisher.start()
    try:
        source.next_registry = {
            "light.desk": {"en": "Renamed Desk Light", "di": "desk-device", "ai": "office"},
            "sensor.temperature": {"en": "Office Temperature", "ai": "office"},
        }
        source.emit("registry")

        await _wait_for_generation(publisher, 2)
        snapshot = publisher.snapshot()
        assert snapshot is not None
        assert source.registry_refreshes == 1
        assert next(entity for entity in snapshot.entities if entity.entity_id == "light.desk").name == (
            "Renamed Desk Light"
        )
    finally:
        await publisher.stop()
