import inspect
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from intent_bridge import bootstrap
from intent_bridge.config import environment, settings
from intent_bridge.home_assistant import catalog as ha_catalog
from intent_bridge.indicators import policy as indicator_policy


def test_process_configuration_is_owned_by_bootstrap(monkeypatch):
    load = Mock()
    tracing = Mock()
    logging_setup = Mock()
    monkeypatch.setattr(bootstrap, "load_dotenv", load)
    monkeypatch.setattr(bootstrap, "set_tracing_disabled", tracing)
    monkeypatch.setattr(bootstrap.logging, "basicConfig", logging_setup)

    bootstrap.configure_process()

    load.assert_called_once_with()
    tracing.assert_called_once_with(True)
    logging_setup.assert_called_once_with(
        level=getattr(logging, settings.api.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def test_settings_module_has_no_process_bootstrap_side_effects():
    source = inspect.getsource(environment)
    assert "load_dotenv" not in source
    assert "basicConfig" not in source
    assert "set_tracing_disabled" not in source


def test_only_config_environment_reads_process_environment():
    package = Path(environment.__file__).parents[1]
    readers = []
    for path in package.rglob("*.py"):
        if "os.getenv" in path.read_text(encoding="utf-8"):
            readers.append(path.relative_to(package).as_posix())
    assert readers == []

    environment_source = inspect.getsource(environment)
    assert "os.environ" in environment_source


def test_catalog_policy_queries_plain_cached_data():
    catalog = SimpleNamespace(
        states={
            "light.office": {"attributes": {"friendly_name": "Desk"}},
            "switch.office": {"attributes": None},
        },
        entity_registry={
            "light.office": {"di": "device", "ai": "office", "en": "Desk light"},
            "switch.office": {"di": "device", "ai": "office"},
        },
        devices={"device": {"name_by_user": "Voice device"}},
        areas={"office": {"name": "Office"}, "kitchen": {"name": "Kitchen"}},
    )

    context = ha_catalog.entity_context(catalog, "light.office", catalog.states["light.office"])
    assert context["device_name"] == "Voice device"
    assert context["area_name"] == "Office"
    assert ha_catalog.resolve_area_reference(catalog, area_name=" office ") == (
        "office",
        "Office",
    )
    assert ha_catalog.area_mentioned_in_text(catalog, "lights in the kitchen") == (
        "kitchen",
        "Kitchen",
    )
    assert ha_catalog.entities_in_area(catalog, "light", "office") == ["light.office"]


def test_indicator_policy_is_parameterized_and_transport_free():
    assert indicator_policy.parse_indicator_rgb("blue") == [0, 0, 255]
    assert indicator_policy.parse_indicator_rgb("#01020f") == [1, 2, 15]
    assert indicator_policy.parse_indicator_rgb("1, bad, 3") is None
    assert (
        indicator_policy.find_native_effect({"effect_list": ["None", "Slow Pulse"]}, "pulse")
        == "Slow Pulse"
    )
    assert indicator_policy.find_native_effect({}, "pulse") is None
    assert indicator_policy.effect_wants_software_pulse("auto")
    assert indicator_policy.find_neutral_effect({"effect_list": ["Solid"]}) == "Solid"
    assert indicator_policy.light_supports_colour({"supported_color_modes": ["xy"]})
    assert not indicator_policy.light_supports_colour({"supported_color_modes": "rgb"})

    restored = indicator_policy.snapshot_restore_light_data(
        {
            "brightness": 0,
            "effect": "Pulse",
            "color_mode": "color_temp",
            "color_temp_kelvin": 2700.8,
        }
    )
    assert restored == {
        "brightness": 1,
        "effect": "Pulse",
        "color_temp_kelvin": 2700,
    }
