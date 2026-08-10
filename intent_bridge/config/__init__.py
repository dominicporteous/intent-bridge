"""Typed application configuration."""

import logging

from intent_bridge.config.environment import ConfigurationError, load_settings
from intent_bridge.config.models import BridgeSettings

settings = load_settings()
log = logging.getLogger("home-intent-proxy")

__all__ = ["BridgeSettings", "ConfigurationError", "load_settings", "log", "settings"]
