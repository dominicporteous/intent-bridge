"""Executable-process setup kept outside importable policy modules."""

import logging

from agents import set_tracing_disabled
from dotenv import load_dotenv


def configure_process() -> None:
    """Load deployment environment and configure process-wide SDK behavior."""
    load_dotenv()
    set_tracing_disabled(True)

    from intent_bridge.config import settings

    logging.basicConfig(
        level=getattr(logging, settings.api.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


__all__ = ["configure_process"]
