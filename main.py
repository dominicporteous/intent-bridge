"""Legacy ASGI entry point for ``uvicorn main:app``."""

from intent_bridge.bootstrap import configure_process


def _configured_app():
    configure_process()
    from intent_bridge.application import app

    return app


app = _configured_app()

__all__ = ["app"]
