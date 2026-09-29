"""ASGI entry point for ``uvicorn datasleuth.api.factory:app`` (auto-reload during development)."""

from __future__ import annotations

from datasleuth.api.app import create_app

app = create_app()
