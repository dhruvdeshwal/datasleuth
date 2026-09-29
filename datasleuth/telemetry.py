"""Optional tracing. Everything here is a no-op unless configured.

* LangSmith: set ``LANGSMITH_TRACING=true`` and ``LANGSMITH_API_KEY``; LangGraph/LangChain
  pick these up automatically, no code needed.
* OpenTelemetry: set ``OTEL_EXPORTER_OTLP_ENDPOINT`` (and ``pip install datasleuth[otel]``);
  each graph node and LLM call then emits a span.
"""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Iterator
from typing import Any

log = logging.getLogger(__name__)
_tracer: Any = None
_initialised = False


def _init() -> None:
    global _tracer, _initialised
    _initialised = True
    if not os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        log.warning("OTEL_EXPORTER_OTLP_ENDPOINT is set but opentelemetry is not installed")
        return
    provider = TracerProvider(resource=Resource.create({"service.name": "datasleuth"}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer("datasleuth")


def set_tracer(tracer: Any) -> None:
    """Install a tracer explicitly (tests, or apps that configure OpenTelemetry themselves)."""
    global _tracer, _initialised
    _tracer, _initialised = tracer, True


@contextlib.contextmanager
def span(name: str, **attributes: Any) -> Iterator[Any]:
    """Context manager yielding an OpenTelemetry span, or ``None`` when tracing is off."""
    if not _initialised:
        _init()
    if _tracer is None:
        yield None
        return
    with _tracer.start_as_current_span(name) as current:
        for key, value in attributes.items():
            current.set_attribute(key, value)
        yield current
