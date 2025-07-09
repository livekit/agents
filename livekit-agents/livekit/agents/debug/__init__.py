import base64
import os

from opentelemetry import trace

from . import types
from .tracing import Tracing, TracingGraph, TracingHandle

# TODO(long): support different tracing backends
langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY")
langfuse_host = os.getenv("LANGFUSE_HOST")

if langfuse_public_key and langfuse_secret_key and langfuse_host:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    langfuse_auth = base64.b64encode(
        f"{langfuse_public_key}:{langfuse_secret_key}".encode()
    ).decode()
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = f"{langfuse_host}/api/public/otel"
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {langfuse_auth}"

    trace_provider = TracerProvider()
    trace_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(trace_provider)

tracer = trace.get_tracer("livekit-agents")


__all__ = [
    "Tracing",
    "TracingGraph",
    "TracingHandle",
    "tracer",
    "types",
]

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
