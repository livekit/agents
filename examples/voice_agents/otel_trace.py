import logging

from dotenv import load_dotenv
from opentelemetry.context import Context
from opentelemetry.sdk.trace import Span, SpanProcessor, TracerProvider

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    RunContext,
    cli,
    get_job_context,
    inference,
    metrics,
)
from livekit.agents.llm import FallbackAdapter as FallbackLLMAdapter, function_tool
from livekit.agents.stt import FallbackAdapter as FallbackSTTAdapter
from livekit.agents.telemetry import set_tracer_provider
from livekit.agents.tts import FallbackAdapter as FallbackTTSAdapter
from livekit.agents.voice import MetricsCollectedEvent
from livekit.plugins import openai

logger = logging.getLogger("otel-trace-example")

load_dotenv()

# This example shows how to trace the agent session with OpenTelemetry.
# It exports spans over OTLP/HTTP, so it works with any OTLP-compatible backend
# (Langfuse, Jaeger, Grafana Tempo, Honeycomb, etc.). Set up the provider once at
# module scope so jobs in THREAD mode share its exporters and background threads.
#
# Configure the destination either by passing `endpoint`/`headers` to `setup_otel`, or
# by leaving them unset and exporting the standard OTLP environment variables:
#   OTEL_EXPORTER_OTLP_ENDPOINT=https://my-collector.example.com
#   OTEL_EXPORTER_OTLP_HEADERS=Authorization=Bearer <token>
#
# Worked example — Langfuse: the endpoint is `<LANGFUSE_HOST>/api/public/otel` and auth
# is a base64-encoded `Authorization: Basic` header built from the public/secret keys:
#   import base64
#   auth = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
#   setup_otel(
#       endpoint=f"{host.rstrip('/')}/api/public/otel",
#       headers={"Authorization": f"Basic {auth}", "x-langfuse-ingestion-version": "4"},
#   )
# Refer to their docs for latest instructions: https://langfuse.com/integrations/native/opentelemetry#opentelemetry-endpoint


class SessionSpanProcessor(SpanProcessor):
    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        if (ctx := get_job_context(required=False)) is not None:
            # Use the grouping key expected by your backend, e.g. langfuse.session.id.
            span.set_attribute("session.id", ctx.job.room.name)

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def setup_otel(
    *,
    endpoint: str | None = None,
    headers: dict[str, str] | None = None,
) -> TracerProvider:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    # When endpoint/headers are None, the exporter falls back to the standard
    # OTEL_EXPORTER_OTLP_* environment variables.
    trace_provider = TracerProvider()
    trace_provider.add_span_processor(SessionSpanProcessor())
    trace_provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, headers=headers))
    )
    set_tracer_provider(trace_provider)
    return trace_provider


@function_tool
async def lookup_weather(context: RunContext, location: str) -> str:
    """Called when the user asks for weather related information.

    Args:
        location: The location they are asking for
    """

    logger.info(f"Looking up weather for {location}")

    return "sunny with a temperature of 70 degrees."


class Kelly(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="Your name is Kelly.",
            llm=FallbackLLMAdapter(
                llm=[
                    inference.LLM("openai/gpt-4.1-mini"),
                    inference.LLM("google/gemini-2.5-flash"),
                ]
            ),
            stt=FallbackSTTAdapter(
                stt=[
                    inference.STT("deepgram/nova-3"),
                    inference.STT("cartesia/ink-whisper"),
                ]
            ),
            tts=FallbackTTSAdapter(
                tts=[
                    inference.TTS("cartesia"),
                    inference.TTS("rime/coda", voice="astra"),
                ]
            ),
            tools=[lookup_weather],
        )

    async def on_enter(self) -> None:
        logger.info("Kelly is entering the session")
        self.session.generate_reply()

    @function_tool
    async def transfer_to_alloy(self) -> Agent:
        """Transfer the call to Alloy."""
        logger.info("Transferring the call to Alloy")
        return Alloy()


class Alloy(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="Your name is Alloy.",
            llm=openai.realtime.RealtimeModel(voice="alloy"),
            tools=[lookup_weather],
        )

    async def on_enter(self) -> None:
        logger.info("Alloy is entering the session")
        self.session.generate_reply()

    @function_tool
    async def transfer_to_kelly(self) -> Agent:
        """Transfer the call to Kelly."""

        logger.info("Transferring the call to Kelly")
        return Kelly()


trace_provider = setup_otel()
server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    # (optional) add a shutdown callback to flush the trace before process exit
    async def flush_trace() -> None:
        trace_provider.force_flush()

    ctx.add_shutdown_callback(flush_trace)

    session: AgentSession = AgentSession()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent) -> None:
        metrics.log_metrics(ev.metrics)

    await session.start(agent=Kelly(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
