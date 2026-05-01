import logging
import os

from dotenv import load_dotenv
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    RunContext,
    cli,
    inference,
    metrics,
)
from livekit.agents.llm import function_tool
from livekit.agents.telemetry import set_tracer_provider
from livekit.agents.voice import MetricsCollectedEvent
from livekit.plugins import silero

logger = logging.getLogger("mlflow-trace-example")

load_dotenv()

# This example shows how to use MLflow to trace the agent session.
# MLflow accepts OpenTelemetry traces via its OTLP endpoint.
#
# Prerequisites:
#   pip install mlflow
#   mlflow server --port 5000
#
# Environment variables:
#   OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:5000
#   OTEL_EXPORTER_OTLP_HEADERS=x-mlflow-experiment-id=0
#
# Learn more: https://mlflow.org/docs/latest/genai/tracing/integrations/listing/livekit


def setup_mlflow(
    *,
    endpoint: str | None = None,
    experiment_id: str | None = None,
    service_name: str | None = None,
) -> TracerProvider:
    """Configure OpenTelemetry to send traces to MLflow.

    Args:
        endpoint: MLflow tracking server URL (default: OTEL_EXPORTER_OTLP_ENDPOINT or http://localhost:5000)
        experiment_id: MLflow experiment ID (default: OTEL_EXPORTER_OTLP_HEADERS or "0")
        service_name: Service name for spans (default: OTEL_SERVICE_NAME or "livekit-agent")
    """
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    endpoint = endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:5000")
    experiment_id = experiment_id or "0"
    service_name = service_name or os.getenv("OTEL_SERVICE_NAME", "livekit-agent")

    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = endpoint
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = os.getenv(
        "OTEL_EXPORTER_OTLP_HEADERS", f"x-mlflow-experiment-id={experiment_id}"
    )

    resource = Resource.create({SERVICE_NAME: service_name})
    trace_provider = TracerProvider(resource=resource)
    trace_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
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


class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a helpful voice assistant. Keep your responses concise and conversational.",
            llm=inference.LLM("openai/gpt-4.1-mini"),
            stt=inference.STT("deepgram/nova-3"),
            tts=inference.TTS("cartesia/sonic-3"),
            tools=[lookup_weather],
        )

    async def on_enter(self):
        self.session.generate_reply()


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    trace_provider = setup_mlflow()

    async def flush_trace():
        trace_provider.force_flush()

    ctx.add_shutdown_callback(flush_trace)

    session = AgentSession(vad=silero.VAD.load())

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)

    await session.start(agent=MyAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
