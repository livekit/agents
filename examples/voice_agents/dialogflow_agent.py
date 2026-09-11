"""LiveKit voice agent using Google Dialogflow CX as the conversational engine.

This example demonstrates how to use Dialogflow CX as the LLM provider
in a LiveKit Agents STT -> LLM -> TTS pipeline.

All conversation logic (intents, flows, pages, fulfillment) is managed
in the Dialogflow CX Console — the ``instructions`` field has no effect.

Required environment variables:
    LIVEKIT_URL: WebSocket URL of your LiveKit server
    LIVEKIT_API_KEY: API key for LiveKit authentication
    LIVEKIT_API_SECRET: API secret for LiveKit authentication
    GOOGLE_APPLICATION_CREDENTIALS: Path to GCP service account key JSON
    GOOGLE_CLOUD_PROJECT: GCP project ID (or pass project_id to LLM)
    DEEPGRAM_API_KEY: API key for Deepgram STT

Usage:
    python dialogflow_agent.py dev
"""

import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    MetricsCollectedEvent,
    cli,
    metrics,
    room_io,
)
from livekit.plugins import dialogflow, google as google_tts, silero

logger = logging.getLogger("dialogflow-agent")

load_dotenv()


class DialogflowAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            # instructions are NOT used by Dialogflow CX — all conversation logic
            # is configured in the Dialogflow CX console
            instructions="",
            llm=dialogflow.LLM(
                # project_id falls back to GOOGLE_CLOUD_PROJECT env var
                agent_id="YOUR_DIALOGFLOW_AGENT_ID",
                location="us-central1",  # or "global"
                language_code="en",
            ),
            stt=google_tts.STT(),
            tts=google_tts.TTS(),
        )

    async def on_enter(self):
        self.session.generate_reply()


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        vad=silero.VAD.load(),
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=DialogflowAgent(),
        room=ctx.room,
    )


if __name__ == "__main__":
    cli.run_app(server)
