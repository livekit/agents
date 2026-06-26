"""Gnani Vachana STT + TTS testing agent for LiveKit.

Uses Gnani's streaming STT and TTS services for Indian language
voice conversations. Requires GNANI_API_KEY, LIVEKIT_URL,
LIVEKIT_API_KEY, and LIVEKIT_API_SECRET environment variables.
"""

import logging
from pathlib import Path

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    MetricsCollectedEvent,
    RunContext,
    TurnHandlingOptions,
    cli,
    inference,
    metrics,
    room_io,
)
from livekit.agents.llm import function_tool
from livekit.plugins import gnani, groq

logger = logging.getLogger("gnani-agent")

_env_file = Path(__file__).resolve().parent / ".env.gnani"
load_dotenv(dotenv_path=_env_file)


class GnaniTestAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a helpful voice assistant powered by Gnani Vachana speech AI. "
                "You interact with users via voice, so keep your responses concise and natural. "
                "Do not use emojis, asterisks, markdown, or other special characters. "
                "You can help with general questions, weather lookups, and casual conversation."
            ),
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(
            instructions="Greet the user and introduce yourself as a Gnani-powered voice assistant."
        )

    @function_tool
    async def lookup_weather(
        self, context: RunContext, location: str, latitude: str, longitude: str
    ) -> str:
        """Called when the user asks for weather related information.
        Ensure the user's location (city or region) is provided.
        When given a location, estimate the latitude and longitude.

        Args:
            location: The location they are asking for
            latitude: The latitude of the location
            longitude: The longitude of the location
        """
        logger.info(f"Looking up weather for {location}")
        return "sunny with a temperature of 30 degrees celsius."


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    session: AgentSession = AgentSession(
        stt=gnani.STT(language="en-IN"),
        llm=groq.LLM(model="llama-3.1-8b-instant"),
        tts=gnani.TTS(voice="Karan"),
        vad=inference.VAD(),
        turn_handling=TurnHandlingOptions(
            interruption={
                "resume_false_interruption": True,
                "false_interruption_timeout": 1.0,
            },
        ),
        aec_warmup_duration=3.0,
    )

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent) -> None:
        metrics.log_metrics(ev.metrics)

    async def log_usage():
        logger.info(f"Usage: {session.usage}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=GnaniTestAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(),
        ),
    )


if __name__ == "__main__":
    cli.run_app(server)
