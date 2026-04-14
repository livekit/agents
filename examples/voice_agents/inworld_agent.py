"""Inworld STT + TTS voice agent example.

Demonstrates using Inworld for both speech-to-text and text-to-speech
in a LiveKit voice agent.

Run in console mode:
    uv run examples/voice_agents/inworld_agent.py console

Run with LiveKit Cloud (set LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET in .env):
    uv run examples/voice_agents/inworld_agent.py dev

Then connect via https://agents-playground.livekit.io
"""

import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    cli,
    metrics,
    room_io,
)
from livekit.plugins import inworld, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("inworld-agent")

load_dotenv()


class InworldAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "Your name is Nova. You interact with users via voice. "
                "Keep your responses concise and to the point. "
                "Do not use emojis, asterisks, markdown, or other special characters. "
                "You are helpful, curious, and friendly."
            ),
        )

    async def on_enter(self):
        self.session.generate_reply()


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    session = AgentSession(
        stt=inworld.STT(),  # model="assemblyai/universal-streaming-multilingual" for alt model
        llm="openai/gpt-4.1-mini",
        tts=inworld.TTS(voice="Clive"),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics(ev):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        logger.info(f"Usage: {usage_collector.get_summary()}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=InworldAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(),
    )


if __name__ == "__main__":
    cli.run_app(server)
