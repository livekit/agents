"""
Telnyx Voice Agent Example.

This example demonstrates a voice agent using:
- Telnyx STT (Speech-to-Text)
- Telnyx TTS (Text-to-Speech)
- OpenAI LLM (GPT-4.1-mini)

Usage:
    export TELNYX_API_KEY='your_telnyx_api_key'
    export OPENAI_API_KEY='your_openai_api_key'
    export LIVEKIT_URL='wss://your-livekit-server'
    export LIVEKIT_API_KEY='your_livekit_api_key'
    export LIVEKIT_API_SECRET='your_livekit_api_secret'

    python telnyx_voice_agent.py dev      # Development mode with hot reload
    python telnyx_voice_agent.py console  # Terminal mode (no server needed)
"""

import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    RunContext,
    cli,
    metrics,
    room_io,
)
from livekit.agents.llm import function_tool
from livekit.plugins import openai, silero, telnyx

logger = logging.getLogger("telnyx-voice-agent")

load_dotenv()


class TelnyxVoiceAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a helpful voice assistant powered by Telnyx. "
                "Keep your responses concise and conversational. "
                "Do not use emojis, asterisks, or markdown in your responses. "
                "You are friendly and professional."
            ),
        )

    async def on_enter(self):
        self.session.generate_reply(allow_interruptions=False)

    @function_tool
    async def get_current_time(self, context: RunContext):
        """Called when the user asks for the current time."""
        import datetime

        now = datetime.datetime.now()
        return f"The current time is {now.strftime('%I:%M %p')}."

    @function_tool
    async def lookup_weather(self, context: RunContext, location: str):
        """Called when the user asks about the weather.

        Args:
            location: The city or location to get weather for.
        """
        logger.info(f"Looking up weather for {location}")
        return f"The weather in {location} is sunny with a temperature of 72 degrees Fahrenheit."


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    session = AgentSession(
        stt=telnyx.STT(
            language="en",
            transcription_engine="telnyx",
        ),
        llm=openai.LLM(model="gpt-4.1-mini"),
        tts=telnyx.TTS(
            voice="Telnyx.NaturalHD.astra",
        ),
        vad=ctx.proc.userdata["vad"],
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=TelnyxVoiceAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(),
    )


if __name__ == "__main__":
    cli.run_app(server)
