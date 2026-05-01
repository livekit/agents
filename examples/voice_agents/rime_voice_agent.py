"""Voice agent example using Rime TTS with WebSocket streaming.

Run with:
    python rime_voice_agent.py start

Required environment variables:
    LIVEKIT_URL        — LiveKit server URL
    LIVEKIT_API_KEY    — LiveKit API key
    LIVEKIT_API_SECRET — LiveKit API secret
    RIME_API_KEY       — Rime API key (https://rime.ai)
    OPENAI_API_KEY     — OpenAI API key (or swap llm= for another provider)

The agent uses Rime's /ws3 WebSocket endpoint for low-latency TTS streaming,
which minimises time-to-first-byte on every agent turn. The HTTP synthesize()
path is preserved and can be used by swapping stream() for synthesize() in
AgentSession, or by replacing the tts= argument with a non-streaming adapter.
"""

import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RunContext,
    TurnHandlingOptions,
    cli,
    metrics,
    room_io,
)
from livekit.agents.beta import EndCallTool
from livekit.agents.llm import function_tool
from livekit.plugins import openai, rime, silero

load_dotenv()

logger = logging.getLogger("rime-voice-agent")


class RimeAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "Your name is Aria, a friendly and helpful voice assistant powered by Rime TTS. "
                "Keep responses concise and conversational — you are speaking, not writing. "
                "Do not use markdown, emojis, or special characters."
            ),
            tools=[EndCallTool()],
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(
            instructions="Greet the user warmly and ask how you can help them today."
        )

    @function_tool
    async def lookup_weather(
        self,
        context: RunContext,
        location: str,
        latitude: str,
        longitude: str,
    ) -> str:
        """Called when the user asks about the weather.

        Args:
            location: City or region name.
            latitude: Estimated latitude — do not ask the user.
            longitude: Estimated longitude — do not ask the user.
        """
        logger.info("Weather lookup for %s", location)
        return "It's sunny with a temperature of 72 degrees Fahrenheit."


server = AgentServer()


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    ctx.log_context_fields = {"room": ctx.room.name}

    session = AgentSession(
        stt=openai.STT(model="whisper-1"),
        llm=openai.LLM(model="gpt-4o-mini"),
        # Rime TTS via WebSocket — stream() is called automatically because
        # streaming=True is declared in TTSCapabilities.
        tts=rime.TTS(
            model="arcana",
            speaker="astra",
            # word_timestamps=True enables aligned transcripts for interruption
            # handling; disable if not needed to reduce message volume.
            word_timestamps=True,
        ),
        vad=ctx.proc.userdata["vad"],
        turn_handling=TurnHandlingOptions(
            interruption={"resume_false_interruption": True, "false_interruption_timeout": 1.0},
        ),
        preemptive_generation=True,
        aec_warmup_duration=3.0,
    )

    @session.on("metrics_collected")
    def _on_metrics(ev: MetricsCollectedEvent) -> None:
        metrics.log_metrics(ev.metrics)

    await session.start(
        agent=RimeAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(),
    )


if __name__ == "__main__":
    cli.run_app(server)
