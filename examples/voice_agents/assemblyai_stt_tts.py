"""A voice agent that uses AssemblyAI for both ears and voice.

- Speech-to-text: AssemblyAI Universal-3.6 Pro, with AssemblyAI's own end-of-turn detection
- Text-to-speech: AssemblyAI Streaming TTS, with word-aligned transcripts

Requires ASSEMBLYAI_API_KEY, plus LiveKit credentials for the LLM through LiveKit Inference.

    python examples/voice_agents/assemblyai_stt_tts.py console
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
    inference,
    metrics,
)
from livekit.agents.llm import function_tool
from livekit.plugins import assemblyai, silero

logger = logging.getLogger("assemblyai-stt-tts")

load_dotenv()

# AssemblyAI recommends matching Silero's activation threshold (used here for barge-in)
# to the STT's vad_threshold, its default on Universal-3.6 Pro.
VAD_THRESHOLD = 0.2


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "Your name is Ava. You are a friendly voice assistant built on AssemblyAI "
                "speech-to-text and text-to-speech. Keep replies short and conversational: "
                "one or two sentences, no lists, no markdown, no emojis. Spell out anything "
                "that should be read aloud exactly, but write numbers, dates and prices "
                "normally; the voice expands them on its own."
            ),
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(instructions="greet the user and introduce yourself")

    @function_tool
    async def lookup_weather(self, context: RunContext, location: str) -> str:
        """Called when the user asks about the weather.

        Args:
            location: The city or region the user asked about
        """
        logger.info("looking up weather for %s", location)
        return "sunny with a temperature of 72 degrees."


server = AgentServer()


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load(activation_threshold=VAD_THRESHOLD)


server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    ctx.log_context_fields = {"room": ctx.room.name}

    tts = assemblyai.TTS(
        # Any preset voice works; the language is derived from it. English voices:
        # alba, anna, charles, eve, george, jane, jean, mary, michael, paul, vera.
        # Other languages: lola (es), juergen (de), giovanni (it), rafael (pt), estelle (fr).
        voice="jane",
        # For data residency, pin a region: "wss://streaming-tts.us.assemblyai.com"
        # or "wss://streaming-tts.eu.assemblyai.com".
    )
    # Open the TTS socket now, so the first reply doesn't pay for the handshake.
    tts.prewarm()

    session: AgentSession = AgentSession(
        stt=assemblyai.STT(
            model="universal-3-6-pro",
            # with turn_detection="stt": min_turn_silence=128, max_turn_silence=1280
            mode="balanced",
            vad_threshold=VAD_THRESHOLD,
        ),
        llm=inference.LLM("openai/gpt-4.1-mini"),
        tts=tts,
        vad=ctx.proc.userdata["vad"],
        turn_handling=TurnHandlingOptions(
            # AssemblyAI decides when the user has finished speaking
            turn_detection="stt",
            # LiveKit's delay is added on top of AssemblyAI's own endpointing
            endpointing={"min_delay": 0},
            interruption={"resume_false_interruption": True},
        ),
        tts_text_transforms=["filter_emoji", "filter_markdown"],
    )

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent) -> None:
        metrics.log_metrics(ev.metrics)

    async def log_usage() -> None:
        logger.info("usage: %s", session.usage)

    ctx.add_shutdown_callback(log_usage)

    await session.start(agent=Assistant(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
