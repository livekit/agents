"""
Example agent demonstrating the Filler Word Filter feature.

This agent shows how filler words like "uh", "umm", "hmm", "haan" are
intelligently filtered when the agent is speaking, while still allowing
genuine interruptions.

To test:
1. Start the agent and let it speak
2. Say "uh" or "umm" while agent is speaking - it should continue
3. Say "wait" or "stop" while agent is speaking - it should stop immediately
4. Say "umm" when agent is quiet - it should be registered as valid speech
"""

import logging
import os

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    cli,
    room_io,
)
from livekit.plugins import silero

logger = logging.getLogger("filler-filter-demo")

load_dotenv()


class FillerFilterDemoAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a helpful assistant named Alex. "
                "You speak clearly and concisely. "
                "When the user interrupts you, respond appropriately. "
                "Keep your responses to 2-3 sentences to allow for testing interruptions."
            )
        )

    async def on_enter(self):
      
        self.session.generate_reply(
            user_input=(
                "Please explain what artificial intelligence is. "
                "Make it detailed enough that I can interrupt you mid-sentence."
            )
        )


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Configure filler filter
    # You can customize the ignored words list
    ignored_words = os.getenv(
        "LIVEKIT_IGNORED_FILLER_WORDS", "uh,umm,hmm,haan"
    ).split(",")

    session = AgentSession(
        stt="deepgram/nova-3",
        llm="openai/gpt-4.1-mini",
        tts="cartesia/sonic-2:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
        resume_false_interruption=True,
        false_interruption_timeout=1.0,
        # Enable filler filter with custom configuration
        filler_filter_enabled=True,
        filler_ignored_words=[w.strip() for w in ignored_words if w.strip()],
        filler_min_confidence=0.0,  # Adjust based on your ASR confidence needs
    )

    # Log when transcripts are filtered
    @session.on("user_input_transcribed")
    def _on_user_input_transcribed(ev):
        logger.info(
            f"User transcript: '{ev.transcript}' (final={ev.is_final})",
            extra={"transcript": ev.transcript, "is_final": ev.is_final},
        )

    await session.start(
        agent=FillerFilterDemoAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(),
        ),
    )


if __name__ == "__main__":
    # For console mode, set dummy LiveKit credentials if not already set
    # Console mode uses simulate_job() and doesn't actually connect to a server
    import sys
    if "console" in sys.argv and not os.getenv("LIVEKIT_URL"):
        os.environ.setdefault("LIVEKIT_URL", "ws://localhost:7880")
        os.environ.setdefault("LIVEKIT_API_KEY", "devkey")
        os.environ.setdefault("LIVEKIT_API_SECRET", "devsecret")
    
    cli.run_app(server)

