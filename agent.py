"""
LiveKit Voice Agent with Intelligent Interruption Handling
Filters filler words to prevent false interruptions during agent speech.
"""

import logging
import os
import asyncio
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    AgentStateChangedEvent,
)
from livekit.plugins import silero, openai, deepgram, cartesia
from livekit import rtc

from interruption_handler import InterruptionHandler, InterruptionConfig

logger = logging.getLogger("voice-agent")
logging.basicConfig(level=logging.INFO)
load_dotenv()


class VoiceAgent(Agent):
    """Voice agent with intelligent interruption handling."""

    def __init__(self, interruption_handler: InterruptionHandler):
        super().__init__(
            instructions=(
                "You are a helpful voice assistant. "
                "Keep your responses concise and natural. "
                "Speak clearly and at a moderate pace."
            )
        )
        self.interruption_handler = interruption_handler
        self._is_speaking = False
        self._agent_session = None

    async def on_enter(self):
        """Called when agent enters the session."""
        logger.info("Agent entered session")
        # Greet the user
        await self.session.generate_reply(
            instructions="Greet the user warmly and ask how you can help them today."
        )

    async def on_exit(self):
        """Called when agent exits the session."""
        logger.info("Agent exited session")


def prewarm(proc: JobProcess):
    """Prewarm function to load models before handling requests."""
    logger.info("Prewarming models...")
    proc.userdata["vad"] = silero.VAD.load()
    logger.info("VAD model loaded")


async def entrypoint(ctx: JobContext):
    """Main entrypoint for the voice agent."""

    # Set up logging context
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    logger.info(f"Starting agent for room: {ctx.room.name}")

    # Load ignored words from environment or use defaults
    ignored_words_str = os.getenv("IGNORED_WORDS", "uh,umm,hmm,haan,um,er,ah")
    ignored_words = [word.strip() for word in ignored_words_str.split(",")]

    confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))

    # Create interruption handler configuration
    config = InterruptionConfig.from_word_list(
        words=ignored_words,
        confidence_threshold=confidence_threshold,
        enable_dynamic_updates=True
    )

    interruption_handler = InterruptionHandler(config)
    logger.info(f"Interruption handler initialized with {len(ignored_words)} ignored words")

    # Create the agent
    agent = VoiceAgent(interruption_handler)

    # Connect to the room first
    await ctx.connect()

    # Create agent session with STT-LLM-TTS pipeline
    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(model="nova-3"),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=cartesia.TTS(),
        # Enable false interruption handling as a fallback
        resume_false_interruption=True,
        false_interruption_timeout=1.5,
        # Minimum interruption settings
        min_interruption_duration=0.3,
        min_interruption_words=0,  # We handle word filtering ourselves
    )

    agent._agent_session = session

    # Track agent speaking state
    @session.on("agent_state_changed")
    def on_agent_state_changed(ev: AgentStateChangedEvent):
        if ev.new_state == "speaking":
            agent._is_speaking = True
            logger.info("Agent started speaking")
        elif ev.new_state in ["listening", "thinking", "idle"]:
            was_speaking = agent._is_speaking
            agent._is_speaking = False
            if was_speaking:
                logger.info(f"Agent stopped speaking, now: {ev.new_state}")

    # Monitor VAD events to detect potential interruptions early
    vad_buffer = []
    vad_start_time = None

    @session.on("user_started_speaking")
    def on_user_started_speaking(ev):
        """Detect when user starts speaking."""
        nonlocal vad_start_time
        vad_start_time = asyncio.get_event_loop().time()

        if agent._is_speaking:
            logger.debug("User started speaking while agent is speaking (potential interruption)")

    @session.on("user_stopped_speaking")
    def on_user_stopped_speaking(ev):
        """Detect when user stops speaking."""
        nonlocal vad_start_time
        if vad_start_time:
            duration = asyncio.get_event_loop().time() - vad_start_time
            logger.debug(f"User speech duration: {duration:.2f}s")
            vad_start_time = None

    # Hook into transcription events to filter filler words
    @session.on("user_transcript")
    def on_user_transcript(ev):
        """Handle interim and final transcripts."""
        # Get the transcript text
        text = ev.text if hasattr(ev, 'text') else ""
        confidence = ev.confidence if hasattr(ev, 'confidence') else None
        is_final = ev.is_final if hasattr(ev, 'is_final') else False

        if not text:
            return

        logger.debug(f"Transcript ({'final' if is_final else 'interim'}): '{text}' (confidence: {confidence})")

        # Only filter during agent speech (potential interruption)
        if agent._is_speaking and is_final:
            should_ignore = interruption_handler.should_ignore_speech(text, confidence)

            if should_ignore:
                logger.info(f"🔇 Filtered filler interruption: '{text}' (confidence: {confidence})")
                # This is a filler - don't let it interrupt the agent
                # The agent will continue speaking
                return
            else:
                logger.info(f"✅ Valid interruption detected: '{text}'")

    # Start the session
    await session.start(
        agent=agent,
        room=ctx.room,
    )

    logger.info("Agent session started successfully")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))

