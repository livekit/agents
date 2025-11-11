"""
Filler-Aware Interruption Handling Agent

This example demonstrates how to use the FillerAwareInterruptFilter
to intelligently handle user interruptions, ignoring filler words when
the agent is speaking while allowing genuine interruptions.
"""

import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    AgentStateChangedEvent,
    JobContext,
    JobProcess,
    UserInputTranscribedEvent,
    WorkerOptions,
    cli,
)
from livekit.plugins import deepgram, silero

# Handle both relative and absolute imports
try:
    from .filler_interrupt_filter import FillerAwareInterruptFilter
except ImportError:
    # If running as a script directly, add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent))
    from filler_interrupt_filter import FillerAwareInterruptFilter

logger = logging.getLogger("filler-interrupt-agent")

# Load environment variables
load_dotenv()

# Hardcoded sentence for the agent to speak
HARDCODED_SENTENCE = (
    "Hello! This is a test of the filler-aware interruption handling system. "
    "I will continue speaking this long sentence so you can test interrupting me. "
    "Try saying filler words like 'uh' or 'umm' and I should ignore them. "
    "But if you say 'stop' or 'wait', I will immediately stop speaking. "
    "This is a demonstration of intelligent interruption handling."
)


class FillerAwareAgent(Agent):
    """Agent that uses filler-aware interruption handling with hardcoded speech."""

    def __init__(self, interrupt_filter: FillerAwareInterruptFilter):
        # No LLM needed - we'll use hardcoded text
        super().__init__(
            instructions="This agent speaks hardcoded text for testing interruption handling."
        )
        self._interrupt_filter = interrupt_filter

    async def on_enter(self):
        """Called when the agent enters the session."""
        # Speak the hardcoded sentence
        logger.info(f"Agent starting to speak: {HARDCODED_SENTENCE}")
        self.session.say(HARDCODED_SENTENCE, allow_interruptions=True)


def prewarm(proc: JobProcess):
    """Preload VAD model."""
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    """Main entrypoint for the agent worker."""

    # Initialize the filler interrupt filter
    # These can be configured via CLI args or environment variables
    ignored_words = os.getenv("IGNORED_WORDS", "uh,umm,hmm,haan").split(",")
    interrupt_keywords = os.getenv(
        "INTERRUPT_KEYWORDS", "stop,wait,hold on,no,cancel,pause"
    ).split(",")

    filter_instance = FillerAwareInterruptFilter(
        ignored_words=[w.strip() for w in ignored_words if w.strip()],
        interrupt_keywords=[kw.strip() for kw in interrupt_keywords if kw.strip()],
        min_asr_confidence=float(os.getenv("MIN_ASR_CONFIDENCE", "0.55")),
        interrupt_confidence=float(os.getenv("INTERRUPT_CONFIDENCE", "0.70")),
        min_meaningful_tokens=int(os.getenv("MIN_MEANINGFUL_TOKENS", "2")),
    )

    # Create agent session with Deepgram STT and TTS (no LLM needed)
    # Note: Using Deepgram for both STT and TTS. You can also use AssemblyAI STT
    # by replacing with: stt=assemblyai.STT()
    # Using VAD-based turn detection (no turn-detector plugin required)
    # To use MultilingualModel turn detection, install: pip install livekit-plugins-turn-detector
    # and uncomment: from livekit.plugins.turn_detector.multilingual import MultilingualModel
    # then change turn_detection="vad" to turn_detection=MultilingualModel()
    session = AgentSession(
        stt=deepgram.STT(model="nova-2", language="en"),
        # No LLM - we use hardcoded text via session.say()
        tts=deepgram.TTS(model="aura-asteria-en", voice="asteria"),
        turn_detection="vad",  # Use VAD-based turn detection (simpler, no extra plugin needed)
        vad=ctx.proc.userdata["vad"],
        allow_interruptions=True,  # Enable interruptions
        resume_false_interruption=True,
        false_interruption_timeout=2.0,
    )

    # Track agent speaking state
    @session.on("agent_state_changed")
    def _on_agent_state_changed(ev: AgentStateChangedEvent):
        """Update filter when agent state changes."""
        is_speaking = ev.new_state == "speaking"
        filter_instance.update_speaking_state(is_speaking)
        logger.info(f"Agent state: {ev.old_state} -> {ev.new_state}")

    # Handle user transcriptions
    @session.on("user_input_transcribed")
    def _on_user_input_transcribed(ev: UserInputTranscribedEvent):
        """Process user transcriptions through the filter."""
        if not ev.is_final:
            # Only process final transcripts for interruption decisions
            return

        # Note: UserInputTranscribedEvent doesn't include confidence.
        # We use a default confidence of 0.7 (medium-high) for decision making.
        # For more accurate filtering, you could subscribe to STT events directly
        # to get confidence scores, but that requires more complex integration.
        confidence = 0.7  # Default confidence (can be adjusted based on your needs)

        # Process through filter
        decision = filter_instance.handle_transcript(
            text=ev.transcript,
            confidence=confidence,
            is_final=ev.is_final,
        )

        # Log the decision
        logger.info(
            f"Transcript decision: {decision.action} | "
            f"Reason: {decision.reason} | "
            f"Text: '{ev.transcript}' | "
            f"Confidence: {confidence:.2f}"
        )

        # If valid interrupt detected and agent is speaking, interrupt the session
        if decision.action == "interrupt" and filter_instance.is_agent_speaking:
            logger.warning(f"Interrupting agent: {ev.transcript}")
            session.interrupt()

    # Start the session
    await session.start(
        agent=FillerAwareAgent(interrupt_filter=filter_instance),
        room=ctx.room,
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))

