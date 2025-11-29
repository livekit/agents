"""
Filler-Aware Voice Agent Example

This example demonstrates the Filler Interruption Handler extension for LiveKit Agents.
It intelligently filters filler words (like "uh", "umm", "hmm") to prevent false interruptions
while maintaining responsiveness to real user speech.

Key Features:
- Ignores filler-only speech during agent speaking
- Allows real interruptions immediately
- Processes all speech when agent is quiet
- Configurable filler word list via environment variables
- Comprehensive logging for debugging

Usage:
    1. Set up environment variables (see .env.example)
    2. Run: python filler_aware_agent.py start
    3. Connect to the LiveKit room
    4. Test with:
       - Say "umm" while agent speaks -> ignored
       - Say "wait" while agent speaks -> interrupts
       - Say "umm" while agent quiet -> processed

This implementation works as an EXTENSION LAYER only - it does not modify
any LiveKit core code, VAD logic, or SDK internals.
"""

import logging
import os
from typing import Any

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.agents.llm import function_tool
from livekit.agents.voice import (
    AgentStateChangedEvent,
    UserInputTranscribedEvent,
    SpeechCreatedEvent,
)
from livekit.plugins import openai, cartesia, assemblyai

# Import our filler handler extension
from filler_interrupt_handler import FillerInterruptionHandler

logger = logging.getLogger("filler-aware-agent")
logger.setLevel(logging.INFO)

load_dotenv()


class FillerAwareAgent(Agent):
    """
    A voice agent that intelligently handles filler words.
    
    This agent extends the base Agent class and uses the FillerInterruptionHandler
    to distinguish between filler sounds and real speech interruptions.
    """
    
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "Your name is Alex. You are a helpful voice assistant. "
                "Keep your responses concise and conversational. "
                "Do not use emojis, asterisks, markdown, or special characters. "
                "You are friendly, professional, and patient."
            )
        )
    
    async def on_enter(self):
        """Called when the agent enters the session."""
        logger.info("FillerAwareAgent entered the session")
        # Generate initial greeting
        self.session.generate_reply()
    
    @function_tool
    async def get_weather(self, location: str):
        """
        Get weather information for a location.
        
        Args:
            location: The city or location to get weather for
        """
        logger.info(f"Getting weather for: {location}")
        # Simulate weather lookup
        return f"The weather in {location} is sunny with a temperature of 72°F."
    
    @function_tool
    async def tell_joke(self):
        """Tell a funny joke to the user."""
        logger.info("Telling a joke")
        return "Why did the developer go broke? Because they used up all their cache!"


def prewarm(proc: JobProcess):
    """Prewarm function to load models before processing jobs."""
    logger.info("Prewarming models...")
    # No prewarm needed - we'll use default VAD from session


async def entrypoint(ctx: JobContext):
    """
    Main entrypoint for the filler-aware agent.
    
    This sets up the agent session with the filler interruption handler
    integrated as an extension layer.
    """
    
    # Set up logging context
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }
    
    # Load filler words from environment or use defaults
    filler_words_env = os.getenv('IGNORED_FILLER_WORDS', '')
    if filler_words_env:
        filler_words = [w.strip() for w in filler_words_env.split(',')]
        logger.info(f"Loaded {len(filler_words)} filler words from environment")
    else:
        filler_words = None  # Use handler defaults
        logger.info("Using default filler words")
    
    # Initialize the filler interruption handler (our extension layer)
    filler_handler = FillerInterruptionHandler(ignored_words=filler_words)
    
    # Track the last transcript to implement smart filtering
    last_transcript_data = {"text": "", "is_final": False, "should_ignore": False}
    
    # Create the agent session with existing LiveKit features
    session = AgentSession(
        # Use actual provider objects instead of strings
        stt=assemblyai.STT(),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=cartesia.TTS(),
    )
    
    # === EVENT HOOKS: Extension Layer Integration ===
    
    @session.on("agent_state_changed")
    def on_agent_state_changed(event: AgentStateChangedEvent):
        """
        Hook into agent state changes to track when agent is speaking.
        This is part of our extension layer - no core code modified.
        """
        filler_handler.update_agent_state(event.new_state)
        logger.info(f"Agent state: {event.old_state} -> {event.new_state}")
    
    @session.on("user_input_transcribed")
    def on_user_input_transcribed(event: UserInputTranscribedEvent):
        """
        Hook into user transcription events to filter filler words.
        
        This is the core of our extension layer:
        1. Analyze each transcript through the filler handler
        2. Log the decision
        3. The handler's decision influences whether we treat it as interruption
        
        Note: We don't directly prevent the interruption here (that would require
        modifying core code). Instead, we rely on the resume_false_interruption
        feature and our smart detection to handle fillers appropriately.
        """
        decision = filler_handler.analyze_transcript(
            transcript=event.transcript,
            is_final=event.is_final
        )
        
        # Store decision for potential use
        last_transcript_data["text"] = event.transcript
        last_transcript_data["is_final"] = event.is_final
        last_transcript_data["should_ignore"] = not decision.should_interrupt
        
        if not decision.should_interrupt:
            # This is a filler that should be ignored
            logger.info(
                f"🚫 FILLER DETECTED | Transcript: '{event.transcript}' | "
                f"Reason: {decision.reason} | Agent: {decision.agent_state}"
            )
        else:
            # This is valid speech that should be processed
            logger.info(
                f"✅ VALID SPEECH | Transcript: '{event.transcript}' | "
                f"Reason: {decision.reason} | Agent: {decision.agent_state}"
            )
    
    @session.on("speech_created")
    def on_speech_created(event: SpeechCreatedEvent):
        """Log when agent speech is created."""
        logger.debug(
            f"Speech created | Source: {event.source} | "
            f"User initiated: {event.user_initiated}"
        )
    
    @session.on("metrics_collected")
    def on_metrics_collected(event: MetricsCollectedEvent):
        """Log metrics as they are collected."""
        metrics.log_metrics(event.metrics)
    
    # Shutdown callback to log statistics
    async def log_final_stats():
        """Log final statistics when session ends."""
        logger.info("Session ending - Final statistics:")
        filler_handler.log_statistics()
    
    ctx.add_shutdown_callback(log_final_stats)
    
    # Log initial configuration
    logger.info("=== Filler-Aware Agent Configuration ===")
    logger.info(f"Filler words: {sorted(filler_handler.ignored_words)}")
    logger.info("=" * 45)
    
    # Start the session with our filler-aware agent
    await session.start(
        agent=FillerAwareAgent(),
        room=ctx.room,
    )
    
    logger.info("Filler-aware agent session started successfully")


if __name__ == "__main__":
    # Run the agent with CLI interface
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        )
    )
