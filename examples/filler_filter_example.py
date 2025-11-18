"""
Example Voice Agent with Filler Filter

This example demonstrates how to use the filler filter in a LiveKit voice agent.
The agent will ignore filler words like "umm", "hmm", "haan" when detecting interruptions.

Author: Raghav
Date: November 18, 2025
"""

import logging
from dotenv import load_dotenv

from livekit.agents import Agent, AgentSession, JobContext, cli
from livekit.plugins import openai, deepgram, cartesia, silero

# Load environment variables (for API keys)
load_dotenv()

# Configure logging to see filler filter decisions
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("filler-filter-example")


async def entrypoint(ctx: JobContext):
    """Main entry point for the voice agent."""
    
    logger.info("Starting voice agent with filler filter...")
    
    # Create AgentSession with filler filter configuration
    session = AgentSession(
        # Voice components
        vad=silero.VAD.load(),
        stt=deepgram.STT(),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=cartesia.TTS(),
        
        # Filler filter configuration
        # Option 1: Use default fillers (loaded from IGNORED_WORDS env var)
        # (No need to specify anything)
        
        # Option 2: Specify custom filler words
        ignored_filler_words=[
            # English fillers
            "uh", "umm", "hmm", "er", "ah", "oh", 
            "yeah", "yep", "okay", "ok", "mm", "mhm",
            # Hindi/Indian fillers (uncomment if needed)
            "haan", "arey", "accha", "theek", "yaar", "bas",
        ],
        
        # Confidence threshold for filtering
        # Transcripts with confidence below this are treated as fillers
        filler_confidence_threshold=0.5,
        
        # Other interruption settings
        allow_interruptions=True,
        min_interruption_duration=0.5,  # Minimum 0.5s of speech to consider
        min_interruption_words=0,  # No minimum word count (filter handles it)
    )
    
    # Create the agent
    agent = Agent(
        instructions="""You are a helpful and friendly voice assistant.
        
        When users speak to you:
        - Respond naturally and conversationally
        - If you hear filler words like 'umm' or 'hmm' while speaking, continue speaking
        - Only stop if you hear a real interruption like 'wait', 'stop', or a question
        
        Your goal is to have natural conversations without being interrupted by speech disfluencies.
        """
    )
    
    # Event handlers to monitor filter behavior
    @session.on("agent_state_changed")
    def on_agent_state_changed(ev):
        logger.info(f"Agent state: {ev.old_state} → {ev.new_state}")
    
    @session.on("user_state_changed")
    def on_user_state_changed(ev):
        logger.info(f"User state: {ev.old_state} → {ev.new_state}")
    
    @session.on("user_input_transcribed")
    def on_user_input(ev):
        logger.info(f"User said: '{ev.transcript}' (final={ev.is_final})")
    
    # Start the agent session
    await session.start(agent=agent, room=ctx.room)
    
    logger.info("""
    ╔════════════════════════════════════════════════════════════════╗
    ║                 FILLER FILTER AGENT READY                      ║
    ╟────────────────────────────────────────────────────────────────╢
    ║  The agent will now ignore filler words during interruptions   ║
    ║                                                                ║
    ║  Try these test scenarios:                                     ║
    ║  1. Say "Tell me a long story" and let agent speak            ║
    ║  2. While agent speaks, say "umm" → Should be IGNORED         ║
    ║  3. While agent speaks, say "stop" → Should INTERRUPT         ║
    ║  4. While agent speaks, say "umm wait" → Should INTERRUPT     ║
    ║                                                                ║
    ║  Watch the logs for [IGNORED_FILLER] and [VALID_INTERRUPT]   ║
    ╚════════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    # Run the agent
    cli.run_app(entrypoint)
