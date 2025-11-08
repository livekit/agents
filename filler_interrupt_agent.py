"""
Filler Word Interrupt Handler Agent - FREE VERSION
Uses Groq (free LLM), Deepgram (free STT), Cartesia (free TTS)

Author: Guttula Viswa Venkata Yashwanth
"""

import logging
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
    llm,
)
from livekit.plugins import deepgram, silero, cartesia, groq

from filler_filter import FillerInterruptionFilter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("filler-interrupt-agent")


class FillerAwareAgent(Agent):
    """Agent with intelligent filler word interruption filtering"""
    
    def __init__(self, filler_filter: FillerInterruptionFilter):
        super().__init__(
            instructions=(
                "You are a helpful and friendly voice assistant. "
                "Engage in natural conversation with the user. "
                "You can discuss any topic they're interested in. "
                "Be concise but informative in your responses."
            )
        )
        self.filler_filter = filler_filter
    
    async def on_enter(self):
        """Called when agent becomes active"""
        logger.info("✨ Agent started - filler filtering is active")
        await self.session.generate_reply(
            instructions="Greet the user warmly and ask how you can help them today."
        )


async def entrypoint(ctx: JobContext):
    """Main entry point for the agent"""
    
    # Connect to the LiveKit room
    await ctx.connect()
    logger.info(f"🔗 Connected to room: {ctx.room.name}")
    
    # Define filler words to ignore (configurable list)
    # These will be filtered ONLY when agent is speaking
    ignored_filler_words = [
        # English fillers
        'uh', 'uhh', 'um', 'umm', 'hmm', 'hmmm', 'er', 'err',
        'ah', 'ahh', 'oh', 'ehh', 'mhm', 'mmm',
        # Hindi fillers
        'haan', 'achha', 'theek', 'thik', 'accha',
    ]
    
    # Create the filler filter with configuration
    filler_filter = FillerInterruptionFilter(
        ignored_words=ignored_filler_words,
        confidence_threshold=0.65  # Ignore fillers with confidence < 65%
    )
    
    # Create agent session with FREE providers
    session = AgentSession(
        vad=silero.VAD.load(),  # Voice Activity Detection (built-in, free)
        stt=deepgram.STT(model="nova-2"),  # Speech-to-Text ($200 free credit)
        llm=groq.LLM(model="llama-3.1-8b-instant"),  # LLM (completely free!)
        tts=cartesia.TTS(),  # Text-to-Speech (free tier available)
    )
    
    # Track agent speaking state
    @session.on("agent_speech_started")
    def on_agent_speech_started():
        """Called when agent starts speaking"""
        filler_filter.set_agent_speaking(True)
        logger.info("🗣️  Agent started speaking")
    
    @session.on("agent_speech_stopped")
    def on_agent_speech_stopped():
        """Called when agent stops speaking"""
        filler_filter.set_agent_speaking(False)
        logger.info("🤫 Agent stopped speaking")
    
    # Process user transcriptions
    @session.on("user_speech_committed")
    def on_user_speech_committed(msg: llm.ChatMessage):
        """Process user speech and filter fillers"""
        user_text = msg.content
        # Get confidence if available (default to high confidence)
        confidence = getattr(msg, 'confidence', 1.0)
        
        # Check if this should be filtered as a filler interruption
        should_ignore = filler_filter.should_ignore_interruption(
            text=user_text,
            confidence=confidence
        )
        
        if should_ignore:
            # Log ignored filler
            logger.info(
                f"🚫 IGNORED filler interruption: '{user_text}' "
                f"(confidence: {confidence:.2f}, agent speaking: {filler_filter.agent_is_speaking})"
            )
            filler_filter.log_ignored_interruption(user_text, confidence)
        else:
            # Log valid interruption/speech
            logger.info(
                f"✅ VALID user speech: '{user_text}' "
                f"(confidence: {confidence:.2f}, agent speaking: {filler_filter.agent_is_speaking})"
            )
            filler_filter.log_valid_interruption(user_text, confidence)
    
    # Start the agent
    agent = FillerAwareAgent(filler_filter)
    await session.start(agent=agent, room=ctx.room)
    
    # Log statistics when shutting down
    @ctx.on("shutdown")
    async def on_shutdown():
        """Print statistics on shutdown"""
        stats = filler_filter.get_statistics()
        logger.info("\n" + "=" * 70)
        logger.info("📊 FILLER FILTER STATISTICS")
        logger.info("=" * 70)
        logger.info(f"✅ Valid interruptions processed: {stats['valid_interruptions']}")
        logger.info(f"🚫 Filler interruptions ignored: {stats['ignored_fillers']}")
        logger.info(f"📈 Total speech events: {stats['total_events']}")
        if stats['total_events'] > 0:
            logger.info(f"📉 Filter rate: {stats['filter_rate']:.1%}")
        logger.info("=" * 70 + "\n")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
