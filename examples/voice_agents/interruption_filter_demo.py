"""
Demo of interruption filtering with LiveKit agents.
Shows how filler words are filtered while real interruptions pass through.
"""
import logging
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from livekit.agents import (
    Agent,
    AgentSession,
    AgentServer,
    JobContext,
    cli,
)
from livekit.plugins import deepgram, openai, silero

from src.interruption_filter import InterruptionFilteredSession

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Create server instance
server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    """Entry point with interruption filtering enabled."""
    # No need for ctx.connect() when using @server.rtc_session()
    
    # Create the interruption filter
    filter_session = InterruptionFilteredSession()
    
    logger.info("🎤 Interruption Filter Demo Started")
    logger.info(f"📋 Ignored words: {list(filter_session.get_config().get_ignored_words())[:10]}")
    
    # Create agent
    agent = Agent(
        instructions=(
            "You are a helpful voice assistant that gives detailed explanations. "
            "When asked about topics, provide thorough 20-30 second responses. "
            "This will help demonstrate that you won't be interrupted by filler words."
        ),
    )
    
    # Create session
    session = AgentSession(
        vad=silero.VAD.load(),
        stt=deepgram.STT(model="nova-3"),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=openai.TTS(voice="alloy"),
    )
    
    # Hook into session events to track agent state and filter interruptions
    @session.on("agent_started_speaking")
    def on_agent_speaking():
        filter_session.set_speaking(True)
    
    @session.on("agent_stopped_speaking")
    def on_agent_stopped():
        filter_session.set_speaking(False)
    
    @session.on("user_speech_committed")
    def on_user_speech(text: str):
        """Filter user speech to prevent filler word interruptions."""
        # Check if this should cause an interruption
        should_interrupt = filter_session.should_interrupt(text)
        
        if not should_interrupt:
            # Prevent interruption by not processing further
            logger.info(f"🔇 Silently ignored: '{text}'")
            return False  # Stop event propagation
        
        return True  # Allow normal processing
    
    # Start the session
    await session.start(agent=agent, room=ctx.room)
    
    # Initial greeting
    await session.generate_reply(
        instructions=(
            "Greet the user warmly and explain that you're a demo of interruption filtering. "
            "Tell them they can try saying filler words like 'umm' or 'hmm' while you speak, "
            "and you'll keep talking. But if they say 'wait' or 'stop', you'll pause immediately."
        )
    )


if __name__ == "__main__":
    cli.run_app(server)
