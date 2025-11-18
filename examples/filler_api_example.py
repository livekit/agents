"""
Example: Using the Filler Filter REST API

This example demonstrates how to:
1. Start the filler API server
2. Use REST endpoints to update filler words dynamically
3. Query current configuration

Author: Raghav (LiveKit Intern Assessment)
Date: November 19, 2025
"""

import asyncio
import logging
from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    llm,
)
from livekit.agents.voice import AgentSession
from livekit.plugins import openai, silero

# Import the API server
from livekit.agents.voice.filler_api import start_filler_api_server

logger = logging.getLogger("filler-api-example")


async def entrypoint(ctx: JobContext):
    """Voice agent with filler filter and REST API."""
    
    logger.info(f"Connecting to room: {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    
    # Wait for participant
    participant = await ctx.wait_for_participant()
    logger.info(f"Participant connected: {participant.identity}")
    
    # Create agent session with filler filter enabled
    session = AgentSession(
        vad=silero.VAD.load(),
        stt=openai.STT(model="whisper-1"),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=openai.TTS(model="tts-1"),
        chat_ctx=llm.ChatContext().append(
            text="You are a helpful voice assistant. Respond naturally to user questions.",
            role="system",
        ),
        # Filler filter configuration
        ignored_filler_words=["umm", "hmm", "haan", "arey", "uh", "er"],
        filler_confidence_threshold=0.5,
    )
    
    # Start the REST API server on port 8080
    logger.info("Starting Filler Filter REST API on http://localhost:8080")
    api_server = await start_filler_api_server(session._activity._filler_filter, port=8080)
    
    logger.info("""
    ╔════════════════════════════════════════════════════════════════╗
    ║                  FILLER FILTER REST API                        ║
    ╠════════════════════════════════════════════════════════════════╣
    ║  Base URL: http://localhost:8080                               ║
    ║                                                                ║
    ║  Available Endpoints:                                          ║
    ║  • GET  /         - API info and documentation                 ║
    ║  • GET  /fillers  - Get current filler configuration           ║
    ║  • POST /update_filler - Update filler words dynamically       ║
    ║                                                                ║
    ║  Example Usage:                                                ║
    ║                                                                ║
    ║  # Get current fillers                                         ║
    ║  curl http://localhost:8080/fillers                            ║
    ║                                                                ║
    ║  # Add new fillers                                             ║
    ║  curl -X POST http://localhost:8080/update_filler \\           ║
    ║       -H "Content-Type: application/json" \\                   ║
    ║       -d '{"add": ["yaar", "bas", "theek"]}'                   ║
    ║                                                                ║
    ║  # Remove fillers                                              ║
    ║  curl -X POST http://localhost:8080/update_filler \\           ║
    ║       -H "Content-Type: application/json" \\                   ║
    ║       -d '{"remove": ["okay", "ok"]}'                          ║
    ║                                                                ║
    ║  # Add and remove simultaneously                               ║
    ║  curl -X POST http://localhost:8080/update_filler \\           ║
    ║       -H "Content-Type: application/json" \\                   ║
    ║       -d '{"add": ["arre"], "remove": ["yeah"]}'               ║
    ╚════════════════════════════════════════════════════════════════╝
    """)
    
    # Start the agent session
    session.start(ctx.room, participant)
    
    logger.info("Agent started. You can now:")
    logger.info("  1. Talk to the agent in the room")
    logger.info("  2. Use the REST API to update filler words dynamically")
    logger.info("  3. Monitor logs for [IGNORED_FILLER] and [VALID_INTERRUPT] events")
    
    # Run the session
    await session.wait_for_completion()
    
    logger.info("Session completed")


if __name__ == "__main__":
    """
    To run this example:
    
    1. Install dependencies:
       pip install livekit livekit-agents livekit-plugins-openai livekit-plugins-silero
    
    2. Set environment variables:
       export LIVEKIT_URL=<your-livekit-url>
       export LIVEKIT_API_KEY=<your-api-key>
       export LIVEKIT_API_SECRET=<your-api-secret>
       export OPENAI_API_KEY=<your-openai-key>
    
    3. Run the agent:
       python filler_api_example.py start
    
    4. In another terminal, test the API:
       
       # Get current fillers
       curl http://localhost:8080/fillers
       
       # Add Hindi fillers
       curl -X POST http://localhost:8080/update_filler \
            -H "Content-Type: application/json" \
            -d '{"add": ["yaar", "bas", "theek", "arre"]}'
       
       # Remove English fillers
       curl -X POST http://localhost:8080/update_filler \
            -H "Content-Type: application/json" \
            -d '{"remove": ["yeah", "yep", "okay"]}'
       
       # Check updated fillers
       curl http://localhost:8080/fillers
    """
    
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
        ),
    )
