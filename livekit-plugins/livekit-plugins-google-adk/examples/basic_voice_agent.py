"""
LiveKit Google ADK Plugin - Basic Voice Agent Example

This example demonstrates how to create a LiveKit voice agent using Google ADK
for LLM orchestration and tool execution.

Prerequisites:
1. Start your ADK server: `adk api_server`
2. Configure your ADK application with the name "orchestrator"
3. Set up Google Cloud credentials for STT/TTS (or use other providers)

Architecture:
- LiveKit handles: Voice session management, STT (speech-to-text), TTS (text-to-speech)
- ADK handles: LLM responses, orchestration, tool execution, multi-agent coordination
- This plugin bridges: Converts LiveKit text → ADK → streams back to LiveKit

Important Notes:
- instructions="" is intentional - ADK manages prompts internally, not LiveKit
- Tools are configured in ADK, not passed through LiveKit
- Session management is handled by ADK based on user_id

Usage:
    python basic_voice_agent.py dev
"""

import asyncio
from pathlib import Path

from dotenv import load_dotenv

from livekit.agents import Agent, JobContext, WorkerOptions, cli
from livekit.agents.voice import AgentSession
from livekit.plugins import google, google_adk, silero

load_dotenv(dotenv_path=Path(__file__).parent / ".env.example")

# ADK Server Configuration
API_BASE_URL = "http://127.0.0.1:8000"  # ADK local server
APP_NAME = "orchestrator"  # Must match your ADK application name
USER_ID = "user_123"  # Unique user identifier for session management


async def entrypoint(ctx: JobContext):
    """
    Main entrypoint for the LiveKit agent.

    This creates a voice agent that:
    1. Uses Google STT to convert speech to text
    2. Sends text to ADK for LLM processing and orchestration
    3. Converts ADK's text responses to speech via Google TTS
    4. Uses Silero VAD for voice activity detection
    """
    await ctx.connect()
    print(f"Connected to room: {ctx.room.name}")

    session = AgentSession()
    adk_agent = Agent(
        # NOTE: instructions are NOT passed to ADK
        # ADK handles prompts internally through its own configuration
        instructions="",
        llm=google_adk.LLM(
            api_base_url=API_BASE_URL,
            app_name=APP_NAME,
            user_id=USER_ID,
        ),
        stt=google.STT(),
        tts=google.TTS(),
        vad=silero.VAD.load(),
    )

    await session.start(agent=adk_agent, room=ctx.room)
    session.say("Hello! Welcome to the ADK agent.")

    # Keep the agent running
    await asyncio.Future()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
