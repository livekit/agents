"""Ojin avatar agent example.

This example demonstrates how to use the Ojin avatar plugin with LiveKit Agents.
The Ojin avatar captures TTS audio, sends it to the Ojin WebSocket API, and
publishes synchronized avatar video/audio tracks to the room.

Environment variables required:
    OJIN_API_KEY: Your Ojin API key
    OJIN_CONFIG_ID: Your Ojin avatar configuration ID
    OJIN_WS_URL: (Optional) Ojin WebSocket URL (default: wss://models.ojin.ai/realtime)
    LIVEKIT_URL: LiveKit server URL
    LIVEKIT_API_KEY: LiveKit API key
    LIVEKIT_API_SECRET: LiveKit API secret
    OPENAI_API_KEY: OpenAI API key (for the LLM)
"""

import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    cli,
)
from livekit.plugins import ojin, openai

logger = logging.getLogger("ojin-avatar-example")
logger.setLevel(logging.INFO)

load_dotenv()

server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        llm=openai.realtime.RealtimeModel(voice="ash"),
        resume_false_interruption=False,
    )

    # Create and start the Ojin avatar session
    # avatar.start() replaces normal TTS publishing to avoid dual audio
    ojin_avatar = ojin.AvatarSession()
    await ojin_avatar.start(session, room=ctx.room)

    await session.start(
        agent=Agent(instructions="You are a helpful assistant with an avatar."),
        room=ctx.room,
    )
    session.generate_reply(instructions="say hello to the user")


if __name__ == "__main__":
    cli.run_app(server)
