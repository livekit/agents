import asyncio
import logging
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RunContext,
    WorkerOptions,
    cli,
)
from livekit.plugins import silero


from agent.session_manager import SessionManager
from agent.config import load_config


logger = logging.getLogger("voice_agent")
logging.basicConfig(level=logging.INFO)


async def entrypoint(ctx: JobContext):
    """
    Clean, simple, production-ready LiveKit voice agent.
    """
    await ctx.connect()
    cfg = load_config()

    # Build the agent persona (instructions + tools)
    agent = Agent(
        instructions="You are a helpful voice assistant. Reply concisely.",
        tools=[],
    )

    # Create LiveKit session
    session = AgentSession(
        vad=silero.VAD.load(),
        stt="deepgram/nova-3",
        llm="openai/gpt-4.1-mini",
        tts="cartesia/sonic-2:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
    )

    # Wrap with our session manager (tracks speaking state, attaches handlers)
    sm = SessionManager(
        session=session,
        agent=agent,
        config=cfg,
    )

    await sm.initialize(ctx.room)

    # Initial greeting
    await sm.say("Hello! You can talk to me anytime.")

    # Keep running until worker shuts down
    await asyncio.Event().wait()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))

