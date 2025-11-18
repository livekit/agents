# default_agent.py
from __future__ import annotations

import logging

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# Load .env so LIVEKIT_* and OpenAI etc. are available
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("default_agent")


class BaselineAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a helpful voice AI assistant. "
                "You speak clearly and keep your answers short."
            )
        )


async def entrypoint(ctx: agents.JobContext):
    """
    This is a SIMPLE baseline agent:
    - Uses default turn detection and interruptions.
    - NO custom filler logic.
    - allow_interruptions is left as default (True).
    """
    logger.info("Starting BASELINE session (no custom filler handling)")

    session = AgentSession(
        stt="assemblyai/universal-streaming:en",
        llm="openai/gpt-4.1-mini",
        tts="cartesia/sonic-3:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    await session.start(
        room=ctx.room,
        agent=BaselineAssistant(),
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await session.generate_reply(
        instructions="Greet the user briefly and then pause."
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
