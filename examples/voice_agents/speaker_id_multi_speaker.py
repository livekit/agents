"""
Example of speaker ID using Speechmatics STT provider.
"""

import datetime

from dotenv import load_dotenv

from livekit import agents
from livekit.agents import Agent, AgentSession, RoomInputOptions
from livekit.plugins import openai, silero, speechmatics

# Load environment variables from .env file
# Required: OPENAI_API_KEY, ELEVENLABS_API_KEY, SPEECHMATICS_API_KEY
load_dotenv()


MASTER_PROMPT = """
You are a friendly AI assistant called Lively.

# Conversation:
- Engage in natural, empathetic conversations with one or more speakers.
- Use a fun and snappy tone unless a longer response is requested.
- Include natural hesitations where appropriate.

# Multiple Speakers:
- Different unknown speakers are indicated with `<Sn/>` tags.
- Known speakers are indicated with `<Name/>` tags.
- Use the context of the conversation to establish the names of the unknown speakers.
- Do not include `<Sn/>` or `<Name/>` tags in your responses.

# Context
- The conversation started at {time}.
"""


class Assistant(Agent):
    def __init__(self) -> None:
        formatted_master_prompt = MASTER_PROMPT.format(
            time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        super().__init__(
            instructions=formatted_master_prompt,
        )


async def entrypoint(ctx: agents.JobContext) -> None:
    session = AgentSession(
        stt=speechmatics.STT(
            transcription_config=speechmatics.types.TranscriptionConfig(
                additional_vocab=[
                    {"content": "LiveKit", "sounds_like": ["live kit"]},
                ],
            ),
        ),
        llm=openai.LLM(),
        tts=openai.TTS(),
        vad=silero.VAD.load(),
    )

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=RoomInputOptions(),
    )

    await ctx.connect()

    # Generate an initial greeting to start the conversation
    await session.generate_reply(
        instructions="Kick off with a friendly Hello and get the conversation started!"
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint, agent_name="lively"))
