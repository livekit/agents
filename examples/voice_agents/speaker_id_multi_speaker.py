"""
Example of speaker ID using Speechmatics STT provider.
"""

import datetime

from dotenv import load_dotenv

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli
from livekit.agents.stt import MultiSpeakerAdapter
from livekit.plugins import deepgram, openai, silero, speechmatics  # noqa: F401

# Load environment variables from .env file
# Required: SPEECHMATICS_API_KEY, OPENAI_API_KEY
load_dotenv()


# This example demonstrates how to use the MultiSpeakerAdapter with STT that supports diarization.
# It works for a single audio track, and it will detect the primary speaker and suppress the background speaker.
# It can also be used to format the transcript differently for the primary and background speakers.

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


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    session = AgentSession(
        vad=silero.VAD.load(),
        llm=openai.LLM(),
        tts=openai.TTS(),
        stt=MultiSpeakerAdapter(
            stt=speechmatics.STT(enable_diarization=True, end_of_utterance_silence_trigger=0.3),
            # stt=deepgram.STT(model="nova-3", enable_diarization=True),
            detect_primary_speaker=True,
            suppress_background_speaker=True,  # set to True to suppress background speaker
            primary_format="<{speaker_id}>{text}</{speaker_id}>",
            background_format="<{speaker_id}>{text}</{speaker_id}>",
        ),
        min_interruption_words=3,  # require transcripts to interrupt the agent
    )

    await session.start(room=ctx.room, agent=Assistant())

    # Generate an initial greeting to start the conversation
    await session.generate_reply(
        instructions="Kick off with a friendly Hello and get the conversation started!"
    )


if __name__ == "__main__":
    cli.run_app(server)
