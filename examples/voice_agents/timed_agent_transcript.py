import asyncio
import logging
from collections.abc import AsyncGenerator, AsyncIterable

from dotenv import load_dotenv

from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.agents.voice.agent import ModelSettings
from livekit.agents.voice.io import TimedString
from livekit.plugins import cartesia, deepgram, openai, silero

logger = logging.getLogger("my-worker")
logger.setLevel(logging.INFO)

load_dotenv()


# This example shows how to obtain the timed transcript from the TTS.
# Right now, it's supported for Cartesia and ElevenLabs TTS (word level timestamps)
# and non-streaming TTS with StreamAdapter (sentence level timestamps).


class MyAgent(Agent):
    def __init__(self):
        super().__init__(instructions="You are a helpful assistant.")

        self._closing_task: asyncio.Task[None] | None = None

    async def transcription_node(
        self, text: AsyncIterable[str | TimedString], model_settings: ModelSettings
    ) -> AsyncGenerator[str | TimedString, None]:
        async for chunk in text:
            if isinstance(chunk, TimedString):
                logger.info(f"TimedString: '{chunk}' ({chunk.start_time} - {chunk.end_time})")
            yield chunk


async def entrypoint(ctx: JobContext):
    session = AgentSession(
        stt=deepgram.STT(),
        llm=openai.LLM(),
        tts=cartesia.TTS(word_timestamps=True),
        vad=silero.VAD.load(),
    )

    await session.start(agent=MyAgent(), room=ctx.room)

    session.generate_reply(instructions="say hello to the user")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
