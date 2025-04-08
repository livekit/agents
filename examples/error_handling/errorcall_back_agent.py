import logging
import os
import pathlib

from dotenv import load_dotenv

from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents._exceptions import APIStatusError
from livekit.agents.llm import LLMStream
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
from livekit.agents.utils.audio import audio_frames_from_file
from livekit.agents.voice import Agent, AgentSession
from livekit.agents.voice.events import ErrorEvent
from livekit.plugins import cartesia, deepgram, silero
from livekit.plugins.openai import LLM

logger = logging.getLogger("my-worker")
logger.setLevel(logging.INFO)

load_dotenv()


"""
This is a custom LLM that simulates an LLM provider being down.
It will raise an error after a predefined number of calls.
"""


class ErrorLLM(LLM):
    def __init__(self, fail_after: int):
        super().__init__()
        self._num_llm_calls = 0
        self._fail_after = fail_after
        logger.info(f"Error LLM initialized with fail_after={self._fail_after}")

    def chat(self, *args, **kwargs):
        if self._num_llm_calls < self._fail_after:
            self._num_llm_calls += 1
            logger.info(f"Normal LLM call #{self._num_llm_calls}")
            return super().chat(*args, **kwargs)
        else:
            logger.info(f"Error LLM call #{self._num_llm_calls}")
            return ErrorLLMStream(
                self,
                chat_ctx=None,
                tools=[],
                conn_options=DEFAULT_API_CONNECT_OPTIONS,
            )


class ErrorLLMStream(LLMStream):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def _run(self):
        raise APIStatusError(  # noqa: B904
            message="This is a test error",
            status_code=500,
            request_id="123",
            body="test",
            retryable=False,
        )


class MyTask(Agent):
    def __init__(self):
        super().__init__(
            instructions="Repeat exactly what you are told.",  # noqa: E501
        )


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    # Create session
    session = AgentSession(
        stt=deepgram.STT(),
        llm=ErrorLLM(fail_after=0),  # pass in a custom LLM that raises an error
        tts=cartesia.TTS(),
        vad=silero.VAD.load(),
    )

    custor_error_audio = os.path.join(pathlib.Path(__file__).parent.absolute(), "error_message.ogg")

    @session.on("error")
    def on_error(ev: ErrorEvent):
        if ev.error.recoverable:
            return
        logger.info(f"Session is closing due to error in {ev.source.__class__.__name__}")
        logger.info(f"Playing error audio file from: {custor_error_audio}")
        session.say(
            "I'm having trouble connecting right now. Let me transfer your call.",
            # If you define a custom audio file, it will play out even if the TTS provider is down.
            audio=audio_frames_from_file(custor_error_audio),
        )
        session.drain()

    # wait for a participant to join the room
    await ctx.wait_for_participant()

    await session.start(agent=MyTask(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
