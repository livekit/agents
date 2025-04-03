import logging

from dotenv import load_dotenv

from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents._exceptions import APIStatusError
from livekit.agents.llm import LLM, LLMStream
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
from livekit.agents.voice import Agent, AgentSession
from livekit.agents.voice.events import SessionCloseEvent
from livekit.plugins import cartesia, deepgram, silero

logger = logging.getLogger("my-worker")
logger.setLevel(logging.INFO)

load_dotenv()


"""
Simulate an LLM provider being down.
"""


class ErrorLLM(LLM):
    def __init__(self):
        super().__init__()

    def chat(self, *args, **kwargs):
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
            instructions="You are a helpful assistant that can answer questions and help with tasks.",  # noqa: E501
        )


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    # Create session
    session = AgentSession(
        stt=deepgram.STT(),
        llm=ErrorLLM(),  # pass in a custom LLM that raises an error
        tts=cartesia.TTS(),
        vad=silero.VAD.load(),
    )

    @session.on("session_close")
    def on_session_close(ev: SessionCloseEvent):
        if ev.session_error is not None:
            logger.info(
                f"Session is closing due to error in {
                    ev.session_error.component.__class__.__name__
                }"
            )

    await session.start(agent=MyTask(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
