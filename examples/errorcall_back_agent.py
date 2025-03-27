import logging

from dotenv import load_dotenv

from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.llm import function_tool
from livekit.agents.voice import Agent, AgentSession
from livekit.agents.voice.agent_session import ErrorCallbackResult, UnrecoverableErrorInfo
from livekit.plugins import cartesia, deepgram, openai, silero

logger = logging.getLogger("my-worker")
logger.setLevel(logging.INFO)

load_dotenv()

class MyTask(Agent):
    def __init__(self):
        super().__init__(
            instructions="You are a helpful assistant that can answer questions and help with tasks.",  # noqa: E501
        )


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    def unrecoverable_error_callback(error_info: UnrecoverableErrorInfo) -> ErrorCallbackResult:
        logger.info(f"++++++ Unrecoverable error: {error_info.message}")
        ## do something with the error
        return ErrorCallbackResult.END_SESSION

    session = AgentSession(
        stt=deepgram.STT(),
        llm=openai.LLM(),
        tts=cartesia.TTS(),
        vad=silero.VAD.load(),
        unrecoverable_error_callback=unrecoverable_error_callback,
    )
    await session.start(agent=MyTask(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
