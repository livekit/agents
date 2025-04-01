import asyncio
import logging
import os

from dotenv import load_dotenv

from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.voice import Agent, AgentSession, MetricsCollectedEvent
from livekit.agents.metrics import LLMMetrics
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

    session = AgentSession(
        stt=deepgram.STT(),
        llm=openai.LLM(),
        tts=cartesia.TTS(),
        vad=silero.VAD.load(),

    )

    @session.on("metrics_collected")
    def on_metrics_collected(ev: MetricsCollectedEvent):
        if isinstance(ev.metrics, LLMMetrics) and ev.metrics.error:
            if ev.metrics.error and not ev.metrics.error.retryable:
                logger.info("Ran into an unrecoverable LLM error, ending session.")
                # do something with the error
                close_task = asyncio.create_task(session.aclose())
                close_task.add_done_callback(lambda _: logger.info("Session closed"))
        return

    await session.start(agent=MyTask(), room=ctx.room)


if __name__ == "__main__":
    # set the OPENAI_API_KEY to a invalid key to simulate an error
    os.environ["OPENAI_API_KEY"] = "invalid"
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
