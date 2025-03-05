import logging

from dotenv import load_dotenv
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.voice import VoiceAgent
from livekit.plugins import openai

logger = logging.getLogger("my-worker")
logger.setLevel(logging.INFO)

load_dotenv()


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    agent = VoiceAgent(
        instructions="You are a helpful assistant that can answer questions and help with tasks.",
        # llm=openai.realtime.RealtimeModel(base_url="http://localhost:1234/v1"),
        llm=openai.realtime.RealtimeModel(),
    )
    await agent.start(room=ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
