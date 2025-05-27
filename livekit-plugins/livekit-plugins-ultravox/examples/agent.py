import datetime
import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    function_tool,
)
from livekit.plugins import silero
from livekit.plugins.ultravox.realtime import RealtimeModel

logger = logging.getLogger("ultravox-agent")
logger.setLevel(logging.INFO)

load_dotenv()


@function_tool
def get_time() -> str:
    return datetime.now().isoformat()


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session = AgentSession(
        allow_interruptions=True,
        vad=ctx.proc.userdata["vad"],
        llm=RealtimeModel(
            model_id="fixie-ai/ultravox",
        ),
    )
    await session.start(
        agent=Agent(instructions="You are a helpful assistant.", tools=[get_time]), room=ctx.room
    )


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
