import logging
import os

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    WorkerType,
    JobRequest,
    cli,
)
from livekit.plugins import dataspike, openai

logger = logging.getLogger("dataspike-deepfake-example")
logger.setLevel(logging.DEBUG)

load_dotenv()


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session = AgentSession(
        llm=openai.realtime.RealtimeModel(voice="alloy"),
        resume_false_interruption=False,
    )

    logger.info("starting dataspike detector")
    dataspike_detector = dataspike.DataspikeDetector()
    await dataspike_detector.start(session, room=ctx.room)

    # start the agent, it will join the room
    logger.info("starting agent")
    await session.start(
        agent=Agent(instructions="Talk to me!"),
        room=ctx.room,
    )


async def on_request(req: JobRequest):
    logger.info(f"[worker] job request for room={req.room.name} id={req.id}")
    # you can set the agent's participant name/identity here:
    await req.accept(name="dataspike-agent")


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            request_fnc=on_request,
            worker_type=WorkerType.ROOM,
        )
    )
