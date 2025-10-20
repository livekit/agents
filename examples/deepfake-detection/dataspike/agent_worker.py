import logging
import os

from dotenv import load_dotenv

from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, WorkerType, cli
from livekit.plugins import dataspike

logger = logging.getLogger("dataspike-deepfake-example")
logger.setLevel(logging.INFO)

load_dotenv()


async def entrypoint(ctx: JobContext):
    session = AgentSession(
        llm=openai.realtime.RealtimeModel(voice="alloy"),
        resume_false_interruption=False,
    )

    dataspike_api_key = os.getenv("DATASPIKE_API_KEY")
    if not dataspike_api_key:
        raise ValueError("DATASPIKE_API_KEY is not set")

    dataspike_detector = dataspike.DataspikeDetector(
        api_key=dataspike_api_key,
        conn_options=APIConnectOptions(timeout=10),
    )
    await dataspike_detector.start(session, room=ctx.room)
   
    # start the agent, it will join the room
    await session.start(
        agent=Agent(instructions="Talk to me!"),
        room=ctx.room,
    )

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, worker_type=WorkerType.ROOM))
