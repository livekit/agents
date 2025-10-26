import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobRequest,
    WorkerOptions,
    WorkerType,
    cli,
)
from livekit.plugins import dataspike, openai

logger = logging.getLogger("dataspike-deepfake-example")
logger.setLevel(logging.DEBUG)

load_dotenv()


async def entrypoint(ctx: JobContext):
    session = AgentSession(
        llm=openai.realtime.RealtimeModel(voice="alloy"),
        resume_false_interruption=False,
    )

    # Configure the Dataspike detector.
    # - api_key: optional; if omitted, uses DATASPIKE_API_KEY from environment
    # - notification_cb: optional; callback for handling notifications
    #   (defaults to publishing results in the room data channel)
    # - other config: see plugin docs for available parameters
    detector = dataspike.DataspikeDetector(
        # api_key="YOUR_API_KEY",
        # notification_cb=on_notification,
    )

    # Start the Dataspike detector and attach it to the current session and room.
    await detector.start(session, room=ctx.room)

    # Launch the main agent, which will automatically join the same room.
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
