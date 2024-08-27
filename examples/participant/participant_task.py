import asyncio
import logging

from livekit import rtc
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, WorkerType, cli
from dataclasses import dataclass

logger = logging.getLogger("my-worker")
logger.setLevel(logging.INFO)

@dataclass
class ParticipantPreWorkResult:
    p: rtc.RemoteParticipant
    some_data: str

async def entrypoint(ctx: JobContext):
    logger.info("starting entrypoint")
    await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL)
    logger.info("connected to the room")

    async def participant_prework(p: rtc.RemoteParticipant):
        # Do something with p.attributes, p.identity, p.metadata, etc.

        # Return some result
        return ParticipantPreWorkResult(p, "Some result, can be anything")

    participant_prework_result = await ctx.add_participant_task(task_fnc=participant_prework, filter_fnc=lambda p: True)

    async def participant_main_task(p: rtc.RemoteParticipant):
        logger.info("Doing work with data: %s", participant_prework_result.some_data)
        await asyncio.sleep(60)
        logger.info("Done with work")

    await ctx.add_participant_task(task_fnc=participant_main_task, identity=participant_prework_result.p.identity)

if __name__ == "__main__":
    # WorkerType.ROOM is the default worker type which will create an agent for every room.
    # You can also use WorkerType.PUBLISHER to create a single agent for all participants that publish a track.
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, worker_type=WorkerType.ROOM))
