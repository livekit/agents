import argparse
import logging
import sys
from dataclasses import asdict, dataclass
from functools import partial

import httpx
from dotenv import load_dotenv
from livekit import api
from livekit.agents import JobContext, WorkerOptions, WorkerType, cli
from livekit.agents.pipeline import AgentTask, PipelineAgent
from livekit.agents.pipeline.datastream_io import DataStreamOutput
from livekit.agents.pipeline.io import PlaybackFinishedEvent
from livekit.agents.pipeline.room_io import RoomInput, RoomInputOptions
from livekit.plugins import openai

logger = logging.getLogger("avatar-example")
logger.setLevel(logging.INFO)

load_dotenv()


AVATAR_IDENTITY = "avatar_worker"


@dataclass
class AvatarConnectionInfo:
    room_name: str
    url: str  # LiveKit server URL
    token: str  # Token for avatar worker to join


async def launch_avatar_worker(
    ctx: JobContext, avatar_dispatcher_url: str, avatar_identity: str
) -> None:
    """Wait for worker participant to join and start streaming"""
    # create a token for the avatar worker
    # TODO(long): do we need to set agent=True here? in playground if not the video track is not automatically displayed
    token = (
        api.AccessToken()
        .with_identity(avatar_identity)
        .with_name("Avatar Worker")
        .with_grants(api.VideoGrants(room_join=True, room=ctx.room.name, agent=True))
        .with_metadata("avatar_worker")
        .to_jwt()
    )

    logger.info(f"Sending connection info to avatar dispatcher {avatar_dispatcher_url}")
    connection_info = AvatarConnectionInfo(
        room_name=ctx.room.name, url=ctx._info.url, token=token
    )
    async with httpx.AsyncClient() as client:
        response = await client.post(
            avatar_dispatcher_url, json=asdict(connection_info)
        )
        response.raise_for_status()
    logger.info("Avatar worker connected")

    # wait for the remote participant to join
    await ctx.wait_for_participant(identity=avatar_identity)


async def entrypoint(ctx: JobContext, avatar_dispatcher_url: str):
    await ctx.connect()

    agent = PipelineAgent(
        task=AgentTask(
            instructions="Talk to me!",
            llm=openai.realtime.RealtimeModel(),
        )
    )

    room_input = RoomInput(ctx.room, options=RoomInputOptions(audio_sample_rate=24000))
    agent.input.audio = room_input.audio

    ds_output = DataStreamOutput(ctx.room, destination_identity=AVATAR_IDENTITY)
    agent.output.audio = ds_output.audio

    await launch_avatar_worker(ctx, avatar_dispatcher_url, AVATAR_IDENTITY)
    await room_input.wait_for_participant()
    await agent.start()

    @agent.output.audio.on("playback_finished")
    def on_playback_finished(ev: PlaybackFinishedEvent) -> None:
        logger.info(
            "playback_finished",
            extra={
                "playback_position": ev.playback_position,
                "interrupted": ev.interrupted,
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--avatar-url", type=str, default="http://localhost:8089/launch"
    )
    args, remaining_args = parser.parse_known_args()
    print(sys.argv, remaining_args)
    sys.argv = sys.argv[:1] + remaining_args

    # WorkerType.ROOM is the default worker type which will create an agent for every room.
    # You can also use WorkerType.PUBLISHER to create a single agent for all participants that publish a track.
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=partial(entrypoint, avatar_dispatcher_url=args.avatar_url),
            worker_type=WorkerType.ROOM,
        )
    )
