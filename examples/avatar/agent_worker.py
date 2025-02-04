import argparse
import logging
import sys
from functools import partial

from dotenv import load_dotenv
from livekit.agents import JobContext, WorkerOptions, WorkerType, cli
from livekit.agents.avatar import AvatarOutput
from livekit.agents.pipeline import AgentTask, PipelineAgent
from livekit.agents.pipeline.io import PlaybackFinishedEvent
from livekit.agents.pipeline.room_io import RoomInput, RoomInputOptions
from livekit.plugins import openai

logger = logging.getLogger("avatar-example")
logger.setLevel(logging.INFO)

load_dotenv()


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

    avatar_output = AvatarOutput(ctx, avatar_dispatcher_url=avatar_dispatcher_url)
    agent.output.audio = avatar_output.audio

    await room_input.wait_for_participant()
    await avatar_output.start()

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
