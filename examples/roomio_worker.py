import logging

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, WorkerType, cli
from livekit.agents.pipeline import AgentTask, ChatCLI, PipelineAgent
from livekit.agents.pipeline.io import PlaybackFinishedEvent
from livekit.agents.pipeline.room_io import RoomOutput, RoomInput, RoomInputOptions
from livekit.plugins import cartesia, deepgram, openai, silero

logger = logging.getLogger("my-worker")
logger.setLevel(logging.INFO)

load_dotenv()


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    agent = PipelineAgent(
        task=AgentTask(
            instructions="Talk to me!",
            llm=openai.realtime.RealtimeModel(),
        )
    )

    # default use RoomIO if room is provided
    await agent.start(room=ctx.room)

    # # Or use RoomInput and RoomOutput explicitly
    # room_input = RoomInput(
    #     ctx.room,
    #     options=RoomInputOptions(
    #         audio_enabled=True,
    #         video_enabled=False,
    #     ),
    # )
    # room_output = RoomOutput(ctx.room, sample_rate=24000, num_channels=1)

    # agent.input.audio = room_input.audio
    # agent.output.audio = room_output.audio

    # await room_input.wait_for_participant()
    # await room_output.start()

    # TODO: the interrupted flag is not set correctly
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
    # WorkerType.ROOM is the default worker type which will create an agent for every room.
    # You can also use WorkerType.PUBLISHER to create a single agent for all participants that publish a track.
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, worker_type=WorkerType.ROOM))
