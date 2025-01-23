import logging

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, WorkerType, cli
from livekit.agents.pipeline import AgentTask, ChatCLI, PipelineAgent
from livekit.agents.pipeline.io import PlaybackFinishedEvent
from livekit.agents.pipeline.room_io import RoomAudioSink, RoomInput, RoomInputOptions
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
    agent.start()

    # # start a chat inside the CLI
    # chat_cli = ChatCLI(agent)
    # await chat_cli.run()

    room_input = RoomInput(
        ctx.room,
        options=RoomInputOptions(
            subscribe_audio=True,
            subscribe_video=False,
        ),
    )
    audio_output = RoomAudioSink(ctx.room, sample_rate=24000, num_channels=1)

    agent.input.audio = room_input.audio
    agent.output.audio = audio_output

    # TODO: the interrupted flag is not set correctly
    @audio_output.on("playback_finished")
    def on_playback_finished(ev: PlaybackFinishedEvent) -> None:
        logger.info(
            "playback_finished",
            extra={
                "playback_position": ev.playback_position,
                "interrupted": ev.interrupted,
            },
        )

    await room_input.wait_for_participant()
    await audio_output.start()


if __name__ == "__main__":
    # WorkerType.ROOM is the default worker type which will create an agent for every room.
    # You can also use WorkerType.PUBLISHER to create a single agent for all participants that publish a track.
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, worker_type=WorkerType.ROOM))
