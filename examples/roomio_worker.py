import logging

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import JobContext, WorkerOptions, WorkerType, cli
from livekit.agents.pipeline import AgentTask, PipelineAgent
from livekit.agents.pipeline.io import PlaybackFinishedEvent, TextSink
from livekit.agents.pipeline.room_io import RoomInputOptions
from livekit.agents.transcription.transcription_sync import TranscriptionSyncIO
from livekit.agents.transcription.tts_forwarder import TTSRoomForwarder
from livekit.plugins import openai

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
    await agent.start(
        room=ctx.room,
        room_input_options=RoomInputOptions(
            audio_enabled=True,
            video_enabled=False,
            audio_sample_rate=24000,
            audio_num_channels=1,
        ),
    )

    # # Or use RoomInput and RoomOutput explicitly
    # room_input = RoomInput(
    #     ctx.room,
    #     options=RoomInputOptions(
    #         audio_enabled=True,
    #         video_enabled=False,
    #         audio_sample_rate=24000,
    #         audio_num_channels=1,
    #     ),
    # )
    # room_output = RoomOutput(ctx.room, sample_rate=24000, num_channels=1)

    # agent.input.audio = room_input.audio
    # agent.output.audio = room_output.audio

    # await room_input.wait_for_participant()
    # await room_output.start()

    # TTS transcription forward
    transcription_sync = TranscriptionSyncIO.from_agent(agent)
    transcription_sync.on(
        "transcription_segment", TTSRoomForwarder(ctx.room, ctx.room.local_participant)
    )

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
