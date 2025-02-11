import logging

from dotenv import load_dotenv
from livekit.agents import JobContext, WorkerOptions, WorkerType, cli
from livekit.agents.pipeline import AgentTask, PipelineAgent
from livekit.agents.pipeline.io import PlaybackFinishedEvent
from livekit.agents.pipeline.room_io import RoomInput, RoomInputOptions, RoomOutput
from livekit.plugins import openai

logger = logging.getLogger("roomio-example")
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

    room_input = RoomInput(ctx.room, options=RoomInputOptions(audio_sample_rate=24000))
    room_output = RoomOutput(room=ctx.room, sample_rate=24000, num_channels=1)

    # wait for the participant to join the room and subscribe to the output audio
    await room_input.wait_for_participant()
    await room_output.start()
    
    # connect the input and output audio to the agent
    agent.input.audio = room_input.audio
    agent.output.audio = room_output.audio
    await agent.start()

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
