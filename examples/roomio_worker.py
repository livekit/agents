import logging

from dotenv import load_dotenv
from livekit.agents import JobContext, WorkerOptions, WorkerType, cli
from livekit.agents.llm import ai_function
from livekit.agents.pipeline import AgentContext, AgentTask, PipelineAgent
from livekit.agents.pipeline.io import PlaybackFinishedEvent
from livekit.agents.pipeline.room_io import RoomInput, RoomOutput, RoomOutputOptions
from livekit.agents.transcription import TranscriptionDataStreamForwarder
from livekit.plugins import cartesia, deepgram, openai

logger = logging.getLogger("roomio-example")
logger.setLevel(logging.INFO)

load_dotenv()


class EchoTask(AgentTask):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are Echo, always speak in English even if the user speaks in another language or wants to use another language.",
            llm=openai.realtime.RealtimeModel(voice="echo"),
        )

    @ai_function
    async def talk_to_alloy(self, context: AgentContext):
        return AlloyTask(), "Transfering you to Alloy."


class AlloyTask(AgentTask):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are Alloy, always speak in English even if the user speaks in another language or wants to use another language.",
            llm=openai.realtime.RealtimeModel(voice="alloy"),
        )

    @ai_function
    async def talk_to_echo(self, context: AgentContext):
        return EchoTask(), "Transfering you to Echo."


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    agent = PipelineAgent(
        task=AlloyTask(),
        # stt=deepgram.STT(),
        # llm=openai.LLM(),
        # tts=cartesia.TTS(),
    )

    room_input = RoomInput(ctx.room)
    room_output = RoomOutput(
        room=ctx.room, options=RoomOutputOptions(sync_transcription=True)
    )

    # wait for the participant to join the room and subscribe to the output audio
    await room_input.wait_for_participant()
    await room_output.start()

    # connect the input and output audio to the agent
    agent.input.audio = room_input.audio
    agent.output.audio = room_output.audio
    agent.output.text = room_output.text

    await agent.start()

    # (optional) forward transcription using data stream
    ds_forwarder = TranscriptionDataStreamForwarder(
        room=ctx.room,
        attributes={"transcription_track": "agent"},
    )
    room_output.on("transcription_segment", ds_forwarder.update)

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
