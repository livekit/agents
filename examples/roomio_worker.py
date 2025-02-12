import logging

from dotenv import load_dotenv
from livekit.agents import JobContext, WorkerOptions, WorkerType, cli
from livekit.agents.llm import ai_function
from livekit.agents.pipeline import AgentContext, AgentTask, PipelineAgent
from livekit.agents.pipeline.io import PlaybackFinishedEvent
from livekit.agents.pipeline.room_io import RoomInput, RoomOutput, RoomOutputOptions
from livekit.agents.transcription import (
    TranscriptionDataStreamForwarder,
    TranscriptionRoomForwarder,
)
from livekit.plugins import cartesia, deepgram, openai

logger = logging.getLogger("roomio-example")
logger.setLevel(logging.INFO)

load_dotenv()


class EchoTask(AgentTask):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are Echo, always speak in English even if the user speaks in another language or wants to use another language.",
            # llm=openai.realtime.RealtimeModel(voice="echo"),
            stt=deepgram.STT(),
            llm=openai.LLM(model="gpt-4o-mini"),
            tts=cartesia.TTS(),
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
        task=EchoTask(),
    )

    room_input = RoomInput(ctx.room)
    room_output = RoomOutput(
        room=ctx.room, options=RoomOutputOptions(sync_transcription=True)
    )

    # set the agent io and wait for the participant to join, subscribe to the output audio
    await room_input.start(agent)
    await room_output.start(agent)

    await agent.start()

    # # (optional) forward transcription using data stream
    # ds_agent_fwd = TranscriptionDataStreamForwarder(
    #     room=ctx.room,
    #     attributes={"track": "agent"},
    # )
    # room_output.on("agent_transcript_updated", ds_agent_fwd.update)

    # ds_user_fwd = TranscriptionDataStreamForwarder(
    #     room=ctx.room,
    #     attributes={"track": "user"},
    # )
    # room_input.on("user_transcript_updated", ds_user_fwd.update)

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
