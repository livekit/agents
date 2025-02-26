import logging

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import AgentState, JobContext, WorkerOptions, WorkerType, cli
from livekit.agents.llm import ai_function
from livekit.agents.pipeline import AgentTask, CallContext, PipelineAgent
from livekit.agents.pipeline.io import PlaybackFinishedEvent
from livekit.agents.pipeline.room_io import RoomInputOptions, RoomOutputOptions
from livekit.plugins import cartesia, deepgram, openai

logger = logging.getLogger("roomio-example")
logger.setLevel(logging.INFO)

load_dotenv()


class EchoTask(AgentTask):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are Echo.",
            # llm=openai.realtime.RealtimeModel(voice="echo"),
            stt=deepgram.STT(),
            llm=openai.LLM(model="gpt-4o-mini"),
            tts=cartesia.TTS(),
        )

    @ai_function
    async def talk_to_alloy(self, context: CallContext):
        return AlloyTask(), "Transferring you to Alloy."


class AlloyTask(AgentTask):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are Alloy.",
            llm=openai.realtime.RealtimeModel(voice="alloy"),
        )

    @ai_function
    async def talk_to_echo(self, context: CallContext):
        return EchoTask(), "Transferring you to Echo."


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    agent = PipelineAgent(
        task=AlloyTask(),
    )

    @agent.on("agent_state_changed")
    def on_agent_state_changed(state: AgentState):
        logger.info("agent_state_changed", extra={"state": state})

    await agent.start(
        room=ctx.room,
        # optionally override the default input and output options
        room_input_options=RoomInputOptions(audio_enabled=True, text_enabled=True),
        room_output_options=RoomOutputOptions(
            audio_enabled=True,
            text_enabled=True,
            agent_text_sync_with_audio=True,
        ),
    )

    @ctx.room.local_participant.register_rpc_method("toggle_input_audio")
    async def toggle_input_audio(data: rtc.RpcInvocationData) -> None:
        enable = data.payload.lower() == "true"
        agent.input.toggle_audio(enable)

    @ctx.room.local_participant.register_rpc_method("toggle_output_audio")
    async def toggle_output_audio(data: rtc.RpcInvocationData) -> None:
        enable = data.payload.lower() == "true"
        agent.output.toggle_audio(enable)

        if agent.room_io:
            text_sync_enable = agent.output.audio and agent.output.text
            agent.room_io.toggle_text_audio_sync(text_sync_enable)

    @ctx.room.local_participant.register_rpc_method("toggle_output_text")
    async def toggle_output_text(data: rtc.RpcInvocationData) -> None:
        enable = data.payload.lower() == "true"
        agent.output.toggle_text(enable)

        if agent.room_io:
            audio_sync_enable = agent.output.audio and agent.output.text
            agent.room_io.toggle_text_audio_sync(audio_sync_enable)

    def on_playback_finished(ev: PlaybackFinishedEvent) -> None:
        logger.info(
            "playback_finished",
            extra={
                "playback_position": ev.playback_position,
                "interrupted": ev.interrupted,
            },
        )

    if agent.output.audio is not None:
        agent.output.audio.on("playback_finished", on_playback_finished)


if __name__ == "__main__":
    # WorkerType.ROOM is the default worker type which will create an agent for every room.
    # You can also use WorkerType.PUBLISHER to create a single agent for all participants that publish a track.
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, worker_type=WorkerType.ROOM))
