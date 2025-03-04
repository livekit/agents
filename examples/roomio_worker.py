import logging

from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import AgentState, JobContext, WorkerOptions, WorkerType, cli
from livekit.agents.llm import ai_function
from livekit.agents.voice import AgentTask, CallContext, VoiceAgent
from livekit.agents.voice.io import PlaybackFinishedEvent
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

    agent = VoiceAgent(
        task=AlloyTask(),
    )

    @agent.on("agent_state_changed")
    def on_agent_state_changed(state: AgentState):
        logger.info("agent_state_changed", extra={"state": state})

    await agent.start(room=ctx.room)

    @ctx.room.local_participant.register_rpc_method("toggle_audio_input")
    async def toggle_audio_input(data: rtc.RpcInvocationData) -> None:
        enable = data.payload.lower() == "on"
        if agent.input.set_audio_enabled(enable):
            logger.info("toggled audio input", extra={"enable": enable})

    @ctx.room.local_participant.register_rpc_method("toggle_audio_output")
    async def toggle_audio_output(data: rtc.RpcInvocationData) -> None:
        enable = data.payload.lower() == "on"
        if agent.output.set_audio_enabled(enable):
            logger.info("toggled audio output", extra={"enable": enable})

    @ctx.room.local_participant.register_rpc_method("toggle_transcription_output")
    async def toggle_transcription_output(data: rtc.RpcInvocationData) -> None:
        enable = data.payload.lower() == "on"
        if agent.output.set_transcription_enabled(enable):
            logger.info("toggled transcription output", extra={"enable": enable})

    if agent.output.audio is not None:

        def on_playback_finished(ev: PlaybackFinishedEvent) -> None:
            logger.info(
                "playback_finished",
                extra={
                    "playback_position": ev.playback_position,
                    "interrupted": ev.interrupted,
                },
            )

        agent.output.audio.on("playback_finished", on_playback_finished)


if __name__ == "__main__":
    # WorkerType.ROOM is the default worker type which will create an agent for every room.
    # You can also use WorkerType.PUBLISHER to create a single agent for all participants that publish a track.
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, worker_type=WorkerType.ROOM))
