import logging

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import JobContext, WorkerOptions, WorkerType, cli
from livekit.agents.llm import ai_function
from livekit.agents.pipeline import AgentTask, CallContext, PipelineAgent
from livekit.agents.pipeline.io import PlaybackFinishedEvent
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
        task=EchoTask(),
    )

    await agent.start(room=ctx.room)

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

    @ctx.room.local_participant.register_rpc_method("set_participant")
    async def on_set_participant(data: rtc.RpcInvocationData) -> None:
        logger.info(
            "set_participant called",
            extra={"caller_identity": data.caller_identity, "payload": data.payload},
        )
        if not agent.room_input:
            logger.warning("room_input not set, skipping set_participant")
            return

        target_identity = data.payload or data.caller_identity
        agent.room_input.set_participant(target_identity)

    @ctx.room.local_participant.register_rpc_method("unset_participant")
    async def on_unset_participant(data: rtc.RpcInvocationData) -> None:
        logger.info(
            "unset_participant called",
            extra={"caller_identity": data.caller_identity, "payload": data.payload},
        )
        if not agent.room_input:
            logger.warning("room_input not set, skipping unset_participant")
            return

        agent.room_input.set_participant(None)


if __name__ == "__main__":
    # WorkerType.ROOM is the default worker type which will create an agent for every room.
    # You can also use WorkerType.PUBLISHER to create a single agent for all participants that publish a track.
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, worker_type=WorkerType.ROOM))
