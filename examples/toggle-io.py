import logging

from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import Agent, AgentSession, JobContext, RoomIO, WorkerOptions, cli
from livekit.plugins import openai

logger = logging.getLogger("toggle-io")
logger.setLevel(logging.INFO)

load_dotenv()

## This example demonstrates a more complex application that allows the user to
## toggle audio and text input/output on the fly.
## The example makes use of LiveKit's RPC system to exchange messages between the
## client and the server.


class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a helpful assistant that interfaces with the user via voice.",
        )


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session = AgentSession(llm=openai.realtime.RealtimeModel())
    room_io = RoomIO(session, room=ctx.room)
    await room_io.start()
    await session.start(agent=MyAgent())

    @ctx.room.local_participant.register_rpc_method("set_participant")
    async def on_set_participant(data: rtc.RpcInvocationData) -> None:
        target_identity = data.payload or data.caller_identity
        logger.info(
            "set participant called",
            extra={
                "caller_identity": data.caller_identity,
                "payload": data.payload,
                "target_identity": target_identity,
            },
        )
        room_io.set_participant(target_identity)
        return "participant set"

    @ctx.room.local_participant.register_rpc_method("unset_participant")
    async def on_unset_participant(data: rtc.RpcInvocationData) -> None:
        logger.info(
            "unset participant called",
            extra={"caller_identity": data.caller_identity, "payload": data.payload},
        )
        room_io.unset_participant()
        return "participant unset"

    @ctx.room.local_participant.register_rpc_method("toggle_input")
    async def on_toggle_input(data: rtc.RpcInvocationData) -> None:
        logger.info(
            "toggle input called",
            extra={"caller_identity": data.caller_identity, "payload": data.payload},
        )
        if data.payload == "audio_on":
            session.input.set_audio_enabled(True)
        elif data.payload == "audio_off":
            session.input.set_audio_enabled(False)
        else:
            return "invalid payload"
        return "success"

    @ctx.room.local_participant.register_rpc_method("toggle_output")
    async def on_toggle_output(data: rtc.RpcInvocationData) -> None:
        logger.info(
            "toggle output called",
            extra={"caller_identity": data.caller_identity, "payload": data.payload},
        )
        if data.payload == "audio_on":
            session.output.set_audio_enabled(True)
        elif data.payload == "audio_off":
            session.output.set_audio_enabled(False)
        elif data.payload == "transcription_on":
            session.output.set_transcription_enabled(True)
        elif data.payload == "transcription_off":
            session.output.set_transcription_enabled(False)
        else:
            return "invalid payload"
        return "success"


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
