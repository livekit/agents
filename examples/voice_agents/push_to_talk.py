import logging

from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import Agent, AgentSession, JobContext, JobRequest, RoomIO, WorkerOptions, cli
from livekit.agents.llm import ChatContext, ChatMessage, StopResponse
from livekit.plugins import cartesia, deepgram, openai

logger = logging.getLogger("push-to-talk")
logger.setLevel(logging.INFO)

load_dotenv()

## This example demonstrates how to use the push-to-talk for multi-participant
## conversations with a voice agent
## It disables audio input by default, and only enables it when the client explicitly
## triggers the `start_turn` RPC method


class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a helpful assistant.",
            stt=deepgram.STT(),
            llm=openai.LLM(model="gpt-4o-mini"),
            tts=cartesia.TTS(),
            # llm=openai.realtime.RealtimeModel(voice="alloy", turn_detection=None),
        )

    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        # callback before generating a reply after user turn committed
        if not new_message.text_content:
            # for example, raise StopResponse to stop the agent from generating a reply
            logger.info("ignore empty user turn")
            raise StopResponse()


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session = AgentSession(turn_detection="manual")
    room_io = RoomIO(session, room=ctx.room)
    await room_io.start()

    agent = MyAgent()
    await session.start(agent=agent)

    # disable input audio at the start
    session.input.set_audio_enabled(False)

    @ctx.room.local_participant.register_rpc_method("start_turn")
    async def start_turn(data: rtc.RpcInvocationData):
        session.interrupt()
        session.clear_user_turn()

        # listen to the caller if multi-user
        room_io.set_participant(data.caller_identity)
        session.input.set_audio_enabled(True)

    @ctx.room.local_participant.register_rpc_method("end_turn")
    async def end_turn(data: rtc.RpcInvocationData):
        session.input.set_audio_enabled(False)
        session.commit_user_turn(
            # the timeout for the final transcript to be received after committing the user turn
            # increase this value if the STT is slow to respond
            transcript_timeout=10.0,
        )

    @ctx.room.local_participant.register_rpc_method("cancel_turn")
    async def cancel_turn(data: rtc.RpcInvocationData):
        session.input.set_audio_enabled(False)
        session.clear_user_turn()
        logger.info("cancel turn")


async def handle_request(request: JobRequest) -> None:
    await request.accept(
        identity="ptt-agent",
        # this attribute communicates to frontend that we support PTT
        attributes={"push-to-talk": "1"},
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, request_fnc=handle_request))
