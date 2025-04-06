import logging

from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import Agent, AgentSession, JobContext, RoomIO, WorkerOptions, cli
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
        )
        self.should_answer = False

    async def on_end_of_turn(
        self, chat_ctx: ChatContext, new_message: ChatMessage, generating_reply: bool
    ) -> None:
        if not self.should_answer:
            # stops the agent from generating a reply
            raise StopResponse()

        # callback when user input is transcribed
        chat_ctx = chat_ctx.copy()
        chat_ctx.items.append(new_message)
        await self.update_chat_ctx(chat_ctx)
        logger.info("add user message to chat context", extra={"content": new_message.content})


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session = AgentSession(turn_detection="manual")
    room_io = RoomIO(session, room=ctx.room)
    agent = MyAgent()
    await room_io.start()
    await session.start(agent=agent)

    # disable input audio at the start
    session.input.set_audio_enabled(False)

    @ctx.room.local_participant.register_rpc_method("start_turn")
    async def start_turn(data: rtc.RpcInvocationData):
        agent.should_answer = True
        session.interrupt()

        # listen to the caller if multi-user
        room_io.set_participant(data.caller_identity)
        session.input.set_audio_enabled(True)

    @ctx.room.local_participant.register_rpc_method("end_turn")
    async def end_turn(data: rtc.RpcInvocationData):
        session.input.set_audio_enabled(False)
        session.generate_reply()

    @ctx.room.local_participant.register_rpc_method("cancel_turn")
    async def cancel_turn(data: rtc.RpcInvocationData):
        agent.should_answer = False
        session.input.set_audio_enabled(False)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
