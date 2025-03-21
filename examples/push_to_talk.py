import logging
from typing import Coroutine

from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.llm import function_tool
from livekit.agents.llm.chat_context import ChatContext, ChatMessage
from livekit.agents.voice import Agent, AgentSession, RunContext
from livekit.plugins import cartesia, deepgram, openai

logger = logging.getLogger("push-to-talk-example")
logger.setLevel(logging.INFO)

load_dotenv()


class EchoAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are Echo.",
            stt=deepgram.STT(),
            llm=openai.LLM(model="gpt-4o-mini"),
            tts=cartesia.TTS(),
        )

    @function_tool
    async def talk_to_alloy(self, context: RunContext):
        """Called when want to talk to Alloy."""
        return AlloyAgent(), "Transferring you to Alloy."

    async def on_end_of_turn(
        self, chat_ctx: ChatContext, new_message: ChatMessage, generating_reply: bool
    ) -> None:
        chat_ctx = chat_ctx.copy()
        chat_ctx.items.append(new_message)
        await self.update_chat_ctx(chat_ctx)  # TODO(long): this is a little bit annoying for LLM

        logger.info("add user message to chat context", extra={"content": new_message.content})


class AlloyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are Alloy.",
            llm=openai.realtime.RealtimeModel(voice="alloy", turn_detection=None),
        )

    @function_tool
    async def talk_to_echo(self, context: RunContext):
        """Called when want to talk to Echo."""
        return EchoAgent(), "Transferring you to Echo."


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session = AgentSession(turn_detection="manual")

    await session.start(
        agent=EchoAgent(),
        room=ctx.room,
    )

    # disable input audio if in PPT mode
    session.input.set_audio_enabled(False)

    @ctx.room.local_participant.register_rpc_method("push")
    async def on_pushed(data: rtc.RpcInvocationData):
        session.interrupt()
        session.input.set_audio_enabled(True)

    @ctx.room.local_participant.register_rpc_method("talk")
    async def on_talk(data: rtc.RpcInvocationData):
        session.generate_reply()
        session.input.set_audio_enabled(False)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
