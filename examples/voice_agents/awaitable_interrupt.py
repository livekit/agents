import logging

from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
)
from livekit.plugins import openai, silero

logger = logging.getLogger("interrupt-example")
logger.setLevel(logging.INFO)

load_dotenv()


# this example demonstrates how to manually interrupt voice agent.
# session.interrupt() provides a safe way to cancel the current generation. this method
# returns a Future that must be awaited to ensure the chatcontext is fully updated before
# accessing it.


class Agent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a helpful voice assistant",
        )


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=openai.STT(),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=openai.TTS(voice="alloy"),
    )

    @ctx.room.local_participant.register_rpc_method("emergency_interrupt")
    async def on_emergency_interrupt(data: rtc.RpcInvocationData) -> None:
        logger.info("Emergency interrupt requested via RPC")

        # await interrupt to ensure chat context is fully updated
        await session.interrupt()

        # now safely access chat context
        chat_history = session.history
        last_message = next((msg for msg in reversed(chat_history.items)), None)

        if last_message:
            print(f"Interrupted message: {last_message.text_content}")

        session.generate_reply(
            instructions="Acknowledge the emergency interrupt and ask if user needs assistance"
        )

    await session.start(agent=Agent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
