import logging

from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
)
from livekit.plugins import openai, silero

logger = logging.getLogger("interrupt-example")
logger.setLevel(logging.INFO)

load_dotenv()


class TravelAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a helpful voice assistant. Your goal is to answer "
            "questions about the user's travel history.",
        )

    async def on_enter(self):
        self.session.generate_reply(
            instructions="Greet the user and ask them about their travel history"
        )

    @function_tool
    async def search_travel_history(
        self,
        context: RunContext,
        destination: str,
    ) -> str:
        """Called when the user asks about trips to a specific destination."""
        # Simulate database lookup
        destinations = {
            "paris": "You visited Paris in June 2022 for 5 days. You stayed at Hotel des Arts.",
            "rome": "you visited Rome in March 2023 for 7 days. You stayed at Hotel Artemide.",
            "tokyo": "you visited Tokyo in November 2024 for 6 days. You stayed at Park Hotel.",
        }
        return destinations.get(destination.lower(), f"No record found for {destination}.")

    @function_tool
    async def get_latest_trip(
        self,
        context: RunContext,
    ) -> str:
        """Called when the user asks about their most recent trip."""
        return "Your most recent trip was to Tokyo in November 2024."


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=openai.STT(),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=openai.TTS(voice="alloy"),
    )

    # example of RPC method that uses awaitable interrupt
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

    await session.start(agent=TravelAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
