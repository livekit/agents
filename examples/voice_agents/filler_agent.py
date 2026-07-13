"""Demo: ctx.update() + ctx.with_filler() in a slow tool.

``ctx.update()`` delivers a real status to the LLM; the agent then naturally voices
something like "let me check flights, this might take a couple minutes."

``ctx.with_filler()`` is the acoustic lane — it goes straight through session.say and
only plays during continuous-idle gaps in the conversation. Use it to fill quiet
during opaque work without burning an LLM round-trip.
"""

import asyncio
import logging
import random
from datetime import datetime

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    RunContext,
    cli,
    inference,
    llm,
)
from livekit.agents.llm.async_toolset import AsyncToolset
from livekit.plugins import silero

logger = logging.getLogger("filler-example")
load_dotenv()


class TravelToolset(AsyncToolset):
    def __init__(self) -> None:
        super().__init__(id="travel")

    @llm.function_tool
    async def book_flight(self, ctx: RunContext, origin: str, destination: str, date: str) -> str:
        """Book a flight (mock — ~60 seconds of opaque work).

        Args:
            origin: Departure city or airport code.
            destination: Arrival city or airport code.
            date: Travel date (e.g. "2026-04-15").
        """
        # one real update — the LLM voices a natural intro
        await ctx.update(
            f"Searching flights from {origin} to {destination} on {date}. "
            "This will take a couple of minutes."
        )

        # phase 1: searching — single filler if the user stays quiet for 5s
        async with ctx.with_filler("Still searching, hang on a sec.", delay=5):
            await asyncio.sleep(20)
            airlines = random.sample(["United", "Delta", "American", "JetBlue"], k=3)
            cheapest = airlines[0]
            price = random.randint(180, 650)
        logger.info("Found airlines and prices, booking the flight...")

        # phase 2: confirming — rotating fillers, up to 3 fires
        followups = [
            "Almost there, just confirming.",
            "Still working on it, won't be long.",
            "Hang tight — almost done.",
        ]
        async with ctx.with_filler(
            lambda step: followups[step], delay=5, interval=10, max_steps=len(followups)
        ):
            await asyncio.sleep(40)
            confirmation = f"FL-{random.randint(100000, 999999)}"

        return (
            f"Booked! {cheapest} from {origin} to {destination} on {date}. "
            f"Price ${price}. Confirmation {confirmation}."
        )


class TravelAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a friendly travel assistant that communicates via voice. "
                "Use book_flight to book flights — don't make up details. "
                "Speak naturally and concisely; no markdown. "
                f"Today is {datetime.now().strftime('%Y-%m-%d')}."
            ),
            tools=[TravelToolset()],
        )

    async def on_enter(self):
        self.session.generate_reply(instructions="Greet the user briefly.")


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    session = AgentSession(
        stt=inference.STT("deepgram/nova-3"),
        llm=inference.LLM("openai/gpt-5.3-chat-latest"),
        tts=inference.TTS("cartesia/sonic-3", voice="e07c00bc-4134-4eae-9ea4-1a55fb45746b"),
        vad=silero.VAD.load(),
        turn_handling={"interruption": {"mode": "vad"}},
    )
    await session.start(agent=TravelAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
