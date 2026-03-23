import asyncio
import logging
import random
from datetime import datetime

from dotenv import load_dotenv

try:
    from ddgs import DDGS
except ImportError as e:
    raise ImportError(
        "ddgs (duckduckgo search) is required for this example. Install it with: pip install ddgs"
    ) from e

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli, inference, llm
from livekit.agents.llm.async_toolset import AsyncRunContext, AsyncToolset
from livekit.plugins import silero

logger = logging.getLogger("async-travel-helper")

_annoying_loggers = ["h2", "rustls", "hyper_util", "cookie_store", "primp"]
for name in _annoying_loggers:
    logging.getLogger(name).setLevel(logging.WARNING)

load_dotenv()


class TravelToolset(AsyncToolset):
    def __init__(self) -> None:
        super().__init__(id="travel")
        self._thinking_llm = inference.LLM(
            "openai/gpt-5.4", extra_kwargs={"reasoning_effort": "medium"}
        )
        self._ddgs = DDGS()

    # -- Tool 1: Mock flight booking (takes ~2 minutes with progress updates) --

    @llm.function_tool
    async def book_flight(
        self, ctx: AsyncRunContext, origin: str, destination: str, date: str
    ) -> str:
        """Book a flight ticket. This takes a while to search airlines and confirm the booking.

        Args:
            origin: Departure city or airport code.
            destination: Arrival city or airport code.
            date: Travel date (e.g. "2026-04-15").
        """
        ctx.pending(
            f"Searching flights from {origin} to {destination} on {date}. "
            "This will take a couple of minutes."
        )

        # Phase 1: searching airlines (~60s)
        await asyncio.sleep(40)

        airlines = random.sample(
            ["United", "Delta", "American", "JetBlue", "Southwest", "Alaska"], k=3
        )
        prices = {a: random.randint(180, 650) for a in airlines}
        cheapest = min(prices, key=lambda a: prices[a])

        ctx.update(
            f"Found {len(airlines)} options. Best price: ${prices[cheapest]} on {cheapest}. "
            "Confirming the booking now.",
        )

        # Phase 2: confirming booking (~60s)
        await asyncio.sleep(40)

        logger.info("Flight booked")
        confirmation = f"FL-{random.randint(100000, 999999)}"
        return (
            f"Flight booked! {cheapest} from {origin} to {destination} on {date}. "
            f"Price: ${prices[cheapest]}. Confirmation: {confirmation}."
        )

    # -- Tool 2: Travel advice via web search --

    @llm.function_tool
    async def travel_advice(self, ctx: AsyncRunContext, destination: str, question: str) -> str:
        """Look up travel advice for a destination using web research.
        Use this when the user asks about things to do, where to eat,
        safety tips, best time to visit, local customs, or any travel question.

        Args:
            destination: The travel destination (city or region).
            question: The specific travel question to research.
        """
        ctx.pending(f"Researching travel tips for {destination}. Let me look that up.")

        sources = await self._search(destination, question, chat_ctx=ctx.session.history)

        if not sources:
            return f"Could not find travel information for {destination}."

        logger.info(f"Found {len(sources)} sources for travel advice")

        advice = await self._summarize(destination, question, sources, chat_ctx=ctx.session.history)
        logger.info(f"Travel advice ready for {destination}")
        return advice

    async def _search(
        self, destination: str, question: str, chat_ctx: llm.ChatContext
    ) -> list[dict]:
        logger.info(f"Searching travel info: {destination} — {question}")
        plan_ctx = chat_ctx.copy(exclude_function_call=True, exclude_instructions=True)
        plan_ctx.add_message(
            role="system",
            content=(
                "You are a travel research assistant. Output 3-4 web search queries "
                f"to find practical travel advice about {destination}, "
                f"specifically: {question}. "
                "Output only the queries, one per line, nothing else."
            ),
        )

        plan_response = await self._thinking_llm.chat(chat_ctx=plan_ctx).collect()
        queries = [q.strip() for q in plan_response.text.strip().split("\n") if q.strip()]
        logger.info(f"Search queries: {queries}")

        all_results: list[dict] = []
        for query in queries[:4]:
            results = await asyncio.to_thread(self._ddgs.text, query, max_results=3)
            all_results.extend(results)
        return all_results

    async def _summarize(
        self,
        destination: str,
        question: str,
        sources: list[dict],
        chat_ctx: llm.ChatContext,
    ) -> str:
        summary_ctx = chat_ctx.copy(exclude_function_call=True, exclude_instructions=True)
        source_text = "\n\n".join(f"- {s.get('title', '')}: {s.get('body', '')}" for s in sources)
        summary_ctx.add_message(
            role="system",
            content=(
                f"You are a helpful travel advisor. The user asked about {destination}: "
                f'"{question}". Based on the conversation context, '
                "summarize the search results below in no more than 300 words."
                f"Search results:\n{source_text}"
            ),
        )

        response = await self._thinking_llm.chat(chat_ctx=summary_ctx).collect()
        return (
            f"Summarized from search results: {response.text}. "
            "Give a short, conversational answer with the most useful tips."
            f"The response will be spoken aloud, so keep it brief — 2 to 4 key points, "
            f"no bullet points or markdown."
        )


class TravelAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a friendly travel assistant that communicates via voice. "
                "Avoid emojis and markdown — speak naturally and concisely. "
                "You can help with two things: booking flights and giving travel advice. "
                "Use the book_flight tool when the user wants to book a flight. "
                "Use the travel_advice tool when the user asks about destinations, "
                "things to do, local tips, safety, food, or anything travel-related. "
                "Both tools run in the background, so keep chatting while they work. "
                f"Today is {datetime.now().strftime('%Y-%m-%d')}. "
                "Don't make up flight details or travel advice — always use the tools."
            ),
            tools=[TravelToolset()],
        )

    async def on_enter(self):
        self.session.generate_reply(
            instructions=(
                "Greet the user and let them know you can help book flights "
                "and give travel advice. Keep it short."
            )
        )


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        stt=inference.STT("deepgram/nova-3"),
        llm=inference.LLM("openai/gpt-4.1-mini"),
        tts=inference.TTS("inworld"),
        vad=silero.VAD.load(),
        turn_handling={"interruption": {"mode": "vad"}},
    )

    await session.start(agent=TravelAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
