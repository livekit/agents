import asyncio
import logging
import random
from datetime import datetime

from dotenv import load_dotenv

from livekit.agents.beta.workflows import GetEmailTask

try:
    from ddgs import DDGS
except ImportError as e:
    raise ImportError(
        "ddgs (duckduckgo search) is required for this example. Install it with: pip install ddgs"
    ) from e

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    RunContext,
    ToolExecutionUpdatedEvent,
    cli,
    inference,
    llm,
)
from livekit.agents.utils import aio

logger = logging.getLogger("async-travel-helper")

_annoying_loggers = [
    "h2",
    "rustls",
    "hyper_util",
    "cookie_store",
    "primp",
    "hpack",
    "hickory_net",
    "hickory_dns",
    "hickory_resolver",
    "hickory_proto",
    "reqwest",
    "ddgs.ddgs",
]
for name in _annoying_loggers:
    logging.getLogger(name).setLevel(logging.WARNING)

load_dotenv()


class TravelAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a friendly travel assistant that communicates via voice. "
                "Avoid emojis and markdown — speak naturally and concisely. "
                "You can help with two things: booking flights and recommending what "
                "to see, eat, and do at a destination. "
                "Use the book_flight tool when the user wants to book a flight. "
                "Use the tour_guide tool when the user asks about places to visit, "
                "restaurants, sightseeing, nightlife, or things to do somewhere. "
                "Summarize the results in a concise and natural manner that suitable for voice communication. "
                f"Today is {datetime.now().strftime('%Y-%m-%d %A')}. "
                "When user is not asking, don't repeat the messages you have already said in the conversation. "
                "Don't make up flight details or ask for flight preferences — always use the tools. "
            ),
        )
        self._thinking_llm = inference.LLM(
            "openai/gpt-5.4", extra_kwargs={"reasoning_effort": "medium"}
        )
        self._ddgs = DDGS()
        self._user_email: str | None = None

    async def on_enter(self):
        self.session.generate_reply(instructions="Greet the user and introduce yourself.")

    @llm.function_tool(flags=llm.ToolFlag.CANCELLABLE, on_duplicate="confirm")
    async def book_flight(self, ctx: RunContext, origin: str, destination: str, date: str) -> str:
        """Called when user wants to book a flight.

        Args:
            origin: Departure city or airport code.
            destination: Arrival city or airport code.
            date: Travel date (e.g. "2026-04-15").
        """
        # This update is delivered immediately — the agent will say something like:
        # "Sure, let me search for flights from New York to Tokyo on April 15th.
        #  This might take a couple of minutes, hang tight!"
        await ctx.update(
            f"Searching flights from {origin} to {destination} on {date}. "
            "This will take a couple of minutes."
        )

        # Phase 1: searching airlines
        await asyncio.sleep(30)

        airlines = random.sample(
            ["United", "Delta", "American", "JetBlue", "Southwest", "Alaska"], k=3
        )
        prices = {a: random.randint(180, 650) for a in airlines}
        cheapest = min(prices, key=lambda a: prices[a])

        logger.info("Found airlines and prices, booking the flight...")
        # This update is delivered when the agent is idle — the agent will say something like:
        # "Good news, I found 3 options. The best price is $289 on Delta.
        #  I'm confirming that booking for you now."
        await ctx.update(
            f"Found {len(airlines)} options. Best price: ${prices[cheapest]} on {cheapest}. "
            "Confirming the booking now.",
        )

        # Phase 2: confirming booking
        if self._user_email is None:
            logger.info("Getting user's email address")
            async with ctx.foreground():
                # collect user input in an async tool
                email = await GetEmailTask(
                    extra_instructions="Tell user we need the email addrss to confirm "
                    "the flight booking when you first ask for it."
                )
            self._user_email = email.email_address
            logger.info(f"User's email address: {self._user_email}")
            await ctx.update(
                "Thanks for providing your email address, we are confirming the booking now."
            )

        await asyncio.sleep(40)

        logger.info("Flight booked")
        confirmation = f"FL-{random.randint(100000, 999999)}"
        # The final return value is also delivered when the agent is idle — it will say:
        # "All done! Your Delta flight from New York to Tokyo on April 15th is booked.
        #  It was $289 and your confirmation number is FL-847293."
        return (
            f"Flight booked! {cheapest} from {origin} to {destination} on {date}. "
            f"Price: ${prices[cheapest]}. Confirmation: {confirmation}. "
            f"The details will be sent to your email."
        )

    # -- Tool 2: Tour guide via web search --

    @llm.function_tool(flags=llm.ToolFlag.CANCELLABLE, on_duplicate="confirm")
    async def tour_guide(self, ctx: RunContext, destination: str, interests: str) -> str:
        """Called when user wants to know about a destination, including
        sightseeing spots, restaurants, local food, nightlife, or neighborhood tips.

        Args:
            destination: The city or area the user is visiting.
            interests: What the user is interested in (e.g. "street food and temples", "museums and nightlife", "family-friendly activities").
        """
        await ctx.update(f"Looking up the best spots in {destination} for you.")

        sources = await self._search(destination, interests, chat_ctx=ctx.session.history)

        if not sources:
            return f"Could not find information about {destination}."

        logger.info(f"Found {len(sources)} sources for tour guide")

        tips = await self._summarize(destination, interests, sources, chat_ctx=ctx.session.history)
        logger.info(f"Tour guide tips ready for {destination}")
        return tips

    async def _search(
        self, destination: str, interests: str, chat_ctx: llm.ChatContext
    ) -> list[dict]:
        logger.info(f"Searching: {destination} — {interests}")
        plan_ctx = chat_ctx.copy(exclude_function_call=True, exclude_instructions=True)
        plan_ctx.add_message(
            role="system",
            content=(
                "You are a travel research assistant. Output 3-4 web search queries "
                f"to find the best places to visit, eat, and explore in {destination} "
                f"for someone interested in: {interests}. "
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
        interests: str,
        sources: list[dict],
        chat_ctx: llm.ChatContext,
    ) -> str:
        summary_ctx = chat_ctx.copy(exclude_function_call=True, exclude_instructions=True)
        source_text = "\n\n".join(f"- {s.get('title', '')}: {s.get('body', '')}" for s in sources)
        summary_ctx.add_message(
            role="system",
            content=(
                f"You are a local tour guide for {destination}. The user is interested in: "
                f'"{interests}". Based on the search results below, recommend specific '
                f"places to visit, restaurants to try, and things to do. "
                f"Be specific — give actual names and neighborhoods. "
                f"This will be spoken aloud, so keep it conversational and brief — "
                f"3 to 5 top picks, no more than 200 words. No bullet points or markdown.\n\n"
                f"Search results:\n{source_text}"
            ),
        )

        response = await self._thinking_llm.chat(chat_ctx=summary_ctx).collect()
        return response.text


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        stt=inference.STT("deepgram/nova-3"),
        # llm=inference.LLM("openai/gpt-5.3-chat-latest"),
        llm=inference.LLM("google/gemini-3.1-flash-lite"),
        tts=inference.TTS("cartesia/sonic-3", voice="e07c00bc-4134-4eae-9ea4-1a55fb45746b"),
        # llm=google.realtime.RealtimeModel(),
        turn_handling={"interruption": {"mode": "vad"}},
    )

    # stream tool execution (calls, ctx.update progress, reply lifecycle) to the frontend
    status_ch = aio.Chan[ToolExecutionUpdatedEvent]()

    @session.on("tool_execution_updated")
    def _on_tool_status(ev: ToolExecutionUpdatedEvent) -> None:
        status_ch.send_nowait(ev)

    async def _publish_status() -> None:
        async for ev in status_ch:
            await ctx.room.local_participant.publish_data(ev.model_dump_json(), topic="tool_status")

    publish_task = asyncio.create_task(_publish_status())

    async def _close_status() -> None:
        status_ch.close()
        await aio.cancel_and_wait(publish_task)

    ctx.add_shutdown_callback(_close_status)

    await session.start(agent=TravelAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
