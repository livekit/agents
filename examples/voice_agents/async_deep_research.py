import asyncio
import logging

from dotenv import load_dotenv

try:
    from ddgs import DDGS
except ImportError as e:
    raise ImportError(
        "ddgs (duckduckgo search) is required for this example. Install it with: pip install ddgs"
    ) from e

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli, inference, llm
from livekit.agents.llm.async_toolset import AsyncRunContext, AsyncToolset
from livekit.plugins import openai, silero

logger = logging.getLogger("async-deep-research")

_annoying_loggers = ["h2", "rustls", "hyper_util", "cookie_store", "primp"]
for name in _annoying_loggers:
    logging.getLogger(name).setLevel(logging.WARNING)

load_dotenv()


class ResearchToolset(AsyncToolset):
    def __init__(self) -> None:
        super().__init__(id="research")
        self._thinking_llm: llm.LLM = openai.LLM(model="gpt-5.4", reasoning_effort="medium")
        self._ddgs = DDGS()

    @llm.function_tool
    async def deep_research(self, ctx: AsyncRunContext, question: str) -> str:
        """Research a question thoroughly using web search and analysis.
        Use this when the user asks something that requires up-to-date
        information or in-depth research. Please avoid calling this tool again
        if there is already a running task for similar topic. If there is, notify
        the user that the research is already in progress, or cancel the running
        one before creating a new tool call.

        Args:
            question: The research question to investigate.
        """
        ctx.pending("task started, notify the user shortly without giving the answer.")

        # Step 1: Plan search queries and run web searches
        sources = await self._search(question, chat_ctx=ctx.session.history)

        if not sources:
            return f"Could not find relevant information about: {question}"

        logger.info(f"Found {len(sources)} sources, summarizing...")

        # only push a progress update if the conversation is idle
        ctx.update("Web searching finished, analyzing and summarizing.", role="tool_output")

        # Step 2: Synthesize results
        summary = await self._analyze(question, sources, chat_ctx=ctx.session.history)

        logger.info(f"Summary response: {summary}")
        return f"Summarized from deep research: {summary}"

    async def _search(self, question: str, chat_ctx: llm.ChatContext) -> list[dict]:
        """Run web searches for the given queries."""

        logger.info(f"Planning search queries for: {question}")
        plan_ctx = chat_ctx.copy(exclude_function_call=True, exclude_instructions=True)
        plan_ctx.add_message(
            role="system",
            content=(
                "You are a research planner. Based on the conversation so far, "
                "output 3-4 focused web search queries that would help answer "
                "the user's research question. "
                "Output only the queries, one per line, nothing else."
            ),
        )

        plan_response = await self._thinking_llm.chat(chat_ctx=plan_ctx).collect()
        queries = [q.strip() for q in plan_response.text.strip().split("\n") if q.strip()]
        logger.info(f"Search queries: {queries}")

        all_results: list[dict] = []
        for query in queries[:5]:
            results = await asyncio.to_thread(self._ddgs.text, query, max_results=3)
            all_results.extend(results)
        return all_results

    async def _analyze(self, question: str, sources: list[dict], chat_ctx: llm.ChatContext) -> str:
        """Synthesize search results into a summary, incorporating any new user input."""
        summary_ctx = chat_ctx.copy(exclude_function_call=True, exclude_instructions=True)
        source_text = "\n\n".join(f"- {s.get('title', '')}: {s.get('body', '')}" for s in sources)
        summary_ctx.add_message(
            role="system",
            content=(
                "You are a research analyst. Based on the conversation context "
                "and the search results below, provide a clear, concise answer "
                "to the user's question. Cite key facts. Keep it under 300 words.\n\n"
                f"Search results:\n{source_text}"
            ),
        )

        response = await self._thinking_llm.chat(chat_ctx=summary_ctx).collect()
        return response.text


class ResearchAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a friendly research assistant that communicates via voice. "
                "So please avoid using emojis, markdowns, answer in spoken words shortly and concisely."
                "When the user asks something that needs research or up-to-date info, "
                "use the deep_research tool. "
                "While research runs in the background, you can keep chatting with the user. "
                "Keep your spoken responses concise. Don't answer the questions directly without "
                "using the deep_research tool."
            ),
            tools=[ResearchToolset()],
        )

    async def on_enter(self):
        self.session.generate_reply(
            instructions="Greet the user and let them know you can research any topic for them."
        )


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        stt=inference.STT("deepgram/nova-3"),
        llm=inference.LLM("openai/gpt-4.1-mini"),  # fast LLM for conversation
        tts=inference.TTS("inworld"),
        vad=silero.VAD.load(),
    )

    await session.start(agent=ResearchAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
