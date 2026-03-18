"""Async tool example: deep research agent.

The agent uses a fast LLM for conversation. When the user asks a research
question, the deep_research tool runs in the background: it uses a thinking
LLM to plan search queries, searches the web via DuckDuckGo, then summarizes
the findings. The agent keeps chatting while the research runs.
"""

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

_annoying_loggers = ["h2", "rustls", "hyper_util", "cookie_store"]
for name in _annoying_loggers:
    logging.getLogger(name).setLevel(logging.WARNING)

load_dotenv()


async def _web_search(ddgs: DDGS, query: str, max_results: int = 3) -> list[dict]:
    """Search the web using DuckDuckGo."""
    return await asyncio.to_thread(ddgs.text, query, max_results=max_results)


class ResearchToolset(AsyncToolset):
    def __init__(self) -> None:
        super().__init__(id="research")
        self._thinking_llm: llm.LLM = openai.LLM(model="gpt-5.4", reasoning_effort="medium")
        self._ddgs = DDGS()

    @llm.function_tool
    async def deep_research(self, ctx: AsyncRunContext, question: str) -> str:
        """Research a question thoroughly using web search and analysis.
        Use this when the user asks something that requires up-to-date
        information or in-depth research.

        Args:
            question: The research question to investigate.
        """
        speech_handle = ctx.pending(
            f"Starting deep research on: {question}. The task is running in background."
            "Notify the user shortly without giving the actual answer."
        )

        # Step 1: Use the conversation history + thinking LLM to plan search queries
        logger.info(f"Planning search queries for: {question}")
        plan_ctx = ctx.session.history.copy(exclude_function_call=True, exclude_instructions=True)
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

        # Step 2: Run web searches
        all_results = []
        for query in queries[:5]:
            results = await _web_search(self._ddgs, query)
            all_results.append(results)

        sources = []
        for query, results in zip(queries, all_results, strict=False):
            for r in results:
                sources.append(f"[{query}] {r.get('title', '')}: {r.get('body', '')}")

        if not sources:
            return f"Could not find relevant information about: {question}"

        logger.info(f"Found {len(sources)} sources, summarizing...")
        await speech_handle  # wait for the pervious generation to finish
        speech_handle = ctx.update(
            "Web searching finished, analyzing and summarizing. "
            "Give a short and concise notification to the user. Don't give anything else except the notification.",
            role="assistant",
        )

        # Step 3: Use conversation history + search results with thinking LLM to synthesize
        summary_ctx = ctx.session.history.copy(
            exclude_function_call=True, exclude_instructions=True
        )
        summary_ctx.add_message(
            role="system",
            content=(
                "You are a research analyst. Based on the conversation context "
                "and the search results below, provide a clear, concise answer "
                "to the user's question. Cite key facts. Keep it under 300 words.\n\n"
                "Search results:\n" + "\n\n".join(sources)
            ),
        )

        summary_response = await self._thinking_llm.chat(chat_ctx=summary_ctx).collect()

        await asyncio.sleep(5)  # simulate a longer response time
        await speech_handle

        logger.info(f"Summary response: {summary_response.text}")

        await speech_handle  # wait for the pervious generation to finish
        return f"Summarized from deep research: {summary_response.text}"


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

    async def llm_node(self, chat_ctx: llm.ChatContext, tools: list[llm.Tool], model_settings):
        import pprint

        print("LLM node")
        pprint.pprint(chat_ctx.to_dict())

        answer = ""
        async for chunk in Agent.default.llm_node(self, chat_ctx, tools, model_settings):
            if isinstance(chunk, str):
                answer += chunk
            elif isinstance(chunk, llm.ChatChunk) and chunk.delta and chunk.delta.content:
                answer += chunk.delta.content
            yield chunk
        print(f"Answer: {answer}")


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        stt=inference.STT("deepgram/nova-3"),
        llm=inference.LLM("openai/gpt-5.3-chat-latest"),  # fast LLM for conversation
        tts=inference.TTS("inworld"),
        vad=silero.VAD.load(),
    )

    await session.start(agent=ResearchAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
