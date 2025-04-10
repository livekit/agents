from dotenv import load_dotenv


import logging
import asyncio
from dataclasses import dataclass
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
    function_tool,
    ToolError,
    RunContext,
)
from livekit.plugins import openai
from duckduckgo_search import DDGS

load_dotenv()

logger = logging.getLogger("web_search")


@dataclass
class AppData:
    ddgs_client: DDGS


@function_tool
async def search_web(ctx: RunContext[AppData], query: str):
    """
    Performs a web search using the DuckDuckGo search engine.

    Args:
        query: The search term or question you want to look up online.
    """
    ddgs_client = ctx.userdata.ddgs_client

    logger.info(f"Searching for {query}")

    # using asyncio.to_thread because the DDGS client is not asyncio compatible
    search = await asyncio.to_thread(ddgs_client.text, query)
    if len(search) == 0:
        raise ToolError("Tell the user that no results were found for the query.")

    return search


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    app_data = AppData(ddgs_client=DDGS())

    agent = Agent(instructions="You are a helpful assistant.", tools=[search_web])
    session = AgentSession(llm=openai.realtime.RealtimeModel(), userdata=app_data)

    await session.start(agent=agent, room=ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
