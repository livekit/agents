import asyncio

from livekit.agents import Agent, AgentHandler, RunContext, RunResult
from livekit.agents.a2a import A2ASessionContext

MODEL = "google/gemma-4-31b-it"

agent = Agent(
    instructions=(
        "You look things up for the user. What you say reaches them through a voice "
        "assistant. Use look_up and reply in one plain sentence."
    ),
    llm=MODEL,
)


@agent.tool
async def look_up(ctx: RunContext, topic: str) -> str:
    """Look a topic up in the knowledge base.

    Args:
        topic: what to look up
    """
    ctx.update(f"Searching for '{topic}'.")
    await asyncio.sleep(5)
    return f"found 3 documents about '{topic}'; the most recent is from last week"


lookup = AgentHandler()


@lookup.a2a_session(partition_key="lookup", endpoint="lookup", description="Looks things up.")
async def endpoint(ctx: A2ASessionContext) -> RunResult[str]:
    return await agent.run(ctx.message, context=ctx.store.bind())
