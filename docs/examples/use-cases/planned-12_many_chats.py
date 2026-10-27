"""Acme's assistant: one agent, many people, several chat threads per person, with a sidebar."""

from dataclasses import dataclass, field

from livekit import api
from livekit.agents import Agent, AgentServer, RunContext, RunResult, persisted
from livekit.agents.a2a import A2ASessionContext, AgentClient
from livekit.agents.storage import AgentDBStorage

APP = "https://acme.livekit.cloud/agents/assistant/production/assistant"


@dataclass
class Thread:
    """This conversation's title and pinned facts."""

    title: str | None = None
    pinned_facts: list[str] = field(default_factory=list)


@dataclass
class AppContext:
    thread: Thread = persisted(default_factory=Thread)


assistant = Agent[AppContext](
    instructions="Acme's assistant. Answer from the pinned facts when they apply.",
    llm="google/gemma-4-31b-it",
)


@assistant.instructions
def pinned(ctx: RunContext[AppContext]) -> str:
    return "Pinned facts for this thread:\n" + "\n".join(ctx.context.thread.pinned_facts)


@assistant.tool
async def pin(ctx: RunContext[AppContext], fact: str) -> str:
    """Keep a fact for the rest of this thread."""
    ctx.context.thread.pinned_facts.append(fact)
    return "pinned"


@assistant.tool
async def rename_thread(ctx: RunContext[AppContext], title: str) -> str:
    ctx.context.thread.title = title
    await ctx.thread.update(attributes={"title": title})
    await ctx.store.save_context(ctx.context)
    return "renamed"


server = AgentServer(storage=AgentDBStorage())


@server.a2a_session(endpoint="assistant", description="Acme's assistant.")
async def chat(ctx: A2ASessionContext) -> RunResult[str]:
    app = AppContext()
    await ctx.store.load_context(app)
    result = await assistant.run(ctx.message, message_history=ctx.message_history, context=app)
    await ctx.store.save_context(app)
    return result


async def new_thread(lk: api.LiveKitAPI, user_id: str) -> str:
    """The "new chat" button: every click is a fresh thread."""
    sess = await lk.threads.create(scope=user_id)
    return sess.id


@dataclass
class ChatSummary:
    id: str
    title: str
    last_active: int


async def sidebar(lk: api.LiveKitAPI, user_id: str) -> list[ChatSummary]:
    """List the subject's threads and read their display metadata."""
    threads = await lk.threads.list(scope=user_id, limit=30)
    summaries = []
    for thread in threads:
        info = await thread.info()  # A listed handle can use its already loaded info.
        summaries.append(
            ChatSummary(
                id=thread.id,
                title=info.attributes.get("title", "New chat"),
                last_active=info.last_active,
            )
        )
    return summaries


async def send(token: str, thread_id: str, text: str) -> None:
    client = AgentClient(APP, token=token)
    try:
        reply = await client.send(text, thread_id=thread_id)
        async for item in reply:
            print(item)
    finally:
        await client.aclose()
