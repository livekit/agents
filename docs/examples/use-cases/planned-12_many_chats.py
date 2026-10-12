"""
- Acme's assistant: one agent, many people, several chat threads per person, with a sidebar.
- Every chat is its own thread; `Thread` is its state, with pinned facts rendered by
  `@assistant.instructions`.
- `rename_thread` updates thread attributes, which is what the sidebar lists.
- The backend creates threads with `lk.threads.create(scope=)` and lists them with
  `lk.threads.list(scope=)`; the browser sends with `AgentClient`.
- Isolation by construction: a token scopes what a client may send to, the scope what it may
  list.
"""

from dataclasses import dataclass, field

from livekit import api
from livekit.agents import Agent, AgentServer, RunContext
from livekit.agents.a2a import A2ASessionContext, AgentClient
from livekit.agents.storage import AgentDBStorage

APP = "https://acme.livekit.cloud/agents/assistant/production"


@dataclass
class Thread:
    """This thread's state, saved with its thread. Every thread is its own thread."""

    title: str | None = None
    pinned_facts: list[str] = field(default_factory=list)


assistant = Agent(
    instructions="Acme's assistant. Answer from the pinned facts when they apply.",
    llm="google/gemma-4-31b-it",
    state_type=Thread,
)


@assistant.instructions
def pinned(ctx: RunContext[None, Thread]) -> str:
    return "Pinned facts for this thread:\n" + "\n".join(ctx.state.pinned_facts)


@assistant.tool
async def pin(ctx: RunContext[None, Thread], fact: str) -> str:
    """Keep a fact for the rest of this thread."""
    ctx.state.pinned_facts.append(fact)
    return "pinned"


@assistant.tool
async def rename_thread(ctx: RunContext[None, Thread], title: str) -> str:
    ctx.state.title = title
    await ctx.thread.update(attributes={"title": title})  # what the sidebar shows
    return "renamed"


server = AgentServer(storage=AgentDBStorage())


@server.a2a_session(endpoint="assistant", description="Acme's assistant.", state=Thread)
async def chat(ctx: A2ASessionContext[Thread]) -> str:
    # One call per message, for every person and every thread at once. The thread behind ctx is
    # this thread's: its history and state are loaded before this runs and saved after it returns.
    result = await assistant.run(ctx.message, message_history=ctx.message_history, state=ctx.state)
    return result.output


# ---- the web app's backend, with the project credential


async def new_thread(lk: api.LiveKitAPI, user_id: str) -> str:
    """The "new chat" button: every click is a fresh thread."""
    sess = await lk.threads.create(scope=user_id)
    return sess.id


async def sidebar(lk: api.LiveKitAPI, user_id: str) -> list[dict]:
    """The chat list: the platform keeps this user's threads; nothing to keep on your side."""
    # handles, with the thread record loaded; the history is a typed await away, never fetched here
    threads = await lk.threads.list(scope=user_id, limit=30)
    return [
        {"id": s.id, "title": s.attributes.get("title", "New chat"), "at": s.last_active}
        for s in threads
    ]


# ---- the browser


async def send(token: str, thread_id: str, text: str) -> None:
    client = AgentClient(APP, token=token)
    reply = await client.send(text, thread_id=thread_id)  # the thread id is the contextId
    async for item in reply:
        print(item)


# A person with three chats has three threads, each with its own history and state, and one
# agent serves all of them. The threads, the history, the load before and the append after each
# run are the framework's; your code is the agent and the two backend calls above. Isolation is by
# construction: a token scopes what a client may send to, and the scope what it may list.
