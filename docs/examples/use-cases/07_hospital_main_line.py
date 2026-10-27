"""Mercy General department selection with application-owned context and explicit agent runs."""

from dataclasses import dataclass
from typing import Literal

from livekit.agents import (
    Agent,
    AgentServer,
    JobContext,
    RunContext,
    RunResult,
    VoiceSession,
    persisted,
)
from livekit.agents.a2a import A2ASessionContext, RemoteAgent
from livekit.agents.storage import AgentDBStorage
from livekit.plugins import openai

LLM = "google/gemma-4-31b-it"
Department = Literal["billing", "scheduling", "nurse"]


@dataclass
class Caller:
    """The department selected for this caller."""

    department: Department | None = None


@dataclass
class AppContext:
    caller: Caller = persisted(default_factory=Caller)


billing = Agent[AppContext](
    instructions="Mercy General billing: balances, payment plans, itemized statements.", llm=LLM
)
scheduling = Agent[AppContext](
    instructions="Mercy General scheduling: book, confirm or move appointments in any department.",
    llm=LLM,
)
nurse_triage = Agent[AppContext](
    instructions="Mercy General nurse line. Assess symptoms and advise the right level of care. On any emergency sign, tell the caller to hang up and call 911 now.",
    llm=LLM,
)
main_line = Agent[AppContext](
    instructions="Mercy General main line. Find out what the caller needs and route them.", llm=LLM
)
DEPARTMENTS: dict[Department, Agent] = {
    "billing": billing,
    "scheduling": scheduling,
    "nurse": nurse_triage,
}


@main_line.tool
async def quick_balance(ctx: RunContext[AppContext], account_number: str) -> str:
    result = await billing.run(
        f"Balance for account {account_number}.", usage=ctx.usage, context=ctx.context
    )
    if result.status == "input_required":
        return f"Balance lookup needs more information: {result.question}"
    return str(result.output)


@main_line.tool
async def route(ctx: RunContext[AppContext], department: Department) -> str:
    ctx.context.caller.department = department
    await ctx.store.save_context(ctx.context)
    return f"Department selected: {department}. Ask the caller what they need help with."


server = AgentServer(storage=AgentDBStorage())


@server.a2a_session(endpoint="main-line", description="Mercy General main line.")
async def text(ctx: A2ASessionContext) -> RunResult[str]:
    app = AppContext()
    await ctx.store.load_context(app)
    selected = main_line if app.caller.department is None else DEPARTMENTS[app.caller.department]
    result = await selected.run(ctx.message, message_history=ctx.message_history, context=app)
    await ctx.store.save_context(app)
    return result


MAIN_LINE_URL = "https://mercy.livekit.cloud/agents/hospital/production/main-line"
voice = Agent[AppContext](
    instructions="Mercy General's voice on the phone. Hand every request to the main line.",
    tools=[RemoteAgent(MAIN_LINE_URL).as_tool()],
)


@server.rtc_session()
async def phone(ctx: JobContext) -> None:
    session = VoiceSession(
        llm=openai.realtime.RealtimeModel(), thread=ctx.thread, context=AppContext(caller=Caller())
    )
    await session.start(agent=voice, room=ctx.room)
