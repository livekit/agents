"""
- Mercy General's main line: billing, scheduling and nurse triage behind one number.
- Three ways agents work together: delegation inside a tool, handoff by returning an `Agent`,
  and a thin voice door delegating to the same brain.
- A handoff continues the same history within the turn and is saved on the lane's agent stack;
  in text `main_line.run(message)` runs whichever agent is on top of the stack, so the handler
  stays one line.
- The voice door, whose one tool is the main line's own text door, and that text endpoint run the same agents on the
  same thread.
"""

from dataclasses import dataclass
from typing import Literal

from livekit.agents import Agent, AgentServer, JobContext, LiveSession, RunContext
from livekit.agents.a2a import A2ASessionContext
from livekit.agents.delegation import A2ADelegate
from livekit.agents.storage import AgentDBStorage
from livekit.plugins import openai

LLM = "google/gemma-4-31b-it"
Department = Literal["billing", "scheduling", "nurse"]


@dataclass
class Caller:
    """State: saved with the lane after every run, shared by every agent in it."""

    department: Department | None = None


billing = Agent(
    instructions="Mercy General billing: balances, payment plans, itemized statements.",
    llm=LLM,
    state_type=Caller,
)
scheduling = Agent(
    instructions="Mercy General scheduling: book, confirm or move appointments in any department.",
    llm=LLM,
    state_type=Caller,
)
nurse_triage = Agent(
    instructions=(
        "Mercy General nurse line. Assess symptoms and advise the right level of care. "
        "On any emergency sign, tell the caller to hang up and call 911 now."
    ),
    llm=LLM,
    state_type=Caller,
)
main_line = Agent(
    instructions="Mercy General main line. Find out what the caller needs and route them.",
    llm=LLM,
    state_type=Caller,
)
DEPARTMENTS: dict[Department, Agent] = {
    "billing": billing,
    "scheduling": scheduling,
    "nurse": nurse_triage,
}


# 1. Delegation inside a tool: the main line asks billing and phrases the answer itself.
#    Billing runs on a fresh history; the main line's history gets the tool call and its result.
@main_line.tool
async def quick_balance(ctx: RunContext[None, Caller], account_number: str) -> str:
    result = await billing.run(f"Balance for account {account_number}.", usage=ctx.usage)
    return result.output


# 2. Handoff: the tool returns the agent that takes over, on the same history.
@main_line.tool
async def route(ctx: RunContext[None, Caller], department: Department) -> Agent:
    ctx.state.department = department
    return DEPARTMENTS[department]


server = AgentServer(storage=AgentDBStorage())


# The text door names the root agent. After a handoff, main_line.run() runs whichever agent is
# on top of the lane's agent stack, so the next message goes straight to the department.
@server.a2a_session(endpoint="main-line", description="Mercy General main line.", state=Caller)
async def text(ctx: A2ASessionContext[Caller]) -> str:
    result = await main_line.run(ctx.message, message_history=ctx.message_history, state=ctx.state)
    return result.output


# 3. The same brain behind the phone: the voice door is thin and delegates every request to the
#    main line, so a call and a text run the same agents on the same history. A remote expert in
#    another deployment is a tool that delegates over A2A (03).
MAIN_LINE_URL = "https://mercy.livekit.cloud/agents/hospital/production/main-line"
voice = Agent(
    instructions="Mercy General's voice on the phone. Hand every request to the main line.",
    tools=[A2ADelegate(MAIN_LINE_URL)],
)


@server.rtc_session()
async def phone(ctx: JobContext) -> None:
    session = LiveSession(llm=openai.realtime.RealtimeModel(), state=Caller(), thread=ctx.thread)
    await session.start(agent=voice, room=ctx.room)


# 1 is a tool call, so its result is in the history and never runs twice. 2 is a turn on the
# history; the lane's agent stack picks the agent of the next text turn. 3 runs main_line in
# process, on the call's history. This is the top inbound use case, the conversational IVR: the caller says it once.
