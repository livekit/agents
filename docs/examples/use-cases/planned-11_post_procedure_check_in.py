"""
- Northside Surgical Center's recovery line: check on a patient on day 1, 3 and 7, and escalate at
  once when something is wrong.
- `CheckIn` is an `AgentTask` submitted `at=` a date; `on_enter` returns `InputRequired(question,
  timeout=12h)`, and its model reads the answer and calls `report`.
- `report` submits `Escalate`, a pure code task, when the pain or a concern warrants it, then
  schedules the next check-in.
- `on_timeout` moves on when the patient does not answer.
- The backend can start the chain with `thread.submit()` and no conversation.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta

import httpx

from livekit.agents import (
    Agent,
    AgentServer,
    AgentTask,
    Completed,
    InputRequired,
    RunContext,
    function_tool,
)
from livekit.agents.a2a import A2ASessionContext
from livekit.agents.storage import AgentDBStorage

CLINIC = "https://api.northsidesurgical.example"
DAYS = [1, 3, 7]  # days after the procedure


@dataclass
class Recovery:
    """State: saved with the lane after every run, read by the agent and its tasks."""

    procedure_date: datetime | None = None


@dataclass
class Deps:
    clinic: httpx.AsyncClient


care_team = Agent(
    instructions="Northside Surgical Center recovery line. Warm, brief, never diagnose.",
    llm="google/gemma-4-31b-it",
    deps_type=Deps,
    state_type=Recovery,
)


# A check-in is an AgentTask submitted at a date. on_enter returns InputRequired with the question
# in the code's words; the model turns the reply into a call of report. It submits the next
# check-in before completing.


@dataclass
class CheckIn(AgentTask[None]):
    step: int = 0

    def __post_init__(self) -> None:  # talks to the patient; needs no outside system itself
        super().__init__(
            instructions="When the patient answers, call report.", llm="google/gemma-4-31b-it"
        )

    async def on_enter(self, ctx: RunContext[None, Recovery]) -> InputRequired:
        day = DAYS[self.step]
        return InputRequired(
            f"Day {day} after your procedure: pain from 0 to 10? Anything worrying you?",
            timeout=timedelta(hours=12),
        )

    @function_tool
    async def report(
        self, ctx: RunContext[None, Recovery], pain: int, concern: str | None = None
    ) -> Completed[None]:
        """How the patient is doing: pain from 0 to 10, and anything worrying them."""
        if pain >= 7 or concern:
            await ctx.submit(Escalate(pain, concern))
        await self.schedule_next(ctx)
        return Completed(None)

    async def on_timeout(self, ctx: RunContext[None, Recovery]) -> Completed[None]:  # no reply
        await self.schedule_next(ctx)
        return Completed(None)

    async def schedule_next(self, ctx) -> None:
        if self.step + 1 < len(DAYS):
            next_date = ctx.state.procedure_date + timedelta(days=DAYS[self.step + 1])
            await ctx.submit(CheckIn(self.step + 1), at=next_date)


@dataclass
class Escalate(AgentTask[str]):  # pure code
    pain: int
    concern: str | None

    def __post_init__(self) -> None:
        super().__init__(deps=Deps(clinic=httpx.AsyncClient(base_url=CLINIC)))

    async def on_enter(self, ctx: RunContext[Deps, Recovery]) -> Completed[str]:
        task = {"kind": "nurse-callback", "pain": self.pain, "concern": self.concern}
        headers = {"Idempotency-Key": ctx.task_id}
        await ctx.deps.clinic.post("/tasks", json=task, headers=headers)
        return Completed("A nurse will call you today.")  # delivered to the patient


@care_team.tool
async def discharged(ctx: RunContext[Deps, Recovery], procedure_date: datetime) -> str:
    ctx.state.procedure_date = procedure_date
    first = procedure_date + timedelta(days=DAYS[0])
    await ctx.submit(CheckIn(), at=first)
    return "follow-ups scheduled"


server = AgentServer(storage=AgentDBStorage())


@server.a2a_session(endpoint="recovery", description="Northside Surgical Center.", state=Recovery)
async def text(ctx: A2ASessionContext[Recovery]) -> str:
    deps = Deps(clinic=httpx.AsyncClient(base_url=CLINIC))
    result = await care_team.run(
        ctx.message, message_history=ctx.message_history, state=ctx.state, deps=deps
    )
    return result.output


# Each task runs on whichever worker holds the thread then. "About a 4, and the wound is a bit
# red" arrives as a turn of care_team, which sees the check-in waiting and hands the conversation
# to it; CheckIn's model calls report(pain=4, concern="the wound is a bit red"). The clinic's
# backend can start the chain with no conversation:
# lk.threads.get(thread_id).submit(agent="recovery", task=CheckIn()), the deployment's agent name.
