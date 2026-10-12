"""
- Northside Family Clinic's appointment line: confirm, book or move an appointment, by phone and
  by text.
- One stateless `scheduler` agent serves both doors: a `LiveSession` running it on a call, and an
  `a2a_session` handler running one turn per text.
- `Patient` is the saved state: `state_type=` on the agent, `ctx.state` in the tools, loaded
  before every run and committed after it.
- `Deps` carries the clinic's HTTP client: `deps_type=` on the agent, `ctx.deps` in the tools, built
  by the door for the call or the turn and passed to `LiveSession` or `run()`, never saved.
- `@scheduler.instructions` renders the known appointment into the prompt every turn.
- A call continues the thread its dispatch names; a text continues a thread by its id. On one
  thread both doors run one lane, so what was found by text is known on the call, on any worker.
"""

from dataclasses import dataclass
from datetime import datetime

import httpx

from livekit.agents import Agent, AgentServer, JobContext, LiveSession, RunContext
from livekit.agents.a2a import A2ASessionContext
from livekit.agents.storage import AgentDBStorage
from livekit.plugins import openai

CLINIC = "https://api.northsideclinic.example"


@dataclass
class Appointment:
    id: str
    provider: str
    when: datetime


@dataclass
class Patient:
    """This patient's state. Saved with the lane after every run, loaded on the next one."""

    appointment: Appointment | None = None


@dataclass
class Deps:
    """What the tools need for one run. Never saved."""

    clinic: httpx.AsyncClient


scheduler = Agent(
    instructions="Northside Family Clinic scheduling. Confirm, book or move appointments.",
    llm="google/gemma-4-31b-it",
    deps_type=Deps,
    state_type=Patient,
)


@scheduler.instructions
def known_appointment(ctx: RunContext[Deps, Patient]) -> str:
    """Rendered every turn, so the model always knows what the state holds."""
    a = ctx.state.appointment
    if a is None:
        return "No appointment known yet. Ask for the patient's last name and date of birth."
    return f"The patient's next appointment: {a.when:%A %d %B at %H:%M} with {a.provider}."


@scheduler.tool
async def find_appointment(
    ctx: RunContext[Deps, Patient], last_name: str, date_of_birth: str
) -> Appointment | str:
    """Find the patient's next appointment in the clinic's record.

    Args:
        last_name: as spelled on the patient's file
        date_of_birth: YYYY-MM-DD
    """
    params = {"last_name": last_name, "dob": date_of_birth}
    r = await ctx.deps.clinic.get("/appointments/next", params=params)
    if r.status_code != 200:
        return "no upcoming appointment for that patient"
    ctx.state.appointment = Appointment(**r.json())
    return ctx.state.appointment


@scheduler.tool
async def reschedule(ctx: RunContext[Deps, Patient], when: datetime) -> str:
    """Move the known appointment. The clinic's calendar confirms or refuses."""
    a = ctx.state.appointment
    if a is None:
        return "find the appointment first"
    body = {"when": when.isoformat()}
    r = await ctx.deps.clinic.post(f"/appointments/{a.id}/move", json=body)
    if r.status_code != 200:
        return f"{when:%A at %H:%M} is taken. Nearest free: {r.json()['nearest']}"
    ctx.state.appointment = Appointment(**r.json())
    return f"moved to {when:%A %d %B at %H:%M}"


server = AgentServer(storage=AgentDBStorage())


@server.rtc_session()
async def phone(ctx: JobContext) -> None:
    # the realtime model runs the scheduler and its tools itself: fastest, fine for two small tools.
    # a brain with heavier tools stays behind its own door, an A2ADelegate in tools=, and the voice door stays thin (03)
    deps = Deps(clinic=httpx.AsyncClient(base_url=CLINIC))
    session = LiveSession(
        llm=openai.realtime.RealtimeModel(), state=Patient(), deps=deps, thread=ctx.thread
    )
    await session.start(agent=scheduler, room=ctx.room)
    await session.generate_reply(instructions="Greet the caller and ask about their appointment.")


@server.a2a_session(
    endpoint="scheduling", description="Book, confirm or move an appointment.", state=Patient
)
async def text(ctx: A2ASessionContext[Patient]) -> str:
    deps = Deps(clinic=httpx.AsyncClient(base_url=CLINIC))
    result = await scheduler.run(
        ctx.message, message_history=ctx.message_history, state=ctx.state, deps=deps
    )
    return result.output


# A patient texts, and the backend keeps the thread id against the patient; a week later it
# dispatches their call with that id in the dispatch metadata: one thread,
# and the appointment found by text is in the state on the call, on any worker. run() is one turn
# on the history you pass; the handler's return is the answer. The state is saved after every run,
# the deps never: the door builds them for the run and passes them.
