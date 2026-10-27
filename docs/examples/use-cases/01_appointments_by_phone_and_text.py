"""Northside Family Clinic's appointment line: confirm, book or move an appointment, by phone and"""

from dataclasses import dataclass
from datetime import datetime

import httpx

from livekit.agents import (
    Agent,
    AgentServer,
    JobContext,
    RunContext,
    RunResult,
    VoiceSession,
    persisted,
)
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
    """This patient's state. Restored with load_context and retained with explicit save_context calls."""

    appointment: Appointment | None = None


@dataclass
class Deps:
    """What the tools need for one run. Never saved."""

    clinic: httpx.AsyncClient


@dataclass
class AppContext:
    services: Deps
    patient: Patient = persisted(default_factory=Patient)


scheduler = Agent[AppContext](
    instructions="Northside Family Clinic scheduling. Confirm, book or move appointments.",
    llm="google/gemma-4-31b-it",
)


@scheduler.instructions
def known_appointment(ctx: RunContext[AppContext]) -> str:
    """Rendered every turn, so the model always knows what the state holds."""
    a = ctx.context.patient.appointment
    if a is None:
        return "No appointment known yet. Ask for the patient's last name and date of birth."
    return f"The patient's next appointment: {a.when:%A %d %B at %H:%M} with {a.provider}."


@scheduler.tool
async def find_appointment(
    ctx: RunContext[AppContext], last_name: str, date_of_birth: str
) -> Appointment | str:
    """Find the patient's next appointment in the clinic's record.

    Args:
        last_name: as spelled on the patient's file
        date_of_birth: YYYY-MM-DD
    """
    params = {"last_name": last_name, "dob": date_of_birth}
    r = await ctx.context.services.clinic.get("/appointments/next", params=params)
    if r.status_code != 200:
        await ctx.store.save_context(ctx.context)
        return "no upcoming appointment for that patient"
    ctx.context.patient.appointment = Appointment(**r.json())
    await ctx.store.save_context(ctx.context)
    return ctx.context.patient.appointment


@scheduler.tool
async def reschedule(ctx: RunContext[AppContext], when: datetime) -> str:
    """Move the known appointment. The clinic's calendar confirms or refuses."""
    a = ctx.context.patient.appointment
    if a is None:
        await ctx.store.save_context(ctx.context)
        return "find the appointment first"
    body = {"when": when.isoformat()}
    r = await ctx.context.services.clinic.post(f"/appointments/{a.id}/move", json=body)
    if r.status_code != 200:
        await ctx.store.save_context(ctx.context)
        return f"{when:%A at %H:%M} is taken. Nearest free: {r.json()['nearest']}"
    ctx.context.patient.appointment = Appointment(**r.json())
    await ctx.store.save_context(ctx.context)
    return f"moved to {when:%A %d %B at %H:%M}"


server = AgentServer(storage=AgentDBStorage())


@server.rtc_session()
async def phone(ctx: JobContext) -> None:
    deps = Deps(clinic=httpx.AsyncClient(base_url=CLINIC))
    session = VoiceSession(
        llm=openai.realtime.RealtimeModel(),
        thread=ctx.thread,
        context=AppContext(services=deps, patient=Patient()),
    )
    await session.start(agent=scheduler, room=ctx.room)
    await session.store.load_context(session.context)
    await session.generate_reply(instructions="Greet the caller and ask about their appointment.")


@server.a2a_session(endpoint="scheduling", description="Book, confirm or move an appointment.")
async def text(ctx: A2ASessionContext) -> RunResult[str]:
    deps = Deps(clinic=httpx.AsyncClient(base_url=CLINIC))
    app = AppContext(services=deps)
    await ctx.store.load_context(app)
    result = await scheduler.run(ctx.message, message_history=ctx.message_history, context=app)
    await ctx.store.save_context(app)
    return result
