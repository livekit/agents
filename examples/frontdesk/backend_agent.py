from __future__ import annotations

import asyncio
import datetime

from calendar_api import TIMEZONE, Calendar, Slot, SlotUnavailable, calendar
from pydantic import BaseModel

from livekit.agents import Agent, AgentHandler, InputRequired, RunContext, RunResult, ToolError
from livekit.agents.a2a import A2ASessionContext
from livekit.agents.llm import ToolFlag


class Patient(BaseModel):
    """What the back office keeps about the patient across turns, on its lane."""

    email: str | None = None


backend = Agent(
    instructions=(
        "You are the back office of Pearl Dental. What you say, and the questions you ask, reach "
        "the patient through the front desk. Answer in English, from your tools, in one or two plain "
        "sentences, and never name a time a tool did not list. Every visit, a cleaning, a "
        "check-up or a toothache, is one half-hour slot. For openings, list_available_slots, "
        "with the day or the month the patient named, and offer a few at a time; a time that "
        "is not listed is not available, say so and offer the nearest listed ones. A booking "
        "is schedule_appointment with the slot's id and the patient's email, never one they "
        "did not give; it asks for a missing email, then reads the appointment back for the "
        "patient's yes. Their answer is the next request: an email means calling it again with "
        "that email, a yes means calling it again the same way, a no means other openings. "
        "Insurance is check_insurance; a check the patient no longer wants, you cancel."
    ),
    llm="google/gemma-4-31b-it",
    deps_type=Calendar,
    state_type=Patient,
)


def spoken(when: datetime.datetime) -> str:
    return f"{when:%A, %B %d, %Y at %H:%M}"


def midnight(day: datetime.date) -> datetime.datetime:
    return datetime.datetime.combine(day, datetime.time(), tzinfo=TIMEZONE)


@backend.instructions
def today(ctx: RunContext[Calendar, Patient]) -> str:
    return f"It is now {spoken(ctx.deps.now())}; dates are in the future."


@backend.tool
async def list_available_slots(
    ctx: RunContext[Calendar, Patient],
    after: datetime.date | None = None,
    until: datetime.date | None = None,
) -> str:
    """The open slots, one per line, as `<id> - <day> at <time> (<how soon>)`. Without dates,
    the next two weeks; for a day the patient named, that day; for a month, that month.

    Args:
        after: the first day to look at, YYYY-MM-DD
        until: the last day to look at, YYYY-MM-DD
    """
    now = ctx.deps.now()
    start = max(now, midnight(after)) if after else now
    end = (
        midnight(until) + datetime.timedelta(days=1)
        if until
        else start + datetime.timedelta(days=14)
    )
    await ctx.update("Checking the calendar.")
    slots = await ctx.deps.available(start, end)
    lines = []
    for slot in slots:
        days = (slot.start.date() - now.date()).days
        soon = (
            "today"
            if days == 0
            else "tomorrow"
            if days == 1
            else f"in {days} days"
            if days < 7
            else f"in {days // 7} weeks"
        )
        lines.append(f"{slot.id} - {spoken(slot.start)} ({soon})")
    return "\n".join(lines) or "no openings between those dates"


async def open_slot(ctx: RunContext[Calendar, Patient], slot_id: str) -> Slot:
    now = ctx.deps.now()
    slots = await ctx.deps.available(now, now + datetime.timedelta(days=90))
    slot = next((s for s in slots if s.id == slot_id), None)
    if slot is None:
        raise ToolError("that slot is no longer open; list the openings again")
    return slot


async def read_back(
    ctx: RunContext[Calendar, Patient], slot_id: str, email: str | None = None
) -> str | InputRequired:
    """What the patient approves before anything is written; the email first, when missing."""
    slot = await open_slot(ctx, slot_id)
    email = email or ctx.state.email
    if email is None or "@" not in email:
        return InputRequired("What email address should the confirmation go to?")
    return f"Please confirm: an appointment on {spoken(slot.start)}, confirmation to {email}. Yes or no?"


@backend.tool(requires_approval=read_back)
async def schedule_appointment(
    ctx: RunContext[Calendar, Patient], slot_id: str, email: str | None = None
) -> str:
    """Book the slot. It asks for the email when it is missing, then reads the appointment back
    and asks the patient; each answer comes back as this call's output, and then it is called
    again with it.

    Args:
        slot_id: the slot's id, as list_available_slots wrote it
        email: where the confirmation goes, as the patient gave it
    """
    slot = await open_slot(ctx, slot_id)
    email = email or ctx.state.email
    if email is None:
        raise ToolError("no email yet; call again with the patient's email")
    try:
        await ctx.deps.schedule(slot, email)
    except SlotUnavailable:
        raise ToolError("that slot was just taken; list the openings again") from None
    ctx.state.email = email
    return f"booked {spoken(slot.start)}; the confirmation goes to {email}"


INSURANCE_CHECK_SECONDS = 8
"""How long an insurance check takes: it stands in for a slow payer system."""
ACCEPTED_INSURERS = ("aetna", "cigna", "delta dental", "metlife")


@backend.tool(flags=ToolFlag.CANCELLABLE)
async def check_insurance(ctx: RunContext[Calendar, Patient], provider: str) -> str:
    """Whether the practice takes the patient's dental insurance; it takes a while.

    Args:
        provider: the insurance company, as the patient named it
    """
    await ctx.update(f"Checking with {provider}.")
    await asyncio.sleep(INSURANCE_CHECK_SECONDS)
    if any(name in provider.lower() for name in ACCEPTED_INSURERS):
        return f"{provider} is accepted"
    return f"{provider} is not accepted; visits are self-pay"


cal = calendar()


back_office = AgentHandler(lane="backend")


@back_office.a2a_session(
    endpoint="backend",
    description="Pearl Dental's back office: openings and appointments.",
    public=False,
    inbox="live",
    state=Patient,
)
async def text(ctx: A2ASessionContext[Patient]) -> RunResult[str]:
    # the run as it is: its question while it waits, else its answer, none while work runs on
    return await backend.run(
        ctx.message, message_history=ctx.message_history, state=ctx.state, deps=cal
    )
