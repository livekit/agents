from __future__ import annotations

import datetime

from calendar_api import TIMEZONE, Calendar, Slot, SlotUnavailable, calendar
from pydantic import BaseModel

from livekit.agents import Agent, AgentHandler, InputRequired, RunContext, ToolError
from livekit.agents.a2a import A2ASessionContext


class Patient(BaseModel):
    """What the back office keeps about the patient across turns, on its lane."""

    email: str | None = None


backend = Agent(
    instructions=(
        "You are the back office of Pearl Dental. What you say, and the questions you ask, reach "
        "the patient through the front desk. Answer in English, from your tools, in one or two plain "
        "sentences, and never name a time a tool did not list. Every visit, a cleaning, a "
        "check-up or a toothache, is one half-hour slot, so never ask what it is for. For "
        "openings, list_available_slots, with the day or the month the patient named, or with "
        "none, and offer a few at a time in your reply; a time that is not listed is not "
        "available, say so and offer the nearest listed ones. "
        "A booking is schedule_appointment with the slot's id, and the patient's email only "
        "when they gave it; never ask for the email yourself: it asks for it when it is "
        "missing, then reads the appointment back and asks the patient to agree. The answer "
        "to either question is the next message of the same request: an email address means "
        "calling it again with the address, a yes means calling it again exactly the same way, "
        "a no, a refusal or a change of mind means booking nothing and offering other "
        "openings, and anything else gets a short answer beside the same call again, which "
        "asks again: the question is the tool's, never your own words."
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
    now = ctx.deps.now()
    return f"Today is {now:%A, %B %d, %Y}, {now:%H:%M}; today and its later openings count."


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
    and asks the patient; their answer is the next message, and then it is called again.

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


cal = calendar()


back_office = AgentHandler(lane="backend")


@back_office.a2a_session(
    endpoint="backend",
    description="Pearl Dental's back office: openings and appointments.",
    public=False,
    inbox="live",
    state=Patient,
)
async def text(ctx: A2ASessionContext[Patient]) -> str:
    result = await backend.run(
        ctx.message, message_history=ctx.message_history, state=ctx.state, deps=cal
    )
    return result.input_required or str(result.output)
