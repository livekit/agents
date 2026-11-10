# /// script
# requires-python = ">=3.10"
# dependencies = ["google-adk[a2a,openai]>=2.14", "aiohttp", "uvicorn", "python-dotenv"]
# ///
"""Pearl Dental's back office as a third-party A2A server: the tools of backend_agent.py on the
same calendar, as a Google ADK agent with nothing from LiveKit.

What ADK's A2A server does differently: a question is a plain answer that completes the task,
since ADK's own `input-required` carries no text, so the read-back is held in the session state;
the session lives in memory.

usage: uv run adk_backend.py   (serves http://localhost:$BACKEND_PORT)
Its dependencies, also in adk_requirements.txt for `uv pip install -r`, clash with the
project's pins, so they go in an environment of their own.
ADK_MODEL picks the model: gpt-5.4-mini with OPENAI_API_KEY, or a Gemini one with GOOGLE_API_KEY.
"""

import asyncio
import datetime
import os

import uvicorn
from a2a.types import TaskStatusUpdateEvent
from calendar_api import TIMEZONE, SlotUnavailable, calendar
from dotenv import find_dotenv, load_dotenv
from google.adk.a2a.executor.a2a_agent_executor import A2aAgentExecutor
from google.adk.a2a.executor.config import A2aAgentExecutorConfig, ExecuteInterceptor
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.agents import LlmAgent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.events.event import Event
from google.adk.tools import ToolContext
from google.genai import types

load_dotenv(find_dotenv(usecwd=True))

PORT = int(os.environ.get("BACKEND_PORT", "8001"))
# as backend_agent.py: a slow payer system and the insurers the practice takes
INSURANCE_CHECK_SECONDS = 8
ACCEPTED_INSURERS = ("Aetna", "Cigna", "Delta Dental", "MetLife")

cal = calendar()


def spoken(when: datetime.datetime) -> str:
    return f"{when:%A, %B %d, %Y at %H:%M}"


def midnight(day: str) -> datetime.datetime:
    return datetime.datetime.combine(datetime.date.fromisoformat(day), datetime.time(), TIMEZONE)


def today(ctx: ReadonlyContext) -> str:
    return (
        "You are the back office of Pearl Dental. What you say, and the questions you ask, reach "
        "the patient through the front desk. Answer in English, from your tools, in one or two "
        "plain sentences, and never name a time a tool did not list. Every visit, a cleaning, a "
        "check-up or a toothache, is one half-hour slot. For openings, list_available_slots, "
        "with the day or the month the patient named, and offer a few at a time; a time that is "
        "not listed is not available, say so and offer the nearest listed ones. A booking is "
        "schedule_appointment with the slot's id and the patient's email, never one they did "
        "not give; it asks for a missing email, then reads the appointment back for the "
        "patient's yes, and what it asks is your answer, as written. Their answer is the next "
        "request: a yes, call it again with the same slot and email; a no, other openings; else "
        "ask again. Only its booked means booked. "
        f"Insurance is check_insurance. It is now {spoken(cal.now())}; dates are in the future."
    )


async def list_available_slots(after: str | None = None, until: str | None = None) -> str:
    """The open slots, one per line, as `<id> - <day> at <time> (<how soon>)`. Without dates,
    the next two weeks; for a day the patient named, that day; for a month, that month.

    Args:
        after: the first day to look at, YYYY-MM-DD
        until: the last day to look at, YYYY-MM-DD
    """
    now = cal.now()
    start = max(now, midnight(after)) if after else now
    end = (
        midnight(until) + datetime.timedelta(days=1)
        if until
        else start + datetime.timedelta(days=14)
    )
    lines = []
    for slot in await cal.available(start, end):
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


async def schedule_appointment(
    slot_id: str, tool_context: ToolContext, email: str | None = None
) -> str:
    """Book the slot. It asks for the email when it is missing, then reads the appointment back
    for the patient's yes; once they said yes, call it again with the same slot and email.

    Args:
        slot_id: the slot's id, as list_available_slots wrote it
        email: where the confirmation goes, as the patient gave it
    """
    now = cal.now()
    slots = await cal.available(now, now + datetime.timedelta(days=90))
    slot = next((s for s in slots if s.id == slot_id), None)
    if slot is None:
        return "that slot is no longer open; list the openings again"
    email = email or tool_context.state.get("email")
    if email is None or "@" not in email:
        return "What email address should the confirmation go to?"

    # booked only on a later request than the one that read it back, which the patient answered
    read_back = tool_context.state.get("read_back")
    if (
        read_back != [slot_id, email]
        or tool_context.state.get("read_in") == tool_context.invocation_id
    ):
        tool_context.state["read_back"] = [slot_id, email]
        tool_context.state["read_in"] = tool_context.invocation_id
        return (
            f"Please confirm: an appointment on {spoken(slot.start)}, confirmation to {email}. "
            "Yes or no?"
        )
    try:
        await cal.schedule(slot, email)
    except SlotUnavailable:
        return "that slot was just taken; list the openings again"
    tool_context.state["email"] = email
    tool_context.state["read_back"] = None
    return f"booked {spoken(slot.start)}; the confirmation goes to {email}"


async def check_insurance(provider: str | None = None):
    """Whether the practice takes the patient's dental insurance; it takes a while, and it asks
    which insurance they have when they named none.

    Args:
        provider: the insurance company, as the patient named it; left out when they named none
    """
    if not provider:
        yield "Which dental insurance do you have?"
        return
    # a text event while the tool runs, tagged so the caller tells it as progress
    yield Event(
        content=types.Content(role="model", parts=[types.Part(text=f"Checking with {provider}.")]),
        custom_metadata={"progress": True},
    )
    await asyncio.sleep(INSURANCE_CHECK_SECONDS)
    if any(name.lower() in provider.lower() for name in ACCEPTED_INSURERS):
        yield f"{provider} is accepted"
    else:
        yield f"{provider} is not accepted; visits are self-pay"


back_office = LlmAgent(
    name="pearl_dental_back_office",
    model=os.environ.get("ADK_MODEL", "gpt-5.4-mini"),
    description="Pearl Dental's back office: openings and appointments.",
    instruction=today,
    tools=[list_available_slots, schedule_appointment, check_insurance],
)


async def flag_progress(ctx: object, a2a_event: object, adk_event: Event) -> object:
    """Flags a tagged event's ``working`` message ``livekit/progress``: plain A2A gives a
    ``working`` message no meaning, and ADK sends its answer there too."""
    if (
        isinstance(a2a_event, TaskStatusUpdateEvent)
        and (adk_event.custom_metadata or {}).get("progress")
        and a2a_event.status.HasField("message")
    ):
        a2a_event.status.message.metadata.update({"livekit/progress": True})
    return a2a_event


app = to_a2a(
    back_office,
    port=PORT,
    agent_executor_factory=lambda runner: A2aAgentExecutor(
        runner=runner,
        config=A2aAgentExecutorConfig(
            execute_interceptors=[ExecuteInterceptor(after_event=flag_progress)]
        ),
    ),
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
