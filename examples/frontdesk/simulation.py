"""Simulation glue for the front-desk agent.

A scenario's ``userdata`` (see scenarios.yaml) drives the whole run:

- ``available_slots`` seeds a :class:`FakeCalendar` with deterministic
  availability, replacing the random/cal.com calendar used in production.
- the agent's tools are mocked through the SDK's ``mock_tools``: both mocks
  close over the same calendar, so booking through the mocked
  ``schedule_appointment`` changes what the mocked ``list_available_slots``
  returns on the next call.
- ``expected_booking`` grades the run on final calendar state in
  :func:`on_simulation_end`.

Scenario dates are absolute; run with ``FRONTDESK_NOW`` pinned (see
scenarios.yaml) so they never go stale.
"""

from __future__ import annotations

import datetime
from collections.abc import Callable
from zoneinfo import ZoneInfo

from calendar_api import AvailableSlot, FakeCalendar, now

from livekit.agents import RunContext, SimulationContext, ToolError, beta

SLOT_DURATION_MIN = 30


def parse_slot(value: str, tz: ZoneInfo) -> datetime.datetime:
    """Parse an ISO datetime from scenario userdata (e.g. "2026-06-15T14:30:00")."""
    return datetime.datetime.fromisoformat(value).replace(tzinfo=tz)


def fake_calendar(sim: SimulationContext, *, timezone: str) -> FakeCalendar:
    """Seed a FakeCalendar with the scenario's ``available_slots``."""
    tz = ZoneInfo(timezone)
    slots = [
        AvailableSlot(start_time=parse_slot(s, tz), duration_min=SLOT_DURATION_MIN)
        for s in sim.userdata().get("available_slots", [])
    ]
    return FakeCalendar(timezone=timezone, slots=slots)


def tool_mocks(cal: FakeCalendar, tz: ZoneInfo) -> dict[str, Callable]:
    """Tool mocks sharing live state: a booking changes what the mocked
    list_available_slots returns on the next call.

    The LLM keeps seeing the real tool schemas, only execution is intercepted,
    and a mock may declare any subset of the real tool's parameters (here,
    list_available_slots drops the ``range`` argument).
    """
    slots_map: dict[str, AvailableSlot] = {}

    async def list_available_slots() -> str:
        start = now(tz)
        slots = await cal.list_available_slots(
            start_time=start, end_time=start + datetime.timedelta(days=90)
        )
        slots_map.update({s.unique_hash: s for s in slots})
        return (
            "\n".join(
                f"{s.unique_hash} – {s.start_time.astimezone(tz):%A, %B %d, %Y at %H:%M}"
                for s in slots
            )
            or "No slots available at the moment."
        )

    async def schedule_appointment(ctx: RunContext, slot_id: str) -> str | None:
        if not (slot := slots_map.get(slot_id)):
            raise ToolError(f"error: slot {slot_id} was not found")

        # the mock still collects the attendee email, like the real tool
        email_result = await beta.workflows.GetEmailTask(
            chat_ctx=ctx.session.current_agent.chat_ctx
        )
        if ctx.speech_handle.interrupted:
            return None

        ctx.disallow_interruptions()

        # mutates the shared calendar: the slot disappears from future listings
        await cal.schedule_appointment(
            start_time=slot.start_time, attendee_email=email_result.email_address
        )
        local = slot.start_time.astimezone(tz)
        return f"The appointment was successfully scheduled for {local:%A, %B %d, %Y at %H:%M}."

    return {
        "list_available_slots": list_available_slots,
        "schedule_appointment": schedule_appointment,
    }


async def on_simulation_end(ctx: SimulationContext) -> None:
    # Grade the run on final calendar state. The effective result is the AND of
    # this check and the simulator's conversation judgment, so a mismatch fails
    # a run the simulator passed; a match leaves the simulator's verdict to stand.
    userdata = ctx.userdata()
    if "expected_booking" not in userdata:
        return  # scenario graded on conversation only

    cal: FakeCalendar = ctx.job_context.primary_session.userdata.cal
    booked = cal.scheduled_appointments

    def speak(dt: datetime.datetime) -> str:
        return dt.astimezone(cal.tz).isoformat()

    if (expected_raw := userdata["expected_booking"]) is None:
        if booked:
            times = ", ".join(speak(b.slot.start_time) for b in booked)
            ctx.fail(reason=f"no booking was expected, but the agent booked: {times}")
        return

    expected = parse_slot(expected_raw, cal.tz)
    if len(booked) != 1 or booked[0].slot.start_time != expected:
        times = ", ".join(speak(b.slot.start_time) for b in booked) or "nothing"
        ctx.fail(reason=f"expected a single booking at {speak(expected)}, got {times}")
