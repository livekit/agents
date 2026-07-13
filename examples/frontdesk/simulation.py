from __future__ import annotations

import datetime
from collections.abc import Callable
from zoneinfo import ZoneInfo

from calendar_api import AvailableSlot, FakeCalendar

from livekit.agents import RunContext, SimulationContext, ToolError, beta

SLOT_DURATION_MIN = 30

# The clock the scenarios in scenarios.yaml are authored against (a Friday).
# Every scenario shares this baseline unless it overrides it via userdata `now`.
SIMULATION_NOW = "2026-06-12T09:00:00"


def parse_slot(value: str, tz: ZoneInfo) -> datetime.datetime:
    """Parse an ISO datetime from scenario userdata."""
    return datetime.datetime.fromisoformat(value).replace(tzinfo=tz)


def scenario_now(sim: SimulationContext, *, timezone: str) -> datetime.datetime:
    """The clock to pin for a scenario: its userdata ``now`` when set, else the
    shared :data:`SIMULATION_NOW` baseline the scenarios are authored against."""
    tz = ZoneInfo(timezone)
    return parse_slot(sim.userdata().get("now", SIMULATION_NOW), tz)


def fake_calendar(sim: SimulationContext, *, timezone: str) -> FakeCalendar:
    """Seed a FakeCalendar with the scenario's ``available_slots``, pinned to the
    scenario's clock so its absolute dates stay in the future where intended."""
    tz = ZoneInfo(timezone)
    slots = [
        AvailableSlot(start_time=parse_slot(s, tz), duration_min=SLOT_DURATION_MIN)
        for s in sim.userdata().get("available_slots", [])
    ]
    return FakeCalendar(timezone=timezone, slots=slots, now=scenario_now(sim, timezone=timezone))


def tool_mocks(cal: FakeCalendar, tz: ZoneInfo) -> dict[str, Callable]:
    """Tool mocks sharing live state: a booking changes what the mocked
    list_available_slots returns on the next call."""
    slots_map: dict[str, AvailableSlot] = {}

    async def list_available_slots() -> str:
        start = cal.now()
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

        email_result = await beta.workflows.GetEmailTask(
            chat_ctx=ctx.session.current_agent.chat_ctx
        )
        if ctx.speech_handle.interrupted:
            return None

        ctx.disallow_interruptions()

        await cal.schedule_appointment(
            start_time=slot.start_time, attendee_email=email_result.email_address
        )
        local = slot.start_time.astimezone(tz)
        return f"The appointment was successfully scheduled for {local:%A, %B %d, %Y at %H:%M}."

    return {
        "list_available_slots": list_available_slots,
        "schedule_appointment": schedule_appointment,
    }
