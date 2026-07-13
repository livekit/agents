from __future__ import annotations

import datetime
import logging
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal
from zoneinfo import ZoneInfo

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import simulation
from calendar_api import (
    AvailableSlot,
    CalComCalendar,
    Calendar,
    FakeCalendar,
    SlotUnavailableError,
)
from dotenv import load_dotenv
from ui_view import UIView

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    RunContext,
    SimulationContext,
    ToolError,
    beta,
    cli,
    function_tool,
    get_job_context,
    inference,
    mock_tools,
)
from livekit.agents.evals import (
    JudgeGroup,
    accuracy_judge,
    coherence_judge,
    conciseness_judge,
    handoff_judge,
    relevancy_judge,
    safety_judge,
    task_completion_judge,
    tool_use_judge,
)
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv()


@dataclass
class Userdata:
    cal: Calendar
    slot_unavailable_count: int = 0
    # Optional UI for the LiveKit Playground. ``None`` when the agent
    # is running anywhere else — the tool handlers no-op on it and
    # the rest of the code stays oblivious to the playground.
    ui: UIView | None = None


logger = logging.getLogger("front-desk")


class FrontDeskAgent(Agent):
    def __init__(self, *, timezone: str, now: Callable[[], datetime.datetime]) -> None:
        self.tz = ZoneInfo(timezone)
        # the calendar's clock, so the agent's sense of "today" matches the
        # availability it sees (fixed under simulation, wall-clock otherwise).
        # The current date/time is injected per turn (see on_user_turn_completed)
        # rather than baked into these instructions, which keeps the prompt cache
        # prefix stable across turns and sessions.
        self._now = now

        super().__init__(
            instructions=(
                "You are Front-Desk, a helpful and efficient voice assistant. "
                "Your main goal is to schedule an appointment for the user. "
                "You do not inherently know the current date or time — call get_current_time "
                "whenever you need to reason about dates, such as interpreting a request like "
                "'next Tuesday' or checking whether a date the caller mentions has already passed. "
                "This is a voice conversation — speak naturally, clearly, and concisely. "
                "When the user says hello or greets you, don’t just respond with a greeting — use it as an opportunity to move things forward. "
                "For example, follow up with a helpful question like: 'Would you like to book a time?' "
                "When asked for availability, call list_available_slots and offer a few clear, simple options. "
                "Say things like 'Monday at 2 PM' — avoid timezones, timestamps, and avoid saying 'AM' or 'PM'. "
                "Use natural phrases like 'in the morning' or 'in the evening', and don’t mention the year unless it’s different from the current one. "
                "Offer a few options at a time, pause for a response, then guide the user to confirm. "
                "If the time is no longer available, let them know gently and offer the next options. "
                "Always keep the conversation flowing — be proactive, human, and focused on helping the user schedule with ease."
            )
        )

        self._slots_map: dict[str, AvailableSlot] = {}

    async def on_enter(self) -> None:
        await self.session.say("Hello, I can help you schedule an appointment!")

    @function_tool
    async def get_current_time(self) -> str:
        """Get the current date and time.

        Call this whenever you need to reason about dates — to interpret relative
        requests like "next Tuesday", or to check whether a date the caller
        mentions has already passed.
        """
        # Kept out of the (cached) system instructions and served on demand, so the
        # prompt-cache prefix stays stable and the time is always current.
        return f"The current date and time is {self._now():%A, %B %d, %Y at %H:%M %Z}."

    @function_tool
    async def schedule_appointment(
        self,
        ctx: RunContext[Userdata],
        slot_id: str,
    ) -> str | None:
        """
        Schedule an appointment at the given slot.

        Args:
            slot_id: The identifier for the selected time slot (as shown in the list of available slots).
        """
        if not (slot := self._slots_map.get(slot_id)):
            raise ToolError(f"error: slot {slot_id} was not found")

        email_result = await beta.workflows.GetEmailTask(chat_ctx=self.chat_ctx)

        if ctx.speech_handle.interrupted:
            return

        ctx.disallow_interruptions()

        try:
            await ctx.userdata.cal.schedule_appointment(
                start_time=slot.start_time, attendee_email=email_result.email_address
            )
        except SlotUnavailableError:
            ctx.userdata.slot_unavailable_count += 1
            try:
                get_job_context().tagger.add(
                    "slot:unavailable",
                    metadata={"count": ctx.userdata.slot_unavailable_count},
                )
            except RuntimeError:
                pass
            # exceptions other than ToolError are treated as "An internal error occurred" for the LLM.
            # Tell the LLM this slot isn't available anymore
            raise ToolError("This slot isn't available anymore") from None

        # the booking is recorded by the calendar (the system of record); no
        # parallel bookkeeping here that the simulation mock would have to mirror
        local = slot.start_time.astimezone(self.tz)
        try:
            get_job_context().tagger.add(
                "appointment:booked",
                metadata={"time": local.isoformat()},
            )
        except RuntimeError:
            pass

        if ctx.userdata.ui is not None:
            ctx.userdata.ui.appointment_booked(slot, self.tz)

        return f"The appointment was successfully scheduled for {local.strftime('%A, %B %d, %Y at %H:%M %Z')}."

    @function_tool
    async def list_available_slots(
        self, ctx: RunContext[Userdata], range: Literal["+2week", "+1month", "+3month", "default"]
    ) -> str:
        """
        Return a plain-text list of available slots, one per line.

        <slot_id> – <Weekday>, <Month> <Day>, <Year> at <HH:MM> <TZ> (<relative time>)

        You must infer the appropriate ``range`` implicitly from the
        conversational context and **must not** prompt the user to pick a value
        explicitly.

        Args:
            range: Determines how far ahead to search for free time slots.
        """
        current_time = self._now()
        lines: list[str] = []

        if range == "+2week" or range == "default":
            range_days = 14
        elif range == "+1month":
            range_days = 30
        elif range == "+3month":
            range_days = 90

        slots = await ctx.userdata.cal.list_available_slots(
            start_time=current_time, end_time=current_time + datetime.timedelta(days=range_days)
        )

        for slot in slots:
            local = slot.start_time.astimezone(self.tz)
            delta = local - current_time
            days = delta.days
            seconds = delta.seconds

            if local.date() == current_time.date():
                if seconds < 3600:
                    rel = "in less than an hour"
                else:
                    rel = "later today"
            elif local.date() == (current_time.date() + datetime.timedelta(days=1)):
                rel = "tomorrow"
            elif days < 7:
                rel = f"in {days} days"
            elif days < 14:
                rel = "in 1 week"
            else:
                rel = f"in {days // 7} weeks"

            lines.append(
                f"{slot.unique_hash} – {local.strftime('%A, %B %d, %Y')} at "
                f"{local:%H:%M} {local.tzname()} ({rel})"
            )
            self._slots_map[slot.unique_hash] = slot

        if ctx.userdata.ui is not None:
            ctx.userdata.ui.slots_listed(slots, current_time, self.tz, range_days)

        return "\n".join(lines) or "No slots available at the moment."


server = AgentServer()


async def on_simulation_end(ctx: SimulationContext) -> None:
    # grade the run on final calendar state; a mismatch vetoes the run
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

    expected = simulation.parse_slot(expected_raw, cal.tz)
    if len(booked) != 1 or booked[0].slot.start_time != expected:
        times = ", ".join(speak(b.slot.start_time) for b in booked) or "nothing"
        ctx.fail(reason=f"expected a single booking at {speak(expected)}, got {times}")


async def on_session_end(ctx: JobContext) -> None:
    # `on_session_end` runs even if the job crashed before the AgentSession
    # started (e.g. a bad timezone, a calendar fault) — make_session_report
    # raises in that case, and there's nothing to evaluate anyway.
    try:
        report = ctx.make_session_report()
    except RuntimeError:
        return

    # Skip evaluation for very short conversations
    chat = report.chat_history.copy(exclude_function_call=True, exclude_instructions=True)
    if len(chat.items) < 3:
        return

    judges = JudgeGroup(
        llm="openai/gpt-4o-mini",
        judges=[
            task_completion_judge(),
            accuracy_judge(),
            tool_use_judge(),
            handoff_judge(),
            safety_judge(),
            relevancy_judge(),
            coherence_judge(),
            conciseness_judge(),
        ],
    )

    await judges.evaluate(report.chat_history)

    userdata = ctx.primary_session.userdata
    if userdata.cal.scheduled_appointments:
        ctx.tagger.success()
    else:
        ctx.tagger.fail(reason="Appointment was not booked")

    logger.info("session tags: %s", ctx.tagger.tags)


@server.rtc_session(on_session_end=on_session_end, on_simulation_end=on_simulation_end)
async def frontdesk_agent(ctx: JobContext):
    await ctx.connect()

    timezone = "UTC"
    tool_mocks: dict[str, Callable] = {}

    if sim := ctx.simulation_context():
        # the scenario's userdata seeds the calendar (pinned to the scenario's
        # clock so its absolute dates line up); the tools run mocked
        cal = simulation.fake_calendar(sim, timezone=timezone)
        tool_mocks = simulation.tool_mocks(cal, ZoneInfo(timezone))
    elif cal_api_key := os.getenv("CAL_API_KEY", None):
        logger.info("CAL_API_KEY detected, using cal.com calendar")
        cal = CalComCalendar(api_key=cal_api_key, timezone=timezone)
    else:
        logger.warning(
            "CAL_API_KEY is not set. Falling back to FakeCalendar; set CAL_API_KEY to enable Cal.com integration."
        )
        cal = FakeCalendar(timezone=timezone)

    await cal.initialize()

    userdata = Userdata(cal=cal, ui=UIView(ctx))

    session = AgentSession[Userdata](
        userdata=userdata,
        stt=inference.STT("deepgram/nova-3"),
        llm=inference.LLM("google/gemma-4-31b-it"),
        tts=inference.TTS("cartesia/sonic-3", voice="39b376fc-488e-4d0c-8b37-e00b72059fdd"),
        turn_detection=MultilingualModel(),
        vad=silero.VAD.load(),
        # default (3): lets a turn chain get_current_time -> list_available_slots
    )

    mock_tools(FrontDeskAgent, tool_mocks, session=session)
    await session.start(agent=FrontDeskAgent(timezone=timezone, now=cal.now), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
