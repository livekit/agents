from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import pytest

from livekit.agents import AgentSession, beta, inference, llm

from .calendar_api import AvailableSlot, FakeCalendar
from .frontdesk_agent import FrontDeskAgent, Userdata

TIMEZONE = "UTC"


def _llm_model() -> llm.LLM:
    return inference.LLM(
        model="openai/gpt-4.1", extra_kwargs={"parallel_tool_calls": False, "temperature": 0.45}
    )


@pytest.mark.asyncio
async def test_slot_scheduling() -> None:
    tz = tz = ZoneInfo(TIMEZONE)
    today = datetime.now(tz).date()

    # fmt: off
    slots = [
        AvailableSlot(start_time=datetime.combine(today, time(9, 0), tzinfo=tz), duration_min=30),
        AvailableSlot(start_time=datetime.combine(today, time(9, 30), tzinfo=tz), duration_min=30),
        AvailableSlot(start_time=datetime.combine(today, time(10, 0), tzinfo=tz), duration_min=30),

        AvailableSlot(start_time=datetime.combine(today + timedelta(days=1), time(14, 0), tzinfo=tz), duration_min=30),
        AvailableSlot(start_time=datetime.combine(today + timedelta(days=1), time(14, 30), tzinfo=tz), duration_min=30),
        AvailableSlot(start_time=datetime.combine(today + timedelta(days=1), time(15, 0), tzinfo=tz), duration_min=30),

        AvailableSlot(start_time=datetime.combine(today + timedelta(days=2), time(11, 0), tzinfo=tz), duration_min=30),
        AvailableSlot(start_time=datetime.combine(today + timedelta(days=2), time(11, 30), tzinfo=tz), duration_min=30),
    ]
    # fmt: on

    userdata = Userdata(cal=FakeCalendar(timezone=TIMEZONE, slots=slots))

    async with _llm_model() as llm, AgentSession(llm=llm, userdata=userdata) as sess:
        await sess.start(FrontDeskAgent(timezone=TIMEZONE))
        result = await sess.run(user_input="Can I get an appointment tomorrow?")
        result.expect.skip_next_event_if(type="message", role="assistant")
        result.expect.next_event().is_function_call(name="list_available_slots")
        result.expect.next_event().is_function_call_output()

        tomorrow = today + timedelta(days=1)
        expected_tomorrow_slots = [slot for slot in slots if slot.start_time.date() == tomorrow]
        expected_times_text = ", ".join(
            slot.start_time.strftime("%-I:%M %p") for slot in expected_tomorrow_slots
        )

        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(
                llm,
                intent=(
                    "must suggest one or more available appointment time slots for tomorrow "
                    f"({today + timedelta(days=1):%B %-d, %Y}). For reference, today is {today:%B %-d, %Y}"
                    f"Must only suggest times that are present in the calendar slots for tomorrow, "
                    f"which are: {expected_times_text}."
                ),
            )
        )

        result = await sess.run(user_input="2 in the afternoon sounds good")
        result.expect.skip_next_event_if(type="message", role="assistant")

        slot_id = next(
            s.unique_hash
            for s in slots
            if s.start_time == datetime.combine(today + timedelta(days=1), time(14, 0), tzinfo=tz)
        )

        result.expect.next_event().is_function_call(
            name="schedule_appointment", arguments={"slot_id": slot_id}
        )
        result.expect.next_event().is_agent_handoff(new_agent_type=beta.workflows.GetEmailTask)
        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(llm, intent="must ask for the email address")
        )

        result = await sess.run(
            user_input="My email address is theo@livekit.io",
            input_modality="audio",  # simulate audio input
        )
        result.expect.next_event().is_function_call(
            name="update_email_address", arguments={"email": "theo@livekit.io"}
        )
        result.expect.next_event().is_function_call_output()
        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(llm, intent="must ask for the email address confirmation/validation")
        )

        result = await sess.run(user_input="Yes, it's valid")
        result.expect.next_event().is_function_call(name="confirm_email_address")
        result.expect.next_event().is_function_call_output()
        result.expect.next_event().is_agent_handoff(new_agent_type=FrontDeskAgent)

        result.expect.next_event().is_function_call_output()  # output of the schedule_appointment
        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(llm, intent="must confirm the appointment was scheduled")
        )


@pytest.mark.asyncio
async def test_no_availability() -> None:
    userdata = Userdata(cal=FakeCalendar(timezone=TIMEZONE, slots=[]))  # no slots

    async with _llm_model() as llm, AgentSession(llm=llm, userdata=userdata) as sess:
        await sess.start(FrontDeskAgent(timezone=TIMEZONE))
        result = await sess.run(
            user_input="Hello, can I need an appointment, what's your availability for the next 2 weeks?"
        )
        result.expect.skip_next_event_if(type="message", role="assistant")
        result.expect.next_event().is_function_call(name="list_available_slots")
        result.expect.next_event().is_function_call_output()
        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(
                llm,
                intent="must say that there is no availability, especially in the requested time range. optionally, it can offer to look at other times",
            )
        )
