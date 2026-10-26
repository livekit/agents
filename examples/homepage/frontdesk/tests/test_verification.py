import json
from dataclasses import dataclass
from datetime import date

import pytest
from livekit.agents.llm import ToolError

from clinic import Clinic
from receptionist import CallState, Receptionist

MONDAY = date(2026, 9, 28)


@dataclass
class FakeContext:
    userdata: CallState


@pytest.fixture
def clinic() -> Clinic:
    return Clinic(today=MONDAY)


@pytest.fixture
def agent(monkeypatch: pytest.MonkeyPatch, clinic) -> Receptionist:
    monkeypatch.setenv("LIVEKIT_API_KEY", "unused")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "unused")
    return Receptionist(clinic)


@pytest.fixture
def context() -> FakeContext:
    return FakeContext(CallState())


async def test_calendar_is_locked_until_the_caller_is_verified(agent, context):
    with pytest.raises(ToolError, match="isn't verified"):
        await agent.check_availability(context, "2026-10-05", "2026-10-05")


async def test_booking_and_moving_are_locked_until_verified(agent, context):
    with pytest.raises(ToolError, match="isn't verified"):
        await agent.book_appointment(context, "2026-10-05T09:30", "Checkup")
    with pytest.raises(ToolError, match="isn't verified"):
        await agent.reschedule_appointment(context, "A-101", "2026-10-05T09:30")


def test_callers_are_looked_up_only_by_spelled_name(agent):
    tools = {t.info.name for t in agent.tools}
    assert {"look_up_patient", "register_new_patient"} <= tools
    assert "verify_patient" not in tools


async def test_spelled_name_and_birth_date_find_the_chart(agent, context):
    result = json.loads(
        await agent.look_up_patient(context, "T A Y L O R", "S M I T H", "1988-03-14")
    )

    assert result["found"] is True
    assert result["upcoming"][0]["reason"] == "Knee pain follow-up"
    assert context.userdata.caller.name == "Taylor Smith"
    assert json.loads(
        await agent.check_availability(context, "2026-10-05", "2026-10-05")
    )


async def test_no_chart_reveals_nothing_on_file(agent, context):
    result = json.loads(
        await agent.look_up_patient(context, "T A Y L O R", "S M I T H", "1990-01-01")
    )

    assert result["found"] is False
    assert result["letters"] == "T A Y L O R, S M I T H"
    assert "upcoming" not in result
    assert context.userdata.caller is None


async def test_new_patient_is_registered_only_after_a_lookup(agent, context, clinic):
    with pytest.raises(ToolError, match="spelled name"):
        await agent.register_new_patient(context)

    result = json.loads(
        await agent.look_up_patient(
            context, "S H A Y N E", "P A R M E L E E", "1989-08-01"
        )
    )

    assert result["found"] is False
    assert result["letters"] == "S H A Y N E, P A R M E L E E"
    assert context.userdata.caller is None
    assert "Shayne Parmelee" not in clinic.patients

    await agent.register_new_patient(context)

    assert context.userdata.caller.name == "Shayne Parmelee"
    assert clinic.patients["Shayne Parmelee"] == date(1989, 8, 1)


async def test_spelled_names_keep_their_hyphens_and_apostrophes(agent, context):
    result = json.loads(
        await agent.look_up_patient(
            context, "M A R Y - K A T E", "O ' B R I E N", "1990-01-01"
        )
    )

    assert result["letters"] == "M A R Y hyphen K A T E, O apostrophe B R I E N"
    await agent.register_new_patient(context)
    assert context.userdata.caller.name == "Mary-Kate O'Brien"


async def test_misheard_birth_date_cannot_open_a_second_chart(agent, context, clinic):
    await agent.look_up_patient(context, "T A Y L O R", "S M I T H", "1988-03-04")

    with pytest.raises(ToolError, match="already has a chart"):
        await agent.register_new_patient(context)
    assert clinic.patients["Taylor Smith"] == date(1988, 3, 14)
    assert context.userdata.caller is None


async def test_corrected_spelling_that_finds_the_chart_registers_nothing(
    agent, context
):
    await agent.look_up_patient(context, "T A Y L E R", "S M I T H", "1988-03-14")
    await agent.look_up_patient(context, "T A Y L O R", "S M I T H", "1988-03-14")

    assert context.userdata.caller.name == "Taylor Smith"
    with pytest.raises(ToolError):
        await agent.register_new_patient(context)


async def test_new_patient_is_verified_once_registered(agent, context):
    await agent.look_up_patient(context, "C A S E Y", "M O R G A N", "1995-06-01")
    await agent.register_new_patient(context)

    assert context.userdata.caller.name == "Casey Morgan"
    [opening, *_] = json.loads(
        await agent.check_availability(context, "2026-10-05", "2026-10-05")
    )
    assert (
        await agent.book_appointment(context, opening["starts_at"], "Sore throat")
    ).startswith("Booked Sore throat for")
