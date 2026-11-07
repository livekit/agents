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
def agent(clinic) -> Receptionist:
    return Receptionist(clinic)


@pytest.fixture
def context() -> FakeContext:
    return FakeContext(CallState())


async def test_calendar_is_locked_until_the_caller_is_verified(agent, context):
    with pytest.raises(ToolError, match="isn't verified"):
        await agent.check_availability(context, "2026-10-05", "2026-10-05")


async def test_a_corrected_spelling_finds_the_chart_behind_a_misheard_name(
    agent, context
):
    heard = json.loads(
        await agent.look_up_birthday(context, "1992-07-08", "Jordan Leigh")
    )
    spelled = json.loads(
        await agent.look_up_birthday(context, "1992-07-08", "Jordan Lee")
    )

    assert heard["charts"] == []
    assert [c["name"] for c in spelled["charts"]] == ["Jordan Lee"]


async def test_only_a_chart_on_the_looked_up_birthday_opens(agent, context):
    with pytest.raises(ToolError, match="date of birth first"):
        await agent.open_chart(context, "Taylor Smith")

    await agent.look_up_birthday(context, "1992-07-08", "Taylor Smith")
    with pytest.raises(ToolError, match="No chart named"):
        await agent.open_chart(context, "Taylor Smith")

    await agent.look_up_birthday(context, "1992-07-08", "Jordan Smith")
    with pytest.raises(ToolError, match="No chart named"):
        await agent.open_chart(context, "Jordan Lee")
    assert context.userdata.caller is None


async def test_new_patient_is_verified_once_registered(agent, context):
    await agent.look_up_birthday(context, "1995-06-01", "Casey Morgan")
    await agent.register_new_patient(context, "C A S E Y", "M O R G A N")

    assert context.userdata.caller.name == "Casey Morgan"
    [opening, *_] = json.loads(
        await agent.check_availability(context, "2026-10-05", "2026-10-05")
    )
    assert (
        await agent.book_appointment(context, opening["starts_at"], "Sore throat")
    ).startswith("Booked Sore throat for")
