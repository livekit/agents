import json
from dataclasses import dataclass
from datetime import date, datetime, timezone

import pytest
from livekit.agents.llm import ToolError

import clinic as clinic_module
import receptionist
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


def test_callers_are_found_by_birthday_then_name(agent):
    tools = {t.info.name for t in agent.tools}
    assert {"look_up_birthday", "open_chart", "register_new_patient"} <= tools
    assert not tools & {"look_up_patient", "verify_patient"}


async def test_birthday_lists_the_charts_to_match_the_name_against(agent, context):
    result = json.loads(
        await agent.look_up_birthday(context, "1988-03-14", "Taylor Smith")
    )

    assert result["charts"] == [
        {"name": "Taylor Smith", "letters": "T A Y L O R, S M I T H"}
    ]
    assert context.userdata.caller is None


async def test_a_shared_birthday_lists_only_the_chart_spelled_as_given(
    agent, context, clinic
):
    clinic.patients["Tyler Smith"] = date(1988, 3, 14)
    clinic.patients["Taylor Smythe"] = date(1988, 3, 14)

    result = json.loads(
        await agent.look_up_birthday(context, "1988-03-14", "taylor smith")
    )
    assert [c["name"] for c in result["charts"]] == ["Taylor Smith"]

    result = json.loads(
        await agent.look_up_birthday(context, "1988-03-14", "Tailor Smith")
    )
    assert result["charts"] == []


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


async def test_opening_the_matched_chart_verifies_the_caller(agent, context):
    await agent.look_up_birthday(context, "1988-03-14", "Taylor Smith")
    result = json.loads(await agent.open_chart(context, "Taylor Smith"))

    assert result["upcoming"][0]["reason"] == "Knee pain follow-up"
    assert context.userdata.caller.name == "Taylor Smith"
    assert json.loads(
        await agent.check_availability(context, "2026-10-05", "2026-10-05")
    )


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


async def test_an_unknown_birthday_lists_no_charts(agent, context):
    result = json.loads(
        await agent.look_up_birthday(context, "1989-08-01", "Shayne Parmelee")
    )

    assert result["charts"] == []
    assert context.userdata.caller is None


async def test_new_patient_is_registered_under_the_confirmed_spelling(
    agent, context, clinic
):
    with pytest.raises(ToolError, match="date of birth before registering"):
        await agent.register_new_patient(context, "S H A Y N E", "P A R M E L E E")

    await agent.look_up_birthday(context, "1989-08-01", "Shayne Parmelee")
    await agent.register_new_patient(context, "S H A Y N E", "P A R M E L E E")

    assert context.userdata.caller.name == "Shayne Parmelee"
    assert clinic.patients["Shayne Parmelee"] == date(1989, 8, 1)


async def test_spelled_names_keep_their_hyphens_and_apostrophes(agent, context):
    await agent.look_up_birthday(context, "1990-01-01", "Mary-Kate O'Brien")
    await agent.register_new_patient(context, "M A R Y - K A T E", "O ' B R I E N")

    assert context.userdata.caller.name == "Mary-Kate O'Brien"
    result = json.loads(
        await agent.look_up_birthday(context, "1990-01-01", "mary kate obrien")
    )
    assert result["charts"][0]["letters"] == (
        "M A R Y hyphen K A T E, O apostrophe B R I E N"
    )


async def test_misheard_birth_date_cannot_open_a_second_chart(agent, context, clinic):
    await agent.look_up_birthday(context, "1988-03-04", "Taylor Smith")

    with pytest.raises(ToolError, match="already has a chart"):
        await agent.register_new_patient(context, "T A Y L O R", "S M I T H")
    assert clinic.patients["Taylor Smith"] == date(1988, 3, 14)
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


class LateEvening(datetime):
    """1:12 AM on the East Coast, still the evening before on the West Coast."""

    @classmethod
    def now(cls, tz=None):
        return datetime(2026, 10, 22, 5, 12, tzinfo=timezone.utc).astimezone(tz)


def test_riley_works_in_pacific_time(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(clinic_module, "datetime", LateEvening)
    monkeypatch.setattr(receptionist, "datetime", LateEvening)
    monkeypatch.setenv("LIVEKIT_API_KEY", "unused")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "unused")

    clinic = Clinic.for_now()
    riley = Receptionist(clinic)

    assert clinic.today == date(2026, 10, 21)
    assert "It's Wednesday, October 21, 10:12 PM Pacific time." in riley.instructions
