import json
from dataclasses import dataclass
from datetime import date

import pytest
from livekit.agents.llm import ToolError

from clinic import Clinic
from receptionist import (
    CallState,
    ConfirmationCall,
    outbound_caller_script,
    outbound_greeting,
)

MONDAY = date(2026, 9, 28)


@dataclass
class FakeContext:
    userdata: CallState


@pytest.fixture
def published() -> dict:
    return {}


@pytest.fixture
def clinic(published: dict) -> Clinic:
    async def publish(key, value):
        published[key] = value

    return Clinic(today=MONDAY, publish=publish)


@pytest.fixture
def appointment(clinic: Clinic):
    return clinic.appointments[0]


@pytest.fixture
def agent(monkeypatch: pytest.MonkeyPatch, clinic, appointment) -> ConfirmationCall:
    monkeypatch.setenv("LIVEKIT_API_KEY", "unused")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "unused")
    return ConfirmationCall(clinic, appointment)


@pytest.fixture
def context() -> FakeContext:
    return FakeContext(CallState())


async def verify(agent, context, clinic, appointment) -> dict:
    dob = clinic.patients[appointment.patient].isoformat()
    return json.loads(await agent.verify_patient(context, dob))


async def test_nothing_changes_before_the_patient_is_verified(
    agent, context, appointment
):
    with pytest.raises(ToolError):
        await agent.confirm_appointment(context, appointment.id)
    with pytest.raises(ToolError):
        await agent.add_note_for_doctor(context, appointment.id, "knee hurts")


async def test_any_date_of_birth_verifies_in_the_demo(
    agent, context, appointment, published
):
    verified = json.loads(await agent.verify_patient(context, "2000-01-01"))
    assert verified["verified"] and context.userdata.caller.name == appointment.patient
    assert published["caller"] == {"name": appointment.patient, "new": False}


async def test_confirming_and_a_note_reach_the_published_calendar(
    agent, context, clinic, appointment, published
):
    verified = await verify(agent, context, clinic, appointment)
    assert verified["upcoming"][0]["appointment_id"] == appointment.id

    await agent.confirm_appointment(context, appointment.id)
    await agent.add_note_for_doctor(context, appointment.id, "Knee swells after runs.")
    [published_appointment] = [
        a for a in published["appointments"] if a["id"] == appointment.id
    ]
    assert published_appointment["confirmed"] is True
    assert published_appointment["notes"] == ["Knee swells after runs."]


async def test_an_empty_note_is_refused(agent, context, clinic, appointment):
    await verify(agent, context, clinic, appointment)
    with pytest.raises(ToolError):
        await agent.add_note_for_doctor(context, appointment.id, "  ")


async def test_only_the_patients_own_appointment_can_be_confirmed(
    agent, context, clinic, appointment
):
    await verify(agent, context, clinic, appointment)
    other = next(a for a in clinic.appointments if a.patient != appointment.patient)
    with pytest.raises(ToolError):
        await agent.confirm_appointment(context, other.id)


def test_inbound_only_tools_are_not_on_the_outbound_call(agent):
    tools = {t.info.name for t in agent.tools}
    assert {"confirm_appointment", "add_note_for_doctor", "verify_patient"} <= tools
    assert not tools & {
        "look_up_birthday",
        "open_chart",
        "register_new_patient",
        "book_appointment",
    }


def test_the_greeting_names_the_patient_but_nothing_medical(appointment, clinic):
    greeting = outbound_greeting(appointment)
    assert appointment.patient in greeting
    assert appointment.reason not in greeting
    assert clinic.patients[appointment.patient].isoformat() not in greeting


async def test_script_follows_the_call_from_verified_to_the_note(
    agent, context, clinic, appointment
):
    script = lambda: outbound_caller_script(clinic, context.userdata, appointment)  # noqa: E731
    assert "date of birth" in script()
    await verify(agent, context, clinic, appointment)
    assert "say yes" in script()
    await agent.confirm_appointment(context, appointment.id)
    assert "ibuprofen" in script().splitlines()[-1]
    await agent.add_note_for_doctor(context, appointment.id, "Knee swells after runs.")
    assert "goodbye" in script()
