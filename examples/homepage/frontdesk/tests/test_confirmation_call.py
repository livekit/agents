import json
from dataclasses import dataclass
from datetime import date

import pytest

from clinic import Clinic
from receptionist import (
    CallState,
    ConfirmationCall,
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
def agent(clinic, appointment) -> ConfirmationCall:
    return ConfirmationCall(clinic, appointment)


@pytest.fixture
def context() -> FakeContext:
    return FakeContext(CallState())


async def test_any_date_of_birth_verifies_in_the_demo(
    agent, context, appointment, published
):
    verified = json.loads(await agent.verify_patient(context, "2000-01-01"))
    assert verified["verified"] and context.userdata.caller.name == appointment.patient
    assert published["caller"] == {"name": appointment.patient, "new": False}
