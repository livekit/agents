from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest
from common import Userdata
from fake_data.seed import build_seed_bytes
from hotel_db import TODAY, HotelDB
from phone_handoff import HANDOFF_GUEST_CODE, greet_on_phone_handoff

from livekit import rtc


@dataclass
class FakeParticipant:
    kind: int
    attributes: dict[str, str] = field(default_factory=dict)


@dataclass
class FakeRoom:
    handlers: dict[str, list[Any]] = field(default_factory=dict)

    def on(self, event: str, handler: Any) -> None:
        self.handlers.setdefault(event, []).append(handler)

    def emit(self, event: str, *args: Any) -> None:
        for handler in self.handlers.get(event, []):
            handler(*args)


@dataclass
class FakeAgent:
    instructions: str = "base instructions"

    async def update_instructions(self, instructions: str) -> None:
        self.instructions = instructions


@dataclass
class FakeSession:
    userdata: Userdata
    replies: list[str] = field(default_factory=list)

    def generate_reply(self, *, instructions: str) -> None:
        self.replies.append(instructions)


async def _settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


@pytest.fixture
async def setup() -> Any:
    db = HotelDB.from_bytes(build_seed_bytes(TODAY))
    session = FakeSession(userdata=Userdata(db=db))
    agent = FakeAgent()
    room = FakeRoom()
    greet_on_phone_handoff(session, agent, room)
    yield session, agent, room
    await db.aclose()


async def test_waits_for_the_call_to_be_answered(setup: Any) -> None:
    session, agent, room = setup
    phone = FakeParticipant(
        kind=rtc.ParticipantKind.PARTICIPANT_KIND_SIP, attributes={"sip.callStatus": "dialing"}
    )
    room.emit("participant_connected", phone)
    await _settle()
    assert session.replies == []
    assert session.userdata.verified_booking is None

    phone.attributes["sip.callStatus"] = "active"
    room.emit("participant_attributes_changed", {"sip.callStatus": "active"}, phone)
    await _settle()

    assert len(session.replies) == 1
    assert "Eleanor" in session.replies[0]
    booking = session.userdata.verified_booking
    assert booking is not None and booking.code == HANDOFF_GUEST_CODE
    assert agent.instructions.startswith("You're a receptionist at The LiveKit Hotel")
    assert "champagne or sparkling water" in agent.instructions
    assert "record_followup" in agent.instructions


async def test_starts_the_script_only_once(setup: Any) -> None:
    session, _agent, room = setup
    phone = FakeParticipant(
        kind=rtc.ParticipantKind.PARTICIPANT_KIND_SIP, attributes={"sip.callStatus": "active"}
    )
    room.emit("participant_connected", phone)
    room.emit("participant_attributes_changed", {"sip.callStatus": "active"}, phone)
    await _settle()
    assert len(session.replies) == 1


async def test_ignores_web_visitors(setup: Any) -> None:
    session, agent, room = setup
    room.emit(
        "participant_connected",
        FakeParticipant(kind=rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD),
    )
    await _settle()
    assert session.replies == []
    assert agent.instructions == "base instructions"
