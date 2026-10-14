from __future__ import annotations

import asyncio
import datetime as dt
import json
from types import SimpleNamespace
from typing import Any, cast

import pytest
from agent import PERSONA, NickAgent, RepairCallAgent, Userdata, _decided, _opener, _voicemail
from backend import PAYMENT, demo_desk
from call import NO_RESPONSE, SILENT_GOODBYE, STILL_THERE, hang_up_when_silent
from page import STATE_RPC_METHOD, dial_error, visitor_name

from livekit import api

pytestmark = pytest.mark.unit

TODAY = dt.date(2026, 9, 29)


class _Participant:
    def __init__(self, identity: str, name: str = "") -> None:
        self.identity = identity
        self.name = name


class _LocalParticipant:
    def __init__(self) -> None:
        self.sent: list[tuple[str, dict[str, Any]]] = []

    async def perform_rpc(self, *, destination_identity: str, method: str, payload: str) -> str:
        assert method == STATE_RPC_METHOD
        self.sent.append((destination_identity, json.loads(payload)))
        return "ok"


class _Room:
    """Just enough of rtc.Room for the page: who's in it, and RPCs out."""

    def __init__(self, *identities: str) -> None:
        self.remote_participants = {i: _Participant(i) for i in identities}
        self.local_participant = _LocalParticipant()
        self.handlers: dict[str, Any] = {}

    def on(self, event: str, fn: Any) -> None:
        self.handlers[event] = fn


class _Session:
    def on(self, event: str) -> Any:
        return lambda fn: fn


def _watched(room: _Room, **kw: Any) -> Userdata:
    ud = Userdata(desk=demo_desk(today=TODAY))
    ud.page.watch(cast(Any, room), cast(Any, _Session()), **kw)
    return ud


async def test_without_a_page_nothing_is_sent() -> None:
    room = _Room("mock_user")
    ud = _watched(room, caller=room.remote_participants["mock_user"])
    ud.page.record("lookup_customer", "Alex Morgan · 2 vehicles on file")
    await asyncio.sleep(0)
    assert not ud.page.watching and room.local_participant.sent == []
    # a test session without a room: the state is kept, nothing goes anywhere
    Userdata(desk=demo_desk(today=TODAY)).page.record("get_slots", "all days: 22 open")


async def test_the_browser_caller_and_viewers_get_every_push() -> None:
    room = _Room("caller-ab12", "viewer-cd34")
    ud = _watched(room, caller=room.remote_participants["caller-ab12"])
    ud.page.push()
    ud.desk.lookup_customer("4155550142", "Alex Morgan")
    ud.page.record("lookup_customer", "Alex Morgan · 2 vehicles on file")
    await asyncio.sleep(0)
    sent = room.local_participant.sent
    # the opener keeps the greeting; after that the stage follows the desk
    assert [m["stage"] for to, m in sent if to == "viewer-cd34"] == ["greeting", "vehicle"]
    await ud.page.ended("Call complete")
    assert {to for to, _ in sent} == {"caller-ab12", "viewer-cd34"}
    last = sent[-1][1]
    assert last["stage"] == "ended" and last["endReason"] == "Call complete"
    assert last["persona"] == PERSONA and last["direction"] == "inbound"
    assert [e["tool"] for e in last["log"]] == ["lookup_customer", "end_call"]
    assert last["data"]["customer"] == {
        "name": "Alex Morgan",
        "phone": "(415) 555-0142",
        "isNew": False,
    }


async def test_a_viewer_who_joins_late_sees_the_call_so_far() -> None:
    room = _Room()
    ud = _watched(room, page="viewer-ab12")
    ud.page.push(stage="reach")
    room.handlers["participant_connected"](_Participant("viewer-ef56"))
    await asyncio.sleep(0)
    assert [to for to, _ in room.local_participant.sent] == ["viewer-ab12", "viewer-ef56"]


class _Spoken:
    async def wait_for_playout(self) -> None:
        return None


class _QuietLine:
    """Just enough of AgentSession for hang_up_when_silent: its events, and what it says."""

    def __init__(self) -> None:
        self.handlers: dict[str, Any] = {}
        self.said: list[str] = []
        self.resets = 0

    def on(self, event: str) -> Any:
        def register(fn: Any) -> Any:
            self.handlers[event] = fn
            return fn

        return register

    def say(self, text: str, **_: Any) -> _Spoken:
        self.said.append(text)
        return _Spoken()

    def reset_away_timer(self) -> None:
        self.resets += 1

    def user(self, state: str) -> None:
        self.handlers["user_state_changed"](SimpleNamespace(new_state=state))


async def test_a_silent_line_gets_one_check_then_a_goodbye() -> None:
    room = _Room("viewer-ab12")
    ud = _watched(room)
    line = _QuietLine()
    hang_up_when_silent(cast(Any, line), ud.page)
    line.user("away")
    assert line.said == [STILL_THERE] and line.resets == 1
    # the check itself flips them back to listening: that isn't the caller speaking
    line.user("listening")
    line.user("away")
    await asyncio.sleep(0)
    assert line.said == [STILL_THERE, SILENT_GOODBYE]
    assert ud.page.stage == "ended" and ud.page.end_reason == NO_RESPONSE
    # nothing was decided on the silence
    assert ud.desk.appointment is None and ud.desk.pending is None


async def test_speaking_again_starts_the_silence_count_over() -> None:
    ud = _watched(_Room())
    line = _QuietLine()
    hang_up_when_silent(cast(Any, line), ud.page)
    line.user("away")
    line.user("speaking")
    line.user("listening")
    line.user("away")
    await asyncio.sleep(0)
    assert line.said == [STILL_THERE, STILL_THERE] and ud.page.stage != "ended"


def test_neither_agent_hands_the_call_off() -> None:
    inbound = NickAgent(demo_desk(today=TODAY))
    outbound = RepairCallAgent(demo_desk(today=TODAY, direction="outbound"))
    for agent in (inbound, outbound):
        assert not [t.id for t in agent.tools if "transfer" in t.id]
        said = str(agent.instructions)
        assert "never offer" in said and "transfer" not in said.replace("to transfer to", "")
        assert PAYMENT in said and "no discount codes" in said
        # a phone call: nothing to look at
        assert '"Does that sound right?"' in said and "sees them" not in said
    assert {"move_appointment", "cancel_appointment", "change_booking"} <= {
        t.id for t in inbound.tools
    }


def test_a_decision_leads_with_the_new_total() -> None:
    desk = demo_desk(today=TODAY, direction="outbound")
    desk.confirm_customer()
    f, notes = desk.decide_repair("front_brakes", "approved")
    said = _decided(desk, f"Approved: {f.title}, ${f.price:.0f}.", notes)
    assert said.startswith(
        "TOTAL DUE AT PICKUP IS NOW $648 (diagnostic $159 + front brake pads and rotors $489)."
    )


def test_the_follow_up_asks_for_the_customer_by_name() -> None:
    desk = demo_desk("Jordan Lee", today=TODAY, direction="outbound")
    assert (
        _opener(desk)
        == "Hi, this is Nick calling from Bayside BMW service. May I speak with Jordan?"
    )
    said = str(RepairCallAgent(desk).instructions)
    assert "outbound call to Jordan Lee" in said and "asked for Jordan." in said
    assert _voicemail(desk).startswith(
        "Hi, this is Nick from Bayside BMW service, calling for Jordan Lee."
    )


def test_without_a_name_the_follow_up_names_nobody() -> None:
    desk = demo_desk("", today=TODAY, direction="outbound")
    assert desk.customer is not None and desk.customer.name == ""
    assert desk.snapshot()["customer"]["name"] == "Customer"
    assert _opener(desk) == (
        "Hi, this is Nick from Bayside BMW service, calling about a car you left with us this "
        "morning. Is this the person who dropped it off?"
    )
    agent = RepairCallAgent(desk)
    said = str(agent.instructions)
    assert "never say or guess one" in said and "whoever dropped off the car" in said
    voicemail = _voicemail(desk)
    assert voicemail.startswith("Hi, this is Nick from Bayside BMW service. Please call us back")
    for spoken in (_opener(desk), said, voicemail):
        assert "Alex" not in spoken and "Morgan" not in spoken
        assert "with there" not in spoken and "for there" not in spoken
    # the read-backs say "the customer", never a name, and a name they give goes on the file
    desk.confirm_customer()
    desk.arrange_pickup("self", collected_by="their friend Sam")
    assert "the customer's name" in desk.ready_summary()
    desk.confirm_customer("jordan  lee")
    assert desk.customer.name == "jordan lee"
    desk.confirm_customer("Someone Else")
    assert desk.customer.name == "jordan lee"


async def test_without_a_name_the_wrong_person_hears_no_name() -> None:
    desk = demo_desk("", today=TODAY, direction="outbound")
    ud = Userdata(desk=desk)
    agent: Any = RepairCallAgent(desk)
    said = await agent.end_without_update(SimpleNamespace(userdata=ud), "wrong_person")
    assert "whoever dropped off the car" in said and "there" not in said.split()


def test_the_visitors_name_is_only_a_browser_callers() -> None:
    assert visitor_name(cast(Any, _Participant("caller-ab12", "Alex Kim"))) == "Alex Kim"
    assert visitor_name(cast(Any, _Participant("sip_+14155550142", "Phone"))) == ""
    assert visitor_name(None) == ""


@pytest.mark.parametrize(
    ("status", "want"),
    [
        ("408", "The phone didn't answer."),
        ("486", "The line was busy."),
        ("603", "The call was declined."),
        ("503", "The call couldn't be placed to +14372687803 (SIP 503: Service Unavailable)."),
    ],
)
def test_dial_error(status: str, want: str) -> None:
    reason = {"408": "Request Timeout", "486": "Busy Here", "603": "Decline"}.get(
        status, "Service Unavailable"
    )
    err = api.TwirpError(
        "unavailable",
        "call failed",
        status=503,
        metadata={"sip_status_code": status, "sip_status": reason},
    )
    assert dial_error(err, "+14372687803") == want
