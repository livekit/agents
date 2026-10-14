from __future__ import annotations

import asyncio
import json
from typing import Any, cast

import pytest
from agent import PERSONA, Userdata
from backend import demo_pos
from page import STATE_RPC_METHOD, dial_error, visitor_name

from livekit import api

pytestmark = pytest.mark.unit


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
    ud = Userdata(pos=demo_pos())
    ud.page.watch(cast(Any, room), cast(Any, _Session()), **kw)
    return ud


async def test_without_a_page_nothing_is_sent() -> None:
    room = _Room("mock_user")
    ud = _watched(room, caller=room.remote_participants["mock_user"])
    ud.page.record("add_item", "L1: 1 × large Pepperoni", stage="order")
    await asyncio.sleep(0)
    assert not ud.page.watching and room.local_participant.sent == []
    # a test session without a room: the state is kept, nothing goes anywhere
    Userdata(pos=demo_pos()).page.record("get_menu", "17 items")


async def test_the_browser_caller_and_viewers_get_every_push() -> None:
    room = _Room("caller-ab12", "viewer-cd34")
    ud = _watched(room, caller=room.remote_participants["caller-ab12"])
    line = ud.pos.add("pepperoni", 1, "large")
    ud.page.record("add_item", line.describe(), stage="order")
    await ud.page.ended("Call complete")
    sent = room.local_participant.sent
    assert {to for to, _ in sent} == {"caller-ab12", "viewer-cd34"}
    last = sent[-1][1]
    assert last["stage"] == "ended" and last["endReason"] == "Call complete"
    assert last["persona"] == PERSONA and last["direction"] == "inbound"
    assert [e["tool"] for e in last["log"]] == ["add_item", "end_call"]
    assert last["data"]["cart"]["lines"][0]["size"] == "large"


async def test_a_viewer_who_joins_late_sees_the_call_so_far() -> None:
    room = _Room()
    ud = _watched(room, page="viewer-ab12")
    ud.page.push(stage="update")
    room.handlers["participant_connected"](_Participant("viewer-ef56"))
    await asyncio.sleep(0)
    assert [to for to, _ in room.local_participant.sent] == ["viewer-ab12", "viewer-ef56"]


def test_the_visitors_name_is_only_a_browser_callers() -> None:
    assert visitor_name(cast(Any, _Participant("caller-ab12", "Sam Carter"))) == "Sam Carter"
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
