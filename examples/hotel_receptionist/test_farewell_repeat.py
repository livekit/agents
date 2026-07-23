from __future__ import annotations

import asyncio
import os
import sys
from types import SimpleNamespace
from typing import cast

import pytest

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from common import Userdata
from hotel_db import HotelDB
from tools_services import _arm_close_watchdog, _farewell_instruction


def _userdata() -> Userdata:
    return Userdata(db=cast(HotelDB, None))


def test_first_close_delivers_the_goodbye() -> None:
    userdata = _userdata()
    out = _farewell_instruction(userdata)
    assert "ONE short, warm goodbye" in out
    assert userdata.goodbye_said is True


def test_repeat_close_forbids_another_farewell() -> None:
    # the caller answered the farewell ("you too!") and the model called the tool
    # again - no second goodbye, no "Take care!" loop.
    userdata = _userdata()
    _farewell_instruction(userdata)
    out = _farewell_instruction(userdata)
    assert "goodbye" in out.lower()
    assert "already" in out.lower()
    assert "ONE short, warm goodbye" not in out


def test_repeat_close_stays_forbidden() -> None:
    userdata = _userdata()
    first = _farewell_instruction(userdata)
    assert _farewell_instruction(userdata) != first
    assert _farewell_instruction(userdata) == _farewell_instruction(userdata)


# --- the close watchdog ---------------------------------------------------------


class _FakeSession:
    def __init__(self) -> None:
        self.handlers: dict[str, list] = {}
        self.shutdowns = 0

    def on(self, event: str, cb) -> None:
        self.handlers.setdefault(event, []).append(cb)

    def off(self, event: str, cb) -> None:
        self.handlers[event].remove(cb)

    def shutdown(self) -> None:
        self.shutdowns += 1

    def caller_speaks(self) -> None:
        ev = SimpleNamespace(item=SimpleNamespace(role="user"))
        for cb in list(self.handlers.get("conversation_item_added", [])):
            cb(ev)


@pytest.mark.asyncio
async def test_close_fires_after_silence() -> None:
    session = _FakeSession()
    task = _arm_close_watchdog(session, grace=0.01, reopen_on_caller_speech=True)
    await task
    assert session.shutdowns == 1


@pytest.mark.asyncio
async def test_first_close_is_cancelled_when_the_caller_speaks() -> None:
    # anything the caller says after the first farewell re-opens the conversation
    session = _FakeSession()
    task = _arm_close_watchdog(session, grace=0.05, reopen_on_caller_speech=True)
    await asyncio.sleep(0)
    session.caller_speaks()
    await task
    assert session.shutdowns == 0
    # the listener is removed once the watchdog is done
    assert session.handlers.get("conversation_item_added", []) == []


@pytest.mark.asyncio
async def test_repeat_close_hangs_up_despite_caller_chatter() -> None:
    # after the SECOND close, caller acknowledgments ("okay, heading down now")
    # must not keep the line open forever - the goodbye already happened.
    session = _FakeSession()
    task = _arm_close_watchdog(session, grace=0.05, reopen_on_caller_speech=False)
    await asyncio.sleep(0)
    session.caller_speaks()
    session.caller_speaks()
    await task
    assert session.shutdowns == 1
