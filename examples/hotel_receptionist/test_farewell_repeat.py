from __future__ import annotations

import os
import sys
from typing import cast

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from common import Userdata
from hotel_db import HotelDB
from tools_services import _farewell_instruction


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
