from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from common import Userdata
from fake_data.seed import build_seed_bytes
from hotel_db import TODAY, HotelDB
from tools_rooms import RoomToolsMixin

from livekit.agents import ToolError

pytestmark = pytest.mark.unit


def _userdata() -> Userdata:
    return Userdata(db=HotelDB.from_bytes(build_seed_bytes(TODAY)))


@pytest.mark.asyncio
async def test_start_room_booking_bounces_while_a_flow_is_in_flight() -> None:
    # Parallel inline AgentTasks deadlock the session (the second activation
    # orphans the first call's future), so a same-turn second start_room_booking
    # must return a ToolError instead of starting another BookRoomTask.
    userdata = _userdata()
    userdata.room_booking_in_flight = True
    ctx = SimpleNamespace(userdata=userdata)
    with pytest.raises(ToolError, match="already in progress"):
        await RoomToolsMixin().start_room_booking(ctx)
