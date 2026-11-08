"""RTC host proxies preserve SDK-room construction and disconnected wait behavior."""

import asyncio

import pytest

from livekit import rtc
from livekit.agents import _core

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module", autouse=True)
def initialize_native_runtime() -> None:
    _core.init(_core.Role.Job)


@pytest.mark.asyncio
async def test_disconnected_sdk_room_keeps_the_framework_wait_error() -> None:
    room = rtc.Room()
    capability = _core._core_create_room_session(room)
    try:
        with pytest.raises(RuntimeError, match="room is not connected"):
            await asyncio.wait_for(_core.wait_for_participant(capability, None, None, False), 2)
    finally:
        await room.disconnect()
