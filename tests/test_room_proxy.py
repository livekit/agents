"""RTC host proxies preserve SDK-room construction and disconnected wait behavior."""

import asyncio

import pytest

from livekit import rtc
from livekit.agents import _core
from livekit.agents.voice.room_io import Room

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module", autouse=True)
def initialize_native_runtime() -> None:
    _core.init(_core.Role.Job)


@pytest.mark.asyncio
async def test_disconnected_sdk_room_keeps_the_framework_wait_error() -> None:
    room = rtc.Room()
    supplied = Room.of(room)
    capability = _core._core_create_room_session(supplied)
    try:
        with pytest.raises(RuntimeError, match="room is not connected"):
            await asyncio.wait_for(_core.wait_for_participant(capability, None, None, False), 2)
    finally:
        await room.disconnect()


@pytest.mark.asyncio
async def test_native_avatar_output_accepts_the_sdk_room_used_by_its_public_alias() -> None:
    room = rtc.Room()
    try:
        output = _core.DataStreamAudioOutput(room, destination_identity="avatar")
        await output.aclose()
    finally:
        await room.disconnect()
