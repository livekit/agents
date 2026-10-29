"""The native receiver keeps outer EOF distinct from its inner audio/segment union."""

import pytest

from livekit import rtc
from livekit.agents import _core
from livekit.agents.voice.room_io import Room

pytestmark = pytest.mark.unit


async def test_a_closed_native_receiver_returns_the_typed_terminal_receipt() -> None:
    receiver = _core.DataStreamAudioReceiver(Room.of(rtc.Room()))
    try:
        receiver.aclose()
        assert await receiver.next_audio() == (0, False, None, None)
    finally:
        receiver.aclose()
