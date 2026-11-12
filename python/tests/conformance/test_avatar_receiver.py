"""The native receiver keeps outer EOF distinct from its inner audio/segment union."""

import pytest

from livekit import rtc
from livekit.agents import _core
from livekit.agents._generated.avatar import AudioReceiverObservation

pytestmark = pytest.mark.unit


async def test_a_closed_native_receiver_returns_the_typed_terminal_receipt() -> None:
    receiver = _core.DataStreamAudioReceiver(rtc.Room())
    try:
        receiver.aclose()
        assert await receiver.next_audio() == AudioReceiverObservation(
            epoch=0, starts_segment=False, segment=None, frame=None
        )
    finally:
        receiver.aclose()
