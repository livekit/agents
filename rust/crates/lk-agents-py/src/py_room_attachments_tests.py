"""Native attachment ownership must not keep an otherwise unused SDK room alive."""

import gc
import weakref

import pytest

from livekit import rtc
from livekit.agents import RoomBinding, RoomIO, _core
from livekit.agents.voice.room_io import Room


@pytest.mark.asyncio
async def test_room_attachment_cache_does_not_create_an_opaque_native_reference_cycle() -> None:
    room = rtc.Room()
    reference = weakref.ref(room)
    Room(room)
    first = _core._core_create_room_session(room)
    second = _core._core_create_room_session(room)
    del first, second, room
    gc.collect()
    assert reference() is None


@pytest.mark.asyncio
async def test_named_bindings_share_reservations_and_reuse_keys_only_after_close() -> None:
    room = rtc.Room()
    io = RoomIO(room)
    another_view = RoomIO(room)
    first = io.bind("voice")
    assert isinstance(first, RoomBinding)
    assert first.key == "voice"
    assert first.room is room
    with pytest.raises(RuntimeError, match="already reserved"):
        another_view.bind("voice")
    await first.aclose()
    replacement = another_view.bind("voice")
    await replacement.aclose()
    await io.aclose()
    with pytest.raises(RuntimeError, match="closed"):
        another_view.bind("later")
