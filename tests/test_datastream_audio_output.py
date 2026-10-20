from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest

from livekit import rtc
from livekit.agents.voice.avatar import _datastream_io as ds
from livekit.agents.voice.recorder_io.recorder_io import RecorderAudioOutput

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]


def _frame() -> rtc.AudioFrame:
    return rtc.AudioFrame.create(sample_rate=24000, num_channels=1, samples_per_channel=240)


def _finish(output: ds.DataStreamAudioOutput, *, interrupted: bool = False) -> None:
    data = MagicMock(spec=rtc.RpcInvocationData)
    data.caller_identity = "avatar"
    data.payload = json.dumps({"playback_position": 0.01, "interrupted": interrupted})
    assert output._handle_playback_finished(data) == "ok"


@pytest.fixture
async def sink(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[tuple[ds.DataStreamAudioOutput, MagicMock]]:
    room = MagicMock()
    room.isconnected.return_value = True
    room.local_participant._rpc_handlers = {}
    room.local_participant.perform_rpc = AsyncMock(return_value="")
    writer = MagicMock(spec=rtc.ByteStreamWriter)
    writer.write = AsyncMock()
    writer.aclose = AsyncMock()
    room.local_participant.stream_bytes = AsyncMock(return_value=writer)
    monkeypatch.setattr(ds.utils, "wait_for_participant", AsyncMock())

    output = ds.DataStreamAudioOutput(room, destination_identity="avatar")
    await output._start_task()
    try:
        yield output, room
    finally:
        if output._clear_buffer_timeout_handler:
            output._clear_buffer_timeout_handler.cancel()
        await asyncio.gather(*output._tasks)
        ds.DataStreamAudioOutput._playback_finished_handlers.pop("avatar", None)


@pytest.mark.parametrize("prior_reply", [False, True])
async def test_empty_clear_does_not_finish_next_reply(
    sink: tuple[ds.DataStreamAudioOutput, MagicMock], prior_reply: bool
) -> None:
    leaf, room = sink
    output = RecorderAudioOutput(
        recording_io=MagicMock(recording=False), audio_output=leaf, on_played=MagicMock()
    )
    if prior_reply:
        await output.capture_frame(_frame())
        output.flush()
        _finish(leaf)
        await output.wait_for_playout()

    # The interrupted reply never reached capture_frame().
    output.flush()
    output.clear_buffer()
    await asyncio.gather(*leaf._tasks)

    async def slow_write(_data: bytes) -> None:
        await asyncio.sleep(3.0)

    room.local_participant.stream_bytes.return_value.write.side_effect = slow_write
    await output.capture_frame(_frame())
    output.flush()
    _finish(leaf)

    event = await asyncio.wait_for(output.wait_for_playout(), timeout=1.0)
    assert not event.interrupted


@pytest.mark.parametrize("next_reply_started", [False, True])
async def test_clear_response_after_finish_does_not_arm_timeout_for_next_reply(
    sink: tuple[ds.DataStreamAudioOutput, MagicMock], next_reply_started: bool
) -> None:
    output, room = sink
    await output.capture_frame(_frame())
    output.flush()

    response = asyncio.Event()
    called = asyncio.Event()

    async def clear_rpc(**_kwargs: object) -> str:
        called.set()
        await response.wait()
        return ""

    room.local_participant.perform_rpc.side_effect = clear_rpc
    output.clear_buffer()
    await called.wait()
    _finish(output, interrupted=True)
    await output.wait_for_playout()

    if next_reply_started:
        await output.capture_frame(_frame())
    response.set()
    await asyncio.gather(*output._tasks)
    if not next_reply_started:
        await output.capture_frame(_frame())
    await asyncio.sleep(3.0)
    output.flush()
    _finish(output)

    event = await asyncio.wait_for(output.wait_for_playout(), timeout=1.0)
    assert not event.interrupted


async def test_remote_finish_cancels_clear_timeout(
    sink: tuple[ds.DataStreamAudioOutput, MagicMock],
) -> None:
    output, _ = sink
    await output.capture_frame(_frame())
    output.flush()
    output.clear_buffer()
    await asyncio.gather(*output._tasks)
    _finish(output, interrupted=True)
    event = await asyncio.wait_for(output.wait_for_playout(), timeout=1.0)
    assert event.interrupted

    await output.capture_frame(_frame())
    await asyncio.sleep(3.0)
    output.flush()
    _finish(output)
    event = await asyncio.wait_for(output.wait_for_playout(), timeout=1.0)
    assert not event.interrupted


@pytest.mark.parametrize("rpc_fails", [False, True])
async def test_clear_fallback_releases_waiter_and_preserves_next_reply(
    sink: tuple[ds.DataStreamAudioOutput, MagicMock], rpc_fails: bool
) -> None:
    output, room = sink
    if rpc_fails:
        room.local_participant.perform_rpc.side_effect = RuntimeError("clear failed")

    await output.capture_frame(_frame())
    output.flush()
    waiter = asyncio.create_task(output.wait_for_playout())
    await asyncio.sleep(0)
    output.clear_buffer()

    event = await asyncio.wait_for(waiter, timeout=3.0)
    assert event.interrupted
    assert event.playback_position == pytest.approx(0.01)

    await output.capture_frame(_frame())
    output.flush()
    _finish(output)
    event = await asyncio.wait_for(output.wait_for_playout(), timeout=1.0)
    assert not event.interrupted
