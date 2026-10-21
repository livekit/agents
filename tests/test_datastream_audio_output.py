from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest

from livekit import rtc
from livekit.agents.voice.avatar import _datastream_io as ds
from livekit.agents.voice.recorder_io.recorder_io import RecorderAudioOutput
from livekit.agents.voice.transcription.synchronizer import TranscriptSynchronizer

from .fake_io import FakeTextOutput
from .test_audio_sink_proxy import _PassthroughWrapper

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]


def _frame() -> rtc.AudioFrame:
    return rtc.AudioFrame.create(sample_rate=24000, num_channels=1, samples_per_channel=240)


def _finish(output: ds.DataStreamAudioOutput, *, interrupted: bool = False) -> None:
    data = MagicMock(spec=rtc.RpcInvocationData)
    data.caller_identity = "avatar"
    data.payload = json.dumps({"playback_position": 0.01, "interrupted": interrupted})
    assert output._handle_playback_finished(data) == "ok"


@pytest.fixture
def clear_buffer_timeout() -> float | None:
    return 2.0


@pytest.fixture
async def sink(
    monkeypatch: pytest.MonkeyPatch,
    clear_buffer_timeout: float | None,
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

    output = ds.DataStreamAudioOutput(
        room, destination_identity="avatar", clear_buffer_timeout=clear_buffer_timeout
    )
    await output._start_task()
    try:
        yield output, room
    finally:
        await asyncio.gather(*output._tasks)
        ds.DataStreamAudioOutput._playback_finished_handlers.pop("avatar", None)


@pytest.mark.parametrize("clear_buffer_timeout", [None])
@pytest.mark.parametrize("rpc_pending", [False, True])
async def test_clear_without_timeout_does_not_block_next_reply(
    sink: tuple[ds.DataStreamAudioOutput, MagicMock], rpc_pending: bool
) -> None:
    leaf, room = sink
    output = _PassthroughWrapper(next_in_chain=leaf)
    await output.capture_frame(_frame())
    output.flush()
    response = asyncio.Event()
    called = asyncio.Event()
    if not rpc_pending:
        response.set()

    async def clear_rpc(**_kwargs: object) -> str:
        called.set()
        await response.wait()
        return ""

    room.local_participant.perform_rpc.side_effect = clear_rpc
    output.clear_buffer()
    await called.wait()
    try:
        await asyncio.wait_for(output.capture_frame(_frame()), timeout=1.0)
        output.flush()
        await asyncio.sleep(3.0)
        assert room.local_participant.stream_bytes.await_count == 2
        assert output._pending_playback_count == 2

        response.set()
        await asyncio.wait_for(asyncio.gather(*leaf._tasks), timeout=1.0)
        assert output._pending_playback_count == 2
    finally:
        response.set()
        while leaf._pending_playback_count:
            _finish(leaf)


@pytest.mark.parametrize("clear_buffer_timeout", [None])
@pytest.mark.parametrize("rpc_fails", [False, True])
async def test_clear_without_timeout_sends_overlapping_requests(
    sink: tuple[ds.DataStreamAudioOutput, MagicMock], rpc_fails: bool
) -> None:
    output, room = sink
    calls: asyncio.Queue[asyncio.Future[str]] = asyncio.Queue()
    responses: list[asyncio.Future[str]] = []

    async def clear_rpc(**_kwargs: object) -> str:
        response = asyncio.Future[str]()
        calls.put_nowait(response)
        return await response

    room.local_participant.perform_rpc.side_effect = clear_rpc
    try:
        for _ in range(2):
            await asyncio.wait_for(output.capture_frame(_frame()), timeout=1.0)
            output.flush()
            output.clear_buffer()
            responses.append(await asyncio.wait_for(calls.get(), timeout=1.0))

        # The later clear can succeed before the first clear returns.
        responses[1].set_result("")
        if rpc_fails:
            responses[0].set_exception(RuntimeError("clear failed"))
        else:
            responses[0].set_result("")
        await asyncio.wait_for(asyncio.gather(*output._tasks), timeout=1.0)
        await asyncio.sleep(3.0)
        assert output._pending_playback_count == (1 if rpc_fails else 2)
        assert room.local_participant.perform_rpc.await_count == 2
    finally:
        for response in responses:
            if not response.done():
                response.set_result("")
        while output._pending_playback_count:
            _finish(output)


@pytest.mark.parametrize("clear_buffer_timeout", [None])
async def test_clear_rpc_failure_keeps_boundary_before_task_starts(
    sink: tuple[ds.DataStreamAudioOutput, MagicMock],
) -> None:
    output, room = sink
    await output.capture_frame(_frame())
    output.flush()
    room.local_participant.perform_rpc.side_effect = RuntimeError("clear failed")

    output.clear_buffer()
    await output.capture_frame(_frame())
    output.flush()
    await asyncio.gather(*output._tasks)
    assert output._pending_playback_count == 1

    _finish(output)
    event = await asyncio.wait_for(output.wait_for_playout(), timeout=1.0)
    assert not event.interrupted


@pytest.mark.parametrize("clear_buffer_timeout", [None])
@pytest.mark.parametrize("remote_finishes", [0, 1, 2, 3])
async def test_clear_without_timeout_rpc_failure_finishes_only_original_segments(
    sink: tuple[ds.DataStreamAudioOutput, MagicMock], remote_finishes: int
) -> None:
    output, room = sink
    for _ in range(2):
        await output.capture_frame(_frame())
        output.flush()
    response = asyncio.Future[str]()
    called = asyncio.Event()
    on_finish = MagicMock()
    output.on("playback_finished", on_finish)

    async def clear_rpc(**_kwargs: object) -> str:
        called.set()
        return await response

    room.local_participant.perform_rpc.side_effect = clear_rpc
    output.clear_buffer()
    await called.wait()
    try:
        await asyncio.wait_for(output.capture_frame(_frame()), timeout=1.0)
        output.flush()
        for _ in range(remote_finishes):
            _finish(output)
        response.set_exception(RuntimeError("clear failed"))
        await asyncio.wait_for(asyncio.gather(*output._tasks), timeout=1.0)

        assert on_finish.call_count == max(2, remote_finishes)
        assert output._pending_playback_count == (0 if remote_finishes == 3 else 1)
        if output._pending_playback_count:
            _finish(output)
        event = await asyncio.wait_for(output.wait_for_playout(), timeout=1.0)
        assert not event.interrupted
    finally:
        if not response.done():
            response.set_result("")
        while output._pending_playback_count:
            _finish(output)


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
async def test_clear_response_after_finish_does_not_finish_next_reply(
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

    capture_task = None
    if next_reply_started:
        capture_task = asyncio.create_task(output.capture_frame(_frame()))
        await asyncio.sleep(0)
    response.set()
    await asyncio.gather(*output._tasks)
    if capture_task is not None:
        await capture_task
    else:
        await output.capture_frame(_frame())
    await asyncio.sleep(3.0)
    output.flush()
    _finish(output)

    event = await asyncio.wait_for(output.wait_for_playout(), timeout=1.0)
    assert not event.interrupted


async def test_remote_finish_releases_clear_before_timeout(
    sink: tuple[ds.DataStreamAudioOutput, MagicMock],
) -> None:
    output, _ = sink
    await output.capture_frame(_frame())
    output.flush()
    output.clear_buffer()
    await asyncio.sleep(0)
    _finish(output, interrupted=True)
    await asyncio.gather(*output._tasks)
    event = await asyncio.wait_for(output.wait_for_playout(), timeout=1.0)
    assert event.interrupted

    await output.capture_frame(_frame())
    await asyncio.sleep(3.0)
    output.flush()
    _finish(output)
    event = await asyncio.wait_for(output.wait_for_playout(), timeout=1.0)
    assert not event.interrupted


async def test_new_segment_waits_for_clear_rpc_and_keeps_its_finish(
    sink: tuple[ds.DataStreamAudioOutput, MagicMock],
) -> None:
    output, room = sink
    await output.capture_frame(_frame())
    output.flush()
    interrupted_waiter = asyncio.create_task(output.wait_for_playout())
    await asyncio.sleep(0)

    called = asyncio.Event()
    response = asyncio.Event()

    async def clear_rpc(**_kwargs: object) -> str:
        called.set()
        await response.wait()
        return ""

    room.local_participant.perform_rpc.side_effect = clear_rpc
    output.clear_buffer()
    await called.wait()
    capture_task = asyncio.create_task(output.capture_frame(_frame()))
    try:
        await asyncio.sleep(0)
        assert room.local_participant.stream_bytes.await_count == 1
        response.set()
        await asyncio.sleep(0)
        assert room.local_participant.stream_bytes.await_count == 1
        await capture_task
        output.flush()

        event = await asyncio.wait_for(interrupted_waiter, timeout=3.0)
        assert event.interrupted
        assert output._pending_playback_count == 1

        _finish(output)
        event = await asyncio.wait_for(output.wait_for_playout(), timeout=1.0)
        assert not event.interrupted
    finally:
        response.set()
        await asyncio.gather(capture_task, return_exceptions=True)
        interrupted_waiter.cancel()
        await asyncio.gather(interrupted_waiter, return_exceptions=True)


@pytest.mark.parametrize("remote_finished", [False, True])
@pytest.mark.parametrize("with_recorder", [False, True])
async def test_cancelled_capture_releases_wrapper_without_cancelling_clear(
    sink: tuple[ds.DataStreamAudioOutput, MagicMock], remote_finished: bool, with_recorder: bool
) -> None:
    leaf, room = sink
    inner = (
        RecorderAudioOutput(
            recording_io=MagicMock(recording=False), audio_output=leaf, on_played=MagicMock()
        )
        if with_recorder
        else leaf
    )
    output = _PassthroughWrapper(next_in_chain=inner)
    await output.capture_frame(_frame())
    output.flush()
    called = asyncio.Event()
    response = asyncio.Event()

    async def clear_rpc(**_kwargs: object) -> str:
        called.set()
        await response.wait()
        return ""

    room.local_participant.perform_rpc.side_effect = clear_rpc
    output.clear_buffer()
    await called.wait()
    if remote_finished:
        _finish(leaf, interrupted=True)
    capture_task = asyncio.create_task(output.capture_frame(_frame()))
    try:
        await asyncio.sleep(0)
        capture_task.cancel()
        await asyncio.gather(capture_task, return_exceptions=True)
        output.flush()
        output.clear_buffer()
        response.set()
        await asyncio.gather(*leaf._tasks)

        event = await asyncio.wait_for(output.wait_for_playout(), timeout=1.0)
        assert event.interrupted
        assert event.playback_position == 0
        assert room.local_participant.perform_rpc.await_count == 1

        await output.capture_frame(_frame())
        output.flush()
        _finish(leaf)
        event = await asyncio.wait_for(output.wait_for_playout(), timeout=1.0)
        assert not event.interrupted
    finally:
        response.set()
        capture_task.cancel()
        await asyncio.gather(capture_task, return_exceptions=True)


async def test_cancelled_capture_preserves_next_reply_captions(
    sink: tuple[ds.DataStreamAudioOutput, MagicMock],
) -> None:
    leaf, room = sink
    text = FakeTextOutput()
    sync = TranscriptSynchronizer(next_in_chain_audio=leaf, next_in_chain_text=text)
    output = _PassthroughWrapper(next_in_chain=sync.audio_output)
    called = asyncio.Event()
    response = asyncio.Event()

    async def clear_rpc(**_kwargs: object) -> str:
        called.set()
        await response.wait()
        return ""

    room.local_participant.perform_rpc.side_effect = clear_rpc
    try:
        await sync.text_output.capture_text("first reply")
        sync.text_output.flush()
        await output.capture_frame(_frame())
        output.flush()
        output.clear_buffer()
        await called.wait()
        _finish(leaf, interrupted=True)
        await sync.barrier()

        capture_task = asyncio.create_task(output.capture_frame(_frame()))
        for _ in range(3):
            await asyncio.sleep(0)
        capture_task.cancel()
        await asyncio.gather(capture_task, return_exceptions=True)
        output.flush()
        output.clear_buffer()
        await sync.barrier()

        await sync.text_output.capture_text("future reply captions")
        sync.text_output.flush()
        capture_task = asyncio.create_task(output.capture_frame(_frame()))
        for _ in range(3):
            await asyncio.sleep(0)
        response.set()
        await capture_task
        output.flush()
        _finish(leaf)
        await asyncio.wait_for(output.wait_for_playout(), timeout=1.0)
        await sync.barrier()
        assert "future reply captions" in "".join(text._messages)
    finally:
        response.set()
        await asyncio.gather(*leaf._tasks)
        await sync.aclose()


async def test_cancelled_write_preserves_next_reply_captions(
    sink: tuple[ds.DataStreamAudioOutput, MagicMock],
) -> None:
    leaf, room = sink
    text = FakeTextOutput()
    sync = TranscriptSynchronizer(next_in_chain_audio=leaf, next_in_chain_text=text)
    output = sync.audio_output
    writing = asyncio.Event()
    called = asyncio.Event()
    response = asyncio.Event()

    async def write(_data: bytes) -> None:
        if not writing.is_set():
            writing.set()
            await asyncio.Event().wait()

    async def clear_rpc(**_kwargs: object) -> str:
        called.set()
        await response.wait()
        return ""

    room.local_participant.stream_bytes.return_value.write.side_effect = write
    room.local_participant.perform_rpc.side_effect = clear_rpc
    capture_task = asyncio.create_task(output.capture_frame(_frame()))
    try:
        await writing.wait()
        capture_task.cancel()
        await asyncio.gather(capture_task, return_exceptions=True)
        output.flush()
        output.clear_buffer()
        await called.wait()
        await sync.barrier()

        await sync.text_output.capture_text("future reply captions")
        sync.text_output.flush()
        response.set()
        await output.capture_frame(_frame())
        output.flush()
        _finish(leaf)
        await asyncio.wait_for(output.wait_for_playout(), timeout=1.0)
        await sync.barrier()
        assert "future reply captions" in "".join(text._messages)
    finally:
        response.set()
        capture_task.cancel()
        await asyncio.gather(capture_task, *leaf._tasks, return_exceptions=True)
        await sync.aclose()


@pytest.mark.parametrize("remote_finishes", [0, 1, 2])
async def test_clear_fallback_finishes_only_segments_at_clear_boundary(
    sink: tuple[ds.DataStreamAudioOutput, MagicMock], remote_finishes: int
) -> None:
    output, _ = sink
    for _ in range(2):
        await output.capture_frame(_frame())
        output.flush()
    interrupted_waiter = asyncio.create_task(output.wait_for_playout())
    await asyncio.sleep(0)
    output.clear_buffer()
    await asyncio.sleep(0)
    for _ in range(remote_finishes):
        _finish(output, interrupted=True)

    await output.capture_frame(_frame())
    output.flush()
    event = await asyncio.wait_for(interrupted_waiter, timeout=3.0)
    assert event.interrupted
    assert output._pending_playback_count == 1

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
