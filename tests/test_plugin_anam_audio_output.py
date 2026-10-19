"""Unit tests for the interrupt handling of the Anam audio output.

The Anam avatar sends no lk.playback_finished for a lk.clear_buffer that reaches it
before lk.playback_started, so AnamAudioOutput must not forward one before that.
The room is a mock, and the tests call the RPC handlers the avatar would call.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from livekit import rtc
from livekit.agents.voice.avatar import _datastream_io as ds
from livekit.plugins.anam._audio_output import AnamAudioOutput

pytestmark = [pytest.mark.unit, pytest.mark.plugin("anam")]

AVATAR = "anam-avatar"
SAMPLE_RATE = 24000


def _frame() -> rtc.AudioFrame:
    samples = SAMPLE_RATE // 100
    return rtc.AudioFrame(
        data=b"\x00\x00" * samples,
        sample_rate=SAMPLE_RATE,
        num_channels=1,
        samples_per_channel=samples,
    )


@pytest.fixture
async def room():
    room = MagicMock()
    room.isconnected.return_value = True
    room.local_participant._rpc_handlers = {}
    room.local_participant.perform_rpc = AsyncMock(return_value="ok")
    room.local_participant.stream_bytes = AsyncMock(
        side_effect=lambda **kwargs: MagicMock(write=AsyncMock(), aclose=AsyncMock())
    )
    with (
        patch.object(ds.utils, "wait_for_participant", AsyncMock()),
        patch.object(ds.utils, "wait_for_track_publication", AsyncMock()),
    ):
        yield room


async def _output(room, **kwargs) -> AnamAudioOutput:
    output = AnamAudioOutput(room, destination_identity=AVATAR, sample_rate=SAMPLE_RATE, **kwargs)
    await _drain()
    assert output._start_atask is not None
    await output._start_atask
    return output


async def _drain() -> None:
    for _ in range(10):
        await asyncio.sleep(0)


def _clears(room) -> int:
    calls = room.local_participant.perform_rpc.await_args_list
    return sum(call.kwargs["method"] == ds.RPC_CLEAR_BUFFER for call in calls)


def _playback_started(output: AnamAudioOutput) -> None:
    output._handle_playback_started(SimpleNamespace(caller_identity=AVATAR))


def _playback_finished(output: AnamAudioOutput, *, interrupted: bool) -> None:
    payload = json.dumps({"playback_position": 0.5, "interrupted": interrupted})
    output._handle_playback_finished(SimpleNamespace(caller_identity=AVATAR, payload=payload))


async def _interrupt(output: AnamAudioOutput) -> asyncio.Task:
    # the agent flushes and clears from the forwarding task, then clears again
    output.flush()
    output.clear_buffer()
    output.clear_buffer()
    playout = asyncio.create_task(output.wait_for_playout())
    await _drain()
    return playout


async def test_clear_without_pending_segment_sends_no_rpc(room):
    output = await _output(room)
    await output.capture_frame(_frame())
    output.flush()
    _playback_started(output)
    _playback_finished(output, interrupted=False)

    output.clear_buffer()
    await _drain()

    assert _clears(room) == 0


async def test_clear_is_held_until_playback_started(room):
    output = await _output(room)
    await output.capture_frame(_frame())

    playout = await _interrupt(output)
    assert _clears(room) == 0

    _playback_started(output)
    await _drain()
    assert _clears(room) == 1
    assert not playout.done()

    _playback_finished(output, interrupted=True)
    await _drain()
    assert playout.done()


async def test_clear_after_playback_started_is_forwarded_once(room):
    output = await _output(room)
    await output.capture_frame(_frame())
    _playback_started(output)

    playout = await _interrupt(output)
    assert _clears(room) == 1

    _playback_finished(output, interrupted=True)
    await _drain()
    assert playout.done()


async def test_segment_without_byte_stream_finishes_locally(room):
    output = await _output(room)

    async def _never_opens(**kwargs):
        await asyncio.Event().wait()

    room.local_participant.stream_bytes.side_effect = _never_opens
    capture = asyncio.create_task(output.capture_frame(_frame()))
    await _drain()
    assert output.captured_playout_segments == 1

    capture.cancel()
    await asyncio.gather(capture, return_exceptions=True)
    playout = await _interrupt(output)

    assert _clears(room) == 0
    assert playout.done()


async def test_finish_without_playback_started_drops_held_clear(room):
    output = await _output(room)
    await output.capture_frame(_frame())
    playout = await _interrupt(output)

    # the lk.playback_started RPC was lost and the avatar played the segment out
    _playback_finished(output, interrupted=False)
    await _drain()

    assert playout.done()
    assert _clears(room) == 0
    assert output._held_clear_timer is None


async def test_held_clear_is_forwarded_after_timeout(room):
    output = await _output(room, playback_started_timeout=0.05)
    await output.capture_frame(_frame())
    playout = await _interrupt(output)

    await asyncio.sleep(0.1)
    assert _clears(room) == 1

    _playback_finished(output, interrupted=True)
    await _drain()
    assert playout.done()


async def test_later_segment_does_not_delay_held_clear(room):
    output = await _output(room)
    await output.capture_frame(_frame())
    playout = await _interrupt(output)

    await output.capture_frame(_frame())
    _playback_started(output)
    await _drain()
    assert _clears(room) == 1

    _playback_finished(output, interrupted=True)
    await _drain()
    assert playout.done()
