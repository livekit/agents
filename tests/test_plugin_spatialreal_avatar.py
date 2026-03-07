from __future__ import annotations

import asyncio
import time

import pytest

from livekit import rtc
from livekit.agents.voice.avatar import QueueAudioOutput
from livekit.plugins.spatialreal.avatar import AvatarSession, _SegmentState


class _FinalizeInterruptRaceSession:
    def __init__(self, req_id: str) -> None:
        self._req_id = req_id
        self.send_audio_started = asyncio.Event()
        self.interrupt_called = asyncio.Event()
        self.allow_finalize = asyncio.Event()

    async def send_audio(self, *, audio: bytes, end: bool) -> str:
        assert audio == b""
        assert end is True
        self.send_audio_started.set()
        await self.allow_finalize.wait()
        return self._req_id

    async def interrupt(self) -> str:
        self.interrupt_called.set()
        return self._req_id


def _make_audio_frame(duration: float, sample_rate: int = 24000) -> rtc.AudioFrame:
    samples_per_channel = int(sample_rate * duration + 0.5)
    return rtc.AudioFrame(
        data=b"\x00\x00" * samples_per_channel,
        sample_rate=sample_rate,
        num_channels=1,
        samples_per_channel=samples_per_channel,
    )


async def test_interrupt_during_finalize_does_not_create_phantom_segment() -> None:
    req_id = "req-1"
    avatar = AvatarSession(api_key="api-key", app_id="app-id", avatar_id="avatar-id")
    avatarkit_session = _FinalizeInterruptRaceSession(req_id)
    audio_buffer = QueueAudioOutput(sample_rate=24000)
    playback_finished_events = []
    audio_buffer.on("playback_finished", playback_finished_events.append)

    first_frame = _make_audio_frame(0.2)
    await audio_buffer.capture_frame(first_frame)
    audio_buffer.flush()

    avatar._avatarkit_session = avatarkit_session
    avatar._audio_buffer = audio_buffer
    avatar._segments[req_id] = _SegmentState(
        req_id=req_id,
        pushed_duration=first_frame.duration,
        first_frame_at=time.time(),
    )
    avatar._active_req_id = req_id

    finalize_task = asyncio.create_task(avatar._finalize_active_segment(source="segment_end"))
    await avatarkit_session.send_audio_started.wait()

    interrupt_task = asyncio.create_task(avatar._handle_interrupt())
    await avatarkit_session.interrupt_called.wait()

    avatarkit_session.allow_finalize.set()

    assert await finalize_task is True
    await interrupt_task

    assert len(playback_finished_events) == 1
    assert playback_finished_events[0].interrupted is True
    assert avatar._segments == {}
    assert list(avatar._pending_segment_ids) == []
    assert (
        avatar._complete_segment(req_id=req_id, interrupted=False, reason="phantom_check") is False
    )

    second_frame = _make_audio_frame(0.1)
    await audio_buffer.capture_frame(second_frame)
    audio_buffer.flush()
    audio_buffer.notify_playback_finished(
        playback_position=second_frame.duration,
        interrupted=False,
    )

    assert len(playback_finished_events) == 2
    assert playback_finished_events[1].interrupted is False
    assert playback_finished_events[1].playback_position == pytest.approx(second_frame.duration)
