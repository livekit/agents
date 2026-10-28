"""Playback reports retain their admitted segment across endpoint reuse."""

import asyncio

import pytest

from livekit import rtc
from livekit.agents.voice.io import AudioOutput, AudioOutputCapabilities, PlaybackSegment

pytestmark = pytest.mark.unit


class Output(AudioOutput):
    def __init__(self) -> None:
        super().__init__(label="output", capabilities=AudioOutputCapabilities(pause=False))

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await super().capture_frame(frame)

    def flush(self) -> None:
        super().flush()

    def clear_buffer(self) -> None:
        pass


async def test_old_transport_report_cannot_settle_a_new_attachment_on_the_same_output() -> None:
    output = Output()
    reports = []
    output.on("playback_finished", reports.append)
    frame = rtc.AudioFrame(b"\0\0" * 160, 16000, 1, 160)
    old = PlaybackSegment(attachment=1, segment=1)
    new = PlaybackSegment(attachment=2, segment=2)
    output.begin_segment(old)
    await output.capture_frame(frame)
    assert output.playback_segment == old
    output.flush()
    output.on_playback_finished(segment=old, playback_position=0.01, interrupted=False)
    assert len(reports) == 1

    output.begin_segment(new)
    await output.capture_frame(frame)
    output.flush()
    waiting = asyncio.create_task(output.wait_for_playout())
    try:
        await asyncio.sleep(0)
        output.on_playback_finished(segment=old, playback_position=0.01, interrupted=False)
        await asyncio.sleep(0)
        assert not waiting.done()
        assert len(reports) == 1
        output.on_playback_finished(segment=new, playback_position=0.01, interrupted=False)
        result = await asyncio.wait_for(waiting, 1)
        assert result.playback_position == 0.01
        assert len(reports) == 2
        assert reports[-1].segment == new
    finally:
        waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)


async def test_native_audio_view_preserves_keys_through_admission_and_transport_reports() -> None:
    from livekit.agents._core_view import _AudioOutput

    output = Output()
    view = _AudioOutput(output)
    started = []
    progressed = []
    finished = []
    output.on("playback_started", started.append)
    output.on("playback_progressed", progressed.append)
    output.on("playback_finished", finished.append)
    frame = rtc.AudioFrame(b"\0\0" * 160, 16000, 1, 160)
    old = PlaybackSegment(attachment=1, segment=1)
    new = PlaybackSegment(attachment=2, segment=2)

    view._core_begin_segment(old)
    await view._core_capture(frame)
    view._core_flush_segment()
    view._core_on_playback_started(old, 1.0)
    view._core_on_playback_progressed(old, 1.0, 0.0, 0.01)
    view._core_on_playback_finished(old, 0.01, False, "first")
    assert started[-1].segment == progressed[-1].segment == finished[-1].segment == old

    view._core_begin_segment(new)
    await view._core_capture(frame)
    view._core_flush_segment()
    waiting = asyncio.create_task(output.wait_for_playout())
    try:
        await asyncio.sleep(0)
        view._core_on_playback_started(old, 2.0)
        view._core_on_playback_progressed(old, 2.0, 0.0, 0.01)
        view._core_on_playback_finished(old, 0.01, False, "stale")
        await asyncio.sleep(0)
        assert not waiting.done()
        assert len(started) == len(progressed) == len(finished) == 1

        view._core_on_playback_started(new, 3.0)
        view._core_on_playback_progressed(new, 3.0, 0.0, 0.01)
        view._core_on_playback_finished(new, 0.01, False, "replacement")
        result = await asyncio.wait_for(waiting, 1)
        assert result.synchronized_transcript == "replacement"
        assert started[-1].segment == progressed[-1].segment == finished[-1].segment == new
    finally:
        waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)
