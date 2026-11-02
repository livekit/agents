"""A second flush of the same audio segment is a no-op, not an exception.

An application wrapper around ``session.output.audio`` may flush a segment early (a
barge-in that stops the agent mid-sentence) and the framework then flushes it again at the
end of the reply. Before the guards under test, the second flush reached
``SpeakingRateStream.end_input()`` a second time, which re-sent its flush sentinel on a
channel it had already closed and raised ``ChanClosed`` out of ``AudioOutput.flush``.
"""

from __future__ import annotations

import asyncio
import logging
import time

import pytest

from livekit import rtc
from livekit.agents import io, tokenize
from livekit.agents.voice.transcription._speaking_rate import SpeakingRateDetector
from livekit.agents.voice.transcription.synchronizer import (
    TranscriptSynchronizer,
    _SegmentSynchronizerImpl,
    _TextSyncOptions,
)

pytestmark = pytest.mark.unit

_SAMPLE_RATE = 16000
_FRAME_MS = 10


class _CollectorTextOutput(io.TextOutput):
    def __init__(self) -> None:
        super().__init__(label="collector", next_in_chain=None)
        self.words: list[str] = []

    async def capture_text(self, text: str) -> None:
        self.words.append(text)

    def flush(self) -> None:
        pass


class _TextOutput(io.TextOutput):
    def __init__(self) -> None:
        super().__init__(label="text", next_in_chain=None)

    async def capture_text(self, text: str) -> None:
        pass

    def flush(self) -> None:
        pass


class _AudioOutput(io.AudioOutput):
    def __init__(self) -> None:
        super().__init__(
            label="audio",
            capabilities=io.AudioOutputCapabilities(pause=False),
            sample_rate=_SAMPLE_RATE,
        )

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await super().capture_frame(frame)

    def flush(self) -> None:
        super().flush()

    def clear_buffer(self) -> None:
        pass


def _silent_frames(duration: float) -> list[rtc.AudioFrame]:
    samples = _SAMPLE_RATE * _FRAME_MS // 1000
    frame = rtc.AudioFrame(
        data=b"\x00\x00" * samples,
        sample_rate=_SAMPLE_RATE,
        num_channels=1,
        samples_per_channel=samples,
    )
    return [frame] * int(duration * 1000 / _FRAME_MS)


def _opts() -> _TextSyncOptions:
    return _TextSyncOptions(
        speed=1.0,
        hyphenate_word=tokenize.basic.hyphenate_word,
        word_tokenizer=tokenize.basic.WordTokenizer(
            retain_format=True, ignore_punctuation=False, split_character=True
        ),
        speaking_rate_detector=SpeakingRateDetector(),
    )


async def _await_text(collector: _CollectorTextOutput, expected: str) -> str:
    deadline = time.monotonic() + 3.0
    while "".join(collector.words) != expected and time.monotonic() < deadline:
        await asyncio.sleep(0.02)
    return "".join(collector.words)


def _warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.levelno >= logging.WARNING]


async def test_speaking_rate_stream_end_input_twice_does_not_raise() -> None:
    stream = SpeakingRateDetector().stream()
    try:
        for frame in _silent_frames(0.5):
            stream.push_frame(frame)
        stream.end_input()
        stream.end_input()  # raised ChanClosed before the guard
        stream.flush()  # so does a late flush
        async for _ in stream:
            pass
    finally:
        await stream.aclose()


async def test_segment_end_audio_input_twice_keeps_speed_and_still_forwards(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="livekit.agents")
    collector = _CollectorTextOutput()
    impl = _SegmentSynchronizerImpl(options=_opts(), next_in_chain=collector)
    text = "Hello there my friend!"
    try:
        for frame in _silent_frames(1.0):
            impl.push_audio(frame)
        impl.push_text(text)
        impl.end_text_input()
        impl.end_audio_input()
        speed, speed_on_unit = impl._speed, impl._speed_on_speaking_unit

        impl.end_audio_input()  # the second flush of the same segment

        assert impl.audio_input_ended
        assert (impl._speed, impl._speed_on_speaking_unit) == (speed, speed_on_unit)
        impl.on_playback_started(time.time())
        assert await _await_text(collector, text) == text
        assert not _warnings(caplog)
    finally:
        await impl.aclose()


async def test_synced_audio_output_flushed_twice_after_frames(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The reported path: a wrapper flushes early, then the end of the reply flushes again."""
    caplog.set_level(logging.INFO, logger="livekit.agents")
    sync = TranscriptSynchronizer(
        next_in_chain_audio=_AudioOutput(), next_in_chain_text=_TextOutput()
    )
    try:
        await sync.text_output.capture_text("hello")
        sync.text_output.flush()
        for frame in _silent_frames(0.5):
            await sync.audio_output.capture_frame(frame)

        sync.audio_output.flush()
        sync.audio_output.flush()  # raised ChanClosed before the guard

        assert sync._impl.audio_input_ended
        sync.audio_output.on_playback_finished(playback_position=0.5, interrupted=False)
        await sync.barrier()
        assert not _warnings(caplog)
    finally:
        await sync.aclose()
