"""A timed transcript fragment must not hold its last word back.

The word stream buffers its trailing token until a delimiter or the end of the input, which
for a realtime model is the segment close — one word landing after the speech is over.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from livekit import rtc
from livekit.agents import tokenize
from livekit.agents.voice import io
from livekit.agents.voice.transcription._speaking_rate import SpeakingRateDetector
from livekit.agents.voice.transcription.synchronizer import (
    _SegmentSynchronizerImpl,
    _TextSyncOptions,
)

pytestmark = pytest.mark.unit

SAMPLE_RATE = 16000
FRAGMENTS = [("It is", 0.0, 0.2), (" sunny", 0.2, 0.4)]
AUDIO_DURATION = 0.4


class _CollectorTextOutput(io.TextOutput):
    def __init__(self) -> None:
        super().__init__(label="test-collector", next_in_chain=None)
        self.words: list[str] = []

    async def capture_text(self, text: str) -> None:
        self.words.append(str(text))

    def flush(self) -> None:
        pass


def _silent_frames(duration: float) -> list[rtc.AudioFrame]:
    samples_per_frame = SAMPLE_RATE // 100  # 10ms
    frame = rtc.AudioFrame(
        data=b"\x00\x00" * samples_per_frame,
        sample_rate=SAMPLE_RATE,
        num_channels=1,
        samples_per_channel=samples_per_frame,
    )
    return [frame] * int(duration * 100)


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


async def test_timed_fragments_forward_the_last_word_before_the_text_input_ends() -> None:
    collector = _CollectorTextOutput()
    impl = _SegmentSynchronizerImpl(_opts(), next_in_chain=collector)
    expected = "".join(text for text, _, _ in FRAGMENTS)
    try:
        for frame in _silent_frames(AUDIO_DURATION):
            impl.push_audio(frame)
        impl.end_audio_input()
        impl.on_playback_started(time.time())

        for text, start_time, end_time in FRAGMENTS:
            impl.push_text(io.TimedString(text, start_time=start_time, end_time=end_time))

        # the turn is still open: no end_text_input() to release the trailing word
        assert await _await_text(collector, expected) == expected, (
            "transcript stalled while the segment stayed open"
        )
    finally:
        await impl.aclose()


async def test_a_span_ending_mid_word_still_forwards_the_same_transcript() -> None:
    # releasing the trailing word of a span that ends mid-word only splits it across two deltas,
    # which the text output concatenates back
    collector = _CollectorTextOutput()
    impl = _SegmentSynchronizerImpl(_opts(), next_in_chain=collector)
    expected = "It is currently sunny"
    try:
        for frame in _silent_frames(AUDIO_DURATION):
            impl.push_audio(frame)
        impl.end_audio_input()
        impl.on_playback_started(time.time())

        impl.push_text(io.TimedString("It is curren", start_time=0.0, end_time=0.2))
        impl.push_text(io.TimedString("tly sunny", start_time=0.2, end_time=0.4))

        assert await _await_text(collector, expected) == expected
    finally:
        await impl.aclose()
