"""Transcript synchronizer pacing must ignore expressive markup.

The synchronizer forwards the raw LLM text (markup intact — the room output strips
it downstream) but paces the display against the *visible* words only. Markup tags
carry spaces in their attributes, so the word stream shreds them into fragments
(``<expr``, ``type="expression"``, ``label="speak``, ``playfully"/>``); a per-token
strip can't recognize those, and each fragment was paced as if it were spoken — the
transcript drifted seconds behind the audio on every expressive sentence.
"""

from __future__ import annotations

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
AUDIO_DURATION = 3.0

# ~10 visible hyphens of speech, but ~40 hyphens of markup fragments. With the bug
# the markup alone adds >10s of pacing; with the fix the whole transcript paces out
# in about the audio duration.
MARKED_UP_TURN = (
    '<expr type="expression" label="speak with warm surprise and bright energy"/> '
    "Hello there my friend! "
    '<expr type="sound" label="laugh"/> '
    '<expr type="expression" label="speak calmly and evenly, unhurried"/> '
    "How are you today?"
)


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


async def test_markup_fragments_add_no_pacing_delay() -> None:
    opts = _TextSyncOptions(
        speed=1.0,
        hyphenate_word=tokenize.basic.hyphenate_word,
        word_tokenizer=tokenize.basic.WordTokenizer(
            retain_format=True, ignore_punctuation=False, split_character=True
        ),
        speaking_rate_detector=SpeakingRateDetector(),
    )
    collector = _CollectorTextOutput()
    impl = _SegmentSynchronizerImpl(opts, next_in_chain=collector)
    try:
        for frame in _silent_frames(AUDIO_DURATION):
            impl.push_audio(frame)
        impl.end_audio_input()
        impl.push_text(MARKED_UP_TURN)
        impl.end_text_input()

        start = time.monotonic()
        impl.on_playback_started(time.time())

        # forwarding is done once the main task exhausts the word stream and the
        # capture task drains the output channel
        await impl._main_atask
        await impl._capture_atask
        elapsed = time.monotonic() - start

        # every raw token is still forwarded (markup included — stripped downstream)
        assert "".join(collector.words) == MARKED_UP_TURN

        # the pacing budget must cover only the visible words, i.e. roughly the
        # audio duration; with markup fragments paced as speech it exceeds 12s
        assert elapsed < AUDIO_DURATION + 3.0, (
            f"transcript took {elapsed:.1f}s — markup is being paced as spoken text"
        )
    finally:
        await impl.aclose()


# a turn that produces audio but no visible syllables at all: after markup stripping
# the pushed text hyphenates to nothing (with retain_format even a trailing space
# would count as a token), so the re-estimated speed used to become 0
MARKUP_ONLY_TURN = '<expr type="sound" label="laugh"/>'


async def test_markup_only_turn_keeps_nonzero_speed() -> None:
    opts = _TextSyncOptions(
        speed=1.0,
        hyphenate_word=tokenize.basic.hyphenate_word,
        word_tokenizer=tokenize.basic.WordTokenizer(
            retain_format=True, ignore_punctuation=False, split_character=True
        ),
        speaking_rate_detector=SpeakingRateDetector(),
    )
    collector = _CollectorTextOutput()
    impl = _SegmentSynchronizerImpl(opts, next_in_chain=collector)
    try:
        for frame in _silent_frames(1.0):
            impl.push_audio(frame)
        impl.end_audio_input()
        impl.push_text(MARKUP_ONLY_TURN)
        impl.end_text_input()

        impl.on_playback_started(time.time())

        # with a zero speed estimate the pacing division raised ZeroDivisionError here
        await impl._main_atask
        await impl._capture_atask

        assert impl._speed > 0
        # no visible syllables to estimate from, so the speaking-unit speed stays unset
        # rather than being overwritten with a meaningless 0
        assert impl._speed_on_speaking_unit is None
        assert "".join(collector.words) == MARKUP_ONLY_TURN
    finally:
        await impl.aclose()
