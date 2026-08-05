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


# Cartesia-style timed words: alignment searches for each spoken word inside the raw
# sent text, so a word that follows markup carries the full tag as a prefix. The
# silence before the second tag keeps the annotated target behind the forwarded text,
# so the tag's fragments hit the catch-up (negative) branch of the pacing math — with
# markup counted as syllables there, each fragment is paced as multi-second speech.
TIMED_AUDIO_DURATION = 5.5
TIMED_WORDS = [
    (
        '<expr type="expression" label="speak with warm surprise and bright energy"/> Hello ',
        0.0,
        0.35,
    ),
    ("there ", 0.35, 0.7),
    ("my ", 0.7, 1.0),
    ("friend! ", 1.0, 1.4),
    # 1.4 -> 4.4: silence before the expressive sentence
    (
        '<expr type="expression" label="speak calmly and evenly, unhurried and relaxed, '
        'with a gentle reassuring warmth in every word"/> How ',
        4.4,
        4.7,
    ),
    ("are ", 4.7, 4.9),
    ("you ", 4.9, 5.1),
    ("today?", 5.1, 5.5),
]


def _make_opts() -> _TextSyncOptions:
    return _TextSyncOptions(
        speed=1.0,
        hyphenate_word=tokenize.basic.hyphenate_word,
        word_tokenizer=tokenize.basic.WordTokenizer(
            retain_format=True, ignore_punctuation=False, split_character=True
        ),
        speaking_rate_detector=SpeakingRateDetector(),
    )


async def test_markup_fragments_add_no_pacing_delay() -> None:
    opts = _make_opts()
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


async def test_annotated_rate_ignores_markup_in_timed_string() -> None:
    """TTS timing annotations pace via character-offset slices of the raw pushed text;
    the hyphen count of a slice must ignore markup, matching the stripped word_hyphens."""
    opts = _make_opts()
    collector = _CollectorTextOutput()
    impl = _SegmentSynchronizerImpl(opts, next_in_chain=collector)
    try:
        for frame in _silent_frames(TIMED_AUDIO_DURATION):
            impl.push_audio(frame)
        impl.end_audio_input()
        for text, start_time, end_time in TIMED_WORDS:
            impl.push_text(io.TimedString(text, start_time=start_time, end_time=end_time))
        impl.end_text_input()

        start = time.monotonic()
        impl.on_playback_started(time.time())

        await impl._main_atask
        await impl._capture_atask
        elapsed = time.monotonic() - start

        assert "".join(collector.words) == "".join(text for text, _, _ in TIMED_WORDS)

        assert elapsed < TIMED_AUDIO_DURATION + 2.0, (
            f"transcript took {elapsed:.1f}s — markup is being paced as spoken text "
            "in annotated rate path"
        )
    finally:
        await impl.aclose()
