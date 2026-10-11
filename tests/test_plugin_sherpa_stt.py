"""livekit-plugins-sherpa STT contract, driven by scripted recognizers (no models, no network).

The scripted recognizer stands in for sherpa-onnx so each test controls exactly what the
model "hears". The LiveKit side (RecognizeStream, channels, event types) is the real SDK.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import pytest

from livekit import rtc
from livekit.agents import stt
from livekit.plugins import sherpa
from livekit.plugins.sherpa.stt import (
    SAMPLE_RATE,
    TAIL_PADDING_S,
    OfflineResult,
    OfflineStream,
    OnlineResult,
    OnlineStream,
)

pytestmark = pytest.mark.unit

# RecognizeStream adds the wall-clock time its task took to start to start_time_offset.
CLOCK = pytest.approx
TOL = 1e-3

SOS = stt.SpeechEventType.START_OF_SPEECH
INTERIM = stt.SpeechEventType.INTERIM_TRANSCRIPT
FINAL = stt.SpeechEventType.FINAL_TRANSCRIPT
EOS = stt.SpeechEventType.END_OF_SPEECH
USAGE = stt.SpeechEventType.RECOGNITION_USAGE


@dataclass
class Step:
    """What the model reports after one chunk of audio."""

    text: str = ""
    endpoint: bool = False
    tokens: Sequence[str] = ()
    timestamps: Sequence[float] = ()
    start_time: float = 0.0


@dataclass
class _Stream:
    script: list[Step]
    pos: int = -1

    def accept_waveform(self, sample_rate: int, waveform: npt.NDArray[np.float32]) -> None:
        self.pos += 1

    def input_finished(self) -> None:
        pass

    @property
    def current(self) -> Step:
        return self.script[self.pos] if self.pos < len(self.script) else Step()


class ScriptedRecognizer:
    """Serves one stream; each accepted audio chunk advances the script by one step."""

    def __init__(self, script: list[Step]) -> None:
        self.stream = _Stream(script)

    def create_stream(self) -> OnlineStream:
        return self.stream

    def is_ready(self, s: OnlineStream) -> bool:
        return False

    def decode_stream(self, s: OnlineStream) -> None:
        pass

    def get_result_all(self, s: OnlineStream) -> OnlineResult:
        return self.stream.current

    def is_endpoint(self, s: OnlineStream) -> bool:
        return self.stream.current.endpoint

    def reset(self, s: OnlineStream) -> None:
        pass


def _frame(ms: int = 100) -> rtc.AudioFrame:
    n = SAMPLE_RATE * ms // 1000
    return rtc.AudioFrame(b"\x00\x00" * n, SAMPLE_RATE, 1, n)


async def _run(
    script: list[Step], frames: int, *, flush_after: int | None = None, offset: float = 0.0
) -> list[stt.SpeechEvent]:
    engine = sherpa.STT(recognizer=ScriptedRecognizer(script))
    stream = engine.stream()
    stream.start_time_offset = offset
    for i in range(frames):
        stream.push_frame(_frame())
        if flush_after is not None and i == flush_after:
            stream.flush()
    stream.end_input()
    events = [ev async for ev in stream]
    await stream.aclose()
    return events


def _types(events: list[stt.SpeechEvent]) -> list[stt.SpeechEventType]:
    return [ev.type for ev in events]


async def test_utterance_emits_final_before_end_of_speech() -> None:
    events = await _run(
        [Step(), Step("hello"), Step("hello world"), Step("hello world", endpoint=True)],
        frames=4,
    )

    assert _types(events) == [SOS, INTERIM, INTERIM, FINAL, EOS, USAGE]
    assert [ev.alternatives[0].text for ev in events if ev.type in (INTERIM, FINAL)] == [
        "hello",
        "hello world",
        "hello world",
    ]


async def test_unchanged_partial_is_not_repeated() -> None:
    events = await _run([Step("hi"), Step("hi"), Step("hi", endpoint=True)], frames=3)

    assert _types(events) == [SOS, INTERIM, FINAL, EOS, USAGE]


async def test_silence_emits_no_speech_events() -> None:
    events = await _run([Step(), Step(endpoint=True), Step()], frames=3)

    assert events == []


async def test_end_of_input_finalizes_the_open_utterance() -> None:
    # No endpoint in the script: the speaker is cut off when the input ends.
    events = await _run([Step("good"), Step("good morning"), Step("good morning")], frames=2)

    assert _types(events) == [SOS, INTERIM, INTERIM, FINAL, EOS, USAGE]
    assert events[3].alternatives[0].text == "good morning"


async def test_each_utterance_gets_its_own_request_id() -> None:
    events = await _run(
        [Step("one", endpoint=True), Step(), Step("two", endpoint=True)],
        frames=3,
    )

    finals = [ev for ev in events if ev.type == FINAL]
    assert [ev.alternatives[0].text for ev in finals] == ["one", "two"]
    assert finals[0].request_id != finals[1].request_id


async def test_usage_reports_audio_consumed_by_the_utterance() -> None:
    events = await _run([Step("a"), Step("a"), Step("a", endpoint=True)], frames=3)

    usage = next(ev for ev in events if ev.type == USAGE)
    assert usage.recognition_usage is not None
    assert usage.recognition_usage.audio_duration == pytest.approx(0.3)


async def test_words_are_grouped_from_bpe_tokens_on_the_session_clock() -> None:
    final = Step(
        "HELLO WORLD",
        endpoint=True,
        tokens=[" HEL", "LO", " WORLD"],
        timestamps=[0.1, 0.2, 0.5],
        start_time=2.0,
    )
    events = await _run([final], frames=1, offset=10.0)

    data = next(ev for ev in events if ev.type == FINAL).alternatives[0]
    assert data.words is not None
    assert data.words == ["HELLO", "WORLD"]
    assert [w.start_time for w in data.words] == CLOCK([12.1, 12.5], abs=TOL)
    assert [w.end_time for w in data.words] == CLOCK([12.5, 12.5], abs=TOL)
    assert (data.start_time, data.end_time) == CLOCK((12.1, 12.5), abs=TOL)
    assert [w.start_time_offset for w in data.words] == CLOCK([10.0, 10.0], abs=TOL)


async def test_flush_padding_does_not_shift_later_timestamps() -> None:
    # flush() makes the plugin inject TAIL_PADDING_S of silence. sherpa's clock counts it,
    # so the second segment's start must be corrected back to the real audio timeline.
    script = [
        Step("first"),
        Step("first"),  # the flush, consumed by the padding
        Step(
            "second",
            endpoint=True,
            tokens=[" second"],
            timestamps=[0.0],
            start_time=1.0 + TAIL_PADDING_S,
        ),
    ]
    events = await _run(script, frames=2, flush_after=0)

    finals = [ev for ev in events if ev.type == FINAL]
    assert [ev.alternatives[0].text for ev in finals] == ["first", "second"]
    assert finals[1].alternatives[0].start_time == CLOCK(1.0, abs=TOL)


@dataclass
class _OfflineResult:
    text: str


@dataclass
class _OfflineStream:
    samples: int = 0
    rates: list[int] = field(default_factory=list)

    def accept_waveform(self, sample_rate: int, waveform: npt.NDArray[np.float32]) -> None:
        self.samples += len(waveform)
        self.rates.append(sample_rate)

    @property
    def result(self) -> OfflineResult:
        return _OfflineResult(f"  heard {self.samples} samples  ")


class ScriptedOfflineRecognizer:
    def __init__(self) -> None:
        self.streams: list[_OfflineStream] = []

    def create_stream(self) -> OfflineStream:
        s = _OfflineStream()
        self.streams.append(s)
        return s

    def decode_stream(self, s: OfflineStream) -> None:
        pass


def test_an_online_recognizer_streams() -> None:
    engine = sherpa.STT(recognizer=ScriptedRecognizer([]))

    assert engine.capabilities.streaming
    assert engine.capabilities.interim_results
    assert engine.capabilities.aligned_transcript == "word"


def test_an_offline_recognizer_is_batch_only() -> None:
    engine = sherpa.STT(recognizer=ScriptedOfflineRecognizer())

    assert not engine.capabilities.streaming
    assert engine.capabilities.offline_recognize
    assert engine.model == "custom"
    assert engine.provider == "sherpa-onnx"
    with pytest.raises(NotImplementedError, match="StreamAdapter"):
        engine.stream()


async def test_offline_recognize_resamples_room_audio_and_returns_one_final() -> None:
    recognizer = ScriptedOfflineRecognizer()
    engine = sherpa.STT(recognizer=recognizer, language="en")
    n = 48000  # one second of 48 kHz room audio
    frame = rtc.AudioFrame(b"\x00\x01" * n, 48000, 1, n)

    event = await engine.recognize([frame])

    assert event.type == FINAL
    assert recognizer.streams[0].rates == [SAMPLE_RATE]
    assert abs(recognizer.streams[0].samples - SAMPLE_RATE) < SAMPLE_RATE * 0.02
    assert event.alternatives[0].text == f"heard {recognizer.streams[0].samples} samples"
    assert event.alternatives[0].end_time == pytest.approx(1.0, abs=0.02)
