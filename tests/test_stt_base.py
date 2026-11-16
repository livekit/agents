"""Unit tests for base STT `RecognizeStream` fields (start_time, etc.)."""

from __future__ import annotations

import asyncio
import dataclasses
import time
from collections.abc import AsyncIterator
from typing import Any

import pytest

from livekit.agents import APIConnectionError, APIStatusError
from livekit.agents.stt import (
    STT,
    RecognitionUsage,
    RecognizeStream,
    SpeechData,
    SpeechEvent,
    SpeechEventType,
    StreamAdapter,
    STTCapabilities,
)
from livekit.agents.stt.clock import _ConnectionClock, _InputClock, _SessionStart
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions, TimedString
from livekit.agents.utils import is_given
from livekit.agents.utils.audio import AudioBuffer, silence_frame

from .fake_stt import FakeSTT, FakeUserSpeech
from .fake_vad import FakeVAD

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]


class _DummyStream(RecognizeStream):
    """Minimal RecognizeStream for unit tests — does not hit the network."""

    def __init__(
        self,
        *,
        stt: STT,
        fail_first_run: bool = False,
    ) -> None:
        super().__init__(stt=stt, conn_options=DEFAULT_API_CONNECT_OPTIONS)
        self._fail_first_run = fail_first_run
        self._run_count = 0

    async def _run(self) -> None:
        self._run_count += 1
        if self._fail_first_run and self._run_count == 1:
            raise APIConnectionError("fake failure to trigger retry")
        # emit a final and exit so _main_task can complete normally
        self._event_ch.send_nowait(
            SpeechEvent(
                type=SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[SpeechData(language="", text="hello")],
            )
        )


class _DummySTT(STT):
    def __init__(self, *, requires_timestamp_translation: bool = True) -> None:
        super().__init__(
            capabilities=STTCapabilities(streaming=True, interim_results=False),
            requires_timestamp_translation=requires_timestamp_translation,
        )

    async def _recognize_impl(self, buffer: AudioBuffer, *, language, conn_options) -> SpeechEvent:
        raise NotImplementedError

    def stream(self, *, language=None, conn_options=DEFAULT_API_CONNECT_OPTIONS) -> _DummyStream:
        return _DummyStream(stt=self)


class _NonRetryableStream(RecognizeStream):
    def __init__(self, *, stt: STT) -> None:
        super().__init__(stt=stt, conn_options=DEFAULT_API_CONNECT_OPTIONS)
        self.run_count = 0

    async def _run(self) -> None:
        self.run_count += 1
        raise APIStatusError("Unauthorized", status_code=401)


async def test_start_time_seeded_on_init() -> None:
    """start_time is initialized to approximately time.time() when the stream is created."""
    stt = _DummySTT()
    before = time.time()
    stream = stt.stream()
    after = time.time()

    assert before <= stream.start_time <= after
    await stream.aclose()


async def test_start_time_setter_accepts_valid_values() -> None:
    """Plugins can override start_time by assigning to the public property."""
    stt = _DummySTT()
    stream = stt.stream()

    new_anchor = time.time() + 10.0
    stream.start_time = new_anchor
    assert stream.start_time == new_anchor

    await stream.aclose()


async def test_start_time_setter_rejects_negative() -> None:
    """start_time setter validates non-negative, matching start_time_offset behavior."""
    stt = _DummySTT()
    stream = stt.stream()

    with pytest.raises(ValueError, match="start_time must be non-negative"):
        stream.start_time = -1.0

    await stream.aclose()


async def test_start_time_reseeded_on_retry() -> None:
    """When _main_task retries after an APIError, start_time is re-seeded so plugin
    overrides from the previous connection don't leak into the new one."""
    stt = _DummySTT()
    stream = _DummyStream(stt=stt, fail_first_run=True)

    # Simulate a plugin overriding start_time during the first (failing) _run()
    # by assigning a sentinel value before the task picks up.
    sentinel = 1.0
    stream.start_time = sentinel

    # Let the main task run: it should retry past the first-run APIError, and
    # on each attempt re-seed start_time to a fresh time.time() value before
    # _run() is called.
    await asyncio.wait_for(stream._task, timeout=5.0)

    # After the retry, start_time must have been re-seeded (not equal to sentinel).
    assert stream.start_time != sentinel
    # And it should be a recent wall-clock value.
    assert time.time() - stream.start_time < 5.0

    await stream.aclose()


async def test_non_retryable_error_is_not_retried() -> None:
    stt = _DummySTT()
    stream = _NonRetryableStream(stt=stt)

    with pytest.raises(APIStatusError) as exc_info:
        await stream._task

    assert exc_info.value.status_code == 401
    assert stream.run_count == 1

    await stream.aclose()


class _AlwaysFailingSTT(STT):
    """A batch STT whose `_recognize_impl` always raises, counting its attempts."""

    def __init__(self, error: Exception) -> None:
        super().__init__(capabilities=STTCapabilities(streaming=False, interim_results=False))
        self._error = error
        self.attempts = 0

    async def _recognize_impl(self, buffer: AudioBuffer, *, language, conn_options) -> SpeechEvent:
        self.attempts += 1
        raise self._error


def _retry_friendly_options() -> APIConnectOptions:
    # the default 2s retry_interval would make the retryable case take 6s of real time
    return dataclasses.replace(DEFAULT_API_CONNECT_OPTIONS, retry_interval=0.0)


async def test_recognize_does_not_retry_a_non_retryable_error() -> None:
    """`recognize()` must honour `APIError.retryable`, like TTS, the LLM and the streaming
    path in this same module already do.

    Before this, `max_retry` alone gated the retry loop, so a permanent failure such as a
    401 was issued `max_retry + 1` times -- burning quota and delaying a failure that could
    never succeed.
    """
    stt = _AlwaysFailingSTT(APIStatusError("Unauthorized", status_code=401))
    with pytest.raises(APIStatusError) as exc_info:
        await stt.recognize(
            silence_frame(duration=0.1, sample_rate=16_000),
            conn_options=_retry_friendly_options(),
        )

    assert exc_info.value.status_code == 401
    assert stt.attempts == 1


async def test_recognize_still_retries_a_retryable_error() -> None:
    """The retryable path is unchanged: the full budget is still spent."""
    stt = _AlwaysFailingSTT(APIConnectionError("socket closed"))
    with pytest.raises(APIConnectionError):
        await stt.recognize(
            silence_frame(duration=0.1, sample_rate=16_000),
            conn_options=_retry_friendly_options(),
        )

    assert stt.attempts == DEFAULT_API_CONNECT_OPTIONS.max_retry + 1


async def test_recognize_returns_the_first_success_without_retrying() -> None:
    """Guard against the fix swallowing successful calls."""
    stt = FakeSTT(fake_transcript="hello")
    event = await stt.recognize(
        silence_frame(duration=0.1, sample_rate=16_000),
        conn_options=_retry_friendly_options(),
    )

    assert event.alternatives[0].text == "hello"


async def test_stream_adapter_keeps_vad_speech_end_on_delayed_final(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = silence_frame(duration=0.1, sample_rate=16_000)
    monkeypatch.setattr("livekit.agents.stt.stream_adapter.utils.merge_frames", lambda _: frame)
    speech = FakeUserSpeech(
        start_time=0.0,
        end_time=0.1,
        transcript="hello",
        stt_delay=0.0,
    )
    batch_stt = FakeSTT(fake_transcript="hello", fake_timeout=0.5)
    batch_stt._capabilities.streaming = False
    adapter = StreamAdapter(
        stt=batch_stt,
        vad=FakeVAD(
            fake_user_speeches=[speech],
            min_speech_duration=0.01,
            min_silence_duration=0.3,
        ),
    )

    try:
        async with adapter.stream() as stream:
            stream.push_frame(frame)
            stream.end_input()
            events = [event async for event in stream]
    finally:
        await adapter.aclose()

    end_event = next(event for event in events if event.type == SpeechEventType.END_OF_SPEECH)
    final_event = next(event for event in events if event.type == SpeechEventType.FINAL_TRANSCRIPT)

    assert end_event.speech_end_time is not None
    assert final_event.speech_end_time == end_event.speech_end_time
    assert final_event.created_at - end_event.created_at == pytest.approx(0.5, abs=0.01)


class _FlappingStream(RecognizeStream):
    """Every connection succeeds, stays up for `uptime` seconds, then drops."""

    def __init__(self, *, stt: STT, uptime: float, drops: int, emit_transcript: bool) -> None:
        super().__init__(
            stt=stt,
            conn_options=dataclasses.replace(DEFAULT_API_CONNECT_OPTIONS, retry_interval=0.0),
        )
        self.runs = 0
        self._uptime = uptime
        self._drops = drops
        self._emit_transcript = emit_transcript

    async def _run(self) -> None:
        self.runs += 1
        if self._emit_transcript:
            self._event_ch.send_nowait(
                SpeechEvent(
                    type=SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=[SpeechData(language="en", text="hello")],
                )
            )
        await asyncio.sleep(self._uptime)
        if self.runs <= self._drops:
            raise APIConnectionError("socket dropped")


@pytest.mark.parametrize("uptime", [1.0, 6.0, 180.0])
async def test_retry_budget_is_finite_without_recognition_progress(uptime: float) -> None:
    stream = _FlappingStream(stt=_DummySTT(), uptime=uptime, drops=100, emit_transcript=False)

    with pytest.raises(APIConnectionError):
        await stream._task

    assert stream.runs == DEFAULT_API_CONNECT_OPTIONS.max_retry + 1
    await stream.aclose()


class _AudioStream(RecognizeStream):
    """Records ``start_time_offset`` as each run reads its first frame, then returns."""

    def __init__(
        self, *, stt: STT, fail_first_run: bool = False, fail_first_connect: bool = False
    ) -> None:
        super().__init__(stt=stt, conn_options=DEFAULT_API_CONNECT_OPTIONS)
        self._fail_first_run = fail_first_run
        self._fail_first_connect = fail_first_connect
        self.run_count = 0
        self.offsets: list[float] = []

    async def _run(self) -> None:
        self.run_count += 1
        if self._fail_first_connect and self.run_count == 1:
            # the connection fails before reading anything; pushed audio stays queued
            await asyncio.sleep(3.0)
            raise APIConnectionError("fake connect failure to trigger retry")
        async for data in self._input_ch:
            if isinstance(data, RecognizeStream._FlushSentinel):
                continue
            self.offsets.append(self.start_time_offset)
            if self._fail_first_run and self.run_count == 1:
                raise APIConnectionError("fake failure to trigger retry")
            return


async def _wait_for_offsets(stream: _AudioStream, count: int) -> None:
    async def wait() -> None:
        while len(stream.offsets) < count:
            await asyncio.sleep(0.01)

    await asyncio.wait_for(wait(), timeout=10.0)


async def test_start_time_offset_grows_by_the_delay_to_the_first_frame() -> None:
    """The provider counts audio from the first frame it reads, not from the moment the
    offset was set, so the offset must absorb the delay between the two."""
    stream = _AudioStream(stt=_DummySTT())
    set_at = time.time()
    stream.start_time_offset = 30.0
    frame = silence_frame(0.02, 16000)
    await asyncio.sleep(5.0)

    pushed_at = time.time()
    stream.push_frame(frame)
    await _wait_for_offsets(stream, 1)

    assert stream.offsets == [pytest.approx(30.0 + pushed_at - frame.duration - set_at)]
    await stream.aclose()


async def test_start_time_offset_set_after_the_first_frame_counts_back_from_now() -> None:
    """A value set late still means the session timeline began ``value`` seconds ago."""
    stream = _AudioStream(stt=_DummySTT())
    frame = silence_frame(0.02, 16000)
    pushed_at = time.time()
    stream.push_frame(frame)
    await _wait_for_offsets(stream, 1)
    await asyncio.sleep(5.0)

    set_at = time.time()
    stream.start_time_offset = 30.0

    assert stream.start_time_offset == pytest.approx(30.0 - (set_at - (pushed_at - frame.duration)))
    await stream.aclose()


async def test_start_time_offset_reanchors_on_retry_at_the_next_frame() -> None:
    """A new connection restarts the provider's clock with the next frame it reads, so the
    offset is re-anchored on that frame rather than grown by the previous attempt's duration."""
    stream = _AudioStream(stt=_DummySTT(), fail_first_run=True)
    set_at = time.time()
    stream.start_time_offset = 30.0
    frame = silence_frame(0.02, 16000)

    await asyncio.sleep(5.0)
    stream.push_frame(frame)
    await _wait_for_offsets(stream, 1)
    # the first attempt read that frame and failed; the retry waits for the next one
    await asyncio.sleep(1.0)
    pushed_at = time.time()
    stream.push_frame(frame)
    await _wait_for_offsets(stream, 2)

    assert stream.run_count == 2
    assert stream.offsets[1] == pytest.approx(30.0 + pushed_at - frame.duration - set_at)
    await stream.aclose()


async def test_start_time_offset_anchors_on_audio_queued_during_a_failed_connect() -> None:
    """Audio pushed while a connection fails is the next connection's first audio, so the
    offset is anchored on the oldest queued frame, not on the next push."""
    stream = _AudioStream(stt=_DummySTT(), fail_first_connect=True)
    set_at = time.time()
    stream.start_time_offset = 30.0
    frame = silence_frame(0.02, 16000)

    await asyncio.sleep(1.0)
    first_pushed_at = time.time()
    # one second of live audio while the connection is still failing
    for _ in range(50):
        stream.push_frame(frame)
        await asyncio.sleep(frame.duration)
    await _wait_for_offsets(stream, 1)

    assert stream.run_count == 2
    assert stream.offsets == [pytest.approx(30.0 + first_pushed_at - frame.duration - set_at)]
    await stream.aclose()


@pytest.mark.parametrize("pause", [False, True])
async def test_start_time_offset_anchors_on_the_capture_of_queued_audio(pause: bool) -> None:
    """Queued audio is anchored on when its oldest frame was captured: a pause between
    queued frames does not move it forward, and a burst of buffered audio moves it back."""
    stream = _AudioStream(stt=_DummySTT(), fail_first_connect=True)
    set_at = time.time()
    stream.start_time_offset = 30.0
    frame = silence_frame(0.02, 16000)

    await asyncio.sleep(1.0)
    first_pushed_at = time.time()
    if pause:
        # input pauses for a second between two queued frames
        stream.push_frame(frame)
        await asyncio.sleep(1.0)
        stream.push_frame(frame)
        captured_at = first_pushed_at - frame.duration
    else:
        # a second of buffered audio arrives at once
        for _ in range(50):
            stream.push_frame(frame)
        captured_at = first_pushed_at - 50 * frame.duration
    await _wait_for_offsets(stream, 1)

    # the pushes themselves take a few microseconds
    assert stream.offsets == [pytest.approx(30.0 + captured_at - set_at, abs=1e-3)]
    await stream.aclose()


async def test_undeclared_start_time_offset_counts_from_the_first_audio() -> None:
    """With no offset set the timeline starts with the first audio, so a late first frame
    adds nothing, and a retry continues that timeline from the next frame read."""
    stream = _AudioStream(stt=_DummySTT(), fail_first_run=True)
    frame = silence_frame(0.02, 16000)

    await asyncio.sleep(5.0)
    first_pushed_at = time.time()
    stream.push_frame(frame)
    await _wait_for_offsets(stream, 1)
    assert stream.offsets == [0.0]

    await asyncio.sleep(1.0)
    pushed_at = time.time()
    stream.push_frame(frame)
    await _wait_for_offsets(stream, 2)

    assert stream.offsets[1] == pytest.approx(pushed_at - first_pushed_at)
    await stream.aclose()


class _ProviderAudioStream(RecognizeStream):
    """Read input frames until the test closes the stream."""

    def __init__(self, *, stt: STT) -> None:
        super().__init__(stt=stt, conn_options=DEFAULT_API_CONNECT_OPTIONS)

    async def _run(self) -> None:
        async for _ in self._input_ch:
            pass


async def _push_live_audio(
    stream: RecognizeStream, duration: float, *, frame_duration: float = 0.02
) -> None:
    """Push ``duration`` seconds of audio, each frame when its capture ends."""
    frame = silence_frame(frame_duration, 16000)
    for _ in range(round(duration / frame_duration)):
        await asyncio.sleep(frame_duration)
        stream.push_frame(frame)
    await _wait_until_read(stream)


async def _wait_until_read(stream: RecognizeStream) -> None:
    while not stream._input_ch.empty():
        await asyncio.sleep(0)


def _transcript(start: float = 0.0, end: float = 0.0, **fields: Any) -> SpeechEvent:
    return SpeechEvent(
        type=SpeechEventType.FINAL_TRANSCRIPT,
        alternatives=[
            SpeechData(language="en", text="hello", start_time=start, end_time=end, **fields)
        ],
    )


async def _send(
    stream: RecognizeStream, event: SpeechEvent, connection_start: float | None = None
) -> SpeechEvent:
    """Translate with the selected connection clock and return the event."""
    if connection_start is not None:
        return _ConnectionClock(stream._input_clock, connection_start).translate_event_timestamps(
            event
        )
    stream._event_ch.send_nowait(event)
    return await stream.__anext__()


async def _emit(
    stream: RecognizeStream,
    start: float = 0.0,
    end: float = 0.0,
    *,
    connection_start: float | None = None,
    **fields: Any,
) -> SpeechData:
    """Send a transcript and return the consumer's event."""
    event = _transcript(start, end, **fields)
    return (await _send(stream, event, connection_start)).alternatives[0]


async def _end(stream: RecognizeStream, connection_time: float, **kwargs: Any) -> float:
    """Where speech that ends at ``connection_time`` is placed."""
    return (await _emit(stream, end=connection_time, **kwargs)).end_time


async def _start(stream: RecognizeStream, connection_time: float, **kwargs: Any) -> float:
    """Where speech that starts at ``connection_time`` is placed."""
    return (await _emit(stream, start=connection_time, **kwargs)).start_time


async def test_event_channel_places_audio_where_it_was_captured_across_input_gaps() -> None:
    """Preserve current and delayed speech timestamps across repeated input gaps."""
    stream = _ProviderAudioStream(stt=_DummySTT())
    stream.start_time_offset = 30.0

    await _push_live_audio(stream, 1.0)
    before_gap = await _end(stream, 0.5)
    await asyncio.sleep(5.0)
    await _push_live_audio(stream, 1.0)
    await asyncio.sleep(3.0)
    await _push_live_audio(stream, 1.0)

    # the provider received 3s of audio while 11s passed
    assert before_gap == pytest.approx(30.5)
    assert await _end(stream, 0.5) == pytest.approx(30.5)  # a late result keeps its place
    assert await _end(stream, 1.0) == pytest.approx(31.0)  # speech that ends with the audio
    assert await _end(stream, 1.5) == pytest.approx(30.0 + 6.5)
    assert await _end(stream, 2.5) == pytest.approx(30.0 + 10.5)
    spans_two_gaps = await _emit(stream, 0.5, 2.5)
    assert (spans_two_gaps.start_time, spans_two_gaps.end_time) == pytest.approx((30.5, 40.5))
    # the offset stays the position of the provider's zero
    assert stream.start_time_offset == pytest.approx(30.0)
    await stream.aclose()


async def test_event_channel_adds_the_declared_offset_when_the_input_has_no_gap() -> None:
    """Apply ``start_time_offset`` before the first frame and during continuous input."""
    stream = _ProviderAudioStream(stt=_DummySTT())
    stream.start_time_offset = 30.0

    before_audio = await _emit(stream, 0.2, 0.4)
    assert (before_audio.start_time, before_audio.end_time) == pytest.approx((30.2, 30.4))

    await _push_live_audio(stream, 2.0)
    words = [
        TimedString("hello", start_time=0.5, end_time=0.9),
        TimedString("there", start_time=1.0, end_time=1.5),
    ]
    live = await _emit(stream, 0.5, 1.5, words=words)
    assert (live.start_time, live.end_time) == pytest.approx((30.5, 31.5))
    assert live.words is not None
    assert [(w.start_time, w.end_time) for w in live.words] == [
        pytest.approx((30.5, 30.9)),
        pytest.approx((31.0, 31.5)),
    ]
    assert [w.start_time_offset for w in live.words] == pytest.approx([30.0, 30.0])
    await stream.aclose()


async def test_stream_without_a_declared_offset_counts_from_its_first_audio() -> None:
    """A stream used on its own has no session: its times count from its first audio."""
    stream = _ProviderAudioStream(stt=_DummySTT())

    await asyncio.sleep(3.0)  # the stream waits for its first audio
    await _push_live_audio(stream, 1.0)
    await asyncio.sleep(5.0)
    await _push_live_audio(stream, 1.0)

    assert await _end(stream, 0.5) == pytest.approx(0.5)
    assert await _end(stream, 1.5) == pytest.approx(6.5)
    await stream.aclose()


async def test_event_channel_places_the_edge_of_an_input_gap_by_what_happens_there() -> None:
    """Distinguish gap edges despite floating-point error in frame durations.

    Short spans on either side retain their position and duration.
    """
    stream = _ProviderAudioStream(stt=_DummySTT())
    stream.start_time_offset = 30.0

    # ten 100 ms frames add up to 0.9999999999999999
    await _push_live_audio(stream, 1.0, frame_duration=0.1)
    await asyncio.sleep(5.0)
    await _push_live_audio(stream, 1.0, frame_duration=0.1)
    assert stream._input_clock._runs[1].pushed_duration != 1.0

    # compared to 0.1 ms: the wall clock is an epoch float
    assert await _end(stream, 1.0) == pytest.approx(31.0, abs=1e-4)
    assert await _start(stream, 1.0) == pytest.approx(36.0, abs=1e-4)

    spans = {
        (0.997, 0.999): (30.997, 30.999),  # before the gap
        (1.0, 1.003): (36.0, 36.003),  # starts with the audio after the gap
        (1.001, 1.002): (36.001, 36.002),  # after the gap
        (0.998, 1.002): (30.998, 36.002),  # across the gap
        (0.9, 1.0): (30.9, 31.0),  # ends with the audio before the gap
        (1.0, 1.0): (31.0, 31.0),  # no length: one side for both bounds
    }
    for (start, end), expected in spans.items():
        word = TimedString("hello", start_time=start, end_time=end)
        placed = await _emit(stream, start, end, words=[word])
        assert (placed.start_time, placed.end_time) == pytest.approx(expected, abs=1e-4)
        assert placed.end_time >= placed.start_time
        assert placed.words is not None
        placed_word = placed.words[0]
        assert (placed_word.start_time, placed_word.end_time) == pytest.approx(expected, abs=1e-4)
        assert placed_word.end_time >= placed_word.start_time
    await stream.aclose()


async def test_event_channel_keeps_unknown_times_and_every_other_field() -> None:
    """Copy remapped timestamps while preserving unknown bounds and other fields."""
    stream = _ProviderAudioStream(stt=_DummySTT())
    stream.start_time_offset = 30.0
    first_audio_at = time.time()
    await _push_live_audio(stream, 1.0)
    await asyncio.sleep(5.0)
    await _push_live_audio(stream, 1.0)

    words = [
        TimedString("hello", start_time=0.5, end_time=0.9, confidence=0.7, speaker_id="S1"),
        TimedString("there"),  # no times
        TimedString("friend", start_time=1.2),  # no end
    ]
    primary = SpeechData(
        language="en",
        text="hello there friend",
        start_time=0.5,
        end_time=1.4,
        confidence=0.9,
        speaker_id="S1",
        is_primary_speaker=True,
        words=words,
        source_languages=["fr"],
        source_texts=["bonjour"],
        metadata={"provider": {"start": 0.5}},
    )
    untimed = SpeechData(language="en", text="hello")
    start_only = SpeechData(language="en", text="hello", start_time=1.5)
    sent = SpeechEvent(
        type=SpeechEventType.FINAL_TRANSCRIPT,
        request_id="req-1",
        alternatives=[primary, untimed, start_only],
        speech_start_time=0.5,
        speech_end_time=1.4,
        created_at=first_audio_at + 20.0,
    )
    sent_before = dataclasses.replace(
        sent, alternatives=[dataclasses.replace(alt) for alt in sent.alternatives]
    )

    stream._event_ch.send_nowait(sent)
    received = await stream.__anext__()

    placed, placed_untimed, placed_start_only = received.alternatives
    assert (placed.start_time, placed.end_time) == pytest.approx((30.5, 36.4))
    assert placed == dataclasses.replace(
        primary, start_time=placed.start_time, end_time=placed.end_time, words=placed.words
    )
    assert placed.words is not None
    hello, there, friend = placed.words
    assert (str(hello), hello.confidence, hello.speaker_id) == ("hello", 0.7, "S1")
    assert (hello.start_time, hello.end_time) == pytest.approx((30.5, 30.9))
    assert not is_given(there.start_time) and not is_given(there.end_time)
    assert friend.start_time == pytest.approx(36.2) and not is_given(friend.end_time)
    assert placed_untimed == untimed
    assert (placed_start_only.start_time, placed_start_only.end_time) == (
        pytest.approx(36.5),
        0.0,
    )
    # the speech bounds are on the wall clock; the rest of the event is as it was sent
    assert received.speech_start_time == pytest.approx(first_audio_at + 0.5, rel=0, abs=1e-4)
    assert received.speech_end_time == pytest.approx(first_audio_at + 6.4, rel=0, abs=1e-4)
    assert dataclasses.replace(
        received, alternatives=[], speech_start_time=None, speech_end_time=None
    ) == dataclasses.replace(sent, alternatives=[], speech_start_time=None, speech_end_time=None)

    # what the plugin holds is as it was
    assert sent == sent_before
    assert sent.alternatives[0] is primary and primary.words is words
    assert (words[0].start_time, words[0].end_time) == (0.5, 0.9)
    assert placed is not primary and hello is not words[0]

    # an event without alternatives or speech bounds has nothing to place
    marker = SpeechEvent(type=SpeechEventType.START_OF_SPEECH)
    stream._event_ch.send_nowait(marker)
    assert await stream.__anext__() == marker
    await stream.aclose()


async def test_reverse_remap_returns_stream_positions_across_input_gaps() -> None:
    """Reverse remapped times to stream positions across gaps and a nonzero session origin."""
    stream = _ProviderAudioStream(stt=_DummySTT())
    stream.start_time_offset = 30.0
    await _push_live_audio(stream, 1.0)
    await asyncio.sleep(5.0)
    await _push_live_audio(stream, 1.0)
    await asyncio.sleep(3.0)
    await _push_live_audio(stream, 1.0)
    reverse = stream._input_clock.session_to_stream_timestamps

    def positions(start: float, end: float) -> tuple[float, float]:
        reversed_data = reverse(_transcript(start, end).alternatives[0])
        assert reversed_data is not None
        return reversed_data.start_time, reversed_data.end_time

    # one connection that read the stream from its start: connection time is stream position
    placed = await _emit(stream, 0.5, 2.5)
    assert (placed.start_time, placed.end_time) == pytest.approx((30.5, 40.5))
    assert positions(placed.start_time, placed.end_time) == pytest.approx((0.5, 2.5), abs=1e-4)

    # both edges of a gap, and any time within it, share one stream position
    assert positions(31.0, 36.0) == pytest.approx((1.0, 1.0), abs=1e-4)
    assert positions(33.0, 38.5) == pytest.approx((1.0, 2.0), abs=1e-4)
    assert positions(36.2, 39.9) == pytest.approx((1.2, 2.0), abs=1e-4)
    # times outside the audio clamp to its bounds
    assert positions(10.0, 100.0) == pytest.approx((0.0, 3.0), abs=1e-4)
    await stream.aclose()


async def test_reverse_remap_keeps_unknown_times_and_every_other_field() -> None:
    """Copy reversed timestamps while preserving unknown bounds and other fields."""
    stream = _ProviderAudioStream(stt=_DummySTT())
    stream.start_time_offset = 30.0
    await _push_live_audio(stream, 1.0)
    await asyncio.sleep(5.0)
    await _push_live_audio(stream, 1.0)
    reverse = stream._input_clock.session_to_stream_timestamps

    words = [
        TimedString(
            "hello",
            start_time=30.5,
            end_time=30.9,
            confidence=0.7,
            start_time_offset=30.0,
            speaker_id="S1",
        ),
        TimedString("there"),  # no times
        TimedString("friend", start_time=36.2),  # no end
    ]
    metadata = {"provider": {"start": 0.5}}
    remapped = SpeechData(
        language="en",
        text="hello there friend",
        start_time=30.5,
        end_time=36.4,
        confidence=0.9,
        speaker_id="S1",
        is_primary_speaker=True,
        words=words,
        source_languages=["fr"],
        source_texts=["bonjour"],
        metadata=metadata,
    )
    remapped_before = dataclasses.replace(remapped)

    reversed_data = reverse(remapped)

    assert reversed_data is not None and reversed_data is not remapped
    assert (reversed_data.start_time, reversed_data.end_time) == pytest.approx((0.5, 1.4), abs=1e-4)
    assert reversed_data == dataclasses.replace(
        remapped,
        start_time=reversed_data.start_time,
        end_time=reversed_data.end_time,
        words=reversed_data.words,
    )
    # Provider metadata stays unchanged.
    assert reversed_data.metadata is metadata and metadata == {"provider": {"start": 0.5}}
    assert reversed_data.words is not None
    hello, there, friend = reversed_data.words
    assert (str(hello), hello.confidence, hello.speaker_id) == ("hello", 0.7, "S1")
    assert (hello.start_time, hello.end_time) == pytest.approx((0.5, 0.9), abs=1e-4)
    assert not is_given(there.start_time) and not is_given(there.end_time)
    assert friend.start_time == pytest.approx(1.2, abs=1e-4) and not is_given(friend.end_time)
    # stream positions count from the stream's first frame
    assert [w.start_time_offset for w in reversed_data.words] == [0.0, 0.0, 0.0]

    untimed = SpeechData(language="en", text="hello")
    assert reverse(untimed) == untimed
    start_only = reverse(SpeechData(language="en", text="hello", start_time=36.5))
    assert start_only is not None
    assert (start_only.start_time, start_only.end_time) == (pytest.approx(1.5, abs=1e-4), 0.0)

    # Leave the caller's data and words unchanged.
    assert remapped == remapped_before and remapped.words is words
    assert hello is not words[0]
    assert (words[0].start_time, words[0].end_time, words[0].start_time_offset) == (
        30.5,
        30.9,
        30.0,
    )
    await stream.aclose()


async def test_reverse_remap_continues_stream_positions_across_a_reconnect() -> None:
    """Return the position in the whole stream, not the new connection's offset."""
    stream = _ProviderAudioStream(stt=_DummySTT())
    stream.start_time_offset = 30.0
    await _push_live_audio(stream, 1.0)
    await asyncio.sleep(5.0)
    stream._restart_connection_clock()
    await _push_live_audio(stream, 1.0)

    # the new connection counts from its first frame
    words = [TimedString("hello", start_time=0.5, end_time=0.8)]
    placed = await _emit(stream, 0.5, 0.8, words=words)
    assert (placed.start_time, placed.end_time) == pytest.approx((36.5, 36.8))

    reversed_data = stream._input_clock.session_to_stream_timestamps(placed)
    assert reversed_data is not None and reversed_data.words is not None
    assert (reversed_data.start_time, reversed_data.end_time) == pytest.approx((1.5, 1.8), abs=1e-4)
    word = reversed_data.words[0]
    assert (word.start_time, word.end_time) == pytest.approx((1.5, 1.8), abs=1e-4)
    assert word.start_time_offset == 0.0
    await stream.aclose()


async def test_reverse_remap_has_no_stream_position_before_the_first_audio() -> None:
    """Without audio there is no stream clock, with or without a declared offset."""
    data = _transcript(30.5, 30.8).alternatives[0]

    stream = _ProviderAudioStream(stt=_DummySTT())
    assert stream._input_clock.session_to_stream_timestamps(data) is None
    stream.start_time_offset = 30.0
    assert stream._input_clock.session_to_stream_timestamps(data) is None
    assert (data.start_time, data.end_time) == (30.5, 30.8)
    await stream.aclose()


async def _speech_bounds(
    stream: RecognizeStream,
    start: float | None = None,
    end: float | None = None,
    *,
    since: float,
    connection_start: float | None = None,
) -> tuple[float | None, float | None]:
    """Send speech bounds without a transcript and return wall times relative to ``since``.

    Compare with ``abs=1e-4``: the wall clock is an epoch float.
    """
    event_type = SpeechEventType.END_OF_SPEECH if start is None else SpeechEventType.START_OF_SPEECH
    sent = SpeechEvent(type=event_type, speech_start_time=start, speech_end_time=end)
    received = await _send(stream, sent, connection_start)
    assert (sent.speech_start_time, sent.speech_end_time) == (start, end)
    assert (received.type, received.created_at) == (sent.type, sent.created_at)
    placed_start, placed_end = received.speech_start_time, received.speech_end_time
    return (
        None if placed_start is None else placed_start - since,
        None if placed_end is None else placed_end - since,
    )


async def test_event_channel_places_speech_bounds_on_the_wall_clock_across_an_input_gap() -> None:
    """Convert speech bounds from connection timestamps to wall-clock timestamps."""
    stream = _ProviderAudioStream(stt=_DummySTT())
    stream.start_time_offset = 30.0
    t0 = time.time()
    await _push_live_audio(stream, 1.0)
    await asyncio.sleep(5.0)
    await _push_live_audio(stream, 1.0)

    cases: dict[tuple[float | None, float | None], tuple[float | None, float | None]] = {
        (None, None): (None, None),
        (0.0, None): (0.0, None),  # speech from the first sample is not a missing time
        (None, 0.0): (None, 0.0),
        (0.5, None): (0.5, None),
        (1.0, None): (6.0, None),  # speech that starts at the gap starts after it
        (None, 1.0): (None, 1.0),  # speech that ends at the gap ends before it
        (1.5, None): (6.5, None),
        (None, 1.5): (None, 6.5),
        (0.5, 1.5): (0.5, 6.5),
        (1.0, 1.0): (1.0, 1.0),  # no length: one side for both bounds
        (0.0, 0.0): (0.0, 0.0),
    }
    for (start, end), expected in cases.items():
        placed = await _speech_bounds(stream, start, end, since=t0)
        assert placed == pytest.approx(expected, rel=0, abs=1e-4), (start, end)
    await stream.aclose()


async def test_final_transcript_speech_end_is_placed_with_its_alternatives() -> None:
    stream = _ProviderAudioStream(stt=_DummySTT())
    await asyncio.sleep(3.0)  # the stream waits for its first audio
    t0 = time.time()
    await _push_live_audio(stream, 1.0)
    await asyncio.sleep(5.0)
    await _push_live_audio(stream, 1.0)

    stream._event_ch.send_nowait(dataclasses.replace(_transcript(1.2, 1.6), speech_end_time=1.5))
    received = await stream.__anext__()
    placed = received.alternatives[0]
    # the transcript counts from the first audio, the speech end is on the wall clock
    assert (placed.start_time, placed.end_time) == pytest.approx((6.2, 6.6))
    assert received.speech_start_time is None
    assert received.speech_end_time == pytest.approx(t0 + 6.5, rel=0, abs=1e-4)
    await stream.aclose()


async def test_speech_bounds_before_the_first_audio() -> None:
    """Without audio or a declared origin there is no clock to place a bound on."""
    stream = _ProviderAudioStream(stt=_DummySTT())
    assert await _speech_bounds(stream, 0.5, 0.8, since=0.0) == (None, None)

    declared_at = time.time()
    stream.start_time_offset = 30.0  # the provider's zero is declared to be now
    placed = await _speech_bounds(stream, 0.5, 0.8, since=declared_at)
    assert placed == pytest.approx((0.5, 0.8), rel=0, abs=1e-4)
    await stream.aclose()


async def test_speech_bounds_queued_before_a_reconnect_keep_the_clock_of_their_connection() -> None:
    stream = _ProviderAudioStream(stt=_DummySTT())
    stream.start_time_offset = 30.0
    t0 = time.time()

    await _push_live_audio(stream, 1.0)
    send = stream._event_ch.send_nowait
    send(SpeechEvent(type=SpeechEventType.START_OF_SPEECH, speech_start_time=0.5))  # not read yet
    await asyncio.sleep(5.0)
    stream._restart_connection_clock()
    # a result of the closed connection keeps its place until the new one reads audio
    send(SpeechEvent(type=SpeechEventType.END_OF_SPEECH, speech_end_time=0.9))
    await _push_live_audio(stream, 1.0)
    send(SpeechEvent(type=SpeechEventType.START_OF_SPEECH, speech_start_time=0.5))
    # audio replayed from 0.5 s of the input, and audio counted from the input's start
    replayed = [
        await _send(
            stream, SpeechEvent(type=SpeechEventType.END_OF_SPEECH, speech_end_time=1.0), 0.5
        ),
        await _send(
            stream, SpeechEvent(type=SpeechEventType.START_OF_SPEECH, speech_start_time=0.0), 0.0
        ),
    ]

    received = [await stream.__anext__() for _ in range(3)] + replayed
    placed = [
        ev.speech_end_time if ev.speech_start_time is None else ev.speech_start_time
        for ev in received
    ]
    assert [t - t0 for t in placed if t is not None] == pytest.approx(
        [0.5, 0.9, 6.5, 6.5, 0.0], rel=0, abs=1e-4
    )
    await stream.aclose()


async def test_event_channel_places_events_sent_with_await_send() -> None:
    stream = _ProviderAudioStream(stt=_DummySTT())
    stream.start_time_offset = 30.0
    await _push_live_audio(stream, 1.0)
    await asyncio.sleep(5.0)
    await _push_live_audio(stream, 1.0)

    await stream._event_ch.send(_transcript(1.5, 1.8))
    placed = (await stream.__anext__()).alternatives[0]
    assert (placed.start_time, placed.end_time) == pytest.approx((36.5, 36.8))
    await stream.aclose()


async def test_adapter_stream_forwards_placed_events_as_they_are() -> None:
    """Forward already remapped events unchanged, including after an input gap."""
    stream = _ProviderAudioStream(stt=_DummySTT(requires_timestamp_translation=False))
    stream.start_time_offset = 30.0
    await _push_live_audio(stream, 1.0)
    await asyncio.sleep(5.0)
    await _push_live_audio(stream, 1.0)

    sent = dataclasses.replace(
        _transcript(stream.start_time_offset + 1.5, stream.start_time_offset + 1.8),
        speech_start_time=time.time() - 0.5,
        speech_end_time=time.time() - 0.2,
    )
    sent_before = dataclasses.replace(
        sent, alternatives=[dataclasses.replace(sent.alternatives[0])]
    )
    stream._event_ch.send_nowait(sent)
    received = await stream.__anext__()
    assert received is sent and sent == sent_before
    assert (received.alternatives[0].start_time, received.alternatives[0].end_time) == (
        pytest.approx((31.5, 31.8))
    )
    await stream.aclose()


async def test_event_queued_before_a_reconnect_keeps_the_clock_of_its_connection() -> None:
    """Keep the originating connection's mapping for events queued across a reconnect."""
    stream = _ProviderAudioStream(stt=_DummySTT())
    stream.start_time_offset = 30.0

    await _push_live_audio(stream, 1.0)
    stream._event_ch.send_nowait(_transcript(0.5, 0.8))  # not read yet
    await asyncio.sleep(5.0)
    stream._restart_connection_clock()
    # a result of the closed connection keeps its place until the new one reads audio
    stream._event_ch.send_nowait(_transcript(0.6, 0.9))
    await _push_live_audio(stream, 1.0)
    stream._event_ch.send_nowait(_transcript(0.5, 0.8))  # the new connection's clock

    received = [(await stream.__anext__()).alternatives[0] for _ in range(3)]
    assert [(alt.start_time, alt.end_time) for alt in received] == [
        pytest.approx((30.5, 30.8)),
        pytest.approx((30.6, 30.9)),
        pytest.approx((36.5, 36.8)),
    ]
    await stream.aclose()


async def test_event_channel_remaps_each_event_once(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = _ProviderAudioStream(stt=_DummySTT())
    stream.start_time_offset = 30.0
    await _push_live_audio(stream, 1.0)
    translate_event_timestamps = _ConnectionClock.translate_event_timestamps
    remapped: list[SpeechEvent] = []

    def spy(clock: _ConnectionClock, event: SpeechEvent) -> SpeechEvent:
        remapped.append(event)
        return translate_event_timestamps(clock, event)

    monkeypatch.setattr(_ConnectionClock, "translate_event_timestamps", spy)
    sent = _transcript(0.5, 0.8)
    stream._event_ch.send_nowait(sent)

    received = (await stream.__anext__()).alternatives[0]
    assert (received.start_time, received.end_time) == pytest.approx((30.5, 30.8))
    assert len(remapped) == 1 and remapped[0] is sent
    await stream.aclose()


async def test_earlier_connection_clock_keeps_delayed_result_timestamps() -> None:
    """Each connection keeps its start position when a response arrives late."""
    stream = _ProviderAudioStream(stt=_DummySTT())
    stream.start_time_offset = 30.0

    await _push_live_audio(stream, 1.0)
    first_connection_clock = _ConnectionClock(stream._input_clock, 0.0)
    await asyncio.sleep(5.0)
    stream._restart_connection_clock()
    await _push_live_audio(stream, 1.0)

    assert stream._input_clock.dequeued_duration == pytest.approx(2.0)
    # The earlier connection's clock is kept after the next connection reads audio.
    delayed = first_connection_clock.translate_event_timestamps(_transcript(0.5, 0.8)).alternatives[
        0
    ]
    assert (delayed.start_time, delayed.end_time) == pytest.approx((30.5, 30.8))
    # The first connection started at stream position zero.
    assert await _end(stream, 0.5, connection_start=0.0) == pytest.approx(30.5)
    assert await _end(stream, 1.0, connection_start=0.0) == pytest.approx(31.0)
    assert await _start(stream, 1.0, connection_start=0.0) == pytest.approx(36.0)
    assert await _end(stream, 1.5, connection_start=0.0) == pytest.approx(36.5)
    # Another connection started at stream position 0.5 s.
    assert await _end(stream, 0.2, connection_start=0.5) == pytest.approx(30.7)
    assert await _end(stream, 1.0, connection_start=0.5) == pytest.approx(36.5)
    assert await _end(stream, 0.7, connection_start=0.0) == pytest.approx(30.7)
    # the same audio, counted from the first frame of the current connection
    assert await _end(stream, 0.5) == pytest.approx(36.5)
    await stream.aclose()


async def test_start_time_offset_stays_when_the_input_history_is_dropped() -> None:
    """Keep provider zero fixed after pruning history; re-anchor on reconnect."""
    stream = _ProviderAudioStream(stt=_DummySTT())
    stream.start_time_offset = 30.0
    frame = silence_frame(0.02, 16000)

    await _push_live_audio(stream, 1.0)
    for _ in range(70):
        await asyncio.sleep(10.0)
        stream.push_frame(frame)
    await _wait_until_read(stream)

    assert stream._input_clock._runs[0].pushed_duration > 1.0  # the first run is dropped
    assert stream.start_time_offset == pytest.approx(30.0)
    # recent audio keeps its place: frame 70 was captured at 700.98
    recent = 1.0 + 69.5 * frame.duration
    assert await _end(stream, recent) == pytest.approx(30.0 + 700.99)
    assert (await _emit(stream, 0.0, recent)).start_time == pytest.approx(30.0)

    stream._restart_connection_clock()
    await asyncio.sleep(5.0)
    await _push_live_audio(stream, 1.0)
    assert stream.start_time_offset == pytest.approx(30.0 + 706.0)
    assert await _end(stream, 0.5) == pytest.approx(30.0 + 706.5)
    await stream.aclose()


async def test_connection_that_follows_an_input_gap_counts_from_its_first_audio() -> None:
    """Re-anchor on reconnect and account for later input gaps once."""
    stream = _ProviderAudioStream(stt=_DummySTT())
    stream.start_time_offset = 30.0

    await _push_live_audio(stream, 1.0)
    await asyncio.sleep(5.0)
    stream._restart_connection_clock()
    await _push_live_audio(stream, 1.0)
    await asyncio.sleep(2.0)
    await _push_live_audio(stream, 1.0)

    assert stream.start_time_offset == pytest.approx(30.0 + 6.0)
    assert await _end(stream, 0.5) == pytest.approx(30.0 + 6.5)
    assert await _end(stream, 1.5) == pytest.approx(30.0 + 9.5)
    await stream.aclose()


@pytest.mark.parametrize(
    "delays",
    [
        pytest.param([0.5] + [0.0] * 24, id="stall"),
        pytest.param([0.035, 0.005] * 25, id="jitter"),
        pytest.param([0.09, 0.0, 0.0, 0.0, 0.01] * 10, id="chunked"),
    ],
)
async def test_late_frames_are_not_read_as_missing_audio(delays: list[float]) -> None:
    """Treat catch-up frames as buffered audio rather than missing audio."""
    stream = _ProviderAudioStream(stt=_DummySTT())
    stream.start_time_offset = 30.0
    frame = silence_frame(0.02, 16000)

    await _push_live_audio(stream, 1.0)
    for delay in delays:
        await asyncio.sleep(delay)
        stream.push_frame(frame)
    await _wait_until_read(stream)
    await _push_live_audio(stream, 1.0)

    audio_duration = 2.0 + len(delays) * frame.duration
    for connection_time in (0.5, 1.0, 1.2, 1.4, audio_duration - 0.5, audio_duration):
        # a chunk is placed by its last frame, the rest of it up to one frame early
        assert await _end(stream, connection_time) == pytest.approx(
            30.0 + connection_time, abs=frame.duration + 1e-6
        )
    await stream.aclose()


async def test_event_channel_keeps_a_bounded_history_of_input_gaps() -> None:
    """Recent audio keeps its place however many pauses came before it."""
    stream = _ProviderAudioStream(stt=_DummySTT())
    stream.start_time_offset = 30.0
    frame = silence_frame(0.02, 16000)

    for _ in range(100):
        await asyncio.sleep(10.0)
        stream.push_frame(frame)
    await _wait_until_read(stream)

    # frame n was captured 10s after frame n - 1 was pushed
    assert await _end(stream, 99.5 * frame.duration) == pytest.approx(30.0 + 999.99)
    assert await _end(stream, 80.5 * frame.duration) == pytest.approx(30.0 + 809.99)
    assert len(stream._input_clock._runs) < 40
    await stream.aclose()


def _stream_positions(stream: RecognizeStream, start: float, end: float) -> tuple[float, float]:
    reversed_data = stream._input_clock.session_to_stream_timestamps(
        _transcript(start, end).alternatives[0]
    )
    assert reversed_data is not None
    return reversed_data.start_time, reversed_data.end_time


@pytest.mark.parametrize("pause", [0.0, 1.0])
async def test_silence_pushed_in_a_burst_keeps_the_place_of_earlier_speech(pause: float) -> None:
    """Silence pushed at once to flush a turn is ahead of real time. It must not move
    the speech before it, and live audio that resumes is placed where it was captured."""
    stream = _ProviderAudioStream(stt=_DummySTT())
    stream.start_time_offset = 30.0
    await _push_live_audio(stream, 1.0)
    await asyncio.sleep(5.0)
    await _push_live_audio(stream, 3.0)  # captured from 6 s to 9 s

    silence = silence_frame(0.2, 16000)
    for _ in range(10):
        stream.push_frame(silence)
    await _wait_until_read(stream)

    speech = await _emit(stream, 1.1, 3.9)
    assert (speech.start_time, speech.end_time) == pytest.approx((36.1, 38.9))
    assert _stream_positions(stream, 36.1, 38.9) == pytest.approx((1.1, 3.9), abs=1e-4)

    # live audio resumes while the 2 s of silence are still ahead of real time
    await asyncio.sleep(pause)
    await _push_live_audio(stream, 1.0)
    resumed_at = 39.0 + pause

    assert await _end(stream, 6.5) == pytest.approx(resumed_at + 0.5)
    assert _stream_positions(stream, resumed_at + 0.5, resumed_at + 0.9) == pytest.approx(
        (6.5, 6.9), abs=1e-4
    )
    # the earlier speech stays in place, in both directions
    speech = await _emit(stream, 1.1, 3.9)
    assert (speech.start_time, speech.end_time) == pytest.approx((36.1, 38.9))
    assert _stream_positions(stream, 36.1, 38.9) == pytest.approx((1.1, 3.9), abs=1e-4)
    # the silence ends where the live audio resumed; times stay in order
    assert await _end(stream, 5.9) <= await _end(stream, 6.5)
    assert stream.start_time_offset == pytest.approx(30.0)
    await stream.aclose()


async def test_live_audio_right_after_a_burst_is_placed_when_it_is_pushed() -> None:
    """Speech that resumes right after flush silence is not placed behind the silence,
    and its place does not change when more live audio follows."""
    stream = _ProviderAudioStream(stt=_DummySTT())
    stream.start_time_offset = 30.0
    await _push_live_audio(stream, 1.0)
    await asyncio.sleep(5.0)
    await _push_live_audio(stream, 3.0)  # captured from 6 s to 9 s

    silence = silence_frame(0.2, 16000)
    for _ in range(10):  # stream positions 4 s to 6 s, all pushed at 9 s
        stream.push_frame(silence)
    await _push_live_audio(stream, 0.1)  # captured from 9 s to 9.1 s

    def check_resumed_speech(resumed: SpeechData) -> None:
        assert (resumed.start_time, resumed.end_time) == pytest.approx((39.02, 39.08))
        assert _stream_positions(stream, 39.02, 39.08) == pytest.approx((6.02, 6.08), abs=1e-4)

    check_resumed_speech(await _emit(stream, 6.02, 6.08))
    # the silence has no time of its own: it shares the time it was pushed at
    assert await _end(stream, 5.0) == pytest.approx(39.0)

    await _push_live_audio(stream, 0.9)
    check_resumed_speech(await _emit(stream, 6.02, 6.08))
    assert await _end(stream, 5.0) == pytest.approx(39.0)
    assert await _end(stream, 6.9) == pytest.approx(39.9)
    await stream.aclose()


async def test_inverse_bounds_keep_their_side_of_a_stopped_run() -> None:
    input_clock = _InputClock()
    input_clock.session_start = _SessionStart(monotonic_timestamp=0.0, wall_timestamp=0.0)
    clock = _ConnectionClock(input_clock, 0.0)
    for pushed_at in (1.0, 7.0, 8.0, 9.0):
        input_clock.on_audio_enqueued(1.0, pushed_at, clock)
    earlier = clock.connection_to_session_timestamps(
        SpeechData(language="en", text="hello", start_time=3.0, end_time=4.0)
    )
    for _ in range(10):
        input_clock.on_audio_enqueued(0.2, 9.0, clock)
    input_clock.on_audio_enqueued(0.02, 9.02, clock)

    reversed_earlier = input_clock.session_to_stream_timestamps(earlier)
    assert reversed_earlier is not None
    assert (reversed_earlier.start_time, reversed_earlier.end_time) == pytest.approx((3.0, 4.0))
    for bounds, expected in (
        ((8.0, 9.0), (3.0, 4.0)),
        ((9.0, 9.02), (6.0, 6.02)),
        ((9.0, 9.0), (4.0, 4.0)),
    ):
        data = SpeechData(
            language="en",
            text="hello",
            start_time=bounds[0],
            end_time=bounds[1],
            words=[TimedString("hello", start_time=bounds[0], end_time=bounds[1])],
        )
        reversed_data = input_clock.session_to_stream_timestamps(data)
        assert reversed_data is not None
        assert (reversed_data.start_time, reversed_data.end_time) == pytest.approx(expected)
        assert reversed_data.words is not None
        assert (
            reversed_data.words[0].start_time,
            reversed_data.words[0].end_time,
        ) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("stall", "burst", "runs", "speech"),
    [
        (0.5, 0.6, 1, (31.1, 31.4)),  # the burst closes the stall
        (5.0, 1.0, 2, (35.1, 35.4)),  # the burst was captured in the last second of the gap
    ],
)
async def test_delayed_result_does_not_fix_the_start_of_a_run_it_does_not_use(
    stall: float, burst: float, runs: int, speech: tuple[float, float]
) -> None:
    """A result for earlier audio arrives while buffered audio is pushed. The run
    of that audio still moves back to where the audio was captured."""
    stream = _ProviderAudioStream(stt=_DummySTT())
    stream.start_time_offset = 30.0
    frame = silence_frame(0.02, 16000)
    await _push_live_audio(stream, 1.0)
    await asyncio.sleep(stall)

    stream.push_frame(frame)  # the first buffered frame starts a run
    await _wait_until_read(stream)
    delayed = await _emit(stream, 0.2, 0.8)
    assert (delayed.start_time, delayed.end_time) == pytest.approx((30.2, 30.8))
    assert _stream_positions(stream, 30.2, 30.8) == pytest.approx((0.2, 0.8), abs=1e-4)

    for _ in range(round(burst / frame.duration) - 1):
        stream.push_frame(frame)
    await _wait_until_read(stream)

    assert len(stream._input_clock._runs) == runs
    placed = await _emit(stream, 1.1, 1.4)
    assert (placed.start_time, placed.end_time) == pytest.approx(speech, abs=1e-3)
    assert _stream_positions(stream, placed.start_time, placed.end_time) == pytest.approx(
        (1.1, 1.4), abs=1e-4
    )
    await stream.aclose()


async def test_placed_event_keeps_its_stream_position_when_later_audio_arrives_early() -> None:
    """A run that placed an event is fixed: audio that then arrives faster than real
    time neither moves the run nor merges it into the previous one."""
    stream = _ProviderAudioStream(stt=_DummySTT())
    stream.start_time_offset = 30.0
    await _push_live_audio(stream, 1.0, frame_duration=1.0)
    await asyncio.sleep(4.0)
    await _push_live_audio(stream, 1.0, frame_duration=1.0)  # captured from 5 s to 6 s

    # placed on the second run before any later frame confirms its clock
    placed = await _emit(stream, 1.0, 2.0)
    assert (placed.start_time, placed.end_time) == pytest.approx((35.0, 36.0))

    second = silence_frame(1.0, 16000)
    await asyncio.sleep(0.1)
    for early_frames in (1, 4):  # the first would move the run, the rest would merge it
        for _ in range(early_frames):
            stream.push_frame(second)
        await _wait_until_read(stream)
        assert _stream_positions(stream, 35.0, 36.0) == pytest.approx((1.0, 2.0), abs=1e-4)
        again = await _emit(stream, 1.0, 2.0)
        assert (again.start_time, again.end_time) == pytest.approx((35.0, 36.0))
    await stream.aclose()


async def test_catch_up_audio_does_not_move_a_run_that_placed_an_event() -> None:
    """Catch-up audio after a stall merges into the previous run without moving it."""
    stream = _ProviderAudioStream(stt=_DummySTT())
    stream.start_time_offset = 30.0
    frame = silence_frame(0.02, 16000)
    await _push_live_audio(stream, 1.0)
    placed = await _emit(stream, 0.2, 0.8)
    assert (placed.start_time, placed.end_time) == pytest.approx((30.2, 30.8))

    # a 0.5 s stall, then 0.6 s of audio at once: more than the stall held back
    await asyncio.sleep(0.5)
    for _ in range(30):
        stream.push_frame(frame)
    await _wait_until_read(stream)

    assert len(stream._input_clock._runs) == 1
    assert _stream_positions(stream, 30.2, 30.8) == pytest.approx((0.2, 0.8), abs=1e-4)
    assert await _end(stream, 0.8) == pytest.approx(30.8)
    await stream.aclose()


async def test_buffered_audio_after_an_input_gap_is_placed_where_it_was_captured() -> None:
    """A burst that opens a run is buffered audio: the run starts before the push."""
    stream = _ProviderAudioStream(stt=_DummySTT())
    stream.start_time_offset = 30.0
    frame = silence_frame(0.02, 16000)
    await _push_live_audio(stream, 1.0)
    await asyncio.sleep(9.0)
    for _ in range(50):  # captured from 9 s to 10 s, pushed at 10 s
        stream.push_frame(frame)
    await _push_live_audio(stream, 1.0)

    buffered = await _emit(stream, 1.0, 2.0)
    assert (buffered.start_time, buffered.end_time) == pytest.approx((39.0, 40.0))
    assert await _end(stream, 2.5) == pytest.approx(40.5)
    assert _stream_positions(stream, 39.5, 40.5) == pytest.approx((1.5, 2.5), abs=1e-4)
    await stream.aclose()


@pytest.mark.parametrize("step", [3600.0, -3600.0])
async def test_wall_clock_step_adds_and_hides_no_input_gap(
    step: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Input gaps are measured on the monotonic clock. Speech bounds stay on the
    wall clock of the session origin."""
    stream = _ProviderAudioStream(stt=_DummySTT())
    stream.start_time_offset = 30.0
    t0 = time.time()
    await _push_live_audio(stream, 1.0)

    wall_clock = time.time
    monkeypatch.setattr(time, "time", lambda: wall_clock() + step)
    await _push_live_audio(stream, 1.0)  # no gap, whatever the wall clock says
    await asyncio.sleep(5.0)
    await _push_live_audio(stream, 1.0)

    assert await _end(stream, 1.5) == pytest.approx(31.5)
    assert await _end(stream, 2.5) == pytest.approx(37.5)
    assert stream.start_time_offset == pytest.approx(30.0)
    session_start = stream._input_clock.session_start
    assert session_start is not None
    assert session_start.wall_timestamp == pytest.approx(t0 - 30.0, rel=0, abs=1e-4)
    placed = await _speech_bounds(stream, 0.5, 2.5, since=t0)
    assert placed == pytest.approx((0.5, 7.5), rel=0, abs=1e-4)
    await stream.aclose()


async def test_clock_made_by_a_plugin_keeps_its_origin_when_the_input_history_is_dropped() -> None:
    """A clock other than the channel's keeps its first frame's place from its first use."""
    stream = _ProviderAudioStream(stt=_DummySTT())
    stream.start_time_offset = 30.0
    frame = silence_frame(0.02, 16000)
    clock = _ConnectionClock(stream._input_clock, 0.0)  # made before any audio

    await _push_live_audio(stream, 1.0)
    assert clock.start_time_offset == pytest.approx(30.0)
    for _ in range(70):
        await asyncio.sleep(10.0)
        stream.push_frame(frame)
    await _wait_until_read(stream)

    assert stream._input_clock._runs[0].pushed_duration > 1.0  # the first run is dropped
    assert clock.start_time_offset == pytest.approx(30.0)
    recent = 1.0 + 69.5 * frame.duration
    placed = clock.translate_event_timestamps(_transcript(0.0, recent)).alternatives[0]
    assert (placed.start_time, placed.end_time) == pytest.approx((30.0, 30.0 + 700.99))
    await stream.aclose()


async def test_reverse_remap_has_no_stream_position_before_the_retained_history() -> None:
    """A time older than the retained runs has no position; it must not read later audio."""
    stream = _ProviderAudioStream(stt=_DummySTT())
    stream.start_time_offset = 30.0
    frame = silence_frame(0.02, 16000)
    await _push_live_audio(stream, 1.0)
    for _ in range(70):
        await asyncio.sleep(10.0)
        stream.push_frame(frame)
    await _wait_until_read(stream)
    reverse = stream._input_clock.session_to_stream_timestamps

    assert stream._input_clock._runs[0].pushed_duration > 1.0  # the first run is dropped
    assert reverse(_transcript(30.2, 30.8).alternatives[0]) is None
    old_word = [TimedString("hello", start_time=30.2, end_time=30.8)]
    assert reverse(_transcript(0.0, 730.99, words=old_word).alternatives[0]) is None
    # a start of 0 is a known time when the end is known
    assert reverse(_transcript(0.0, 730.99).alternatives[0]) is None
    # frame 70 was captured from 700.98 to 701.0
    recent = 1.0 + 69 * frame.duration
    assert _stream_positions(stream, 30.0 + 700.98, 30.0 + 701.0) == pytest.approx(
        (recent, recent + frame.duration), abs=1e-4
    )
    await stream.aclose()


async def test_successful_recognition_resets_retry_budget() -> None:
    drops = DEFAULT_API_CONNECT_OPTIONS.max_retry * 3
    stream = _FlappingStream(stt=_DummySTT(), uptime=6.0, drops=drops, emit_transcript=True)
    try:
        await stream._task
        assert stream.runs == drops + 1
    finally:
        await stream.aclose()


async def test_recovery_budget_is_finite_after_previously_useful_connections() -> None:
    useful_runs = DEFAULT_API_CONNECT_OPTIONS.max_retry * 2

    class InitiallyUsefulStream(_FlappingStream):
        async def _run(self) -> None:
            self._emit_transcript = self.runs < useful_runs
            await super()._run()

    stream = InitiallyUsefulStream(stt=_DummySTT(), uptime=6.0, drops=100, emit_transcript=True)
    try:
        with pytest.raises(APIConnectionError):
            await stream._task
        assert stream.runs == useful_runs + DEFAULT_API_CONNECT_OPTIONS.max_retry
    finally:
        await stream.aclose()


async def test_provider_progress_without_a_transcript_resets_retry_budget() -> None:
    class ProgressStream(_FlappingStream):
        async def _run(self) -> None:
            self._mark_provider_progress()
            await super()._run()

    drops = DEFAULT_API_CONNECT_OPTIONS.max_retry * 3
    stream = ProgressStream(stt=_DummySTT(), uptime=6.0, drops=drops, emit_transcript=False)
    try:
        await stream._task
        assert stream.runs == drops + 1
    finally:
        await stream.aclose()


@pytest.mark.parametrize(
    "event_type",
    [
        SpeechEventType.START_OF_SPEECH,
        SpeechEventType.END_OF_SPEECH,
        SpeechEventType.RECOGNITION_USAGE,
    ],
)
async def test_local_events_do_not_reset_retry_budget(event_type: SpeechEventType) -> None:
    stream = _FlappingStream(stt=_DummySTT(), uptime=6.0, drops=100, emit_transcript=False)

    async def events() -> AsyncIterator[SpeechEvent]:
        yield SpeechEvent(type=event_type, recognition_usage=RecognitionUsage(audio_duration=1))

    try:
        stream._num_retries = 2
        await stream._metrics_monitor_task(events())
        assert stream._num_retries == 2
    finally:
        await stream.aclose()
