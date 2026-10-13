"""Unit tests for base STT `RecognizeStream` fields (start_time, etc.)."""

from __future__ import annotations

import asyncio
import dataclasses
import time

import pytest

from livekit import rtc
from livekit.agents import APIConnectionError, APIStatusError
from livekit.agents.stt import (
    STT,
    RecognizeStream,
    SpeechData,
    SpeechEvent,
    SpeechEventType,
    StreamAdapter,
    STTCapabilities,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, APIConnectOptions
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
    def __init__(self) -> None:
        super().__init__(capabilities=STTCapabilities(streaming=True, interim_results=False))

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

    def __init__(self, *, stt: STT, uptime: float, drops: int) -> None:
        super().__init__(
            stt=stt,
            conn_options=dataclasses.replace(DEFAULT_API_CONNECT_OPTIONS, retry_interval=0.0),
        )
        self.runs = 0
        self._uptime = uptime
        self._drops = drops

    async def _run(self) -> None:
        self.runs += 1
        await asyncio.sleep(self._uptime)
        if self.runs <= self._drops:
            raise APIConnectionError("socket dropped")


async def test_retry_budget_resets_after_an_attempt_that_outlived_the_connect_timeout() -> None:
    # a caller who never speaks produces no FINAL_TRANSCRIPT, so this is the only reset that
    # keeps an idle socket (Cartesia closes it every ~3 minutes) from exhausting the budget
    drops = DEFAULT_API_CONNECT_OPTIONS.max_retry * 3
    stream = _FlappingStream(stt=_DummySTT(), uptime=180.0, drops=drops)

    await stream._task

    assert stream.runs == drops + 1
    await stream.aclose()


async def test_retry_budget_still_gives_up_on_consecutive_short_lived_attempts() -> None:
    stream = _FlappingStream(stt=_DummySTT(), uptime=1.0, drops=100)

    with pytest.raises(APIConnectionError):
        await stream._task

    assert stream.runs == DEFAULT_API_CONNECT_OPTIONS.max_retry + 1
    await stream.aclose()


class _AudioStream(RecognizeStream):
    """Records ``start_time_offset`` as each run receives its first frame, then returns."""

    def __init__(
        self,
        *,
        stt: STT,
        fail_first_run: bool = False,
        first_run_delay: float = 0.0,
        sample_rate: int | None = None,
    ) -> None:
        super().__init__(
            stt=stt,
            conn_options=DEFAULT_API_CONNECT_OPTIONS,
            sample_rate=sample_rate if sample_rate is not None else NOT_GIVEN,
        )
        self._fail_first_run = fail_first_run
        self._first_run_delay = first_run_delay
        self.run_count = 0
        self.offsets: list[float] = []

    async def _run(self) -> None:
        self.run_count += 1
        if self.run_count == 1:
            await asyncio.sleep(self._first_run_delay)
        async for data in self._input_ch:
            if isinstance(data, RecognizeStream._FlushSentinel):
                continue
            self.offsets.append(self.start_time_offset)
            if self._fail_first_run and self.run_count == 1:
                raise APIConnectionError("fake failure to trigger retry")
            return


async def _wait_for_offsets(stream: _AudioStream, count: int) -> None:
    while len(stream.offsets) < count:
        await asyncio.sleep(0.01)


async def test_start_time_offset_grows_by_the_delay_to_the_first_frame() -> None:
    """The provider counts audio from the first frame it receives, not from the stream's
    creation, so the offset the caller set must absorb the delay between the two."""
    stream = _AudioStream(stt=_DummySTT())
    stream.start_time_offset = 30.0
    frame = silence_frame(0.02, 16000)
    await asyncio.sleep(5.0)

    pushed_at = time.time()
    stream.push_frame(frame)
    await _wait_for_offsets(stream, 1)

    delay = pushed_at - frame.duration - stream._created_at
    assert stream.offsets == [pytest.approx(30.0 + delay)]
    await stream.aclose()


async def test_start_time_offset_set_after_the_first_frame_keeps_the_delay() -> None:
    stream = _AudioStream(stt=_DummySTT())
    frame = silence_frame(0.02, 16000)
    await asyncio.sleep(5.0)
    pushed_at = time.time()
    stream.push_frame(frame)
    await _wait_for_offsets(stream, 1)

    stream.start_time_offset = 30.0

    assert stream.start_time_offset == pytest.approx(
        30.0 + pushed_at - frame.duration - stream._created_at
    )
    await stream.aclose()


async def test_start_time_offset_reanchors_on_retry_at_the_next_frame() -> None:
    """A new connection restarts the provider's clock with the next frame pushed, so the
    offset is re-anchored on that frame rather than grown by the previous attempt's duration."""
    stream = _AudioStream(stt=_DummySTT(), fail_first_run=True)
    stream.start_time_offset = 30.0
    frame = silence_frame(0.02, 16000)

    await asyncio.sleep(5.0)
    stream.push_frame(frame)
    await _wait_for_offsets(stream, 1)
    # the retry backoff is longer than this, so the frame is queued for the next attempt
    await asyncio.sleep(1.0)
    pushed_at = time.time()
    stream.push_frame(frame)
    await _wait_for_offsets(stream, 2)

    assert stream.run_count == 2
    assert stream.offsets[1] == pytest.approx(
        30.0 + pushed_at - frame.duration - stream._created_at
    )
    await stream.aclose()


async def test_start_time_offset_reanchors_on_a_frame_queued_before_the_retry() -> None:
    """A frame queued while the first connection fails is the first audio the next
    connection sends, so its capture time anchors the offset, not a later push."""
    stream = _AudioStream(stt=_DummySTT(), fail_first_run=True, first_run_delay=2.0)
    stream.start_time_offset = 30.0
    frame = silence_frame(0.02, 16000)

    # both frames are queued before the first attempt consumes anything; it takes the
    # first one and fails, so the second is what the next connection starts with
    stream.push_frame(frame)
    await asyncio.sleep(1.0)
    queued_at = time.time()
    stream.push_frame(frame)
    await _wait_for_offsets(stream, 2)

    assert stream.run_count == 2
    assert stream.offsets[1] == pytest.approx(
        30.0 + queued_at - frame.duration - stream._created_at
    )
    await stream.aclose()


async def test_undeclared_start_time_offset_counts_from_the_first_audio() -> None:
    """With no offset set the timeline starts with the first audio, so a late first frame
    adds nothing, and a retry continues that timeline from the next frame consumed."""
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


class _BufferingResampler:
    """Fake resampler: holds the first input and releases it with the second."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        self._held: list[rtc.AudioFrame] = []

    def push(self, frame: rtc.AudioFrame) -> list[rtc.AudioFrame]:
        self._held.append(frame)
        if len(self._held) < 2:
            return []
        out, self._held = self._held, []
        return out

    def flush(self) -> list[rtc.AudioFrame]:
        out, self._held = self._held, []
        return out


async def test_resampled_frames_keep_the_capture_time_of_their_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A resampler can hold audio across pushes and release several frames at once. Each
    output is stamped with the audio it carries, including the gap between the pushes,
    so a retry that starts on the second output is anchored where that audio was captured."""
    monkeypatch.setattr(rtc, "AudioResampler", _BufferingResampler)
    stream = _AudioStream(stt=_DummySTT(), fail_first_run=True, sample_rate=8000)
    stream.start_time_offset = 30.0
    frame = silence_frame(0.02, 16000)

    await asyncio.sleep(5.0)
    first_captured_at = time.time() - frame.duration
    stream.push_frame(frame)  # held by the resampler, no output yet
    await asyncio.sleep(1.0)
    second_captured_at = time.time() - frame.duration
    stream.push_frame(frame)  # releases both; the first attempt takes one and fails
    await _wait_for_offsets(stream, 2)

    assert stream.offsets == [
        pytest.approx(30.0 + first_captured_at - stream._created_at),
        pytest.approx(30.0 + second_captured_at - stream._created_at),
    ]
    await stream.aclose()
