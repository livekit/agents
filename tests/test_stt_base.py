"""Unit tests for base STT `RecognizeStream` fields (start_time, etc.)."""

from __future__ import annotations

import asyncio
import dataclasses
import time

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
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
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
    """Every connection succeeds, then drops. A real reconnect, every time."""

    def __init__(
        self,
        *,
        stt: STT,
        drops: int,
        emit: SpeechEvent | None,
        silent_runs: int = 0,
        yield_after_emit: bool = True,
    ) -> None:
        super().__init__(
            stt=stt,
            conn_options=dataclasses.replace(DEFAULT_API_CONNECT_OPTIONS, retry_interval=0.0),
        )
        self.runs = 0
        self._drops = drops
        self._emit = emit
        self._silent_runs = silent_runs
        self._yield_after_emit = yield_after_emit

    async def _run(self) -> None:
        self.runs += 1
        if self._emit is not None and self.runs > self._silent_runs:
            self._event_ch.send_nowait(self._emit)
            if self._yield_after_emit:
                await asyncio.sleep(0)
        if self.runs <= self._drops:
            raise APIConnectionError("socket dropped")
        await asyncio.sleep(3600)  # healthy at last


async def _survives(
    emit: SpeechEvent | None,
    *,
    drops: int = 10,
    silent_runs: int = 0,
    yield_after_emit: bool = True,
) -> bool:
    stream = _FlappingStream(
        stt=_DummySTT(),
        drops=drops,
        emit=emit,
        silent_runs=silent_runs,
        yield_after_emit=yield_after_emit,
    )
    try:
        for _ in range(500):
            if stream._task.done() or stream.runs > drops:
                break
            await asyncio.sleep(0.01)
        return not stream._task.done()
    finally:
        if not stream._task.done():
            await stream.aclose()


async def test_retry_budget_survives_drops_while_the_caller_is_silent() -> None:
    """The budget counts consecutive failures, not the lifetime of the stream.

    Every reconnect here succeeds, so the stream is healthy throughout. Resetting
    only on FINAL_TRANSCRIPT tied the budget to the caller speaking: an agent
    talking over a silent caller, or a caller on hold, never earned it back, and
    max_retry drops spread over one long call killed the stream for good.
    """
    usage_only = SpeechEvent(
        type=SpeechEventType.RECOGNITION_USAGE,
        recognition_usage=RecognitionUsage(audio_duration=5.0),
    )
    assert await _survives(usage_only)


async def test_retry_budget_still_gives_up_when_nothing_is_ever_delivered() -> None:
    """A connection that has never delivered anything must not retry forever."""
    assert not await _survives(None)


async def test_retry_budget_resets_even_when_the_producer_never_yields() -> None:
    """The reset must land in the producer's turn, not the consumer's.

    `send_nowait` wakes the metrics consumer but does not run it. An attempt that
    publishes an event and then raises in the same event-loop turn would otherwise
    reach the terminal branch of `_main_task` with the budget still exhausted, and
    a plugin can emit its last usage event during teardown, just before the socket
    error propagates. Enters the event-delivering attempt already at max_retry.
    """
    usage_only = SpeechEvent(
        type=SpeechEventType.RECOGNITION_USAGE,
        recognition_usage=RecognitionUsage(audio_duration=5.0),
    )
    assert await _survives(
        usage_only,
        drops=12,
        silent_runs=DEFAULT_API_CONNECT_OPTIONS.max_retry,
        yield_after_emit=False,
    )
