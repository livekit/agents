from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

import pytest

from livekit import rtc
from livekit.agents import APIConnectionError, APIError, LanguageCode, utils
from livekit.agents.stt import (
    STT,
    AvailabilityChangedEvent,
    FallbackAdapter,
    RecognizeStream,
    SpeechData,
    SpeechEvent,
    SpeechEventType,
    STTCapabilities,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from livekit.agents.utils.aio.channel import ChanEmpty
from livekit.agents.utils.audio import AudioBuffer

from .fake_stt import FakeSTT
from .fake_vad import FakeVAD

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]


class FallbackAdapterTester(FallbackAdapter):
    def __init__(
        self,
        stt: list[STT],
        *,
        attempt_timeout: float = 10.0,
        max_retry_per_stt: int = 1,
        retry_interval: float = 5,
    ) -> None:
        super().__init__(
            stt,
            attempt_timeout=attempt_timeout,
            max_retry_per_stt=max_retry_per_stt,
            retry_interval=retry_interval,
        )

        self.on("stt_availability_changed", self._on_stt_availability_changed)

        self._availability_changed_ch: dict[int, utils.aio.Chan[AvailabilityChangedEvent]] = {
            id(t): utils.aio.Chan[AvailabilityChangedEvent]() for t in stt
        }

    def _on_stt_availability_changed(self, ev: AvailabilityChangedEvent) -> None:
        self._availability_changed_ch[id(ev.stt)].send_nowait(ev)

    def availability_changed_ch(
        self,
        stt: STT,
    ) -> utils.aio.ChanReceiver[AvailabilityChangedEvent]:
        return self._availability_changed_ch[id(stt)]


class _NamedSTT(FakeSTT):
    """FakeSTT with a configurable model/provider so tests can tell instances apart."""

    def __init__(self, *, model: str, provider: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._model_name = model
        self._provider_name = provider

    @property
    def model(self) -> str:
        return self._model_name

    @property
    def provider(self) -> str:
        return self._provider_name


class _NonStreamingSTT(FakeSTT):
    def __init__(self) -> None:
        super().__init__()
        self._capabilities = STTCapabilities(streaming=False, interim_results=False)
        self.close_count = 0

    async def aclose(self) -> None:
        self.close_count += 1


def _metrics_listener_count(stt: STT) -> int:
    return len(stt._events.get("metrics_collected", set()))


async def test_aclose_closes_automatically_created_stream_adapters() -> None:
    stt = _NonStreamingSTT()
    baseline = _metrics_listener_count(stt)
    fallback = FallbackAdapter([stt], vad=FakeVAD())

    assert _metrics_listener_count(stt) == baseline + 1

    await fallback.aclose()

    assert _metrics_listener_count(stt) == baseline
    assert stt.close_count == 0


async def test_reports_active_instance_model_and_provider() -> None:
    fake1 = _NamedSTT(
        model="primary-model",
        provider="primary",
        fake_exception=APIConnectionError("fake1 failed"),
        fake_timeout=0.5,
    )
    fake2 = _NamedSTT(model="fallback-model", provider="fallback", fake_transcript="hello world")

    fallback_adapter = FallbackAdapterTester([fake1, fake2])

    # before any traffic, the primary is reported
    assert fallback_adapter.metrics_metadata == {
        "model_name": "primary-model",
        "model_provider": "primary",
    }

    await fallback_adapter.recognize([])

    # the fallback served the request, so metrics must be labeled with it
    assert fallback_adapter.metrics_metadata == {
        "model_name": "fallback-model",
        "model_provider": "fallback",
    }
    # once the primary recovers (its recovery task flips it back to available) the next
    # request goes to it first, so that is what model and provider report
    fallback_adapter._status[0].available = True
    assert fallback_adapter.model == "primary-model"
    assert fallback_adapter.provider == "primary"
    fallback_adapter._status[0].available = False
    assert fallback_adapter.model == "fallback-model"

    assert not fallback_adapter.availability_changed_ch(fake1).recv_nowait().available

    # a successful recovery probe must not relabel: its result is never surfaced
    fake1.update_options(fake_exception=None, fake_transcript="probe")
    assert (
        await asyncio.wait_for(fallback_adapter.availability_changed_ch(fake1).recv(), 1.0)
    ).available, "fake1 should have recovered"

    assert fallback_adapter.metrics_metadata == {
        "model_name": "fallback-model",
        "model_provider": "fallback",
    }

    # once the recovered primary serves real traffic again, the label follows
    await fallback_adapter.recognize([])

    assert fallback_adapter.metrics_metadata == {
        "model_name": "primary-model",
        "model_provider": "primary",
    }

    await fallback_adapter.aclose()


async def test_stream_reports_active_instance_model_and_provider() -> None:
    fake1 = _NamedSTT(
        model="primary-model",
        provider="primary",
        fake_exception=APIConnectionError("fake1 failed"),
    )
    fake2 = _NamedSTT(model="fallback-model", provider="fallback", fake_transcript="hello world")

    fallback_adapter = FallbackAdapterTester([fake1, fake2])

    async with fallback_adapter.stream() as stream:
        stream.end_input()

        async for _ in stream:
            pass

    assert fallback_adapter.metrics_metadata == {
        "model_name": "fallback-model",
        "model_provider": "fallback",
    }

    await fallback_adapter.aclose()


async def test_stt_fallback() -> None:
    fake1 = FakeSTT(fake_exception=APIConnectionError("fake1 failed"))
    fake2 = FakeSTT(fake_transcript="hello world")

    fallback_adapter = FallbackAdapterTester([fake1, fake2])
    ev = await fallback_adapter.recognize([])
    assert ev.alternatives[0].text == "hello world"

    assert fake1.recognize_ch.recv_nowait()
    assert fake2.recognize_ch.recv_nowait()

    assert not fallback_adapter.availability_changed_ch(fake1).recv_nowait().available

    fake2.update_options(fake_exception=APIConnectionError("fake2 failed"))

    with pytest.raises(APIConnectionError):
        await fallback_adapter.recognize([])

    assert not fallback_adapter.availability_changed_ch(fake2).recv_nowait().available

    await fallback_adapter.aclose()

    # stream
    fake1 = FakeSTT(fake_exception=APIConnectionError("fake1 failed"))
    fake2 = FakeSTT(fake_transcript="hello world")

    fallback_adapter = FallbackAdapterTester([fake1, fake2])

    async with fallback_adapter.stream() as stream:
        stream.end_input()

        last_alt = ""

        async for ev in stream:
            last_alt = ev.alternatives[0].text

        assert last_alt == "hello world"

    await fallback_adapter.aclose()


async def test_stt_stream_fallback_propagates_start_time_offset() -> None:
    # A mid-stream fallback must anchor each leg's timestamps to the original input
    # timeline by seeding start_time_offset; otherwise a leg created after the switch
    # emits timestamps relative to the switch moment, placing post-switch transcripts
    # far in the past for consumers that anchor them to the input start.
    fake1 = FakeSTT(fake_exception=APIConnectionError("fake1 failed"))
    fake2 = FakeSTT(fake_transcript="hello world")

    fallback_adapter = FallbackAdapterTester([fake1, fake2])

    stream = fallback_adapter.stream()
    # simulate that audio input started 30s before this stream was created
    stream.start_time_offset = 30.0

    async with stream:
        stream.end_input()
        async for _ in stream:
            pass

    leg1 = fake1.stream_ch.recv_nowait()
    leg2 = fake2.stream_ch.recv_nowait()
    assert leg1.start_time_offset >= 30.0
    assert leg2.start_time_offset >= 30.0

    await fallback_adapter.aclose()


async def test_stt_stream_fallback() -> None:
    fake1 = FakeSTT(fake_exception=APIConnectionError("fake1 failed"))
    fake2 = FakeSTT(fake_transcript="hello world")

    fallback_adapter = FallbackAdapterTester([fake1, fake2])

    async with fallback_adapter.stream() as stream:
        stream.end_input()

        async for _ in stream:
            pass

        assert fake1.stream_ch.recv_nowait()
        assert fake2.stream_ch.recv_nowait()

    assert not fallback_adapter.availability_changed_ch(fake1).recv_nowait().available

    await fallback_adapter.aclose()


async def test_stt_recover() -> None:
    fake1 = FakeSTT(fake_exception=APIConnectionError("fake1 failed"))
    fake2 = FakeSTT(fake_exception=APIConnectionError("fake2 failed"), fake_timeout=0.5)

    fallback_adapter = FallbackAdapterTester([fake1, fake2])

    with pytest.raises(APIConnectionError):
        await fallback_adapter.recognize([])

    fake2.update_options(fake_exception=None, fake_transcript="hello world")

    assert not fallback_adapter.availability_changed_ch(fake1).recv_nowait().available
    assert not fallback_adapter.availability_changed_ch(fake2).recv_nowait().available

    assert (
        await asyncio.wait_for(fallback_adapter.availability_changed_ch(fake2).recv(), 1.0)
    ).available, "fake2 should have recovered"

    await fallback_adapter.recognize([])

    assert fake1.recognize_ch.recv_nowait()
    assert fake2.recognize_ch.recv_nowait()

    with pytest.raises(ChanEmpty):
        fallback_adapter.availability_changed_ch(fake1).recv_nowait()

    with pytest.raises(ChanEmpty):
        fallback_adapter.availability_changed_ch(fake2).recv_nowait()

    await fallback_adapter.aclose()


class _ImmediateFailStream(RecognizeStream):
    """Stream whose _run raises APIConnectionError immediately, triggering fallback."""

    async def _run(self) -> None:
        raise APIConnectionError("immediate fail")


class _BrokenPushStream(RecognizeStream):
    """Stream that raises RuntimeError on push_frame/flush (simulates a closed/broken
    recovering stream). _run blocks forever so it stays in _recovering_streams."""

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        raise RuntimeError("broken recovering stream")

    def flush(self) -> None:
        raise RuntimeError("broken recovering stream")

    async def _run(self) -> None:
        await asyncio.Future()  # block forever


class _RecoveringFailSTT(STT):
    """First stream() call returns _ImmediateFailStream (triggers fallback).
    Subsequent calls return _BrokenPushStream (simulates broken recovery stream)."""

    def __init__(self) -> None:
        super().__init__(capabilities=STTCapabilities(streaming=True, interim_results=False))
        self._call_count = 0

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: str | None,
        conn_options: APIConnectOptions,
    ) -> SpeechEvent:
        raise APIConnectionError("not implemented")

    def stream(
        self,
        *,
        language: str | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> RecognizeStream:
        self._call_count += 1
        if self._call_count == 1:
            return _ImmediateFailStream(stt=self, conn_options=conn_options)
        return _BrokenPushStream(stt=self, conn_options=conn_options)


async def test_stt_stream_recovery_failure_doesnt_block_main() -> None:
    """Regression test: RuntimeError from a broken recovering stream must not
    prevent audio data from being forwarded to the main (fallback) stream.

    With the old code, a single try/except around both recovering and main stream
    forwarding meant a RuntimeError from a recovering stream's push_frame() would
    skip the main stream's push_frame(), starving it of audio data.
    """
    fallback = FallbackAdapterTester(
        [_RecoveringFailSTT(), FakeSTT(fake_transcript="hello world", fake_require_audio=True)],
        max_retry_per_stt=0,
    )

    audio_frame = rtc.AudioFrame(
        data=b"\x00\x00" * 480,
        sample_rate=48000,
        num_channels=1,
        samples_per_channel=480,
    )

    async with fallback.stream() as stream:
        # push audio after a brief delay so the fallback adapter has time to
        # fail over from the first STT to the second STT
        async def _push_delayed() -> None:
            await asyncio.sleep(0.2)
            stream.push_frame(audio_frame)
            stream.end_input()

        push_task = asyncio.create_task(_push_delayed())

        events: list[SpeechEvent] = []
        async for ev in stream:
            events.append(ev)

        await push_task

    assert len(events) == 1, f"expected 1 event, got {len(events)}"
    assert events[0].alternatives[0].text == "hello world"

    await fallback.aclose()


class _ControlledSTT(STT):
    def __init__(self, name: str) -> None:
        super().__init__(capabilities=STTCapabilities(streaming=True, interim_results=True))
        self._label = name
        self.streams: list[_ControlledStream] = []
        self.fail_stream_construction = 0
        self.stream_hook: Callable[[], None] | None = None

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: str | None,
        conn_options: APIConnectOptions,
    ) -> SpeechEvent:
        raise APIConnectionError("not used")

    def stream(
        self,
        *,
        language: str | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> RecognizeStream:
        if self.stream_hook is not None:
            self.stream_hook()
        if self.fail_stream_construction:
            self.fail_stream_construction -= 1
            raise RuntimeError("secret provider response")
        stream = _ControlledStream(stt=self, conn_options=conn_options)
        self.streams.append(stream)
        return stream


class _ControlledStream(RecognizeStream):
    def __init__(self, *, stt: STT, conn_options: APIConnectOptions) -> None:
        super().__init__(stt=stt, conn_options=conn_options)
        self.started = asyncio.Event()
        self.completion: asyncio.Future[None] = asyncio.Future()

    def emit_text(self, text: str) -> None:
        if not self._event_ch.closed:
            self._event_ch.send_nowait(
                SpeechEvent(
                    type=SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=[SpeechData(text=text, language=LanguageCode("en"))],
                )
            )

    def fail(self, error: Exception | None = None) -> None:
        if not self.completion.done():
            self.completion.set_exception(error or APIError("provider connection ended"))

    def finish(self) -> None:
        if not self.completion.done():
            self.completion.set_result(None)

    async def _run(self) -> None:
        self.started.set()
        await self.completion


class _CancellationResistantStream(RecognizeStream):
    def __init__(self, *, stt: STT, conn_options: APIConnectOptions) -> None:
        super().__init__(stt=stt, conn_options=conn_options)
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.terminal_error: Exception | None = None

    def emit_text(self, text: str) -> None:
        if not self._event_ch.closed:
            self._event_ch.send_nowait(
                SpeechEvent(
                    type=SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=[SpeechData(text=text, language=LanguageCode("en"))],
                )
            )

    async def _run(self) -> None:
        self.started.set()
        while not self.release.is_set():
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                pass
        if self.terminal_error is not None:
            raise self.terminal_error


class _CancellationResistantSTT(STT):
    def __init__(self, name: str, *, fail_first: bool = False) -> None:
        super().__init__(capabilities=STTCapabilities(streaming=True, interim_results=False))
        self._label = name
        self.fail_first = fail_first
        self.streams: list[RecognizeStream] = []

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: str | None,
        conn_options: APIConnectOptions,
    ) -> SpeechEvent:
        raise APIConnectionError("not used")

    def stream(
        self,
        *,
        language: str | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> RecognizeStream:
        if self.fail_first and not self.streams:
            stream: RecognizeStream = _ImmediateFailStream(stt=self, conn_options=conn_options)
        else:
            stream = _CancellationResistantStream(stt=self, conn_options=conn_options)
        self.streams.append(stream)
        return stream


async def _get_controlled_stream(provider: _ControlledSTT, index: int = 0) -> _ControlledStream:
    async def _wait() -> _ControlledStream:
        while len(provider.streams) <= index:
            await asyncio.sleep(0)
        stream = provider.streams[index]
        await stream.started.wait()
        return stream

    return await asyncio.wait_for(_wait(), 1)


async def _wait_for(predicate: Callable[[], bool]) -> None:
    async def _wait() -> None:
        while not predicate():
            await asyncio.sleep(0)

    await asyncio.wait_for(_wait(), 1)


class _RetryTimelineSTT(STT):
    def __init__(self) -> None:
        super().__init__(capabilities=STTCapabilities(streaming=True, interim_results=False))
        self.main_streams: list[_RetryTimelineStream] = []

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: str | None,
        conn_options: APIConnectOptions,
    ) -> SpeechEvent:
        raise APIConnectionError("not used")

    def stream(
        self,
        *,
        language: str | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> RecognizeStream:
        is_probe = conn_options.max_retry == 0
        attempt = 0 if is_probe else len(self.main_streams) + 1
        stream = _RetryTimelineStream(stt=self, conn_options=conn_options, attempt=attempt)
        if not is_probe:
            self.main_streams.append(stream)
        return stream


class _RetryTimelineStream(RecognizeStream):
    def __init__(self, *, stt: STT, conn_options: APIConnectOptions, attempt: int) -> None:
        super().__init__(stt=stt, conn_options=conn_options)
        self.attempt = attempt
        self.run_offsets: list[float] = []

    async def _run(self) -> None:
        self.run_offsets.append(self.start_time_offset)
        if self.attempt == 0:
            return
        if self.attempt == 1:
            await asyncio.sleep(0.025)
            raise APIConnectionError("first adapter attempt failed", retryable=False)
        self._event_ch.send_nowait(
            SpeechEvent(
                type=SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[SpeechData(text="recovered", language=LanguageCode("en"))],
            )
        )
        async for _ in self._input_ch:
            pass


def _audio_frame() -> rtc.AudioFrame:
    return rtc.AudioFrame(
        data=b"\x00\x00" * 160,
        sample_rate=16000,
        num_channels=1,
        samples_per_channel=160,
    )


async def test_preserves_timeline_across_outer_stream_retries() -> None:
    stt = _RetryTimelineSTT()
    adapter = FallbackAdapterTester([stt], max_retry_per_stt=1, retry_interval=0)
    stream = adapter.stream(
        conn_options=APIConnectOptions(max_retry=1, timeout=10, retry_interval=0)
    )
    stream.start_time_offset = 30
    stream.end_input()

    events = [event async for event in stream]

    assert [event.alternatives[0].text for event in events] == ["recovered"]
    assert len(stt.main_streams) == 2
    assert stt.main_streams[0].start_time_offset >= 30
    assert stt.main_streams[1].start_time_offset > stt.main_streams[0].start_time_offset
    await stream.aclose()
    await adapter.aclose()


async def test_close_before_run_does_not_open_provider() -> None:
    primary = _ControlledSTT("primary")
    adapter = FallbackAdapterTester([primary], max_retry_per_stt=0)
    stream = adapter.stream()

    await stream.aclose()
    await asyncio.sleep(0)

    assert not primary.streams
    assert adapter._status[0].available
    await adapter.aclose()


@pytest.mark.parametrize("event", ["transcript", "error"])
async def test_parent_close_discards_late_child_completion(event: str) -> None:
    primary = _ControlledSTT("primary")
    secondary = _ControlledSTT("secondary")
    adapter = FallbackAdapterTester([primary, secondary], max_retry_per_stt=0)
    stream = adapter.stream()
    child = await _get_controlled_stream(primary)

    close_task = asyncio.create_task(stream.aclose())
    await asyncio.sleep(0)
    if event == "transcript":
        child.emit_text("late transcript")
    else:
        child.fail(APIError("late provider error"))
    await close_task

    assert adapter._status[0].available
    assert not secondary.streams
    await adapter.aclose()


async def test_availability_listener_close_prevents_recovery_and_fallback() -> None:
    primary = _ControlledSTT("primary")
    secondary = _ControlledSTT("secondary")
    adapter = FallbackAdapterTester([primary, secondary], max_retry_per_stt=0)
    stream = adapter.stream()
    child = await _get_controlled_stream(primary)
    close_tasks: list[asyncio.Task[None]] = []

    adapter.on(
        "stt_availability_changed",
        lambda _: close_tasks.append(asyncio.create_task(stream.aclose())),
    )
    child.fail()
    await _wait_for(lambda: bool(close_tasks))
    await close_tasks[0]

    assert len(primary.streams) == 1
    assert not secondary.streams
    assert adapter._status[0].recovering_stream_task is None
    await adapter.aclose()


async def test_construction_close_request_prevents_fallback() -> None:
    primary = _ControlledSTT("primary")
    secondary = _ControlledSTT("secondary")
    adapter = FallbackAdapterTester([primary, secondary], max_retry_per_stt=0)
    stream = adapter.stream()
    close_tasks: list[asyncio.Task[None]] = []

    def _close_and_fail() -> None:
        close_tasks.append(asyncio.create_task(stream.aclose()))
        raise APIError("provider closed during setup")

    primary.stream_hook = _close_and_fail
    await _wait_for(lambda: bool(close_tasks))
    await close_tasks[0]

    assert not secondary.streams
    assert adapter._status[0].available
    await adapter.aclose()


async def test_parent_close_settles_child_and_discards_in_flight_output() -> None:
    primary = _ControlledSTT("primary")
    secondary = _ControlledSTT("secondary")
    adapter = FallbackAdapterTester([primary, secondary], max_retry_per_stt=0)
    stream = adapter.stream()
    child = await _get_controlled_stream(primary)

    child.emit_text("late transcript")
    await stream.aclose()

    assert child._task.done()
    assert stream._task.done()
    assert adapter._status[0].available
    assert not secondary.streams
    await adapter.aclose()


async def test_premature_child_eof_falls_back_but_post_input_eof_is_clean() -> None:
    primary = _ControlledSTT("primary")
    secondary = _ControlledSTT("secondary")
    adapter = FallbackAdapterTester([primary, secondary], max_retry_per_stt=0)
    stream = adapter.stream()
    (await _get_controlled_stream(primary)).finish()
    fallback = await _get_controlled_stream(secondary)
    fallback.emit_text("continued transcription")
    assert (await anext(stream)).alternatives[0].text == "continued transcription"
    await stream.aclose()
    assert not adapter.availability_changed_ch(primary).recv_nowait().available
    await adapter.aclose()

    primary = _ControlledSTT("primary")
    secondary = _ControlledSTT("secondary")
    adapter = FallbackAdapterTester([primary, secondary], max_retry_per_stt=0)
    stream = adapter.stream()
    child = await _get_controlled_stream(primary)
    stream.end_input()
    child.finish()
    with pytest.raises(StopAsyncIteration):
        await anext(stream)
    assert not secondary.streams
    assert adapter._status[0].available
    await stream.aclose()
    await adapter.aclose()


async def test_concurrent_stream_failure_does_not_interrupt_healthy_stream() -> None:
    primary = _ControlledSTT("primary")
    secondary = _ControlledSTT("secondary")
    adapter = FallbackAdapterTester([primary, secondary], max_retry_per_stt=0)
    failing = adapter.stream()
    failing_child = await _get_controlled_stream(primary)
    healthy = adapter.stream()
    healthy_child = await _get_controlled_stream(primary, 1)

    failing_child.fail()
    await _get_controlled_stream(secondary)
    healthy.end_input()
    healthy_child.finish()

    with pytest.raises(StopAsyncIteration):
        await anext(healthy)
    assert len(secondary.streams) == 1
    await failing.aclose()
    await healthy.aclose()
    await adapter.aclose()


async def test_clean_sibling_eof_preserves_active_probe() -> None:
    primary = _ControlledSTT("primary")
    secondary = _ControlledSTT("secondary")
    adapter = FallbackAdapterTester([primary, secondary], max_retry_per_stt=0)
    owner = adapter.stream()
    (await _get_controlled_stream(primary)).fail()
    probe = await _get_controlled_stream(primary, 1)
    await _get_controlled_stream(secondary)
    sibling = adapter.stream()
    sibling_child = await _get_controlled_stream(secondary, 1)

    sibling.end_input()
    sibling_child.finish()
    with pytest.raises(StopAsyncIteration):
        await anext(sibling)
    probe.emit_text("primary recovered")
    await _wait_for(lambda: adapter._status[0].available)
    await _wait_for(lambda: adapter._status[0].recovering_stream_task is None)

    assert adapter._status[0].recovering_stream_task is None
    await owner.aclose()
    await sibling.aclose()
    await adapter.aclose()


async def test_owner_close_without_waiter_settles_probe_and_fallback() -> None:
    primary = _ControlledSTT("primary")
    secondary = _ControlledSTT("secondary")
    adapter = FallbackAdapterTester([primary, secondary], max_retry_per_stt=0)
    owner = adapter.stream()
    (await _get_controlled_stream(primary)).fail()
    probe = await _get_controlled_stream(primary, 1)
    fallback = await _get_controlled_stream(secondary)
    recovery_task = adapter._status[0].recovering_stream_task

    await owner.aclose()
    await _wait_for(lambda: adapter._status[0].recovering_stream_task is None)

    assert recovery_task is not None and recovery_task.done()
    assert probe._task.done()
    assert fallback._task.done()
    await adapter.aclose()


async def test_parent_close_is_bounded_for_cancellation_resistant_probe_and_child(
    caplog: pytest.LogCaptureFixture,
) -> None:
    primary = _CancellationResistantSTT("primary", fail_first=True)
    secondary = _CancellationResistantSTT("secondary")
    adapter = FallbackAdapterTester([primary, secondary], max_retry_per_stt=0)
    stream = adapter.stream()
    await _wait_for(lambda: len(primary.streams) == 2 and len(secondary.streams) == 1)
    probe = primary.streams[1]
    fallback = secondary.streams[0]
    assert isinstance(probe, _CancellationResistantStream)
    assert isinstance(fallback, _CancellationResistantStream)
    await probe.started.wait()
    await fallback.started.wait()
    caplog.clear()

    await asyncio.wait_for(stream.aclose(), 2)

    assert stream._task.done()
    assert not probe._task.done()
    assert not fallback._task.done()
    probe.emit_text("late recovery transcript")
    probe.terminal_error = RuntimeError("late probe failure")
    fallback.terminal_error = RuntimeError("late fallback failure")
    probe.release.set()
    fallback.release.set()
    await _wait_for(lambda: probe._task.done() and fallback._task.done())
    await _wait_for(lambda: not stream._close_tasks)

    assert not adapter._status[0].available
    assert not [record for record in caplog.records if record.levelno >= logging.WARNING]
    await adapter.aclose()


async def test_single_recovery_probe_transfers_to_live_waiter() -> None:
    primary = _ControlledSTT("primary")
    secondary = _ControlledSTT("secondary")
    adapter = FallbackAdapterTester([primary, secondary], max_retry_per_stt=0)
    owner = adapter.stream()
    (await _get_controlled_stream(primary)).fail()
    old_probe = await _get_controlled_stream(primary, 1)
    await _get_controlled_stream(secondary)
    replacement = adapter.stream()
    replacement_main = await _get_controlled_stream(secondary, 1)
    assert len(primary.streams) == 2

    replacement.push_frame(_audio_frame())
    await asyncio.sleep(0)
    assert old_probe._input_ch.empty()
    assert not replacement_main._input_ch.empty()
    await owner.aclose()

    probe = await _get_controlled_stream(primary, 2)
    replacement.push_frame(_audio_frame())
    await _wait_for(lambda: not probe._input_ch.empty())
    probe.emit_text("primary recovered")
    await _wait_for(lambda: adapter._status[0].available)

    assert old_probe._task.done()
    await replacement.aclose()
    await adapter.aclose()


async def test_recovery_transfer_skips_closed_and_failed_waiters() -> None:
    primary = _ControlledSTT("primary")
    secondary = _ControlledSTT("secondary")
    adapter = FallbackAdapterTester([primary, secondary], max_retry_per_stt=0)
    owner = adapter.stream()
    (await _get_controlled_stream(primary)).fail()
    await _get_controlled_stream(primary, 1)
    await _get_controlled_stream(secondary)
    closed_waiter = adapter.stream()
    await _get_controlled_stream(secondary, 1)
    failed_waiter = adapter.stream()
    await _get_controlled_stream(secondary, 2)
    replacement = adapter.stream()
    await _get_controlled_stream(secondary, 3)

    await closed_waiter.aclose()
    primary.fail_stream_construction = 1
    await owner.aclose()
    probe = await _get_controlled_stream(primary, 2)
    replacement.push_frame(_audio_frame())
    await _wait_for(lambda: not probe._input_ch.empty())

    await failed_waiter.aclose()
    await replacement.aclose()
    await adapter.aclose()


@pytest.mark.parametrize("action", ["close", "recover"])
async def test_no_waiting_probe_after_adapter_close_or_recovery(action: str) -> None:
    primary = _ControlledSTT("primary")
    secondary = _ControlledSTT("secondary")
    adapter = FallbackAdapterTester([primary, secondary], max_retry_per_stt=0)
    owner = adapter.stream()
    (await _get_controlled_stream(primary)).fail()
    probe = await _get_controlled_stream(primary, 1)
    await _get_controlled_stream(secondary)
    waiter = adapter.stream()
    await _get_controlled_stream(secondary, 1)

    if action == "close":
        await adapter.aclose()
    else:
        probe.emit_text("primary recovered")
        await _wait_for(lambda: adapter._status[0].available)
    await _wait_for(lambda: adapter._status[0].recovering_stream_task is None)

    assert len(primary.streams) == 2
    assert not adapter._status[0].waiting_streams
    await owner.aclose()
    await waiter.aclose()
    if action != "close":
        await adapter.aclose()


async def test_all_unavailable_retries_normally_without_duplicate_probe() -> None:
    primary = _ControlledSTT("primary")
    secondary = _ControlledSTT("secondary")
    adapter = FallbackAdapterTester([primary, secondary], max_retry_per_stt=0)
    for status in adapter._status:
        status.available = False

    stream = adapter.stream()
    child = await _get_controlled_stream(primary)
    assert all(status.recovering_stream_task is None for status in adapter._status)
    child.emit_text("direct retry succeeded")
    stream.end_input()
    child.finish()

    assert (await anext(stream)).alternatives[0].text == "direct retry succeeded"
    with pytest.raises(StopAsyncIteration):
        await anext(stream)
    assert [status.available for status in adapter._status] == [False, False]
    await stream.aclose()
    await adapter.aclose()


async def test_unavailable_secondary_is_normal_retry_while_probe_stays_silent() -> None:
    primary = _ControlledSTT("primary")
    secondary = _ControlledSTT("secondary")
    adapter = FallbackAdapterTester([primary, secondary], max_retry_per_stt=0)
    for status in adapter._status:
        status.available = False
    stream = adapter.stream()
    (await _get_controlled_stream(primary)).fail()
    probe = await _get_controlled_stream(primary, 1)
    fallback = await _get_controlled_stream(secondary)

    fallback.emit_text("secondary transcript")
    stream.end_input()
    fallback.finish()
    events = [event async for event in stream]

    assert [event.alternatives[0].text for event in events] == ["secondary transcript"]
    assert len(primary.streams) == 2
    assert probe._task.done()
    assert adapter._status[0].recovering_stream_task is None
    await stream.aclose()
    await adapter.aclose()


@pytest.mark.parametrize("error_type", [APIError, RuntimeError])
async def test_new_fallback_and_probe_receive_immediate_input_eof(
    error_type: type[Exception],
) -> None:
    primary = _ControlledSTT("primary")
    secondary = _ControlledSTT("secondary")
    adapter = FallbackAdapterTester([primary, secondary], max_retry_per_stt=0)
    stream = adapter.stream()
    child = await _get_controlled_stream(primary)
    stream.end_input()
    child.fail(error_type("terminal failure"))

    probe = await _get_controlled_stream(primary, 1)
    fallback = await _get_controlled_stream(secondary)
    await _wait_for(lambda: probe._input_ch.closed and fallback._input_ch.closed)
    fallback.emit_text("final fallback transcript")
    fallback.finish()

    assert (await anext(stream)).alternatives[0].text == "final fallback transcript"
    with pytest.raises(StopAsyncIteration):
        await anext(stream)
    await stream.aclose()
    await adapter.aclose()


async def test_all_unavailable_terminal_failure_settles_all_streams() -> None:
    primary = _ControlledSTT("primary")
    secondary = _ControlledSTT("secondary")
    adapter = FallbackAdapterTester([primary, secondary], max_retry_per_stt=0)
    for status in adapter._status:
        status.available = False
    stream = adapter.stream()
    (await _get_controlled_stream(primary)).fail()
    primary_probe = await _get_controlled_stream(primary, 1)
    (await _get_controlled_stream(secondary)).fail()
    secondary_probe = await _get_controlled_stream(secondary, 1)

    with pytest.raises(APIConnectionError, match="all STTs failed"):
        await anext(stream)
    await _wait_for(
        lambda: all(status.recovering_stream_task is None for status in adapter._status)
    )

    assert primary_probe._task.done()
    assert secondary_probe._task.done()
    assert all(child._task.done() for child in [*primary.streams, *secondary.streams])
    await stream.aclose()
    await adapter.aclose()


class _LateBatchSTT(_ControlledSTT):
    def __init__(self, name: str, results: list[asyncio.Future[SpeechEvent]]) -> None:
        super().__init__(name)
        self.results = results

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: str | None,
        conn_options: APIConnectOptions,
    ) -> SpeechEvent:
        result = self.results.pop(0)
        try:
            return await asyncio.shield(result)
        except asyncio.CancelledError:
            return await result


async def test_adapter_close_ignores_late_batch_recovery_completion(
    caplog: pytest.LogCaptureFixture,
) -> None:
    loop = asyncio.get_running_loop()
    failed: asyncio.Future[SpeechEvent] = loop.create_future()
    failed.set_exception(APIError("primary failed"))
    recovery: asyncio.Future[SpeechEvent] = loop.create_future()
    primary = _LateBatchSTT("primary", [failed, recovery])
    fallback = FakeSTT(fake_transcript="fallback")
    adapter = FallbackAdapterTester([primary, fallback], max_retry_per_stt=0)
    await adapter.recognize([])
    await _wait_for(lambda: adapter._status[0].recovering_recognize_task is not None)
    await _wait_for(lambda: not primary.results)
    recovery_task = adapter._status[0].recovering_recognize_task
    assert recovery_task is not None

    await asyncio.wait_for(adapter.aclose(), 2)
    assert not recovery_task.done()
    recovery.set_result(
        SpeechEvent(
            type=SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[SpeechData(text="late", language=LanguageCode("en"))],
        )
    )
    await recovery_task
    await asyncio.sleep(0)

    assert not adapter._status[0].available
    assert adapter._status[0].recovering_recognize_task is None
    assert not [record for record in caplog.records if record.message == "STT recovery failed"]


def _assert_safe_log(record: logging.LogRecord, error_type: type[Exception]) -> None:
    assert record.stt == "primary"
    assert record.streamed is True
    assert record.error_type == error_type.__name__
    assert "secret provider response" not in record.getMessage()
    assert "secret credentials" not in record.getMessage()
    assert record.exc_info is None


@pytest.mark.parametrize("error_type", [APIError, RuntimeError])
async def test_main_stream_exception_logs_only_safe_metadata(
    caplog: pytest.LogCaptureFixture, error_type: type[Exception]
) -> None:
    caplog.set_level(logging.DEBUG, logger="livekit.agents")
    primary = _ControlledSTT("primary")
    secondary = _ControlledSTT("secondary")
    adapter = FallbackAdapterTester([primary, secondary], max_retry_per_stt=0)
    error = error_type("secret provider response")
    error.__cause__ = RuntimeError("secret credentials")

    def _raise_error() -> None:
        raise error

    primary.stream_hook = _raise_error
    stream = adapter.stream()
    await _get_controlled_stream(secondary)

    records = [
        record
        for record in caplog.records
        if record.message == "STT failed, switching to next provider"
    ]
    assert len(records) == 1
    _assert_safe_log(records[0], error_type)

    await stream.aclose()
    await adapter.aclose()


@pytest.mark.parametrize("error_type", [APIError, RuntimeError])
async def test_stream_recovery_exception_logs_once_with_safe_metadata(
    caplog: pytest.LogCaptureFixture, error_type: type[Exception]
) -> None:
    caplog.set_level(logging.DEBUG, logger="livekit.agents")
    primary = _ControlledSTT("primary")
    secondary = _ControlledSTT("secondary")
    adapter = FallbackAdapterTester([primary, secondary], max_retry_per_stt=0)
    stream = adapter.stream()
    (await _get_controlled_stream(primary)).fail()
    probe = await _get_controlled_stream(primary, 1)
    await _get_controlled_stream(secondary)
    caplog.clear()
    error = error_type("secret provider response")
    error.__cause__ = RuntimeError("secret credentials")
    probe.fail(error)
    await _wait_for(lambda: adapter._status[0].recovering_stream_task is None)

    records = [record for record in caplog.records if record.message == "STT recovery failed"]
    assert len(records) == 1
    _assert_safe_log(records[0], error_type)

    await stream.aclose()
    await adapter.aclose()


@pytest.mark.parametrize("error_type", [APIError, RuntimeError])
async def test_batch_recovery_exception_logs_once_with_safe_metadata(
    caplog: pytest.LogCaptureFixture, error_type: type[Exception]
) -> None:
    caplog.set_level(logging.DEBUG, logger="livekit.agents")
    loop = asyncio.get_running_loop()
    failed: asyncio.Future[SpeechEvent] = loop.create_future()
    failed.set_exception(APIError("primary failed"))
    recovery: asyncio.Future[SpeechEvent] = loop.create_future()
    primary = _LateBatchSTT("primary", [failed, recovery])
    fallback = FakeSTT(fake_transcript="fallback")
    adapter = FallbackAdapterTester([primary, fallback], max_retry_per_stt=0)
    await adapter.recognize([])
    await _wait_for(lambda: adapter._status[0].recovering_recognize_task is not None)
    caplog.clear()
    error = error_type("secret provider response")
    error.__cause__ = RuntimeError("secret credentials")
    recovery.set_exception(error)
    await _wait_for(lambda: adapter._status[0].recovering_recognize_task is None)

    records = [record for record in caplog.records if record.message == "STT recovery failed"]
    assert len(records) == 1
    assert records[0].stt == "primary"
    assert records[0].streamed is False
    assert records[0].error_type == error_type.__name__
    assert "secret provider response" not in records[0].getMessage()
    assert records[0].exc_info is None
    await adapter.aclose()
