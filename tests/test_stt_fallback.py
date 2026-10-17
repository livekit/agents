from __future__ import annotations

import asyncio
from typing import Any

import pytest

from livekit import rtc
from livekit.agents import APIConnectionError, utils
from livekit.agents.metrics import STTMetrics
from livekit.agents.stt import (
    STT,
    AvailabilityChangedEvent,
    FallbackAdapter,
    RecognizeStream,
    SpeechEvent,
    SpeechEventType,
    STTCapabilities,
    STTError,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from livekit.agents.utils.aio.channel import ChanEmpty
from livekit.agents.utils.audio import AudioBuffer
from livekit.agents.vad import VAD, VADCapabilities, VADEvent, VADEventType, VADStream

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
    assert _metrics_listener_count(fallback) == 0


class _FlushVAD(VAD):
    """Treat explicitly flushed audio as one utterance, without wall-clock timing."""

    def __init__(self) -> None:
        super().__init__(capabilities=VADCapabilities(update_interval=0.1))

    def stream(self) -> VADStream:
        return _FlushVADStream(self)


class _FlushVADStream(VADStream):
    async def _main_task(self) -> None:
        frames: list[rtc.AudioFrame] = []
        async for item in self._input_ch:
            if isinstance(item, rtc.AudioFrame):
                frames.append(item)
            elif frames:
                self._event_ch.send_nowait(
                    VADEvent(
                        type=VADEventType.END_OF_SPEECH,
                        samples_index=0,
                        timestamp=0,
                        speech_duration=sum(frame.duration for frame in frames),
                        silence_duration=0,
                        frames=frames,
                    )
                )
                frames = []


class _RecordingBatchSTT(_NonStreamingSTT):
    def __init__(self, *, fail: bool = False) -> None:
        super().__init__()
        self._fake_exception = APIConnectionError("primary unavailable") if fail else None
        self._fake_transcript = "transcribed utterance"
        self.buffers: list[bytes] = []

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: str | None,
        conn_options: APIConnectOptions,
    ) -> SpeechEvent:
        self.buffers.append(utils.merge_frames(buffer).data.tobytes())
        return await super()._recognize_impl(buffer, language=language, conn_options=conn_options)


@pytest.mark.parametrize("fail_primary", [False, True])
@pytest.mark.parametrize("end_input", [False, True])
async def test_batch_fallback_preserves_the_failed_utterance(
    fail_primary: bool, end_input: bool
) -> None:
    primary = _RecordingBatchSTT(fail=fail_primary)
    backup = _RecordingBatchSTT()
    fallback = FallbackAdapter([primary, backup], vad=_FlushVAD(), max_retry_per_stt=0)
    metrics: list[STTMetrics] = []
    fallback.on("metrics_collected", metrics.append)
    frames = [
        rtc.AudioFrame(
            data=bytes([value, 0]) * 160, sample_rate=16000, num_channels=1, samples_per_channel=160
        )
        for value in (1, 2)
    ]
    expected_audio = b"".join(frame.data.tobytes() for frame in frames)
    try:
        async with fallback.stream() as stream:
            for frame in frames:
                stream.push_frame(frame)
            stream.flush()
            if end_input:
                stream.end_input()

            async def final_transcript() -> SpeechEvent:
                async for event in stream:
                    if event.type == SpeechEventType.FINAL_TRANSCRIPT:
                        return event
                pytest.fail("no final transcript")

            event = await asyncio.wait_for(final_transcript(), timeout=1)
            assert event.alternatives[0].text == "transcribed utterance"
            assert primary.buffers and all(buf == expected_audio for buf in primary.buffers)
            assert backup.buffers == ([expected_audio] if fail_primary else [])
            assert [m.audio_duration for m in metrics] == [0.02]

            if not end_input:
                # The same stream must handle another utterance without replaying the first.
                stream.push_frame(frames[1])
                stream.flush()
                next_event = await asyncio.wait_for(final_transcript(), timeout=1)
                assert next_event.alternatives[0].text == "transcribed utterance"
                active = backup if fail_primary else primary
                assert active.buffers == [expected_audio, frames[1].data.tobytes()]
                assert [m.audio_duration for m in metrics] == [0.02, 0.01]
                stream.end_input()

            assert not [
                event async for event in stream if event.type == SpeechEventType.FINAL_TRANSCRIPT
            ]
    finally:
        await fallback.aclose()


async def test_batch_fallback_reports_when_all_providers_fail() -> None:
    fallback = FallbackAdapter(
        [_RecordingBatchSTT(fail=True), _RecordingBatchSTT(fail=True)],
        vad=_FlushVAD(),
        max_retry_per_stt=0,
    )
    errors: list[STTError] = []
    fallback.on("error", errors.append)
    try:
        async with fallback.stream() as stream:
            stream.push_frame(
                rtc.AudioFrame(
                    data=b"\x01\x00" * 160,
                    sample_rate=16000,
                    num_channels=1,
                    samples_per_channel=160,
                )
            )
            stream.end_input()

            async def collect() -> list[SpeechEvent]:
                return [event async for event in stream]

            with pytest.raises(APIConnectionError, match="all STTs failed"):
                await asyncio.wait_for(collect(), timeout=1)
        assert len(errors) == 1
        assert not errors[0].recoverable
    finally:
        await fallback.aclose()


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
