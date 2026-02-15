from __future__ import annotations

import asyncio

import pytest

from livekit import rtc
from livekit.agents import APIConnectionError, utils
from livekit.agents.stt import (
    STT,
    AvailabilityChangedEvent,
    FallbackAdapter,
    RecognizeStream,
    SpeechEvent,
    STTCapabilities,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from livekit.agents.utils.aio.channel import ChanEmpty
from livekit.agents.utils.audio import AudioBuffer

from .fake_stt import FakeSTT


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
        tts: STT,
    ) -> utils.aio.ChanReceiver[AvailabilityChangedEvent]:
        return self._availability_changed_ch[id(tts)]


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
