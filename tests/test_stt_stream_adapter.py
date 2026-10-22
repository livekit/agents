from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest

from livekit import rtc
from livekit.agents import NOT_GIVEN, APIConnectOptions, NotGivenOr, utils
from livekit.agents.stt import (
    RecognizeStream,
    SpeechData,
    SpeechEvent,
    SpeechEventType,
    StreamAdapter,
)
from livekit.agents.vad import VADEvent, VADEventType, VADStream

from .fake_stt import FakeSTT
from .fake_vad import FakeVAD

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time]

START = SpeechEventType.START_OF_SPEECH
END = SpeechEventType.END_OF_SPEECH
FINAL = SpeechEventType.FINAL_TRANSCRIPT


class _ControlledSTT(FakeSTT):
    """Let each test release recognition without relying on provider timing."""

    def __init__(self) -> None:
        super().__init__()
        self.capabilities.streaming = False
        self.calls = utils.aio.Chan[asyncio.Future[SpeechEvent]]()
        self.cancelled = asyncio.Event()

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> SpeechEvent:
        result: asyncio.Future[SpeechEvent] = asyncio.get_running_loop().create_future()
        self.calls.send_nowait(result)
        try:
            return await result
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


class _FlushVAD(FakeVAD):
    """Use real input and flush queues, with a clip ending on each flush."""

    def stream(self) -> VADStream:
        return _FlushVADStream(self)


class _FlushVADStream(VADStream):
    async def _main_task(self) -> None:
        frames: list[rtc.AudioFrame] = []
        async for item in self._input_ch:
            if isinstance(item, rtc.AudioFrame):
                if not frames:
                    self._send(VADEventType.START_OF_SPEECH, [])
                frames.append(item)
            elif frames:
                self._send(VADEventType.END_OF_SPEECH, frames)
                frames = []

    def _send(self, kind: VADEventType, frames: list[rtc.AudioFrame]) -> None:
        self._event_ch.send_nowait(
            VADEvent(
                type=kind,
                samples_index=0,
                timestamp=0.0,
                speech_duration=0.01,
                silence_duration=0.0,
                frames=frames,
            )
        )


def _reply(result: asyncio.Future[SpeechEvent], text: str | None = "hello") -> None:
    result.set_result(
        SpeechEvent(
            type=FINAL,
            alternatives=[] if text is None else [SpeechData(language="en", text=text)],
        )
    )


def _push(stream: RecognizeStream) -> None:
    stream.push_frame(rtc.AudioFrame.create(16000, 1, 160))


async def _next(stream: RecognizeStream, kind: SpeechEventType) -> SpeechEvent:
    event = await asyncio.wait_for(anext(stream), 2.0)
    assert event.type == kind
    return event


@asynccontextmanager
async def _stream(
    *, enabled: bool | None = True, cap: float = 5.0
) -> AsyncIterator[tuple[RecognizeStream, _ControlledSTT]]:
    stt = _ControlledSTT()
    options = {} if enabled is None else {"end_of_speech_after_transcript": enabled}
    adapter = StreamAdapter(stt=stt, vad=_FlushVAD(), max_transcript_wait=cap, **options)
    stream = adapter.stream()
    try:
        yield stream, stt
    finally:
        await stream.aclose()
        await adapter.aclose()
        await stt.aclose()


@pytest.mark.parametrize("enabled", [None, False])
async def test_default_order(enabled: bool | None) -> None:
    async with _stream(enabled=enabled) as (stream, stt):
        _push(stream)
        stream.flush()
        await _next(stream, START)
        await _next(stream, END)
        _reply(await asyncio.wait_for(stt.calls.recv(), 2.0))
        stream.end_input()
        assert [event.type async for event in stream] == [FINAL]


async def test_final_before_end() -> None:
    async with _stream() as (stream, stt):
        _push(stream)
        stream.flush()
        await _next(stream, START)
        _reply(await asyncio.wait_for(stt.calls.recv(), 2.0))
        final = await _next(stream, FINAL)
        end = await _next(stream, END)
        assert final.alternatives[0].text == "hello"
        assert end.speech_end_time == final.speech_end_time
        stream.end_input()
        assert [event async for event in stream] == []


@pytest.mark.parametrize("close_before_first_result", [False, True])
async def test_resumed_speech_waits_for_both_clips(close_before_first_result: bool) -> None:
    async with _stream() as (stream, stt):
        _push(stream)
        stream.flush()
        await _next(stream, START)
        first = await asyncio.wait_for(stt.calls.recv(), 2.0)
        _push(stream)
        # START must arrive while the first recognition is still blocked.
        await _next(stream, START)
        if close_before_first_result:
            stream.flush()
            # Virtual time advances only after queued VAD events have drained.
            await asyncio.sleep(0.1)
        _reply(first, "first")
        await _next(stream, FINAL)
        if not close_before_first_result:
            stream.flush()
        second = await asyncio.wait_for(stt.calls.recv(), 2.0)
        _reply(second, "second")
        stream.end_input()
        remaining = [event async for event in stream]
        assert [event.type for event in remaining] == [FINAL, END]
        assert remaining[0].alternatives[0].text == "second"


@pytest.mark.parametrize("cap", [0.0, 0.02])
async def test_cap_does_not_cancel_late_transcript(cap: float) -> None:
    async with _stream(cap=cap) as (stream, stt):
        _push(stream)
        stream.flush()
        await _next(stream, START)
        result = await asyncio.wait_for(stt.calls.recv(), 2.0)
        await _next(stream, END)
        assert not result.done()
        _reply(result)
        stream.end_input()
        assert [event.type async for event in stream] == [FINAL]


async def test_end_input_drains_held_end() -> None:
    async with _stream() as (stream, stt):
        _push(stream)
        # No explicit flush: end_input must close the queued clip itself.
        stream.end_input()
        await _next(stream, START)
        _reply(await asyncio.wait_for(stt.calls.recv(), 2.0))
        assert [event.type async for event in stream] == [FINAL, END]


@pytest.mark.parametrize("text", ["", None])
async def test_empty_transcript_still_ends_speech(text: str | None) -> None:
    async with _stream() as (stream, stt):
        _push(stream)
        stream.flush()
        await _next(stream, START)
        _reply(await asyncio.wait_for(stt.calls.recv(), 2.0), text)
        await _next(stream, END)
        stream.end_input()
        assert [event async for event in stream] == []


async def test_close_cancels_recognition() -> None:
    async with _stream() as (stream, stt):
        _push(stream)
        stream.flush()
        await _next(stream, START)
        result = await asyncio.wait_for(stt.calls.recv(), 2.0)
        await stream.aclose()
        assert result.cancelled()
        assert stt.cancelled.is_set()
        assert [event async for event in stream] == []


async def test_resumed_speech_restarts_cap() -> None:
    async with _stream(cap=1.0) as (stream, stt):
        _push(stream)
        stream.flush()
        await _next(stream, START)
        first = await asyncio.wait_for(stt.calls.recv(), 2.0)
        _push(stream)
        await _next(stream, START)
        # Pass the first clip's deadline while speech is active.
        await asyncio.sleep(1.5)
        latest_close = asyncio.get_running_loop().time()
        stream.flush()
        _reply(first, "first")
        await _next(stream, FINAL)
        second = await asyncio.wait_for(stt.calls.recv(), 2.0)
        await _next(stream, END)
        assert asyncio.get_running_loop().time() - latest_close >= 1.0
        assert not second.done()
        _reply(second, "second")
        stream.end_input()
        assert [event.type async for event in stream] == [FINAL]


@pytest.mark.parametrize("cap", [-1.0, float("inf"), float("nan")])
async def test_invalid_cap(cap: float) -> None:
    stt = _ControlledSTT()
    try:
        with pytest.raises(ValueError, match="max_transcript_wait"):
            StreamAdapter(stt=stt, vad=_FlushVAD(), max_transcript_wait=cap)
    finally:
        await stt.aclose()


async def test_recognition_failure_is_propagated() -> None:
    async with _stream() as (stream, stt):
        _push(stream)
        stream.flush()
        await _next(stream, START)
        result = await asyncio.wait_for(stt.calls.recv(), 2.0)
        result.set_exception(RuntimeError("recognition failed"))
        with pytest.raises(RuntimeError, match="recognition failed"):
            await anext(stream)
