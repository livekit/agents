from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import pytest

from livekit import rtc
from livekit.agents import NOT_GIVEN, APIConnectionError, APIConnectOptions, NotGivenOr, utils
from livekit.agents.llm import ChatContext
from livekit.agents.stt import (
    FallbackAdapter,
    RecognizeStream,
    SpeechData,
    SpeechEvent,
    SpeechEventType,
    StreamAdapter,
)
from livekit.agents.vad import VAD, VADEvent, VADEventType, VADStream
from livekit.agents.voice import AgentSession
from livekit.agents.voice.audio_recognition import AudioRecognition, _EndOfTurnInfo
from livekit.agents.voice.endpointing import BaseEndpointing

from .fake_stt import FakeSTT
from .fake_vad import FakeVAD
from .test_vad_buffer_resize import vad_impl as vad_impl

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


class _ClipVAD(FakeVAD):
    """Nonzero frames start speech, silence ends it; flush discards open audio."""

    def __init__(self, *, silence: float = 0.0, inference: float = 0.0) -> None:
        super().__init__()
        self.silence = silence
        self.inference = inference

    def stream(self) -> VADStream:
        return _ClipVADStream(self)


class _ClipVADStream(VADStream):
    async def _main_task(self) -> None:
        frames: list[rtc.AudioFrame] = []
        async for item in self._input_ch:
            if isinstance(item, self._FlushSentinel):
                frames = []
            elif any(item.data):
                if not frames:
                    self._send(VADEventType.START_OF_SPEECH, [])
                frames.append(item)
            elif frames:
                self._send(VADEventType.END_OF_SPEECH, frames)
                frames = []

    def _send(self, kind: VADEventType, frames: list[rtc.AudioFrame]) -> None:
        assert isinstance(self._vad, _ClipVAD)
        self._event_ch.send_nowait(
            VADEvent(
                type=kind,
                samples_index=0,
                timestamp=0.0,
                speech_duration=0.01,
                silence_duration=self._vad.silence,
                inference_duration=self._vad.inference,
                frames=frames,
            )
        )


def _reply(
    result: asyncio.Future[SpeechEvent],
    text: str | None = "hello",
    *,
    speech_end_time: float | None = None,
) -> None:
    result.set_result(
        SpeechEvent(
            type=FINAL,
            speech_end_time=speech_end_time,
            alternatives=[] if text is None else [SpeechData(language="en", text=text)],
        )
    )


def _push(stream: RecognizeStream) -> None:
    stream.push_frame(rtc.AudioFrame(bytes([1, 0]) * 160, 16000, 1, 160))


def _close_clip(stream: RecognizeStream) -> None:
    stream.push_frame(rtc.AudioFrame.create(16000, 1, 160))


async def _next(stream: AsyncIterator[SpeechEvent], kind: SpeechEventType) -> SpeechEvent:
    event = await asyncio.wait_for(anext(stream), 2.0)
    assert event.type == kind
    return event


async def _drain(stream: RecognizeStream) -> list[SpeechEvent]:
    async def _collect() -> list[SpeechEvent]:
        return [event async for event in stream]

    return await asyncio.wait_for(_collect(), 2.0)


@asynccontextmanager
async def _stream(
    *, enabled: bool | None = True, cap: float = 5.0, vad: VAD | None = None
) -> AsyncIterator[tuple[RecognizeStream, _ControlledSTT]]:
    stt = _ControlledSTT()
    options = {} if enabled is None else {"end_of_speech_after_transcript": enabled}
    adapter = StreamAdapter(stt=stt, vad=vad or _ClipVAD(), max_transcript_wait=cap, **options)
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
        _close_clip(stream)
        await _next(stream, START)
        await _next(stream, END)
        _reply(await asyncio.wait_for(stt.calls.recv(), 2.0))
        stream.end_input()
        assert [event.type for event in await _drain(stream)] == [FINAL]


@pytest.mark.parametrize("provider_end", [None, 1234.0])
async def test_final_before_end_preserves_known_anchors(provider_end: float | None) -> None:
    async with _stream(vad=_ClipVAD(silence=0.2, inference=0.05)) as (stream, stt):
        _push(stream)
        expected_end = time.time() - 0.25
        _close_clip(stream)
        await _next(stream, START)
        result = await asyncio.wait_for(stt.calls.recv(), 2.0)
        await asyncio.sleep(1.0)
        _reply(result, speech_end_time=provider_end)
        final = await _next(stream, FINAL)
        end = await _next(stream, END)
        assert final.alternatives[0].text == "hello"
        assert end.speech_end_time == pytest.approx(expected_end, abs=0.001)
        assert final.speech_end_time == pytest.approx(
            provider_end if provider_end is not None else expected_end, abs=0.001
        )
        # Passing the old deadline must not emit a second END.
        await asyncio.sleep(5.0)
        stream.end_input()
        assert await _drain(stream) == []


@pytest.mark.parametrize("close_before_first_result", [False, True])
async def test_resumed_speech_waits_for_both_clips(close_before_first_result: bool) -> None:
    async with _stream(vad=_ClipVAD(silence=0.2, inference=0.05)) as (stream, stt):
        _push(stream)
        _close_clip(stream)
        await _next(stream, START)
        first = await asyncio.wait_for(stt.calls.recv(), 2.0)
        _push(stream)
        # START must arrive while the first recognition is still blocked.
        await _next(stream, START)
        await asyncio.sleep(0.4)
        expected_end = time.time() - 0.25
        if close_before_first_result:
            _close_clip(stream)
            # Virtual time advances only after queued VAD events have drained.
            await asyncio.sleep(0.1)
        _reply(first, "first")
        await _next(stream, FINAL)
        if not close_before_first_result:
            _close_clip(stream)
        second = await asyncio.wait_for(stt.calls.recv(), 2.0)
        _reply(second, "second")
        stream.end_input()
        remaining = await _drain(stream)
        assert [event.type for event in remaining] == [FINAL, END]
        assert remaining[0].alternatives[0].text == "second"
        assert remaining[0].speech_end_time == pytest.approx(expected_end, abs=0.001)
        assert remaining[1].speech_end_time == pytest.approx(expected_end, abs=0.001)


@pytest.mark.parametrize("cap", [0.0, 0.02])
async def test_cap_does_not_cancel_late_transcript(cap: float) -> None:
    async with _stream(cap=cap) as (stream, stt):
        _push(stream)
        _close_clip(stream)
        await _next(stream, START)
        result = await asyncio.wait_for(stt.calls.recv(), 2.0)
        await _next(stream, END)
        assert not result.done()
        _reply(result)
        stream.end_input()
        assert [event.type for event in await _drain(stream)] == [FINAL]


async def test_end_input_drains_held_end() -> None:
    async with _stream() as (stream, stt):
        _push(stream)
        _close_clip(stream)
        await _next(stream, START)
        result = await asyncio.wait_for(stt.calls.recv(), 2.0)
        # Only clips already closed by VAD are promised to drain.
        stream.end_input()
        _reply(result)
        assert [event.type for event in await _drain(stream)] == [FINAL, END]


@pytest.mark.parametrize("text", ["", None])
async def test_empty_transcript_still_ends_speech(text: str | None) -> None:
    async with _stream() as (stream, stt):
        _push(stream)
        _close_clip(stream)
        await _next(stream, START)
        _reply(await asyncio.wait_for(stt.calls.recv(), 2.0), text)
        await _next(stream, END)
        stream.end_input()
        assert await _drain(stream) == []


@pytest.mark.parametrize("during_flush", [False, True])
async def test_close_cancels_recognition(during_flush: bool) -> None:
    async with _stream() as (stream, stt):
        _push(stream)
        _close_clip(stream)
        await _next(stream, START)
        result = await asyncio.wait_for(stt.calls.recv(), 2.0)
        if during_flush:
            stream.flush()
            await asyncio.sleep(0)
        await stream.aclose()
        assert result.cancelled()
        assert stt.cancelled.is_set()
        assert await _drain(stream) == []


async def test_resumed_speech_restarts_cap() -> None:
    async with _stream(cap=1.0) as (stream, stt):
        _push(stream)
        _close_clip(stream)
        await _next(stream, START)
        first = await asyncio.wait_for(stt.calls.recv(), 2.0)
        _push(stream)
        await _next(stream, START)
        # Pass the first clip's deadline while speech is active.
        await asyncio.sleep(1.5)
        latest_close = asyncio.get_running_loop().time()
        _close_clip(stream)
        _reply(first, "first")
        await _next(stream, FINAL)
        second = await asyncio.wait_for(stt.calls.recv(), 2.0)
        await _next(stream, END)
        assert asyncio.get_running_loop().time() == pytest.approx(latest_close + 1.0, abs=0.001)
        assert not second.done()
        _reply(second, "second")
        stream.end_input()
        assert [event.type for event in await _drain(stream)] == [FINAL]


@pytest.mark.parametrize("cap", [-1.0, float("inf"), float("nan")])
async def test_invalid_cap(cap: float) -> None:
    stt = _ControlledSTT()
    try:
        with pytest.raises(ValueError, match="max_transcript_wait"):
            StreamAdapter(stt=stt, vad=_ClipVAD(), max_transcript_wait=cap)
    finally:
        await stt.aclose()


async def test_recognition_failure_is_propagated() -> None:
    async with _stream() as (stream, stt):
        _push(stream)
        _close_clip(stream)
        await _next(stream, START)
        result = await asyncio.wait_for(stt.calls.recv(), 2.0)
        result.set_exception(RuntimeError("recognition failed"))
        await _next(stream, END)
        with pytest.raises(RuntimeError, match="recognition failed"):
            await anext(stream)


@pytest.mark.parametrize("active_speech", [False, True])
@pytest.mark.parametrize("abandoned_clip", [False, True])
async def test_later_failure_settles_only_silent_speech(
    active_speech: bool, abandoned_clip: bool
) -> None:
    async with _stream() as (stream, stt):
        _push(stream)
        _close_clip(stream)
        await _next(stream, START)
        first = await asyncio.wait_for(stt.calls.recv(), 2.0)
        _push(stream)
        _close_clip(stream)
        await _next(stream, START)
        _reply(first, "first")
        await _next(stream, FINAL)
        second = await asyncio.wait_for(stt.calls.recv(), 2.0)
        if abandoned_clip:
            _push(stream)
            _close_clip(stream)
            await _next(stream, START)
            await asyncio.sleep(0.1)
        if active_speech:
            _push(stream)
            await _next(stream, START)
        second.set_exception(RuntimeError("second clip failed"))
        if not active_speech:
            await _next(stream, END)
        with pytest.raises(RuntimeError, match="second clip failed"):
            await anext(stream)
        assert stt.calls.qsize() == 0


async def test_flush_discards_open_audio_and_releases_previous_clip() -> None:
    async with _stream(cap=0.5) as (stream, stt):
        _push(stream)
        _close_clip(stream)
        await _next(stream, START)
        first = await asyncio.wait_for(stt.calls.recv(), 2.0)
        _push(stream)
        await _next(stream, START)
        stream.flush()
        _reply(first, "first")
        await _next(stream, FINAL)
        await _next(stream, END)
        # No recognition request is made for the discarded open clip.
        stream.end_input()
        assert await _drain(stream) == []
        assert stt.calls.qsize() == 0


@asynccontextmanager
async def _turn_consumer(
    stream: RecognizeStream,
) -> AsyncIterator[tuple[list[str], utils.aio.Chan[SpeechEvent]]]:
    """Feed actual adapter events through the real consumer and endpoint scheduler."""
    turns: list[str] = []
    hooks = MagicMock()
    hooks.retrieve_chat_ctx.return_value = ChatContext.empty()

    def _commit(info: _EndOfTurnInfo) -> bool:
        turns.append(info.new_transcript)
        return True

    hooks.on_end_of_turn.side_effect = _commit
    recognition = AudioRecognition(
        AgentSession(),
        hooks=hooks,
        endpointing=BaseEndpointing(min_delay=0.01, max_delay=0.01),
        stt=MagicMock(),
        vad=None,
        interruption_detection=None,
        turn_detection="stt",
    )
    events = utils.aio.Chan[SpeechEvent]()

    async def _consume() -> None:
        async for event in stream:
            await recognition._on_stt_event(event)
            events.send_nowait(event)

    task = asyncio.create_task(_consume())
    try:
        yield turns, events
    finally:
        await utils.aio.cancel_and_wait(task)
        await recognition._aclose()


@pytest.mark.parametrize("enabled", [False, True])
async def test_consumer_commits_resumed_clips_once_when_deferred(enabled: bool) -> None:
    async with _stream(enabled=enabled) as (stream, stt), _turn_consumer(stream) as (turns, events):
        _push(stream)
        _close_clip(stream)
        await _next(events, START)
        if not enabled:
            await _next(events, END)
        first = await asyncio.wait_for(stt.calls.recv(), 2.0)
        # Queue the resumed clip before releasing the first recognition. The
        # default path cannot read it yet; the opt-in reader can.
        _push(stream)
        _close_clip(stream)
        if enabled:
            await _next(events, START)
        _reply(first, "first")
        await _next(events, FINAL)
        if not enabled:
            await _next(events, START)
            await _next(events, END)
        second = await asyncio.wait_for(stt.calls.recv(), 2.0)
        await asyncio.sleep(0.1)
        assert turns == ([] if enabled else ["first"])
        _reply(second, "second")
        await _next(events, FINAL)
        if enabled:
            await _next(events, END)
        await asyncio.sleep(0.1)
        # With the default ordering, the first clip has already committed;
        # the second text is now waiting for another boundary. Deferral puts
        # both clips in the same committed turn without needing more audio.
        assert turns == (["first second"] if enabled else ["first"])


async def test_consumer_can_still_split_after_cap() -> None:
    async with _stream(cap=0.5) as (stream, stt), _turn_consumer(stream) as (turns, events):
        _push(stream)
        _close_clip(stream)
        await _next(events, START)
        first = await asyncio.wait_for(stt.calls.recv(), 2.0)
        _push(stream)
        _close_clip(stream)
        await _next(events, START)
        _reply(first, "first")
        await _next(events, FINAL)
        second = await asyncio.wait_for(stt.calls.recv(), 2.0)
        await _next(events, END)
        await asyncio.sleep(0.1)
        assert turns == ["first"]
        _reply(second, "second")
        await _next(events, FINAL)
        # A later VAD segment (even one with no recognized text) can release
        # the late text as a separate turn. Deferral cannot undo the first.
        _push(stream)
        _close_clip(stream)
        await _next(events, START)
        third = await asyncio.wait_for(stt.calls.recv(), 2.0)
        _reply(third, "")
        await _next(events, END)
        await asyncio.sleep(0.1)
        assert turns == ["first", "second"]


async def test_fallback_preserves_turn_after_later_recognition_failure() -> None:
    primary, backup = _ControlledSTT(), _ControlledSTT()
    adapters = [
        StreamAdapter(stt=stt, vad=_ClipVAD(), end_of_speech_after_transcript=True)
        for stt in (primary, backup)
    ]
    fallback = FallbackAdapter(adapters, max_retry_per_stt=0)
    switched = asyncio.Event()
    fallback.on("stt_availability_changed", lambda _: switched.set())
    stream = fallback.stream(conn_options=APIConnectOptions(max_retry=0))
    try:
        async with _turn_consumer(stream) as (turns, events):
            _push(stream)
            _close_clip(stream)
            await _next(events, START)
            first = await asyncio.wait_for(primary.calls.recv(), 2.0)
            _push(stream)
            _close_clip(stream)
            await _next(events, START)
            _reply(first, "first")
            await _next(events, FINAL)
            second = await asyncio.wait_for(primary.calls.recv(), 2.0)
            second.set_exception(APIConnectionError("provider unavailable", retryable=False))
            await _next(events, END)
            await asyncio.wait_for(switched.wait(), 2.0)
            await asyncio.sleep(0.1)
            assert turns == ["first"]
            assert backup.calls.qsize() == 0
            stream.end_input()
    finally:
        await stream.aclose()
        await fallback.aclose()
        for adapter in adapters:
            await adapter.aclose()
        await primary.aclose()
        await backup.aclose()


def _window(value: int) -> rtc.AudioFrame:
    return rtc.AudioFrame(bytes([value, 0]) * 512, 16000, 1, 512)


def _real_clip_end(stream: RecognizeStream) -> None:
    for _ in range(5):
        stream.push_frame(_window(0))


@pytest.mark.real_time
@pytest.mark.parametrize("boundary", ["flush", "end_input", "flush_then_audio"])
async def test_real_vad_reset_settles_closed_clip(vad_impl: VAD, boundary: str) -> None:
    # The VAD implementation and its reset logic are real; only the classifier
    # is replaced by the existing vad_impl fixture's deterministic model.
    async with _stream(vad=vad_impl, cap=0.5) as (stream, stt):
        stream.push_frame(_window(1))
        await _next(stream, START)
        _real_clip_end(stream)
        first = await asyncio.wait_for(stt.calls.recv(), 2.0)
        stream.push_frame(_window(1))
        await _next(stream, START)
        if boundary == "end_input":
            stream.end_input()
        else:
            stream.flush()
        if boundary == "flush_then_audio":
            # Queue audio immediately after the reset, before VAD has drained.
            stream.push_frame(_window(1))
            await _next(stream, START)
        _reply(first, "first")
        await _next(stream, FINAL)
        if boundary == "flush_then_audio":
            # The new open segment must keep END held, then close normally.
            _real_clip_end(stream)
            second = await asyncio.wait_for(stt.calls.recv(), 2.0)
            _reply(second, "second")
            await _next(stream, FINAL)
        await _next(stream, END)
        if boundary != "end_input":
            stream.end_input()
        assert await _drain(stream) == []
        assert stt.calls.qsize() == 0


@pytest.mark.real_time
async def test_real_vad_end_input_does_not_promise_open_audio(vad_impl: VAD) -> None:
    async with _stream(vad=vad_impl) as (stream, stt):
        stream.push_frame(_window(1))
        await _next(stream, START)
        stream.end_input()
        assert await _drain(stream) == []
        assert stt.calls.qsize() == 0


async def test_flush_restores_elapsed_deadline_without_waiting_for_text() -> None:
    async with _stream(cap=1.0) as (stream, stt):
        _push(stream)
        _close_clip(stream)
        await _next(stream, START)
        result = await asyncio.wait_for(stt.calls.recv(), 2.0)
        _push(stream)
        await _next(stream, START)
        await asyncio.sleep(1.5)
        reset_at = asyncio.get_running_loop().time()
        stream.flush()
        await _next(stream, END)
        assert asyncio.get_running_loop().time() == pytest.approx(reset_at, abs=0.001)
        assert not result.done()
        _reply(result)
        await _next(stream, FINAL)
        stream.end_input()
        assert await _drain(stream) == []
