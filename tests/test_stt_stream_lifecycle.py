"""Host-authored recognition retains input, final responses, and cleanup ownership."""

from __future__ import annotations

import asyncio
from typing import Never

import pytest

from livekit.agents import (
    NOT_GIVEN,
    APIConnectionError,
    APIConnectOptions,
    APITimeoutError,
    NotGivenOr,
    stt,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
from livekit.agents.utils import aio
from livekit.agents.utils.audio import AudioBuffer, silence_frame

pytestmark = pytest.mark.unit


class Provider(stt.STT[Never]):
    def __init__(self) -> None:
        super().__init__(capabilities=stt.STTCapabilities(streaming=True, interim_results=True))

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        raise NotImplementedError

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> Stream:
        return Stream(stt=self, conn_options=conn_options)


class Stream(stt.RecognizeStream):
    async def _run(self) -> None:
        async for _ in self.input:
            pass


async def test_recognition_input_is_bounded_while_provider_is_stalled() -> None:
    stream = Provider().stream(conn_options=APIConnectOptions(max_retry=0))
    frame = silence_frame(duration=0.02, sample_rate=16000)
    try:
        with pytest.raises(aio.ChanFull):
            for _ in range(257):
                stream.push_frame(frame)
    finally:
        await stream.aclose()


async def test_recognition_event_queue_refuses_unconsumed_transcripts() -> None:
    stream = Provider().stream(conn_options=APIConnectOptions(max_retry=0))
    event = stt.SpeechEvent(type=stt.SpeechEventType.FINAL_TRANSCRIPT, alternatives=[])
    try:
        with pytest.raises(aio.ChanFull):
            for _ in range(65):
                stream.send(event)
    finally:
        await stream.aclose()


async def test_recognition_passive_listening_does_not_start_read_timeout() -> None:
    stream = Provider().stream(conn_options=APIConnectOptions(max_retry=0, read_timeout=0))
    try:
        await asyncio.sleep(0.01)
        assert not stream.closed
        stream.end_input()
        with pytest.raises(APITimeoutError) as error:
            await anext(stream)
        assert error.value.phase == "read"
    finally:
        await stream.aclose()


async def test_recognition_return_before_input_eof_is_a_protocol_failure() -> None:
    class Premature(Stream):
        async def _run(self) -> None:
            return

    stream = Premature(stt=Provider(), conn_options=APIConnectOptions(max_retry=0))
    try:
        with pytest.raises(APIConnectionError, match="completion"):
            await anext(stream)
    finally:
        await stream.aclose()


async def test_recognition_waits_for_final_transcript_after_input_eof() -> None:
    class DelayedFinal(Stream):
        async def _run(self) -> None:
            async for _ in self.input:
                pass
            await asyncio.sleep(0.01)
            self.send(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=[stt.SpeechData(language="en", text="last words")],
                )
            )

    stream = DelayedFinal(stt=Provider(), conn_options=APIConnectOptions(max_retry=0))
    try:
        stream.end_input()
        assert [event.alternatives[0].text async for event in stream] == ["last words"]
    finally:
        await stream.aclose()


async def test_recognition_close_retains_cancellation_resistant_cleanup() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    settled = asyncio.Event()

    class Resistant(Stream):
        async def _run(self) -> None:
            entered.set()
            try:
                await asyncio.Future[None]()
            except asyncio.CancelledError:
                await release.wait()
            finally:
                settled.set()

    stream = Resistant(stt=Provider(), conn_options=APIConnectOptions(max_retry=0))
    await entered.wait()
    closing = asyncio.create_task(stream.aclose())
    try:
        # The shared native settlement allowance is five seconds. A waiter does not cancel
        # cleanup again when that allowance passes; the operation continues to own it.
        await asyncio.wait_for(asyncio.shield(closing), timeout=6)
        assert not settled.is_set()
    finally:
        release.set()
        await closing
        await asyncio.wait_for(settled.wait(), timeout=1)


async def test_recognition_flush_queued_behind_backpressure_does_not_arm_read_timeout() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    class Blocked(Stream):
        async def _run(self) -> None:
            entered.set()
            await release.wait()
            async for _ in self.input:
                pass

    stream = Blocked(stt=Provider(), conn_options=APIConnectOptions(max_retry=0, read_timeout=0))
    try:
        await entered.wait()
        stream.flush()
        await asyncio.sleep(0.01)
        assert not stream.closed
    finally:
        release.set()
        await stream.aclose()


@pytest.mark.parametrize("completion", ["premature", "read_timeout"])
async def test_core_owned_recognition_keeps_completion_and_read_deadlines(completion: str) -> None:
    class Premature(Stream):
        async def _run(self) -> None:
            return

    class BoundaryProvider(Provider):
        def stream(
            self,
            *,
            language: NotGivenOr[str] = NOT_GIVEN,
            conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        ) -> Stream:
            operation = Premature if completion == "premature" else Stream
            return operation(stt=self, conn_options=conn_options)

    provider = BoundaryProvider()
    stream = await provider._core_stream(None, None, APIConnectOptions(max_retry=3, read_timeout=0))
    assert stream is not None
    try:
        if completion == "read_timeout":
            stream.end_input()
            with pytest.raises(APITimeoutError) as error:
                await anext(stream)
            assert error.value.phase == "read"
        else:
            with pytest.raises(APIConnectionError, match="completion"):
                await anext(stream)
    finally:
        await stream.aclose()


@pytest.mark.parametrize("ended", [False, True])
async def test_recognition_admission_preserves_the_terminal_timeout_cause(ended: bool) -> None:
    stream = Provider().stream(conn_options=APIConnectOptions(max_retry=0, total_timeout=0))
    frame = silence_frame(duration=0.02, sample_rate=16000)
    if ended:
        stream.end_input()
    try:
        with pytest.raises(APITimeoutError) as failed_read:
            await anext(stream)
        assert failed_read.value.phase == "total"
        for admit in (lambda: stream.push_frame(frame), stream.flush):
            with pytest.raises(APITimeoutError) as failed_input:
                admit()
            assert failed_input.value.phase == "total"
    finally:
        await stream.aclose()
