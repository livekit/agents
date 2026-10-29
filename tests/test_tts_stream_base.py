"""A plugin TTS stream's contract: delivered audio is never replayed, and every flushed segment
is spoken."""

from __future__ import annotations

import asyncio

import pytest

from livekit.agents import APIConnectionError, APIConnectOptions, tts

pytestmark = pytest.mark.unit

_CONN = APIConnectOptions()


class _BatchTTS(tts.TTS):
    def __init__(self) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False), sample_rate=24000, num_channels=1
        )
        self.attempts = 0
        self.delivered = asyncio.Event()

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = _CONN
    ) -> tts.ChunkedStream:
        return _FailingAfterAudio(tts=self, input_text=text, conn_options=conn_options)


class _FailingAfterAudio(tts.ChunkedStream):
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        assert isinstance(self._tts, _BatchTTS)
        self._tts.attempts += 1
        output_emitter.initialize(
            request_id="r", sample_rate=24000, num_channels=1, mime_type="audio/pcm"
        )
        output_emitter.push(b"\x00\x00" * 2400)
        output_emitter.flush()
        await self._tts.delivered.wait()
        raise APIConnectionError("failure after delivered output")


class _StreamingTTS(tts.TTS):
    def __init__(self) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True), sample_rate=24000, num_channels=1
        )
        self.sent: list[str] = []

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = _CONN
    ) -> tts.ChunkedStream:
        raise NotImplementedError

    def stream(self, *, conn_options: APIConnectOptions = _CONN) -> tts.SynthesizeStream:
        return _SegmentPerFlush(tts=self, conn_options=conn_options)


class _SegmentPerFlush(tts.SynthesizeStream):
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        assert isinstance(self._tts, _StreamingTTS)
        output_emitter.initialize(
            request_id="r", sample_rate=24000, num_channels=1, mime_type="audio/pcm", stream=True
        )
        text = ""
        async for item in self.input:
            if isinstance(item, str):
                text += item
            elif text:
                self._tts.sent.append(text)
                output_emitter.start_segment(segment_id=str(len(self._tts.sent)))
                output_emitter.push(b"\x00\x00" * 2400)
                output_emitter.end_segment()
                text = ""


# Catches: a batch synthesis retried after its audio reached the reader, which plays it twice.
async def test_batch_audio_delivered_before_a_failure_is_not_replayed() -> None:
    provider = _BatchTTS()
    with pytest.raises(APIConnectionError):
        async with provider.synthesize(
            "hello", conn_options=APIConnectOptions(max_retry=1, retry_interval=0)
        ) as stream:
            async for _ in stream:
                provider.delivered.set()
    assert provider.attempts == 1


# Catches: text pushed after a flush dropped instead of starting the next segment.
async def test_each_flush_ends_a_segment_and_later_text_starts_the_next() -> None:
    provider = _StreamingTTS()
    async with provider.stream(conn_options=APIConnectOptions(max_retry=0)) as stream:
        stream.push_text("first")
        stream.flush()
        stream.push_text("second")
        stream.end_input()
        assert [frame async for frame in stream]
    assert provider.sent == ["first", "second"]


class _FailedStreamTTS(_StreamingTTS):
    def __init__(self) -> None:
        super().__init__()
        self.attempts = 0

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = _CONN
    ) -> tts.ChunkedStream:
        return self._synthesize_with_stream(text, conn_options=conn_options)

    def stream(self, *, conn_options: APIConnectOptions = _CONN) -> tts.SynthesizeStream:
        return _FailedStream(tts=self, conn_options=conn_options)


class _FailedStream(tts.SynthesizeStream):
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        assert isinstance(self._tts, _FailedStreamTTS)
        self._tts.attempts += 1
        raise APIConnectionError("failure before any audio")


async def test_batch_over_stream_uses_one_retry_budget() -> None:
    provider = _FailedStreamTTS()
    try:
        with pytest.raises(APIConnectionError):
            async with provider.synthesize(
                "hello", conn_options=APIConnectOptions(max_retry=1, retry_interval=0)
            ) as stream:
                async for _ in stream:
                    pytest.fail("the provider failed before producing audio")
        assert provider.attempts == 2
    finally:
        await provider.aclose()


class _StalledResponse(tts.ChunkedStream):
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id="r", sample_rate=24000, num_channels=1, mime_type="audio/pcm"
        )
        deadline = self._core_attempts
        # No provider response is owed while the application is idle.
        await asyncio.sleep(0.04)
        deadline.expect_response()
        await asyncio.sleep(0.01)
        deadline.progress()
        await asyncio.sleep(0.01)
        deadline.complete_response()
        await asyncio.sleep(0.04)
        deadline.expect_response()
        await asyncio.Future()


class _StallingTTS(_BatchTTS):
    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = _CONN
    ) -> tts.ChunkedStream:
        return _StalledResponse(tts=self, input_text=text, conn_options=conn_options)


async def test_stalled_provider_response_expires_in_the_read_phase() -> None:
    from livekit.agents import APITimeoutError

    provider = _StallingTTS()
    options = APIConnectOptions(read_timeout=0.02, max_retry=0)
    try:
        async with asyncio.timeout(1):
            with pytest.raises(APITimeoutError) as failure:
                async with provider.synthesize("hello", conn_options=options) as stream:
                    async for _ in stream:
                        pytest.fail("the provider never answered")
        assert failure.value.phase == "read"
    finally:
        await provider.aclose()


class _FailedBatch(tts.ChunkedStream):
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        assert isinstance(self._tts, _FailingBatchTTS)
        self._tts.attempts += 1
        raise APIConnectionError("failure before any audio")


class _FailingBatchTTS(_BatchTTS):
    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = _CONN
    ) -> tts.ChunkedStream:
        return _FailedBatch(tts=self, input_text=text, conn_options=conn_options)


# Catches: a StreamAdapter stream retrying as a whole on top of its per-sentence retries.
async def test_a_stream_adapter_retries_each_sentence_not_the_whole_stream() -> None:
    provider = _FailingBatchTTS()
    adapter = tts.StreamAdapter(tts=provider)
    try:
        async with asyncio.timeout(5):
            async with adapter.stream(
                conn_options=APIConnectOptions(max_retry=1, retry_interval=0)
            ) as stream:
                stream.push_text("Hello there.")
                stream.end_input()
                with pytest.raises(APIConnectionError):
                    async for _ in stream:
                        pytest.fail("the provider failed before producing audio")
        assert provider.attempts == 2
    finally:
        await adapter.aclose()


async def test_a_core_opened_stream_carries_its_expressive_option() -> None:
    provider = _StreamingTTS()
    for expressive in (True, False):
        stream = await provider._core_stream(expressive)
        try:
            assert stream._expressive is expressive
        finally:
            await stream.aclose()
