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
