from __future__ import annotations

import asyncio
import base64
import json
from types import MethodType, SimpleNamespace
from typing import Any

import aiohttp
import pytest

from livekit import rtc
from livekit.agents import APIConnectOptions, APIStatusError
from livekit.plugins import slng


class _FakeStream:
    def __init__(self, results: list[Any]) -> None:
        self.results = list(results)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    def push_text(self, text: str) -> None:
        del text

    def flush(self) -> None:
        pass

    def end_input(self) -> None:
        pass

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.results:
            raise StopAsyncIteration
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    async def aclose(self) -> None:
        pass


def test_tts_requires_voice() -> None:
    with pytest.raises(TypeError):
        slng.TTS(api_key="test-key", model="deepgram/aura:2")  # type: ignore[call-arg]


def test_tts_builds_unmute_bridge_connection() -> None:
    tts = slng.TTS(
        api_key="test-key",
        model="deepgram/aura:2",
        voice="aura-2-thalia-en",
    )
    assert tts._opts.model_endpoint == ("wss://api.slng.ai/v1/bridges/unmute/tts/deepgram/aura:2")


def test_tts_rejects_direct_provider_endpoint() -> None:
    with pytest.raises(ValueError, match="Unmute Bridge"):
        slng.TTS(
            api_key="test-key",
            connections=["wss://api.slng.ai/v1/tts/deepgram/aura:2"],
            voice="aura-2-thalia-en",
        )


def test_tts_keeps_exact_language_and_chunking_options() -> None:
    tts = slng.TTS(
        api_key="test-key",
        model="sarvam/bulbul:v3",
        voice="shubh",
        language="en-IN",
        text_chunking="phrase",
        phrase_max_chars=80,
        pronunciation={"mode": "rewrite", "name": "products"},
    )
    assert tts._opts.language == "en-IN"
    assert tts._opts.text_chunking == "phrase"
    assert tts._opts.phrase_max_chars == 80
    assert tts._opts.model_options["pronunciation"] == {
        "mode": "rewrite",
        "name": "products",
    }


@pytest.mark.asyncio
async def test_tts_fails_over_before_first_audio() -> None:
    tts = slng.TTS(
        api_key="test-key",
        voice="voice-a",
        connections=[
            "provider/model:1",
            slng.TTSConnectionConfig(
                endpoint=("wss://api.slng.ai/v1/bridges/unmute/tts/provider/model:2"),
                voice="voice-b",
            ),
        ],
    )
    first, second = tts._candidate_tts

    def first_stream(self, *, conn_options):
        del self, conn_options
        return _FakeStream([APIStatusError("failed")])

    audio = object()

    def second_stream(self, *, conn_options):
        del self, conn_options
        return _FakeStream([audio])

    first._stream_candidate = MethodType(first_stream, first)  # type: ignore[method-assign]
    second._stream_candidate = MethodType(second_stream, second)  # type: ignore[method-assign]

    stream = tts.stream(conn_options=APIConnectOptions(max_retry=0))
    await stream.__aenter__()
    stream.push_text("hello")
    stream.end_input()
    assert await stream.__anext__() is audio
    assert tts._active_candidate_index == 1
    await stream.aclose()


@pytest.mark.asyncio
async def test_chunked_tts_fails_over_before_first_audio() -> None:
    tts = slng.TTS(
        api_key="test-key",
        voice="voice-a",
        connections=["provider/model:1", "provider/model:2"],
    )
    first, second = tts._candidate_tts

    def first_synthesize(self, text, *, conn_options):
        del self, text, conn_options
        return _FakeStream([APIStatusError("failed")])

    audio = object()

    def second_synthesize(self, text, *, conn_options):
        del self, text, conn_options
        return _FakeStream([audio])

    first._synthesize_candidate = MethodType(  # type: ignore[method-assign]
        first_synthesize, first
    )
    second._synthesize_candidate = MethodType(  # type: ignore[method-assign]
        second_synthesize, second
    )

    stream = tts.synthesize("hello", conn_options=APIConnectOptions(max_retry=0))
    await stream.__aenter__()
    assert await stream.__anext__() is audio
    assert tts._active_candidate_index == 1
    await stream.aclose()


@pytest.mark.asyncio
async def test_tts_does_not_fail_over_after_audio() -> None:
    tts = slng.TTS(
        api_key="test-key",
        voice="voice-a",
        connections=["provider/model:1", "provider/model:2"],
    )
    first = tts._candidate_tts[0]
    audio = object()

    def first_stream(self, *, conn_options):
        del self, conn_options
        return _FakeStream([audio, APIStatusError("late failure")])

    first._stream_candidate = MethodType(first_stream, first)  # type: ignore[method-assign]
    stream = tts.stream(conn_options=APIConnectOptions(max_retry=0))
    await stream.__aenter__()
    stream.push_text("hello")
    stream.end_input()
    assert await stream.__anext__() is audio
    with pytest.raises(APIStatusError, match="late failure"):
        await stream.__anext__()
    assert tts._active_candidate_index == 0


def test_tts_rejects_removed_model_endpoint_kwarg() -> None:
    with pytest.raises(ValueError, match="Unmute Bridge"):
        slng.TTS(
            api_key="test-key",
            model="deepgram/aura:2",
            voice="aura-2-thalia-en",
            model_endpoint="wss://api.slng.ai/v1/tts/deepgram/aura:2",
        )


def test_tts_provider_api_key_becomes_byok_header() -> None:
    tts = slng.TTS(
        api_key="test-key",
        model="deepgram/aura:2",
        voice="aura-2-thalia-en",
        provider_api_key="provider-secret",
    )
    assert tts._opts.extra_headers["X-Slng-Provider-Key"] == "provider-secret"


@pytest.mark.asyncio
async def test_tts_413_is_terminal_without_chain_walk() -> None:
    tts = slng.TTS(
        api_key="test-key",
        voice="voice-a",
        connections=["provider/model:1", "provider/model:2"],
    )
    first, second = tts._candidate_tts

    def first_stream(self, *, conn_options):
        del self, conn_options
        return _FakeStream([APIStatusError("payload too large", status_code=413)])

    second_calls: list[int] = []

    def second_stream(self, *, conn_options):
        del self, conn_options
        second_calls.append(1)
        return _FakeStream([object()])

    first._stream_candidate = MethodType(first_stream, first)  # type: ignore[method-assign]
    second._stream_candidate = MethodType(second_stream, second)  # type: ignore[method-assign]

    stream = tts.stream(conn_options=APIConnectOptions(max_retry=0))
    await stream.__aenter__()
    stream.push_text("hello")
    stream.end_input()
    with pytest.raises(APIStatusError, match="payload too large"):
        await stream.__anext__()
    assert not second_calls
    assert tts._active_candidate_index == 0
    await stream.aclose()


class _ScriptedTtsWs:
    """Fake bridge websocket: records sent frames, answers flush with audio_end."""

    def __init__(self) -> None:
        self.sent: list[dict] = []
        self._q: asyncio.Queue = asyncio.Queue()
        self._q.put_nowait(
            SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data=json.dumps({"type": "ready"}))
        )
        self.closed = False

    async def send_str(self, msg: str) -> None:
        frame = json.loads(msg)
        self.sent.append(frame)
        if frame.get("type") == "flush":
            silence = base64.b64encode(b"\x00" * 4800).decode()
            self._q.put_nowait(
                SimpleNamespace(
                    type=aiohttp.WSMsgType.TEXT,
                    data=json.dumps({"type": "audio_chunk", "data": silence}),
                )
            )
            self._q.put_nowait(
                SimpleNamespace(
                    type=aiohttp.WSMsgType.TEXT,
                    data=json.dumps({"type": "audio_end"}),
                )
            )

    async def receive(self, timeout=None):
        del timeout
        return await self._q.get()

    async def close(self) -> None:
        self.closed = True


async def _sent_text_frames(monkeypatch, text: str, **tts_kwargs) -> list[str]:
    ws = _ScriptedTtsWs()

    async def fake_connect(self, timeout):
        del self, timeout
        return ws

    monkeypatch.setattr(slng.tts.TTS, "_connect_ws", fake_connect)
    tts = slng.TTS(api_key="test-key", model="deepgram/aura:2", **tts_kwargs)
    stream = tts.stream(conn_options=APIConnectOptions(max_retry=0))
    stream.push_text(text)
    stream.end_input()
    async for _ in stream:
        pass
    await stream.aclose()
    return [f["text"] for f in ws.sent if f.get("type") == "text"]


@pytest.mark.asyncio
async def test_tts_phrase_batching_flushes_on_punctuation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    texts = await _sent_text_frames(
        monkeypatch,
        "Hello there my friend, how are you doing today?",
        voice="aura-2-thalia-en",
    )
    assert texts == ["Hello there my friend, ", "how are you doing today? "]


@pytest.mark.asyncio
async def test_tts_word_chunking_merges_letterless_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    texts = await _sent_text_frames(
        monkeypatch,
        "Costs 4.5 million dollars .",
        voice="aura-2-thalia-en",
        text_chunking="word",
    )
    assert texts == ["Costs 4.5 ", "million ", "dollars . "]
    assert all(any(ch.isalpha() for ch in t) for t in texts)


@pytest.mark.asyncio
async def test_tts_fallback_stream_supports_flush() -> None:
    tts = slng.TTS(
        api_key="test-key",
        voice="voice-a",
        connections=["provider/model:1", "provider/model:2"],
    )
    first, second = tts._candidate_tts
    calls: list[str] = []
    audio = object()

    class _RecordingStream(_FakeStream):
        def push_text(self, text: str) -> None:
            calls.append(f"push:{text}")

        def flush(self) -> None:
            calls.append("flush")

        def end_input(self) -> None:
            calls.append("end")

    def first_stream(self, *, conn_options):
        del self, conn_options
        return _FakeStream([APIStatusError("failed")])

    def second_stream(self, *, conn_options):
        del self, conn_options
        return _RecordingStream([audio])

    first._stream_candidate = MethodType(first_stream, first)  # type: ignore[method-assign]
    second._stream_candidate = MethodType(second_stream, second)  # type: ignore[method-assign]

    stream = tts.stream(conn_options=APIConnectOptions(max_retry=0))
    await stream.__aenter__()
    stream.push_text("hello")
    stream.flush()
    stream.end_input()
    assert await stream.__anext__() is audio
    assert calls == ["push:hello", "flush", "end"]
    await stream.aclose()


@pytest.mark.asyncio
async def test_tts_fallback_stream_supports_plain_async_for() -> None:
    tts = slng.TTS(
        api_key="test-key",
        voice="voice-a",
        connections=["provider/model:1", "provider/model:2"],
    )
    first, second = tts._candidate_tts
    audio = object()

    def first_stream(self, *, conn_options):
        del self, conn_options
        return _FakeStream([APIStatusError("failed")])

    def second_stream(self, *, conn_options):
        del self, conn_options
        return _FakeStream([audio])

    first._stream_candidate = MethodType(first_stream, first)  # type: ignore[method-assign]
    second._stream_candidate = MethodType(second_stream, second)  # type: ignore[method-assign]

    stream = tts.stream(conn_options=APIConnectOptions(max_retry=0))
    stream.push_text("hello")
    stream.end_input()
    items = [item async for item in stream]
    assert items == [audio]
    await stream.aclose()


@pytest.mark.asyncio
async def test_tts_fallback_chunked_supports_collect() -> None:
    tts = slng.TTS(
        api_key="test-key",
        voice="voice-a",
        connections=["provider/model:1", "provider/model:2"],
    )
    first, second = tts._candidate_tts
    frame = rtc.AudioFrame(
        data=b"\x00\x00" * 240,
        sample_rate=24000,
        num_channels=1,
        samples_per_channel=240,
    )

    def first_synthesize(self, text, *, conn_options):
        del self, text, conn_options
        return _FakeStream([APIStatusError("failed")])

    def second_synthesize(self, text, *, conn_options):
        del self, text, conn_options
        return _FakeStream([SimpleNamespace(frame=frame)])

    first._synthesize_candidate = MethodType(  # type: ignore[method-assign]
        first_synthesize, first
    )
    second._synthesize_candidate = MethodType(  # type: ignore[method-assign]
        second_synthesize, second
    )

    stream = tts.synthesize("hello", conn_options=APIConnectOptions(max_retry=0))
    combined = await stream.collect()
    assert combined.samples_per_channel == 240
    await stream.aclose()
