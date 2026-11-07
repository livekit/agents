"""Lokutor STT and TTS plugin tests: offline, against a local fake of the API (``WS /ws/stt``,
``WS /ws/tts``, ``POST /stt/transcribe``).

Async tests run by pytest-asyncio (``asyncio_mode = "auto"`` in the repo's pytest config).
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import wave
from typing import Any
from unittest.mock import patch

import aiohttp
import pytest
from aiohttp import web

from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    stt,
)

pytestmark = pytest.mark.unit


def _frame(ms: int = 100, rate: int = 16000) -> rtc.AudioFrame:
    n = rate * ms // 1000
    return rtc.AudioFrame(b"\x10\x00" * n, sample_rate=rate, num_channels=1, samples_per_channel=n)


async def _serve(app: web.Application) -> tuple[web.AppRunner, str]:
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    return runner, f"ws://127.0.0.1:{port}"


def test_stt_requires_api_key():
    from livekit.plugins.lokutor import STT

    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="API key"):
            STT(api_key=None)


def test_stt_properties():
    from livekit.plugins.lokutor import STT

    s = STT(api_key="test-key")
    assert s.model == "conv-2.0"
    assert s.provider == "Lokutor"
    assert s.capabilities.streaming is True
    assert s.capabilities.interim_results is True
    assert s.capabilities.offline_recognize is True


def test_api_language_maps_to_the_nine_and_detects_the_rest():
    from livekit.plugins.lokutor.stt import _api_language

    assert _api_language("en-US") == "en"
    assert _api_language("es") == "es"
    assert _api_language("ca") == "ca"
    assert _api_language("eu") == "eu"
    assert _api_language("ja") == ""  # not served: detected, not forced
    assert _api_language(None) == ""
    assert _api_language("") == ""


async def test_stream_turns_partials_and_finals_into_speech_events():
    seen: dict[str, Any] = {"audio": 0}

    async def ws_stt(request: web.Request) -> web.WebSocketResponse:
        seen["key"] = request.headers.get("X-API-Key")
        seen["query"] = dict(request.query)
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        first = await ws.receive()
        seen["config"] = json.loads(first.data)
        await ws.send_str(json.dumps({"type": "status", "data": "ready", "lang": "es"}))
        sent = False
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.BINARY:
                seen["audio"] += len(msg.data)
                if not sent and seen["audio"] >= 16000:  # ~0.5 s in
                    await ws.send_str(
                        json.dumps({"type": "transcript", "data": "hola", "isFinal": False})
                    )
                    await ws.send_str(
                        json.dumps(
                            {"type": "transcript", "data": "hola, bon dia", "isFinal": False}
                        )
                    )
                    sent = True
            elif msg.type == aiohttp.WSMsgType.TEXT and json.loads(msg.data).get("type") == "end":
                await ws.send_str(
                    json.dumps({"type": "transcript", "data": "Hola, bon dia.", "isFinal": True})
                )
                await ws.send_str(json.dumps({"type": "status", "data": "done"}))
                await ws.close()
        return ws

    async def run() -> list[stt.SpeechEvent]:
        from livekit.plugins.lokutor import STT

        app = web.Application()
        app.router.add_get("/ws/stt", ws_stt)
        runner, base = await _serve(app)
        try:
            async with aiohttp.ClientSession() as session:
                s = STT(api_key="sk-test", base_url=base, language="es-ES", http_session=session)
                stream = s.stream(conn_options=APIConnectOptions(max_retry=0, timeout=5))
                for _ in range(10):
                    stream.push_frame(_frame())
                stream.end_input()
                events = [ev async for ev in stream]
                await stream.aclose()
                return events
        finally:
            await runner.cleanup()

    events = await run()
    kinds = [e.type for e in events]
    assert kinds == [
        stt.SpeechEventType.START_OF_SPEECH,
        stt.SpeechEventType.INTERIM_TRANSCRIPT,
        stt.SpeechEventType.INTERIM_TRANSCRIPT,
        stt.SpeechEventType.FINAL_TRANSCRIPT,
        stt.SpeechEventType.END_OF_SPEECH,
        stt.SpeechEventType.RECOGNITION_USAGE,
    ]
    final = events[3]
    assert final.alternatives[0].text == "Hola, bon dia."
    assert final.alternatives[0].language == "es"
    assert final.request_id == events[1].request_id == events[4].request_id
    # Times are audio positions from the start of the stream: the utterance opens at 0 and the final
    # one ends where the audio ended.
    assert final.alternatives[0].start_time == 0.0
    assert final.alternatives[0].end_time == pytest.approx(1.0, abs=0.01)
    assert 0.0 < events[1].alternatives[0].end_time <= final.alternatives[0].end_time
    usage = events[5].recognition_usage
    assert usage is not None and usage.audio_duration == pytest.approx(1.0, abs=0.01)
    # The key travels in a header, never the URL; the language as the API names it.
    assert seen["key"] == "sk-test"
    assert "api_key" not in seen["query"]
    assert seen["config"] == {"language": "es"}
    assert seen["audio"] == 32000  # 1 s of 16 kHz PCM16


async def test_stream_with_an_unserved_language_asks_for_detection():
    seen: dict[str, Any] = {}

    async def ws_stt(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        seen["config"] = json.loads((await ws.receive()).data)
        await ws.send_str(json.dumps({"type": "status", "data": "ready"}))
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                await ws.send_str(json.dumps({"type": "status", "data": "done"}))
                await ws.close()
        return ws

    async def run() -> None:
        from livekit.plugins.lokutor import STT

        app = web.Application()
        app.router.add_get("/ws/stt", ws_stt)
        runner, base = await _serve(app)
        try:
            async with aiohttp.ClientSession() as session:
                s = STT(api_key="sk-test", base_url=base, http_session=session)
                stream = s.stream(language="ja", conn_options=APIConnectOptions(max_retry=0))
                stream.end_input()
                assert [ev async for ev in stream] == []
                await stream.aclose()
        finally:
            await runner.cleanup()

    await run()
    assert seen["config"] == {"language": ""}


async def test_stream_refused_at_the_handshake_raises_a_status_error():
    async def ws_stt(request: web.Request) -> web.Response:
        return web.json_response({"code": "auth.rate_limited"}, status=429)

    async def run() -> None:
        from livekit.plugins.lokutor import STT

        app = web.Application()
        app.router.add_get("/ws/stt", ws_stt)
        runner, base = await _serve(app)
        try:
            async with aiohttp.ClientSession() as session:
                s = STT(api_key="sk-test", base_url=base, http_session=session)
                stream = s.stream(conn_options=APIConnectOptions(max_retry=0, timeout=5))
                stream.end_input()
                with pytest.raises(APIStatusError) as err:
                    async for _ in stream:
                        pass
                assert err.value.status_code == 429
                await stream.aclose()
        finally:
            await runner.cleanup()

    await run()


async def test_recognize_posts_a_wav_and_returns_a_final_transcript():
    seen: dict[str, Any] = {}

    async def transcribe(request: web.Request) -> web.Response:
        seen["key"] = request.headers.get("X-API-Key")
        body = await request.json()
        seen["format"] = body["format"]
        seen["language"] = body.get("language")
        with wave.open(io.BytesIO(base64.b64decode(body["audio"]))) as w:
            seen["rate"] = w.getframerate()
            seen["frames"] = w.getnframes()
        return web.json_response({"text": " Bon dia. ", "language": "ca", "engine": "lokutor-stt"})

    async def run() -> stt.SpeechEvent:
        from livekit.plugins.lokutor import STT

        app = web.Application()
        app.router.add_post("/stt/transcribe", transcribe)
        runner, base = await _serve(app)
        try:
            async with aiohttp.ClientSession() as session:
                s = STT(api_key="sk-test", base_url=base, language="ca", http_session=session)
                return await s.recognize([_frame(500, 48000)])
        finally:
            await runner.cleanup()

    event = await run()
    assert event.type == stt.SpeechEventType.FINAL_TRANSCRIPT
    assert event.alternatives[0].text == "Bon dia."
    assert event.alternatives[0].language == "ca"
    assert event.alternatives[0].end_time == pytest.approx(0.5)
    assert seen == {
        "key": "sk-test",
        "format": "wav",
        "language": "ca",
        "rate": 48000,
        "frames": 24000,
    }


async def _stream_against(ws_stt, *, timeout: float = 5.0, **stt_kwargs) -> list[stt.SpeechEvent]:
    from livekit.plugins.lokutor import STT

    app = web.Application()
    app.router.add_get("/ws/stt", ws_stt)
    runner, base = await _serve(app)
    try:
        async with aiohttp.ClientSession() as session:
            s = STT(api_key="sk-test", base_url=base, http_session=session, **stt_kwargs)
            stream = s.stream(conn_options=APIConnectOptions(max_retry=0, timeout=timeout))
            stream.push_frame(_frame())
            stream.end_input()
            try:
                return [ev async for ev in stream]
            finally:
                await stream.aclose()
    finally:
        await runner.cleanup()


async def test_stream_closed_before_done_is_an_error():
    # After end_input() the server owes a last final transcript and "done"; closing without
    # them may have lost speech, so it must not end the stream as if it had finished.
    async def ws_stt(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.receive()
        await ws.send_str(json.dumps({"type": "status", "data": "ready"}))
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                await ws.close()
        return ws

    with pytest.raises(APIConnectionError):
        await _stream_against(ws_stt)


async def test_stream_that_stalls_after_end_times_out():
    # A server that accepts "end" and then says nothing, without closing, must not hold the
    # stream open forever.
    async def ws_stt(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.receive()
        await ws.send_str(json.dumps({"type": "status", "data": "ready"}))
        async for _ in ws:
            pass
        return ws

    with pytest.raises(APITimeoutError):
        await asyncio.wait_for(_stream_against(ws_stt, timeout=0.5), 10)


async def test_stream_without_a_language_reports_the_detected_one():
    async def ws_stt(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.receive()
        await ws.send_str(json.dumps({"type": "status", "data": "ready", "lang": ""}))
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                await ws.send_str(
                    json.dumps(
                        {"type": "transcript", "data": "Bon dia.", "isFinal": True, "lang": "ca"}
                    )
                )
                await ws.send_str(json.dumps({"type": "status", "data": "done"}))
                await ws.close()
        return ws

    events = await _stream_against(ws_stt)
    finals = [e for e in events if e.type == stt.SpeechEventType.FINAL_TRANSCRIPT]
    assert finals and finals[0].alternatives[0].language == "ca"


async def test_recognize_without_a_language_reports_the_detected_one():
    async def transcribe(request: web.Request) -> web.Response:
        return web.json_response({"text": "Bon dia.", "language": "ca"})

    from livekit.plugins.lokutor import STT

    app = web.Application()
    app.router.add_post("/stt/transcribe", transcribe)
    runner, base = await _serve(app)
    try:
        async with aiohttp.ClientSession() as session:
            s = STT(api_key="sk-test", base_url=base, http_session=session)
            event = await s.recognize([_frame(500)])
    finally:
        await runner.cleanup()
    assert event.alternatives[0].language == "ca"


# --- TTS ---
def test_tts_requires_api_key():
    from livekit.plugins.lokutor import TTS

    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="API key"):
            TTS(api_key=None)


def test_tts_accepts_api_key_directly():
    from livekit.plugins.lokutor import TTS

    tts = TTS(api_key="test-key")
    assert tts._opts.api_key == "test-key"


def test_tts_accepts_api_key_from_env():
    from livekit.plugins.lokutor import TTS

    with patch.dict("os.environ", {"LOKUTOR_API_KEY": "env-key"}):
        tts = TTS()
        assert tts._opts.api_key == "env-key"


def test_tts_default_voice():
    from livekit.plugins.lokutor import TTS

    tts = TTS(api_key="test-key")
    assert tts._opts.voice == "F1"


def test_tts_custom_voice():
    from livekit.plugins.lokutor import TTS

    tts = TTS(api_key="test-key", voice="M3")
    assert tts._opts.voice == "M3"


def test_tts_all_voices_accepted():
    from livekit.plugins.lokutor import TTS

    voices = ["F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5"]
    for voice in voices:
        tts = TTS(api_key="test-key", voice=voice)
        assert tts._opts.voice == voice


def test_tts_default_language():
    from livekit.plugins.lokutor import TTS

    tts = TTS(api_key="test-key")
    assert tts._opts.language == "en"


def test_tts_custom_language():
    from livekit.plugins.lokutor import TTS

    tts = TTS(api_key="test-key", language="es")
    assert tts._opts.language == "es"


def test_tts_none_language():
    from livekit.plugins.lokutor import TTS

    tts = TTS(api_key="test-key", language=None)
    assert tts._opts.language is None


def test_tts_sample_rate_is_fixed_at_44100():
    # Lokutor's TTS output is always 44.1 kHz; the rate is not configurable
    # (a mismatched rate would corrupt playback), so it is pinned.
    import pytest

    from livekit.plugins.lokutor import TTS

    tts = TTS(api_key="test-key")
    assert tts.sample_rate == 44100
    assert tts._opts.sample_rate == 44100
    with pytest.raises(TypeError):
        TTS(api_key="test-key", sample_rate=22050)


def test_tts_model_property():
    from livekit.plugins.lokutor import TTS

    tts = TTS(api_key="test-key")
    assert tts.model == "versa-2.0"


def test_tts_provider_property():
    from livekit.plugins.lokutor import TTS

    tts = TTS(api_key="test-key")
    assert tts.provider == "Lokutor"


def test_tts_capabilities():
    from livekit.plugins.lokutor import TTS

    tts = TTS(api_key="test-key")
    assert tts.capabilities.streaming is True
    assert tts.capabilities.aligned_transcript is False


def test_tts_num_channels():
    from livekit.plugins.lokutor import TTS

    tts = TTS(api_key="test-key")
    assert tts.num_channels == 1


def test_tts_default_base_url():
    from livekit.plugins.lokutor import TTS

    tts = TTS(api_key="test-key")
    assert tts._opts.base_url == "wss://api.lokutor.com"


def test_tts_ws_url():
    from livekit.plugins.lokutor import TTS

    tts = TTS(api_key="test-key")
    assert tts._opts.get_ws_url() == "wss://api.lokutor.com/ws/tts"


async def test_synthesize_returns_chunked_stream():
    from unittest.mock import MagicMock, patch

    from livekit.plugins.lokutor import TTS, ChunkedStream

    def _fake_create_task(coro, *args, **kwargs):
        coro.close()
        return MagicMock()

    tts = TTS(api_key="test-key")
    with patch("livekit.agents.tts.tts.asyncio.create_task", side_effect=_fake_create_task):
        stream = tts.synthesize("hello")
    assert isinstance(stream, ChunkedStream)


async def test_stream_returns_synthesize_stream():
    from unittest.mock import MagicMock, patch

    from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
    from livekit.plugins.lokutor import TTS
    from livekit.plugins.lokutor.tts import SynthesizeStream

    def _fake_create_task(coro, *args, **kwargs):
        coro.close()
        return MagicMock()

    tts = TTS(api_key="test-key")
    with patch("livekit.agents.tts.tts.asyncio.create_task", side_effect=_fake_create_task):
        stream = tts.stream(conn_options=DEFAULT_API_CONNECT_OPTIONS)
    assert isinstance(stream, SynthesizeStream)


def test_build_request_includes_language():
    from livekit.plugins.lokutor.tts import _build_request, _TTSOptions

    opts = _TTSOptions(
        api_key="key",
        voice="F1",
        language="es",
        base_url="wss://api.lokutor.com",
        sample_rate=44100,
    )
    request = _build_request(opts, "hola")
    assert request["text"] == "hola"
    assert request["voice"] == "F1"
    assert request["language"] == "es"


def test_build_request_without_language():
    from livekit.plugins.lokutor.tts import _build_request, _TTSOptions

    opts = _TTSOptions(
        api_key="key",
        voice="M1",
        language=None,
        base_url="wss://api.lokutor.com",
        sample_rate=44100,
    )
    request = _build_request(opts, "hello")
    assert request["text"] == "hello"
    assert request["voice"] == "M1"
    assert "language" not in request


# --- TTS stream: segments, sentences ---


async def _tts_server(requests: list[str]):
    """A fake ``WS /ws/tts``: for each text request, 50 ms of audio and ``EOS``."""

    async def ws_tts(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                requests.append(json.loads(msg.data)["text"])
                await ws.send_bytes(b"\x10\x00" * 2205)
                await ws.send_str("EOS")
        return ws

    app = web.Application()
    app.router.add_get("/ws/tts", ws_tts)
    return await _serve(app)


async def _synthesize_stream(
    pushes: list[str | None], *, direct: bool = False
) -> tuple[list[str], list[Any]]:
    """Push text (``None`` = flush) through a TTS stream; return the requests the server saw and the audio events.

    ``direct`` writes to the stream's input channel instead of ``push_text()``: LiveKit's base class
    drops text pushed after a flush on the same stream (one segment per instance), and the plugin
    should still handle a second segment correctly if it is ever handed one.
    """
    from livekit.plugins.lokutor import TTS

    requests: list[str] = []
    runner, base = await _tts_server(requests)
    try:
        async with aiohttp.ClientSession() as session:
            t = TTS(api_key="sk-test", base_url=base, http_session=session)
            stream = t.stream(conn_options=APIConnectOptions(max_retry=0, timeout=5.0))
            for p in pushes:
                if direct:
                    stream._input_ch.send_nowait(stream._FlushSentinel() if p is None else p)
                else:
                    stream.flush() if p is None else stream.push_text(p)
            stream.end_input()
            try:
                events = [ev async for ev in stream]
            finally:
                await stream.aclose()
            return requests, events
    finally:
        await runner.cleanup()


async def test_text_after_a_flush_opens_a_new_segment():
    # A flush ends the segment; text that follows must open a new one instead of pushing audio into
    # none (AudioEmitter: "start_segment() must be called before pushing audio data").
    requests, events = await _synthesize_stream(
        ["Hello there, friend. ", None, "How are you doing today? ", None], direct=True
    )
    assert requests == ["Hello there, friend.", "How are you doing today?"]
    assert sum(len(ev.frame.data) for ev in events) > 0
    assert len({ev.segment_id for ev in events}) == 2


async def test_a_flush_with_nothing_after_it_adds_no_segment():
    requests, events = await _synthesize_stream(["Just one sentence here.", None, None])
    assert requests == ["Just one sentence here."]
    assert len({ev.segment_id for ev in events}) == 1


async def test_llm_token_deltas_become_sentence_requests():
    # The input is LLM token deltas; each fragment used to be its own request with its own prosody.
    deltas = [
        " Hel",
        "lo th",
        "ere, my fri",
        "end. How are y",
        "ou doing tod",
        "ay? I hope you are ",
        "well.",
    ]
    requests, _ = await _synthesize_stream(deltas)
    assert len(requests) < len(deltas)
    assert all(len(r) > 12 for r in requests)
    assert " ".join(requests).replace("  ", " ") == "".join(deltas).strip()


async def test_a_stream_with_no_text_makes_no_request():
    requests, events = await _synthesize_stream([None])
    assert requests == []
    assert events == []


# --- STT stream: a reconnect mid-utterance ---


async def test_a_retry_ends_the_utterance_the_dropped_connection_left_open():
    # The first connection drops after START_OF_SPEECH and an interim; the retry must close that
    # utterance and open its own with a new request id, not continue the old one.
    from livekit.plugins.lokutor import STT

    connections = {"n": 0}

    async def ws_stt(request: web.Request) -> web.WebSocketResponse:
        connections["n"] += 1
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.receive()
        await ws.send_str(json.dumps({"type": "status", "data": "ready"}))
        if connections["n"] == 1:
            await ws.send_str(json.dumps({"type": "transcript", "data": "hola", "isFinal": False}))
            await asyncio.sleep(0.05)
            await ws.close()
            return ws
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT and json.loads(msg.data).get("type") == "end":
                await ws.send_str(
                    json.dumps({"type": "transcript", "data": "Adiós.", "isFinal": True})
                )
                await ws.send_str(json.dumps({"type": "status", "data": "done"}))
                await ws.close()
        return ws

    app = web.Application()
    app.router.add_get("/ws/stt", ws_stt)
    runner, base = await _serve(app)
    try:
        async with aiohttp.ClientSession() as session:
            s = STT(api_key="sk-test", base_url=base, http_session=session)
            stream = s.stream(
                conn_options=APIConnectOptions(max_retry=1, retry_interval=0.01, timeout=5.0)
            )
            stream.push_frame(_frame())
            stream.end_input()
            try:
                events = [ev async for ev in stream]
            finally:
                await stream.aclose()
    finally:
        await runner.cleanup()

    kinds = [e.type for e in events if e.type != stt.SpeechEventType.RECOGNITION_USAGE]
    T = stt.SpeechEventType
    assert kinds == [
        T.START_OF_SPEECH,
        T.INTERIM_TRANSCRIPT,
        T.END_OF_SPEECH,
        T.START_OF_SPEECH,
        T.FINAL_TRANSCRIPT,
        T.END_OF_SPEECH,
    ]
    ids = [e.request_id for e in events if e.type in (T.INTERIM_TRANSCRIPT, T.FINAL_TRANSCRIPT)]
    assert ids[0] != ids[1]


def test_build_request_normalizes_language():
    from livekit.plugins.lokutor.tts import _build_request, _TTSOptions

    opts = _TTSOptions(
        api_key="key",
        voice="F1",
        language="es-ES",
        base_url="wss://api.lokutor.com",
        sample_rate=44100,
    )
    request = _build_request(opts, "hola")
    assert request["language"] == "es"
    assert set(request) == {"text", "voice", "language"}
