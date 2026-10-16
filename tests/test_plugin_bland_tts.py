from __future__ import annotations

import asyncio
import json
import struct
import time
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock, patch

import aiohttp
import pytest
from aiohttp import web

from livekit.agents import APIConnectionError, APIConnectOptions, APIError, APIStatusError
from livekit.agents.types import NOT_GIVEN

pytestmark = pytest.mark.plugin("bland")


def _pcm(num_samples: int) -> bytes:
    """Bare little-endian int16 PCM, which is what ``container: raw`` returns."""
    return struct.pack(f"<{num_samples}h", *(((i * 97) % 2000) - 1000 for i in range(num_samples)))


class _Server:
    """Local ``/v2/tts`` stand-in so the response path runs end to end."""

    def __init__(self, handler: Callable) -> None:
        self._handler = handler

    async def __aenter__(self) -> _Server:
        app = web.Application()
        app.router.add_post("/v2/tts", self._handler)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await self._site.start()
        port = self._runner.addresses[0][1]
        self.base_url = f"http://127.0.0.1:{port}/v2"
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.session.close()
        await self._runner.cleanup()


def _assert_audio_is(audio: bytes, expected: bytes) -> None:
    """The emitter packs fixed-size frames, so the final frame is zero-padded. Every
    input sample must appear unchanged, and the only surplus may be that padding."""
    assert audio[: len(expected)] == expected
    assert set(audio[len(expected) :]) <= {0}


async def _collect(tts, text: str = "hello world", **conn: Any) -> tuple[bytes, list[int]]:
    """Drain a synthesis into its concatenated audio bytes and per-frame sample rates."""
    options = APIConnectOptions(max_retry=0, timeout=5, **conn)
    stream = tts.synthesize(text, conn_options=options)
    audio, rates = bytearray(), []
    try:
        async for ev in stream:
            audio.extend(ev.frame.data.tobytes())
            rates.append(ev.frame.sample_rate)
    finally:
        await stream.aclose()
    return bytes(audio), rates


def test_requires_api_key():
    from livekit.plugins.bland import TTS

    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="API key"):
            TTS()


def test_api_key_from_argument():
    from livekit.plugins.bland import TTS

    assert TTS(api_key="test-key")._api_key == "test-key"


def test_api_key_from_env():
    from livekit.plugins.bland import TTS

    with patch.dict("os.environ", {"BLAND_API_KEY": "env-key"}):
        assert TTS()._api_key == "env-key"


def test_provider_property():
    from livekit.plugins.bland import TTS

    assert TTS(api_key="test-key").provider == "Bland"


def test_streaming():
    from livekit.plugins.bland import TTS

    assert TTS(api_key="test-key").capabilities.streaming is True


def test_default_voice_id():
    from livekit.plugins.bland import TTS
    from livekit.plugins.bland.tts import DEFAULT_VOICE_ID

    assert TTS(api_key="test-key")._opts.voice_id == DEFAULT_VOICE_ID


def test_custom_voice_id():
    from livekit.plugins.bland import TTS

    tts = TTS(api_key="test-key", voice_id="c18a1cd5-91ef-4b06-841a-e58b8b487e8c")
    assert tts._opts.voice_id == "c18a1cd5-91ef-4b06-841a-e58b8b487e8c"


def test_default_sample_rate_is_btts_v3_native():
    from livekit.plugins.bland import TTS

    assert TTS(api_key="test-key").sample_rate == 48000


@pytest.mark.parametrize("rate", [8000, 16000, 24000, 44100, 48000])
def test_supported_sample_rates(rate: int):
    from livekit.plugins.bland import TTS

    assert TTS(api_key="test-key", sample_rate=rate).sample_rate == rate


@pytest.mark.parametrize("rate", [12345, 22050, 0, 96000])
def test_rejects_unsupported_sample_rate(rate: int):
    from livekit.plugins.bland import TTS

    with pytest.raises(ValueError, match="sample_rate must be one of"):
        TTS(api_key="test-key", sample_rate=rate)


def test_controls_default_to_unset():
    from livekit.plugins.bland import TTS

    tts = TTS(api_key="test-key")
    assert tts._opts.expressiveness is NOT_GIVEN
    assert tts._opts.stability is NOT_GIVEN


def test_update_options():
    from livekit.plugins.bland import TTS

    tts = TTS(api_key="test-key")
    tts.update_options(
        voice_id="c18a1cd5-91ef-4b06-841a-e58b8b487e8c", expressiveness=0.9, stability=0.4
    )
    assert tts._opts.voice_id == "c18a1cd5-91ef-4b06-841a-e58b8b487e8c"
    assert tts._opts.expressiveness == 0.9
    assert tts._opts.stability == 0.4


def test_update_options_preserves_unset_fields():
    from livekit.plugins.bland import TTS
    from livekit.plugins.bland.tts import DEFAULT_VOICE_ID

    tts = TTS(api_key="test-key")
    tts.update_options(stability=0.25)
    assert tts._opts.voice_id == DEFAULT_VOICE_ID
    assert tts._opts.expressiveness is NOT_GIVEN
    assert tts._opts.stability == 0.25


def test_synthesize_returns_chunked_stream():
    from livekit.plugins.bland import TTS
    from livekit.plugins.bland.tts import ChunkedStream

    def _fake_create_task(coro, *args, **kwargs):
        coro.close()
        return MagicMock()

    tts = TTS(api_key="test-key")
    with patch("livekit.agents.tts.tts.asyncio.create_task", side_effect=_fake_create_task):
        stream = tts.synthesize("hello")
    assert isinstance(stream, ChunkedStream)


def _patch_session_capture_post(tts, captured: dict[str, Any]) -> None:
    class _FakePostCM:
        async def __aenter__(self):
            raise RuntimeError("short-circuit")

        async def __aexit__(self, *exc):
            return None

    def _fake_post(url, *, headers=None, json=None, **kwargs):
        captured.update(url=url, headers=headers, json=json)
        return _FakePostCM()

    fake_session = MagicMock()
    fake_session.post = _fake_post
    tts._session = fake_session


async def _synthesize_and_capture(tts) -> dict[str, Any]:
    captured: dict[str, Any] = {}
    _patch_session_capture_post(tts, captured)
    with pytest.raises(APIConnectionError):
        async for _ in tts.synthesize("hello world"):
            pass
    return captured


async def test_request_url_and_auth_header():
    from livekit.plugins.bland import TTS

    captured = await _synthesize_and_capture(TTS(api_key="test-key"))

    assert captured["url"] == "https://api.bland.ai/v2/tts"
    assert captured["headers"]["authorization"] == "test-key"
    assert captured["headers"]["content-type"] == "application/json"


async def test_trailing_slash_is_normalized_for_http_requests():
    from livekit.plugins.bland import TTS

    captured = await _synthesize_and_capture(
        TTS(api_key="test-key", base_url="https://api.bland.ai/v2/")
    )

    assert captured["url"] == "https://api.bland.ai/v2/tts"


async def test_request_body_defaults():
    from livekit.plugins.bland import TTS
    from livekit.plugins.bland.tts import DEFAULT_VOICE_ID

    body = (await _synthesize_and_capture(TTS(api_key="test-key")))["json"]

    assert body["text"] == "hello world"
    assert body["voice"] == DEFAULT_VOICE_ID
    assert body["audio"] == {"encoding": "pcm_s16le", "sample_rate": 48000}
    assert "controls" not in body
    # fields the request shape does not define
    assert "language" not in body
    assert "output_format" not in body
    assert "voice_id" not in body


async def test_request_body_sample_rate_follows_option():
    from livekit.plugins.bland import TTS

    body = (await _synthesize_and_capture(TTS(api_key="test-key", sample_rate=24000)))["json"]

    assert body["audio"]["sample_rate"] == 24000


async def test_request_body_includes_controls():
    from livekit.plugins.bland import TTS

    tts = TTS(api_key="test-key", expressiveness=0.9, stability=0.4)
    body = (await _synthesize_and_capture(tts))["json"]

    assert body["controls"] == {"expressiveness": 0.9, "stability": 0.4}


async def test_partial_controls_send_only_what_was_set():
    from livekit.plugins.bland import TTS

    body = (await _synthesize_and_capture(TTS(api_key="test-key", stability=0.4)))["json"]

    assert body["controls"] == {"stability": 0.4}


async def test_emits_bare_pcm_unchanged():
    """The bytes on the wire are already frame-ready, so every sample must survive."""
    from livekit.plugins.bland import TTS

    payload = _pcm(4800)

    async def handler(request):
        return web.Response(body=payload, content_type="audio/pcm")

    async with _Server(handler) as srv:
        tts = TTS(api_key="k", base_url=srv.base_url)
        tts._session = srv.session
        audio, rates = await _collect(tts)

    _assert_audio_is(audio, payload)
    assert not audio.startswith(b"RIFF")
    assert set(rates) == {48000}


async def test_reassembles_audio_split_across_chunks():
    """A split at an odd byte lands mid-sample; nothing may be dropped or reordered."""
    from livekit.plugins.bland import TTS

    payload = _pcm(4800)
    splits = [1, 3, 1000, 2001, len(payload)]

    async def handler(request):
        resp = web.StreamResponse(headers={"content-type": "audio/pcm"})
        await resp.prepare(request)
        start = 0
        for end in splits:
            await resp.write(payload[start:end])
            start = end
        await resp.write_eof()
        return resp

    async with _Server(handler) as srv:
        tts = TTS(api_key="k", base_url=srv.base_url)
        tts._session = srv.session
        audio, _ = await _collect(tts)

    _assert_audio_is(audio, payload)


async def test_honors_requested_sample_rate_end_to_end():
    from livekit.plugins.bland import TTS

    payload = _pcm(2400)
    seen: dict[str, Any] = {}

    async def handler(request):
        seen["body"] = await request.json()
        return web.Response(body=payload, content_type="audio/pcm")

    async with _Server(handler) as srv:
        tts = TTS(api_key="k", base_url=srv.base_url, sample_rate=24000)
        tts._session = srv.session
        audio, rates = await _collect(tts)

    assert seen["body"]["audio"]["sample_rate"] == 24000
    assert set(rates) == {24000}
    _assert_audio_is(audio, payload)


async def test_error_envelope_becomes_api_status_error():
    from livekit.plugins.bland import TTS

    async def handler(request):
        return web.json_response(
            {"error": {"code": "voice_not_found", "message": "Voice was not found."}},
            status=404,
        )

    async with _Server(handler) as srv:
        tts = TTS(api_key="k", base_url=srv.base_url)
        tts._session = srv.session
        with pytest.raises(APIStatusError) as excinfo:
            await _collect(tts)

    assert excinfo.value.status_code == 404
    assert "voice_not_found" in str(excinfo.value)
    assert "Voice was not found." in str(excinfo.value)


async def test_non_json_error_body_still_raises():
    from livekit.plugins.bland import TTS

    async def handler(request):
        return web.Response(body=b"<html>gateway</html>", status=502, content_type="text/html")

    async with _Server(handler) as srv:
        tts = TTS(api_key="k", base_url=srv.base_url)
        tts._session = srv.session
        with pytest.raises(APIStatusError) as excinfo:
            await _collect(tts)

    assert excinfo.value.status_code == 502


# --- /v2/tts/ws ----------------------------------------------------------------------


class _WSServer:
    """Local ``/v2/tts/ws`` stand-in speaking the real turn protocol."""

    def __init__(
        self,
        *,
        frames: list[bytes] | None = None,
        init_error: dict[str, Any] | None = None,
        turn_error: dict[str, Any] | None = None,
        end_reason: str = "complete",
        stale_terminator: bool = False,
        ready_encoding: str = "pcm_s16le",
        ready_sample_rate: int | None = None,
        handshake_status: int | None = None,
        acknowledge_cancel: bool = True,
    ) -> None:
        self._frames = frames if frames is not None else [_pcm(480)]
        self._init_error = init_error
        self._turn_error = turn_error
        self._end_reason = end_reason
        self._stale_terminator = stale_terminator
        self._ready_encoding = ready_encoding
        self._ready_sample_rate = ready_sample_rate
        self._handshake_status = handshake_status
        self._acknowledge_cancel = acknowledge_cancel
        self.received: list[dict[str, Any]] = []
        self.headers: dict[str, str] = {}
        self.sessions = 0
        self.started_contexts: set[str] = set()
        self.speak_received = asyncio.Event()
        self.cancel_received = asyncio.Event()
        self.connection_closed = asyncio.Event()

    async def _handle(self, request: web.Request) -> web.StreamResponse:
        if self._handshake_status is not None:
            return web.json_response(
                {"type": "error", "code": "AUTH_REQUIRED"},
                status=self._handshake_status,
            )
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.headers = dict(request.headers)
        self.sessions += 1

        async for msg in ws:
            if msg.type is not aiohttp.WSMsgType.TEXT:
                continue
            message = json.loads(msg.data)
            self.received.append(message)

            if message["type"] == "init":
                if self._init_error is not None:
                    await ws.send_json({"type": "error", **self._init_error})
                    await ws.close()
                    return ws
                await ws.send_json(
                    {
                        "type": "ready",
                        "session_id": "test-session",
                        "encoding": self._ready_encoding,
                        "sample_rate": (
                            self._ready_sample_rate
                            if self._ready_sample_rate is not None
                            else message.get("audio", {}).get("sample_rate", 48000)
                        ),
                    }
                )
            elif message["type"] == "speak":
                context_id = message["context_id"]
                self.speak_received.set()
                if self._turn_error is not None:
                    await ws.send_json(
                        {"type": "error", "context_id": context_id, **self._turn_error}
                    )
                    continue
                if context_id not in self.started_contexts:
                    self.started_contexts.add(context_id)
                    await ws.send_json({"type": "utterance_start", "context_id": context_id})
            elif message["type"] == "end_of_turn":
                context_id = message["context_id"]
                if self._turn_error is not None:
                    continue
                for frame in self._frames:
                    await ws.send_bytes(frame)
                if self._stale_terminator:
                    await ws.send_json(
                        {
                            "type": "utterance_end",
                            "context_id": "someone-else",
                            "reason": "cancelled",
                        }
                    )
                await ws.send_json(
                    {
                        "type": "utterance_end",
                        "context_id": context_id,
                        "reason": self._end_reason,
                        "frames": len(self._frames),
                        "duration_ms": 40 * len(self._frames),
                    }
                )
            elif message["type"] == "cancel":
                self.cancel_received.set()
                if not self._acknowledge_cancel:
                    continue
                await ws.send_json(
                    {
                        "type": "utterance_end",
                        "context_id": message["context_id"],
                        "reason": "cancelled",
                        "frames": 0,
                        "duration_ms": 0,
                    }
                )
            elif message["type"] == "close":
                await ws.send_json({"type": "done", "session_id": "test-session"})
                await ws.close()

        self.connection_closed.set()
        return ws

    async def __aenter__(self) -> _WSServer:
        app = web.Application()
        app.router.add_get("/v2/tts/ws", self._handle)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await self._site.start()
        port = self._runner.addresses[0][1]
        self.base_url = f"http://127.0.0.1:{port}/v2"
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.session.close()
        await self._runner.cleanup()

    def of_type(self, type: str) -> list[dict[str, Any]]:
        return [m for m in self.received if m["type"] == type]


async def _stream_turn(tts, tokens: list[str], **conn: Any) -> bytes:
    """Push one turn's text deltas through a stream and drain the resulting audio.

    One stream is one segment is one turn, so several turns means several streams.
    """
    options = APIConnectOptions(max_retry=0, timeout=5, **conn)
    stream = tts.stream(conn_options=options)
    for text in tokens:
        stream.push_text(text)
    stream.end_input()

    audio = bytearray()
    try:
        async for ev in stream:
            audio.extend(ev.frame.data.tobytes())
    finally:
        await stream.aclose()
    return bytes(audio)


def test_ws_url_derives_from_base_url():
    from livekit.plugins.bland.tts import DEFAULT_BASE_URL, _ws_url

    assert _ws_url(DEFAULT_BASE_URL) == "wss://api.bland.ai/v2/tts/ws"
    assert _ws_url("http://127.0.0.1:8080/v2") == "ws://127.0.0.1:8080/v2/tts/ws"
    assert _ws_url("http://127.0.0.1:8080/v2/") == "ws://127.0.0.1:8080/v2/tts/ws"


def test_stream_returns_synthesize_stream():
    from livekit.plugins.bland import TTS
    from livekit.plugins.bland.tts import SynthesizeStream

    def _fake_create_task(coro, *args, **kwargs):
        coro.close()
        return MagicMock()

    tts = TTS(api_key="test-key")
    with patch("livekit.agents.tts.tts.asyncio.create_task", side_effect=_fake_create_task):
        stream = tts.stream()
    assert isinstance(stream, SynthesizeStream)


async def test_init_sent_once_with_voice_and_format():
    from livekit.plugins.bland import TTS
    from livekit.plugins.bland.tts import DEFAULT_VOICE_ID

    async with _WSServer() as srv:
        tts = TTS(api_key="test-key", base_url=srv.base_url)
        tts._session = srv.session
        await _stream_turn(tts, ["hello world"])
        await tts.aclose()

    inits = srv.of_type("init")
    assert len(inits) == 1
    assert inits[0]["voice"] == DEFAULT_VOICE_ID
    assert inits[0]["audio"] == {"encoding": "pcm_s16le", "sample_rate": 48000}
    assert "controls" not in inits[0]
    assert srv.headers["Authorization"] == "Bearer test-key"


async def test_init_carries_controls_when_set():
    from livekit.plugins.bland import TTS

    async with _WSServer() as srv:
        tts = TTS(api_key="k", base_url=srv.base_url, expressiveness=0.9, stability=0.4)
        tts._session = srv.session
        await _stream_turn(tts, ["hello"])
        await tts.aclose()

    assert srv.of_type("init")[0]["controls"] == {"expressiveness": 0.9, "stability": 0.4}


async def test_tokens_stream_through_verbatim():
    """Bland picks its own synthesis boundaries, so deltas must not be re-tokenized."""
    from livekit.plugins.bland import TTS

    tokens = ["The weather", " is clear", " today."]

    async with _WSServer() as srv:
        tts = TTS(api_key="k", base_url=srv.base_url)
        tts._session = srv.session
        await _stream_turn(tts, tokens)
        await tts.aclose()

    assert [m["text"] for m in srv.of_type("speak")] == tokens


async def test_empty_stream_completes_without_creating_a_turn():
    from livekit.plugins.bland import TTS

    async with _WSServer() as srv:
        tts = TTS(api_key="k", base_url=srv.base_url)
        tts._session = srv.session
        assert await _stream_turn(tts, []) == b""
        await tts.aclose()

    assert srv.of_type("speak") == []
    assert srv.of_type("end_of_turn") == []


async def test_each_turn_gets_its_own_context_id():
    from livekit.plugins.bland import TTS

    async with _WSServer() as srv:
        tts = TTS(api_key="k", base_url=srv.base_url)
        tts._session = srv.session
        await _stream_turn(tts, ["first."])
        await _stream_turn(tts, ["second."])
        await tts.aclose()

    speaks = srv.of_type("speak")
    ends = srv.of_type("end_of_turn")

    assert [m["text"] for m in speaks] == ["first.", "second."]
    assert speaks[0]["context_id"] != speaks[1]["context_id"]
    # each turn is terminated under the id its deltas were sent with
    assert [m["context_id"] for m in ends] == [m["context_id"] for m in speaks]


async def test_binary_frames_reach_the_pipeline_unchanged():
    from livekit.plugins.bland import TTS

    frames = [_pcm(480), _pcm(480), _pcm(240)]

    async with _WSServer(frames=frames) as srv:
        tts = TTS(api_key="k", base_url=srv.base_url)
        tts._session = srv.session
        audio = await _stream_turn(tts, ["hello"])
        await tts.aclose()

    _assert_audio_is(audio, b"".join(frames))


async def test_session_is_reused_across_turns():
    from livekit.plugins.bland import TTS

    async with _WSServer() as srv:
        tts = TTS(api_key="k", base_url=srv.base_url)
        tts._session = srv.session
        for token in ("one.", "two.", "three."):
            await _stream_turn(tts, [token])
        await tts.aclose()

    assert srv.sessions == 1
    assert len(srv.of_type("init")) == 1


async def test_close_settles_the_session():
    from livekit.plugins.bland import TTS

    async with _WSServer() as srv:
        tts = TTS(api_key="k", base_url=srv.base_url)
        tts._session = srv.session
        await _stream_turn(tts, ["hello"])
        await tts.aclose()

    assert len(srv.of_type("close")) == 1


async def test_abandoned_stream_cancels_turn_and_reuses_connection():
    from livekit.plugins.bland import TTS

    async with _WSServer() as srv:
        tts = TTS(api_key="k", base_url=srv.base_url)
        tts._session = srv.session
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=5))
        stream.push_text("hello")
        await asyncio.wait_for(srv.speak_received.wait(), timeout=1)
        first_context = srv.of_type("speak")[0]["context_id"]
        await stream.aclose()
        await asyncio.wait_for(srv.cancel_received.wait(), timeout=1)
        await _stream_turn(tts, ["replacement"])
        await tts.aclose()

    assert srv.of_type("cancel") == [{"type": "cancel", "context_id": first_context}]
    assert srv.sessions == 1


async def test_unacknowledged_cancel_closes_connection_within_turn_timeout():
    from livekit.plugins.bland import TTS

    async with _WSServer(acknowledge_cancel=False) as srv:
        tts = TTS(api_key="k", base_url=srv.base_url)
        tts._session = srv.session
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=0.05))
        stream.push_text("hello")
        await asyncio.wait_for(srv.speak_received.wait(), timeout=1)
        await stream.aclose()
        await asyncio.wait_for(srv.connection_closed.wait(), timeout=1)
        await tts.aclose()


async def test_stale_utterance_end_does_not_end_the_turn():
    from livekit.plugins.bland import TTS

    frames = [_pcm(480), _pcm(480)]

    async with _WSServer(frames=frames, stale_terminator=True) as srv:
        tts = TTS(api_key="k", base_url=srv.base_url)
        tts._session = srv.session
        audio = await _stream_turn(tts, ["hello"])
        await tts.aclose()

    _assert_audio_is(audio, b"".join(frames))


async def test_unfinished_turn_raises():
    from livekit.plugins.bland import TTS

    async with _WSServer(end_reason="failed") as srv:
        tts = TTS(api_key="k", base_url=srv.base_url)
        tts._session = srv.session
        with pytest.raises(APIError) as excinfo:
            await _stream_turn(tts, ["hello"])
        await tts.aclose()

    assert "failed" in str(excinfo.value)


async def test_turn_error_raises_with_code_and_message():
    from livekit.plugins.bland import TTS

    error = {"code": "insufficient_credits", "message": "Your account is out of credits."}

    async with _WSServer(turn_error=error) as srv:
        tts = TTS(api_key="k", base_url=srv.base_url)
        tts._session = srv.session
        with pytest.raises(APIError) as excinfo:
            await _stream_turn(tts, ["hello"])
        await tts.aclose()

    assert "insufficient_credits" in str(excinfo.value)
    assert "Your account is out of credits." in str(excinfo.value)
    assert excinfo.value.retryable is False


async def test_synthesis_failure_stays_retryable():
    from livekit.plugins.bland import TTS

    error = {"code": "synthesis_failed", "message": "Synthesis failed."}

    async with _WSServer(turn_error=error) as srv:
        tts = TTS(api_key="k", base_url=srv.base_url)
        tts._session = srv.session
        with pytest.raises(APIError) as excinfo:
            await _stream_turn(tts, ["hello"])
        await tts.aclose()

    assert excinfo.value.retryable is True


async def test_rejected_init_surfaces_the_reason():
    from livekit.plugins.bland import TTS

    error = {"code": "voice_not_found", "message": "Voice was not found."}

    async with _WSServer(init_error=error) as srv:
        tts = TTS(api_key="k", base_url=srv.base_url)
        tts._session = srv.session
        with pytest.raises(APIError) as excinfo:
            await _stream_turn(tts, ["hello"])
        await tts.aclose()

    assert "voice_not_found" in str(excinfo.value)


async def test_preupgrade_rejection_preserves_http_status():
    from livekit.plugins.bland import TTS

    async with _WSServer(handshake_status=401) as srv:
        tts = TTS(api_key="bad-key", base_url=srv.base_url)
        tts._session = srv.session
        with pytest.raises(APIStatusError) as excinfo:
            await _stream_turn(tts, ["hello"])
        await tts.aclose()

    assert excinfo.value.status_code == 401


@pytest.mark.parametrize(
    ("encoding", "sample_rate"),
    [("mulaw", 48000), ("pcm_s16le", 24000)],
)
async def test_ready_must_acknowledge_requested_audio_format(encoding, sample_rate):
    from livekit.plugins.bland import TTS

    async with _WSServer(
        ready_encoding=encoding,
        ready_sample_rate=sample_rate,
    ) as srv:
        tts = TTS(api_key="k", base_url=srv.base_url)
        tts._session = srv.session
        with pytest.raises(APIError, match="unexpected audio format") as excinfo:
            await _stream_turn(tts, ["hello"])
        await tts.aclose()

    assert excinfo.value.retryable is False
    assert srv.connection_closed.is_set()


async def test_update_options_invalidates_the_session():
    from livekit.plugins.bland import TTS

    async with _WSServer() as srv:
        tts = TTS(api_key="k", base_url=srv.base_url)
        tts._session = srv.session
        await _stream_turn(tts, ["hello"])
        tts.update_options(voice_id="c18a1cd5-91ef-4b06-841a-e58b8b487e8c")
        await _stream_turn(tts, ["hello again"])
        await tts.aclose()

    inits = srv.of_type("init")
    assert len(inits) == 2
    assert inits[1]["voice"] == "c18a1cd5-91ef-4b06-841a-e58b8b487e8c"


def test_update_options_without_changes_keeps_the_session():
    from livekit.plugins.bland import TTS

    tts = TTS(api_key="k")
    with patch.object(tts._pool, "invalidate") as invalidate:
        tts.update_options()
    invalidate.assert_not_called()


class _RefusingWSServer:
    """Refuses admission for the turn's context, holding the reply back until
    ``cancel`` so it is queued behind a cancelling client's receive task."""

    def __init__(self) -> None:
        self.speaks: list[dict[str, Any]] = []
        self.sessions = 0
        self.two_speaks = asyncio.Event()

    async def _handle(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.sessions += 1
        async for msg in ws:
            if msg.type is not aiohttp.WSMsgType.TEXT:
                continue
            message = json.loads(msg.data)
            if message["type"] == "init":
                await ws.send_json(
                    {
                        "type": "ready",
                        "session_id": f"session-{self.sessions}",
                        "encoding": "pcm_s16le",
                        "sample_rate": 48000,
                    }
                )
            elif message["type"] == "speak":
                self.speaks.append(message)
                if len(self.speaks) == 2:
                    self.two_speaks.set()
            elif message["type"] == "cancel":
                # One error per refused context, not per delta: the server records
                # the context it turned away and drops its later deltas silently.
                for context_id in dict.fromkeys(s["context_id"] for s in self.speaks):
                    await ws.send_json(
                        {
                            "type": "error",
                            "context_id": context_id,
                            "code": "insufficient_credits",
                            "message": "wallet depleted",
                        }
                    )
            elif message["type"] == "close":
                await ws.send_json({"type": "done", "session_id": "done"})
                await ws.close()
        return ws

    async def __aenter__(self) -> _RefusingWSServer:
        app = web.Application()
        app.router.add_get("/v2/tts/ws", self._handle)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await self._site.start()
        self.base_url = f"http://127.0.0.1:{self._runner.addresses[0][1]}/v2"
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.session.close()
        await self._runner.cleanup()


async def test_refused_turn_does_not_contaminate_the_next_one():
    """A cancelled turn whose admission was refused must not poison the pool.

    A refused context never becomes a turn, so no terminal arrives for the drain to
    stop on. Handing the socket back on the refusal instead would leave whatever the
    server still had to say queued on it, and the next turn would read someone
    else's failure as its own.
    """
    from livekit.plugins.bland import TTS

    async with _RefusingWSServer() as srv:
        tts = TTS(api_key="k", base_url=srv.base_url, http_session=srv.session)
        options = APIConnectOptions(max_retry=0, timeout=5)

        stream = tts.stream(conn_options=options)
        stream.push_text("first")
        stream.push_text(" second")
        await asyncio.wait_for(srv.two_speaks.wait(), timeout=5)
        await stream.aclose()

        replacement = tts.stream(conn_options=options)
        replacement.push_text("replacement")
        replacement.end_input()
        try:
            async for _ in replacement:
                pass
        except APIError as e:  # a fresh session may still fail, but never on stale state
            assert "insufficient_credits" not in str(e), e
        finally:
            await replacement.aclose()

        # The contaminated socket must be discarded rather than reused.
        assert srv.sessions == 2, srv.sessions
        await tts.aclose()


async def test_failed_cancel_stays_a_cancellation():
    """A barge-in whose cleanup fails must not come back as a retryable error.

    `cancel_and_drain` can time out or raise. If that exception escapes instead of
    the original `CancelledError`, the framework sees a retryable API error and
    replays the buffered text: the caller hears the interrupted sentence a second
    time, and when the cancel came from `aclose()` before `end_input()`, the replay
    waits forever on an input channel nothing will close.
    """
    from livekit.plugins.bland import TTS

    async with _WSServer(acknowledge_cancel=False) as srv:
        tts = TTS(api_key="k", base_url=srv.base_url)
        tts._session = srv.session
        # Retries enabled, unlike the sibling test above — that is what makes a
        # swallowed cancellation observable.
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=3, timeout=0.05))
        stream.push_text("hello")
        await asyncio.wait_for(srv.speak_received.wait(), timeout=1)

        # Must return: a cancellation that turned into a retry would hang here.
        await asyncio.wait_for(stream.aclose(), timeout=5)
        await tts.aclose()

    # The interrupted text must not be spoken again.
    assert [m["text"] for m in srv.of_type("speak")] == ["hello"]


async def test_barge_in_does_not_wait_out_the_connect_budget():
    """A socket that never answers `cancel` must not hold up the next turn.

    Teardown drains the cancelled turn so the session can be reused, but a barge-in
    is waiting on it: the user has already started talking and the agent's reply is
    queued behind this. Bounding the drain by the connect timeout put a
    conversational pause at the mercy of a socket that had stopped answering.
    """
    from livekit.plugins.bland import TTS
    from livekit.plugins.bland.tts import _CANCEL_DRAIN_TIMEOUT

    async with _WSServer(acknowledge_cancel=False) as srv:
        tts = TTS(api_key="k", base_url=srv.base_url)
        tts._session = srv.session
        # A generous connect budget, which is what the bound used to be taken from.
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=30))
        stream.push_text("hello")
        await asyncio.wait_for(srv.speak_received.wait(), timeout=1)

        started = time.monotonic()
        await stream.aclose()
        elapsed = time.monotonic() - started
        await tts.aclose()

    assert elapsed < _CANCEL_DRAIN_TIMEOUT * 2, elapsed


def test_streaming_can_be_disabled():
    from livekit.plugins.bland import TTS

    tts = TTS(api_key="k", streaming=False)

    assert tts.capabilities.streaming is False
    # No pooled session, so no concurrency slot is held for a pipeline that only
    # ever synthesizes complete strings.
    assert tts._pool is None


def test_disabled_streaming_refuses_to_open_a_stream():
    from livekit.plugins.bland import TTS

    tts = TTS(api_key="k", streaming=False)

    with pytest.raises(RuntimeError, match="streaming is disabled"):
        tts.stream()


def test_disabled_streaming_leaves_the_http_path_alone():
    from livekit.plugins.bland import TTS
    from livekit.plugins.bland.tts import ChunkedStream

    def _fake_create_task(coro, *args, **kwargs):
        coro.close()
        return MagicMock()

    tts = TTS(api_key="k", streaming=False)
    with patch("livekit.agents.tts.tts.asyncio.create_task", side_effect=_fake_create_task):
        assert isinstance(tts.synthesize("hello"), ChunkedStream)


async def test_disabled_streaming_teardown_is_a_noop():
    from livekit.plugins.bland import TTS

    tts = TTS(api_key="k", streaming=False)
    tts.prewarm()  # nothing to warm
    await tts.aclose()


def test_streaming_is_on_by_default():
    from livekit.plugins.bland import TTS

    tts = TTS(api_key="k")
    assert tts.capabilities.streaming is True
    assert tts._pool is not None
