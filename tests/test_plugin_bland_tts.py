from __future__ import annotations

import struct
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock, patch

import aiohttp
import pytest
from aiohttp import web

from livekit.agents import APIConnectionError, APIConnectOptions, APIStatusError
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


def test_not_streaming():
    from livekit.plugins.bland import TTS

    assert TTS(api_key="test-key").capabilities.streaming is False


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
