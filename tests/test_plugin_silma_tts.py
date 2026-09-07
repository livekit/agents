"""Tests for the SILMA TTS plugin: configuration, text splitting and wire protocol."""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any
from unittest.mock import patch

import aiohttp
import numpy as np
import pytest

from livekit.agents import APIConnectionError, APIConnectOptions, APIStatusError

pytestmark = pytest.mark.unit


def _float32_bytes(*samples: float) -> bytes:
    return np.array(samples, dtype="<f4").tobytes()


def _pcm16(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype="<i2")


# ---------------------------------------------------------------- configuration


def test_tts_requires_api_key() -> None:
    from livekit.plugins.silma import TTS

    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="API key"):
            TTS(api_key=None)


def test_tts_accepts_api_key_directly() -> None:
    from livekit.plugins.silma import TTS

    assert TTS(api_key="test-key")._opts.api_key == "test-key"


def test_tts_accepts_api_key_from_env() -> None:
    from livekit.plugins.silma import TTS

    with patch.dict("os.environ", {"SILMA_API_KEY": "env-key"}):
        assert TTS()._opts.api_key == "env-key"


def test_tts_defaults() -> None:
    from livekit.plugins.silma import TTS

    tts = TTS(api_key="test-key")
    assert tts.model == "silma-tts-v2-msa"
    assert tts.provider == "SILMA"
    assert tts.sample_rate == 24000
    assert tts.num_channels == 1
    assert tts.capabilities.streaming is True
    assert tts._opts.voice == "sarah"
    # Unset tuning knobs are left to the server rather than guessed at.
    assert tts._opts.creativity is None
    assert tts._opts.speed is None


def test_tts_custom_options() -> None:
    from livekit.plugins.silma import TTS

    tts = TTS(
        api_key="test-key",
        model="silma-tts-v2-english",
        voice="emma",
        creativity=0.4,
        speed=0.3,
    )
    assert tts._opts.model == "silma-tts-v2-english"
    assert tts._opts.voice == "emma"
    assert tts._opts.creativity == 0.4
    assert tts._opts.speed == 0.3


def test_tts_rejects_empty_voice() -> None:
    from livekit.plugins.silma import TTS

    with pytest.raises(ValueError, match="voice"):
        TTS(api_key="test-key", voice="   ")


def test_tts_rejects_custom_voice_without_user_id() -> None:
    from livekit.plugins.silma import TTS

    with pytest.raises(ValueError, match="user_id"):
        TTS(api_key="test-key", custom_audio_id="voice_1769817467123")


def test_tts_accepts_custom_voice_with_user_id() -> None:
    from livekit.plugins.silma import TTS

    tts = TTS(api_key="test-key", user_id="u-1", custom_audio_id="voice_1769817467123")
    assert tts._opts.custom_audio_id == "voice_1769817467123"


def test_any_voice_and_model_pairing_is_accepted() -> None:
    """Pairing is left to the API; the plugin does not second-guess it."""
    from livekit.plugins.silma import TTS

    tts = TTS(api_key="test-key", model="silma-tts-v2-english", voice="sarah")
    assert tts._opts.voice == "sarah"
    assert tts._opts.model == "silma-tts-v2-english"


def test_update_options() -> None:
    from livekit.plugins.silma import TTS

    tts = TTS(api_key="test-key")
    tts.update_options(model="silma-tts-v2-ksa", voice="sultan", speed=0.5)
    assert tts._opts.model == "silma-tts-v2-ksa"
    assert tts._opts.voice == "sultan"
    assert tts._opts.speed == 0.5


def test_update_options_is_atomic_on_invalid_value() -> None:
    from livekit.plugins.silma import TTS

    tts = TTS(api_key="test-key")
    original = tts._opts

    with pytest.raises(ValueError, match="user_id"):
        tts.update_options(voice="salma", custom_audio_id="voice_123")

    assert tts._opts == original


def test_payload_omits_unset_fields() -> None:
    from livekit.plugins.silma import TTS

    payload = TTS(api_key="test-key")._opts.build_payload("مرحبا")
    assert payload == {
        "model_id": "silma-tts-v2-msa",
        "text": "مرحبا",
        "voice_id": "sarah",
    }


def test_payload_includes_set_fields() -> None:
    from livekit.plugins.silma import TTS

    payload = TTS(
        api_key="test-key",
        creativity=0.3,
        speed=0.1,
        user_id="u-1",
        custom_audio_id="voice_1",
        enable_server_pronunciation_overrides=True,
    )._opts.build_payload("hi")
    assert payload["creativity"] == 0.3
    assert payload["speed"] == 0.1
    assert payload["user_id"] == "u-1"
    assert payload["custom_audio_id"] == "voice_1"
    assert payload["enable_server_pronunciation_overrides"] is True


# ------------------------------------------------------------------------ urls


def test_http_stream_url() -> None:
    from livekit.plugins.silma._utils import http_stream_url

    assert http_stream_url("https://api.silma.ai/tts/v2") == "https://api.silma.ai/tts/v2/stream"


def test_websocket_url_from_https() -> None:
    from livekit.plugins.silma._utils import websocket_url

    assert websocket_url("https://api.silma.ai/tts/v2") == "wss://api.silma.ai/tts/v2/ws/stream"


def test_insecure_base_url_rejected() -> None:
    from livekit.plugins.silma._utils import normalize_base_url

    with pytest.raises(ValueError, match="HTTPS"):
        normalize_base_url("http://example.com")


def test_insecure_localhost_allowed() -> None:
    from livekit.plugins.silma._utils import normalize_base_url

    assert normalize_base_url("http://localhost:8080/tts/v2") == "http://localhost:8080/tts/v2"


# -------------------------------------------------------------- float32 decode


def test_float32_decoder_converts_full_scale() -> None:
    from livekit.plugins.silma._utils import Float32Decoder

    pcm = _pcm16(Float32Decoder().decode(_float32_bytes(0.0, 1.0, -1.0, 0.5)))
    assert pcm.tolist() == [0, 32767, -32767, 16383]


def test_float32_decoder_clips_out_of_range_samples() -> None:
    from livekit.plugins.silma._utils import Float32Decoder

    # Without clipping these would wrap to the opposite polarity and click.
    pcm = _pcm16(Float32Decoder().decode(_float32_bytes(1.5, -1.5)))
    assert pcm.tolist() == [32767, -32767]


def test_float32_decoder_buffers_partial_samples() -> None:
    from livekit.plugins.silma._utils import Float32Decoder

    raw = _float32_bytes(0.25, -0.25, 0.75)
    decoder = Float32Decoder()

    out = b""
    # Feed the stream one byte at a time: no chunk boundary lands on a sample.
    for i in range(len(raw)):
        out += decoder.decode(raw[i : i + 1])

    assert _pcm16(out).tolist() == _pcm16(Float32Decoder().decode(raw)).tolist()
    assert len(out) == 6


def test_float32_decoder_emits_nothing_for_partial_sample() -> None:
    from livekit.plugins.silma._utils import Float32Decoder

    assert Float32Decoder().decode(b"\x00\x00\x00") == b""


# --------------------------------------------------------------- text splitting


def test_split_text_respects_provider_limit() -> None:
    from livekit.plugins.silma._utils import split_text

    chunks = split_text("word " * 500, max_characters=250)
    assert " ".join(chunks) == ("word " * 500).strip()
    assert all(0 < len(chunk) <= 250 for chunk in chunks)


def test_split_text_keeps_words_whole() -> None:
    from livekit.plugins.silma._utils import split_text

    assert split_text("alpha beta gamma", max_characters=11) == ["alpha beta", "gamma"]


def test_split_text_keeps_pronunciation_tags_intact() -> None:
    from livekit.plugins.silma._utils import split_text

    text = "Call <STAG_PN>92005455</STAG_PN> or email <STAG_EMAIL>hi@silma.ai</STAG_EMAIL> today"
    for chunk in split_text(text, max_characters=40):
        assert chunk.count("<STAG_") == chunk.count("</STAG_")


def test_split_text_unwraps_tag_that_cannot_fit() -> None:
    from livekit.plugins.silma._utils import split_text

    # A dangling half-tag would be read out literally, so the hint is dropped
    # and the content kept.
    chunks = split_text("<STAG_LINK>www.silma.ai/very/long/path</STAG_LINK>", max_characters=20)
    assert "".join(chunks) == "www.silma.ai/very/long/path"
    assert not any("STAG" in chunk for chunk in chunks)


def test_split_text_hard_splits_an_oversized_word() -> None:
    from livekit.plugins.silma._utils import split_text

    chunks = split_text("a" * 25, max_characters=10)
    assert chunks == ["a" * 10, "a" * 10, "a" * 5]


def test_split_text_on_whitespace_only_input() -> None:
    from livekit.plugins.silma._utils import split_text

    assert split_text("   \n  ", max_characters=250) == []


# ------------------------------------------------------------------- api errors


def test_raise_for_status_marks_client_errors_non_retryable() -> None:
    from livekit.plugins.silma._utils import raise_for_status

    with pytest.raises(APIStatusError) as exc_info:
        raise_for_status(401, json.dumps({"detail": "invalid_api_key"}))

    assert exc_info.value.status_code == 401
    assert exc_info.value.retryable is False


def test_raise_for_status_marks_server_errors_retryable() -> None:
    from livekit.plugins.silma._utils import raise_for_status

    with pytest.raises(APIStatusError) as exc_info:
        raise_for_status(503, "upstream unavailable")

    assert exc_info.value.retryable is True


def test_raise_for_status_does_not_echo_response_body() -> None:
    from livekit.plugins.silma._utils import raise_for_status

    secret = "the full text the customer asked us to synthesize, plus their key"
    with pytest.raises(APIStatusError) as exc_info:
        raise_for_status(500, json.dumps({"detail": secret}))

    assert secret not in str(exc_info.value)
    assert secret not in json.dumps(exc_info.value.body)


# ------------------------------------------------------------- http synthesize


class _FakeContent:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def iter_chunks(self) -> Any:
        for chunk in self._chunks:
            yield chunk, True


class _FakeResponse:
    def __init__(
        self,
        *,
        status: int = 200,
        chunks: list[bytes] | None = None,
        content_type: str = "application/octet-stream",
        text: str = "",
    ) -> None:
        self.status = status
        self.headers = {"Content-Type": content_type}
        self.content = _FakeContent(chunks or [])
        self._text = text

    async def __aenter__(self) -> _FakeResponse:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def text(self) -> str:
        return self._text


class _FakeSession:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    def post(self, url: str, *, headers: Any, json: Any, timeout: Any) -> _FakeResponse:
        self.calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return self._response


async def test_synthesize_decodes_float32_to_pcm() -> None:
    from livekit.plugins.silma import TTS

    raw = _float32_bytes(*([0.5] * 2400))
    session = _FakeSession(_FakeResponse(chunks=[raw[:1001], raw[1001:]]))
    tts = TTS(api_key="test-key", http_session=session)  # type: ignore[arg-type]

    frames = [ev.frame async for ev in tts.synthesize("Hello there.")]
    assert frames
    assert frames[0].sample_rate == 24000
    audio = b"".join(bytes(frame.data) for frame in frames)
    # float32 in, int16 out: half the bytes, plus whatever silence the emitter
    # pads the trailing frame with.
    assert len(audio) >= len(raw) // 2
    samples = _pcm16(audio)
    assert set(samples[: len(raw) // 4].tolist()) == {16383}
    assert set(samples[len(raw) // 4 :].tolist()) <= {0}
    await tts.aclose()


async def test_synthesize_sends_api_key_header_and_payload() -> None:
    from livekit.plugins.silma import TTS

    session = _FakeSession(_FakeResponse(chunks=[_float32_bytes(*([0.0] * 100))]))
    tts = TTS(api_key="test-key", http_session=session)  # type: ignore[arg-type]

    async for _ in tts.synthesize("Hello."):
        pass

    call = session.calls[0]
    assert call["url"] == "https://api.silma.ai/tts/v2/stream"
    assert call["headers"]["apiKey"] == "test-key"
    assert call["json"]["text"] == "Hello."
    await tts.aclose()


async def test_synthesize_sends_the_plugin_user_agent() -> None:
    """SILMA uses this to tell plugin traffic apart from direct API use."""
    from livekit.plugins.silma import TTS
    from livekit.plugins.silma.models import USER_AGENT

    session = _FakeSession(_FakeResponse(chunks=[_float32_bytes(*([0.0] * 100))]))
    tts = TTS(api_key="test-key", http_session=session)  # type: ignore[arg-type]

    async for _ in tts.synthesize("Hello."):
        pass

    assert session.calls[0]["headers"]["User-Agent"] == USER_AGENT
    assert USER_AGENT.startswith("LiveKit-Agents-SILMA/")
    assert "livekit-agents/" in USER_AGENT
    await tts.aclose()


async def test_websocket_handshake_sends_the_plugin_user_agent() -> None:
    from livekit.plugins.silma import TTS
    from livekit.plugins.silma.models import USER_AGENT

    captured: dict[str, Any] = {}

    class FakeHTTPSession:
        async def ws_connect(self, url: str, *, headers: Any) -> Any:
            captured["url"] = url
            captured["headers"] = headers
            # The headers are the subject of the test; fail the connect so the
            # plugin's own error mapping runs and nothing is left open.
            raise ConnectionRefusedError("no server here")

    tts = TTS(api_key="test-key", http_session=FakeHTTPSession())  # type: ignore[arg-type]

    with pytest.raises(APIConnectionError):
        await tts._connect_ws(timeout=1.0)

    assert captured["headers"]["User-Agent"] == USER_AGENT
    assert captured["headers"]["apiKey"] == "test-key"
    await tts.aclose()


async def test_synthesize_splits_text_over_the_character_limit() -> None:
    from livekit.plugins.silma import TTS

    session = _FakeSession(_FakeResponse(chunks=[_float32_bytes(*([0.0] * 100))]))
    tts = TTS(api_key="test-key", http_session=session)  # type: ignore[arg-type]

    async for _ in tts.synthesize("word " * 300):
        pass

    assert len(session.calls) > 1
    assert all(len(call["json"]["text"]) <= 250 for call in session.calls)
    await tts.aclose()


async def test_multi_chunk_synthesis_survives_a_mid_text_failure() -> None:
    """A blip partway through long text must not truncate the speech.

    Long input is split into several requests. The framework's own retry loop
    (`ChunkedStream._main_task`) re-runs `_run` under a fresh emitter and
    request_id, so the plugin's job is only to raise a *retryable* error and let
    that happen. Retrying inside the plugin would stack on top of this and turn
    an outage into a retry storm.
    """
    from livekit.plugins.silma import TTS
    from livekit.plugins.silma._utils import split_text

    class FlakySession:
        """Fails once, on the third request."""

        def __init__(self) -> None:
            self.sent: list[str] = []
            self.failed_once = False

        def post(self, url: str, *, headers: Any, json: Any, timeout: Any) -> _FakeResponse:
            self.sent.append(json["text"])
            if not self.failed_once and len(self.sent) == 3:
                self.failed_once = True
                return _FakeResponse(status=503, text="upstream hiccup")
            return _FakeResponse(chunks=[_float32_bytes(*([0.5] * 2400))])

    text = " ".join(["word"] * 300)
    expected_chunks = len(split_text(text))
    assert expected_chunks > 3, "the text must span more chunks than the failure point"

    session = FlakySession()
    tts = TTS(api_key="test-key", http_session=session)  # type: ignore[arg-type]

    pcm = bytearray()
    async for ev in tts.synthesize(
        text, conn_options=APIConnectOptions(max_retry=2, retry_interval=0.0)
    ):
        pcm.extend(bytes(ev.frame.data))

    # Every chunk contributes 2400 samples of audio; allow the emitter's tail padding.
    delivered = round(len(pcm) / 2 / 2400)
    assert delivered >= expected_chunks, (
        f"only {delivered} of {expected_chunks} chunks reached the listener"
    )
    assert session.failed_once
    await tts.aclose()


async def test_transient_status_is_retryable_but_auth_is_not() -> None:
    """The classification is what lets the framework retry a blip and give up on a bad key."""
    from livekit.plugins.silma._utils import raise_for_status

    for transient in (500, 502, 503, 429):
        with pytest.raises(APIStatusError) as exc:
            raise_for_status(transient, "")
        assert exc.value.retryable is True, f"{transient} should be retryable"

    for fatal in (400, 401, 403, 404, 422):
        with pytest.raises(APIStatusError) as exc:
            raise_for_status(fatal, "")
        assert exc.value.retryable is False, f"{fatal} should not be retryable"


async def test_synthesize_raises_on_error_status() -> None:
    from livekit.plugins.silma import TTS

    session = _FakeSession(
        _FakeResponse(status=401, content_type="application/json", text='{"detail":"bad key"}')
    )
    tts = TTS(api_key="test-key", http_session=session)  # type: ignore[arg-type]

    with pytest.raises(APIStatusError) as exc_info:
        async for _ in tts.synthesize("Hello.", conn_options=APIConnectOptions(max_retry=0)):
            pass

    assert exc_info.value.status_code == 401
    await tts.aclose()


async def test_synthesize_rejects_json_body_on_a_200() -> None:
    from livekit.plugins.silma import TTS

    session = _FakeSession(_FakeResponse(content_type="application/json", text="{}"))
    tts = TTS(api_key="test-key", http_session=session)  # type: ignore[arg-type]

    with pytest.raises(APIStatusError, match="non-audio") as exc_info:
        async for _ in tts.synthesize("Hello.", conn_options=APIConnectOptions(max_retry=0)):
            pass

    assert exc_info.value.status_code == 502
    await tts.aclose()


# ---------------------------------------------------------------- websocket


class _FakeMsg:
    def __init__(self, type_: aiohttp.WSMsgType, data: Any) -> None:
        self.type = type_
        self.data = data
        self.extra = None


class _FakeWS:
    """A scripted WebSocket: each sent utterance replays one canned response."""

    def __init__(self, responses: list[list[_FakeMsg]]) -> None:
        self._responses = responses
        self.sent: list[dict[str, Any]] = []
        self.closed = False
        self.close_code: int | None = None
        self._inbox: list[_FakeMsg] = []

    async def send_str(self, data: str) -> None:
        self.sent.append(json.loads(data))
        self._inbox = list(self._responses.pop(0)) if self._responses else []

    async def receive(self, timeout: float | None = None) -> _FakeMsg:
        if not self._inbox:
            self.closed = True
            return _FakeMsg(aiohttp.WSMsgType.CLOSED, None)
        return self._inbox.pop(0)

    async def close(self) -> None:
        self.closed = True


def _audio_event(*samples: float) -> _FakeMsg:
    payload = {
        "status": "streaming",
        "audio": base64.b64encode(_float32_bytes(*samples)).decode(),
        "text": "chunk",
    }
    return _FakeMsg(aiohttp.WSMsgType.TEXT, json.dumps(payload))


def _completed_event() -> _FakeMsg:
    return _FakeMsg(aiohttp.WSMsgType.TEXT, json.dumps({"status": "completed"}))


def _install_ws(monkeypatch: pytest.MonkeyPatch, tts: Any, ws: Any) -> list[int]:
    """Point the TTS connection pool at ``ws``; returns a connect counter."""
    connects: list[int] = []

    async def fake_connect(timeout: float) -> Any:
        connects.append(1)
        return ws() if callable(ws) else ws

    monkeypatch.setattr(tts._pool, "_connect_cb", fake_connect)
    return connects


async def test_stream_decodes_base64_audio_events(monkeypatch: pytest.MonkeyPatch) -> None:
    from livekit.plugins.silma import TTS

    ws = _FakeWS(
        [
            [
                _FakeMsg(aiohttp.WSMsgType.TEXT, json.dumps({"status": "started"})),
                _audio_event(*([0.5] * 1200)),
                _audio_event(*([0.5] * 1200)),
                _completed_event(),
            ]
        ]
    )
    tts = TTS(api_key="test-key")
    _install_ws(monkeypatch, tts, ws)

    stream = tts.stream(conn_options=APIConnectOptions(max_retry=0))
    stream.push_text("Hello there.")
    stream.end_input()

    frames = [ev.frame async for ev in stream]
    assert frames
    audio = b"".join(bytes(frame.data) for frame in frames)
    assert set(_pcm16(audio).tolist()) <= {16383, 0}
    assert 16383 in set(_pcm16(audio).tolist())
    assert ws.sent[0]["text"] == "Hello there."
    await tts.aclose()


async def test_stream_accepts_binary_audio_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    from livekit.plugins.silma import TTS

    ws = _FakeWS(
        [[_FakeMsg(aiohttp.WSMsgType.BINARY, _float32_bytes(*([0.25] * 2400))), _completed_event()]]
    )
    tts = TTS(api_key="test-key")
    _install_ws(monkeypatch, tts, ws)

    stream = tts.stream(conn_options=APIConnectOptions(max_retry=0))
    stream.push_text("Hello there.")
    stream.end_input()

    frames = [ev.frame async for ev in stream]
    assert frames
    audio = b"".join(bytes(frame.data) for frame in frames)
    assert set(_pcm16(audio).tolist()) <= {8191, 0}
    assert 8191 in set(_pcm16(audio).tolist())
    await tts.aclose()


async def test_stream_raises_on_failed_status(monkeypatch: pytest.MonkeyPatch) -> None:
    from livekit.plugins.silma import TTS

    ws = _FakeWS(
        [
            [
                _FakeMsg(
                    aiohttp.WSMsgType.TEXT,
                    json.dumps(
                        {"status": "failed", "code": "internal_error", "detail": "secret text"}
                    ),
                )
            ]
        ]
    )
    tts = TTS(api_key="test-key")
    _install_ws(monkeypatch, tts, ws)

    stream = tts.stream(conn_options=APIConnectOptions(max_retry=0))
    stream.push_text("Hello there.")
    stream.end_input()

    with pytest.raises(APIStatusError) as exc_info:
        async for _ in stream:
            pass

    assert exc_info.value.body == {"status": "failed", "code": "internal_error"}
    assert "secret text" not in str(exc_info.value)
    await tts.aclose()


async def test_stream_splits_long_sentences(monkeypatch: pytest.MonkeyPatch) -> None:
    from livekit.plugins.silma import TTS

    ws = _FakeWS([[_audio_event(0.0), _completed_event()] for _ in range(10)])
    tts = TTS(api_key="test-key")
    _install_ws(monkeypatch, tts, ws)

    stream = tts.stream(conn_options=APIConnectOptions(max_retry=0))
    stream.push_text("word " * 200 + ".")
    stream.end_input()

    async for _ in stream:
        pass

    assert len(ws.sent) > 1
    assert all(len(sent["text"]) <= 250 for sent in ws.sent)
    await tts.aclose()


async def test_stream_reconnects_once_when_a_pooled_socket_is_stale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pooled connection the server has since closed must not lose the utterance."""
    from livekit.plugins.silma import TTS

    sockets: list[_FakeWS] = []

    def make_ws() -> _FakeWS:
        # The first socket is dead: it returns CLOSED before any audio.
        ws = _FakeWS([[]] if not sockets else [[_audio_event(*([0.5] * 1200)), _completed_event()]])
        sockets.append(ws)
        return ws

    tts = TTS(api_key="test-key")
    connects = _install_ws(monkeypatch, tts, make_ws)

    # Prime the pool so the first acquisition counts as reused.
    stale = await tts._pool.get(timeout=1.0)
    tts._pool.put(stale)

    stream = tts.stream(conn_options=APIConnectOptions(max_retry=0))
    stream.push_text("Hello there.")
    stream.end_input()

    frames = [ev.frame async for ev in stream]
    assert frames, "the utterance should have been replayed on a fresh connection"
    assert len(connects) == 2
    await tts.aclose()


async def test_stream_does_not_replay_after_audio_was_emitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replaying after partial audio would duplicate speech, so it must not happen."""
    from livekit.plugins.silma import TTS

    sockets: list[_FakeWS] = []

    def make_ws() -> _FakeWS:
        # Audio, then the socket dies before `completed`.
        ws = _FakeWS([[_audio_event(*([0.5] * 1200))]])
        sockets.append(ws)
        return ws

    tts = TTS(api_key="test-key")
    connects = _install_ws(monkeypatch, tts, make_ws)

    primed = await tts._pool.get(timeout=1.0)
    tts._pool.put(primed)

    stream = tts.stream(conn_options=APIConnectOptions(max_retry=0))
    stream.push_text("Hello there.")
    stream.end_input()

    with pytest.raises(APIStatusError):
        async for _ in stream:
            pass

    assert len(connects) == 1
    await tts.aclose()


async def test_stream_reuses_one_connection_across_sentences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from livekit.plugins.silma import TTS

    ws = _FakeWS([[_audio_event(0.5), _completed_event()] for _ in range(4)])
    tts = TTS(api_key="test-key")
    connects = _install_ws(monkeypatch, tts, ws)

    stream = tts.stream(conn_options=APIConnectOptions(max_retry=0))
    stream.push_text("First sentence. Second sentence. Third sentence.")
    stream.end_input()

    async for _ in stream:
        pass

    assert len(ws.sent) >= 2, "each sentence should be its own request"
    assert len(connects) == 1, "sentences should share the pooled connection"
    await tts.aclose()


async def test_stream_closed_socket_is_not_returned_to_the_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from livekit.plugins.silma import TTS

    class ClosingWS(_FakeWS):
        async def receive(self, timeout: float | None = None) -> _FakeMsg:
            msg = await super().receive(timeout)
            if isinstance(msg.data, str) and '"completed"' in msg.data:
                # Some deployments hang up after each utterance.
                self.closed = True
            return msg

    ws = ClosingWS([[_audio_event(0.5), _completed_event()]])
    tts = TTS(api_key="test-key")
    _install_ws(monkeypatch, tts, ws)

    stream = tts.stream(conn_options=APIConnectOptions(max_retry=0))
    stream.push_text("Hello there.")
    stream.end_input()

    async for _ in stream:
        pass

    assert ws not in tts._pool._available
    await tts.aclose()


async def test_stream_aclose_is_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    from livekit.plugins.silma import TTS

    ws = _FakeWS([[_audio_event(*([0.5] * 1200)), _completed_event()]])
    tts = TTS(api_key="test-key")
    _install_ws(monkeypatch, tts, ws)

    stream = tts.stream(conn_options=APIConnectOptions(max_retry=0))
    stream.push_text("Hello there.")
    stream.end_input()
    await anext(aiter(stream))

    await asyncio.wait_for(stream.aclose(), timeout=2.0)
    await tts.aclose()
