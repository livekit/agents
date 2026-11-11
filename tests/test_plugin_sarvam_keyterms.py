from __future__ import annotations

import asyncio
import json
from typing import Any
from urllib.parse import parse_qs, urlparse

import aiohttp
import numpy as np
import pytest

from livekit import rtc
from livekit.agents import APIConnectOptions
from livekit.plugins.sarvam import _utils, stt as sarvam_stt, stt_streaming

pytestmark = pytest.mark.unit

TERMS = ["Sarvam", "New Delhi", "Vistaar"]
# 64 Devanagari characters, the longest term Sarvam accepts
DEVANAGARI_64 = ("अआइईउऊएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह" * 2)[:64]


def _query(url: str) -> dict[str, str]:
    return {k: v[0] for k, v in parse_qs(urlparse(url).query).items()}


def _buffer() -> list[rtc.AudioFrame]:
    return [
        rtc.AudioFrame(
            data=np.zeros(1600, dtype=np.int16).tobytes(),
            sample_rate=16000,
            num_channels=1,
            samples_per_channel=1600,
        )
    ]


# ---------------------------------------------------------------------------
# Validation and merging
# ---------------------------------------------------------------------------


def test_validate_keyterms_dedupes_and_keeps_order() -> None:
    assert _utils.validate_keyterms(["a", "b", "a"], "saaras:v4") == ["a", "b"]
    assert _utils.validate_keyterms([], "saaras:v4") is None
    assert _utils.validate_keyterms(None, "saaras:v3") is None


@pytest.mark.parametrize(
    ("terms", "model", "match"),
    [
        ([f"t{i}" for i in range(51)], "saaras:v4", "at most 50 keyterms"),
        (["x" * 65], "saaras:v4", "at most 64 characters"),
        (["  "], "saaras:v4", "non-empty strings"),
        (TERMS, "saaras:v3", "only supported by saaras:v4"),
        (TERMS, "saaras:v3-realtime", "only supported by saaras:v4"),
        ([f"{i:02d}" + DEVANAGARI_64[2:] for i in range(16)], "saaras:v4", "URL-encoded"),
    ],
)
def test_validate_keyterms_rejects_what_sarvam_rejects(
    terms: list[str], model: str, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _utils.validate_keyterms(terms, model)


def test_merge_keyterms_puts_user_terms_first_and_caps_session_terms() -> None:
    user = [f"u{i}" for i in range(48)]
    session = ["u0", "s1", "x" * 65, "s2", "s3"]
    assert _utils.merge_keyterms(user, session, "saaras:v4") == [*user, "s1", "s2"]


def test_merge_keyterms_keeps_the_url_under_the_size_limit() -> None:
    user = [f"{i:02d}" + DEVANAGARI_64[2:] for i in range(15)]
    session = ["a" + DEVANAGARI_64[1:], "b" + DEVANAGARI_64[1:], "Sarvam"]
    merged = _utils.merge_keyterms(user, session, "saaras:v4")
    assert merged == [*user, "Sarvam"]
    assert _utils.keyterms_encoded_size(merged) <= _utils.MAX_KEYTERMS_ENCODED_BYTES


def test_fifteen_full_devanagari_terms_fit() -> None:
    terms = [f"{i:02d}" + DEVANAGARI_64[2:] for i in range(15)]
    assert _utils.validate_keyterms(terms, "saaras:v4") == terms


def test_merge_keyterms_ignores_session_terms_for_other_models() -> None:
    assert _utils.merge_keyterms(None, ["s1"], "saaras:v3-realtime") is None


# ---------------------------------------------------------------------------
# STTRealtime
# ---------------------------------------------------------------------------


def _capture_realtime_streams(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    captured: list[Any] = []

    class _CapturedStream:
        def __init__(self, **kwargs: Any) -> None:
            captured.append(kwargs["opts"])

    monkeypatch.setattr(stt_streaming, "RealtimeSpeechStream", _CapturedStream)
    return captured


def test_realtime_sends_keyterms_on_the_connection_url(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_realtime_streams(monkeypatch)
    stt = stt_streaming.STTRealtime(
        api_key="sk_test",
        model="saaras:v4",
        keyterms=TERMS,
        http_session=object(),  # type: ignore[arg-type]
    )
    assert stt.capabilities.keyterms

    stt.stream()

    url = stt_streaming._build_realtime_ws_url(stt_streaming.SARVAM_STT_REALTIME_URL, captured[0])
    assert json.loads(_query(url)["keyterms"]) == TERMS


def test_realtime_omits_keyterms_when_unset() -> None:
    opts = stt_streaming.RealtimeSTTOptions(language="hi-IN", api_key="sk_test")
    url = stt_streaming._build_realtime_ws_url(stt_streaming.SARVAM_STT_REALTIME_URL, opts)
    assert "keyterms" not in _query(url)


def test_realtime_rejects_keyterms_on_v3_realtime() -> None:
    with pytest.raises(ValueError, match="only supported by saaras:v4"):
        stt_streaming.STTRealtime(api_key="sk_test", keyterms=TERMS)


def test_realtime_merges_session_keyterms_into_new_streams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _capture_realtime_streams(monkeypatch)
    stt = stt_streaming.STTRealtime(
        api_key="sk_test",
        model="saaras:v4",
        keyterms=["Sarvam"],
        http_session=object(),  # type: ignore[arg-type]
    )

    stt._update_session_keyterms(["Vistaar", "Sarvam"])
    stt.stream()
    stt.update_options(keyterms=["Bulbul"])
    stt.stream()

    assert captured[0].keyterms == ["Sarvam", "Vistaar"]
    assert captured[1].keyterms == ["Bulbul", "Vistaar", "Sarvam"]
    assert stt._opts.keyterms == ["Bulbul"]  # session terms never leak into user terms


# ---------------------------------------------------------------------------
# sarvam.STT (WebSocket + REST)
# ---------------------------------------------------------------------------


def test_stt_websocket_url_carries_keyterms() -> None:
    opts = sarvam_stt.SarvamSTTOptions(
        language="hi-IN", api_key="sk_test", model="saaras:v4", keyterms=TERMS
    )
    url = sarvam_stt._build_websocket_url(opts.streaming_url or "", opts)
    assert json.loads(_query(url)["keyterms"]) == TERMS


def test_stt_rejects_keyterms_on_v3() -> None:
    with pytest.raises(ValueError, match="only supported by saaras:v4"):
        sarvam_stt.STT(api_key="sk_test", model="saaras:v3", keyterms=TERMS)


async def test_stt_stream_merges_session_keyterms(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class _DummyStream:
        def __init__(self, *, opts: Any, http_session: Any, **_: Any) -> None:
            captured["opts"] = opts
            captured["http_session"] = http_session

    monkeypatch.setattr(sarvam_stt, "SpeechStream", _DummyStream)
    stt = sarvam_stt.STT(api_key="sk_test", keyterms=["Sarvam"])
    assert stt.capabilities.keyterms
    stt._update_session_keyterms(["Vistaar"])

    stt.stream()
    try:
        assert captured["opts"].keyterms == ["Sarvam", "Vistaar"]
    finally:
        await captured["http_session"].close()


class _OkResponse:
    status = 200

    async def __aenter__(self) -> _OkResponse:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def json(self) -> dict[str, Any]:
        return {
            "request_id": "r1",
            "transcript": "Sarvam",
            "language_code": "hi-IN",
            "timestamps": {"start_time_seconds": [0.1], "end_time_seconds": [0.4]},
        }


class _RecordingSession:
    def __init__(self) -> None:
        self.fields: dict[str, Any] = {}

    def post(self, *, data: Any, **_: Any) -> _OkResponse:
        for type_options, _headers, value in data._fields:
            self.fields[type_options["name"]] = value
        return _OkResponse()


async def test_stt_rest_sends_keyterms() -> None:
    stt = sarvam_stt.STT(api_key="sk_test", keyterms=["Sarvam"])
    stt._update_session_keyterms(["Vistaar"])
    session = _RecordingSession()
    stt._session = session  # type: ignore[assignment]

    event = await stt._recognize_impl(_buffer(), conn_options=APIConnectOptions(max_retry=0))

    assert json.loads(session.fields["keyterms"]) == ["Sarvam", "Vistaar"]
    assert event.alternatives[0].text == "Sarvam"


async def test_stt_rest_omits_keyterms_by_default() -> None:
    stt = sarvam_stt.STT(api_key="sk_test")
    session = _RecordingSession()
    stt._session = session  # type: ignore[assignment]

    await stt._recognize_impl(_buffer(), conn_options=APIConnectOptions(max_retry=0))

    assert "keyterms" not in session.fields


def test_stt_stream_drops_keyterms_for_a_non_v4_model_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class _DummyStream:
        def __init__(self, *, opts: Any, **_: Any) -> None:
            captured["opts"] = opts

    monkeypatch.setattr(sarvam_stt, "SpeechStream", _DummyStream)
    monkeypatch.setattr(aiohttp, "ClientSession", lambda *a, **k: None)
    stt = sarvam_stt.STT(api_key="sk_test", keyterms=TERMS)
    stt._update_session_keyterms(["Vistaar"])

    stt.stream(model="saaras:v3")

    assert captured["opts"].keyterms is None


async def test_stt_rest_drops_keyterms_for_a_non_v4_model_override() -> None:
    stt = sarvam_stt.STT(api_key="sk_test", keyterms=TERMS)
    session = _RecordingSession()
    stt._session = session  # type: ignore[assignment]

    await stt._recognize_impl(
        _buffer(), model="saaras:v3", conn_options=APIConnectOptions(max_retry=0)
    )

    assert "keyterms" not in session.fields


# ---------------------------------------------------------------------------
# STT streams pick up session keyterms by reconnecting between utterances
# ---------------------------------------------------------------------------


class _WebSocket:
    """Like Sarvam's endpoint, it never closes the socket on its own."""

    close_code = 1000

    def __init__(self, url: str) -> None:
        self.url = url
        self.closed = False
        self._closed = asyncio.Event()

    async def send_str(self, data: str) -> None:
        pass

    async def receive(self) -> aiohttp.WSMessage:
        await self._closed.wait()
        return aiohttp.WSMessage(aiohttp.WSMsgType.CLOSED, 1000, "")

    async def close(self) -> None:
        self.closed = True
        self._closed.set()


class _Session:
    def __init__(self) -> None:
        self.closed = False
        self.sockets: list[_WebSocket] = []
        self.connected = asyncio.Event()

    async def ws_connect(self, url: str, **_: object) -> _WebSocket:
        ws = _WebSocket(url)
        self.sockets.append(ws)
        self.connected.set()
        return ws

    async def close(self) -> None:
        self.closed = True

    async def wait_for_connections(self, count: int) -> None:
        async def _wait() -> None:
            while len(self.sockets) < count:
                self.connected.clear()
                await self.connected.wait()

        await asyncio.wait_for(_wait(), timeout=2)

    def keyterms(self, index: int) -> list[str] | None:
        value = _query(self.sockets[index].url).get("keyterms")
        return json.loads(value) if value is not None else None


@pytest.fixture
def ws_session(monkeypatch: pytest.MonkeyPatch) -> _Session:
    monkeypatch.setattr(sarvam_stt, "END_OF_INPUT_GRACE", 0.01)
    session = _Session()
    monkeypatch.setattr(aiohttp, "ClientSession", lambda *args, **kwargs: session)
    return session


async def test_stt_stream_reconnects_with_new_session_keyterms(ws_session: _Session) -> None:
    async with sarvam_stt.STT(api_key="sk_test", keyterms=["Sarvam"]) as stt:
        stream = stt.stream()
        await ws_session.wait_for_connections(1)
        assert ws_session.keyterms(0) == ["Sarvam"]

        stt._update_session_keyterms(["Vistaar"])

        await ws_session.wait_for_connections(2)
        assert ws_session.sockets[0].closed
        assert ws_session.keyterms(1) == ["Sarvam", "Vistaar"]

        # the same terms again change nothing
        stt._update_session_keyterms(["Vistaar"])
        await asyncio.sleep(0.05)
        assert len(ws_session.sockets) == 2
        await stream.aclose()


async def test_stt_stream_waits_for_the_utterance_to_end(ws_session: _Session) -> None:
    async with sarvam_stt.STT(api_key="sk_test") as stt:
        stream = stt.stream()
        await ws_session.wait_for_connections(1)
        await stream._handle_events({"data": {"signal_type": "START_SPEECH"}})

        stt._update_session_keyterms(["Vistaar"])
        await asyncio.sleep(0.05)
        assert len(ws_session.sockets) == 1  # never cut an utterance off

        await stream._handle_events({"data": {"signal_type": "END_SPEECH"}})
        await stream._handle_transcript_data(
            {"data": {"transcript": "Vistaar", "language_code": "hi-IN"}}
        )

        await ws_session.wait_for_connections(2)
        assert ws_session.keyterms(1) == ["Vistaar"]
        await stream.aclose()


async def test_stt_stream_drops_removed_session_keyterms(ws_session: _Session) -> None:
    """With no user terms, session terms the framework removes don't stick around."""
    async with sarvam_stt.STT(api_key="sk_test") as stt:
        stt._update_session_keyterms(["Alice"])
        stream = stt.stream()
        await ws_session.wait_for_connections(1)
        assert ws_session.keyterms(0) == ["Alice"]

        stt._update_session_keyterms([])

        await ws_session.wait_for_connections(2)
        assert ws_session.keyterms(1) is None
        await stream.aclose()


async def test_stt_stream_restores_keyterms_when_switching_back_to_v4(
    ws_session: _Session,
) -> None:
    async with sarvam_stt.STT(api_key="sk_test", keyterms=["Sarvam"]) as stt:
        stt._update_session_keyterms(["Vistaar"])
        stream = stt.stream()
        await ws_session.wait_for_connections(1)

        stream.update_options(language="en-IN", model="saaras:v3")
        await ws_session.wait_for_connections(2)
        assert ws_session.keyterms(1) is None

        stream.update_options(language="en-IN", model="saaras:v4")
        await ws_session.wait_for_connections(3)
        assert ws_session.keyterms(2) == ["Sarvam", "Vistaar"]
        await stream.aclose()


async def test_stt_stream_on_v3_ignores_session_keyterms(ws_session: _Session) -> None:
    async with sarvam_stt.STT(api_key="sk_test", keyterms=["Sarvam"]) as stt:
        stream = stt.stream()
        await ws_session.wait_for_connections(1)
        stream.update_options(language="en-IN", model="saaras:v3")
        await ws_session.wait_for_connections(2)

        stt._update_session_keyterms(["Vistaar"])
        await asyncio.sleep(0.05)
        assert len(ws_session.sockets) == 2  # nothing to send on v3, so no reconnect
        await stream.aclose()


def test_keyterms_capability_follows_the_model() -> None:
    assert sarvam_stt.STT(api_key="sk_test").capabilities.keyterms
    assert not sarvam_stt.STT(api_key="sk_test", model="saaras:v3").capabilities.keyterms
    assert not stt_streaming.STTRealtime(api_key="sk_test").capabilities.keyterms
    assert stt_streaming.STTRealtime(api_key="sk_test", model="saaras:v4").capabilities.keyterms
