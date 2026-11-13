"""Cartesia TTS and STT against a local peer speaking the Cartesia JSON protocol."""

from __future__ import annotations

import asyncio
import base64
import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    http,
    inference,
)
from livekit.agents.types import USERDATA_TIMED_TRANSCRIPT
from livekit.plugins import cartesia
from livekit.plugins.cartesia import STT, TTS

PCM = b"\x13\x00" * 2400
OPTIONS = APIConnectOptions(max_retry=0)
SECRET_API_KEY = "cartesia-secret-api-key-do-not-log"


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


async def test_rejected_handshake_does_not_leak_the_api_key() -> None:
    """https://github.com/livekit/agents/issues/6739"""

    async def reject(request: web.Request) -> web.Response:
        return web.Response(status=401, text="Unauthorized")

    app = web.Application()
    app.router.add_get("/tts/websocket", reject)
    async with TestServer(app) as server, http.open() as session:
        tts = TTS(
            api_key=SECRET_API_KEY,
            base_url=str(server.make_url("/")).rstrip("/"),
            http_session=session,
        )
        async with tts.stream(conn_options=OPTIONS) as stream:
            stream.push_text("Hello.")
            stream.end_input()
            with pytest.raises(APIStatusError) as error:
                async with asyncio.timeout(2):
                    _ = [ev async for ev in stream]
        await tts.aclose()

    assert error.value.status_code == 401
    err: BaseException | None = error.value
    while err is not None:
        assert SECRET_API_KEY not in repr(err)
        assert SECRET_API_KEY not in str(err)
        err = err.__cause__


@pytest.mark.parametrize("include_context_id", [True, False])
async def test_stale_frames_from_a_previous_context_are_not_played(
    include_context_id: bool,
) -> None:
    """A pooled socket's leftover frames from an older context are skipped.

    https://github.com/livekit/agents/pull/7081
    """
    stale = {"context_id": "stale-context-id"}

    async def speak(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        started = False
        async for msg in ws:
            packet = json.loads(msg.data)
            ctx = {"context_id": packet["context_id"]} if include_context_id else {}
            if not started:
                started = True
                await ws.send_json({**stale, "data": b64(b"\x00" * 160)})
                await ws.send_json({**stale, "done": True})
                await ws.send_json({**ctx, "data": b64(b"\x01" * 160)})
            if packet["continue"] is False:
                await ws.send_json({**ctx, "done": True})
        return ws

    app = web.Application()
    app.router.add_get("/tts/websocket", speak)
    async with TestServer(app) as server, http.open() as session:
        tts = TTS(
            api_key=SECRET_API_KEY,
            base_url=str(server.make_url("")).rstrip("/"),
            http_session=session,
        )
        async with tts.stream(conn_options=OPTIONS) as stream:
            stream.push_text("hello world")
            stream.end_input()
            async with asyncio.timeout(2):
                audio = b"".join([bytes(ev.frame.data) async for ev in stream])
        await tts.aclose()

    assert audio.rstrip(b"\x00") == b"\x01" * 160


async def test_changing_the_api_version_reconnects_with_the_new_version() -> None:
    """The version rides on the socket URL, so only a version change reconnects.

    https://github.com/livekit/agents/pull/7140
    """
    versions: list[str] = []

    async def speak(request: web.Request) -> web.WebSocketResponse:
        versions.append(request.query["cartesia_version"])
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            packet = json.loads(msg.data)
            if packet["continue"] is False:
                ctx = {"context_id": packet["context_id"]}
                await ws.send_json({**ctx, "data": b64(PCM)})
                await ws.send_json({**ctx, "done": True})
        return ws

    async def synthesize(tts: TTS) -> None:
        async with tts.stream(conn_options=OPTIONS) as stream:
            stream.push_text("hello")
            stream.end_input()
            async with asyncio.timeout(2):
                _ = [ev async for ev in stream]

    app = web.Application()
    app.router.add_get("/tts/websocket", speak)
    async with TestServer(app) as server, http.open() as session:
        tts = TTS(
            api_key=SECRET_API_KEY,
            base_url=str(server.make_url("")).rstrip("/"),
            http_session=session,
        )
        await synthesize(tts)
        tts.update_options(speed=1.2)
        await synthesize(tts)
        tts.update_options(api_version="2099-01-01")
        await synthesize(tts)
        await tts.aclose()

    assert len(versions) == 2
    assert versions[1] == "2099-01-01"


async def test_rest_request_sends_the_configured_api_version_header() -> None:
    """https://github.com/livekit/agents/pull/7140"""
    versions: list[str] = []

    async def speak(request: web.Request) -> web.Response:
        versions.append(request.headers["Cartesia-Version"])
        return web.Response(body=PCM, content_type="audio/pcm")

    app = web.Application()
    app.router.add_post("/tts/bytes", speak)
    async with TestServer(app) as server, http.open() as session:
        tts = TTS(
            api_key=SECRET_API_KEY,
            api_version="2099-01-01",
            base_url=str(server.make_url("")).rstrip("/"),
            http_session=session,
        )
        async with tts.synthesize("hello", conn_options=OPTIONS) as stream:
            async with asyncio.timeout(2):
                _ = [ev async for ev in stream]
        await tts.aclose()

    assert versions == ["2099-01-01"]


def test_ink2_claims_no_alignment_through_the_gateway_or_the_plugin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ink-2 sends no word timings, so neither path claims alignment.

    https://github.com/livekit/agents/pull/6629
    """
    monkeypatch.setenv("LIVEKIT_URL", "ws://localhost:7880")
    monkeypatch.setenv("LIVEKIT_API_KEY", "fake")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "fakesecret")

    gateway_stt = inference.STT(model="cartesia/ink-2")
    plugin_stt = cartesia.STT(api_key="fake", model="ink-2")

    assert plugin_stt.capabilities.aligned_transcript is False
    assert gateway_stt.capabilities.aligned_transcript is False


@pytest.mark.parametrize(
    ("fallback", "expected"),
    [
        pytest.param("cartesia/ink-whisper", "word", id="aligned-fallback"),
        pytest.param("cartesia/ink-2", False, id="unaligned-fallback"),
        pytest.param("new-provider/new-turn-model", False, id="unknown-fallback"),
    ],
)
def test_fallback_models_constrain_alignment_claim(
    fallback: str, expected: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A gateway STT claims only the alignment its fallbacks share.

    https://github.com/livekit/agents/pull/6633
    """
    monkeypatch.setenv("LIVEKIT_URL", "ws://localhost:7880")
    monkeypatch.setenv("LIVEKIT_API_KEY", "fake")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "fakesecret")

    stt_impl = inference.STT(model="deepgram/nova-3", fallback=fallback)

    assert stt_impl.capabilities.aligned_transcript == expected


@pytest.mark.parametrize(
    "event",
    [
        "flush_done",
        pytest.param(
            "phoneme_timestamps",
            marks=pytest.mark.xfail(
                strict=True,
                reason="https://github.com/livekit/agents/issues/3071 reproduces: benign phoneme timestamps are warned about",
            ),
        ),
    ],
)
async def test_fragmented_input_alignment_and_benign_events(
    event: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Fragments retain text and alignment fallback, ignoring benign events. https://github.com/livekit/agents/pull/3876 https://github.com/livekit/agents/pull/5321 https://github.com/livekit/agents/issues/3084"""
    packets = []

    async def speak(request: web.Request) -> web.StreamResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            packet = json.loads(msg.data)
            packets.append(packet)
            if packet["continue"]:
                continue
            ctx = {"context_id": packet["context_id"]}
            await ws.send_json({**ctx, "type": event})
            await ws.send_json(
                {
                    **ctx,
                    "word_timestamps": {
                        "words": ["Hello,", "substitute", "world!"],
                        "start": [0, 0.02, 0.04],
                        "end": [0.02, 0.04, 0.08],
                    },
                }
            )
            await ws.send_json({**ctx, "data": base64.b64encode(PCM).decode()})
            await ws.send_json({**ctx, "done": True})
        return ws

    app = web.Application()
    app.router.add_get("/tts/websocket", speak)
    async with TestServer(app) as server, http.open() as session:
        provider = TTS(
            api_key="secret",
            base_url=str(server.make_url("")).rstrip("/"),
            http_session=session,
        )
        try:
            async with provider.stream(conn_options=APIConnectOptions(max_retry=0)) as stream:
                for fragment in ["Hel", "lo, ", "world", "!"]:
                    stream.push_text(fragment)
                stream.end_input()
                async with asyncio.timeout(2):
                    frames = [ev async for ev in stream]
            assert b"".join(bytes(ev.frame.data) for ev in frames) == PCM
            assert "".join(p["transcript"] for p in packets).strip() == "Hello, world!"
            assert all(p["transcript"] for p in packets)
            words = [
                w.text
                for ev in frames
                for w in ev.frame.userdata.get(USERDATA_TIMED_TRANSCRIPT, [])
            ]
            assert words == ["Hello,", "substitute ", "world! "]
            assert not [r for r in caplog.records if r.levelno >= 30 and "unexpected" in r.message]
        finally:
            await provider.aclose()


async def test_provider_error_preserves_diagnostic() -> None:
    """Provider errors reach callers with their diagnostic. https://github.com/livekit/agents/pull/3028"""

    async def speak(request: web.Request) -> web.StreamResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        packet = json.loads((await ws.receive()).data)
        await ws.send_json(
            {
                "context_id": packet["context_id"],
                "type": "error",
                "error": "voice unavailable diagnostic",
            }
        )
        async for _ in ws:
            pass
        return ws

    app = web.Application()
    app.router.add_get("/tts/websocket", speak)
    async with TestServer(app) as server, http.open() as session:
        provider = TTS(
            api_key="secret",
            base_url=str(server.make_url("")),
            http_session=session,
        )
        try:
            async with provider.stream(conn_options=APIConnectOptions(max_retry=0)) as stream:
                stream.push_text("Hello.")
                stream.end_input()
                with pytest.raises(APIError, match="voice unavailable diagnostic"):
                    async with asyncio.timeout(2):
                        _ = [ev async for ev in stream]
        finally:
            await provider.aclose()


@pytest.mark.parametrize(
    "speed",
    [
        pytest.param(
            1,
            marks=pytest.mark.xfail(
                strict=True,
                reason="https://github.com/livekit/agents/pull/7945 reproduces: integer speed is rejected",
            ),
        ),
        1.5,
    ],
)
async def test_integer_and_fractional_speed_reach_synthesis(speed: float) -> None:
    """Valid numeric speeds reach synthesis and return intact PCM. https://github.com/livekit/agents/pull/7945"""

    async def speak(request: web.Request) -> web.StreamResponse:
        payload = await request.json()
        assert payload["generation_config"]["speed"] == speed
        return web.Response(body=PCM, content_type="audio/pcm")

    app = web.Application()
    app.router.add_post("/tts/bytes", speak)
    async with TestServer(app) as server, http.open() as session:
        provider = TTS(
            api_key="secret", speed=speed, base_url=str(server.make_url("")), http_session=session
        )
        try:
            async with provider.synthesize(
                "Hello.", conn_options=APIConnectOptions(max_retry=0)
            ) as stream:
                audio = b"".join([bytes(ev.frame.data) async for ev in stream])
                assert audio[: len(PCM)] == PCM
                assert not any(audio[len(PCM) :])
        finally:
            await provider.aclose()


@pytest.mark.parametrize("mode", ["tts", "legacy_stt", "turns_stt"])
async def test_websocket_authentication_never_exposes_key_in_url(mode: str) -> None:
    """STT and TTS authenticate through headers without URL secrets. https://github.com/livekit/agents/pull/5516"""
    connected = asyncio.Event()
    requests = []

    async def speak(request: web.Request) -> web.StreamResponse:
        requests.append((request.raw_path, request.headers))
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        connected.set()
        async for _ in ws:
            pass
        return ws

    app = web.Application()
    app.router.add_get("/{path:.*}", speak)
    secret = "secret+credential&delimiter"
    async with TestServer(app) as server, http.open() as session:
        url = str(server.make_url(""))
        if mode == "tts":
            provider = TTS(api_key=secret, base_url=url, http_session=session)
            try:
                async with provider.stream(conn_options=APIConnectOptions(max_retry=0)) as stream:
                    stream.push_text("Hello.")
                    stream.flush()
                    await asyncio.wait_for(connected.wait(), 1)
            finally:
                await provider.aclose()
        else:
            recognizer = STT(
                api_key=secret,
                model="ink-whisper" if mode == "legacy_stt" else "ink-2",
                base_url=url,
                http_session=session,
            )
            try:
                async with recognizer.stream(conn_options=APIConnectOptions(max_retry=0)) as audio:
                    audio.push_frame(
                        rtc.AudioFrame(
                            data=bytes(3200),
                            sample_rate=16000,
                            num_channels=1,
                            samples_per_channel=1600,
                        )
                    )
                    await asyncio.wait_for(connected.wait(), 1)
            finally:
                await recognizer.aclose()
        raw, headers = requests[0]
        assert secret not in raw and "credential" not in raw
        assert headers.get("X-API-Key") == secret


@pytest.mark.parametrize("mode", ["legacy_stt", "turns_stt"])
async def test_stt_unexpected_close_keeps_code_and_retryability(mode: str) -> None:
    """Unexpected STT closes retain their close code and retryability.

    https://github.com/livekit/agents/issues/3277 https://github.com/livekit/agents/pull/4683
    """

    async def recognize(request: web.Request) -> web.StreamResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.close(code=1011, message=b"upstream failed")
        return ws

    app = web.Application()
    app.router.add_get("/{path:.*}", recognize)
    async with TestServer(app) as server, http.open() as session:
        provider = STT(
            api_key="key",
            model="ink-whisper" if mode == "legacy_stt" else "ink-2",
            base_url=str(server.make_url("")),
            http_session=session,
        )
        try:
            async with provider.stream(conn_options=APIConnectOptions(max_retry=0)) as stream:
                with pytest.raises(APIConnectionError) as error:
                    async with asyncio.timeout(2):
                        _ = [ev async for ev in stream]
                assert error.value.retryable
                assert "(close code 1011)" in error.value.message
                assert error.value.detail == "upstream failed"
        finally:
            await provider.aclose()
