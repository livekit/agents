"""Sarvam streaming TTS against a local peer speaking Sarvam's text-to-speech socket protocol."""

from __future__ import annotations

import asyncio
import base64
import struct
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit.agents import APIConnectOptions, APIError, http
from livekit.plugins.sarvam import tts as sarvam_tts

NO_RETRY = APIConnectOptions(max_retry=0)
PCM = b"\x34\x12" * 1100
Handler = Callable[[web.Request], Awaitable[web.WebSocketResponse]]


@asynccontextmanager
async def _ws_url(handler: Handler) -> AsyncIterator[str]:
    app = web.Application()
    app.router.add_get("/ws", handler)
    async with TestServer(app) as server, http.open():
        yield str(server.make_url("/ws")).replace("http", "ws", 1)


def _speaking_peer(
    configs: list[dict[str, object]] | None = None,
    keys: list[str] | None = None,
    *,
    audio: bytes = PCM,
    close_when_done: bool = False,
    closed: asyncio.Event | None = None,
) -> Handler:
    """Answers every flush with `audio` and a final event, recording configs and API keys."""

    async def handle(request: web.Request) -> web.WebSocketResponse:
        if keys is not None:
            keys.append(request.headers.get("api-subscription-key", ""))
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for incoming in ws:
            message = incoming.json()
            if message["type"] == "config" and configs is not None:
                configs.append(message["data"])
            elif message["type"] == "flush":
                encoded = base64.b64encode(audio).decode()
                await ws.send_json({"type": "audio", "data": {"audio": encoded}})
                await ws.send_json({"type": "event", "data": {"event_type": "final"}})
                if close_when_done:  # Sarvam drops a connection once it sits idle
                    await ws.close()
                    if closed is not None:
                        closed.set()
        return ws

    return handle


def _error_peer(frame: dict[str, object]) -> Handler:
    async def handle(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for incoming in ws:
            if incoming.json()["type"] == "flush":
                await ws.send_json({"type": "error", "data": frame})
        return ws

    return handle


async def _speak(model: sarvam_tts.TTS, text: str) -> bytes:
    async with model.stream(conn_options=NO_RETRY) as stream:
        stream.push_text(text)
        stream.end_input()
        return b"".join([event.frame.data.tobytes() async for event in stream])


def _tts(
    url: str,
    *,
    api_key: str | None = "sk_test",
    output_audio_codec: str = "linear16",
    speech_sample_rate: int = 22050,
) -> sarvam_tts.TTS:
    return sarvam_tts.TTS(
        model="bulbul:v3",
        ws_url=url,
        api_key=api_key,
        output_audio_codec=output_audio_codec,
        speech_sample_rate=speech_sample_rate,
    )


async def test_v3_config_matches_the_model_and_codec_and_language_updates_apply() -> None:
    """bulbul:v3 gets no pitch/loudness, the chosen codec decodes, and language updates apply.

    https://github.com/livekit/agents/pull/3413 https://github.com/livekit/agents/pull/5086
    """
    configs: list[dict[str, object]] = []
    async with _ws_url(_speaking_peer(configs, close_when_done=True)) as url:
        model = _tts(url)
        first = await _speak(model, "Hello there.")
        model.update_options(target_language_code="hi-IN")
        await _speak(model, "Namaste.")
        await model.aclose()

    assert first[: len(PCM)] == PCM
    assert not any(first[len(PCM) :])
    assert "pitch" not in configs[0] and "loudness" not in configs[0]
    assert configs[0]["output_audio_codec"] == "linear16"
    assert [c["target_language_code"] for c in configs] == ["en-IN", "hi-IN"]


async def test_reconnects_after_the_server_drops_an_idle_socket() -> None:
    """A pooled socket Sarvam closed while idle never fails the next synthesis.

    https://github.com/livekit/agents/issues/5681
    """
    keys: list[str] = []
    closed = asyncio.Event()
    async with _ws_url(_speaking_peer(keys=keys, close_when_done=True, closed=closed)) as url:
        model = _tts(url)
        first = await _speak(model, "Hello there.")
        await closed.wait()  # the pooled socket acknowledged the server's close
        second = await _speak(model, "Hello again.")
        await model.aclose()

    assert first[: len(PCM)] == PCM
    assert second[: len(PCM)] == PCM
    assert len(keys) == 2


async def test_reads_the_api_key_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """https://github.com/livekit/agents/pull/2579"""
    monkeypatch.setenv("SARVAM_API_KEY", "sk_from_env")
    keys: list[str] = []
    async with _ws_url(_speaking_peer(keys=keys)) as url:
        model = _tts(url, api_key=None)
        await _speak(model, "Hello there.")
        await model.aclose()

    assert keys == ["sk_from_env"]


@pytest.mark.parametrize(
    ("codec", "encoded", "sample"), [("mulaw", 0x00, -32124), ("alaw", 0x55, -8)]
)
async def test_telephony_codec_is_configured_first_and_decoded_to_pcm(
    codec: str, encoded: int, sample: int
) -> None:
    """The G.711 bytes Sarvam sends decode to PCM. https://github.com/livekit/agents/pull/5561"""
    configs: list[dict[str, object]] = []
    peer = _speaking_peer(configs, audio=bytes([encoded]) * 800)
    async with _ws_url(peer) as url:
        model = _tts(url, output_audio_codec=codec, speech_sample_rate=8000)
        pcm = await _speak(model, "Hello there.")
        await model.aclose()

    assert configs[0]["output_audio_codec"] == codec
    assert pcm[: 800 * 2] == struct.pack("<h", sample) * 800


async def test_interrupting_a_synthesis_closes_quietly_and_the_next_one_plays() -> None:
    """A barge-in mid-audio ends the stream promptly and the TTS keeps working.

    https://github.com/livekit/agents/pull/5561
    """
    flushes = 0

    async def peer(request: web.Request) -> web.WebSocketResponse:
        nonlocal flushes
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for incoming in ws:
            if incoming.json()["type"] != "flush":
                continue
            flushes += 1
            await ws.send_json({"type": "audio", "data": {"audio": base64.b64encode(PCM).decode()}})
            if flushes > 1:  # the first synthesis is still speaking when the user barges in
                await ws.send_json({"type": "event", "data": {"event_type": "final"}})
        return ws

    async with _ws_url(peer) as url:
        model = _tts(url)
        interrupted = model.stream(conn_options=NO_RETRY)
        interrupted.push_text("A long answer the user will cut off.")
        interrupted.end_input()
        await anext(interrupted)
        await asyncio.wait_for(interrupted.aclose(), 1)

        second = await _speak(model, "Sure, go ahead.")
        await model.aclose()

    assert second[: len(PCM)] == PCM


@pytest.mark.parametrize(
    ("frame", "status_code", "retryable"),
    [
        # schema rejections carry an integer `code`
        ({"code": 422, "message": "Input parameters has to be a valid dictionary"}, 422, False),
        ({"code": 400, "message": "Speech sample rate can only be 8000 Hz"}, 400, False),
        # speaker and codec rejections omit `code` and prefix the message instead
        ({"message": "400: Speaker 'shubh' is not compatible with bulbul:v4-flash"}, 400, False),
        # a status sent as a digit string is still a status
        ({"code": "400", "message": "invalid speaker"}, 400, False),
        # transient failures stay retryable
        ({"code": 429, "message": "rate limit exceeded"}, 429, True),
        ({"code": 503, "message": "model unavailable"}, 503, True),
        # an unrecognized frame is a refusal, never retried by default
        ({"message": "something we cannot classify"}, None, False),
        ({"code": "invalid_request_error", "message": "bad input"}, None, False),
    ],
)
async def test_an_error_frame_status_decides_retryability_and_hides_provider_text(
    frame: dict[str, object], status_code: int | None, retryable: bool
) -> None:
    """The provider text goes to the error's detail, never its message.

    https://github.com/livekit/agents/pull/7335
    """
    async with _ws_url(_error_peer(frame)) as url:
        model = _tts(url)
        with pytest.raises(APIError) as caught:
            await _speak(model, "Hello there.")
        await model.aclose()

    assert getattr(caught.value, "status_code", None) == status_code
    assert caught.value.retryable is retryable
    assert str(frame["message"]) not in str(caught.value)
    assert caught.value.detail == frame["message"]


@pytest.mark.xfail(
    strict=True,
    reason="https://github.com/livekit/agents/issues/6774 reproduces: v3 accepts the v2 pace range",
)
def test_bulbul_v3_refuses_a_pace_the_api_rejects() -> None:
    """A pace outside bulbul:v3's 0.5-2.0 fails locally, not as a 400 on every request.

    https://github.com/livekit/agents/issues/6774
    """
    with pytest.raises(ValueError, match="pace"):
        sarvam_tts.TTS(api_key="sk_test", model="bulbul:v3", pace=2.5)
