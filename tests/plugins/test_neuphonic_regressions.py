"""Neuphonic streaming wire and lifecycle regression contracts."""

import asyncio
import base64
import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit.agents import APIConnectOptions, APIError, http
from livekit.plugins.neuphonic import TTS

PCM = b"\x04\x00" * 2205


@pytest.mark.parametrize("jwt_only", [False, True])
async def test_fragmented_audio_options_and_authentication(
    jwt_only: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stop markers retain fragmented audio and current authenticated options. https://github.com/livekit/agents/pull/1688 https://github.com/livekit/agents/pull/3703 https://github.com/livekit/agents/pull/7133 https://github.com/livekit/agents/pull/5691 https://github.com/livekit/agents/pull/4874"""
    monkeypatch.delenv("NEUPHONIC_API_KEY", raising=False)
    requests = []
    key = "secret&with+delimiters"

    async def speak(request: web.Request) -> web.StreamResponse:
        requests.append(
            (request.path, dict(request.query), dict(request.headers), request.raw_path)
        )
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            packet = json.loads(msg.data)
            ctx = {"context_id": packet["context_id"]}
            for fragment in (PCM[:800], PCM[800:]):
                await ws.send_json({"data": {**ctx, "audio": base64.b64encode(fragment).decode()}})
            await ws.send_json({"data": {**ctx, "stop": True}})
        return ws

    app = web.Application()
    app.router.add_get("/speak/{lang}", speak)
    async with TestServer(app) as server, http.open() as session:
        provider = TTS(
            api_key=None if jwt_only else key,
            jwt_token="jwt" if jwt_only else None,
            base_url=str(server.make_url("")),
            http_session=session,
        )
        try:
            for lang, voice, speed in [("en", "voice-a", 1.0), ("es", "voice-b", 1.2)]:
                provider.update_options(lang_code=lang, voice_id=voice, speed=speed)
                async with provider.stream(conn_options=APIConnectOptions(max_retry=0)) as stream:
                    stream.push_text("Hello world.")
                    stream.end_input()
                    async with asyncio.timeout(2):
                        audio = b"".join([bytes(ev.frame.data) async for ev in stream])
                assert audio == PCM
                path, query, headers, raw = requests[-1]
                assert path == "/speak/" + lang
                assert query["voice_id"] == voice and float(query["speed"]) == speed
                assert query["encoding"] == "pcm_linear"
                assert key not in raw and "secret%26" not in raw
                if jwt_only:
                    assert query["jwt_token"] == "jwt"
                else:
                    assert headers["x-api-key"] == key
            assert len(requests) == 2
        finally:
            await provider.aclose()


@pytest.mark.parametrize("failure", ["close", "error"])
async def test_streaming_failure_is_mapped_and_next_request_recovers(failure: str) -> None:
    """A failed socket yields a typed error and cannot poison later requests. https://github.com/livekit/agents/pull/2282 https://github.com/livekit/agents/issues/2281"""
    connections = 0

    async def speak(request: web.Request) -> web.StreamResponse:
        nonlocal connections
        connections += 1
        attempt = connections
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            packet = json.loads(msg.data)
            if attempt == 1:
                if failure == "close":
                    await ws.close(code=1011)
                else:
                    await ws.send_json({"type": "error", "errors": "quota exhausted"})
            else:
                await ws.send_json(
                    {
                        "data": {
                            "context_id": packet["context_id"],
                            "audio": base64.b64encode(PCM).decode(),
                            "stop": True,
                        }
                    }
                )
        return ws

    app = web.Application()
    app.router.add_get("/speak/en", speak)
    async with TestServer(app) as server, http.open() as session:
        provider = TTS(api_key="secret", base_url=str(server.make_url("")), http_session=session)
        try:
            for attempt in range(2):
                async with provider.stream(conn_options=APIConnectOptions(max_retry=0)) as stream:
                    stream.push_text("Hello.")
                    stream.end_input()
                    async with asyncio.timeout(2):
                        if attempt == 0:
                            with pytest.raises(APIError):
                                _ = [ev async for ev in stream]
                        else:
                            assert b"".join([bytes(ev.frame.data) async for ev in stream]) == PCM
            assert connections == 2
        finally:
            await provider.aclose()


async def test_retry_after_ended_input_delivers_replayed_audio() -> None:
    """A failure after text input ends retries with usable segmentation state. https://github.com/livekit/agents/issues/7569"""
    connections = 0

    async def speak(request: web.Request) -> web.StreamResponse:
        nonlocal connections
        connections += 1
        attempt = connections
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            packet = json.loads(msg.data)
            if attempt == 1:
                await ws.close(code=1011)
                break
            await ws.send_json(
                {
                    "data": {
                        "context_id": packet["context_id"],
                        "audio": base64.b64encode(PCM).decode(),
                        "stop": True,
                    }
                }
            )
        return ws

    app = web.Application()
    app.router.add_get("/speak/en", speak)
    async with TestServer(app) as server, http.open() as session:
        provider = TTS(api_key="key", base_url=str(server.make_url("")), http_session=session)
        try:
            async with provider.stream(
                conn_options=APIConnectOptions(max_retry=1, retry_interval=0)
            ) as stream:
                stream.push_text("Hello.")
                stream.end_input()
                async with asyncio.timeout(2):
                    assert b"".join([bytes(ev.frame.data) async for ev in stream]) == PCM
            assert connections == 2
        finally:
            await provider.aclose()
