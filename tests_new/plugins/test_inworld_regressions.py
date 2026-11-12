"""Inworld audio delivery and cancellation contracts."""

import asyncio
import base64
import io
import json
import wave
from typing import Literal

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit.agents import APIConnectOptions, http
from livekit.agents.types import USERDATA_TIMED_TRANSCRIPT
from livekit.plugins.inworld import TTS

PCM = b"\x09\x00" * 2400


@pytest.mark.parametrize("encoding", ["PCM", "LINEAR16"])
async def test_first_chunk_emits_before_provider_finishes(
    encoding: Literal["PCM", "LINEAR16"],
) -> None:
    """Each audio chunk is playable before response completion with faithful spacing. https://github.com/livekit/agents/pull/4983 https://github.com/livekit/agents/pull/5071 https://github.com/livekit/agents/pull/5470"""
    release = asyncio.Event()
    if encoding == "LINEAR16":
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24000)
            wav.writeframes(PCM)
        audio = buf.getvalue()
    else:
        audio = PCM

    async def speak(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(headers={"Content-Type": "application/json"})
        await response.prepare(request)
        result = {
            "audioContent": base64.b64encode(audio).decode(),
            "timestampInfo": {
                "wordAlignment": {
                    "words": ["Hello", " ", "world!"],
                    "wordStartTimeSeconds": [0, 0.02, 0.03],
                    "wordEndTimeSeconds": [0.02, 0.03, 0.08],
                }
            },
        }
        await response.write(json.dumps({"result": result}).encode() + b"\n")
        await release.wait()
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/tts/v1/voice:stream", speak)
    async with TestServer(app) as server, http.open() as session:
        provider = TTS(
            api_key="key",
            encoding=encoding,
            sample_rate=24000,
            timestamp_type="WORD",
            base_url=str(server.make_url("")),
            ws_url=str(server.make_url("")),
            http_session=session,
        )
        try:
            async with provider.synthesize(
                "Hello world!", conn_options=APIConnectOptions(max_retry=0)
            ) as stream:
                try:
                    first = await asyncio.wait_for(anext(stream), 0.5)
                    assert not release.is_set()
                finally:
                    release.set()
                frames = [first, *[ev async for ev in stream]]
            words = [
                w.text
                for ev in frames
                for w in ev.frame.userdata.get(USERDATA_TIMED_TRANSCRIPT, [])
            ]
            assert "".join(words) == "Hello world!"
            audio_out = b"".join(bytes(ev.frame.data) for ev in frames)
            assert audio_out[: len(PCM)] == PCM
            assert not any(audio_out[len(PCM) :])
        finally:
            release.set()
            await provider.aclose()


@pytest.mark.parametrize("ended", [False, True])
@pytest.mark.xfail(
    strict=True,
    reason="https://github.com/livekit/agents/pull/8006 reproduces: abandoned context drains instead of canceling",
)
async def test_abandoning_stream_sends_cancel_context(ended: bool) -> None:
    """Abandoning synthesis cancels server work even after input completion. https://github.com/livekit/agents/pull/8006"""
    packets = []
    termination = asyncio.Event()

    async def speak(request: web.Request) -> web.StreamResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            packet = json.loads(msg.data)
            packets.append(packet)
            ctx = packet["contextId"]
            if "create" in packet:
                await ws.send_json({"result": {"contextId": ctx, "contextCreated": {}}})
            if "send_text" in packet:
                await ws.send_json(
                    {
                        "result": {
                            "contextId": ctx,
                            "audioChunk": {"audioContent": base64.b64encode(PCM).decode()},
                        }
                    }
                )
            if "cancel_context" in packet or "close_context" in packet:
                termination.set()
        return ws

    app = web.Application()
    app.router.add_get("/tts/v1/voice:streamBidirectional", speak)
    async with TestServer(app) as server, http.open() as session:
        provider = TTS(
            api_key="key",
            encoding="PCM",
            base_url=str(server.make_url("")),
            ws_url=str(server.make_url("")),
            http_session=session,
        )
        try:
            async with provider.stream(conn_options=APIConnectOptions(max_retry=0)) as stream:
                stream.push_text("Hello world.")
                stream.flush()
                if ended:
                    stream.end_input()
                assert (await asyncio.wait_for(anext(stream), 1)).frame.samples_per_channel > 0
            await asyncio.wait_for(termination.wait(), 1)
            assert any("cancel_context" in p for p in packets)
        finally:
            await provider.aclose()


async def test_cancelled_handshake_does_not_exhaust_pool() -> None:
    """Cancelled context acquisition releases capacity for later synthesis. https://github.com/livekit/agents/issues/5426"""
    connecting: asyncio.Queue[asyncio.Event] = asyncio.Queue()
    attempts = 0

    async def speak(request: web.Request) -> web.StreamResponse:
        nonlocal attempts
        attempts += 1
        ws = web.WebSocketResponse()
        if attempts <= 6:
            gate = asyncio.Event()
            connecting.put_nowait(gate)
            await gate.wait()
            try:
                await ws.prepare(request)
            except ConnectionResetError:
                return ws
        else:
            await ws.prepare(request)
        async for msg in ws:
            packet = json.loads(msg.data)
            ctx = packet["contextId"]
            if "create" in packet:
                await ws.send_json({"result": {"contextId": ctx, "contextCreated": {}}})
            if "send_text" in packet:
                await ws.send_json(
                    {
                        "result": {
                            "contextId": ctx,
                            "audioChunk": {"audioContent": base64.b64encode(PCM).decode()},
                        }
                    }
                )
            if "close_context" in packet:
                await ws.send_json({"result": {"contextId": ctx, "contextClosed": {}}})
        return ws

    app = web.Application()
    app.router.add_get("/tts/v1/voice:streamBidirectional", speak)
    async with TestServer(app) as server, http.open() as session:
        provider = TTS(
            api_key="key",
            encoding="PCM",
            base_url=str(server.make_url("")),
            ws_url=str(server.make_url("")),
            http_session=session,
        )
        try:
            for _ in range(6):
                stream = provider.stream(conn_options=APIConnectOptions(max_retry=0))
                stream.push_text("Hello.")
                stream.end_input()
                gate = await asyncio.wait_for(connecting.get(), 1)
                try:
                    await asyncio.wait_for(stream.aclose(), 1)
                finally:
                    gate.set()
            async with provider.stream(conn_options=APIConnectOptions(max_retry=0)) as stream:
                stream.push_text("Hello.")
                stream.end_input()
                async with asyncio.timeout(2):
                    assert [ev async for ev in stream]
        finally:
            await provider.aclose()


async def test_stream_preserves_punctuation_and_timed_spacing() -> None:
    """Synthesis preserves punctuation and timed text without added spaces. https://github.com/livekit/agents/pull/4215 https://github.com/livekit/agents/pull/5470"""
    text = "I paid $1,250.50 -- really!"
    sent = []

    async def speak(request: web.Request) -> web.StreamResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            packet = json.loads(msg.data)
            ctx = packet["contextId"]
            if "create" in packet:
                await ws.send_json({"result": {"contextId": ctx, "contextCreated": {}}})
            if "send_text" in packet:
                sent.append(packet["send_text"]["text"])
                await ws.send_json(
                    {
                        "result": {
                            "contextId": ctx,
                            "audioChunk": {
                                "audioContent": base64.b64encode(PCM).decode(),
                                "timestampInfo": {
                                    "wordAlignment": {
                                        "words": ["I ", "paid ", "$1,250.50 ", "-- ", "really!"],
                                        "wordStartTimeSeconds": [0, 0.01, 0.02, 0.03, 0.04],
                                        "wordEndTimeSeconds": [0.01, 0.02, 0.03, 0.04, 0.05],
                                    }
                                },
                            },
                        }
                    }
                )
            if "close_context" in packet:
                await ws.send_json({"result": {"contextId": ctx, "contextClosed": {}}})
        return ws

    app = web.Application()
    app.router.add_get("/tts/v1/voice:streamBidirectional", speak)
    async with TestServer(app) as server, http.open() as session:
        provider = TTS(
            api_key="key",
            encoding="PCM",
            timestamp_type="WORD",
            ws_url=str(server.make_url("")),
            http_session=session,
        )
        try:
            async with provider.stream(conn_options=APIConnectOptions(max_retry=0)) as stream:
                for fragment in ["I paid $1,", "250.50 -- ", "really!"]:
                    stream.push_text(fragment)
                stream.end_input()
                async with asyncio.timeout(2):
                    frames = [ev async for ev in stream]
            assert "".join(sent) == text
            words = [
                w.text
                for ev in frames
                for w in ev.frame.userdata.get(USERDATA_TIMED_TRANSCRIPT, [])
            ]
            assert "".join(words) == text
        finally:
            await provider.aclose()


@pytest.mark.parametrize("diagnostic", ["Context not found", "Invalid voice for context"])
async def test_rejected_context_fails_promptly_and_next_request_recovers(
    diagnostic: str,
) -> None:
    """Rejected contexts raise typed diagnostics without poisoning later synthesis. https://github.com/livekit/agents/issues/5103 https://github.com/livekit/agents/issues/5274"""
    created = 0
    failed = set()

    async def speak(request: web.Request) -> web.StreamResponse:
        nonlocal created
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            packet = json.loads(msg.data)
            ctx = packet["contextId"]
            if "create" in packet:
                created += 1
                if created == 1:
                    failed.add(ctx)
                    await ws.send_json(
                        {"result": {"contextId": ctx, "status": {"code": 5, "message": diagnostic}}}
                    )
                else:
                    await ws.send_json({"result": {"contextId": ctx, "contextCreated": {}}})
            if ctx in failed:
                continue
            if "send_text" in packet:
                await ws.send_json(
                    {
                        "result": {
                            "contextId": ctx,
                            "audioChunk": {"audioContent": base64.b64encode(PCM).decode()},
                        }
                    }
                )
            if "close_context" in packet:
                await ws.send_json({"result": {"contextId": ctx, "contextClosed": {}}})
        return ws

    from livekit.agents import APIError

    app = web.Application()
    app.router.add_get("/tts/v1/voice:streamBidirectional", speak)
    async with TestServer(app) as server, http.open() as session:
        provider = TTS(
            api_key="key",
            voice="workspace__custom",
            encoding="PCM",
            ws_url=str(server.make_url("")),
            http_session=session,
        )
        try:
            for attempt in range(2):
                async with provider.stream(conn_options=APIConnectOptions(max_retry=0)) as stream:
                    stream.push_text("Hello.")
                    stream.end_input()
                    async with asyncio.timeout(2):
                        if attempt == 0:
                            with pytest.raises(APIError, match=diagnostic):
                                _ = [ev async for ev in stream]
                        else:
                            audio = b"".join([bytes(ev.frame.data) async for ev in stream])
                            assert audio[: len(PCM)] == PCM
                            assert not any(audio[len(PCM) :])
        finally:
            await provider.aclose()
