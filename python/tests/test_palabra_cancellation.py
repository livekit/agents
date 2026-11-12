"""Palabra SDK cancellation is scoped to the stream's own connection."""

from __future__ import annotations

import asyncio
import base64

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from palabra_ai import SessionError, TtsSession
from palabra_ai.client import REGIONS, Region

from livekit.agents import APIConnectionError, APIConnectOptions
from livekit.plugins.palabra import TTS

pytestmark = pytest.mark.unit


async def test_cancelling_one_stream_stops_its_provider_work_and_keeps_a_sibling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancelled = asyncio.Event()
    first_audio = b"\x11\x00" * 2400
    second_audio = b"\x22\x00" * 2400

    async def synthesize(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        stopped = False
        pending: set[asyncio.Task[None]] = set()

        async def finish_second(generation_id: str) -> None:
            await cancelled.wait()
            if not stopped:
                await ws.send_json(
                    {
                        "message_type": "audio_chunk",
                        "data": {
                            "generation_id": generation_id,
                            "audio": base64.b64encode(second_audio).decode(),
                            "last_chunk": True,
                        },
                    }
                )

        try:
            async for message in ws:
                if message.type != aiohttp.WSMsgType.TEXT:
                    continue
                payload = message.json()
                if payload["type"] == "cancel":
                    # The SDK supports connection-wide cancellation, without a generation id.
                    if "generation_id" not in payload:
                        stopped = True
                        cancelled.set()
                elif payload["type"] == "text":
                    if payload["text"].startswith("First"):
                        await ws.send_json(
                            {
                                "message_type": "audio_chunk",
                                "data": {
                                    "generation_id": payload["generation_id"],
                                    "audio": base64.b64encode(first_audio).decode(),
                                    "last_chunk": False,
                                },
                            }
                        )
                    else:
                        task = asyncio.create_task(finish_second(payload["generation_id"]))
                        pending.add(task)
                        task.add_done_callback(pending.discard)
        finally:
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        return ws

    app = web.Application()
    app.router.add_get("/tts", synthesize)
    async with TestServer(app) as server:
        monkeypatch.setitem(
            REGIONS, "eu", Region(tts=str(server.make_url("/tts").with_scheme("ws")))
        )
        tts = TTS(api_key="test", region="eu")
        try:
            async with tts.stream() as first, tts.stream() as second:
                first.push_text("First reply that will be interrupted.")
                first.flush()
                second.push_text("Second reply must remain audible.")
                second.end_input()
                async with asyncio.timeout(2):
                    assert (await anext(first)).frame.data.tobytes() in first_audio
                    await first.aclose()
                    await cancelled.wait()
                    received = b"".join([event.frame.data.tobytes() async for event in second])
                    assert received == second_audio
        finally:
            await tts.aclose()


async def test_a_stalled_sdk_send_does_not_block_a_sibling_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_sending = asyncio.Event()
    release_first = asyncio.Event()
    audio = b"\x33\x00" * 2400
    original_send = TtsSession.send_text

    async def stalled_send(
        session: TtsSession,
        text: str,
        *,
        eos: bool = False,
        generation_id: str | None = None,
        **voice_overrides: object,
    ) -> None:
        if text.startswith("Stalled"):
            first_sending.set()
            await release_first.wait()
        await original_send(session, text, eos=eos, generation_id=generation_id, **voice_overrides)

    async def synthesize(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for message in ws:
            if message.type != aiohttp.WSMsgType.TEXT:
                continue
            payload = message.json()
            if payload["type"] == "text":
                await ws.send_json(
                    {
                        "message_type": "audio_chunk",
                        "data": {
                            "generation_id": payload["generation_id"],
                            "audio": base64.b64encode(audio).decode(),
                            "last_chunk": True,
                        },
                    }
                )
        return ws

    app = web.Application()
    app.router.add_get("/tts", synthesize)
    async with TestServer(app) as server:
        monkeypatch.setitem(
            REGIONS, "eu", Region(tts=str(server.make_url("/tts").with_scheme("ws")))
        )
        monkeypatch.setattr(TtsSession, "send_text", stalled_send)
        model = TTS(api_key="test", region="eu")
        try:
            async with model.stream() as stalled, model.stream() as sibling:
                stalled.push_text("Stalled transport must not hold the shared rate lock.")
                stalled.flush()
                async with asyncio.timeout(2):
                    await first_sending.wait()
                    sibling.push_text("Sibling reply remains audible.")
                    sibling.end_input()
                    received = b"".join([event.frame.data.tobytes() async for event in sibling])
                assert received == audio
                assert not release_first.is_set()
        finally:
            release_first.set()
            await model.aclose()


async def test_close_failure_does_not_replace_the_provider_failure(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    original_close = TtsSession.close

    async def failed_send(session: TtsSession, text: str, **kwargs: object) -> None:
        raise SessionError("primary provider failure")

    async def failed_close(session: TtsSession) -> None:
        await original_close(session)
        raise RuntimeError("secondary cleanup failure")

    async def peer(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for _ in ws:
            pass
        return ws

    app = web.Application()
    app.router.add_get("/tts", peer)
    async with TestServer(app) as server:
        monkeypatch.setitem(
            REGIONS, "eu", Region(tts=str(server.make_url("/tts").with_scheme("ws")))
        )
        monkeypatch.setattr(TtsSession, "send_text", failed_send)
        monkeypatch.setattr(TtsSession, "close", failed_close)
        model = TTS(api_key="test", region="eu")
        try:
            async with model.stream(conn_options=APIConnectOptions(max_retry=0)) as stream:
                stream.push_text("Provider failure remains the primary diagnostic.")
                stream.end_input()
                with pytest.raises(APIConnectionError, match="SessionError"):
                    _ = [event async for event in stream]
        finally:
            await model.aclose()
    assert sum("palabra TTS cleanup failed" in record.message for record in caplog.records) == 1
