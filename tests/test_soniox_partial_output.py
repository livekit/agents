"""A later provider failure never restarts a synthesis that already delivered audio."""

from __future__ import annotations

import asyncio
import base64

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit.agents import APIConnectOptions, APIStatusError
from livekit.plugins.soniox import TTS

pytestmark = pytest.mark.unit


async def test_a_failure_after_audio_is_not_replayed() -> None:
    requests: list[str] = []
    first_received = asyncio.Event()

    async def synthesize(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for incoming in ws:
            if incoming.type != aiohttp.WSMsgType.TEXT:
                continue
            message = incoming.json()
            if not message.get("text"):
                continue
            requests.append(message["text"])
            stream_id = message["stream_id"]
            if len(requests) == 1:
                await ws.send_json(
                    {
                        "stream_id": stream_id,
                        "audio": base64.b64encode(b"\x00\x00" * 2400).decode(),
                    }
                )
                await ws.send_json({"stream_id": stream_id, "audio_end": True, "terminated": True})
                first_received.set()
            else:
                await ws.send_json(
                    {"stream_id": stream_id, "error_code": 429, "error_message": "later failure"}
                )
        return ws

    app = web.Application()
    app.router.add_get("/tts", synthesize)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    assert site._server is not None
    port = site._server.sockets[0].getsockname()[1]
    try:
        async with aiohttp.ClientSession() as session:
            tts = TTS(
                api_key="test", websocket_url=f"ws://127.0.0.1:{port}/tts", http_session=session
            )
            try:
                options = APIConnectOptions(max_retry=1, retry_interval=0)
                async with tts.stream(conn_options=options) as stream:
                    stream.push_text("First sentence is long enough to start synthesis. Next ")
                    # Wait for delivered audio, not just the peer's send acknowledgement.
                    async with asyncio.timeout(2):
                        audio = await anext(stream)
                        assert audio.frame.samples_per_channel > 0
                        await first_received.wait()
                        stream.push_text("sentence must report the provider failure. ")
                        stream.end_input()
                        with pytest.raises(APIStatusError, match="later failure"):
                            async for _ in stream:
                                pass
                assert len(requests) == 2
            finally:
                await tts.aclose()
    finally:
        await runner.cleanup()


@pytest.mark.parametrize("status_code", [429, 401])
async def test_batch_failure_preserves_audio_and_continues_only_transient_errors(
    status_code: int,
) -> None:
    requests: list[tuple[str, str]] = []
    failed = asyncio.Event()
    prefix = b"\x01\x00" * 2400
    suffix = b"\x02\x00" * 2400

    async def synthesize(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(autoping=False)
        await ws.prepare(request)
        async for incoming in ws:
            if incoming.type == aiohttp.WSMsgType.PONG:
                failed.set()
                continue
            if incoming.type != aiohttp.WSMsgType.TEXT:
                continue
            message = incoming.json()
            if not message.get("text"):
                continue
            stream_id, text = message["stream_id"], message["text"]
            requests.append((stream_id, text))
            if text.lstrip().startswith("First"):
                await ws.send_json(
                    {"stream_id": stream_id, "audio": base64.b64encode(prefix).decode()}
                )
            elif text.lstrip().startswith("Second"):
                await ws.send_json(
                    {
                        "stream_id": stream_id,
                        "error_code": status_code,
                        "error_message": "mid-reply",
                    }
                )
                # The pong proves the client consumed the preceding error frame.
                await ws.ping(b"failed")
            else:
                await ws.send_json(
                    {"stream_id": stream_id, "audio": base64.b64encode(suffix).decode()}
                )
                await ws.send_json({"stream_id": stream_id, "audio_end": True, "terminated": True})
        return ws

    app = web.Application()
    app.router.add_get("/tts", synthesize)
    async with TestServer(app) as server, aiohttp.ClientSession() as session:
        tts = TTS(api_key="test", websocket_url=str(server.make_url("/tts")), http_session=session)
        try:
            options = APIConnectOptions(max_retry=1, retry_interval=0)
            async with tts.stream(conn_options=options) as stream:
                stream.push_text(
                    "First sentence is long enough to start synthesis. Second sentence "
                )
                async with asyncio.timeout(2):
                    first = await anext(stream)
                    audio = bytearray(first.frame.data.tobytes())
                    stream.push_text("encounters the transient provider failure. Third sentence ")
                    await failed.wait()
                    # This text was never submitted to the failed provider stream.
                    stream.push_text("still needs its own synthesized audio. ")
                    stream.end_input()
                    with pytest.raises(APIStatusError, match="mid-reply"):
                        async for event in stream:
                            audio.extend(event.frame.data.tobytes())
                assert requests[0][0] == requests[1][0]
                if status_code == 429:
                    assert bytes(audio) == prefix + suffix
                    assert requests[2][0] != requests[1][0]
                    assert [text.split()[0] for _, text in requests] == ["First", "Second", "Third"]
                else:
                    assert bytes(audio) == prefix
                    assert [text.split()[0] for _, text in requests] == ["First", "Second"]
        finally:
            await tts.aclose()
