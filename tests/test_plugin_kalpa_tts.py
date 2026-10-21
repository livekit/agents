from __future__ import annotations

import asyncio
import base64
import contextlib
from collections.abc import AsyncIterator

import aiohttp
import pytest
from aiohttp import web

from livekit.agents import APIConnectOptions, APIStatusError, APITimeoutError
from livekit.plugins import kalpa

pytestmark = pytest.mark.unit

CHUNK = base64.b64encode(b"\x01\x00" * 480).decode()  # 20 ms at 24 kHz
NO_RETRY = APIConnectOptions(max_retry=0, timeout=2.0)


class FakeKalpa:
    """A minimal Kalpa TTS API: the WebSocket answers a flush after `delay` seconds and streams
    `chunks` audio frames, 20 ms apart."""

    def __init__(
        self,
        *,
        chunks: int = 3,
        delay: float = 0.0,
        ack_cancel: bool = True,
        error: dict | None = None,
    ):
        self.chunks = chunks
        self.delay = delay
        self.ack_cancel = ack_cancel
        self.error = error  # sent instead of sessionCreated when fatal, else instead of audio
        self.connections = 0
        self.received: list[dict] = []
        self.flushed: list[str] = []  # each flushed utterance's text

    async def ws(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.connections += 1
        self.received.append(await ws.receive_json())
        if self.error and self.error["fatal"]:
            await ws.send_json(self.error)
            await ws.close(code=4401)
            return ws
        await ws.send_json({"type": "sessionCreated", "session_id": "s", "sample_rate": 24000})
        generation: asyncio.Task[None] | None = None
        text = ""
        async for msg in ws:
            event = msg.json()
            self.received.append(event)
            if event["type"] == "sendText":
                text += event.get("text", "")
                if event.get("flush"):
                    self.flushed.append(text)
                    generation = asyncio.create_task(self._respond(ws, silent=not text.strip()))
                    text = ""
            elif event["type"] == "cancelResponse":
                text = ""  # a bare cancel also abandons unflushed text
                if generation and not generation.done():
                    generation.cancel()
                    await ws.send_json(
                        {"type": "responseDone", "response_id": "r", "status": "cancelled"}
                    )
                if self.ack_cancel:
                    await ws.send_json({"type": "cancelled"})
        if generation:
            generation.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await generation
        return ws

    async def _respond(self, ws: web.WebSocketResponse, *, silent: bool) -> None:
        await asyncio.sleep(self.delay)
        await ws.send_json({"type": "responseCreated", "response_id": "r", "sample_rate": 24000})
        if self.error:
            await ws.send_json(self.error)
            await ws.send_json({"type": "responseDone", "response_id": "r", "status": "cancelled"})
            return
        for _ in range(0 if silent else self.chunks):
            await ws.send_json({"type": "responseAudio", "response_id": "r", "pcm_b64": CHUNK})
            await asyncio.sleep(0.02)
        await ws.send_json({"type": "responseDone", "response_id": "r", "status": "completed"})

    async def post(self, request: web.Request) -> web.StreamResponse:
        self.received.append(await request.json())
        if self.error:
            return web.json_response({"error": self.error["error"]}, status=503)
        resp = web.StreamResponse(headers={"Content-Type": "audio/pcm"})
        await resp.prepare(request)
        for _ in range(self.chunks):
            await resp.write(b"\x01\x00" * 480)
        return resp


@contextlib.asynccontextmanager
async def serve(fake: FakeKalpa) -> AsyncIterator[kalpa.TTS]:
    app = web.Application()
    app.router.add_get("/v1/tts/{voice}/stream", fake.ws)
    app.router.add_post("/v1/tts/{voice}/stream", fake.post)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    async with aiohttp.ClientSession() as session:
        tts = kalpa.TTS(api_key="key", base_url=f"http://127.0.0.1:{port}", http_session=session)
        try:
            yield tts
        finally:
            await tts.aclose()
            await runner.cleanup()


async def _speak(tts: kalpa.TTS, text: str) -> int:
    stream = tts.stream(conn_options=NO_RETRY)
    if text:
        stream.push_text(text)
    stream.end_input()
    async with stream:
        return len([ev async for ev in stream])


async def test_replies_share_one_socket() -> None:
    fake = FakeKalpa()
    async with serve(fake) as tts:
        assert await _speak(tts, "Hello there.") > 0
        assert await _speak(tts, "") == 0  # an empty reply is silent, not an error
        assert await _speak(tts, "Hello again.") > 0
    assert fake.connections == 1
    assert fake.flushed == ["Hello there.", "", "Hello again."]
    assert fake.received[0]["generation_config"] == {}
    assert fake.received[0]["sample_rate"] == 24000


@pytest.mark.parametrize("ack_cancel", [True, False])
async def test_barge_in_reuses_socket_after_cancel_ack(ack_cancel: bool) -> None:
    fake = FakeKalpa(chunks=100, ack_cancel=ack_cancel)
    async with serve(fake) as tts:
        stream = tts.stream(conn_options=NO_RETRY)
        stream.push_text("A long reply that the user interrupts.")
        stream.end_input()
        await stream.__anext__()
        await stream.aclose()
        await asyncio.sleep(kalpa.tts.CANCEL_ACK_TIMEOUT + 0.2)
        fake.chunks = 3
        assert await _speak(tts, "The next reply.") > 0
    assert {"type": "cancelResponse"} in fake.received
    assert fake.connections == (1 if ack_cancel else 2)


async def test_flush_before_text_is_ignored() -> None:
    fake = FakeKalpa()
    async with serve(fake) as tts:
        stream = tts.stream(conn_options=NO_RETRY)
        stream.flush()
        stream.push_text("Hello there.")
        stream.end_input()
        async with stream:
            assert len([ev async for ev in stream]) > 0
    assert fake.flushed == ["Hello there."]


async def test_barge_in_before_flush_discards_buffered_text() -> None:
    fake = FakeKalpa()
    async with serve(fake) as tts:
        stream = tts.stream(conn_options=NO_RETRY)
        stream.push_text("Half a sentence that")
        await asyncio.sleep(0.1)
        await stream.aclose()
        await asyncio.sleep(0.2)
        assert await _speak(tts, "The next reply.") > 0
    # text reaches the API as it arrives, so audio can start before the reply ends
    assert {"type": "sendText", "text": "Half a sentence that"} in fake.received
    assert fake.flushed == ["The next reply."]
    assert fake.connections == 1


async def test_timeout_counts_from_the_flush() -> None:
    fake = FakeKalpa(delay=0.5)
    async with serve(fake) as tts:
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=1.0))
        stream.push_text("A reply from a slow LLM.")
        await asyncio.sleep(0.8)  # a pause before the flush does not count
        stream.end_input()
        async with stream:
            assert len([ev async for ev in stream]) > 0

        fake.delay = 3.0  # the API goes silent after the flush
        with pytest.raises(APITimeoutError):
            await asyncio.wait_for(_speak(tts, "Hello there."), 2.5)


@pytest.mark.parametrize(
    ("error", "status", "retryable"),
    [
        ({"type": "authentication_error", "message": "Invalid key."}, 401, False),
        ({"type": "overloaded", "message": "Busy."}, 503, True),
    ],
)
async def test_errors_map_to_status(error: dict, status: int, retryable: bool) -> None:
    fake = FakeKalpa(error={"type": "error", "fatal": status == 401, "error": error})
    async with serve(fake) as tts:
        with pytest.raises(APIStatusError) as exc:
            await _speak(tts, "Hello there.")
    assert (exc.value.status_code, exc.value.retryable) == (status, retryable)


async def test_synthesize_streams_pcm_over_http() -> None:
    fake = FakeKalpa(chunks=50)
    async with serve(fake) as tts:
        frame = await tts.synthesize("Hello there.", conn_options=NO_RETRY).collect()
        fake.error = {"error": {"type": "overloaded", "message": "Busy."}}
        with pytest.raises(APIStatusError) as exc:
            await tts.synthesize("Hello there.", conn_options=NO_RETRY).collect()
    assert frame.duration == pytest.approx(1.0, abs=0.02)
    assert fake.received[0] == {
        "text": "Hello there.",
        "model": "kalpa-tts-1",
        "sample_rate": 24000,
    }
    assert exc.value.status_code == 503
