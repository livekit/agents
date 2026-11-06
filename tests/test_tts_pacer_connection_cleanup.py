from __future__ import annotations

import asyncio
import base64
from typing import Any, Literal

import aiohttp
import pytest
from aiohttp import web

from livekit.agents import APIConnectOptions, APIStatusError, tokenize, tts
from livekit.agents.tts.stream_pacer import StreamPacerWrapper
from livekit.plugins import cartesia, minimax, murf

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]

_AUDIO = b"\x01\x00" * 480


class _ObservedPacer(tts.SentenceStreamPacer):
    def __init__(self) -> None:
        super().__init__()
        self.streams: list[StreamPacerWrapper] = []
        self.tasks: set[asyncio.Task[Any]] = set()

    def wrap(
        self, sent_stream: tokenize.SentenceStream, audio_emitter: tts.AudioEmitter
    ) -> StreamPacerWrapper:
        before = asyncio.all_tasks()
        stream = super().wrap(sent_stream, audio_emitter)
        self.streams.append(stream)
        self.tasks.update(asyncio.all_tasks() - before)
        return stream


@pytest.mark.parametrize(
    "provider", [cartesia.TTS, minimax.TTS, murf.TTS], ids=["cartesia", "minimax", "murf"]
)
@pytest.mark.parametrize("outcome", ["reject", "retry", "cancel", "success"])
@pytest.mark.asyncio
async def test_tts_closes_pacer_across_websocket_acquisition(
    provider: type[tts.TTS], outcome: Literal["reject", "retry", "cancel", "success"]
) -> None:
    handshake_started = asyncio.Event()
    release_handshake = asyncio.Event()
    requests = 0

    async def handle(request: web.Request) -> web.StreamResponse:
        nonlocal requests
        requests += 1
        handshake_started.set()
        if outcome == "cancel":
            await release_handshake.wait()
        if outcome != "success":
            return web.Response(status=503 if outcome == "retry" else 401)

        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for message in ws:
            if message.type != aiohttp.WSMsgType.TEXT:
                continue
            data = message.json()
            if provider is minimax.TTS:
                if data["event"] == "task_start":
                    await ws.send_json({"event": "task_started", "session_id": "test"})
                elif data["event"] == "task_continue":
                    await ws.send_json({"event": "task_continued", "data": {"audio": _AUDIO.hex()}})
                elif data["event"] == "task_finish":
                    await ws.send_json({"event": "task_finished"})
            else:
                context = {"context_id": data["context_id"]}
                if provider is cartesia.TTS:
                    if data["continue"]:
                        await ws.send_json({**context, "data": base64.b64encode(_AUDIO).decode()})
                    else:
                        await ws.send_json({**context, "done": True})
                elif data.get("end"):
                    await ws.send_json({**context, "final": True})
                else:
                    await ws.send_json({**context, "audio": base64.b64encode(_AUDIO).decode()})
        return ws

    app = web.Application()
    app.router.add_route("GET", "/{path:.*}", handle)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    pacer = _ObservedPacer()
    try:
        async with aiohttp.ClientSession() as session:
            options: dict[str, Any] = {
                "api_key": "test-key",
                "base_url": f"http://127.0.0.1:{runner.addresses[0][1]}",
                "http_session": session,
                "text_pacing": pacer,
            }
            if provider is minimax.TTS:
                options["audio_format"] = "pcm"
            model = provider(**options)
            stream = model.stream(
                conn_options=APIConnectOptions(max_retry=1, retry_interval=0, timeout=5)
            )
            stream.push_text("Hello from a local test.")
            stream.end_input()

            async def collect() -> bytes:
                return b"".join([bytes(event.frame.data) async for event in stream])

            try:
                if outcome == "cancel":
                    await asyncio.wait_for(handshake_started.wait(), 5)
                elif outcome == "success":
                    audio = await asyncio.wait_for(collect(), 5)
                    assert audio.startswith(_AUDIO)
                    # The emitter may append a silent end-of-segment marker.
                    assert not any(audio[len(_AUDIO) :])
                else:
                    with pytest.raises(APIStatusError) as exc_info:
                        await asyncio.wait_for(collect(), 5)
                    assert exc_info.value.status_code == (503 if outcome == "retry" else 401)
            finally:
                await asyncio.wait_for(stream.aclose(), 5)
                await asyncio.wait_for(model.aclose(), 5)

            assert requests == (2 if outcome == "retry" else 1)
            assert len(pacer.streams) == requests
            assert all(stream.closed for stream in pacer.streams)
            assert all(task.done() for task in pacer.tasks)
    finally:
        # Clean up leaked tasks too, so the regression fails without affecting later tests.
        for stream in pacer.streams:
            await stream.aclose()
        release_handshake.set()
        await runner.cleanup()
