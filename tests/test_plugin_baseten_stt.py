"""Baseten Whisper STT: end-of-input flush.

When the input ends, the stream must send the last partial audio buffer and an
``end_audio`` message, then keep reading until the server reports
``{"type": "end_audio", "body": {"status": "finished"}}``. Transcripts for audio the
server still had buffered arrive in between and must reach the caller.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import aiohttp
import pytest
from aiohttp import web

from livekit import rtc
from livekit.agents import stt
from livekit.plugins.baseten import STT

pytestmark = pytest.mark.unit

_SAMPLE_RATE = 16000


class _FakeWhisperServer:
    """Speaks the Baseten streaming Whisper protocol: metadata, binary PCM, end_audio."""

    def __init__(self, *, reply_to_end_audio: bool = True) -> None:
        self.reply_to_end_audio = reply_to_end_audio
        self.metadata: dict[str, Any] | None = None
        self.audio_bytes = 0
        self.text_messages: list[dict[str, Any]] = []

    async def handle(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.BINARY:
                self.audio_bytes += len(msg.data)
                continue
            data = json.loads(msg.data)
            if self.metadata is None:
                self.metadata = data
                continue
            self.text_messages.append(data)
            if data.get("type") == "end_audio" and self.reply_to_end_audio:
                await ws.send_json({"type": "end_audio", "body": {"status": "acknowledged"}})
                await ws.send_json(
                    {
                        "type": "transcription",
                        "is_final": True,
                        "segments": [
                            {"text": "trailing words", "start_time": 0.0, "end_time": 1.0}
                        ],
                    }
                )
                await ws.send_json({"type": "end_audio", "body": {"status": "finished"}})
                await ws.close()
        return ws


async def _run_stream(server: _FakeWhisperServer, samples: int) -> list[stt.SpeechEvent]:
    app = web.Application()
    app.router.add_get("/ws", server.handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]

    try:
        async with aiohttp.ClientSession() as session:
            baseten_stt = STT(
                api_key="test-key",
                model_endpoint=f"ws://127.0.0.1:{port}/ws",
                http_session=session,
            )
            stream = baseten_stt.stream()
            stream.push_frame(
                rtc.AudioFrame(
                    data=b"\x00\x00" * samples,
                    sample_rate=_SAMPLE_RATE,
                    num_channels=1,
                    samples_per_channel=samples,
                )
            )
            stream.end_input()

            async def collect() -> list[stt.SpeechEvent]:
                return [ev async for ev in stream]

            try:
                return await asyncio.wait_for(collect(), timeout=10)
            finally:
                await stream.aclose()
    finally:
        await runner.cleanup()


async def test_end_input_flushes_buffered_audio() -> None:
    server = _FakeWhisperServer()
    # 1.5 buffers: the trailing 256 samples only go out when the input ends
    events = await _run_stream(server, samples=768)

    assert server.audio_bytes == 768 * 2
    assert server.text_messages == [{"type": "end_audio"}]
    finals = [ev for ev in events if ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT]
    assert [ev.alternatives[0].text for ev in finals] == ["trailing words"]


async def test_end_input_completes_when_server_ignores_end_audio() -> None:
    # runtimes without end_audio support never answer; the stream still has to finish
    server = _FakeWhisperServer(reply_to_end_audio=False)
    events = await _run_stream(server, samples=512)

    assert server.text_messages == [{"type": "end_audio"}]
    assert not [ev for ev in events if ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT]
