"""Palabra plugin: STT and TTS realtime wire contracts.

https://github.com/livekit/agents/pull/6681
"""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from palabra_ai.client import REGIONS, Region

from livekit import rtc
from livekit.agents import stt as agents_stt, tts as agents_tts
from livekit.plugins.palabra import STT, TTS
from tests_new.fakes import until


@pytest.fixture(autouse=True)
def _clean_palabra_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tests hermetic: ignore Palabra credentials/region from the dev machine env."""
    monkeypatch.delenv("PALABRA_API_KEY", raising=False)
    monkeypatch.delenv("PALABRA_REGION", raising=False)


# Any: wire JSON frames, read and written by Palabra's documented shape
Frame = dict[str, Any]


@dataclass
class PalabraTTS:
    """The local Palabra TTS endpoint: ``received`` holds the client's frames; frames put on
    ``send_queue`` are pushed to it; setting ``close_event`` closes its sockets."""

    received: list[Frame] = field(default_factory=list)
    send_queue: asyncio.Queue[Frame] = field(default_factory=asyncio.Queue)
    close_event: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass
class PalabraSTT:
    """The local Palabra STT endpoint: each connection's query, the binary audio it got, and
    frames put on ``send_queue`` pushed to the client."""

    queries: list[dict[str, str]] = field(default_factory=list)
    audio: list[bytes] = field(default_factory=list)
    send_queue: asyncio.Queue[Frame] = field(default_factory=asyncio.Queue)


@pytest.fixture
async def mock_palabra_ws(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[PalabraTTS]:
    """A local Palabra realtime TTS endpoint, installed as the SDK's "eu" region."""
    peer = PalabraTTS()
    received, send_queue, close_event = peer.received, peer.send_queue, peer.close_event

    connections: dict[str, web.WebSocketResponse] = {}
    next_generation: asyncio.Queue[tuple[web.WebSocketResponse, str]] = asyncio.Queue()

    async def pusher() -> None:
        pending_target: tuple[web.WebSocketResponse, str] | None = None
        while True:
            try:
                msg = await asyncio.wait_for(send_queue.get(), timeout=0.1)
            except TimeoutError:
                if close_event.is_set():
                    for ws in set(connections.values()):
                        await ws.close()
                    return
                continue
            data = msg.get("data")
            if msg.get("message_type") == "audio_chunk" and data is not None:
                generation_id = data.get("generation_id")
                if generation_id is None:
                    ws, generation_id = pending_target or await next_generation.get()
                    pending_target = None
                    data["generation_id"] = generation_id
                else:
                    live = connections.get(generation_id)
                    if live is not None and not live.closed:
                        ws = live
                    else:
                        # A stale generation is delivered on the next live connection.
                        pending_target = pending_target or await next_generation.get()
                        ws, _ = pending_target
                await ws.send_json(msg)
            else:
                for ws in set(connections.values()):
                    if not ws.closed:
                        await ws.send_json(msg)

    async def ws_handler(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for raw in ws:
            if raw.type == web.WSMsgType.TEXT:
                frame = json.loads(raw.data)
                received.append(frame)
                if frame.get("type") == "text" and frame.get("generation_id"):
                    generation_id = frame["generation_id"]
                    connections[generation_id] = ws
                    next_generation.put_nowait((ws, generation_id))
        return ws

    pusher_task = asyncio.create_task(pusher())
    app = web.Application()
    app.router.add_get("/v1/text-to-speech/stream", ws_handler)
    server = TestServer(app)
    await server.start_server()
    url = str(server.make_url("/v1/text-to-speech/stream").with_scheme("ws"))

    monkeypatch.setitem(REGIONS, "eu", Region(tts=url))

    yield peer
    pusher_task.cancel()
    await asyncio.gather(pusher_task, return_exceptions=True)
    await server.close()


def _audio_chunk(pcm: bytes, *, last_chunk: bool, generation_id: str | None = None) -> Frame:
    data: Frame = {
        "audio": base64.b64encode(pcm).decode(),
        "size": len(pcm),
        "last_chunk": last_chunk,
    }
    if generation_id is not None:
        data["generation_id"] = generation_id
    return {"message_type": "audio_chunk", "data": data}


async def _collect(stream: agents_tts.SynthesizeStream) -> bytes:
    out: list[bytes] = []
    async for ev in stream:
        if ev.frame is not None:
            out.append(bytes(ev.frame.data))
    return b"".join(out)


# --- TTS end-to-end stream against the local WS mock ---


async def test_synthesize_stream_full_flow(mock_palabra_ws: PalabraTTS) -> None:
    """A streamed reply sends one init, ends with an EOS text frame and plays the audio."""
    ws = mock_palabra_ws

    tts = TTS(api_key="test-key")
    stream = tts.stream()
    stream.push_text("Hello world.")
    stream.end_input()

    pcm = b"\x11" * 4800
    ws.send_queue.put_nowait(_audio_chunk(pcm, last_chunk=True))

    audio = await _collect(stream)

    assert pcm in audio
    init_frames = [m for m in ws.received if m.get("type") == "init"]
    assert len(init_frames) == 1
    assert init_frames[0]["language"] == "en"
    assert init_frames[0]["output"] == {"format": "pcm", "sample_rate": 24000}
    text_frames = [m for m in ws.received if m.get("type") == "text"]
    assert any(m.get("is_eos") for m in text_frames)

    await tts.aclose()


# --- TTS interruption and generation routing ---


async def test_interruption_flow(mock_palabra_ws: PalabraTTS) -> None:
    """Closing an interrupted reply stops provider work and filters stale audio."""
    ws = mock_palabra_ws
    tts = TTS(api_key="test-key")

    ws.send_queue.put_nowait(_audio_chunk(b"\x44" * 2000, last_chunk=False))
    stream = tts.stream()
    stream.push_text("A long reply that will be interrupted.")
    stream.flush()

    async for ev in stream:  # wait for the first audio frame, then interrupt
        assert ev.frame is not None
        break

    loop = asyncio.get_running_loop()
    started = loop.time()
    await stream.aclose()
    assert loop.time() - started < 1.5

    await until(
        lambda: any(m.get("type") == "cancel" for m in ws.received), "Palabra gets a cancel"
    )
    cancels = [m for m in ws.received if m.get("type") == "cancel"]
    assert cancels
    assert all("generation_id" not in m for m in cancels)
    cancelled_gen = next(m["generation_id"] for m in ws.received if m.get("type") == "text")

    # The next turn must filter late audio from the cancelled generation.
    junk = b"\xde\xad" * 500
    real = b"\x88" * 2000
    ws.send_queue.put_nowait(_audio_chunk(junk, last_chunk=True, generation_id=cancelled_gen))
    ws.send_queue.put_nowait(_audio_chunk(real, last_chunk=True))

    stream2 = tts.stream()
    stream2.push_text("You said: hello.")
    stream2.end_input()
    audio = await asyncio.wait_for(_collect(stream2), timeout=8.0)

    assert real in audio
    assert junk not in audio  # stale generation was filtered, not played
    await tts.aclose()


# --- STT fixtures and helpers ---


@pytest.fixture
async def mock_palabra_stt_ws(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[PalabraSTT]:
    """A local Palabra realtime STT endpoint, installed as the SDK's "eu" region."""
    peer = PalabraSTT()
    queries, audio, send_queue = peer.queries, peer.audio, peer.send_queue

    async def ws_handler(request: web.Request) -> web.WebSocketResponse:
        queries.append(dict(request.query))
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        async def pusher() -> None:
            while not ws.closed:
                try:
                    msg = await asyncio.wait_for(send_queue.get(), timeout=0.1)
                except TimeoutError:
                    continue
                await ws.send_json(msg)

        pusher_task = asyncio.create_task(pusher())
        try:
            async for raw in ws:
                if raw.type == web.WSMsgType.BINARY:
                    audio.append(raw.data)
                elif raw.type in (web.WSMsgType.CLOSE, web.WSMsgType.CLOSED, web.WSMsgType.ERROR):
                    break
        finally:
            pusher_task.cancel()
        return ws

    app = web.Application()
    app.router.add_get("/asr/v1/speech-to-text/stream", ws_handler)
    server = TestServer(app)
    await server.start_server()
    url = str(server.make_url("/asr/v1/speech-to-text/stream").with_scheme("ws"))

    monkeypatch.setitem(REGIONS, "eu", Region(stt=url))

    yield peer
    await server.close()


SAMPLE_RATE = 16000
FRAME_SAMPLES = SAMPLE_RATE // 100  # 10 ms


def _frame() -> rtc.AudioFrame:
    return rtc.AudioFrame(
        data=b"\x01\x00" * FRAME_SAMPLES,
        sample_rate=SAMPLE_RATE,
        num_channels=1,
        samples_per_channel=FRAME_SAMPLES,
    )


def _transcription(
    text: str,
    *,
    is_eos: bool,
    language: str = "en",
    message_type: str = "transcription",
    transcription_id: str = "tr-1",
) -> Frame:
    return {
        "message_type": message_type,
        "transcription_id": transcription_id,
        "language": language,
        "is_eos": is_eos,
        "segment": {"text": text, "start_time": 0.5, "end_time": 1.5},
        "delta": {"text": text, "start_time": 1.2, "end_time": 1.5},
    }


# --- STT streaming recognition against the local WS mock ---


async def test_stt_stream_interim_and_final(mock_palabra_stt_ws: PalabraSTT) -> None:
    """Audio goes up in 100 ms chunks; interim and EOS transcripts become speech events."""
    ws = mock_palabra_stt_ws
    stt_ = STT(api_key="test-key")
    stream = stt_.stream()

    for _ in range(20):  # 200 ms of audio -> two 100 ms binary chunks
        stream.push_frame(_frame())
    await until(lambda: len(ws.audio) >= 2, "two audio chunks reach Palabra")
    assert len(ws.audio[0]) == SAMPLE_RATE // 10 * 2  # 100 ms of s16le mono

    query = ws.queries[0]
    assert query["token"] == "test-key"
    assert query["format"] == "pcm_s16le"
    assert query["sample_rate"] == str(SAMPLE_RATE)

    ws.send_queue.put_nowait(_transcription("Hello", is_eos=False))
    ws.send_queue.put_nowait(_transcription("Hello world.", is_eos=True))

    events = []
    async for ev in stream:
        events.append(ev)
        if ev.type == agents_stt.SpeechEventType.END_OF_SPEECH:
            break

    types = [ev.type for ev in events]
    assert types == [
        agents_stt.SpeechEventType.START_OF_SPEECH,
        agents_stt.SpeechEventType.INTERIM_TRANSCRIPT,
        agents_stt.SpeechEventType.FINAL_TRANSCRIPT,
        agents_stt.SpeechEventType.END_OF_SPEECH,
    ]
    interim, final = events[1].alternatives[0], events[2].alternatives[0]
    assert interim.text == "Hello"
    assert final.text == "Hello world."
    assert final.language == "en"

    stream.end_input()
    await stream.aclose()
    await stt_.aclose()
