"""Tests for the Palabra TTS plugin (built on the palabra-ai SDK).

Covers option handling, end-to-end streaming, and the interruption path.
The WS mock defined below stands in for the Palabra realtime TTS endpoint.
"""

from __future__ import annotations

import asyncio
import base64
import json

import pytest
from aiohttp import web
from palabra_ai.client import REGIONS, Region

from livekit.agents import APIConnectOptions, APITimeoutError
from livekit.plugins.palabra import TTS

pytestmark = pytest.mark.plugin("palabra")


@pytest.fixture(autouse=True)
def _clean_palabra_env(monkeypatch):
    """Keep tests hermetic: ignore Palabra credentials/region from the dev machine env."""
    monkeypatch.delenv("PALABRA_API_KEY", raising=False)
    monkeypatch.delenv("PALABRA_REGION", raising=False)


@pytest.fixture
async def mock_palabra_ws(monkeypatch):
    """Aiohttp WS server standing in for the Palabra realtime TTS endpoint.

    Patches the SDK's "eu" region, so any `TTS()` created in the test connects here.

    Yields a dict:
      - `url`: ws://127.0.0.1:<port>/v1/text-to-speech/stream
      - `received`: list[dict] of decoded JSON frames the client sent
      - `send_queue`: asyncio.Queue — put dicts here to push as server frames
      - `close_event`: asyncio.Event — set to instruct the server to close the WS
    """
    received: list[dict] = []
    send_queue: asyncio.Queue = asyncio.Queue()
    close_event = asyncio.Event()

    async def ws_handler(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        last_gen = {"id": ""}

        async def pusher():
            while not ws.closed:
                try:
                    msg = await asyncio.wait_for(send_queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    if close_event.is_set():
                        await ws.close()
                        return
                    continue
                # Like the real server, audio_chunk frames echo the client's generation_id:
                #   - a frame queued without an id waits for the next text frame
                #     and takes its id;
                #   - a frame queued with an explicit id goes out as-is (a stale generation).
                data = msg.get("data")
                if (
                    msg.get("message_type") == "audio_chunk"
                    and data is not None
                    and "generation_id" not in data
                ):
                    seen = last_gen["id"]
                    while last_gen["id"] == seen:
                        await asyncio.sleep(0.005)
                        if ws.closed or close_event.is_set():
                            return
                    data["generation_id"] = last_gen["id"]
                await ws.send_json(msg)

        pusher_task = asyncio.create_task(pusher())
        try:
            async for raw in ws:
                if raw.type == web.WSMsgType.TEXT:
                    frame = json.loads(raw.data)
                    received.append(frame)
                    if frame.get("type") == "text" and frame.get("generation_id"):
                        last_gen["id"] = frame["generation_id"]
                elif raw.type in (web.WSMsgType.CLOSE, web.WSMsgType.CLOSED, web.WSMsgType.ERROR):
                    break
        finally:
            pusher_task.cancel()
        return ws

    app = web.Application()
    app.router.add_get("/v1/text-to-speech/stream", ws_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    url = f"ws://127.0.0.1:{port}/v1/text-to-speech/stream"

    monkeypatch.setitem(REGIONS, "eu", Region(tts=url))

    yield {
        "url": url,
        "received": received,
        "send_queue": send_queue,
        "close_event": close_event,
    }
    await runner.cleanup()


# --- Configuration ---


def test_tts_requires_credentials():
    with pytest.raises(ValueError, match="api_key"):
        TTS()


def test_tts_accepts_api_key_directly():
    tts = TTS(api_key="direct-key")
    assert tts._client.api_key == "direct-key"


def test_tts_reads_api_key_from_env(monkeypatch):
    monkeypatch.setenv("PALABRA_API_KEY", "env-key")
    tts = TTS()
    assert tts._client.api_key == "env-key"


def test_tts_defaults():
    tts = TTS(api_key="key")
    assert tts._opts.language == "en"
    assert tts._opts.voice_id == "default_low"
    assert tts._opts.speed is None  # server default
    assert tts._opts.model == "auto"
    assert tts._opts.deaccent_strength == 1.0
    assert tts._client.region == "eu"


def test_tts_capabilities_and_audio():
    tts = TTS(api_key="key")
    assert tts.capabilities.streaming is True
    assert tts.sample_rate == 24000
    assert tts.num_channels == 1


def test_tts_custom_sample_rate():
    tts = TTS(api_key="key", sample_rate=48000)
    assert tts.sample_rate == 48000


def test_tts_model_and_provider_properties():
    tts = TTS(api_key="key")
    assert tts.model == "auto"
    assert tts.provider == "Palabra"


def test_update_options():
    tts = TTS(api_key="key")
    tts.update_options(language="ru", voice_id="default_high", speed=0.7)
    assert tts._opts.language == "ru"
    assert tts._opts.voice_id == "default_high"
    assert tts._opts.speed == 0.7
    assert tts._opts_dirty is True  # session reconnects on next stream


# --- End-to-end stream against the local WS mock ---


async def test_synthesize_stream_full_flow(mock_palabra_ws):
    ws = mock_palabra_ws

    tts = TTS(api_key="test-key")
    stream = tts.stream()
    stream.push_text("Hello world.")
    stream.end_input()

    pcm = b"\x11" * 4800
    ws["send_queue"].put_nowait(
        {
            "message_type": "audio_chunk",
            "data": {"audio": base64.b64encode(pcm).decode(), "size": len(pcm), "last_chunk": True},
        }
    )

    collected: list[bytes] = []
    async for ev in stream:
        if ev.frame is not None:
            collected.append(bytes(ev.frame.data))

    assert pcm in b"".join(collected)
    init_frames = [m for m in ws["received"] if m.get("type") == "init"]
    assert len(init_frames) == 1
    assert init_frames[0]["language"] == "en"
    assert init_frames[0]["output"] == {"format": "pcm", "sample_rate": 24000}
    text_frames = [m for m in ws["received"] if m.get("type") == "text"]
    assert any(m.get("is_eos") for m in text_frames)

    await tts.aclose()


async def test_synthesize_one_shot(mock_palabra_ws):
    ws = mock_palabra_ws

    tts = TTS(api_key="test-key")
    chunked = tts.synthesize("Hello there.")

    pcm = b"\xcc" * 1000
    ws["send_queue"].put_nowait(
        {
            "message_type": "audio_chunk",
            "data": {"audio": base64.b64encode(pcm).decode(), "size": len(pcm), "last_chunk": True},
        }
    )

    collected: list[bytes] = []
    async for ev in chunked:
        if ev.frame is not None:
            collected.append(bytes(ev.frame.data))

    assert pcm in b"".join(collected)
    text_frames = [m for m in ws["received"] if m.get("type") == "text"]
    assert text_frames and any(m.get("is_eos") for m in text_frames)
    await tts.aclose()


async def test_session_reused_across_turns(mock_palabra_ws):
    """Two turns on one TTS reuse the SDK session.

    The Palabra `init` frame is sent once (one WS connection) and both turns produce audio.
    """
    ws = mock_palabra_ws
    tts = TTS(api_key="test-key")

    async def _turn(text: str, pcm: bytes) -> bytes:
        ws["send_queue"].put_nowait(
            {
                "message_type": "audio_chunk",
                "data": {
                    "audio": base64.b64encode(pcm).decode(),
                    "size": len(pcm),
                    "last_chunk": True,
                },
            }
        )
        out: list[bytes] = []
        stream = tts.stream()
        stream.push_text(text)
        stream.end_input()
        async for ev in stream:
            if ev.frame is not None:
                out.append(bytes(ev.frame.data))
        return b"".join(out)

    first = await _turn("First turn.", b"\x11" * 2000)
    second = await _turn("Second turn.", b"\x22" * 2000)

    assert b"\x11" * 2000 in first
    assert b"\x22" * 2000 in second  # 2nd turn on the reused session still works
    init_frames = [m for m in ws["received"] if m.get("type") == "init"]
    assert len(init_frames) == 1  # init sent once (session reused)

    await tts.aclose()


# --- Interruption and generation routing ---


async def test_stale_generation_chunks_filtered(mock_palabra_ws):
    """Audio chunks whose generation_id this stream never sent are dropped.

    Their audio is not emitted, and their last_chunk does not clear this stream's pending_gens.
    """
    ws = mock_palabra_ws
    tts = TTS(api_key="test-key")

    junk = b"\xde\xad" * 500
    real = b"\x33" * 2000
    # Explicit foreign generation_id goes out immediately, before our text.
    ws["send_queue"].put_nowait(
        {
            "message_type": "audio_chunk",
            "data": {
                "audio": base64.b64encode(junk).decode(),
                "size": len(junk),
                "last_chunk": True,
                "generation_id": "stale-gen",
            },
        }
    )
    ws["send_queue"].put_nowait(
        {
            "message_type": "audio_chunk",
            "data": {
                "audio": base64.b64encode(real).decode(),
                "size": len(real),
                "last_chunk": True,
            },
        }
    )

    stream = tts.stream()
    stream.push_text("Hello world.")
    stream.end_input()

    collected: list[bytes] = []
    async for ev in stream:
        if ev.frame is not None:
            collected.append(bytes(ev.frame.data))
    audio = b"".join(collected)

    assert real in audio
    assert junk not in audio  # stale generation was filtered, not played
    await tts.aclose()


async def test_interruption_sends_cancel_and_exits_fast(mock_palabra_ws):
    """Interruption: closing the stream mid-generation sends `cancel` to the server.

    aclose() must also return quickly — the framework awaits it before the next reply starts.
    """
    ws = mock_palabra_ws
    tts = TTS(api_key="test-key")

    pcm = b"\x44" * 2000
    ws["send_queue"].put_nowait(
        {
            "message_type": "audio_chunk",
            "data": {
                "audio": base64.b64encode(pcm).decode(),
                "size": len(pcm),
                "last_chunk": False,
            },
        }
    )

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

    for _ in range(100):
        if any(m.get("type") == "cancel" for m in ws["received"]):
            break
        await asyncio.sleep(0.01)
    cancels = [m for m in ws["received"] if m.get("type") == "cancel"]
    assert cancels
    sent_gens = {m["generation_id"] for m in ws["received"] if m.get("type") == "text"}
    assert all(m.get("generation_id") in sent_gens for m in cancels)
    await tts.aclose()


async def test_no_terminal_frame_no_audio_raises_timeout(mock_palabra_ws):
    """The server can end a generation without a terminal chunk.

    With zero audio received the stream raises APITimeoutError instead of ending empty.
    """
    tts = TTS(api_key="test-key")
    stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=0.5))
    stream.push_text("Hello world.")
    stream.end_input()

    with pytest.raises(APITimeoutError):
        async for _ in stream:
            pass
    await tts.aclose()


async def test_partial_audio_without_terminal_completes(mock_palabra_ws):
    """Partial audio without a terminal chunk ends the stream normally.

    The audio already received is delivered instead of failing a reply the user heard.
    """
    ws = mock_palabra_ws
    tts = TTS(api_key="test-key")

    pcm = b"\x55" * 2000
    ws["send_queue"].put_nowait(
        {
            "message_type": "audio_chunk",
            "data": {
                "audio": base64.b64encode(pcm).decode(),
                "size": len(pcm),
                "last_chunk": False,
            },
        }
    )

    stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=0.6))
    stream.push_text("Hello world.")
    stream.end_input()

    collected: list[bytes] = []
    async for ev in stream:
        if ev.frame is not None:
            collected.append(bytes(ev.frame.data))

    assert pcm in b"".join(collected)
    await tts.aclose()


async def test_slow_input_does_not_time_out(mock_palabra_ws):
    """A pause in the input longer than the receive timeout is not a failure.

    The receive deadline applies only while sent generations are outstanding.
    """
    ws = mock_palabra_ws
    tts = TTS(api_key="test-key")

    pcm = b"\x99" * 2000
    ws["send_queue"].put_nowait(
        {
            "message_type": "audio_chunk",
            "data": {"audio": base64.b64encode(pcm).decode(), "size": len(pcm), "last_chunk": True},
        }
    )

    stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=0.5))

    async def _late_push() -> None:
        await asyncio.sleep(1.2)  # longer than the timeout: the LLM is still producing
        stream.push_text("Hello world.")
        stream.end_input()

    push_task = asyncio.create_task(_late_push())
    collected: list[bytes] = []
    async for ev in stream:
        if ev.frame is not None:
            collected.append(bytes(ev.frame.data))
    await push_task

    assert pcm in b"".join(collected)
    await tts.aclose()


async def test_empty_reply_completes_fast(mock_palabra_ws):
    """A reply with no text ends quickly and cleanly instead of waiting out the timeout."""
    tts = TTS(api_key="test-key")
    stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=2.0))
    await asyncio.sleep(0.3)  # let _run start and park in the receive loop first

    loop = asyncio.get_running_loop()
    started = loop.time()
    stream.end_input()
    async for _ in stream:
        pass
    assert loop.time() - started < 1.0
    await tts.aclose()


async def test_dead_session_reconnects(mock_palabra_ws):
    """A server-closed connection is detected by _session_alive().

    The next stream opens a fresh session, which the mock sees as a second init frame.
    """
    ws = mock_palabra_ws
    tts = TTS(api_key="test-key")

    async def _turn(text: str, pcm: bytes) -> bytes:
        ws["send_queue"].put_nowait(
            {
                "message_type": "audio_chunk",
                "data": {
                    "audio": base64.b64encode(pcm).decode(),
                    "size": len(pcm),
                    "last_chunk": True,
                },
            }
        )
        out: list[bytes] = []
        stream = tts.stream()
        stream.push_text(text)
        stream.end_input()
        async for ev in stream:
            if ev.frame is not None:
                out.append(bytes(ev.frame.data))
        return b"".join(out)

    first = await _turn("First turn.", b"\x66" * 2000)
    assert b"\x66" * 2000 in first

    ws["close_event"].set()
    await asyncio.sleep(0.4)  # let the server close and the client notice
    ws["close_event"].clear()

    second = await _turn("Second turn.", b"\x77" * 2000)
    assert b"\x77" * 2000 in second

    init_frames = [m for m in ws["received"] if m.get("type") == "init"]
    assert len(init_frames) == 2  # dead session was replaced, not reused
    await tts.aclose()


async def test_reply_after_interruption_reuses_session(mock_palabra_ws):
    """The turn after an interruption synthesizes normally on the same session.

    The cancel must not leave _send_lock held or the shared WebSocket unusable.
    """
    ws = mock_palabra_ws
    tts = TTS(api_key="test-key")

    pcm1 = b"\x44" * 2000
    ws["send_queue"].put_nowait(
        {
            "message_type": "audio_chunk",
            "data": {
                "audio": base64.b64encode(pcm1).decode(),
                "size": len(pcm1),
                "last_chunk": False,
            },
        }
    )
    stream = tts.stream()
    stream.push_text("A long reply that will be interrupted.")
    stream.flush()
    async for _ev in stream:  # first audio frame arrived — now interrupt
        break
    await stream.aclose()

    async def _next_turn() -> bytes:
        pcm2 = b"\x88" * 2000
        ws["send_queue"].put_nowait(
            {
                "message_type": "audio_chunk",
                "data": {
                    "audio": base64.b64encode(pcm2).decode(),
                    "size": len(pcm2),
                    "last_chunk": True,
                },
            }
        )
        out: list[bytes] = []
        stream2 = tts.stream()
        stream2.push_text("You said: hello.")
        stream2.end_input()
        async for ev in stream2:
            if ev.frame is not None:
                out.append(bytes(ev.frame.data))
        return b"".join(out)

    audio = await asyncio.wait_for(_next_turn(), timeout=8.0)
    assert b"\x88" * 2000 in audio
    await tts.aclose()


async def test_overlapping_streams_get_their_own_audio(mock_palabra_ws):
    """Two live streams on one session receive their own audio in crossed order.

    The session reader routes frames by generation_id, so neither stream can
    consume or drop the other stream's chunks.
    """
    ws = mock_palabra_ws
    tts = TTS(api_key="test-key")

    s1 = tts.stream()
    s1.push_text("First reply.")
    s1.end_input()
    s2 = tts.stream()
    s2.push_text("Second reply.")
    s2.end_input()

    for _ in range(200):
        if len([m for m in ws["received"] if m.get("type") == "text"]) >= 2:
            break
        await asyncio.sleep(0.01)
    by_text = {m["text"]: m["generation_id"] for m in ws["received"] if m.get("type") == "text"}
    g1, g2 = by_text["First reply."], by_text["Second reply."]

    a1, a2 = b"\x21" * 2000, b"\x42" * 2000
    for pcm, gen in ((a2, g2), (a1, g1)):  # crossed: second stream's audio first
        ws["send_queue"].put_nowait(
            {
                "message_type": "audio_chunk",
                "data": {
                    "audio": base64.b64encode(pcm).decode(),
                    "size": len(pcm),
                    "last_chunk": True,
                    "generation_id": gen,
                },
            }
        )

    async def collect(stream) -> bytes:
        out: list[bytes] = []
        async for ev in stream:
            if ev.frame is not None:
                out.append(bytes(ev.frame.data))
        return b"".join(out)

    r1, r2 = await asyncio.wait_for(asyncio.gather(collect(s1), collect(s2)), timeout=10.0)
    assert a1 in r1
    assert a2 in r2
    await tts.aclose()
