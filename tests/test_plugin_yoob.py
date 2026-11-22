"""Yoob STT and TTS plugin tests: offline, against a local fake of the API
(``WS /v1/audio/transcriptions/stream``, ``WS /v1/audio/speech/stream``, ``POST /v1/audio/transcriptions``).

Async tests run by pytest-asyncio (``asyncio_mode = "auto"`` in the repo's pytest config).
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import numpy as np
import pytest
from aiohttp import web

from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APITimeoutError,
    stt,
)

pytestmark = pytest.mark.unit

_READY = {"type": "ready", "id": "s1", "sample_rate": 16000}


def _frame(ms: int = 100, rate: int = 16000, channels: int = 1) -> rtc.AudioFrame:
    n = rate * ms // 1000
    return rtc.AudioFrame(
        b"\x10\x00" * n * channels, sample_rate=rate, num_channels=channels, samples_per_channel=n
    )


async def _serve(routes: dict[str, Any]) -> tuple[web.AppRunner, str]:
    app = web.Application()
    for path, handler in routes.items():
        if path.endswith("/stream"):
            app.router.add_get(path, handler)
        else:
            app.router.add_post(path, handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    return runner, f"http://127.0.0.1:{port}/v1"


async def _stt_stream(ws_handler, *, frames: list[rtc.AudioFrame], timeout: float = 5.0):
    from livekit.plugins.yoob import STT

    runner, base = await _serve({"/v1/audio/transcriptions/stream": ws_handler})
    try:
        async with aiohttp.ClientSession() as session:
            s = STT(api_key="k", base_url=base, http_session=session)
            stream = s.stream(conn_options=APIConnectOptions(max_retry=0, timeout=timeout))
            for f in frames:
                stream.push_frame(f)
            stream.end_input()
            try:
                return [ev async for ev in stream]
            finally:
                await stream.aclose()
    finally:
        await runner.cleanup()


def _kinds(events: list[stt.SpeechEvent]) -> list[stt.SpeechEventType]:
    return [e.type for e in events if e.type != stt.SpeechEventType.RECOGNITION_USAGE]


# --- STT stream: turns ---


async def test_a_final_after_vad_silence_closes_the_turn_it_belongs_to():
    # VAD start, VAD stop, then the final: one turn (start, final, end), not a second turn opened
    # by the final that would cancel the end of the first.
    async def ws_stt(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_str(json.dumps(_READY))
        sent = False
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.BINARY and not sent:
                sent = True
                await ws.send_str(json.dumps({"type": "vad", "speech": True}))
                await ws.send_str(
                    json.dumps({"type": "partial", "text": "Hel", "start": 0.0, "end": 0.2})
                )
                await ws.send_str(json.dumps({"type": "vad", "speech": False}))
                # a late partial after VAD silence still belongs to the same turn
                await ws.send_str(
                    json.dumps({"type": "partial", "text": "Hello", "start": 0.0, "end": 0.3})
                )
                await ws.send_str(
                    json.dumps({"type": "final", "text": "Hello.", "start": 0.0, "end": 0.4})
                )
            elif msg.type == aiohttp.WSMsgType.TEXT and json.loads(msg.data)["type"] == "stop":
                await ws.send_str(json.dumps({"type": "done"}))
                await ws.close()
        return ws

    started = time.time()
    events = await _stt_stream(ws_stt, frames=[_frame() for _ in range(5)])
    assert _kinds(events) == [
        stt.SpeechEventType.START_OF_SPEECH,
        stt.SpeechEventType.INTERIM_TRANSCRIPT,
        stt.SpeechEventType.INTERIM_TRANSCRIPT,
        stt.SpeechEventType.FINAL_TRANSCRIPT,
        stt.SpeechEventType.END_OF_SPEECH,
    ]
    final = next(e for e in events if e.type == stt.SpeechEventType.FINAL_TRANSCRIPT)
    end = next(e for e in events if e.type == stt.SpeechEventType.END_OF_SPEECH)
    assert final.alternatives[0].text == "Hello."
    # The end carries where the speech ended (the final's end, on the wall clock), so endpointing
    # counts from there and not from when the event arrived.
    assert end.speech_end_time is not None
    assert started - 1 < end.speech_end_time < time.time() + 1


async def test_new_speech_before_a_final_ends_the_old_turn_first():
    async def ws_stt(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_str(json.dumps(_READY))
        sent = False
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.BINARY and not sent:
                sent = True
                for speech in (True, False, True):
                    await ws.send_str(json.dumps({"type": "vad", "speech": speech}))
                await ws.send_str(
                    json.dumps({"type": "final", "text": "Again.", "start": 0.5, "end": 0.9})
                )
            elif msg.type == aiohttp.WSMsgType.TEXT and json.loads(msg.data)["type"] == "stop":
                await ws.send_str(json.dumps({"type": "done"}))
                await ws.close()
        return ws

    events = await _stt_stream(ws_stt, frames=[_frame() for _ in range(5)])
    assert _kinds(events) == [
        stt.SpeechEventType.START_OF_SPEECH,
        stt.SpeechEventType.END_OF_SPEECH,
        stt.SpeechEventType.START_OF_SPEECH,
        stt.SpeechEventType.FINAL_TRANSCRIPT,
        stt.SpeechEventType.END_OF_SPEECH,
    ]


# --- STT stream: audio ---


async def test_stereo_input_is_sent_as_mono_at_its_own_length():
    seen: dict[str, Any] = {"bytes": 0, "first": None}

    async def ws_stt(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_str(json.dumps(_READY))
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.BINARY:
                seen["bytes"] += len(msg.data)
                if seen["first"] is None:
                    seen["first"] = np.frombuffer(msg.data, dtype=np.int16)[:4].tolist()
            elif json.loads(msg.data)["type"] == "stop":
                await ws.send_str(json.dumps({"type": "done"}))
                await ws.close()
        return ws

    # 16 kHz stereo, left 1000 and right 3000: already at the socket's rate, so nothing resamples it
    n = 1600
    pcm = np.empty(n * 2, dtype=np.int16)
    pcm[0::2], pcm[1::2] = 1000, 3000
    stereo = rtc.AudioFrame(pcm.tobytes(), sample_rate=16000, num_channels=2, samples_per_channel=n)
    await _stt_stream(ws_stt, frames=[stereo] * 5)
    assert seen["bytes"] == 5 * n * 2  # 0.5 s of mono PCM16, not 1 s
    assert seen["first"] == [2000, 2000, 2000, 2000]


# --- STT stream: timeouts and dead sockets ---


async def test_no_ready_message_is_a_timeout():
    # A timeout (retryable), not a bare asyncio.TimeoutError that would stop the session's STT.
    async def ws_stt(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for _ in ws:
            pass
        return ws

    with pytest.raises(APITimeoutError):
        await asyncio.wait_for(_stt_stream(ws_stt, frames=[_frame()], timeout=0.5), 10)


async def test_a_stream_that_stalls_after_stop_times_out_for_good():
    # The audio is spent: the caller gets a final, non-retryable timeout instead of a hang.
    async def ws_stt(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_str(json.dumps(_READY))
        async for _ in ws:
            pass
        return ws

    with pytest.raises(APITimeoutError) as exc:
        await asyncio.wait_for(_stt_stream(ws_stt, frames=[_frame()], timeout=0.5), 10)
    assert exc.value.retryable is False


def _live_stt_stream(*, answer_pings: bool) -> tuple[AsyncMock, stt.SpeechStream]:
    """Keep input open against a fake socket, optionally answering the monitor's probes."""
    from livekit.plugins.yoob import STT

    incoming: asyncio.Queue[aiohttp.WSMessage] = asyncio.Queue()
    incoming.put_nowait(aiohttp.WSMessage(aiohttp.WSMsgType.TEXT, json.dumps(_READY), ""))
    ws = AsyncMock(spec=aiohttp.ClientWebSocketResponse)
    ws.receive.side_effect = incoming.get

    if answer_pings:

        async def pong() -> None:
            incoming.put_nowait(aiohttp.WSMessage(aiohttp.WSMsgType.PONG, b"", ""))

        ws.ping.side_effect = pong

    async def connect(url: str, *, autoping: bool, **kwargs: Any) -> AsyncMock:
        assert autoping is False  # the monitor must receive control frames itself
        return ws

    session = MagicMock(spec=aiohttp.ClientSession)
    session.closed = False
    session.ws_connect = connect
    s = STT(api_key="k", http_session=session)
    stream = s.stream(conn_options=APIConnectOptions(max_retry=0, liveness_timeout=1.0))
    stream.push_frame(_frame())
    return ws, stream


@pytest.mark.virtual_time
async def test_a_half_open_socket_fails_within_the_liveness_timeout():
    ws, stream = _live_stt_stream(answer_pings=False)
    try:
        loop = asyncio.get_running_loop()
        started = loop.time()
        with pytest.raises(APIConnectionError, match="WebSocket connection lost"):

            async def drain() -> None:
                async for _ in stream:
                    pass

            await asyncio.wait_for(drain(), 5)
        assert loop.time() - started == pytest.approx(1.0)
        ws.ping.assert_awaited_once()
    finally:
        await stream.aclose()


@pytest.mark.virtual_time
async def test_a_quiet_stream_that_answers_pings_stays_open():
    ws, stream = _live_stt_stream(answer_pings=True)
    events: list[stt.SpeechEvent] = []

    async def collect() -> None:
        async for ev in stream:
            events.append(ev)

    task = asyncio.create_task(collect())
    try:
        await asyncio.sleep(2.5)  # more than two liveness windows without application messages
        assert not task.done()
        assert ws.ping.await_count >= 6
        assert events == []
    finally:
        await stream.aclose()
        await asyncio.gather(task, return_exceptions=True)


# --- STT recognize ---


async def test_recognize_honours_the_connect_options_timeout():
    # FallbackAdapter passes its attempt timeout as conn_options.timeout and relies on it.
    from livekit.plugins.yoob import STT

    async def slow(request: web.Request) -> web.Response:
        await asyncio.sleep(5)
        return web.json_response({"text": "late"})

    runner, base = await _serve({"/v1/audio/transcriptions": slow})
    try:
        async with aiohttp.ClientSession() as session:
            s = STT(api_key="k", base_url=base, http_session=session)
            started = time.monotonic()
            with pytest.raises(APITimeoutError):
                await s.recognize(
                    [_frame()], conn_options=APIConnectOptions(max_retry=0, timeout=0.5)
                )
            assert time.monotonic() - started < 2
    finally:
        await runner.cleanup()


# --- TTS stream ---


async def _tts_stream(ws_handler, pushes: list[str], *, pause: float = 0.0, timeout: float = 5.0):
    from livekit.plugins.yoob import TTS

    runner, base = await _serve({"/v1/audio/speech/stream": ws_handler})
    try:
        async with aiohttp.ClientSession() as session:
            t = TTS(api_key="k", base_url=base, http_session=session)
            stream = t.stream(conn_options=APIConnectOptions(max_retry=0, timeout=timeout))

            async def feed() -> None:
                for i, text in enumerate(pushes):
                    if i and pause:
                        await asyncio.sleep(pause)
                    stream.push_text(text)
                stream.end_input()

            feeder = asyncio.create_task(feed())
            try:
                return [ev async for ev in stream]
            finally:
                await feeder
                await stream.aclose()
    finally:
        await runner.cleanup()


def _tts_server(*, stall: bool = False):
    async def ws_tts(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.receive()  # options
        await ws.send_str(json.dumps({"type": "ready", "request_id": "r1"}))
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT and json.loads(msg.data).get("end"):
                if stall:
                    continue  # all text received; never renders, never closes
                await ws.send_bytes(b"\x00\x01" * 4800)  # 100 ms at 48 kHz
                await ws.send_str(json.dumps({"type": "done"}))
                await ws.close()
        return ws

    return ws_tts


async def test_a_render_that_stalls_after_the_last_text_times_out():
    with pytest.raises(APITimeoutError):
        await asyncio.wait_for(
            _tts_stream(_tts_server(stall=True), ["Hello there."], timeout=0.5), 10
        )


async def test_a_slow_writer_does_not_time_out_while_text_is_still_coming():
    # Gaps in the LLM's text longer than the timeout are normal before the last text is sent.
    events = await asyncio.wait_for(
        _tts_stream(_tts_server(), ["Hello ", "there."], pause=1.2, timeout=0.5), 10
    )
    assert sum(len(ev.frame.data) for ev in events) > 0


async def test_the_render_deadline_starts_when_the_last_text_is_sent():
    # The writer pauses almost the whole timeout before its last text; the render then needs less than
    # the timeout. The deadline must count from the end message, not from a read started in the pause.
    async def ws_tts(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.receive()  # options
        await ws.send_str(json.dumps({"type": "ready", "request_id": "r1"}))
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT and json.loads(msg.data).get("end"):
                await asyncio.sleep(0.3)
                await ws.send_bytes(b"\x00\x01" * 4800)
                await ws.send_str(json.dumps({"type": "done"}))
                await ws.close()
        return ws

    events = await asyncio.wait_for(
        _tts_stream(ws_tts, ["Hello ", "there."], pause=0.45, timeout=0.5), 10
    )
    assert sum(len(ev.frame.data) for ev in events) > 0


async def test_a_failed_send_after_end_input_is_not_retried():
    # The frames the dead connection consumed cannot be replayed, so a retry would quietly drop them.
    from livekit.plugins.yoob import STT

    incoming: asyncio.Queue[aiohttp.WSMessage] = asyncio.Queue()
    incoming.put_nowait(aiohttp.WSMessage(aiohttp.WSMsgType.TEXT, json.dumps(_READY), ""))
    ws = AsyncMock(spec=aiohttp.ClientWebSocketResponse)
    ws.receive.side_effect = incoming.get
    ws.send_bytes.side_effect = aiohttp.ClientConnectionError("connection reset")

    async def connect(url: str, *, autoping: bool, **kwargs: Any) -> AsyncMock:
        return ws

    session = MagicMock(spec=aiohttp.ClientSession)
    session.closed = False
    session.ws_connect = connect
    s = STT(api_key="k", http_session=session)
    stream = s.stream(conn_options=APIConnectOptions(max_retry=0))
    for _ in range(3):
        stream.push_frame(_frame())
    stream.end_input()
    try:
        with pytest.raises(APIConnectionError) as exc:
            async for _ in stream:
                pass
        assert exc.value.retryable is False
    finally:
        await stream.aclose()


async def test_recognize_bounds_a_trickling_answer():
    # One budget for the whole request: an answer that trickles in a byte at a time must not reset it.
    from livekit.plugins.yoob import STT

    async def trickle(request: web.Request) -> web.StreamResponse:
        await request.read()
        resp = web.StreamResponse(headers={"Content-Type": "application/json"})
        await resp.prepare(request)
        for ch in json.dumps({"text": "late"}):
            await resp.write(ch.encode())
            await asyncio.sleep(0.2)
        await resp.write_eof()
        return resp

    runner, base = await _serve({"/v1/audio/transcriptions": trickle})
    try:
        async with aiohttp.ClientSession() as session:
            s = STT(api_key="k", base_url=base, http_session=session)
            started = time.monotonic()
            with pytest.raises(APITimeoutError):
                await s.recognize(
                    [_frame()], conn_options=APIConnectOptions(max_retry=0, timeout=0.5)
                )
            assert time.monotonic() - started < 2
    finally:
        await runner.cleanup()
