"""Baseten validates transcript timing before exposing speech events."""

from __future__ import annotations

import asyncio
import gc

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit.agents import APIConnectionError, APIConnectOptions, APIStatusError, stt
from livekit.plugins.baseten import STT, TTS
from livekit.plugins.baseten.models import STTModels

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("model", ["whisper", "qwen3-asr"])
@pytest.mark.parametrize("invalid_timing", [False, True])
async def test_transcript_timing_is_validated_before_delivery(
    model: STTModels, invalid_timing: bool
) -> None:
    async def transcribe(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.receive()
        await ws.send_json(
            {
                "type": "transcription",
                "is_final": True,
                "is_end_of_audio_flush": True,
                "language_code": "en",
                "segments": [
                    {
                        "text": "Hello",
                        "start_time": 1.0,
                        "end_time": 1.5,
                        "word_timestamps": [
                            {
                                "word": "Hello",
                                "start_time": "invalid" if invalid_timing else 1.0,
                                "end_time": 1.5,
                                "prob": 0.9,
                            }
                        ],
                    }
                ],
            }
        )
        await ws.close()
        return ws

    app = web.Application()
    app.router.add_get("/stt", transcribe)
    async with TestServer(app) as server, aiohttp.ClientSession() as session:
        provider = STT(
            model=model,
            api_key="test",
            model_endpoint=str(server.make_url("/stt").with_scheme("ws")),
            show_word_timestamps=True,
            http_session=session,
        )
        async with provider.stream(conn_options=APIConnectOptions(max_retry=0)) as stream:
            stream.start_time_offset = 0.25
            stream.end_input()
            if invalid_timing:
                with pytest.raises(APIConnectionError, match="invalid transcript") as failure:
                    async for _ in stream:
                        pass
                assert failure.value.retryable is False
            else:
                async for event in stream:
                    if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                        speech = event.alternatives[0]
                        break
                else:
                    pytest.fail("No final transcript was delivered")
                assert speech.text == "Hello"
                assert speech.start_time == pytest.approx(1.0 + stream.start_time_offset)
                assert speech.end_time == pytest.approx(1.5 + stream.start_time_offset)
                assert speech.words is not None
                assert speech.words[0].start_time == speech.start_time
                assert speech.words[0].end_time == speech.end_time


async def test_qwen_voice_changes_apply_to_new_streams_without_changing_an_open_stream() -> None:
    async def synthesize(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        voice = ""
        async for message in ws:
            if message.type != aiohttp.WSMsgType.TEXT:
                continue
            payload = message.json()
            if payload["type"] == "session.config":
                voice = payload["voice"]
            elif payload["type"] == "input.done":
                await ws.send_json({"type": "audio.start"})
                await ws.send_bytes((b"\x01\x00" if voice == "first" else b"\x02\x00") * 2400)
                await ws.send_json({"type": "audio.done"})
                await ws.send_json({"type": "session.done"})
        return ws

    app = web.Application()
    app.router.add_get("/tts", synthesize)
    async with TestServer(app) as server, aiohttp.ClientSession() as session:
        provider = TTS(
            model="qwen3-tts",
            api_key="test",
            voice="first",
            model_endpoint=str(server.make_url("/tts").with_scheme("ws")),
            http_session=session,
        )
        try:
            async with provider.stream(conn_options=APIConnectOptions(max_retry=0)) as first:
                provider.update_options(voice="second")
                first.push_text("First voice.")
                first.end_input()
                heard = b"".join([event.frame.data.tobytes() async for event in first])
                assert heard == b"\x01\x00" * 2400
            async with provider.stream(conn_options=APIConnectOptions(max_retry=0)) as second:
                second.push_text("Second voice.")
                second.end_input()
                heard = b"".join([event.frame.data.tobytes() async for event in second])
                assert heard == b"\x02\x00" * 2400
        finally:
            await provider.aclose()


@pytest.mark.parametrize("stop_after_transcript", [False, True])
@pytest.mark.parametrize("expected_close", [False, True])
async def test_closed_transcription_peer_settles_all_transport_tasks(
    stop_after_transcript: bool, expected_close: bool
) -> None:
    terminations: list[object] = []

    async def transcribe(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.receive()
        if expected_close:
            message = await asyncio.wait_for(ws.receive(), timeout=1)
            terminations.append(message.json())
        await ws.send_json({"type": "transcription", "transcript": "Hello", "is_final": True})
        await ws.close()
        return ws

    app = web.Application()
    app.router.add_get("/stt", transcribe)
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    unhandled: list[dict[str, object]] = []
    loop.set_exception_handler(lambda _loop, context: unhandled.append(context))
    try:
        async with TestServer(app) as server, aiohttp.ClientSession() as session:
            provider = STT(
                model="whisper",
                api_key="test",
                model_endpoint=str(server.make_url("/stt").with_scheme("ws")),
                http_session=session,
            )
            try:
                async with provider.stream(conn_options=APIConnectOptions(max_retry=0)) as stream:
                    if expected_close:
                        stream.end_input()
                    if stop_after_transcript:
                        async for event in stream:
                            if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                                assert event.alternatives[0].text == "Hello"
                                break
                    elif expected_close:
                        events = [event async for event in stream]
                        assert any(
                            event.type == stt.SpeechEventType.FINAL_TRANSCRIPT
                            and event.alternatives[0].text == "Hello"
                            for event in events
                        )
                    else:
                        with pytest.raises(APIStatusError, match="closed unexpectedly"):
                            _ = [event async for event in stream]
                assert terminations == ([{"terminate_session": True}] if expected_close else [])
            finally:
                await provider.aclose()
            del stream, provider
        for _ in range(3):
            gc.collect()
            await asyncio.sleep(0)
        assert unhandled == []
    finally:
        loop.set_exception_handler(previous_handler)
