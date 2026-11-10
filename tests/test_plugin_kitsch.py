import asyncio
import base64
import json
import wave
from contextlib import asynccontextmanager
from io import BytesIO

import pytest
from aiohttp import web

from livekit import rtc
from livekit.agents import APIConnectOptions, APIError, APIStatusError, stt
from livekit.plugins.kitsch import STT, TTS

pytestmark = pytest.mark.unit

PCM = b"\x01\x00" * 2400
OPTIONS = APIConnectOptions(timeout=2, max_retry=3)
PREFIX = "conversation.item.input_audio_transcription."


@asynccontextmanager
async def backend(*, failure=None, hold_http=False, automatic_final=False, delayed_final=False):
    calls = []
    complete_http = asyncio.Event()
    tts_closed = asyncio.Event()
    stt_closed = asyncio.Event()
    app = web.Application()

    async def tts_http(request):
        body = await request.json()
        calls.append((request.path, body))
        assert request.headers["Authorization"] == "Bearer test-key"
        assert request.query["output_format"] == "pcm_24000"
        if failure == "status":
            return web.json_response(
                {"error": "PRIVATE provider detail"},
                status=503,
                headers={"x-request-id": "request-1"},
            )
        if failure == "redirect":
            raise web.HTTPTemporaryRedirect("/unexpected")
        response = web.StreamResponse(headers={"Content-Type": "audio/pcm"})
        await response.prepare(request)
        data = b"\x01" if failure == "odd" else PCM
        await response.write(data[:17])
        await response.write(data[17:])
        if hold_http:
            await complete_http.wait()
        await response.write_eof()
        return response

    async def tts_ws(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        assert request.headers["Authorization"] == "Bearer test-key"
        calls.append(("tts-connect", dict(request.query)))
        sent = False
        try:
            async for message in ws:
                data = json.loads(message.data)
                calls.append(("tts", data))
                if failure == "ws-error":
                    await ws.send_json({"error": {"message": "PRIVATE provider detail"}})
                    continue
                if failure == "early-close":
                    await ws.close()
                    break
                if data.get("text") and (not sent):
                    payload = "???" if failure == "base64" else base64.b64encode(PCM).decode()
                    await ws.send_json({"audio": payload, "isFinal": False})
                    sent = True
                if data.get("flush") and failure != "no-final":
                    await ws.send_json({"audio": None, "isFinal": True})
        finally:
            tts_closed.set()
        return ws

    async def stt_http(request):
        reader = await request.multipart()
        fields = {}
        async for field in reader:
            fields[field.name] = bytes(await field.read())
        calls.append(("recognize", fields))
        if failure == "status":
            return web.json_response({}, status=429)
        with wave.open(BytesIO(fields["file"])) as wav:
            assert wav.getnchannels() == 1
            assert wav.getsampwidth() == 2
        return web.json_response({"text": "안녕하세요"}, headers={"x-request-id": "stt-1"})

    async def stt_ws(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        calls.append(("stt-connect", {}))
        await ws.send_json({"type": "session.created", "session": {"id": "session-1"}})
        count = 0
        dirty = False

        async def final():
            nonlocal count, dirty
            count += 1
            dirty = False
            result = {
                "type": PREFIX + "completed",
                "item_id": str(count),
                "transcript": "오늘은 네 시",
                "usage": {"seconds": 0.1},
            }
            await ws.send_json(result)
            await ws.send_json(result)
            await ws.send_json(
                {
                    "type": PREFIX + "partial",
                    "item_id": str(count),
                    "stable_text": "stale",
                    "provisional_text": "",
                }
            )

        try:
            async for message in ws:
                data = json.loads(message.data)
                calls.append(("stt", data))
                if data["type"] == "session.update":
                    session = data["session"]
                    if failure == "no-partial":
                        session["audio"]["input"]["transcription"]["include_partial"] = False
                    await ws.send_json({"type": "session.updated", "session": session})
                elif data["type"] == "input_audio_buffer.append":
                    assert len(base64.b64decode(data["audio"])) <= 256 * 1024
                    dirty = True
                    if automatic_final:
                        await ws.send_json(
                            {"type": "input_audio_buffer.committed", "item_id": str(count + 1)}
                        )
                    for provisional in ("세 시", "네 시", ""):
                        await ws.send_json(
                            {
                                "type": PREFIX + "partial",
                                "item_id": str(count + 1),
                                "stable_text": "오늘은 ",
                                "provisional_text": provisional,
                            }
                        )
                    if automatic_final:
                        if delayed_final:
                            dirty = False
                        else:
                            await final()
                elif data["type"] == "input_audio_buffer.commit":
                    if failure == "early-close":
                        await ws.close()
                        break
                    if dirty:
                        await ws.send_json(
                            {
                                "type": "input_audio_buffer.committed",
                                "item_id": str(count + 1),
                            }
                        )
                        if failure != "no-final":
                            await final()
                    else:
                        await ws.send_json({"type": "error", "error": {"code": "buffer_empty"}})
                        if delayed_final and failure != "no-final":
                            await asyncio.sleep(0.05)
                            await final()
        finally:
            stt_closed.set()
        return ws

    app.router.add_post("/v1/text-to-speech/{voice}/stream", tts_http)
    app.router.add_get("/v1/text-to-speech/{voice}/stream-input", tts_ws)
    app.router.add_post("/v1/speech-to-text", stt_http)
    app.router.add_get("/v1/speech-to-text/stream", stt_ws)
    runner = web.AppRunner(app, shutdown_timeout=1)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        yield (f"http://127.0.0.1:{port}", calls, complete_http, tts_closed, stt_closed)
    finally:
        complete_http.set()
        await runner.cleanup()


async def collect(stream):
    async with stream:
        return [event async for event in stream]


async def test_http_tts_yields_before_eof_and_preserves_pcm():
    async with (
        backend(hold_http=True) as (url, calls, done, *_),
        TTS(voice_id="minjun", api_key="test-key", base_url=url) as tts,
    ):
        async with tts.synthesize("안녕", conn_options=OPTIONS) as stream:
            first = await asyncio.wait_for(anext(stream), 2)
            assert not done.is_set()
            assert first.frame.sample_rate == 24000
            assert first.frame.num_channels == 1
            done.set()
            events = [first] + [e async for e in stream]
        assert b"".join(e.frame.data.cast("B") for e in events) == PCM
        assert len(calls) == 1


async def test_ws_tts_yields_before_flush_and_finishes_once():
    async with backend() as (url, calls, _, closed, _):
        async with (
            TTS(voice_id="minjun", language_code="ko", api_key="test-key", base_url=url) as tts,
            tts.stream(conn_options=OPTIONS) as stream,
        ):
            stream.push_text("안녕하세요. ")
            first = await asyncio.wait_for(anext(stream), 2)
            assert not any((d.get("flush") for p, d in calls if p == "tts"))
            stream.push_text("다음 문장입니다.")
            stream.end_input()
            events = [first] + [e async for e in stream]
            assert sum(e.is_final for e in events) == 1
            assert b"".join(e.frame.data.cast("B") for e in events) == PCM
        await asyncio.wait_for(closed.wait(), 2)
        assert calls[0][1] == {"output_format": "pcm_24000", "language_code": "ko"}
        assert not any((d.get("text") == "" for p, d in calls if p == "tts"))


@pytest.mark.parametrize("failure", ["ws-error", "early-close", "base64"])
async def test_ws_tts_errors_never_retry(failure):
    async with (
        backend(failure=failure) as (url, calls, *_),
        TTS(voice_id="minjun", api_key="test-key", base_url=url) as tts,
    ):
        stream = tts.stream(conn_options=OPTIONS)
        stream.push_text("안녕")
        stream.end_input()
        with pytest.raises(APIError) as exc:
            await collect(stream)
        assert "PRIVATE" not in str(exc.value)
        assert sum((p == "tts-connect" for p, _ in calls)) == 1


@pytest.mark.parametrize("failure", ["status", "redirect", "odd"])
async def test_http_tts_errors_never_retry(failure):
    async with (
        backend(failure=failure) as (url, calls, *_),
        TTS(voice_id="minjun", api_key="test-key", base_url=url) as tts,
    ):
        with pytest.raises(APIError) as exc:
            await collect(tts.synthesize("안녕", conn_options=OPTIONS))
        assert "PRIVATE" not in str(exc.value)
        assert len(calls) == 1
        if failure == "status":
            assert isinstance(exc.value, APIStatusError)
            assert exc.value.request_id == "request-1"


async def test_tts_interrupt_closes_socket():
    async with (
        backend() as (url, _, _, closed, _),
        TTS(voice_id="minjun", api_key="test-key", base_url=url) as tts,
    ):
        stream = tts.stream(conn_options=OPTIONS)
        stream.push_text("안녕")
        await asyncio.wait_for(anext(stream), 2)
        await stream.aclose()
        await asyncio.wait_for(closed.wait(), 2)


async def test_file_stt_and_language_override():
    async with (
        backend() as (url, calls, *_),
        STT(api_key="test-key", base_url=url, language="en", prompt="Kitsch") as engine,
    ):
        frame = rtc.AudioFrame(PCM, 24000, 1, 2400)
        result = await engine.recognize(frame, language="ko", conn_options=OPTIONS)
        assert result.alternatives[0].text == "안녕하세요"
        assert result.request_id == "stt-1"
        assert calls[0][1]["language"] == b"ko"
        assert calls[0][1]["model"] == b"kitsch-stt-v1"


async def test_stt_partial_replaces_and_final_is_not_duplicated():
    async with backend() as (url, calls, _, _, closed):
        async with STT(api_key="test-key", base_url=url, language="ko") as engine:
            stream = engine.stream(conn_options=OPTIONS)
            stream.push_frame(rtc.AudioFrame(PCM, 48000, 1, 2400))
            stream.end_input()
            events = await asyncio.wait_for(collect(stream), 4)
            texts = [e.alternatives[0].text for e in events if e.alternatives]
            assert "오늘은 세 시" in texts
            assert "오늘은 네 시" in texts
            assert "오늘은 " in texts
            assert "stale" not in texts
            finals = [e for e in events if e.type == stt.SpeechEventType.FINAL_TRANSCRIPT]
            assert len(finals) == 1
            assert finals[0].alternatives[0].text == "오늘은 네 시"
        await asyncio.wait_for(closed.wait(), 2)
        chunks = [
            base64.b64decode(d["audio"])
            for p, d in calls
            if p == "stt" and d["type"] == "input_audio_buffer.append"
        ]
        assert abs(sum(map(len, chunks)) - 2400) <= 4


@pytest.mark.parametrize("failure", ["early-close", "no-partial"])
async def test_stt_stream_fails_without_reconnect(failure):
    async with (
        backend(failure=failure) as (url, calls, *_),
        STT(api_key="test-key", base_url=url) as engine,
    ):
        stream = engine.stream(conn_options=OPTIONS)
        stream.push_frame(rtc.AudioFrame(PCM, 24000, 1, 2400))
        stream.end_input()
        with pytest.raises(APIError):
            await asyncio.wait_for(collect(stream), 4)
        assert sum((p == "stt-connect" for p, _ in calls)) == 1


async def test_file_stt_never_retries():
    async with (
        backend(failure="status") as (url, calls, *_),
        STT(api_key="test-key", base_url=url) as engine,
    ):
        with pytest.raises(APIStatusError):
            await engine.recognize(rtc.AudioFrame(PCM, 24000, 1, 2400), conn_options=OPTIONS)
        assert len(calls) == 1


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com",
        "https://key@host",
        "https://host/path",
        "https://host?key=x",
    ],
)
def test_reject_unsafe_origin(url):
    with pytest.raises(ValueError):
        TTS(voice_id="minjun", api_key="test-key", base_url=url)


def test_missing_api_key(monkeypatch):
    monkeypatch.delenv("KITSCH_API_KEY", raising=False)
    with pytest.raises(ValueError):
        STT()


async def test_stt_continuous_turns_use_one_connection_and_drain_last_final():
    async with (
        backend() as (url, calls, _, _, closed),
        STT(api_key="test-key", base_url=url) as engine,
    ):
        stream = engine.stream(conn_options=OPTIONS)
        events = []
        first_final = asyncio.Event()

        async def consume():
            async for event in stream:
                events.append(event)
                if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                    first_final.set()

        async with stream:
            consumer = asyncio.create_task(consume())
            try:
                stream.push_frame(rtc.AudioFrame(PCM, 24000, 1, 2400))
                stream.flush()
                await asyncio.wait_for(first_final.wait(), 2)
                stream.push_frame(rtc.AudioFrame(PCM, 24000, 1, 2400))
                stream.end_input()
                await asyncio.wait_for(consumer, 3)
            finally:
                consumer.cancel()
                await asyncio.gather(consumer, return_exceptions=True)
        assert sum(e.type == stt.SpeechEventType.FINAL_TRANSCRIPT for e in events) == 2
        assert sum(p == "stt-connect" for p, _ in calls) == 1
        await asyncio.wait_for(closed.wait(), 2)


async def test_server_vad_final_followed_by_flush_does_not_duplicate_or_hang():
    async with (
        backend(automatic_final=True) as (url, calls, *_),
        STT(api_key="test-key", base_url=url) as engine,
    ):
        async with engine.stream(conn_options=OPTIONS) as stream:
            stream.push_frame(rtc.AudioFrame(PCM, 24000, 1, 2400))
            events = []
            async for event in stream:
                events.append(event)
                if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                    break
            stream.end_input()
            events.extend([e async for e in stream])
        assert sum(e.type == stt.SpeechEventType.FINAL_TRANSCRIPT for e in events) == 1
        assert sum(p == "stt-connect" for p, _ in calls) == 1


@pytest.mark.parametrize("failure", [None, "no-final"])
async def test_server_vad_empty_buffer_waits_for_pending_final(failure):
    async with (
        backend(automatic_final=True, delayed_final=True, failure=failure) as (url, calls, *_),
        STT(api_key="test-key", base_url=url) as engine,
    ):
        options = APIConnectOptions(timeout=0.2, max_retry=3)
        async with engine.stream(conn_options=options) as stream:
            stream.push_frame(rtc.AudioFrame(PCM, 24000, 1, 2400))
            # The preceding committed event must be read before we flush.
            await asyncio.wait_for(anext(stream), 2)
            stream.end_input()
            if failure:
                with pytest.raises(APIError):
                    await asyncio.wait_for(collect(stream), 2)
            else:
                events = await asyncio.wait_for(collect(stream), 2)
                finals = [e for e in events if e.type == stt.SpeechEventType.FINAL_TRANSCRIPT]
                assert [e.alternatives[0].text for e in finals] == ["오늘은 네 시"]
        assert sum(p == "stt-connect" for p, _ in calls) == 1


async def test_stt_cancel_closes_socket_without_replay():
    async with (
        backend() as (url, calls, _, _, closed),
        STT(api_key="test-key", base_url=url) as engine,
    ):
        stream = engine.stream(conn_options=OPTIONS)
        stream.push_frame(rtc.AudioFrame(PCM, 24000, 1, 2400))
        await asyncio.wait_for(anext(stream), 2)
        await stream.aclose()
        await asyncio.wait_for(closed.wait(), 2)
        assert sum(p == "stt-connect" for p, _ in calls) == 1


async def test_tts_empty_stream_does_not_open_connection():
    async with (
        backend() as (url, calls, *_),
        TTS(voice_id="minjun", api_key="test-key", base_url=url) as engine,
    ):
        stream = engine.stream(conn_options=OPTIONS)
        stream.end_input()
        assert await collect(stream) == []
        assert calls == []


async def test_local_validation_rejects_new_tts_segment_and_stereo_stt():
    async with (
        TTS(voice_id="minjun", api_key="test-key") as tts_engine,
        STT(api_key="test-key") as stt_engine,
    ):
        stream = tts_engine.stream(conn_options=OPTIONS)
        stream.flush()
        with pytest.raises(ValueError, match="new TTS stream"):
            stream.push_text("second segment")
        await stream.aclose()
        stream = stt_engine.stream(conn_options=OPTIONS)
        with pytest.raises(ValueError, match="mono"):
            stream.push_frame(rtc.AudioFrame(PCM, 24000, 2, 1200))
        await stream.aclose()


@pytest.mark.parametrize("kind", ["tts", "stt"])
async def test_final_timeout_never_replays(kind):
    async with backend(failure="no-final") as (url, calls, *_):
        options = APIConnectOptions(timeout=0.2, max_retry=3)
        if kind == "tts":
            engine = TTS(voice_id="minjun", api_key="test-key", base_url=url)
            stream = engine.stream(conn_options=options)
            stream.push_text("안녕")
        else:
            engine = STT(api_key="test-key", base_url=url)
            stream = engine.stream(conn_options=options)
            stream.push_frame(rtc.AudioFrame(PCM, 24000, 1, 2400))
        async with engine:
            stream.end_input()
            with pytest.raises(APIError):
                await asyncio.wait_for(collect(stream), 3)
        assert sum(p == kind + "-connect" for p, _ in calls) == 1
