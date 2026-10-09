from __future__ import annotations

import asyncio
import base64
import json

import aiohttp
import pytest
from aiohttp import web

from livekit import rtc
from livekit.agents import APIStatusError, vad
from livekit.agents.stt import SpeechEventType
from livekit.plugins.qwen_asr import STT

pytestmark = pytest.mark.unit


async def _serve(app: web.Application) -> tuple[web.AppRunner, int]:
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    sockets = site._server.sockets if site._server is not None else None
    assert sockets
    return runner, sockets[0].getsockname()[1]


def _frame(samples: int = 1600) -> rtc.AudioFrame:
    return rtc.AudioFrame(b"\x00\x01" * samples, 16000, 1, samples)


@pytest.mark.asyncio
async def test_batch_omits_prompt_and_language_when_unset() -> None:
    seen: dict[str, object] = {}

    async def transcribe(request: web.Request) -> web.Response:
        form = await request.post()
        seen["fields"] = set(form.keys())
        return web.json_response({"text": "Evet buyurun."})

    app = web.Application()
    app.router.add_post("/v1/audio/transcriptions", transcribe)
    runner, port = await _serve(app)
    try:
        stt = STT(base_url=f"http://127.0.0.1:{port}/v1", model="qwen3-asr-1.7b")
        event = await stt.recognize(_frame())
        await stt.aclose()
    finally:
        await runner.cleanup()

    assert event.alternatives[0].text == "Evet buyurun."
    assert seen["fields"] == {"file", "model"}


@pytest.mark.asyncio
async def test_batch_sends_prompt_language_and_keyterms() -> None:
    seen: dict[str, str] = {}

    async def transcribe(request: web.Request) -> web.Response:
        form = await request.post()
        seen["language"] = form["language"]
        seen["prompt"] = form["prompt"]
        seen["model"] = form["model"]
        return web.json_response({"text": "türevi alınmış"})

    app = web.Application()
    app.router.add_post("/v1/audio/transcriptions", transcribe)
    runner, port = await _serve(app)
    try:
        stt = STT(
            base_url=f"http://127.0.0.1:{port}/v1",
            model="qwen3-asr-1.7b",
            language="tr",
            prompt="Türkçe ders. Konu türev.",
        )
        stt._update_session_keyterms(["Fibabanka", "diferansiyel"])
        event = await stt.recognize(_frame())
        await stt.aclose()
    finally:
        await runner.cleanup()

    assert event.alternatives[0].text == "türevi alınmış"
    assert seen["model"] == "qwen3-asr-1.7b"
    assert seen["language"] == "tr"
    assert "Türkçe ders. Konu türev." in seen["prompt"]
    assert "Vocabulary: Fibabanka, diferansiyel" in seen["prompt"]


@pytest.mark.asyncio
async def test_realtime_maps_delta_and_done() -> None:
    seen: list[dict[str, object]] = []

    async def realtime(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_json({"type": "session.created", "id": "sess-test"})
        async for msg in ws:
            if msg.type != aiohttp.WSMsgType.TEXT:
                continue
            event = json.loads(msg.data)
            seen.append(event)
            if event.get("type") == "input_audio_buffer.commit" and event.get("final") is True:
                await ws.send_json({"type": "transcription.delta", "delta": "Evet "})
                await ws.send_json(
                    {
                        "type": "transcription.done",
                        "text": "Evet buyurun.",
                        "usage": {"prompt_tokens": 4, "completion_tokens": 2},
                    }
                )
                break
        await ws.close()
        return ws

    app = web.Application()
    app.router.add_get("/v1/realtime", realtime)
    runner, port = await _serve(app)
    try:
        stt = STT(
            base_url=f"http://127.0.0.1:{port}/v1",
            model="qwen3-asr-1.7b",
            language="tr",
            prompt="Türkçe telefon.",
            use_realtime=True,
            vad=None,
        )
        stream = stt.stream()
        stream.push_frame(_frame(800))
        stream.end_input()
        events = [event async for event in stream]
        await stt.aclose()
    finally:
        await runner.cleanup()

    assert seen[0] == {
        "type": "session.update",
        "model": "qwen3-asr-1.7b",
        "language": "tr",
        "prompt": "Türkçe telefon.",
    }
    assert {"type": "input_audio_buffer.commit", "final": False} in seen
    assert any(event.get("type") == "input_audio_buffer.append" for event in seen)
    assert seen[-1] == {"type": "input_audio_buffer.commit", "final": True}

    kinds = [event.type for event in events]
    assert SpeechEventType.START_OF_SPEECH in kinds
    assert SpeechEventType.INTERIM_TRANSCRIPT in kinds
    assert SpeechEventType.FINAL_TRANSCRIPT in kinds
    final = next(event for event in events if event.type == SpeechEventType.FINAL_TRANSCRIPT)
    interim = next(event for event in events if event.type == SpeechEventType.INTERIM_TRANSCRIPT)
    assert interim.alternatives[0].text == "Evet "
    assert final.alternatives[0].text == "Evet buyurun."
    usage = next(event for event in events if event.type == SpeechEventType.RECOGNITION_USAGE)
    assert usage.recognition_usage is not None
    assert usage.recognition_usage.input_tokens == 4
    assert usage.recognition_usage.output_tokens == 2


@pytest.mark.asyncio
async def test_realtime_hides_language_preamble() -> None:
    async def realtime(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_json({"type": "session.created", "id": "sess-test"})
        async for msg in ws:
            if msg.type != aiohttp.WSMsgType.TEXT:
                continue
            event = json.loads(msg.data)
            if event.get("type") == "input_audio_buffer.commit" and event.get("final") is True:
                for delta in ("language Turkish", "<asr_text>", "Evet"):
                    await ws.send_json({"type": "transcription.delta", "delta": delta})
                await ws.send_json(
                    {
                        "type": "transcription.done",
                        "text": "language Turkish<asr_text>Evet",
                    }
                )
                break
        await ws.close()
        return ws

    app = web.Application()
    app.router.add_get("/v1/realtime", realtime)
    runner, port = await _serve(app)
    try:
        stt = STT(
            base_url=f"http://127.0.0.1:{port}/v1",
            model="qwen3-asr-1.7b",
            use_realtime=True,
            vad=None,
        )
        stream = stt.stream()
        stream.push_frame(_frame(800))
        stream.end_input()
        events = [event async for event in stream]
        await stt.aclose()
    finally:
        await runner.cleanup()

    texts = [
        event.alternatives[0].text
        for event in events
        if event.type in (SpeechEventType.INTERIM_TRANSCRIPT, SpeechEventType.FINAL_TRANSCRIPT)
    ]
    assert texts
    assert all("<asr_text>" not in text and not text.startswith("language") for text in texts)
    assert texts[-1] == "Evet"


@pytest.mark.asyncio
async def test_batch_error_omits_response_body() -> None:
    secret = "müşteri Ayşe, prompt: Fibabanka kredisi"

    async def transcribe(_request: web.Request) -> web.Response:
        return web.json_response({"error": {"message": secret}}, status=400)

    app = web.Application()
    app.router.add_post("/v1/audio/transcriptions", transcribe)
    runner, port = await _serve(app)
    try:
        stt = STT(base_url=f"http://127.0.0.1:{port}/v1", model="qwen3-asr-1.7b")
        with pytest.raises(APIStatusError) as caught:
            await stt.recognize(_frame())
        await stt.aclose()
    finally:
        await runner.cleanup()

    assert caught.value.status_code == 400
    assert caught.value.body is None
    assert secret not in str(caught.value)
    assert secret not in repr(caught.value)


class _LateSpeech:
    """VAD that classifies only after the input loop has finished."""

    def stream(self) -> _LateSpeechStream:
        return _LateSpeechStream()


class _LateSpeechStream:
    def __init__(self) -> None:
        self._pcm = bytearray()
        self._events: asyncio.Queue[vad.VADEvent | None] = asyncio.Queue()

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        self._pcm.extend(bytes(frame.data))

    def flush(self) -> None:
        # RecognizeStream.end_input() flushes before the VAD task runs.
        # Queued audio has to stay available for the late start event.
        return None

    def end_input(self) -> None:
        # Snapshot ends before the last half-second, which the input loop has
        # already buffered. Both pieces must reach the server.
        consumed = max(0, len(self._pcm) // 2 - 8000)
        onset = bytes(self._pcm[: consumed * 2])
        self._events.put_nowait(
            vad.VADEvent(
                type=vad.VADEventType.START_OF_SPEECH,
                samples_index=consumed,
                timestamp=0.0,
                speech_duration=consumed / 16000,
                silence_duration=0.0,
                frames=[rtc.AudioFrame(onset, 16000, 1, consumed)] if onset else [],
            )
        )
        self._events.put_nowait(
            vad.VADEvent(
                type=vad.VADEventType.END_OF_SPEECH,
                samples_index=len(self._pcm) // 2,
                timestamp=0.0,
                speech_duration=len(self._pcm) / 2 / 16000,
                silence_duration=0.2,
                frames=[],
            )
        )
        self._events.put_nowait(None)

    async def aclose(self) -> None:
        return None

    def __aiter__(self) -> _LateSpeechStream:
        return self

    async def __anext__(self) -> vad.VADEvent:
        event = await self._events.get()
        if event is None:
            raise StopAsyncIteration
        return event


@pytest.mark.asyncio
async def test_realtime_keeps_speech_when_vad_lags() -> None:
    audio = bytearray()

    async def realtime(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_json({"type": "session.created", "id": "sess-test"})
        async for msg in ws:
            if msg.type != aiohttp.WSMsgType.TEXT:
                continue
            event = json.loads(msg.data)
            if event.get("type") == "input_audio_buffer.append":
                audio.extend(base64.b64decode(event["audio"]))
            if event.get("type") == "input_audio_buffer.commit" and event.get("final") is True:
                await ws.send_json({"type": "transcription.done", "text": "tamamı"})
                break
        await ws.close()
        return ws

    spoken = b"\x11\x22" * 16000 + b"\x33\x44" * 16000
    app = web.Application()
    app.router.add_get("/v1/realtime", realtime)
    runner, port = await _serve(app)
    try:
        stt = STT(
            base_url=f"http://127.0.0.1:{port}/v1",
            model="qwen3-asr-1.7b",
            use_realtime=True,
            vad=_LateSpeech(),  # type: ignore[arg-type]
        )
        stream = stt.stream()
        for start in range(0, len(spoken), 3200):
            chunk = spoken[start : start + 3200]
            stream.push_frame(rtc.AudioFrame(chunk, 16000, 1, len(chunk) // 2))
        stream.end_input()
        events = [event async for event in stream]
        await stt.aclose()
    finally:
        await runner.cleanup()

    assert bytes(audio) == spoken
    final = next(event for event in events if event.type == SpeechEventType.FINAL_TRANSCRIPT)
    assert final.alternatives[0].text == "tamamı"


class _TwoTurnVAD:
    """Two utterances already buffered before either VAD event is delivered."""

    def __init__(self, first: bytes, second: bytes) -> None:
        self._first = first
        self._second = second

    def stream(self) -> _TwoTurnStream:
        return _TwoTurnStream(self._first, self._second)


class _TwoTurnStream:
    def __init__(self, first: bytes, second: bytes) -> None:
        self._first = first
        self._second = second
        self._events: asyncio.Queue[vad.VADEvent | None] = asyncio.Queue()

    def push_frame(self, _frame: rtc.AudioFrame) -> None:
        return None

    def flush(self) -> None:
        return None

    def end_input(self) -> None:
        cuts = (len(self._first) // 2, (len(self._first) + len(self._second)) // 2)
        pieces = (self._first, self._second)
        for end, piece in zip(cuts, pieces, strict=True):
            self._events.put_nowait(
                vad.VADEvent(
                    type=vad.VADEventType.START_OF_SPEECH,
                    samples_index=end,
                    timestamp=0.0,
                    speech_duration=len(piece) / 2 / 16000,
                    silence_duration=0.0,
                    frames=[rtc.AudioFrame(piece, 16000, 1, len(piece) // 2)],
                )
            )
            self._events.put_nowait(
                vad.VADEvent(
                    type=vad.VADEventType.END_OF_SPEECH,
                    samples_index=end,
                    timestamp=0.0,
                    speech_duration=len(piece) / 2 / 16000,
                    silence_duration=0.2,
                    frames=[],
                )
            )
        self._events.put_nowait(None)

    async def aclose(self) -> None:
        return None

    def __aiter__(self) -> _TwoTurnStream:
        return self

    async def __anext__(self) -> vad.VADEvent:
        event = await self._events.get()
        if event is None:
            raise StopAsyncIteration
        return event


@pytest.mark.asyncio
async def test_realtime_keeps_queued_turns_separate() -> None:
    turns: list[bytes] = []
    current = bytearray()

    async def realtime(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_json({"type": "session.created", "id": "sess-test"})
        async for msg in ws:
            if msg.type != aiohttp.WSMsgType.TEXT:
                continue
            event = json.loads(msg.data)
            if event.get("type") == "input_audio_buffer.append":
                current.extend(base64.b64decode(event["audio"]))
            if event.get("type") == "input_audio_buffer.commit" and event.get("final") is True:
                turns.append(bytes(current))
                current.clear()
                await ws.send_json({"type": "transcription.done", "text": f"t{len(turns)}"})
        return ws

    # Identical PCM: a byte search would keep the later copy and drop the first.
    phrase = b"\x11\x22" * 16000
    app = web.Application()
    app.router.add_get("/v1/realtime", realtime)
    runner, port = await _serve(app)
    try:
        stt = STT(
            base_url=f"http://127.0.0.1:{port}/v1",
            model="qwen3-asr-1.7b",
            use_realtime=True,
            vad=_TwoTurnVAD(phrase, phrase),  # type: ignore[arg-type]
        )
        stream = stt.stream()
        for start in range(0, len(phrase) * 2, 3200):
            chunk = (phrase + phrase)[start : start + 3200]
            stream.push_frame(rtc.AudioFrame(chunk, 16000, 1, len(chunk) // 2))
        stream.end_input()
        _events = [event async for event in stream]
        await stt.aclose()
    finally:
        await runner.cleanup()

    assert turns == [phrase, phrase]


class _EarlyStart:
    """VAD that opens the turn after the first frame, then ends at input end."""

    def stream(self) -> _EarlyStartStream:
        return _EarlyStartStream()


class _EarlyStartStream:
    def __init__(self) -> None:
        self._pcm = bytearray()
        self._started = False
        self._events: asyncio.Queue[vad.VADEvent | None] = asyncio.Queue()

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        self._pcm.extend(bytes(frame.data))
        if not self._started:
            self._started = True
            samples = len(self._pcm) // 2
            self._events.put_nowait(
                vad.VADEvent(
                    type=vad.VADEventType.START_OF_SPEECH,
                    samples_index=samples,
                    timestamp=0.0,
                    speech_duration=samples / 16000,
                    silence_duration=0.0,
                    frames=[rtc.AudioFrame(bytes(self._pcm), 16000, 1, samples)],
                )
            )

    def flush(self) -> None:
        return None

    def end_input(self) -> None:
        samples = len(self._pcm) // 2
        self._events.put_nowait(
            vad.VADEvent(
                type=vad.VADEventType.END_OF_SPEECH,
                samples_index=samples,
                timestamp=0.0,
                speech_duration=samples / 16000,
                silence_duration=0.2,
                frames=[],
            )
        )
        self._events.put_nowait(None)

    async def aclose(self) -> None:
        return None

    def __aiter__(self) -> _EarlyStartStream:
        return self

    async def __anext__(self) -> vad.VADEvent:
        event = await self._events.get()
        if event is None:
            raise StopAsyncIteration
        return event


@pytest.mark.asyncio
async def test_realtime_streams_audio_after_speech_starts() -> None:
    audio = bytearray()

    async def realtime(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_json({"type": "session.created", "id": "sess-test"})
        async for msg in ws:
            if msg.type != aiohttp.WSMsgType.TEXT:
                continue
            event = json.loads(msg.data)
            if event.get("type") == "input_audio_buffer.append":
                audio.extend(base64.b64decode(event["audio"]))
            if event.get("type") == "input_audio_buffer.commit" and event.get("final") is True:
                await ws.send_json({"type": "transcription.done", "text": "devam"})
                break
        await ws.close()
        return ws

    spoken = b"\x11\x22" * 1600 + b"\x33\x44" * 16000
    app = web.Application()
    app.router.add_get("/v1/realtime", realtime)
    runner, port = await _serve(app)
    try:
        stt = STT(
            base_url=f"http://127.0.0.1:{port}/v1",
            model="qwen3-asr-1.7b",
            use_realtime=True,
            vad=_EarlyStart(),  # type: ignore[arg-type]
        )
        stream = stt.stream()
        for start in range(0, len(spoken), 3200):
            chunk = spoken[start : start + 3200]
            stream.push_frame(rtc.AudioFrame(chunk, 16000, 1, len(chunk) // 2))
        stream.end_input()
        _events = [event async for event in stream]
        await stt.aclose()
    finally:
        await runner.cleanup()

    assert bytes(audio) == spoken


class _LateEnd:
    """START is early. END for the midpoint arrives only after later audio is queued."""

    def stream(self) -> _LateEndStream:
        return _LateEndStream()


class _LateEndStream:
    def __init__(self) -> None:
        self._pcm = bytearray()
        self._started = False
        self._events: asyncio.Queue[vad.VADEvent | None] = asyncio.Queue()

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        self._pcm.extend(bytes(frame.data))
        if not self._started:
            self._started = True
            samples = len(self._pcm) // 2
            self._events.put_nowait(
                vad.VADEvent(
                    type=vad.VADEventType.START_OF_SPEECH,
                    samples_index=samples,
                    timestamp=0.0,
                    speech_duration=samples / 16000,
                    silence_duration=0.0,
                    frames=[rtc.AudioFrame(bytes(self._pcm), 16000, 1, samples)],
                )
            )

    def flush(self) -> None:
        return None

    def end_input(self) -> None:
        midpoint = 3200
        second = bytes(self._pcm[midpoint * 2 :])
        self._events.put_nowait(
            vad.VADEvent(
                type=vad.VADEventType.END_OF_SPEECH,
                samples_index=midpoint,
                timestamp=0.0,
                speech_duration=midpoint / 16000,
                silence_duration=0.25,
                frames=[],
            )
        )
        self._events.put_nowait(
            vad.VADEvent(
                type=vad.VADEventType.START_OF_SPEECH,
                samples_index=len(self._pcm) // 2,
                timestamp=0.0,
                speech_duration=len(second) / 2 / 16000,
                silence_duration=0.0,
                frames=[rtc.AudioFrame(second, 16000, 1, len(second) // 2)],
            )
        )
        self._events.put_nowait(
            vad.VADEvent(
                type=vad.VADEventType.END_OF_SPEECH,
                samples_index=len(self._pcm) // 2,
                timestamp=0.0,
                speech_duration=len(second) / 2 / 16000,
                silence_duration=0.25,
                frames=[],
            )
        )
        self._events.put_nowait(None)

    async def aclose(self) -> None:
        return None

    def __aiter__(self) -> _LateEndStream:
        return self

    async def __anext__(self) -> vad.VADEvent:
        event = await self._events.get()
        if event is None:
            raise StopAsyncIteration
        return event


@pytest.mark.asyncio
async def test_realtime_does_not_merge_audio_before_end() -> None:
    turns: list[bytes] = []
    current = bytearray()

    async def realtime(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_json({"type": "session.created", "id": "sess-test"})
        async for msg in ws:
            if msg.type != aiohttp.WSMsgType.TEXT:
                continue
            event = json.loads(msg.data)
            if event.get("type") == "input_audio_buffer.append":
                current.extend(base64.b64decode(event["audio"]))
            if event.get("type") == "input_audio_buffer.commit" and event.get("final") is True:
                turns.append(bytes(current))
                current.clear()
                await ws.send_json({"type": "transcription.done", "text": f"t{len(turns)}"})
        return ws

    first = b"\x11\x22" * 3200
    second = b"\x33\x44" * 3200
    app = web.Application()
    app.router.add_get("/v1/realtime", realtime)
    runner, port = await _serve(app)
    try:
        stt = STT(
            base_url=f"http://127.0.0.1:{port}/v1",
            model="qwen3-asr-1.7b",
            use_realtime=True,
            vad=_LateEnd(),  # type: ignore[arg-type]
        )
        stream = stt.stream()
        for start in range(0, len(first + second), 3200):
            chunk = (first + second)[start : start + 3200]
            stream.push_frame(rtc.AudioFrame(chunk, 16000, 1, len(chunk) // 2))
        stream.end_input()
        _events = [event async for event in stream]
        await stt.aclose()
    finally:
        await runner.cleanup()

    assert turns == [first, second]


class _FlushBoundary:
    def stream(self) -> _FlushBoundaryStream:
        return _FlushBoundaryStream()


class _FlushBoundaryStream:
    """First END index lags the pushed length; the next epoch must not reuse it."""

    def __init__(self) -> None:
        self._pcm = bytearray()
        self._flushed = False
        self._events: asyncio.Queue[vad.VADEvent | None] = asyncio.Queue()

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        self._pcm.extend(bytes(frame.data))

    def flush(self) -> None:
        if self._flushed:
            return
        self._flushed = True
        end = 1536
        onset = bytes(self._pcm[: end * 2])
        self._events.put_nowait(
            vad.VADEvent(
                type=vad.VADEventType.START_OF_SPEECH,
                samples_index=end,
                timestamp=0.0,
                speech_duration=end / 16000,
                silence_duration=0.0,
                frames=[rtc.AudioFrame(onset, 16000, 1, end)],
            )
        )
        self._events.put_nowait(
            vad.VADEvent(
                type=vad.VADEventType.END_OF_SPEECH,
                samples_index=end,
                timestamp=0.0,
                speech_duration=end / 16000,
                silence_duration=0.2,
                frames=[],
            )
        )

    def end_input(self) -> None:
        rest = bytes(self._pcm[1600 * 2 :])
        samples = len(rest) // 2
        self._events.put_nowait(
            vad.VADEvent(
                type=vad.VADEventType.START_OF_SPEECH,
                samples_index=samples,
                timestamp=0.0,
                speech_duration=samples / 16000,
                silence_duration=0.0,
                frames=[rtc.AudioFrame(rest, 16000, 1, samples)] if rest else [],
            )
        )
        self._events.put_nowait(
            vad.VADEvent(
                type=vad.VADEventType.END_OF_SPEECH,
                samples_index=samples,
                timestamp=0.0,
                speech_duration=samples / 16000,
                silence_duration=0.2,
                frames=[],
            )
        )
        self._events.put_nowait(None)

    async def aclose(self) -> None:
        return None

    def __aiter__(self) -> _FlushBoundaryStream:
        return self

    async def __anext__(self) -> vad.VADEvent:
        event = await self._events.get()
        if event is None:
            raise StopAsyncIteration
        return event


@pytest.mark.asyncio
async def test_realtime_flush_keeps_the_next_turn_aligned() -> None:
    turns: list[bytes] = []
    current = bytearray()

    async def realtime(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_json({"type": "session.created", "id": "sess-test"})
        async for msg in ws:
            if msg.type != aiohttp.WSMsgType.TEXT:
                continue
            event = json.loads(msg.data)
            if event.get("type") == "input_audio_buffer.append":
                current.extend(base64.b64decode(event["audio"]))
            if event.get("type") == "input_audio_buffer.commit" and event.get("final") is True:
                turns.append(bytes(current))
                current.clear()
                await ws.send_json({"type": "transcription.done", "text": f"t{len(turns)}"})
        return ws

    first = b"\x11\x22" * 1600
    second = b"\x33\x44" * 512
    app = web.Application()
    app.router.add_get("/v1/realtime", realtime)
    runner, port = await _serve(app)
    try:
        stt = STT(
            base_url=f"http://127.0.0.1:{port}/v1",
            model="qwen3-asr-1.7b",
            use_realtime=True,
            vad=_FlushBoundary(),  # type: ignore[arg-type]
        )
        stream = stt.stream()
        stream.push_frame(rtc.AudioFrame(first, 16000, 1, len(first) // 2))
        stream.flush()
        stream.push_frame(rtc.AudioFrame(second, 16000, 1, len(second) // 2))
        stream.end_input()
        _events = [event async for event in stream]
        await stt.aclose()
    finally:
        await runner.cleanup()

    assert turns == [first[: 1536 * 2], second]
