from __future__ import annotations

import asyncio
import base64
import functools
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, TypeVar

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
)
from livekit.agents.metrics import TTSMetrics
from livekit.agents.utils import http_context
from livekit.plugins.sprag import LLM, STT, TTS, __version__, tts as sprag_tts

pytestmark = pytest.mark.plugin("sprag")

ATTRIBUTION = f"livekit-agents/{__version__}"
NO_RETRY = APIConnectOptions(max_retry=0, timeout=5)
ONE_RETRY = APIConnectOptions(max_retry=1, retry_interval=0, timeout=5)
PCM = b"\x01\x00" * 2400  # 0.1 s of 24 kHz audio

F = TypeVar("F", bound=Callable[..., Awaitable[Any]])


def bounded(fn: F) -> F:
    """Fail a test that hangs, such as one waiting on a socket that was never closed."""

    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        return await asyncio.wait_for(fn(*args, **kwargs), timeout=10)

    return wrapper  # type: ignore[return-value]


async def _until(condition: Callable[[], bool]) -> None:
    while not condition():
        await asyncio.sleep(0.01)


@pytest.fixture(autouse=True)
def _api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPRAG_API_KEY", "test-key")


@pytest.fixture
async def http() -> AsyncIterator[aiohttp.ClientSession]:
    async with aiohttp.ClientSession() as session:
        yield session


async def _serve(routes: dict[str, Any]) -> TestServer:
    app = web.Application()
    for path, handler in routes.items():
        app.router.add_route("*", path, handler)
    test_server = TestServer(app)
    await test_server.start_server()
    return test_server


@pytest.mark.parametrize("cls", [LLM, STT, TTS])
def test_missing_api_key_raises(cls: type, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SPRAG_API_KEY", raising=False)
    with pytest.raises(ValueError, match="SPRAG_API_KEY"):
        cls()


@pytest.mark.parametrize(
    ("base_url", "ws_url"),
    [
        ("https://api.sprag.ai/v1", "wss://api.sprag.ai/v1"),
        ("https://api.sprag.ai/v1/", "wss://api.sprag.ai/v1"),
        ("http://localhost:8080/v1", "ws://localhost:8080/v1"),
    ],
)
def test_tts_derives_the_websocket_url(base_url: str, ws_url: str) -> None:
    assert TTS(base_url=base_url)._ws_url == ws_url


# -- LLM and STT --------------------------------------------------------------------------


@bounded
async def test_llm_sends_the_model_key_and_attribution() -> None:
    requests: list[tuple[web.Request, dict[str, Any]]] = []

    async def completions(request: web.Request) -> web.StreamResponse:
        requests.append((request, await request.json()))
        chunk = {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "symphony",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "hi"}}],
        }
        body = f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n"
        return web.Response(text=body, content_type="text/event-stream")

    from livekit.agents import llm as lk_llm

    test_server = await _serve({"/v1/chat/completions": completions})
    llm = LLM(base_url=str(test_server.make_url("/v1")))
    try:
        ctx = lk_llm.ChatContext.empty()
        ctx.add_message(role="user", content="Hello")
        text = ""
        async with llm.chat(chat_ctx=ctx) as stream:
            async for chunk in stream:
                if chunk.delta and chunk.delta.content:
                    text += chunk.delta.content
    finally:
        await llm.aclose()
        await test_server.close()

    assert text == "hi"
    request, body = requests[0]
    assert request.headers["Authorization"] == "Bearer test-key"
    assert request.headers["X-Sprag-Integration"] == ATTRIBUTION
    assert body["model"] == "symphony"
    assert llm.provider == "Sprag"


@bounded
async def test_stt_rest_sends_the_model_without_a_language() -> None:
    requests: list[tuple[web.Request, dict[str, Any]]] = []

    async def transcriptions(request: web.Request) -> web.Response:
        form = await request.post()
        requests.append((request, {k: v for k, v in form.items() if k != "file"}))
        return web.json_response({"text": "hello there"})

    test_server = await _serve({"/v1/audio/transcriptions": transcriptions})
    stt = STT(use_realtime=False, base_url=str(test_server.make_url("/v1")))
    try:
        frame = rtc.AudioFrame(
            data=PCM, sample_rate=24000, num_channels=1, samples_per_channel=2400
        )
        event = await stt.recognize(frame)
    finally:
        await stt.aclose()
        await test_server.close()

    assert event.alternatives[0].text == "hello there"
    request, form = requests[0]
    assert request.headers["Authorization"] == "Bearer test-key"
    assert request.headers["X-Sprag-Integration"] == ATTRIBUTION
    assert form["model"] == "rhythm"
    assert "language" not in form
    assert "languages" not in form


@bounded
async def test_stt_realtime_opens_a_transcription_session() -> None:
    handshakes: list[web.Request] = []
    first_event: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()

    async def realtime(request: web.Request) -> web.WebSocketResponse:
        handshakes.append(request)
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            if not first_event.done():
                first_event.set_result(json.loads(msg.data))
        return ws

    test_server = await _serve({"/v1/audio/transcriptions": realtime})
    async with http_context.open():
        stt = STT(
            base_url=str(test_server.make_url("/v1")),
            turn_detection={"silence_duration_ms": 200},
        )
        stream = stt.stream()
        try:
            silence = b"\x00\x00" * 480
            stream.push_frame(
                rtc.AudioFrame(
                    data=silence, sample_rate=24000, num_channels=1, samples_per_channel=480
                )
            )
            update = await asyncio.wait_for(first_event, timeout=5)
        finally:
            await stream.aclose()
            await stt.aclose()
            await test_server.close()

    request = handshakes[0]
    assert request.query["model"] == "rhythm"
    assert request.headers["Authorization"] == "Bearer test-key"
    assert request.headers["X-Sprag-Integration"] == ATTRIBUTION
    assert update["type"] == "session.update"
    session = update["session"]
    assert session["type"] == "transcription"
    assert session["audio"]["input"]["transcription"] == {"model": "rhythm"}
    turn_detection = session["audio"]["input"]["turn_detection"]
    assert turn_detection["type"] == "server_vad"
    assert turn_detection["silence_duration_ms"] == 200


# -- TTS against a fake speech session ---------------------------------------------------


class _SpeechServer:
    """Serves Sprag's streaming speech socket: responses queue and are spoken in order.

    ``script`` maps a response's index, counted across connections, to a behaviour:
    ``ok`` (the default), ``error:<code>`` (refused as it is asked for, or once the
    response ahead of it has begun to play), ``failed:<code>``,
    ``close:<code>``, ``stall`` (no reply), ``hang`` (one audio delta, then nothing), or
    ``slow`` (a pause mid-response), ``stale`` or ``stale:failed`` (a whole response
    tagged for another segment, then the requested one), or ``untagged`` (the response
    without its metadata).
    """

    BEHAVIOURS = {"ok", "error", "failed", "close", "stall", "hang", "slow", "stale", "untagged"}

    def __init__(self, *, reject_voice: bool = False, script: dict[int, str] | None = None):
        self.reject_voice = reject_voice
        self.script = script or {}
        self.connections: list[dict[str, str]] = []
        self.received: list[dict[str, Any]] = []
        self.spoken: list[tuple[int, str]] = []
        self.metadata: list[dict[str, str]] = []
        self.asked = 0
        self.asked_behind_another = 0
        self.closed_count = 0
        self.closed = asyncio.Event()
        self.first_audio = asyncio.Event()

    async def handle(self, request: web.Request) -> web.WebSocketResponse:
        index = len(self.connections)
        self.connections.append(
            {
                "model": request.query.get("model", ""),
                "authorization": request.headers.get("Authorization", ""),
                "attribution": request.headers.get("X-Sprag-Integration", ""),
            }
        )
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_json({"type": "session.created", "session": {}})
        pending: list[str] = []
        waiting: asyncio.Queue[tuple[str, str, dict[str, str]]] = asyncio.Queue()
        speaking = False

        async def speak_in_order() -> None:
            nonlocal speaking
            while True:
                behaviour, text, metadata = await waiting.get()
                self.spoken.append((index, text))
                self.metadata.append(metadata)
                speaking = True
                await self._respond(ws, behaviour, text, metadata)
                speaking = False

        speaker = asyncio.create_task(speak_in_order())
        async for msg in ws:
            event = json.loads(msg.data)
            self.received.append(event)
            if event["type"] == "session.update":
                if self.reject_voice:
                    error = {"code": "unknown_voice", "message": "unknown voice 'nobody'"}
                    await ws.send_json({"type": "error", "error": error})
                else:
                    await ws.send_json({"type": "session.updated", "session": event["session"]})
            elif event["type"] == "conversation.item.create":
                item = event["item"]
                parts = item.get("content") or []
                if item.get("type") != "message" or not all(
                    p.get("type") in ("input_text", "text") for p in parts
                ):
                    error = {"code": "unsupported_item", "message": "a speech session accepts text"}
                    await ws.send_json({"type": "error", "error": error})
                    continue
                pending.append("".join(p["text"] for p in parts))
                await ws.send_json({"type": "conversation.item.added", "item": {"id": "item_1"}})
                await ws.send_json({"type": "conversation.item.done", "item": {"id": "item_1"}})
            elif event["type"] == "response.create":
                behaviour = self.script.get(self.asked, "ok")
                self.asked += 1
                kind, _, code = behaviour.partition(":")
                if kind == "error":
                    if speaking or not waiting.empty():
                        await self.first_audio.wait()
                    self.spoken.append((index, "".join(pending)))
                    await ws.send_json({"type": "error", "error": {"code": code, "message": code}})
                    continue
                metadata = (event.get("response") or {}).get("metadata") or {}
                if speaking or not waiting.empty():
                    self.asked_behind_another += 1
                waiting.put_nowait((behaviour, "".join(pending), metadata))
                pending.clear()
        speaker.cancel()
        self.closed_count += 1
        self.closed.set()
        return ws

    async def _respond(
        self, ws: web.WebSocketResponse, behaviour: str, text: str, metadata: dict[str, str]
    ) -> None:
        kind, _, arg = behaviour.partition(":")
        assert kind in self.BEHAVIOURS, f"unknown behaviour {behaviour!r}"
        if kind == "close":
            await ws.close(code=int(arg))
            return
        if kind == "stall":
            await asyncio.Event().wait()
        if kind == "stale":
            stale = "failed:backend_error" if arg == "failed" else "ok"
            await self._respond(ws, stale, "stale", {"segment_id": "another"})
        if kind == "untagged":
            metadata = {}
        response = {"id": "resp_1", "metadata": metadata}
        await ws.send_json({"type": "response.created", "response": response})
        await ws.send_json({"type": "response.output_item.added", "item": {"id": "item_2"}})
        await ws.send_json({"type": "response.content_part.added", "part": {"type": "audio"}})
        await ws.send_json({"type": "response.output_audio_transcript.delta", "delta": text})
        if kind == "failed":
            details = {"type": "failed", "error": {"type": "server_error", "code": arg}}
            failed = {**response, "status": "failed", "status_details": details}
            await ws.send_json({"type": "response.done", "response": failed})
            return
        halves = (PCM[: len(PCM) // 2], PCM[len(PCM) // 2 :])
        for i, half in enumerate(halves):
            delta = base64.b64encode(half).decode()
            await ws.send_json({"type": "response.output_audio.delta", "delta": delta})
            self.first_audio.set()
            if kind == "hang":
                await asyncio.Event().wait()
            if kind == "slow" and i == 0:
                await asyncio.sleep(0.3)
        for done in (
            "response.output_audio.done",
            "response.output_audio_transcript.done",
            "response.content_part.done",
            "response.output_item.done",
        ):
            await ws.send_json({"type": done})
        completed = {**response, "status": "completed"}
        await ws.send_json({"type": "response.done", "response": completed})

    @property
    def texts(self) -> list[str]:
        return [text for _, text in self.spoken]


@pytest.fixture
async def speech() -> AsyncIterator[Callable[..., Awaitable[tuple[_SpeechServer, str]]]]:
    servers: list[TestServer] = []

    async def start(**kwargs: Any) -> tuple[_SpeechServer, str]:
        server = _SpeechServer(**kwargs)
        test_server = await _serve({"/v1/audio/speech": server.handle})
        servers.append(test_server)
        return server, str(test_server.make_url("/v1"))

    yield start
    for test_server in servers:
        await test_server.close()


async def _speak(tts: TTS, *chunks: str, conn_options: APIConnectOptions = NO_RETRY) -> float:
    stream = tts.stream(conn_options=conn_options)
    for chunk in chunks:
        stream.push_text(chunk)
    stream.flush()
    stream.end_input()
    duration = 0.0
    try:
        async for audio in stream:
            duration += audio.frame.duration
    finally:
        await stream.aclose()
    return duration


@bounded
async def test_tts_streams_one_response_per_sentence(
    speech: Any, http: aiohttp.ClientSession
) -> None:
    server, base_url = await speech()
    tts = TTS(base_url=base_url, http_session=http)
    try:
        duration = await _speak(
            tts, "It is lovely to meet you today. ", "How has your ", "week been going so far?"
        )
    finally:
        await tts.aclose()

    assert server.texts == [
        "It is lovely to meet you today.",
        "How has your week been going so far?",
    ]
    assert duration == pytest.approx(0.2, abs=0.01)


@bounded
async def test_tts_connects_with_the_key_attribution_and_voice(
    speech: Any, http: aiohttp.ClientSession
) -> None:
    server, base_url = await speech()
    tts = TTS(base_url=base_url, voice="serena", http_session=http)
    try:
        await _speak(tts, "Hello there.")
    finally:
        await tts.aclose()

    assert server.connections == [
        {"model": "chorus", "authorization": "Bearer test-key", "attribution": ATTRIBUTION}
    ]
    assert server.received[0]["session"]["audio"]["output"]["voice"] == "serena"


@bounded
async def test_tts_declares_its_sample_rate(speech: Any, http: aiohttp.ClientSession) -> None:
    server, base_url = await speech()
    tts = TTS(base_url=base_url, sample_rate=16000, http_session=http)
    try:
        duration = await _speak(tts, "Hello there.")
    finally:
        await tts.aclose()

    output = server.received[0]["session"]["audio"]["output"]
    assert output["format"] == {"type": "audio/pcm", "rate": 16000}
    # the fake sends 2400 samples, which last 0.15 s at 16 kHz
    assert duration == pytest.approx(0.15, abs=0.01)


@bounded
async def test_tts_sends_each_sentence_without_waiting_for_the_last(
    speech: Any, http: aiohttp.ClientSession
) -> None:
    server, base_url = await speech(script={0: "slow"})
    tts = TTS(base_url=base_url, http_session=http)
    try:
        duration = await _speak(tts, "The first sentence is slow. ", "The second one waits.")
    finally:
        await tts.aclose()

    assert server.asked_behind_another == 1
    assert server.texts == ["The first sentence is slow.", "The second one waits."]
    assert duration == pytest.approx(0.2, abs=0.01)
    assert len({m["segment_id"] for m in server.metadata}) == 1


@bounded
async def test_tts_ignores_audio_tagged_for_another_segment(
    speech: Any, http: aiohttp.ClientSession
) -> None:
    _, base_url = await speech(script={0: "stale"})
    tts = TTS(base_url=base_url, http_session=http)
    try:
        duration = await _speak(tts, "Only this sentence is heard.")
    finally:
        await tts.aclose()

    assert duration == pytest.approx(0.1, abs=0.01)


@bounded
async def test_tts_ignores_a_failure_tagged_for_another_segment(
    speech: Any, http: aiohttp.ClientSession
) -> None:
    _, base_url = await speech(script={0: "stale:failed"})
    tts = TTS(base_url=base_url, http_session=http)
    try:
        duration = await _speak(tts, "Only this sentence is heard.")
    finally:
        await tts.aclose()

    assert duration == pytest.approx(0.1, abs=0.01)


@bounded
async def test_tts_reports_a_response_without_a_segment_tag(
    speech: Any, http: aiohttp.ClientSession
) -> None:
    server, base_url = await speech(script={0: "untagged"})
    tts = TTS(base_url=base_url, http_session=http)
    try:
        with pytest.raises(APIStatusError, match="segment_id") as exc_info:
            await _speak(tts, "Hello there.", conn_options=ONE_RETRY)
    finally:
        await tts.aclose()

    assert not exc_info.value.retryable
    assert len(server.connections) == 1


@bounded
async def test_tts_fails_the_turn_when_a_queued_sentence_is_refused(
    speech: Any, http: aiohttp.ClientSession
) -> None:
    server, base_url = await speech(script={0: "slow", 1: "error:response_queue_full"})
    tts = TTS(base_url=base_url, http_session=http)
    try:
        with pytest.raises(APIStatusError, match="response_queue_full") as exc_info:
            await _speak(
                tts,
                "The first sentence is speaking. ",
                "The second is refused.",
                conn_options=ONE_RETRY,
            )
    finally:
        await tts.aclose()

    assert not exc_info.value.retryable
    assert server.first_audio.is_set()
    assert len(server.connections) == 1


@bounded
async def test_tts_turn_after_an_interruption_uses_a_fresh_connection(
    speech: Any, http: aiohttp.ClientSession
) -> None:
    server, base_url = await speech(script={0: "hang"})
    tts = TTS(base_url=base_url, http_session=http)
    try:
        stream = tts.stream(conn_options=NO_RETRY)
        stream.push_text("The first sentence hangs. The second is queued.")
        stream.end_input()
        await _until(lambda: server.asked == 2)
        await server.first_audio.wait()
        await stream.aclose()
        await asyncio.wait_for(server.closed.wait(), timeout=2)
        duration = await _speak(tts, "The next turn.")
    finally:
        await tts.aclose()

    assert duration == pytest.approx(0.1, abs=0.01)
    assert server.spoken[-1] == (1, "The next turn.")


@bounded
async def test_tts_waits_out_a_pause_between_sentences(
    speech: Any, http: aiohttp.ClientSession
) -> None:
    server, base_url = await speech()
    tts = TTS(base_url=base_url, http_session=http)
    stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, timeout=0.3))
    try:
        stream.push_text("The first sentence arrives at once. ")
        stream.push_text("The second sentence ")
        await _until(lambda: server.asked == 1)
        await asyncio.sleep(0.6)
        stream.push_text("comes after a pause longer than the timeout.")
        stream.end_input()
        duration = sum([audio.frame.duration async for audio in stream])
    finally:
        await stream.aclose()
        await tts.aclose()

    assert server.asked == 2
    assert duration == pytest.approx(0.2, abs=0.01)


@bounded
async def test_tts_synthesize_speaks_the_whole_text(
    speech: Any, http: aiohttp.ClientSession
) -> None:
    server, base_url = await speech()
    tts = TTS(base_url=base_url, http_session=http)
    try:
        text = "This is the first sentence to speak. This is the second one."
        async with tts.synthesize(text, conn_options=NO_RETRY) as stream:
            audio = b"".join([bytes(a.frame.data) async for a in stream])
    finally:
        await tts.aclose()

    assert server.texts == ["This is the first sentence to speak.", "This is the second one."]
    # the framework closes the clip with a short silence
    assert audio.startswith(PCM * 2)
    assert not audio[len(PCM) * 2 :].strip(b"\x00")


@bounded
@pytest.mark.parametrize("text", ["   ", ""])
async def test_tts_blank_input_opens_no_connection(
    speech: Any, http: aiohttp.ClientSession, text: str
) -> None:
    server, base_url = await speech()
    tts = TTS(base_url=base_url, http_session=http)
    try:
        assert await _speak(tts, text) == 0.0
    finally:
        await tts.aclose()
    assert server.connections == []


@bounded
async def test_tts_reuses_the_connection_across_turns(
    speech: Any, http: aiohttp.ClientSession
) -> None:
    server, base_url = await speech()
    tts = TTS(base_url=base_url, http_session=http)
    try:
        await _speak(tts, "First turn.")
        await _speak(tts, "Second turn.")
    finally:
        await tts.aclose()

    assert server.spoken == [(0, "First turn."), (0, "Second turn.")]


@bounded
async def test_tts_voice_change_reconnects_with_the_new_voice(
    speech: Any, http: aiohttp.ClientSession
) -> None:
    server, base_url = await speech()
    tts = TTS(base_url=base_url, voice="wade", http_session=http)
    try:
        await _speak(tts, "First turn.")
        tts.update_options(voice="wade")
        await _speak(tts, "Same voice.")
        tts.update_options(voice="serena")
        await _speak(tts, "New voice.")
    finally:
        await tts.aclose()

    assert [conn for conn, _ in server.spoken] == [0, 0, 1]
    updates = [e for e in server.received if e["type"] == "session.update"]
    assert [u["session"]["audio"]["output"]["voice"] for u in updates] == ["wade", "serena"]


@bounded
async def test_tts_model_change_reconnects_with_the_new_model(
    speech: Any, http: aiohttp.ClientSession
) -> None:
    server, base_url = await speech()
    tts = TTS(base_url=base_url, http_session=http)
    try:
        await _speak(tts, "First turn.")
        tts.update_options(model="chorus-next")
        await _speak(tts, "Second turn.")
    finally:
        await tts.aclose()

    assert [conn for conn, _ in server.spoken] == [0, 1]
    assert [c["model"] for c in server.connections] == ["chorus", "chorus-next"]


@bounded
async def test_tts_reports_connection_reuse_in_its_metrics(
    speech: Any, http: aiohttp.ClientSession
) -> None:
    _, base_url = await speech()
    tts = TTS(base_url=base_url, http_session=http)
    metrics: list[TTSMetrics] = []
    tts.on("metrics_collected", metrics.append)
    try:
        await _speak(tts, "First turn.")
        await _speak(tts, "Second turn.")
    finally:
        await tts.aclose()

    assert [m.connection_reused for m in metrics] == [False, True]
    assert metrics[0].acquire_time > 0 and metrics[1].acquire_time == 0
    assert all(m.audio_duration == pytest.approx(0.1, abs=0.01) for m in metrics)
    assert metrics[0].characters_count == len("First turn.")


@bounded
async def test_tts_prewarmed_connection_serves_the_first_turn(
    speech: Any, http: aiohttp.ClientSession
) -> None:
    server, base_url = await speech()
    tts = TTS(base_url=base_url, http_session=http)
    metrics: list[TTSMetrics] = []
    tts.on("metrics_collected", metrics.append)
    try:
        tts.prewarm()
        await _until(lambda: bool(server.connections))
        await asyncio.sleep(0.05)
        await _speak(tts, "Hello there.")
    finally:
        await tts.aclose()

    assert len(server.connections) == 1
    assert metrics[0].connection_reused


@bounded
async def test_tts_concurrent_streams_use_separate_connections(
    speech: Any, http: aiohttp.ClientSession
) -> None:
    server, base_url = await speech()
    tts = TTS(base_url=base_url, http_session=http)
    try:
        durations = await asyncio.gather(
            _speak(tts, "The first speaker."), _speak(tts, "The second speaker.")
        )
    finally:
        await tts.aclose()

    assert durations == [pytest.approx(0.1, abs=0.01)] * 2
    assert len(server.connections) == 2
    assert sorted(server.spoken) in (
        [(0, "The first speaker."), (1, "The second speaker.")],
        [(0, "The second speaker."), (1, "The first speaker.")],
    )


@bounded
async def test_tts_replaces_an_idle_connection(
    speech: Any, http: aiohttp.ClientSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sprag_tts, "IDLE_SOCKET_MAX_AGE", 0.3)
    server, base_url = await speech()
    tts = TTS(base_url=base_url, http_session=http)
    try:
        # each gap is shorter than the idle limit, though together they outlast it
        for turn in ("First turn.", "Second turn.", "Third turn."):
            await _speak(tts, turn)
            await asyncio.sleep(0.15)
        await asyncio.sleep(0.35)
        await _speak(tts, "Fourth turn.")
    finally:
        await tts.aclose()

    assert [conn for conn, _ in server.spoken] == [0, 0, 0, 1]
    await asyncio.wait_for(_until(lambda: server.closed_count == 2), timeout=2)


@bounded
async def test_tts_replaces_a_connection_past_its_lifetime(
    speech: Any, http: aiohttp.ClientSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sprag_tts, "SOCKET_MAX_LIFETIME", 1.0)
    server, base_url = await speech()
    tts = TTS(base_url=base_url, http_session=http)
    try:
        await _speak(tts, "First turn.")
        await _speak(tts, "Second turn.")
        await asyncio.sleep(1.1)
        await _speak(tts, "Third turn.")
        await asyncio.wait_for(server.closed.wait(), timeout=2)
    finally:
        await tts.aclose()

    assert [conn for conn, _ in server.spoken] == [0, 0, 1]


@bounded
async def test_tts_voice_change_mid_turn_applies_to_the_next_turn(
    speech: Any, http: aiohttp.ClientSession
) -> None:
    server, base_url = await speech(script={0: "slow"})
    tts = TTS(base_url=base_url, http_session=http)
    try:
        turn = asyncio.create_task(_speak(tts, "A slow first turn."))
        await server.first_audio.wait()
        tts.update_options(voice="serena")
        assert await turn == pytest.approx(0.1, abs=0.01)
        await _speak(tts, "Second turn.")
        await asyncio.wait_for(server.closed.wait(), timeout=2)
    finally:
        await tts.aclose()

    assert [conn for conn, _ in server.spoken] == [0, 1]
    updates = [e for e in server.received if e["type"] == "session.update"]
    assert updates[-1]["session"]["audio"]["output"]["voice"] == "serena"


@bounded
async def test_tts_reports_a_refused_voice_without_retrying(
    speech: Any, http: aiohttp.ClientSession
) -> None:
    server, base_url = await speech(reject_voice=True)
    tts = TTS(base_url=base_url, voice="nobody", http_session=http)
    try:
        with pytest.raises(APIStatusError, match="unknown voice") as exc_info:
            await _speak(tts, "Hello.", conn_options=ONE_RETRY)
        await asyncio.wait_for(server.closed.wait(), timeout=2)
    finally:
        await tts.aclose()

    assert not exc_info.value.retryable
    assert len(server.connections) == 1


def _holds(exc: BaseException, secret: str) -> bool:
    """Whether the exception, or any it chains to, holds the secret in its attributes."""
    pending: list[BaseException | None] = [exc]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        if secret in repr(current) or secret in repr(vars(current)):
            return True
        pending += [current.__cause__, current.__context__]
    return False


@bounded
async def test_tts_rejected_handshake_keeps_the_key_out_of_the_error(
    http: aiohttp.ClientSession,
) -> None:
    async def reject(request: web.Request) -> web.Response:
        return web.Response(status=403, text="refused")

    test_server = await _serve({"/v1/audio/speech": reject})
    tts = TTS(base_url=str(test_server.make_url("/v1")), http_session=http)
    try:
        with pytest.raises(APIStatusError) as streamed:
            await _speak(tts, "Hello.")
        # the stream re-raises inside the framework's own handlers, which replaces the
        # exception's context, so the connect step is checked where the error is built
        with pytest.raises(APIStatusError) as connected:
            await tts._connect_ws(timeout=5)
    finally:
        await tts.aclose()
        await test_server.close()

    assert streamed.value.status_code == 403
    assert not streamed.value.retryable
    assert not _holds(streamed.value, "test-key")
    assert not _holds(connected.value, "test-key")


@bounded
async def test_tts_retries_a_failed_handshake(http: aiohttp.ClientSession) -> None:
    server = _SpeechServer()
    attempts = 0

    async def flaky(request: web.Request) -> web.StreamResponse:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return web.Response(status=503, text="rolling out")
        return await server.handle(request)

    test_server = await _serve({"/v1/audio/speech": flaky})
    tts = TTS(base_url=str(test_server.make_url("/v1")), http_session=http)
    try:
        assert await _speak(tts, "Hello there.", conn_options=ONE_RETRY) == pytest.approx(
            0.1, abs=0.01
        )
    finally:
        await tts.aclose()
        await test_server.close()

    assert attempts == 2


@bounded
@pytest.mark.parametrize(
    ("behaviour", "retryable"),
    [
        ("error:input_too_long", False),
        ("failed:generation_timeout", False),
        ("failed:backend_error", True),
        ("error:model_at_capacity", True),
        ("error:model_not_running", False),
        ("close:1008", False),
        ("close:1011", True),
    ],
)
async def test_tts_retries_only_transient_failures(
    speech: Any, http: aiohttp.ClientSession, behaviour: str, retryable: bool
) -> None:
    server, base_url = await speech(script={0: behaviour})
    tts = TTS(base_url=base_url, http_session=http)
    try:
        if retryable:
            duration = await _speak(tts, "Hello there.", conn_options=ONE_RETRY)
            assert duration == pytest.approx(0.1, abs=0.01)
        else:
            with pytest.raises(APIStatusError) as exc_info:
                await _speak(tts, "Hello there.", conn_options=ONE_RETRY)
            assert not exc_info.value.retryable
    finally:
        await tts.aclose()

    if retryable:
        assert [conn for conn, _ in server.spoken] == [0, 1]
    else:
        assert len(server.spoken) == 1


@bounded
async def test_tts_times_out_a_stalled_response(speech: Any, http: aiohttp.ClientSession) -> None:
    server, base_url = await speech(script={0: "stall"})
    tts = TTS(base_url=base_url, http_session=http)
    try:
        with pytest.raises(APITimeoutError):
            await _speak(tts, "Hello.", conn_options=APIConnectOptions(max_retry=0, timeout=0.3))
    finally:
        await tts.aclose()


@bounded
async def test_tts_reports_a_handshake_that_never_answers(http: aiohttp.ClientSession) -> None:
    async def never(request: web.Request) -> web.Response:
        await asyncio.Event().wait()
        raise AssertionError

    test_server = await _serve({"/v1/audio/speech": never})
    tts = TTS(base_url=str(test_server.make_url("/v1")), http_session=http)
    try:
        with pytest.raises(APIConnectionError) as exc_info:
            await _speak(tts, "Hello.", conn_options=APIConnectOptions(max_retry=0, timeout=0.3))
    finally:
        await tts.aclose()
        await test_server.close()

    assert exc_info.value.retryable


@bounded
async def test_tts_reports_an_unreachable_server(http: aiohttp.ClientSession) -> None:
    test_server = await _serve({})
    base_url = str(test_server.make_url("/v1"))
    await test_server.close()
    tts = TTS(base_url=base_url, http_session=http)
    try:
        with pytest.raises(APIConnectionError) as exc_info:
            await _speak(tts, "Hello.")
    finally:
        await tts.aclose()

    assert exc_info.value.retryable
    assert not _holds(exc_info.value, "test-key")


@bounded
async def test_tts_aclose_closes_idle_and_active_connections(
    speech: Any, http: aiohttp.ClientSession
) -> None:
    server, base_url = await speech(script={2: "hang"})
    tts = TTS(base_url=base_url, http_session=http)
    await asyncio.gather(_speak(tts, "The first speaker."), _speak(tts, "The second speaker."))
    stream = tts.stream(conn_options=NO_RETRY)
    stream.push_text("A sentence still being spoken.")
    stream.flush()
    await server.first_audio.wait()
    await _until(lambda: server.asked == 3)
    await tts.aclose()

    await asyncio.wait_for(_until(lambda: server.closed_count == 2), timeout=2)
