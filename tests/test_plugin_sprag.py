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
from livekit.agents import APIConnectOptions, APIStatusError, APITimeoutError
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


# -- LLM and STT on the wire -------------------------------------------------------------


@bounded
async def test_llm_request_on_the_wire() -> None:
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
async def test_stt_rest_request_on_the_wire() -> None:
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
async def test_stt_realtime_session_on_the_wire() -> None:
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

    test_server = await _serve({"/v1/realtime": realtime})
    async with http_context.open():
        stt = STT(base_url=str(test_server.make_url("/v1")))
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
    assert request.query["intent"] == "transcription"
    assert request.headers["Authorization"] == "Bearer test-key"
    assert update["type"] == "session.update"
    session = update["session"]
    assert session["type"] == "transcription"
    assert session["audio"]["input"]["transcription"] == {"model": "rhythm"}
    assert session["audio"]["input"]["turn_detection"]["type"] == "server_vad"


# -- TTS against a fake speech session ---------------------------------------------------


class _SpeechServer:
    """Serves the realtime speech-session events the gateway sends, in its order.

    ``script`` maps a response's index, counted across connections, to a behaviour:
    ``ok`` (the default), ``error:<code>``, ``failed:<code>``, ``close:<code>``, ``stall``
    (no reply), ``hang`` (one audio delta, then nothing), or ``slow`` (a pause mid-response).
    """

    BEHAVIOURS = {"ok", "error", "failed", "close", "stall", "hang", "slow"}

    def __init__(self, *, reject_voice: bool = False, script: dict[int, str] | None = None):
        self.reject_voice = reject_voice
        self.script = script or {}
        self.connections: list[dict[str, str]] = []
        self.received: list[dict[str, Any]] = []
        self.spoken: list[tuple[int, str]] = []
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
                response_index = sum(1 for _ in self.spoken)
                text = "".join(pending)
                self.spoken.append((index, text))
                pending.clear()
                await self._respond(ws, self.script.get(response_index, "ok"), text)
        self.closed_count += 1
        self.closed.set()
        return ws

    async def _respond(self, ws: web.WebSocketResponse, behaviour: str, text: str) -> None:
        kind, _, arg = behaviour.partition(":")
        assert kind in self.BEHAVIOURS, f"unknown behaviour {behaviour!r}"
        if kind == "error":
            await ws.send_json({"type": "error", "error": {"code": arg, "message": arg}})
            return
        if kind == "close":
            await ws.close(code=int(arg))
            return
        if kind == "stall":
            return
        await ws.send_json({"type": "response.created", "response": {"id": "resp_1"}})
        await ws.send_json({"type": "response.output_item.added", "item": {"id": "item_2"}})
        await ws.send_json({"type": "response.content_part.added", "part": {"type": "audio"}})
        await ws.send_json({"type": "response.output_audio_transcript.delta", "delta": text})
        if kind == "failed":
            details = {"type": "failed", "error": {"type": "server_error", "code": arg}}
            response = {"status": "failed", "status_details": details}
            await ws.send_json({"type": "response.done", "response": response})
            return
        halves = (PCM[: len(PCM) // 2], PCM[len(PCM) // 2 :])
        for i, half in enumerate(halves):
            delta = base64.b64encode(half).decode()
            await ws.send_json({"type": "response.output_audio.delta", "delta": delta})
            self.first_audio.set()
            if kind == "hang":
                return
            if kind == "slow" and i == 0:
                await asyncio.sleep(0.3)
        for done in (
            "response.output_audio.done",
            "response.output_audio_transcript.done",
            "response.content_part.done",
            "response.output_item.done",
        ):
            await ws.send_json({"type": done})
        await ws.send_json({"type": "response.done", "response": {"status": "completed"}})

    @property
    def texts(self) -> list[str]:
        return [text for _, text in self.spoken]


@pytest.fixture
async def speech() -> AsyncIterator[Callable[..., Awaitable[tuple[_SpeechServer, str]]]]:
    servers: list[TestServer] = []

    async def start(**kwargs: Any) -> tuple[_SpeechServer, str]:
        server = _SpeechServer(**kwargs)
        test_server = await _serve({"/v1/realtime": server.handle})
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
    tts = TTS(base_url=base_url, voice="serena", instructions="Warm", http_session=http)
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
    assert server.connections == [
        {"model": "chorus-clone", "authorization": "Bearer test-key", "attribution": ATTRIBUTION}
    ]
    session = server.received[0]["session"]
    assert session["audio"]["output"]["voice"] == "serena"
    assert session["instructions"] == "Warm"


@bounded
async def test_tts_trailing_slash_in_base_url(speech: Any, http: aiohttp.ClientSession) -> None:
    server, base_url = await speech()
    tts = TTS(base_url=base_url + "/", http_session=http)
    try:
        assert await _speak(tts, "Hello there.") == pytest.approx(0.1, abs=0.01)
    finally:
        await tts.aclose()
    assert len(server.connections) == 1


@bounded
async def test_tts_synthesize(speech: Any, http: aiohttp.ClientSession) -> None:
    server, base_url = await speech()
    tts = TTS(base_url=base_url, http_session=http)
    try:
        duration = 0.0
        async with tts.synthesize("A single sentence to speak.", conn_options=NO_RETRY) as stream:
            async for audio in stream:
                duration += audio.frame.duration
    finally:
        await tts.aclose()

    assert server.texts == ["A single sentence to speak."]
    assert duration == pytest.approx(0.1, abs=0.01)


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
    metrics: list[TTSMetrics] = []
    tts.on("metrics_collected", metrics.append)
    try:
        await _speak(tts, "First turn.")
        await _speak(tts, "Second turn.")
        tts.update_options(voice="wade")
        await _speak(tts, "Third turn.")
        tts.update_options(voice="serena")
        await _speak(tts, "Fourth turn.")
    finally:
        await tts.aclose()

    assert server.texts == ["First turn.", "Second turn.", "Third turn.", "Fourth turn."]
    assert len(server.connections) == 2
    updates = [e for e in server.received if e["type"] == "session.update"]
    assert [u["session"]["audio"]["output"]["voice"] for u in updates] == ["wade", "serena"]
    assert [m.connection_reused for m in metrics] == [False, True, True, False]
    assert metrics[0].acquire_time > 0 and metrics[1].acquire_time == 0
    assert all(m.audio_duration == pytest.approx(0.1, abs=0.01) for m in metrics)
    assert metrics[0].characters_count == len("First turn.")


@bounded
async def test_tts_prewarm(speech: Any, http: aiohttp.ClientSession) -> None:
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
        await asyncio.gather(_speak(tts, "The first speaker."), _speak(tts, "The second speaker."))
    finally:
        await tts.aclose()

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
        await _speak(tts, "First turn.")
        await _speak(tts, "Second turn.")
        await asyncio.sleep(0.5)
        await _speak(tts, "Third turn.")
    finally:
        await tts.aclose()

    assert [conn for conn, _ in server.spoken] == [0, 0, 1]
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
async def test_tts_update_while_a_connection_is_in_use(
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
async def test_tts_surfaces_a_refused_session(speech: Any, http: aiohttp.ClientSession) -> None:
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


@bounded
async def test_tts_rejected_handshake_keeps_the_key_out_of_the_error(
    http: aiohttp.ClientSession,
) -> None:
    async def reject(request: web.Request) -> web.Response:
        return web.Response(status=403, text="refused")

    test_server = await _serve({"/v1/realtime": reject})
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
    exc = connected.value
    assert exc.__cause__ is None
    assert exc.__context__ is None
    assert "test-key" not in repr(exc)


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

    test_server = await _serve({"/v1/realtime": flaky})
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
async def test_tts_does_not_reuse_a_connection_after_a_failure(
    speech: Any, http: aiohttp.ClientSession
) -> None:
    server, base_url = await speech(script={0: "error:invalid_value"})
    tts = TTS(base_url=base_url, http_session=http)
    try:
        with pytest.raises(APIStatusError):
            await _speak(tts, "First turn.")
        await _speak(tts, "Second turn.")
        await asyncio.wait_for(server.closed.wait(), timeout=2)
    finally:
        await tts.aclose()

    assert [conn for conn, _ in server.spoken] == [0, 1]


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
async def test_tts_closes_an_interrupted_connection_promptly(
    speech: Any, http: aiohttp.ClientSession
) -> None:
    server, base_url = await speech(script={0: "hang"})
    tts = TTS(base_url=base_url, http_session=http)
    try:
        stream = tts.stream(conn_options=NO_RETRY)
        stream.push_text("A sentence the caller talks over.")
        stream.flush()
        await stream.__anext__()
        await stream.aclose()
        await asyncio.wait_for(server.closed.wait(), timeout=2)
    finally:
        await tts.aclose()
