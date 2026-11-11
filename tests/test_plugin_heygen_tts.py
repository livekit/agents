from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any
from unittest.mock import patch

import aiohttp
import pytest
from aiohttp import web

from livekit.agents import (
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tts,
)
from livekit.agents.types import USERDATA_TIMED_TRANSCRIPT
from livekit.plugins.heygen import TTS
from livekit.plugins.heygen.tts import DEFAULT_VOICE_ID

pytestmark = pytest.mark.unit

SAMPLE_RATE = 24000
# Long enough that the default sentence tokenizer keeps them apart.
FIRST = "This is the first sentence of the turn."
SECOND = "And this is the second one that follows it."
NO_RETRY = APIConnectOptions(max_retry=0, timeout=2)
RETRY = APIConnectOptions(max_retry=2, timeout=2, retry_interval=0.05)


def _pcm(ms: int) -> bytes:
    return b"\x10\x00" * (SAMPLE_RATE * ms // 1000)


def _frame(context_id: str, text: str, audio_ms: int = 100) -> dict[str, Any]:
    """One audio message, with character timings that restart at 0 as the server's do."""
    step = audio_ms // len(text)
    return {
        "audio": base64.b64encode(_pcm(audio_ms)).decode(),
        "isFinal": None,
        "contextId": context_id,
        "alignment": {
            "chars": list(text),
            "charStartTimesMs": [i * step for i in range(len(text))],
            "charDurationsMs": [step] * len(text),
        },
    }


def _final(context_id: str) -> dict[str, Any]:
    return {"isFinal": True, "contextId": context_id, "synthesizedChars": 1}


Script = Callable[[web.WebSocketResponse, "_Server"], Awaitable[None]]


async def _answer_each_closed_context(ws: web.WebSocketResponse, server: _Server) -> None:
    async for msg in ws:
        data = json.loads(msg.data)
        server.messages.append(data)
        if data.get("close_context"):
            await ws.send_json(_frame(data["context_id"], "spoken "))
            await ws.send_json(_final(data["context_id"]))


class _Server:
    """Local stand-in for the HeyGen API, so the real wire path runs end to end."""

    def __init__(
        self,
        *scripts: Script,
        refusals: list[tuple[int, dict[str, str]]] | None = None,
    ) -> None:
        self.scripts = list(scripts)
        self.refusals = list(refusals or [])
        self.handshakes = 0
        self.connections: list[web.Request] = []
        self.messages: list[dict[str, Any]] = []
        self.disconnected = asyncio.Event()
        self.posts: list[tuple[web.Request, dict[str, Any]]] = []
        self.post_responses: list[web.Response] = []

    async def _on_socket(self, request: web.Request) -> web.StreamResponse:
        self.handshakes += 1
        if self.refusals:
            status, headers = self.refusals.pop(0)
            return web.json_response({"error": {"code": "nope"}}, status=status, headers=headers)

        self.connections.append(request)
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        script = self.scripts.pop(0) if self.scripts else _answer_each_closed_context
        try:
            await script(ws, self)
        finally:
            self.disconnected.set()
        return ws

    async def _on_post(self, request: web.Request) -> web.StreamResponse:
        self.posts.append((request, await request.json()))
        if self.post_responses:
            return self.post_responses.pop(0)
        return web.Response(body=_pcm(100), content_type="audio/pcm")

    async def __aenter__(self) -> _Server:
        app = web.Application()
        app.router.add_get("/v3/models/audio/tts/multi-stream-input", self._on_socket)
        app.router.add_post("/v3/models/audio/tts/stream", self._on_post)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await self._site.start()
        self.base_url = f"http://127.0.0.1:{self._runner.addresses[0][1]}"
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.session.close()
        await self._runner.cleanup()

    def tts(self, **kwargs: Any) -> TTS:
        return TTS(api_key="test-key", base_url=self.base_url, http_session=self.session, **kwargs)

    def sent(self, field: str) -> list[dict[str, Any]]:
        return [m for m in self.messages if field in m]


async def _turn(
    tts_instance: tts.TTS, *tokens: str, conn_options: APIConnectOptions = NO_RETRY
) -> tuple[bytes, list[Any]]:
    stream = tts_instance.stream(conn_options=conn_options)
    for token in tokens:
        stream.push_text(token)
    stream.end_input()
    audio, timed = bytearray(), []
    try:
        async for ev in stream:
            audio.extend(ev.frame.data.tobytes())
            timed.extend(ev.frame.userdata.get(USERDATA_TIMED_TRANSCRIPT, []))
    finally:
        await stream.aclose()
    return bytes(audio), timed


async def _synthesize(
    tts_instance: TTS, text: str = "hello there", conn_options: APIConnectOptions = NO_RETRY
) -> bytes:
    stream = tts_instance.synthesize(text, conn_options=conn_options)
    audio = bytearray()
    try:
        async for ev in stream:
            audio.extend(ev.frame.data.tobytes())
    finally:
        await stream.aclose()
    return bytes(audio)


@pytest.fixture
async def server() -> AsyncIterator[_Server]:
    async with _Server() as s:
        yield s


def test_requires_api_key() -> None:
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="API key"):
            TTS()


async def test_api_key_from_env(server: _Server) -> None:
    with patch.dict("os.environ", {"HEYGEN_API_KEY": "env-key"}):
        tts_instance = TTS(base_url=server.base_url, http_session=server.session)
    await _synthesize(tts_instance)

    assert server.posts[0][0].headers["x-api-key"] == "env-key"


def test_warns_that_http_ignores_enable_logging(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING", logger="livekit.plugins.heygen"):
        TTS(api_key="test-key", streaming=False, enable_logging=False)

    assert "enable_logging" in caplog.text


def test_refuses_a_sample_rate_the_api_does_not_serve() -> None:
    with pytest.raises(ValueError, match="sample_rate"):
        TTS(api_key="test-key", sample_rate=32000)


async def test_handshake_authenticates_and_carries_the_options(server: _Server) -> None:
    tts_instance = server.tts(
        language="en-US",
        sample_rate=16000,
        end_user_id="user-1",
        apply_text_normalization="off",
        enable_ssml_parsing=True,
        enable_logging=False,
    )
    await _turn(tts_instance, FIRST)

    request = server.connections[0]
    assert request.headers["x-api-key"] == "test-key"
    assert {
        "voice_id": DEFAULT_VOICE_ID,
        "model": "heygen-voice-1-turbo",
        "language": "en",
        "end_user_id": "user-1",
        "output_format": "pcm_16000",
        "auto_mode": "true",
        "apply_text_normalization": "off",
        "enable_ssml_parsing": "true",
        "enable_logging": "false",
    }.items() <= dict(request.query).items()


async def test_turn_is_sent_as_spaced_sentences_with_one_flush_and_one_close(
    server: _Server,
) -> None:
    await _turn(server.tts(), f"{FIRST} ", SECOND)

    context_id = server.messages[0]["context_id"]
    assert server.messages == [
        {"text": f"{FIRST} ", "context_id": context_id},
        {"text": f"{SECOND} ", "context_id": context_id},
        {"context_id": context_id, "flush": True},
        {"context_id": context_id, "close_context": True},
    ]


async def test_audio_and_word_times_follow_the_audio_clock(server: _Server) -> None:
    async def speak(ws: web.WebSocketResponse, server: _Server) -> None:
        async for msg in ws:
            data = json.loads(msg.data)
            if data.get("close_context"):
                context_id = data["context_id"]
                # The second word splits across messages. The clock restarts in each message.
                await ws.send_json(_frame(context_id, "Hi the", audio_ms=600))
                await ws.send_json(_frame(context_id, "re pal", audio_ms=600))
                await ws.send_json({"audio": None, "alignment": None, "contextId": context_id})
                await ws.send_json(_final(context_id))

    async with _Server(speak) as s:
        audio, timed = await _turn(s.tts(), FIRST)

    assert audio[: len(_pcm(1200))] == _pcm(1200)
    assert [str(word) for word in timed] == ["Hi ", "there ", "pal"]
    assert [(word.start_time, word.end_time) for word in timed] == [
        (0.0, 0.3),
        (0.3, 0.9),
        (0.9, 1.2),
    ]


async def test_parsed_break_is_not_a_timed_word() -> None:
    async def speak(ws: web.WebSocketResponse, server: _Server) -> None:
        async for msg in ws:
            data = json.loads(msg.data)
            if data.get("close_context"):
                context_id = data["context_id"]
                await ws.send_json(_frame(context_id, "Wait. <pause:1s> Now<pause:1s>.", 1000))
                await ws.send_json(_final(context_id))

    async with _Server(speak) as s:
        _, timed = await _turn(s.tts(), FIRST)

    assert [str(word) for word in timed] == ["Wait. ", "Now."]
    assert timed[1].start_time > timed[0].start_time


async def test_turn_without_text_sends_nothing(server: _Server) -> None:
    audio, _ = await _turn(server.tts())

    assert audio == b""
    assert server.messages == []


async def test_turns_share_a_socket_but_never_a_context(server: _Server) -> None:
    tts_instance = server.tts()
    await _turn(tts_instance, FIRST)
    await _turn(tts_instance, SECOND)
    await tts_instance.aclose()

    assert server.handshakes == 1
    first, second = (m["context_id"] for m in server.sent("close_context"))
    assert first != second


async def test_socket_the_server_closed_while_idle_is_replaced() -> None:
    async def close_after_turn(ws: web.WebSocketResponse, server: _Server) -> None:
        async for msg in ws:
            data = json.loads(msg.data)
            if data.get("close_context"):
                await ws.send_json(_frame(data["context_id"], "spoken "))
                await ws.send_json(_final(data["context_id"]))
                await ws.close(code=1000, message=b"no message within inactivity_timeout")

    async with _Server(close_after_turn) as s:
        tts_instance = s.tts()
        await _turn(tts_instance, FIRST)
        await asyncio.sleep(0.05)
        audio, _ = await _turn(tts_instance, SECOND, conn_options=RETRY)
        await tts_instance.aclose()

    assert audio
    assert s.handshakes == 2


async def test_barge_in_drops_the_running_context_with_its_socket() -> None:
    async def speak_forever(ws: web.WebSocketResponse, server: _Server) -> None:
        msg = await ws.receive()
        context_id = json.loads(msg.data)["context_id"]
        while not ws.closed:
            await ws.send_json(_frame(context_id, "stale "))
            await asyncio.sleep(0.01)

    async with _Server(speak_forever) as s:
        tts_instance = s.tts()
        stream = tts_instance.stream(conn_options=NO_RETRY)
        stream.push_text(FIRST)
        stream.end_input()
        await stream.__anext__()
        await stream.aclose()
        await asyncio.wait_for(s.disconnected.wait(), 2)

        _, timed = await _turn(tts_instance, SECOND)
        await tts_instance.aclose()

    assert s.handshakes == 2
    assert "stale" not in "".join(str(word) for word in timed)


async def test_cancel_before_any_text_sends_nothing_and_keeps_the_socket(server: _Server) -> None:
    tts_instance = server.tts()
    stream = tts_instance.stream(conn_options=NO_RETRY)
    await asyncio.sleep(0.05)
    await stream.aclose()
    assert server.messages == []

    await _turn(tts_instance, FIRST)
    await tts_instance.aclose()

    assert server.handshakes == 1


async def test_frames_of_another_context_stay_out_of_the_turn() -> None:
    async def speak(ws: web.WebSocketResponse, server: _Server) -> None:
        async for msg in ws:
            data = json.loads(msg.data)
            if data.get("close_context"):
                await ws.send_json(_frame("old-context", "stale "))
                await ws.send_json(_final("old-context"))
                await ws.send_json(_frame(data["context_id"], "spoken "))
                await ws.send_json(_final(data["context_id"]))

    async with _Server(speak) as s:
        audio, timed = await _turn(s.tts(), FIRST)

    assert [str(word) for word in timed] == ["spoken "]
    assert audio[: len(_pcm(100))] == _pcm(100)


async def test_context_the_server_ended_is_not_reused() -> None:
    async def end_after_first_text(ws: web.WebSocketResponse, server: _Server) -> None:
        msg = await ws.receive()
        first = json.loads(msg.data)
        server.messages.append(first)
        await ws.send_json(_frame(first["context_id"], "first "))
        await ws.send_json(_final(first["context_id"]))
        await _answer_each_closed_context(ws, server)

    async with _Server(end_after_first_text) as s:
        tts_instance = s.tts()
        stream = tts_instance.stream(conn_options=NO_RETRY)
        stream.push_text(f"{FIRST} {SECOND} ")
        await stream.__anext__()
        await asyncio.sleep(0.05)
        stream.end_input()
        async for _ in stream:
            pass
        await stream.aclose()
        await tts_instance.aclose()

    first_context = s.messages[0]["context_id"]
    closed = s.sent("close_context")
    assert len(closed) == 1
    assert closed[0]["context_id"] != first_context


async def test_turn_ends_when_the_server_already_ended_its_only_context() -> None:
    async def end_early(ws: web.WebSocketResponse, server: _Server) -> None:
        msg = await ws.receive()
        context_id = json.loads(msg.data)["context_id"]
        await ws.send_json(_frame(context_id, "spoken "))
        await ws.send_json(_final(context_id))
        await _answer_each_closed_context(ws, server)

    async with _Server(end_early) as s:
        tts_instance = s.tts()
        stream = tts_instance.stream(conn_options=NO_RETRY)
        stream.push_text(FIRST)
        stream.flush()
        await stream.__anext__()
        await asyncio.sleep(0.05)
        stream.end_input()

        async def drain() -> None:
            async for _ in stream:
                pass

        await asyncio.wait_for(drain(), 2)
        await stream.aclose()
        await tts_instance.aclose()

    assert s.sent("flush") == []


async def test_pause_in_the_text_is_not_a_timeout(server: _Server) -> None:
    stream = server.tts().stream(conn_options=APIConnectOptions(max_retry=0, timeout=0.1))
    stream.push_text(f"{FIRST} {SECOND} ")
    await asyncio.sleep(0.4)
    stream.end_input()
    audio = bytearray()
    async for ev in stream:
        audio.extend(ev.frame.data.tobytes())
    await stream.aclose()

    assert audio


async def test_silence_is_a_timeout_even_when_the_server_keeps_pinging() -> None:
    async def only_ping(ws: web.WebSocketResponse, server: _Server) -> None:
        await ws.receive()
        while not ws.closed:
            await ws.ping()
            await asyncio.sleep(0.02)

    async with _Server(only_ping) as s:
        with pytest.raises(APITimeoutError):
            await _turn(s.tts(), FIRST, conn_options=APIConnectOptions(max_retry=0, timeout=0.1))


def _closes_with(code: int, error: dict[str, Any] | None = None) -> Script:
    async def script(ws: web.WebSocketResponse, server: _Server) -> None:
        await ws.receive()
        if error is not None:
            await ws.send_json({"message": error["message"], "error": error})
        await ws.close(code=code, message=b"closing")

    return script


async def test_refused_turn_is_not_retried_and_keeps_code_and_request_id() -> None:
    refusal = {
        "code": "rate_limit_exceeded",
        "message": "Too many requests",
        "request_id": "req-1",
    }
    async with _Server(_closes_with(1008, refusal)) as s:
        with pytest.raises(APIStatusError) as raised:
            await _turn(s.tts(), FIRST, conn_options=RETRY)

    assert s.handshakes == 1
    assert raised.value.status_code == 1008
    assert raised.value.request_id == "req-1"
    assert "rate_limit_exceeded" in raised.value.message
    assert raised.value.retryable is False


@pytest.mark.parametrize(
    ("code", "error"),
    [
        pytest.param(1011, {"code": "internal_error", "message": "failed"}, id="inference-failed"),
        pytest.param(1012, None, id="restart"),
        pytest.param(1013, {"code": "rate_limit_exceeded", "message": "full"}, id="capacity"),
    ],
)
async def test_failed_turn_is_retried_on_a_new_socket(
    code: int, error: dict[str, Any] | None
) -> None:
    async with _Server(_closes_with(code, error)) as s:
        audio, _ = await _turn(s.tts(), FIRST, conn_options=RETRY)

    assert audio
    assert s.handshakes == 2


@pytest.mark.parametrize(
    ("status", "headers"),
    [
        pytest.param(401, {}, id="bad-key"),
        pytest.param(429, {"Retry-After": "60"}, id="rate-limit"),
    ],
)
async def test_refused_handshake_is_not_retried_and_keeps_request_id(
    status: int, headers: dict[str, str]
) -> None:
    async with _Server(refusals=[(status, {"X-Request-ID": "req-2"} | headers)]) as s:
        with pytest.raises(APIStatusError) as raised:
            await _turn(s.tts(), FIRST, conn_options=RETRY)

    assert s.handshakes == 1
    assert raised.value.status_code == status
    assert raised.value.request_id == "req-2"


async def test_full_socket_cap_is_retried() -> None:
    async with _Server(refusals=[(429, {"Retry-After": "1"})]) as s:
        audio, _ = await _turn(s.tts(), FIRST, conn_options=RETRY)

    assert audio
    assert s.handshakes == 2


async def test_refused_voice_points_at_the_http_path() -> None:
    async with _Server(refusals=[(400, {})]) as s:
        with pytest.raises(APIStatusError, match="streaming=False"):
            await _turn(s.tts(), FIRST, conn_options=RETRY)

    assert s.handshakes == 1


@pytest.mark.parametrize(
    ("enable_ssml_parsing", "expected"),
    [
        pytest.param(False, "Wait for the next part of it now.", id="removed"),
        pytest.param(
            True, 'Wait <break time="1s"></break> for the next part of it now.', id="kept"
        ),
    ],
)
async def test_break_tags_only_reach_a_server_told_to_accept_them(
    server: _Server, enable_ssml_parsing: bool, expected: str
) -> None:
    tts_instance = server.tts(enable_ssml_parsing=enable_ssml_parsing)
    text = 'Wait <break time="1s"></break> for the next part of it now.'
    await _turn(tts_instance, text)
    await _synthesize(tts_instance, text)

    assert server.sent("text")[0]["text"] == f"{expected} "
    assert server.posts[0][1]["text"] == expected


async def test_pronunciation_dictionary_goes_on_the_first_message_only(server: _Server) -> None:
    dictionary = {"Acme": "/ˈækmi/"}
    tts_instance = server.tts(pronunciation_dictionary=dictionary)
    _, timed = await _turn(tts_instance, FIRST, " ", SECOND)

    assert timed == []
    assert [m.get("pronunciation_dictionary") for m in server.sent("text")] == [dictionary, None]


async def test_update_options_reconnects_only_when_the_socket_is_affected(server: _Server) -> None:
    tts_instance = server.tts()
    await _turn(tts_instance, FIRST)

    tts_instance.update_options(voice_id=DEFAULT_VOICE_ID)
    await _turn(tts_instance, SECOND)
    assert server.handshakes == 1

    tts_instance.update_options(voice_id="other-voice", model="heygen-voice-1", language="de")
    await _turn(tts_instance, SECOND)
    await tts_instance.aclose()

    assert server.handshakes == 2
    query = server.connections[1].query
    assert (query["voice_id"], query["model"], query["language"]) == (
        "other-voice",
        "heygen-voice-1",
        "de",
    )


async def test_synthesize_requests_raw_pcm_with_the_supported_fields(server: _Server) -> None:
    tts_instance = server.tts(
        language="en-GB",
        sample_rate=44100,
        expressiveness_boost=0.5,
        pronunciation_dictionary={"Acme": "/ˈækmi/"},
        end_user_id="user-1",
        # The HTTP endpoint refuses these three fields.
        enable_logging=False,
        enable_ssml_parsing=True,
        apply_text_normalization="on",
    )
    audio = await _synthesize(tts_instance)

    request, body = server.posts[0]
    assert audio
    assert request.headers["x-api-key"] == "test-key"
    assert body == {
        "text": "hello there",
        "voice_id": DEFAULT_VOICE_ID,
        "model": "heygen-voice-1-turbo",
        "language": "en",
        "expressiveness_boost": 0.5,
        "pronunciation_dictionary": {"Acme": "/ˈækmi/"},
        "end_user_id": "user-1",
        "output_format": {"encoding": "pcm", "sample_rate_hertz": 44100},
    }
    assert server.handshakes == 0


async def test_synthesize_error_keeps_code_and_request_id(server: _Server) -> None:
    server.post_responses.append(
        web.json_response(
            {
                "error": {
                    "code": "invalid_parameter",
                    "message": "Voice not found",
                    "request_id": "req-3",
                }
            },
            status=400,
        )
    )
    with pytest.raises(APIStatusError) as raised:
        await _synthesize(server.tts())

    assert raised.value.status_code == 400
    assert raised.value.request_id == "req-3"
    assert raised.value.message == "invalid_parameter: Voice not found"
    assert raised.value.retryable is False


async def test_synthesize_does_not_retry_a_rate_limit(server: _Server) -> None:
    server.post_responses.append(web.Response(status=429, headers={"Retry-After": "60"}))
    with pytest.raises(APIStatusError):
        await _synthesize(server.tts(), conn_options=RETRY)

    assert len(server.posts) == 1


async def test_synthesize_retries_full_capacity(server: _Server) -> None:
    server.post_responses.append(web.Response(status=429))
    assert await _synthesize(server.tts(), conn_options=RETRY)
    assert len(server.posts) == 2


async def test_synthesize_survives_a_non_json_error(server: _Server) -> None:
    server.post_responses.append(web.Response(text="bad gateway", status=502))
    with pytest.raises(APIStatusError) as raised:
        await _synthesize(server.tts())

    assert raised.value.status_code == 502


async def test_streaming_off_serves_every_sentence_over_http(server: _Server) -> None:
    tts_instance = server.tts(streaming=False, voice_id="professional-voice")

    assert tts_instance.capabilities.streaming is False
    assert tts_instance.capabilities.aligned_transcript is False
    with pytest.raises(RuntimeError, match="streaming"):
        tts_instance.stream()

    audio, _ = await _turn(tts.StreamAdapter(tts=tts_instance), f"{FIRST} {SECOND}")

    assert audio
    assert server.handshakes == 0
    assert [body["text"] for _, body in server.posts] == [FIRST, SECOND]
    assert server.posts[0][1]["voice_id"] == "professional-voice"
