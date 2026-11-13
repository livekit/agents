"""Soniox TTS: sentences share provider streams on one socket, rotate on idle gaps, and fail."""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncIterator, Callable

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit.agents import APIConnectOptions, APIError, APIStatusError, http
from livekit.agents.tts import SynthesizeStream
from livekit.plugins.soniox import TTS

AUDIO = base64.b64encode(b"\x00\x00" * 2400).decode()


class SonioxPeer:
    """``/tts``: one socket multiplexing provider streams keyed by ``stream_id``.

    By default each text answers with audio and ``text_end`` with ``audio_end`` + ``terminated``;
    ``on_text`` replaces the answer to the n-th (1-based) text message.
    """

    def __init__(self) -> None:
        self.authorizations: list[str | None] = []
        self.configs: list[dict[str, object]] = []
        self.texts: list[dict[str, object]] = []
        self.ended: list[str] = []
        self.an_end = asyncio.Event()
        self.open: set[str] = set()
        self.max_open = 0
        self.on_text: Callable[[int, str], list[dict[str, object]] | None] = lambda n, sid: None

    async def handle(self, request: web.Request) -> web.WebSocketResponse:
        self.authorizations.append(request.headers.get("Authorization"))
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for incoming in ws:
            message = json.loads(incoming.data)
            sid = message.get("stream_id", "")
            if "model" in message:
                self.configs.append(message)
                self.open.add(sid)
                self.max_open = max(self.max_open, len(self.open))
                continue
            replies: list[dict[str, object]] = []
            if message.get("text"):
                self.texts.append(message)
                custom = self.on_text(len(self.texts), sid)
                replies = custom if custom is not None else [{"stream_id": sid, "audio": AUDIO}]
            if message.get("text_end"):
                self.ended.append(sid)
                self.an_end.set()
            if message.get("text_end") or message.get("cancel"):
                replies.append({"stream_id": sid, "audio_end": True, "terminated": True})
            for reply in replies:
                if reply.get("terminated") or reply.get("error_code"):
                    self.open.discard(sid)
                await ws.send_json(reply)
        return ws


@pytest.fixture
async def peer() -> AsyncIterator[tuple[SonioxPeer, str]]:
    peer = SonioxPeer()
    app = web.Application()
    app.router.add_get("/tts", peer.handle)
    async with TestServer(app) as server, http.open():
        yield peer, str(server.make_url("/tts").with_scheme("ws"))


async def speak(tts: TTS, *sentences: str, retries: int = 0) -> int:
    """Pushes ``sentences`` and returns the audio frames played."""
    stream = tts.stream(conn_options=APIConnectOptions(max_retry=retries, retry_interval=0))
    for sentence in sentences:
        stream.push_text(sentence)
    stream.end_input()
    frames = len([ev async for ev in stream])
    await stream.aclose()
    return frames


async def test_streams_share_one_authenticated_socket(peer: tuple[SonioxPeer, str]) -> None:
    """Consecutive syntheses reuse one socket opened with the key as a bearer header.

    https://github.com/livekit/agents/pull/7698
    """
    server, url = peer
    tts = TTS(api_key="test-key", websocket_url=url)

    await speak(tts, "Hello!")
    await speak(tts, "Hello again!")
    await tts.aclose()

    assert server.authorizations == ["Bearer test-key"]
    assert [c["model"] for c in server.configs] == [tts.model, tts.model]
    assert all("api_key" not in c for c in server.configs)
    assert len({c["stream_id"] for c in server.configs}) == 2
    assert [t["stream_id"] for t in server.texts] == [c["stream_id"] for c in server.configs]


@pytest.mark.parametrize("stream_id", [None, ""], ids=["no-stream-id", "empty-stream-id"])
@pytest.mark.parametrize("status", [401, 429])
async def test_a_connection_error_fails_every_open_stream(
    peer: tuple[SonioxPeer, str], stream_id: str | None, status: int
) -> None:
    """A socket-level error fails each synthesis sharing the socket with its status.

    https://github.com/livekit/agents/pull/7698
    """
    server, url = peer
    error: dict[str, object] = {
        "error_code": status,
        "error_message": "rejected",
        "request_id": "r1",
    }
    if stream_id is not None:
        error["stream_id"] = stream_id
    server.on_text = lambda n, sid: [{"stream_id": sid, "audio": AUDIO}] if n == 1 else [error]
    tts = TTS(api_key="key", websocket_url=url)
    # the options `speak` uses: streams share a socket only when their options match
    first = tts.stream(conn_options=APIConnectOptions(max_retry=0, retry_interval=0))
    # the stage sends a sentence once the next one starts: "speaking." could be "Dr."
    first.push_text("The first synthesis is still speaking. The")
    await anext(first)

    second = await asyncio.gather(
        speak(tts, "The second one breaks the socket. "), return_exceptions=True
    )
    ended: list[BaseException] = []
    try:
        first.end_input()
    except APIStatusError as refused:  # the first already failed: its input answers with that
        ended.append(refused)
    [first_error] = await asyncio.gather(_drain(first), return_exceptions=True)
    errors = [*second, first_error, *ended]
    await first.aclose()
    await tts.aclose()

    for raised in errors:
        assert isinstance(raised, APIStatusError)
        assert (raised.status_code, raised.message, raised.request_id) == (status, "rejected", "r1")
        assert raised.retryable is (status == 429)


async def _drain(stream: SynthesizeStream) -> None:
    async for _ in stream:
        pass


async def test_idle_gap_finalizes_the_stream_and_opens_a_new_one(
    peer: tuple[SonioxPeer, str],
) -> None:
    """A stalled LLM ends the open stream; the next sentence starts a fresh one.

    https://github.com/livekit/agents/issues/6225
    """
    server, url = peer
    tts = TTS(api_key="key", websocket_url=url, stream_idle_timeout=0.001)
    stream = tts.stream(conn_options=APIConnectOptions(max_retry=0, retry_interval=0))
    # the stage sends a sentence once the next one starts: "now." could be "Dr."
    stream.push_text("The first sentence is spoken now. The")
    await server.an_end.wait()  # the LLM stalls: the idle stream is finalized
    stream.push_text(" second one comes after the stall.")
    stream.end_input()
    frames = len([ev async for ev in stream])
    await stream.aclose()
    await tts.aclose()

    assert frames > 0
    first, second = (c["stream_id"] for c in server.configs)
    assert [(t["stream_id"], str(t["text"]).strip()) for t in server.texts] == [
        (first, "The first sentence is spoken now."),
        (second, "The second one comes after the stall."),
    ]
    assert server.ended == [first, second]
    assert server.max_open == 1


async def test_a_transient_failure_before_audio_replays_the_sentence(
    peer: tuple[SonioxPeer, str],
) -> None:
    """A retryable stream error before any audio retries the same text on a new stream.

    https://github.com/livekit/agents/pull/6434
    """
    server, url = peer
    server.on_text = lambda n, sid: (
        [{"stream_id": sid, "error_code": 429, "error_message": "busy"}] if n == 1 else None
    )
    tts = TTS(api_key="key", websocket_url=url)

    frames = await speak(tts, "Hello there, this is the first sentence. ", retries=1)
    await tts.aclose()

    assert frames > 0
    first, replay = server.texts[:2]
    assert first["stream_id"] != replay["stream_id"]
    assert str(replay["text"]).strip() == str(first["text"]).strip()
    assert server.max_open == 1


async def test_a_failure_after_audio_is_not_replayed(peer: tuple[SonioxPeer, str]) -> None:
    """A provider failure after audio was played is final, never a replay."""
    server, url = peer
    server.on_text = lambda n, sid: (
        [{"stream_id": sid, "error_code": 429, "error_message": "later failure"}]
        if n == 2
        else [{"stream_id": sid, "audio": AUDIO}]
    )
    tts = TTS(api_key="key", websocket_url=url)

    with pytest.raises(APIStatusError, match="later failure"):
        await speak(tts, "First sentence is long enough. ", "Second must fail. ", retries=1)
    await tts.aclose()

    assert [str(t["text"]).strip() for t in server.texts] == [
        "First sentence is long enough.",
        "Second must fail.",
    ]


async def test_server_abort_without_audio_is_a_retryable_failure(
    peer: tuple[SonioxPeer, str],
) -> None:
    """A stream the server terminates with no audio fails retryably instead of going silent.

    https://github.com/livekit/agents/pull/5770
    """
    server, url = peer
    server.on_text = lambda n, sid: [{"stream_id": sid, "terminated": True}]
    tts = TTS(api_key="key", websocket_url=url)

    with pytest.raises(APIError) as raised:
        await speak(tts, "This sentence must produce audio.")
    await tts.aclose()

    assert raised.value.retryable
    assert "terminated" in raised.value.message
