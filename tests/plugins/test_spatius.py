"""Spatius avatar: the session token, the ingress socket's protobuf handshake, the agent's audio
as one Ogg Opus stream per segment, interrupts and server errors, against a local Spatius."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from google.protobuf import empty_pb2
from google.protobuf.unknown_fields import UnknownFieldSet

from livekit import rtc
from livekit.agents import APIConnectOptions, APIError, VoiceSession, http
from livekit.agents.voice.avatar import QueueAudioOutput
from livekit.plugins import spatius
from livekit.plugins.spatius import avatar as spatius_avatar, warmup
from tests.fakes import until

if TYPE_CHECKING:
    from tests.plugins.conftest import AvatarHost

OPTIONS = APIConnectOptions(max_retry=0, connect_timeout=1, read_timeout=1)
# Message{type: SERVER_CONFIRM_SESSION, server_confirm_session: {connection_id: "conn-1"}}
CONFIRM = b"\x08\x02\x1a\x08\x0a\x06conn-1"


def fields(data: bytes) -> dict[int, list[Any]]:
    """A protobuf message's fields by number, read without the plugin's codec."""
    message = empty_pb2.Empty()
    message.ParseFromString(data)
    found: dict[int, list[Any]] = {}
    unknown = UnknownFieldSet(message)
    for field in (unknown[i] for i in range(len(unknown))):
        found.setdefault(field.field_number, []).append(field.data)
    return found


def server_error(code: int, message: str) -> bytes:
    body = b"\x18" + bytes([code]) + b"\x22" + bytes([len(message)]) + message.encode()
    return b"\x08\x04\x2a" + bytes([len(body)]) + body


class Spatius:
    """The console's session-token endpoint and the ingress socket: confirms the session (or
    answers ``handshake``) and keeps every client message after the configuration."""

    def __init__(self) -> None:
        self.url = ""
        self.token_posts = 0
        self.upgrade: web.Request | None = None
        self.configuration: bytes = b""
        self.handshake = CONFIRM
        self.messages: list[dict[int, list[Any]]] = []
        self.socket: web.WebSocketResponse | None = None

    async def token(self, request: web.Request) -> web.Response:
        self.token_posts += 1
        return web.json_response({"sessionToken": f"token-{self.token_posts}"})

    async def ingress(self, request: web.Request) -> web.WebSocketResponse:
        self.upgrade = request
        socket = self.socket = web.WebSocketResponse()
        await socket.prepare(request)
        self.configuration = await socket.receive_bytes()
        await socket.send_bytes(self.handshake)
        async for message in socket:
            self.messages.append(fields(message.data))
        return socket

    def audio(self) -> list[tuple[str, bool, bytes]]:
        """Each audio input's request id, end flag and audio."""
        out = []
        for message in self.messages:
            if message[1] == [3]:
                body = fields(message[4][0])
                audio = body.get(3, [b""])[0]
                out.append((body[1][0].decode(), body.get(2, [0])[0] == 1, audio))
        return out

    def ends(self) -> int:
        return sum(end for _, end, _ in self.audio())

    def interrupts(self) -> list[str]:
        return [fields(m[8][0])[1][0].decode() for m in self.messages if m[1] == [7]]


@pytest.fixture
async def provider(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Spatius]:
    monkeypatch.setattr(spatius_avatar, "_tokens", {})
    monkeypatch.setattr(spatius_avatar, "_regions", {})
    peer = Spatius()
    app = web.Application()
    app.router.add_post("/v1/console/session-tokens", peer.token)
    app.router.add_get("/websocket", peer.ingress)
    async with TestServer(app) as server, http.open():
        peer.url = str(server.make_url("")).rstrip("/")
        yield peer


def avatar(peer: Spatius, **kwargs: Any) -> spatius.AvatarSession:
    return spatius.AvatarSession(
        api_key="key",
        app_id="app",
        avatar_id="avatar",
        console_endpoint_url=peer.url + "/v1/console",
        ingress_endpoint_url=peer.url,
        conn_options=OPTIONS,
        **kwargs,
    )


async def start(
    peer: Spatius, host: AvatarHost, **kwargs: Any
) -> tuple[spatius.AvatarSession, QueueAudioOutput]:
    session = avatar(peer, **kwargs)
    voice = VoiceSession(vad=None)
    await session.start(voice, host.room)
    audio = voice.output.audio
    assert isinstance(audio, QueueAudioOutput)
    return session, audio


def frame(seconds: float, rate: int = 24000) -> rtc.AudioFrame:
    samples = round(rate * seconds)
    return rtc.AudioFrame(b"\1\0" * samples, rate, 1, samples)


async def test_the_session_is_configured_for_the_room_with_the_session_token(
    provider: Spatius, avatar_host: AvatarHost
) -> None:
    """The ingress gets the session token, the avatar id, and a configuration naming the room,
    the avatar participant and the agent it publishes for."""
    session, _ = await start(provider, avatar_host)
    await session.aclose()

    assert provider.upgrade is not None
    assert provider.upgrade.query["id"] == "avatar"
    assert provider.upgrade.headers["X-App-ID"] == "app"
    assert provider.upgrade.headers["X-Session-Key"] == "token-1"
    message = fields(provider.configuration)
    assert message[1] == [1]
    configure = fields(message[2][0])
    assert (configure[1], configure[3], configure[5]) == ([24000], [1], [1])
    egress = fields(configure[6][0])
    assert egress[1] == [b"wss://lk.test"]
    assert egress[4] == [b"room-1"]
    attributes = {fields(entry)[1][0]: fields(entry)[2][0] for entry in egress[6]}
    assert attributes[b"livekit_agent_identity"] == b"agent-1"


async def test_each_segment_is_one_ogg_opus_stream_ended_by_its_last_message(
    provider: Spatius, avatar_host: AvatarHost
) -> None:
    """A segment's audio goes out as one Ogg Opus stream under one request id, its last message
    ends it, and the next segment starts a new request."""
    session, audio = await start(provider, avatar_host)
    try:
        await audio.capture_frame(frame(0.1))
        audio.flush()
        await until(lambda: provider.ends() == 1, "the first segment's end reaches Spatius")
        await audio.capture_frame(frame(0.1))
        audio.flush()
        await until(lambda: provider.ends() == 2, "the second segment's end reaches Spatius")
    finally:
        await session.aclose()

    sent = provider.audio()
    first = [item for item in sent if item[0] == sent[0][0]]
    assert len({request for request, _, _ in sent}) == 2
    assert [end for _, end, _ in first] == [False] * (len(first) - 1) + [True]
    stream = b"".join(chunk for _, _, chunk in first)
    assert stream.startswith(b"OggS") and b"OpusHead" in stream
    # the last page carries the end-of-stream flag
    assert stream[stream.rindex(b"OggS") + 5] & 0x04


async def test_a_clear_interrupts_the_last_request(
    provider: Spatius, avatar_host: AvatarHost
) -> None:
    """An interruption interrupts the request last sent, even once its audio ended; nothing is
    interrupted before any audio."""
    session, audio = await start(provider, avatar_host)
    try:
        audio.clear_buffer()
        await audio.capture_frame(frame(0.1))
        audio.flush()
        await until(lambda: any(end for _, end, _ in provider.audio()), "the segment ends")
        audio.clear_buffer()
        await until(lambda: provider.interrupts(), "the interrupt reaches Spatius")
    finally:
        await session.aclose()

    assert provider.interrupts() == [provider.audio()[0][0]]


async def test_a_server_error_fails_the_avatar_by_its_code(
    provider: Spatius, avatar_host: AvatarHost
) -> None:
    """A ServerError on the running session is the avatar's failure, its kind from its code."""
    session, _ = await start(provider, avatar_host)
    try:
        assert provider.socket is not None
        await provider.socket.send_bytes(server_error(16, "unauthorized egress"))
        await until(lambda: avatar_host.lifecycle.fail.called, "the avatar fails")
    finally:
        await session.aclose()

    error = avatar_host.lifecycle.fail.call_args.args[0]
    assert isinstance(error, APIError)
    assert (error.kind, error.detail) == ("auth", "unauthorized egress")


async def test_a_refused_handshake_fails_start_by_its_code(
    provider: Spatius, avatar_host: AvatarHost
) -> None:
    """A ServerError instead of the session's confirmation fails start with its kind."""
    provider.handshake = server_error(3, "bad sample rate")

    with pytest.raises(APIError) as failure:
        await start(provider, avatar_host)

    assert (failure.value.kind, failure.value.detail) == ("invalid_request", "bad sample rate")


async def test_a_prewarmed_token_serves_the_next_start(
    provider: Spatius, avatar_host: AvatarHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    """prewarm() fetches the session token once for the process; a start reuses it."""
    for name, value in {
        "SPATIUS_APP_ID": "app",
        "SPATIUS_API_KEY": "key",
        "SPATIUS_CONSOLE_ENDPOINT": provider.url + "/v1/console",
        "SPATIUS_INGRESS_ENDPOINT": provider.url,
    }.items():
        monkeypatch.setenv(name, value)

    await warmup._prewarm("app", "key")
    session, _ = await start(provider, avatar_host)
    await session.aclose()

    assert provider.token_posts == 1
    assert provider.upgrade is not None
    assert provider.upgrade.headers["X-Session-Key"] == "token-1"


async def test_an_auto_region_is_resolved_once_and_falls_back_when_bootstrap_fails(
    provider: Spatius, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bootstrap API names the region; a failed lookup falls back to us-west."""
    answers = [web.json_response({"region": {"current": "eu-central"}})]

    async def bootstrap(request: web.Request) -> web.Response:
        return answers.pop(0) if answers else web.Response(status=503)

    app = web.Application()
    app.router.add_post("/bootstrap", bootstrap)
    async with TestServer(app) as server:
        monkeypatch.setattr(spatius_avatar, "_BOOTSTRAP_URL", str(server.make_url("/bootstrap")))
        resolve = spatius_avatar._resolve_region
        assert await resolve("app", "auto", OPTIONS) == "eu-central"
        assert await resolve("app", "auto", OPTIONS) == "eu-central"  # cached
        assert await resolve("other", "auto", OPTIONS) == "us-west"
        assert await resolve("app", "us-east", OPTIONS) == "us-east"
