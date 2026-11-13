"""LiveAvatar avatar: session start/stop, LITE speech commands and playback correlation."""

from __future__ import annotations

import asyncio
import base64
import logging
import uuid
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import TypedDict

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    CloseEvent,
    VoiceSession,
    http,
)
from livekit.agents.voice.avatar import QueueAudioOutput
from livekit.agents.voice.events import CloseReason
from livekit.plugins.liveavatar import AvatarSession
from livekit.plugins.liveavatar.api import LiveAvatarAPI
from tests.fakes import until

SAMPLE_RATE = 24000
NOT_CONNECTED = "the session is not connected yet"
OPTIONS = APIConnectOptions(max_retry=2, retry_interval=0, connect_timeout=1, read_timeout=1)


class Command(TypedDict, total=False):
    """A LITE-mode command the plugin sends on the event socket."""

    type: str
    event_id: str
    audio: str


Started = tuple[AvatarSession, VoiceSession, QueueAudioOutput]


class LiveAvatarPeer:
    """The LiveAvatar REST API (token, start, stop) and its LITE-mode event socket. Until
    ``connected`` is set the socket sends only a ``warning`` (``NOT_CONNECTED``), and keeps what
    the plugin sends meanwhile in ``early``."""

    def __init__(self) -> None:
        self.connected = asyncio.Event()
        self.connected.set()
        self.handshake = asyncio.Event()
        self.handshake.set()
        self.handshakes = 0
        self.commands: asyncio.Queue[Command] = asyncio.Queue()
        self.early: list[Command] = []
        self.stops: list[tuple[str | None, dict[str, object]]] = []
        self.start_status = 200
        self.starts = 0
        self.ws_url = ""
        self.api_url = ""
        self.socket: web.WebSocketResponse | None = None

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_post("/sessions/token", self.token)
        app.router.add_post("/sessions/start", self.start)
        app.router.add_post("/sessions/stop", self.stop)
        app.router.add_get("/events", self.events)
        return app

    async def token(self, request: web.Request) -> web.Response:
        return web.json_response({"data": {"session_id": "S1", "session_token": "session-token"}})

    async def start(self, request: web.Request) -> web.Response:
        self.starts += 1
        if self.start_status != 200:
            body = {"detail": f"provider says {self.start_status}"}
            return web.json_response(body, status=self.start_status)
        return web.json_response({"data": {"ws_url": self.ws_url}})

    async def stop(self, request: web.Request) -> web.Response:
        self.stops.append((request.headers.get("Authorization"), await request.json()))
        return web.json_response({"code": 100})

    async def events(self, request: web.Request) -> web.WebSocketResponse:
        self.handshakes += 1
        await self.handshake.wait()
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        told = False

        async def connect() -> None:
            nonlocal told
            if not self.connected.is_set():
                event = {"type": "warning", "event_id": "W1", "warning": NOT_CONNECTED}
                await socket.send_json(event)
                await self.connected.wait()
            told = True
            await socket.send_json({"type": "session.state_updated", "state": "connected"})
            self.socket = socket

        connecting = asyncio.create_task(connect())
        try:
            async for message in socket:
                command = message.json()
                if not told:
                    self.early.append(command)
                self.commands.put_nowait(command)
        finally:
            connecting.cancel()
            await asyncio.gather(connecting, return_exceptions=True)
        return socket

    async def command(self, kind: str, seen: list[Command] | None = None) -> Command:
        while True:
            message = await self.commands.get()
            if seen is not None:
                seen.append(message)
            if message["type"] == kind:
                return message

    async def send(self, kind: str, source: str | None, **fields: str) -> None:
        assert self.socket is not None
        event = {"type": kind, "event_id": str(uuid.uuid4()), "source_event_id": source}
        await self.socket.send_json({**event, **fields})


@pytest.fixture
async def peer(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[LiveAvatarPeer]:
    monkeypatch.setattr("livekit.plugins.liveavatar.avatar.KEEP_ALIVE_INTERVAL", 0.02)
    peer = LiveAvatarPeer()
    async with TestServer(peer.app()) as server, http.open():
        peer.ws_url = str(server.make_url("/events").with_scheme("ws"))
        peer.api_url = str(server.make_url("/sessions"))
        yield peer


@pytest.fixture
async def started(peer: LiveAvatarPeer, avatar_host: SimpleNamespace) -> AsyncIterator[Started]:
    """An avatar started for a voice session: the session's audio sink is the avatar's."""
    voice = VoiceSession(vad=None)
    avatar = AvatarSession(
        api_key="key",
        avatar_id="avatar",
        api_url=peer.api_url,
        conn_options=OPTIONS,
    )
    await avatar.start(voice, avatar_host.room, **avatar_host.credentials)
    audio = voice.output.audio
    assert isinstance(audio, QueueAudioOutput)
    try:
        yield avatar, voice, audio
    finally:
        peer.connected.set()
        await avatar.aclose()


def frame(seconds: float) -> rtc.AudioFrame:
    samples = round(SAMPLE_RATE * seconds)
    return rtc.AudioFrame(b"\0\0" * samples, SAMPLE_RATE, 1, samples)


def playback_events(audio: QueueAudioOutput) -> list[str]:
    events: list[str] = []
    audio.on("playback_started", lambda _: events.append("started"))
    audio.on("playback_finished", lambda _: events.append("finished"))
    return events


async def speak(peer: LiveAvatarPeer, audio: QueueAudioOutput, seconds: float) -> str:
    """Plays one utterance into the avatar and returns its speak event id."""
    await audio.capture_frame(frame(seconds))
    audio.flush()
    source = (await peer.command("agent.speak"))["event_id"]
    await peer.command("agent.speak_end")
    return source


@pytest.fixture
def unconnected(peer: LiveAvatarPeer) -> LiveAvatarPeer:
    peer.connected.clear()
    return peer


async def test_speech_and_keepalives_wait_for_the_connected_state(
    unconnected: LiveAvatarPeer, started: Started, caplog: pytest.LogCaptureFixture
) -> None:
    """Nothing is sent before LiveAvatar says connected; then speech goes out in 0.6 s chunks.

    https://github.com/livekit/agents/issues/4204 https://github.com/livekit/agents/pull/5453
    """
    caplog.set_level(logging.WARNING)
    _, _, audio = started
    for _ in range(10):
        await audio.capture_frame(frame(0.1))
    audio.flush()
    # an ungated keepalive or chunk goes out at once; the plugin reading the provider's first
    # event is a round trip later
    await until(lambda: NOT_CONNECTED in caplog.text, "the avatar read the provider's warning")

    unconnected.connected.set()
    seen: list[Command] = []
    await unconnected.command("agent.speak_end", seen)
    await unconnected.command("session.keep_alive")
    assert unconnected.early == [], "sent before the session was connected"

    speaks = [c for c in seen if c["type"] == "agent.speak"]
    seconds = [len(base64.b64decode(c["audio"])) / (2 * SAMPLE_RATE) for c in speaks]
    assert seconds == pytest.approx([0.6, 0.4])
    assert len({c["event_id"] for c in speaks}) == 1


@pytest.mark.parametrize(
    ("acks", "expected"),
    [
        (["agent.audio_buffer_appended", "agent.audio_buffer_committed"], ["finished"]),
        (["agent.state_updated"], ["started", "finished"]),
        (["agent.speak_started", "agent.state_updated"], ["started", "finished"]),
    ],
    ids=["buffer-acks", "talking", "started-then-talking"],
)
async def test_playback_starts_once_on_talking_never_on_buffer_acks(
    peer: LiveAvatarPeer, started: Started, acks: list[str], expected: list[str]
) -> None:
    """Playback starts on speak_started or talking, once; buffer acks never start it.

    https://github.com/livekit/agents/issues/7232
    """
    _, _, audio = started
    events = playback_events(audio)
    source = await speak(peer, audio, 0.6)

    for kind in acks:
        await peer.send(kind, source, new_state="talking")
    await peer.send("agent.speak_ended", source)
    await audio.wait_for_playout()

    assert events == expected


@pytest.mark.parametrize("stage", ["submitted", "playing"])
async def test_late_events_for_interrupted_speech_cannot_finish_the_next(
    peer: LiveAvatarPeer, started: Started, stage: str
) -> None:
    """Acknowledgements for interrupted or foreign speech never finish the current utterance.

    https://github.com/livekit/agents/pull/7245 https://github.com/livekit/agents/pull/4725
    """
    _, _, audio = started
    retired: list[tuple[str, str]] = []
    for _ in range(2):
        source = await speak(peer, audio, 0.6)
        if stage == "playing":
            await peer.send("agent.speak_started", source)
        audio.clear_buffer()
        interrupt = (await peer.command("agent.interrupt"))["event_id"]
        await peer.send("agent.audio_buffer_cleared", interrupt)
        assert (await audio.wait_for_playout()).interrupted
        retired.append((source, interrupt))
    events = playback_events(audio)
    current = await speak(peer, audio, 0.6)
    waiting = asyncio.ensure_future(audio.wait_for_playout())

    for source, interrupt in retired:
        for kind in ["agent.speak_started", "agent.speak_ended", "agent.speak_interrupted"]:
            await peer.send(kind, source)
        await peer.send("agent.audio_buffer_cleared", interrupt)
    foreign = str(uuid.uuid4())
    for kind in ["agent.speak_started", "agent.speak_ended", "agent.audio_buffer_cleared"]:
        await peer.send(kind, foreign)
    await peer.send("agent.state_updated", None, new_state="listening")
    await peer.send("agent.speak_started", current)
    await until(lambda: events, "the current utterance to start")

    # events are handled in order, so every late event above was handled first
    assert events == ["started"], "a late event finished the current utterance"
    assert not waiting.done()
    await peer.send("agent.speak_ended", current)
    result = await waiting

    assert not result.interrupted
    assert result.playback_position == pytest.approx(0.6)


async def test_queued_utterances_report_in_order_with_their_full_duration(
    peer: LiveAvatarPeer, started: Started
) -> None:
    """Chunks of one utterance share its id; finishes are reported in speaking order.

    https://github.com/livekit/agents/issues/4406 https://github.com/livekit/agents/issues/4568
    """
    _, _, audio = started
    reports: list[float] = []
    audio.on("playback_finished", lambda ev: reports.append(ev.playback_position))
    await audio.capture_frame(frame(0.6))
    first = (await peer.command("agent.speak"))["event_id"]
    await audio.capture_frame(frame(1.0))
    assert (await peer.command("agent.speak"))["event_id"] == first
    audio.flush()
    await peer.command("agent.speak_end")
    second = await speak(peer, audio, 0.6)

    await peer.send("agent.speak_ended", second)
    await peer.send("agent.speak_ended", first)
    result = await audio.wait_for_playout()

    assert second != first
    assert not result.interrupted
    assert reports == pytest.approx([1.6, 0.6])


@pytest.mark.parametrize("close", ["voice_session_closed", "avatar_aclose"])
async def test_ending_stops_the_provider_session_quietly(
    peer: LiveAvatarPeer, started: Started, caplog: pytest.LogCaptureFixture, close: str
) -> None:
    """Ending the avatar stops the LiveAvatar session with its token and logs no warning.

    https://github.com/livekit/agents/issues/4193 https://github.com/livekit/agents/pull/7341
    """
    avatar, voice, _ = started
    await peer.command("session.keep_alive")
    caplog.set_level(logging.WARNING)

    if close == "voice_session_closed":
        voice.emit("close", CloseEvent(reason=CloseReason.USER_INITIATED))
        await until(lambda: peer.stops, "the session to be stopped")
    await avatar.aclose()

    assert peer.stops == [
        ("Bearer session-token", {"session_id": "S1", "reason": "USER_DISCONNECTED"})
    ]
    assert [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING] == []


@pytest.mark.parametrize(("status", "attempts"), [(400, 1), (401, 1), (503, OPTIONS.max_retry + 1)])
async def test_provider_status_and_body_survive_and_only_transient_errors_retry(
    peer: LiveAvatarPeer, status: int, attempts: int
) -> None:
    """https://github.com/livekit/agents/pull/6306"""
    peer.start_status = status
    api = LiveAvatarAPI("key", api_url=peer.api_url, conn_options=OPTIONS)

    with pytest.raises(APIStatusError) as raised:
        await api.start_streaming_session("S1", "session-token")

    assert raised.value.status_code == status
    assert f"provider says {status}" in str(raised.value.body)
    assert raised.value.retryable == (status >= 500)
    assert peer.starts == attempts


@pytest.mark.xfail(
    strict=True,
    reason="the event socket is opened outside the cleanup that calls /sessions/stop",
)
async def test_closing_during_a_stalled_handshake_stops_the_provider_session(
    peer: LiveAvatarPeer, avatar_host: SimpleNamespace
) -> None:
    """An avatar closed while its event socket handshake hangs still stops the LiveAvatar session."""
    peer.handshake.clear()
    avatar = AvatarSession(api_key="key", avatar_id="avatar", api_url=peer.api_url)
    await avatar.start(VoiceSession(vad=None), avatar_host.room, **avatar_host.credentials)
    await until(lambda: peer.handshakes, "the event socket handshake begins")

    await avatar.aclose()
    peer.handshake.set()

    assert [body for _, body in peer.stops] == [{"session_id": "S1", "reason": "USER_DISCONNECTED"}]


async def test_zero_retries_still_starts_the_session(
    peer: LiveAvatarPeer, avatar_host: SimpleNamespace
) -> None:
    """With max_retry=0 the session is still created and started, once.

    https://github.com/livekit/agents/issues/7604
    """
    avatar = AvatarSession(
        api_key="key",
        avatar_id="avatar",
        api_url=peer.api_url,
        conn_options=APIConnectOptions(max_retry=0),
    )

    await avatar.start(VoiceSession(vad=None), avatar_host.room, **avatar_host.credentials)
    await avatar.aclose()

    assert peer.starts == 1


async def test_event_socket_handshake_obeys_read_timeout(
    peer: LiveAvatarPeer, avatar_host: SimpleNamespace
) -> None:
    """An event socket whose handshake never answers is dropped after the read timeout."""
    hung_up = asyncio.Event()

    async def silent(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await reader.read()  # the upgrade request, then EOF once the plugin hangs up
        hung_up.set()
        writer.close()

    server = await asyncio.start_server(silent, "127.0.0.1", 0)
    peer.ws_url = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}/events"
    options = APIConnectOptions(max_retry=0, connect_timeout=30, read_timeout=0.05)
    avatar = AvatarSession(
        api_key="key", avatar_id="avatar", api_url=peer.api_url, conn_options=options
    )
    await avatar.start(VoiceSession(vad=None), avatar_host.room, **avatar_host.credentials)

    try:
        await until(hung_up.is_set, "the plugin gives up on the stalled handshake")
    finally:
        await avatar.aclose()
        server.close()
        await server.wait_closed()


async def test_invalid_api_url_is_not_classified_as_invalid_provider_response(
    avatar_host: SimpleNamespace,
) -> None:
    """A malformed api_url fails as a connection error, never as a bad LiveAvatar answer."""
    avatar = AvatarSession(
        api_key="key",
        avatar_id="avatar",
        api_url="http://[",
        conn_options=APIConnectOptions(max_retry=0),
    )

    async with http.open():
        with pytest.raises(APIConnectionError) as error:
            await avatar.start(VoiceSession(vad=None), avatar_host.room, **avatar_host.credentials)

    assert "invalid response" not in str(error.value)
