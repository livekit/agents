"""LiveAvatar LITE events correlate to real utterance and interrupt command IDs."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import Any

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from pydantic import JsonValue, ValidationError

from livekit import rtc
from livekit.agents import utils
from livekit.agents.voice.avatar import QueueAudioOutput
from livekit.plugins.liveavatar.avatar import (
    _SERVER_EVENT,
    AvatarSession,
    _ServerEvent,
    _Utterance,
)

pytestmark = pytest.mark.plugin("liveavatar")


class _FakeAudioBuffer:
    def __init__(self) -> None:
        self.started = 0
        self.finished: list[tuple[float, bool]] = []

    def notify_playback_started(self) -> None:
        self.started += 1

    def notify_playback_finished(self, playback_position: float, interrupted: bool) -> None:
        self.finished.append((playback_position, interrupted))


@pytest.fixture
def avatar(monkeypatch: pytest.MonkeyPatch) -> tuple[AvatarSession, _FakeAudioBuffer, str]:
    def fake_api(*_args: Any, **_kwargs: Any) -> object:
        return object()

    monkeypatch.setattr("livekit.plugins.liveavatar.avatar.LiveAvatarAPI", fake_api)
    session = AvatarSession(api_key="test-key", avatar_id="av-1")
    audio = _FakeAudioBuffer()
    session._audio_buffer = audio  # type: ignore[assignment]
    source = str(uuid.uuid4())
    session._utterances.append(_Utterance(event_id=source, playback_position=1.25))
    return session, audio, source


def test_session_connected_still_unblocks_forwarding(avatar):
    session, _, _ = avatar
    session._handle_server_event({"type": "session.state_updated", "state": "connected"})
    assert session._session_connected.is_set()


def test_state_updated_talking_marks_playback_started(avatar):
    session, audio, source = avatar
    session._handle_server_event(
        {"type": "agent.state_updated", "new_state": "talking", "source_event_id": source}
    )
    assert audio.started == 1
    assert audio.finished == []


def test_uncorrelated_listening_does_not_finish_playback(avatar):
    session, audio, source = avatar
    session._handle_server_event({"type": "agent.speak_started", "source_event_id": source})
    session._handle_server_event(
        {"type": "agent.state_updated", "new_state": "listening", "source_event_id": None}
    )
    assert audio.finished == []
    session._handle_server_event({"type": "agent.speak_ended", "source_event_id": source})
    assert audio.finished == [(1.25, False)]


def test_buffer_acks_do_not_drive_speaking_state(avatar):
    session, audio, source = avatar
    for kind in ["agent.audio_buffer_appended", "agent.audio_buffer_committed"]:
        session._handle_server_event({"type": kind, "source_event_id": source})
    assert audio.started == 0
    assert audio.finished == []


def test_speak_started_and_talking_are_idempotent(avatar):
    session, audio, source = avatar
    session._handle_server_event({"type": "agent.speak_started", "source_event_id": source})
    session._handle_server_event(
        {"type": "agent.state_updated", "new_state": "talking", "source_event_id": source}
    )
    assert audio.started == 1


def test_audio_buffer_cleared_matches_the_interrupt_command(avatar):
    session, audio, source = avatar
    session._audio_playing = True
    session._on_clear_buffer()
    command = session._msg_ch.recv_nowait()
    session._handle_server_event(
        {"type": "agent.audio_buffer_cleared", "source_event_id": command["event_id"]}
    )
    session._handle_server_event({"type": "agent.speak_interrupted", "source_event_id": source})
    assert audio.finished == [(0.0, True)]


@pytest.mark.parametrize("invalid_source", [False, 42, {}])
def test_wire_event_rejects_a_malformed_source_id(invalid_source):
    with pytest.raises(ValidationError):
        _SERVER_EVENT.validate_python(
            {"type": "agent.speak_ended", "source_event_id": invalid_source}, strict=True
        )


@dataclass
class _Peer:
    session: AvatarSession
    audio: QueueAudioOutput
    websocket: web.WebSocketResponse
    commands: asyncio.Queue[dict[str, Any]]
    handled: asyncio.Queue[str]

    async def command(self, kind: str) -> dict[str, Any]:
        while True:
            message = await asyncio.wait_for(self.commands.get(), 1)
            if message["type"] == kind:
                return message

    async def respond(self, kind: str, source: str | None, **fields: JsonValue) -> None:
        event_id = str(uuid.uuid4())
        await self.websocket.send_json(
            {"type": kind, "event_id": event_id, "source_event_id": source, **fields}
        )
        # Observe actual dispatch before asserting playout state; no timing-based barrier.
        assert await asyncio.wait_for(self.handled.get(), 1) == event_id


@pytest.fixture
async def peer(monkeypatch: pytest.MonkeyPatch):
    commands: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    ready: asyncio.Future[web.WebSocketResponse] = asyncio.get_running_loop().create_future()

    async def websocket(request: web.Request) -> web.WebSocketResponse:
        connection = web.WebSocketResponse()
        await connection.prepare(request)
        await connection.send_json({"type": "session.state_updated", "state": "connected"})
        ready.set_result(connection)
        async for message in connection:
            commands.put_nowait(message.json())
        return connection

    app = web.Application()
    app.router.add_get("/events", websocket)
    async with TestServer(app) as server, utils.http_context.open():
        session = AvatarSession(api_key="test", avatar_id="test")
        audio = QueueAudioOutput(sample_rate=24000, wait_playback_start=True)
        session._audio_buffer = audio
        session._ws_url = str(server.make_url("/events").with_scheme("ws"))
        audio.on("clear_buffer", session._on_clear_buffer)
        handled: asyncio.Queue[str] = asyncio.Queue()
        original = session._handle_server_event

        def observe(event: _ServerEvent) -> None:
            original(event)
            event_id = event.get("event_id")
            if event_id is not None:
                handled.put_nowait(event_id)

        monkeypatch.setattr(session, "_handle_server_event", observe)
        session._main_atask = asyncio.create_task(session._main_task())
        try:
            yield _Peer(session, audio, await asyncio.wait_for(ready, 1), commands, handled)
        finally:
            await session.aclose()


def frame(duration: float) -> rtc.AudioFrame:
    samples = round(24000 * duration)
    return rtc.AudioFrame(b"\0\0" * samples, 24000, 1, samples)


@pytest.mark.parametrize("stage", ["buffered", "submitted", "playing"])
async def test_delayed_correlated_peer_events_cannot_finish_the_next_utterance(peer, stage):
    audio = peer.audio
    retired: list[tuple[str | None, str]] = []
    for _ in range(2):
        await audio.capture_frame(frame(0.1 if stage == "buffered" else 0.6))
        if stage == "buffered":
            async with asyncio.timeout(1):
                while not peer.session._audio_playing:
                    await asyncio.sleep(0)
            source = None
        else:
            audio.flush()
            source = (await peer.command("agent.speak"))["event_id"]
            await peer.command("agent.speak_end")
            if stage == "playing":
                await peer.respond("agent.speak_started", source)
        audio.clear_buffer()
        interrupt = (await peer.command("agent.interrupt"))["event_id"]
        await peer.respond("agent.audio_buffer_cleared", interrupt)
        assert (await asyncio.wait_for(audio.wait_for_playout(), 1)).interrupted
        retired.append((source, interrupt))

    await audio.capture_frame(frame(0.6))
    audio.flush()
    current = (await peer.command("agent.speak"))["event_id"]
    await peer.command("agent.speak_end")
    await peer.respond("agent.speak_started", current)
    waiting = asyncio.create_task(audio.wait_for_playout())
    try:
        for source, interrupt in retired:
            for kind in ["agent.speak_started", "agent.speak_ended", "agent.speak_interrupted"]:
                await peer.respond(kind, source)
            await peer.respond("agent.audio_buffer_cleared", interrupt)
        foreign_session = str(uuid.uuid4())
        for kind in ["agent.speak_started", "agent.speak_ended", "agent.audio_buffer_cleared"]:
            await peer.respond(kind, foreign_session)
        await peer.respond("agent.state_updated", None, new_state="listening")
        await asyncio.sleep(0)
        assert not waiting.done(), "old acknowledgements must not finish new audio"
        await peer.respond("agent.speak_ended", current)
        result = await asyncio.wait_for(waiting, 1)
        assert not result.interrupted
        assert result.playback_position == pytest.approx(0.6)
    finally:
        waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)


async def test_queued_utterances_keep_their_first_chunk_id_and_duration(peer):
    audio = peer.audio
    reports = []
    audio.on("playback_finished", reports.append)
    await audio.capture_frame(frame(0.6))
    first = (await peer.command("agent.speak"))["event_id"]
    await audio.capture_frame(frame(1.0))
    later_chunk = (await peer.command("agent.speak"))["event_id"]
    assert later_chunk == first
    audio.flush()
    await peer.command("agent.speak_end")
    await audio.capture_frame(frame(0.6))
    audio.flush()
    second = (await peer.command("agent.speak"))["event_id"]
    await peer.command("agent.speak_end")
    waiting = asyncio.create_task(audio.wait_for_playout())
    try:
        assert second != first
        await peer.respond("agent.speak_ended", second)
        assert reports == []
        await peer.respond("agent.speak_ended", first)
        result = await asyncio.wait_for(waiting, 1)
        assert not result.interrupted
        assert [event.playback_position for event in reports] == pytest.approx([1.6, 0.6])
    finally:
        waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)


@pytest.mark.parametrize("sample_rate,samples", [(48000, 1), (24000, 0)])
async def test_empty_resampled_segment_does_not_block_later_playout(peer, sample_rate, samples):
    audio = peer.audio
    # Exercise both real resampler buffering and an empty PCM frame.
    await audio.capture_frame(rtc.AudioFrame(b"\0\0" * samples, sample_rate, 1, samples))
    audio.flush()
    result = await asyncio.wait_for(audio.wait_for_playout(), 1)
    assert not result.interrupted
    assert result.playback_position == 0.0
    barrier = str(uuid.uuid4())
    peer.session.send_event({"type": "session.keep_alive", "event_id": barrier})
    while True:
        command = await asyncio.wait_for(peer.commands.get(), 1)
        assert command["type"] == "session.keep_alive", "empty audio must not send speech commands"
        if command["event_id"] == barrier:
            break
    await audio.capture_frame(frame(0.6))
    audio.flush()
    source = (await peer.command("agent.speak"))["event_id"]
    await peer.command("agent.speak_end")
    await peer.respond("agent.speak_ended", source)
    assert not (await asyncio.wait_for(audio.wait_for_playout(), 1)).interrupted


async def test_normal_end_in_flight_preserves_played_position_during_interrupt(peer):
    reports = []
    peer.audio.on("playback_finished", reports.append)
    sources = []
    for _ in range(2):
        await peer.audio.capture_frame(frame(0.6))
        peer.audio.flush()
        sources.append((await peer.command("agent.speak"))["event_id"])
        await peer.command("agent.speak_end")
    peer.audio.clear_buffer()
    interrupt = (await peer.command("agent.interrupt"))["event_id"]
    await peer.respond("agent.speak_ended", sources[0])
    assert [(event.playback_position, event.interrupted) for event in reports] == [(0.6, False)]
    await peer.respond("agent.audio_buffer_cleared", interrupt)
    assert [(event.playback_position, event.interrupted) for event in reports] == [
        (0.6, False),
        (0.0, True),
    ]
    assert (await asyncio.wait_for(peer.audio.wait_for_playout(), 1)).interrupted


async def test_rejected_first_chunk_retains_the_actual_utterance_id(peer):
    await peer.audio.capture_frame(frame(0.6))
    first = (await peer.command("agent.speak"))["event_id"]
    await peer.respond(
        "error", first, error={"type": "invalid_request_error", "message": "first chunk rejected"}
    )
    await peer.audio.capture_frame(frame(1.0))
    accepted = (await peer.command("agent.speak"))["event_id"]
    assert accepted == first
    peer.audio.flush()
    await peer.command("agent.speak_end")
    await peer.respond("agent.speak_ended", accepted)
    assert not (await asyncio.wait_for(peer.audio.wait_for_playout(), 1)).interrupted
