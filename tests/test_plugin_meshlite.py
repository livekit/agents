"""Unit tests for the MeshLite avatar plugin, in a faked LiveKit room.

MeshLite draws the avatar in the visitor's page: the plugin only routes the
agent's voice there, over the avatar data-stream protocol. These tests check
that wiring, the protocol the MeshLite web SDK implements, and that the plugin
never removes the visitor from the room.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from types import SimpleNamespace

import pytest

from livekit import rtc
from livekit.agents import Plugin
from livekit.agents.voice.avatar import AvatarSession as BaseAvatarSession, DataStreamAudioOutput
from livekit.plugins import meshlite

pytestmark = [pytest.mark.unit, pytest.mark.plugin("meshlite")]

STANDARD = rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD
AGENT = rtc.ParticipantKind.PARTICIPANT_KIND_AGENT
ACTIVE = rtc.ParticipantState.PARTICIPANT_STATE_ACTIVE


class FakeWriter:
    def __init__(self, options: dict) -> None:
        self.options = options
        self.chunks: list[bytes] = []
        self.closed = False

    async def write(self, data: bytes) -> None:
        self.chunks.append(bytes(data))

    async def aclose(self) -> None:
        self.closed = True


class FakeLocalParticipant:
    identity = "agent-1"
    kind = AGENT

    def __init__(self) -> None:
        self._rpc_handlers: dict = {}
        self.streams: list[FakeWriter] = []
        self.rpcs: list[dict] = []
        self.rpc_result = "ok"

    def register_rpc_method(self, name, handler) -> None:
        self._rpc_handlers[name] = handler

    async def stream_bytes(self, **options) -> FakeWriter:
        writer = FakeWriter(options)
        self.streams.append(writer)
        return writer

    async def perform_rpc(self, **options) -> str:
        self.rpcs.append(options)
        if isinstance(self.rpc_result, Exception):
            raise self.rpc_result
        return self.rpc_result


class FakeRoom:
    name = "room-1"

    def __init__(self, connected: bool = True) -> None:
        self.connected = connected
        self.local_participant = FakeLocalParticipant()
        self.remote_participants: dict = {}
        self._handlers: dict[str, list] = {}

    def isconnected(self) -> bool:
        return self.connected

    def on(self, event, callback=None):
        self._handlers.setdefault(event, []).append(callback)
        return callback

    def off(self, event, callback) -> None:
        if callback in self._handlers.get(event, []):
            self._handlers[event].remove(callback)

    def emit(self, event, *args) -> None:
        for callback in list(self._handlers.get(event, [])):
            callback(*args)

    def join(self, identity: str, kind=STANDARD):
        participant = SimpleNamespace(identity=identity, kind=kind, state=ACTIVE)
        self.remote_participants[identity] = participant
        self.emit("participant_connected", participant)
        self.emit("participant_active", participant)
        return participant

    def rpc(self, method: str, caller: str, payload: str = "") -> str:
        handler = self.local_participant._rpc_handlers[method]
        return handler(
            rtc.RpcInvocationData(
                request_id=uuid.uuid4().hex,
                caller_identity=caller,
                payload=payload,
                response_timeout=15.0,
            )
        )


class FakeSession(rtc.EventEmitter):
    def __init__(self) -> None:
        super().__init__()
        self.output = SimpleNamespace(audio=None)


def fake_session() -> FakeSession:
    return FakeSession()


def visitor() -> str:
    # The agents SDK keeps playback handlers per caller identity at class level.
    return f"visitor-{uuid.uuid4().hex[:8]}"


def frame(samples: int = 240, rate: int = 24000) -> rtc.AudioFrame:
    return rtc.AudioFrame.create(rate, 1, samples)


def test_constructor_validates_options() -> None:
    with pytest.raises(ValueError):
        meshlite.AvatarSession(visitor_identity="  ")
    with pytest.raises(ValueError):
        meshlite.AvatarSession(sample_rate=4000)
    with pytest.raises(ValueError):
        meshlite.AvatarSession(sample_rate=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        meshlite.AvatarSession(clear_buffer_timeout=-1)
    avatar = meshlite.AvatarSession()
    assert avatar.provider == "meshlite"
    assert avatar.visitor_identity is None
    with pytest.raises(RuntimeError):
        _ = avatar.avatar_identity


async def test_explicit_visitor_routes_voice_without_waiting() -> None:
    room, session, identity = FakeRoom(), fake_session(), visitor()
    avatar = meshlite.AvatarSession(visitor_identity=identity)
    await avatar.start(session, room)  # the visitor has not joined yet
    await asyncio.sleep(0)  # RPC handlers are registered on the next loop turn
    output = session.output.audio
    assert isinstance(output, DataStreamAudioOutput)
    assert output is avatar.audio_output
    assert avatar.avatar_identity == identity
    assert output.sample_rate == meshlite.DEFAULT_SAMPLE_RATE
    assert output.can_pause is False
    assert "lk.playback_finished" in room.local_participant._rpc_handlers
    assert "lk.playback_started" in room.local_participant._rpc_handlers
    await avatar.aclose()


async def test_discovers_first_standard_participant() -> None:
    room, session = FakeRoom(), fake_session()
    room.join("other-agent", kind=AGENT)
    avatar = meshlite.AvatarSession()
    task = asyncio.create_task(avatar.start(session, room))
    await asyncio.sleep(0.01)
    assert not task.done(), "an agent participant is not a visitor"
    identity = visitor()
    room.join(identity)
    await asyncio.wait_for(task, 1)
    assert avatar.visitor_identity == identity
    assert session.output.audio is avatar.audio_output


async def test_discovery_needs_a_connected_room() -> None:
    avatar = meshlite.AvatarSession()
    with pytest.raises(RuntimeError, match="connect"):
        await avatar.start(fake_session(), FakeRoom(connected=False))
    # A failed start can be retried once the room is connected.
    room, identity = FakeRoom(), visitor()
    room.join(identity)
    await avatar.start(fake_session(), room)
    assert avatar.visitor_identity == identity


async def test_start_twice_is_refused() -> None:
    room, session = FakeRoom(), fake_session()
    avatar = meshlite.AvatarSession(visitor_identity=visitor())
    await avatar.start(session, room)
    with pytest.raises(RuntimeError, match="once"):
        await avatar.start(session, room)
    await avatar.aclose()


async def test_replacing_an_existing_output_warns(caplog) -> None:
    room, session = FakeRoom(), fake_session()
    session.output.audio = SimpleNamespace(label="RoomIO")
    with caplog.at_level(logging.WARNING, logger="livekit.plugins.meshlite"):
        avatar = meshlite.AvatarSession(visitor_identity=visitor())
        await avatar.start(session, room)
    assert isinstance(session.output.audio, DataStreamAudioOutput)
    assert "before AgentSession.start()" in caplog.text
    await avatar.aclose()


async def test_aclose_never_removes_a_participant_and_is_idempotent() -> None:
    room, session = FakeRoom(), fake_session()
    avatar = meshlite.AvatarSession(visitor_identity=visitor())
    await avatar.start(session, room)
    await avatar.aclose()
    await avatar.aclose()
    assert session.output.audio is None
    assert room.local_participant.rpcs == []
    with pytest.raises(RuntimeError, match="closed"):
        await avatar.start(session, room)


async def test_aclose_keeps_an_output_installed_later() -> None:
    room, session = FakeRoom(), fake_session()
    avatar = meshlite.AvatarSession(visitor_identity=visitor())
    await avatar.start(session, room)
    replacement = SimpleNamespace(label="other")
    session.output.audio = replacement
    await avatar.aclose()
    assert session.output.audio is replacement


async def test_speech_segment_matches_the_web_sdk_protocol() -> None:
    room, session, identity = FakeRoom(), fake_session(), visitor()
    await meshlite.AvatarSession(visitor_identity=identity).start(session, room)
    output: DataStreamAudioOutput = session.output.audio
    room.join(identity)

    await output.capture_frame(frame())
    await output.capture_frame(frame())
    output.flush()
    await asyncio.sleep(0)

    [writer] = room.local_participant.streams
    assert writer.options["topic"] == "lk.audio_stream"
    assert writer.options["destination_identities"] == [identity]
    assert writer.options["attributes"] == {"sample_rate": "24000", "num_channels": "1"}
    assert sum(len(chunk) for chunk in writer.chunks) == 2 * 240 * 2  # int16 mono
    assert writer.closed

    assert room.rpc("lk.playback_started", identity) == "ok"
    payload = json.dumps({"playback_position": 0.02, "interrupted": False})
    assert room.rpc("lk.playback_finished", identity, payload) == "ok"
    event = await asyncio.wait_for(output.wait_for_playout(), 1)
    assert event.interrupted is False
    assert event.playback_position == pytest.approx(0.02)


async def test_stream_attributes_follow_the_frames() -> None:
    # AgentSession resamples speech to output.sample_rate (24 kHz) before
    # capture_frame; frames pushed directly keep their own rate, which the
    # web SDK converts. The attributes must always describe the bytes.
    room, session, identity = FakeRoom(), fake_session(), visitor()
    await meshlite.AvatarSession(visitor_identity=identity, sample_rate=16000).start(session, room)
    room.join(identity)
    output = session.output.audio
    assert output.sample_rate == 16000
    await output.capture_frame(frame(480, 48000))
    output.flush()
    await asyncio.sleep(0)
    [writer] = room.local_participant.streams
    assert writer.options["attributes"] == {"sample_rate": "48000", "num_channels": "1"}


async def test_interruption_sends_clear_buffer_to_the_visitor() -> None:
    room, session, identity = FakeRoom(), fake_session(), visitor()
    await meshlite.AvatarSession(visitor_identity=identity).start(session, room)
    room.join(identity)
    output = session.output.audio
    await output.capture_frame(frame())
    output.flush()
    output.clear_buffer()
    await asyncio.sleep(0.01)
    assert [rpc["method"] for rpc in room.local_participant.rpcs] == ["lk.clear_buffer"]
    assert room.local_participant.rpcs[0]["destination_identity"] == identity
    payload = json.dumps({"playback_position": 0.004, "interrupted": True})
    assert room.rpc("lk.playback_finished", identity, payload) == "ok"
    event = await asyncio.wait_for(output.wait_for_playout(), 1)
    assert event.interrupted is True


async def test_clear_without_page_answer_falls_back() -> None:
    room, session, identity = FakeRoom(), fake_session(), visitor()
    avatar = meshlite.AvatarSession(visitor_identity=identity, clear_buffer_timeout=0.05)
    await avatar.start(session, room)
    room.join(identity)
    output = session.output.audio
    await output.capture_frame(frame())
    output.flush()
    output.clear_buffer()
    event = await asyncio.wait_for(output.wait_for_playout(), 1)
    assert event.interrupted is True


async def test_playback_events_from_someone_else_are_rejected() -> None:
    room, session, identity = FakeRoom(), fake_session(), visitor()
    avatar = meshlite.AvatarSession(visitor_identity=identity)
    await avatar.start(session, room)
    await asyncio.sleep(0)
    payload = json.dumps({"playback_position": 1.0, "interrupted": False})
    assert room.rpc("lk.playback_finished", "intruder", payload) == "reject"
    assert room.rpc("lk.playback_started", "intruder") == "reject"
    await avatar.aclose()


def test_registered_as_a_livekit_avatar_plugin() -> None:
    assert any(p.package == "livekit.plugins.meshlite" for p in Plugin.registered_plugins)
    assert isinstance(meshlite.AvatarSession(), BaseAvatarSession)


async def test_wait_for_join_returns_once_the_visitor_is_in_the_room() -> None:
    room, session, identity = FakeRoom(), fake_session(), visitor()
    metrics: list = []
    avatar = meshlite.AvatarSession(visitor_identity=identity)
    avatar.on("metrics_collected", metrics.append)
    await avatar.start(session, room)
    with pytest.raises(asyncio.TimeoutError):
        await avatar.wait_for_join(timeout=0.05)
    room.join(identity)  # no video track: the face is drawn in the visitor's page
    await avatar.wait_for_join(timeout=1)
    [joined] = metrics
    assert joined.metadata.model_provider == "meshlite"
    assert joined.avatar_joined_time is not None
    await avatar.aclose()


async def test_aclose_does_not_use_the_base_participant_removal(monkeypatch) -> None:
    # The base class removes ``avatar_identity`` from the room on close; for
    # MeshLite that identity is the visitor, who must stay in the room.
    async def removal(self) -> None:
        raise AssertionError("the base aclose would remove the visitor from the room")

    monkeypatch.setattr(BaseAvatarSession, "aclose", removal)
    room, session, identity = FakeRoom(), fake_session(), visitor()
    room.join(identity)
    avatar = meshlite.AvatarSession(visitor_identity=identity)
    await avatar.start(session, room)
    await avatar.aclose()
    assert identity in room.remote_participants
    assert session.output.audio is None
    assert avatar._wait_avatar_join_task is None
