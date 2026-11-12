"""A Python `AudioOutput` stage wrapping the native `DataStreamAudioOutput`: the chain behaves as
over a Python sink. The avatar worker is a fake room participant answering the output's RPCs."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from livekit import rtc
from livekit.agents import Agent, VoiceSession
from livekit.agents.voice.avatar import DataStreamAudioOutput
from livekit.agents.voice.io import AudioOutput, AudioOutputCapabilities

from .fake_tts import FakeTTS

pytestmark = pytest.mark.unit


class Writer:
    def __init__(self, stream_id: str) -> None:
        self.info = SimpleNamespace(stream_id=stream_id)
        self.chunks: list[bytes] = []
        self.closed = False

    async def write(self, data: bytes) -> None:
        self.chunks.append(bytes(data))

    async def aclose(self) -> None:
        self.closed = True


class LocalParticipant:
    identity = "agent"
    sid = "PA_agent"
    kind = rtc.ParticipantKind.PARTICIPANT_KIND_AGENT
    track_publications: dict[str, object] = {}

    def __init__(self) -> None:
        self.routes: dict[str, object] = {}
        self.streams: list[Writer] = []
        self.clears = 0

    async def stream_bytes(self, *, name, topic, destination_identities, attributes) -> Writer:
        writer = Writer(f"STREAM_{len(self.streams)}")
        self.streams.append(writer)
        return writer

    def register_rpc_method(self, method, handler) -> None:
        self.routes[method] = handler

    def unregister_rpc_method(self, method) -> None:
        self.routes.pop(method, None)

    async def perform_rpc(self, *, destination_identity, method, payload) -> str:
        assert (destination_identity, method) == ("avatar", "lk.clear_buffer")
        self.clears += 1
        return "ok"


class Room:
    """An `rtc.Room` the avatar participant is already in."""

    name = "room"
    connection_state = rtc.ConnectionState.CONN_CONNECTED

    def __init__(self, *, avatar_joined: bool = True) -> None:
        self.local_participant = LocalParticipant()
        self.remote_participants: dict[str, object] = {}
        self.handlers: dict[str, object] = {}
        if avatar_joined:
            self.join_avatar()

    def on(self, event, callback=None):
        self.handlers[event] = callback
        return callback

    def join_avatar(self) -> None:
        avatar = SimpleNamespace(
            identity="avatar",
            sid="PA_avatar",
            kind=rtc.ParticipantKind.PARTICIPANT_KIND_AGENT,
            state=rtc.ParticipantState.PARTICIPANT_STATE_ACTIVE,
            attributes={},
            track_publications={},
        )
        self.remote_participants["avatar"] = avatar
        if (connected := self.handlers.get("participant_connected")) is not None:
            connected(avatar)

    def finish_playback(self, position: float, *, interrupted: bool) -> None:
        """The avatar reports the open segment played, as its worker does over RPC."""
        # the open segment's own route, beside the plain v1 method
        (method,) = [m for m in self.local_participant.routes if m.startswith("lk.playback_finished/")]
        payload = json.dumps({"playback_position": position, "interrupted": interrupted})
        reply = self.local_participant.routes[method](
            SimpleNamespace(caller_identity="avatar", payload=payload)
        )
        assert reply == "ok"


class Stage(AudioOutput):
    """A user stage: counts the frames and forwards everything to the output below it."""

    def __init__(
        self, next_in_chain: DataStreamAudioOutput | AudioOutput, *, label: str = "stage"
    ) -> None:
        super().__init__(
            label=label,
            capabilities=AudioOutputCapabilities(pause=True),
            next_in_chain=next_in_chain,
        )
        self.below = next_in_chain
        self.frames = 0
        self.bytes = 0
        self.events: list[tuple[str, object]] = []
        for event in ("playback_started", "playback_finished"):
            self.on(event, lambda ev, event=event: self.events.append((event, ev)))

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await super().capture_frame(frame)
        self.frames += 1
        self.bytes += frame.data.nbytes
        await self.below.capture_frame(frame)

    def flush(self) -> None:
        super().flush()
        self.below.flush()

    def clear_buffer(self) -> None:
        self.below.clear_buffer()


async def until(condition, timeout: float = 5.0) -> None:
    async with asyncio.timeout(timeout):
        while not condition():
            await asyncio.sleep(0.01)


async def test_a_python_stage_plays_through_the_avatar_output_and_hears_its_playback():
    room = Room()
    output = DataStreamAudioOutput(room, destination_identity="avatar")
    stage = Stage(output)
    assert stage.can_pause is False, "the avatar cannot pause, so neither can its chain"
    outside: list[object] = []
    stage.on("playback_finished", outside.append)
    session = VoiceSession(
        vad=None, tts=FakeTTS(fake_audio_duration=0.3), turn_handling={"turn_detection": None}
    )
    session.output.audio = stage
    try:
        async with session:
            await session.start(Agent(instructions="x", name="desk"))

            handle = session.say("Hello.")
            local = room.local_participant
            await until(lambda: local.streams and local.streams[0].closed)
            assert stage.frames > 0
            sent = sum(len(chunk) for chunk in local.streams[0].chunks)
            assert sent == stage.bytes, "every frame reached the avatar's stream"
            assert [name for name, _ in stage.events] == ["playback_started"]
            assert not handle.done(), "the speech waits for the avatar's playback"
            room.finish_playback(0.3, interrupted=False)
            await asyncio.wait_for(handle, 5)
            assert not handle.interrupted
            finished = stage.events[-1]
            assert finished[0] == "playback_finished"
            assert (finished[1].playback_position, finished[1].interrupted) == (0.3, False)
            assert [(ev.playback_position, ev.interrupted) for ev in outside] == [(0.3, False)]
            played = await asyncio.wait_for(stage.wait_for_playout(), 5)
            assert (played.playback_position, played.interrupted) == (0.3, False)
            played = await asyncio.wait_for(output.wait_for_playout(), 5)
            assert (played.playback_position, played.interrupted) == (0.3, False)

            interrupted = session.say("A much longer reply the caller talks over.")
            await until(lambda: len(local.streams) == 2 and local.streams[1].chunks)
            interrupted.interrupt()
            await until(lambda: local.clears == 1)
            room.finish_playback(0.05, interrupted=True)
            await asyncio.wait_for(interrupted, 5)
            assert interrupted.interrupted
            name, ev = stage.events[-1]
            assert (name, ev.interrupted) == ("playback_finished", True)
            assert [ev.interrupted for ev in outside] == [False, True]
    finally:
        await output.aclose()


async def test_a_listener_on_a_middle_stage_hears_the_reports_the_head_also_gets():
    room = Room()
    output = DataStreamAudioOutput(room, destination_identity="avatar")
    middle = Stage(output, label="middle")
    head = Stage(middle, label="head")
    heard: list[str] = []
    middle.on("playback_started", lambda _: heard.append("started"))
    middle.on("playback_finished", lambda ev: heard.append(f"finished {ev.playback_position}"))
    session = VoiceSession(
        vad=None, tts=FakeTTS(fake_audio_duration=0.3), turn_handling={"turn_detection": None}
    )
    session.output.audio = head
    try:
        async with session:
            await session.start(Agent(instructions="x", name="desk"))

            handle = session.say("Hello.")
            local = room.local_participant
            await until(lambda: local.streams and local.streams[0].closed)
            room.finish_playback(0.3, interrupted=False)
            await asyncio.wait_for(handle, 5)
            assert not handle.interrupted
            assert heard == ["started", "finished 0.3"]
            assert [name for name, _ in middle.events] == ["playback_started", "playback_finished"]
            assert [name for name, _ in head.events] == ["playback_started", "playback_finished"]
            assert head.events[-1][1].playback_position == 0.3
    finally:
        await output.aclose()


async def test_an_output_belongs_to_one_chain():
    output = DataStreamAudioOutput(Room(), destination_identity="avatar")
    try:
        stage = Stage(output)
        output._set_parent(stage)
        with pytest.raises(ValueError, match="belongs to one chain"):
            Stage(output)

        head = Stage(stage, label="head")
        stage._set_parent(head)
        with pytest.raises(ValueError, match="belongs to one chain"):
            Stage(stage, label="another head")
    finally:
        await output.aclose()


async def test_an_interrupt_while_the_avatar_joins_leaves_the_chain_to_the_next_speech():
    room = Room(avatar_joined=False)
    output = DataStreamAudioOutput(room, destination_identity="avatar")
    stage = Stage(output)
    session = VoiceSession(
        vad=None, tts=FakeTTS(fake_audio_duration=0.3), turn_handling={"turn_detection": None}
    )
    session.output.audio = stage
    try:
        async with session:
            await session.start(Agent(instructions="x", name="desk"))

            waiting = session.say("Are you there?")
            await until(lambda: stage.frames > 0)
            await asyncio.sleep(0.1)
            assert not waiting.done(), "the first frame waits for the avatar to join"
            waiting.interrupt()
            await asyncio.wait_for(waiting, 5)
            assert waiting.interrupted
            local = room.local_participant
            assert not local.streams, "nothing reached an avatar that was not there"

            room.join_avatar()
            handle = session.say("Hello.")
            await until(lambda: local.streams and local.streams[0].closed)
            room.finish_playback(0.3, interrupted=False)
            await asyncio.wait_for(handle, 5)
            assert not handle.interrupted
            name, ev = stage.events[-1]
            assert (name, ev.playback_position, ev.interrupted) == ("playback_finished", 0.3, False)
    finally:
        await output.aclose()
