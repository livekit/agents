"""The avatar's data-stream audio output in a real room: the stage chain over it, and which output
an avatar's close ends."""

from __future__ import annotations

import uuid

import pytest

from livekit import rtc
from livekit.agents import Agent, VoiceSession
from livekit.agents.voice.avatar import AvatarSession, DataStreamAudioOutput
from tests.fakes import FakeAudioStage, FakeTTS, until

from .conftest import JoinAvatar
from .sfu import connected


class _Avatar(AvatarSession):
    """An avatar plugin streaming the agent's speech to ``identity``, as the plugins do."""

    def __init__(self, identity: str, *, wait_remote_track: int | None = None) -> None:
        super().__init__()
        self._identity = identity
        self._wait_remote_track = wait_remote_track

    @property
    def avatar_identity(self) -> str:
        return self._identity

    async def start(self, agent_session: VoiceSession, room: rtc.Room) -> None:
        await super().start(agent_session, room)
        output = DataStreamAudioOutput(
            room,
            destination_identity=self._identity,
            wait_remote_track=self._wait_remote_track,
        )
        self._attach_audio_output(agent_session, output)


def new_room() -> str:
    return "avatar-" + uuid.uuid4().hex


async def test_a_listener_on_a_middle_stage_hears_the_reports_the_head_also_gets(
    sfu: str, join_avatar: JoinAvatar
) -> None:
    """A stage in the middle of the chain hears the avatar's playback reports, as the head does."""
    name = new_room()
    heard: list[str] = []
    async with connected(sfu, name, "agent") as room, join_avatar(sfu, name) as avatar:
        output = DataStreamAudioOutput(room, destination_identity="avatar")
        middle = FakeAudioStage(output)
        head = FakeAudioStage(middle)
        middle.on("playback_started", lambda _: heard.append("started"))
        middle.on("playback_finished", lambda ev: heard.append(f"finished {ev.playback_position}"))
        session = VoiceSession(vad=None, tts=FakeTTS(seconds=0.3))
        session.output.audio = head
        try:
            async with session:
                await session.start(Agent(instructions="Help."))
                speech = session.say("Hello.")

                segment, _ = await avatar.segments.get()
                await avatar.played(segment, 0.3)
                await speech
        finally:
            await output.aclose()

    assert heard == ["started", "finished 0.3"]
    assert [(f.playback_position, f.interrupted) for f in head.finished] == [(0.3, False)]


async def test_a_data_stream_output_belongs_to_one_chain(sfu: str) -> None:
    """A second stage cannot be built over an avatar output another stage already wraps."""
    async with connected(sfu, new_room(), "agent") as room:
        output = DataStreamAudioOutput(room, destination_identity="avatar")
        FakeAudioStage(output)
        try:
            with pytest.raises(ValueError, match="belongs to one chain"):
                FakeAudioStage(output)
        finally:
            await output.aclose()


async def test_a_stage_over_the_avatar_output_belongs_to_one_chain(sfu: str) -> None:
    """A second head cannot be built over a stage another head already wraps."""
    async with connected(sfu, new_room(), "agent") as room:
        output = DataStreamAudioOutput(room, destination_identity="avatar")
        stage = FakeAudioStage(output)
        FakeAudioStage(stage)
        try:
            with pytest.raises(ValueError, match="belongs to one chain"):
                FakeAudioStage(stage)
        finally:
            await output.aclose()


async def test_closing_a_replaced_avatar_leaves_the_replacement_speaking(
    sfu: str, join_avatar: JoinAvatar
) -> None:
    """Closing an avatar whose output another avatar replaced ends only its own output: the
    agent's next speech still plays through the replacement."""
    name = new_room()
    async with connected(sfu, name, "agent") as room, join_avatar(sfu, name) as avatar:
        session = VoiceSession(vad=None, tts=FakeTTS(seconds=0.3))
        replaced, replacement = _Avatar("former-avatar"), _Avatar("avatar")
        async with session:
            await session.start(Agent(instructions="Help."))
            await replaced.start(session, room)
            await replacement.start(session, room)
            try:
                await replaced.aclose()

                speech = session.say("Hello.")
                segment, audio = await avatar.segments.get()
                await avatar.played(segment, 0.3)
                await speech
            finally:
                await replacement.aclose()

    assert audio and not speech.interrupted


async def test_closing_an_avatar_that_never_joined_ends_the_speech_waiting_for_it(
    sfu: str,
) -> None:
    """Closing an avatar whose output waits for the avatar's video ends the speech held at that
    wait, interrupted; a second close returns too."""
    async with connected(sfu, new_room(), "agent") as room:
        session = VoiceSession(vad=None, tts=FakeTTS(seconds=0.3))
        avatar = _Avatar("avatar", wait_remote_track=rtc.TrackKind.KIND_VIDEO)
        async with session:
            await session.start(Agent(instructions="Help."))
            await avatar.start(session, room)
            assert session.output.audio is not None
            stage = FakeAudioStage(session.output.audio)
            session.output.audio = stage
            speech = session.say("Are you there?")
            await until(lambda: stage.frames > 0, "the speech never reached the avatar's output")

            await avatar.aclose()
            await avatar.aclose()
            await speech

    assert speech.interrupted
