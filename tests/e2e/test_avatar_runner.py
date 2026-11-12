"""``AvatarRunner`` publishing an avatar in a real room: its microphone first, its camera once
someone listens, and nothing left behind once the runner closes.

The avatar is the real ``AvatarRunner`` over a ``QueueAudioOutput`` with ``FakeVideoGenerator``'s
face; the person is an SDK participant in the isolated ``sfu`` room."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from livekit import rtc
from livekit.agents.voice.avatar import AvatarOptions, AvatarRunner, QueueAudioOutput
from tests.fakes import FakeVideoGenerator, until

from .sfu import connected
from .v1_clients import track_of

AVATAR = "avatar"
OPTIONS = AvatarOptions(FakeVideoGenerator.SIZE, FakeVideoGenerator.SIZE, 25, 24000, 1)
MICROPHONE = rtc.TrackSource.SOURCE_MICROPHONE
CAMERA = rtc.TrackSource.SOURCE_CAMERA


def new_room() -> str:
    return "avatar-runner-" + uuid.uuid4().hex


@asynccontextmanager
async def avatar_runner(url: str, name: str) -> AsyncIterator[AvatarRunner]:
    async with connected(url, name, AVATAR) as room:
        audio = QueueAudioOutput(sample_rate=24000)
        runner = AvatarRunner(
            room, audio_recv=audio, video_gen=FakeVideoGenerator(), options=OPTIONS
        )
        await runner.start()
        try:
            yield runner
        finally:
            await runner.aclose()
            await audio.aclose()


def published(
    participant: rtc.RemoteParticipant, source: rtc.TrackSource.ValueType
) -> rtc.RemoteTrackPublication | None:
    return next((p for p in participant.track_publications.values() if p.source == source), None)


async def avatar_in(room: rtc.Room) -> rtc.RemoteParticipant:
    await until(lambda: AVATAR in room.remote_participants, "the avatar joins")
    return room.remote_participants[AVATAR]


async def test_the_avatar_publishes_its_camera_only_once_its_microphone_has_a_listener(
    sfu: str,
) -> None:
    """The avatar publishes its microphone first and its camera only once someone subscribes to
    the microphone, so its face never shows ahead of its voice."""
    name = new_room()
    unsubscribed = rtc.RoomOptions(auto_subscribe=False)
    async with (
        connected(sfu, name, "person", options=unsubscribed) as person,
        avatar_runner(sfu, name),
    ):
        avatar = await avatar_in(person)
        await until(lambda: published(avatar, MICROPHONE), "the avatar publishes its microphone")
        camera_before_listening = published(avatar, CAMERA)

        microphone = published(avatar, MICROPHONE)
        assert microphone is not None
        microphone.set_subscribed(True)

        await until(lambda: published(avatar, CAMERA), "the avatar publishes its camera")
    assert camera_before_listening is None


async def test_closing_the_runner_unpublishes_the_avatars_tracks(sfu: str) -> None:
    """Closing the runner takes the avatar's microphone and camera out of the room."""
    name = new_room()
    async with (
        connected(sfu, name, "person") as person,
        avatar_runner(sfu, name) as runner,
    ):
        avatar = await avatar_in(person)
        await track_of(avatar, CAMERA)

        await runner.aclose()

        await until(lambda: not avatar.track_publications, "the avatar's tracks are gone")
