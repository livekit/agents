"""Avatars in a real room (``COMPAT.md`` #4): the person sees and hears the avatar speak the
agent's replies; an avatar provider's worker still on v1 works with a 2.0 agent, and one
upgraded to 2.0 plays a v1 agent's speech.

The person is the v1 client of ``v1_clients.py``. The 2.0 agent is the compat worker's job; the
2.0 avatar worker is the real ``DataStreamAudioReceiver`` and ``AvatarRunner`` with
``FakeVideoGenerator``'s face, written as v1 providers write it. The v1 peers
(``V1AvatarWorker``, ``V1AvatarAgent``) do on the wire what v1 does, checked against the
published v1 release."""

from __future__ import annotations

import asyncio
import uuid

import pytest

from livekit import rtc
from tests.fakes import until

from .v1_clients import (
    AVATAR_IDENTITY,
    V1_AGENT_IDENTITY,
    V1AvatarAgent,
    avatar_worker,
    hearing,
    joined,
    next_transcription,
    publish_microphone,
    read_transcriptions,
    speak,
    starter_token,
    v1_agent_token,
    v1_avatar_worker,
    wait_for_agent,
    wait_for_agent_state,
    wait_for_avatar,
    watching,
)

REPLIES = {"hello there agent": "hi, how can I help you today", "what time is it": "it is noon"}
LONG = "this answer runs long enough for the caller to cut in before it ends"
CUT_IN = {"hello there agent": LONG, "what time is it": "it is noon"}


def new_room() -> str:
    return "avatar-" + uuid.uuid4().hex


@pytest.mark.usefixtures("compat_worker")
async def test_the_person_hears_and_sees_the_avatar_speak_the_reply(sfu: str) -> None:
    """The agent's speech reaches the person only through the avatar: its microphone carries the
    reply while its camera shows the face mouthing it, and the agent listens again once the
    avatar reports the reply played."""
    name = new_room()
    async with joined(
        sfu, starter_token(name, metadata={"avatar": True, "replies": REPLIES})
    ) as room:
        agent = await wait_for_agent(room)
        async with avatar_worker(sfu, name, agent.identity):
            avatar = await wait_for_avatar(room, agent.identity)
            await wait_for_agent_state(room, "listening")
            microphone, _ = await publish_microphone(room)
            async with hearing(avatar) as heard, watching(avatar) as seen:
                await speak(microphone, "hello there agent")
                await until(lambda: heard.seconds > 1.0, "the avatar speaks the reply")
                await wait_for_agent_state(room, "listening")

        assert seen.lit > 10 and seen.dark > 0
        assert not [
            p for p in agent.track_publications.values() if p.kind == rtc.TrackKind.KIND_AUDIO
        ]


@pytest.mark.usefixtures("compat_worker")
async def test_talking_over_the_avatar_clears_it_and_the_next_reply_follows(sfu: str) -> None:
    """Talking over the avatar clears its buffer: the person stops hearing the long reply, its
    transcript ends cut, and the answer to the new words plays."""
    name = new_room()
    async with joined(
        sfu, starter_token(name, metadata={"avatar": True, "replies": CUT_IN})
    ) as room:
        transcriptions = read_transcriptions(room)
        agent = await wait_for_agent(room)
        async with avatar_worker(sfu, name, agent.identity) as face:
            avatar = await wait_for_avatar(room, agent.identity)
            await wait_for_agent_state(room, "listening")
            microphone, _ = await publish_microphone(room)
            async with hearing(avatar) as heard:
                await speak(microphone, "hello there agent")
                _, long_reply = await next_transcription(transcriptions, agent.identity)
                await until(lambda: heard.seconds > 0.3, "the avatar starts the long reply")

                await speak(microphone, "what time is it")
                _, answer = await next_transcription(transcriptions, agent.identity)

                assert await answer.read_all() == "it is noon"
                await wait_for_agent_state(room, "listening")

        cut = await long_reply.read_all()
        assert face.clears == 1
        assert LONG.startswith(cut) and len(cut) < len(LONG), cut
        assert heard.seconds < 0.06 * len(LONG), "the long reply was heard to its end"


@pytest.mark.usefixtures("compat_worker")
async def test_legacy_transcription_packets_name_the_avatars_microphone(sfu: str) -> None:
    """As in v1, the agent's ``TranscriptionReceived`` packets name the avatar and its microphone,
    the audio clients play (components-react ``useTrackTranscription``), while its
    ``lk.transcription`` stream names no track (v1 ``room_io/_output.py:600-613, 937-950``)."""
    name = new_room()
    async with joined(
        sfu, starter_token(name, metadata={"avatar": True, "replies": REPLIES})
    ) as room:
        packets: list[tuple[str | None, str | None, rtc.TranscriptionSegment]] = []
        room.on(
            "transcription_received",
            lambda segments, participant, publication: packets.extend(
                (participant and participant.identity, publication and publication.sid, s)
                for s in segments
            ),
        )
        transcriptions = read_transcriptions(room)
        agent = await wait_for_agent(room)
        async with avatar_worker(sfu, name, agent.identity):
            avatar = await wait_for_avatar(room, agent.identity)
            await wait_for_agent_state(room, "listening")
            microphone, _ = await publish_microphone(room)
            await speak(microphone, "hello there agent")
            header, reply = await next_transcription(transcriptions, agent.identity)
            text = await reply.read_all()
            await until(
                lambda: any(s.final and s.text == text for _, _, s in packets),
                "the reply's final packet",
            )
            [avatar_microphone] = [
                p.sid
                for p in avatar.track_publications.values()
                if p.source == rtc.TrackSource.SOURCE_MICROPHONE
            ]

        assert {(who, sid) for who, sid, s in packets if s.text == text} == {
            (AVATAR_IDENTITY, avatar_microphone)
        }
        assert "lk.transcribed_track_id" not in header


@pytest.mark.usefixtures("compat_worker")
async def test_a_v1_avatar_worker_plays_the_agents_reply(sfu: str) -> None:
    """A provider's v1 avatar worker reports on the plain ``lk.playback_finished``; the 2.0 agent
    accepts it and listens again."""
    name = new_room()
    async with joined(
        sfu, starter_token(name, metadata={"avatar": True, "replies": REPLIES})
    ) as room:
        agent = await wait_for_agent(room)
        async with v1_avatar_worker(sfu, name, agent.identity) as worker:
            avatar = await wait_for_avatar(room, agent.identity)
            await wait_for_agent_state(room, "listening")
            microphone, _ = await publish_microphone(room)
            async with hearing(avatar) as heard:
                await speak(microphone, "hello there agent")
                await until(lambda: worker.reports, "the worker reports the reply played")
                await wait_for_agent_state(room, "listening")

        assert heard.seconds > 1.0
        [(payload, answer)] = worker.reports
        assert payload["interrupted"] is False
        assert answer == "ok"


@pytest.mark.usefixtures("compat_worker")
async def test_a_v1_avatar_worker_is_cleared_when_the_person_talks_over_it(sfu: str) -> None:
    """Talking over a provider's v1 avatar worker sends it ``lk.clear_buffer``; its interrupted
    report on the plain method settles the cut reply, and the next reply plays through it."""
    name = new_room()
    async with joined(
        sfu, starter_token(name, metadata={"avatar": True, "replies": CUT_IN})
    ) as room:
        transcriptions = read_transcriptions(room)
        agent = await wait_for_agent(room)
        async with v1_avatar_worker(sfu, name, agent.identity) as worker:
            avatar = await wait_for_avatar(room, agent.identity)
            await wait_for_agent_state(room, "listening")
            microphone, _ = await publish_microphone(room)
            async with hearing(avatar) as heard:
                await speak(microphone, "hello there agent")
                _, long_reply = await next_transcription(transcriptions, agent.identity)
                await until(lambda: heard.seconds > 0.3, "the worker starts the long reply")

                await speak(microphone, "what time is it")
                _, answer = await next_transcription(transcriptions, agent.identity)

                assert await answer.read_all() == "it is noon"
                await until(lambda: len(worker.reports) == 2, "the worker plays the answer")

        cut = await long_reply.read_all()
        assert LONG.startswith(cut) and len(cut) < len(LONG), cut
        assert worker.clears == 1
        assert [(p["interrupted"], answer) for p, answer in worker.reports] == [
            (True, "ok"),
            (False, "ok"),
        ]


async def test_a_2_0_avatar_worker_plays_a_v1_agents_speech(sfu: str) -> None:
    """A provider's worker upgraded to 2.0 plays a v1 agent's ``AUDIO_<shortuuid>`` stream and
    reports it finished on the plain ``lk.playback_finished``, even when the agent joins after the
    worker and streams the moment it is in the room."""
    name = new_room()
    async with joined(sfu, starter_token(name, agent=None)) as room:
        async with avatar_worker(sfu, name, V1_AGENT_IDENTITY):
            avatar = await wait_for_avatar(room, V1_AGENT_IDENTITY)
            async with (
                joined(sfu, v1_agent_token(name)) as agent_room,
                hearing(avatar) as heard,
                watching(avatar) as seen,
            ):
                agent = V1AvatarAgent(agent_room)
                await agent.say(1.0)
                async with asyncio.timeout(10):  # a lost segment is never reported
                    report = await agent.finished.get()

    assert report["interrupted"] is False
    assert 0.8 < float(str(report["playback_position"])) < 1.2
    assert heard.seconds > 0.6
    assert seen.lit > 10


async def test_a_v1_agent_interrupts_a_2_0_avatar_worker(sfu: str) -> None:
    """A v1 agent's ``lk.clear_buffer`` stops a provider's 2.0 worker mid-segment, and the
    worker reports the segment interrupted, at what it played, on the plain
    ``lk.playback_finished``."""
    name = new_room()
    async with joined(sfu, starter_token(name, agent=None)) as room:
        async with avatar_worker(sfu, name, V1_AGENT_IDENTITY) as face:
            avatar = await wait_for_avatar(room, V1_AGENT_IDENTITY)
            async with joined(sfu, v1_agent_token(name)) as agent_room, hearing(avatar) as heard:
                agent = V1AvatarAgent(agent_room)
                await agent.say(4.0)
                await until(lambda: heard.seconds > 0.5, "the avatar speaks")

                await agent.interrupt()
                report = await agent.finished.get()

    assert report["interrupted"] is True
    assert float(str(report["playback_position"])) < 3.0
    assert face.clears == 1
    assert heard.seconds < 3.0
