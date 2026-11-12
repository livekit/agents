"""End-to-end encryption on an isolated local SFU: what an encrypted agent publishes (speech and
data tracks) decrypts only for a peer holding the room's key, what such a peer publishes
(microphone, pre-connect buffer) reaches the agent, and room text keeps working for keyed peers.
Text and byte streams are asserted to work, never to be confidential."""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path

import numpy as np
import pytest

from livekit import rtc
from livekit.agents import Agent, VoiceSession, _core
from livekit.agents.job import JobContext
from livekit.agents.voice.events import ErrorEvent
from livekit.agents.voice.room_io import RoomOptions
from livekit.protocol import agent, models
from tests_new.fakes import FakeLLM, FakeSTT, FakeTTS, FakeVAD, until

from .processes import job_runner, running_job
from .sfu import DATA_KEY, connected, read_marker, room_token, tone, write_marker
from .test_audio_processors import listening_session, loudness
from .v1_clients import publish_microphone, send_preconnect_buffer


def keyed(key: bytes = DATA_KEY) -> rtc.RoomOptions:
    return rtc.RoomOptions(
        encryption=rtc.E2EEOptions(key_provider_options=rtc.KeyProviderOptions(shared_key=key))
    )


def pitch(frame: rtc.AudioFrame) -> int:
    """The dominant frequency of a loud frame, to 10 Hz; 0 for a quiet one."""
    samples = np.frombuffer(frame.data, dtype=np.int16).astype(np.float64)
    if np.sqrt(np.mean(samples**2)) < 300:
        return 0
    frequencies = np.fft.rfftfreq(samples.size, 1 / frame.sample_rate)
    return round(float(frequencies[np.argmax(abs(np.fft.rfft(samples)))]) / 10) * 10


async def launch(
    own_executor: Callable[[_core.JobSupervisor], _core.JobSupervisor],
    entrypoint: Callable[[JobContext], Awaitable[None]],
    sfu: str,
    room: str,
    marker: Path,
) -> _core.JobSupervisor:
    """Dispatches ``entrypoint`` to ``room`` in a job process; the job gets ``marker`` as its
    metadata."""
    process = own_executor(job_runner(entrypoint, mp_ctx=mp.get_context("spawn")))
    await process.start()
    await process.initialize()
    info = running_job(
        agent.Job(id="AJ_" + uuid.uuid4().hex, room=models.Room(name=room), metadata=str(marker)),
        fake_job=False,
    )
    info.url, info.token = sfu, room_token(room, "agent")
    await process.launch_job(info)
    return process


async def _job_publishing_a_data_track(ctx: JobContext) -> None:
    marker = Path(ctx.job.metadata)
    await ctx.connect(encryption=keyed().encryption)
    track = await ctx.room.local_participant.publish_data_track(name="private")
    write_marker(marker, track.info.uses_e2ee)
    async with asyncio.timeout(10):
        while not marker.with_suffix(".send").exists():
            await asyncio.sleep(0.01)
        track.try_push(rtc.DataTrackFrame(payload=b"private payload"))
        while not marker.with_suffix(".received").exists():
            await asyncio.sleep(0.01)
    ctx.shutdown("delivered")


async def test_an_encrypted_jobs_data_track_decrypts_only_for_a_keyed_peer(
    sfu: str,
    own_executor: Callable[[_core.JobSupervisor], _core.JobSupervisor],
    tmp_path: Path,
) -> None:
    """A job connected with `encryption=` encrypts its data tracks like its media: a peer with
    the key reads the payload, a peer without it does not.

    https://github.com/livekit/agents/issues/4991 https://github.com/livekit/agents/pull/4992
    """
    name = "e2ee-" + uuid.uuid4().hex
    marker = tmp_path / "data"
    async with (
        connected(sfu, name, "keyed", options=keyed()) as with_key,
        connected(sfu, name, "keyless") as without_key,
    ):
        streams: dict[str, rtc.DataTrackStream] = {}
        with_key.on("data_track_published", lambda t: streams.setdefault("keyed", t.subscribe()))
        without_key.on(
            "data_track_published", lambda t: streams.setdefault("keyless", t.subscribe())
        )
        try:
            process = await launch(own_executor, _job_publishing_a_data_track, sfu, name, marker)
            assert await read_marker(marker) is True
            await until(lambda: len(streams) == 2, "a peer never saw the data track")
            keyless_read = asyncio.ensure_future(anext(streams["keyless"]))
            marker.with_suffix(".send").touch()

            received = await anext(streams["keyed"])
            await asyncio.wait([keyless_read], timeout=0.5)

            assert received.payload == b"private payload"
            assert not keyless_read.done() or keyless_read.result().payload != b"private payload"
            marker.with_suffix(".received").touch()
            await process.join()
            assert process.exitcode == 0
        finally:
            marker.with_suffix(".received").touch()
            await asyncio.gather(*(stream.aclose() for stream in streams.values()))


async def _job_listening_to_the_microphone(ctx: JobContext) -> None:
    marker = Path(ctx.job.metadata)
    await ctx.connect(encryption=keyed().encryption)
    track: asyncio.Future[rtc.Track] = asyncio.get_running_loop().create_future()

    @ctx.room.on("track_subscribed")
    def subscribed(subscribed_track: rtc.Track, *_: object) -> None:
        if not track.done():
            track.set_result(subscribed_track)

    for participant in ctx.room.remote_participants.values():
        for publication in participant.track_publications.values():
            if publication.track is not None and not track.done():
                track.set_result(publication.track)
    stream = rtc.AudioStream(await track, sample_rate=48000, num_channels=1)
    pitches: list[int] = []
    async with asyncio.timeout(10):
        async for event in stream:
            pitches.append(pitch(event.frame))
            if len(pitches) == 100:
                break
    await stream.aclose()
    write_marker(marker, pitches)
    ctx.shutdown("heard")


async def test_an_encrypted_job_hears_encrypted_microphone_audio(
    sfu: str,
    own_executor: Callable[[_core.JobSupervisor], _core.JobSupervisor],
    tmp_path: Path,
) -> None:
    """A job connected with the room's key decrypts the person's audio into real samples, not
    silence. https://github.com/livekit/agents/issues/5231

    The job subscribes mid-stream: its first frames are the decoder starting up, or a packet that
    arrived before the key applied decoded as up to ~0.4 s of noise (livekit rtc), so the tone is
    asserted in every frame after half a second."""
    name = "e2ee-" + uuid.uuid4().hex
    marker = tmp_path / "heard"
    async with connected(sfu, name, "person", options=keyed()) as person:
        microphone, publication = await publish_microphone(person)
        assert publication.encryption_type != rtc.EncryptionType.NONE
        speaking = asyncio.create_task(play(microphone, tone(1000)))
        try:
            process = await launch(
                own_executor, _job_listening_to_the_microphone, sfu, name, marker
            )

            pitches = await read_marker(marker)

            assert isinstance(pitches, list) and pitches[50:] == [1000] * 50
            await process.join()
            assert process.exitcode == 0
        finally:
            speaking.cancel()
            await asyncio.gather(speaking, return_exceptions=True)
            await microphone.aclose()


async def play(source: rtc.AudioSource, frames: AsyncIterator[rtc.AudioFrame]) -> None:
    async for frame in frames:
        await source.capture_frame(frame)


async def pitches_of(room: rtc.Room, identity: str) -> AsyncIterator[int]:
    """The pitch of each frame ``identity`` publishes, as ``room`` decodes it."""
    track: asyncio.Future[rtc.Track] = asyncio.get_running_loop().create_future()

    def subscribed(t: rtc.Track, _: object, participant: rtc.RemoteParticipant) -> None:
        if participant.identity == identity and not track.done():
            track.set_result(t)

    room.on("track_subscribed", subscribed)
    stream = rtc.AudioStream(await track, sample_rate=24000, num_channels=1, frame_size_ms=100)
    try:
        async for event in stream:
            yield pitch(event.frame)
    finally:
        await stream.aclose()


async def test_the_agents_speech_is_audible_only_with_the_key(sfu: str) -> None:
    """An agent in an encrypted room speaks: a peer with the key hears its voice, a peer
    without it hears none.

    The keyless peer decodes the ciphertext as noise, whose dominant pitch matches the voice's
    in a few percent of frames by chance; the voice itself is in nearly every keyed frame."""
    name = "e2ee-" + uuid.uuid4().hex
    async with (
        connected(sfu, name, "keyed", options=keyed()) as with_key,
        connected(sfu, name, "keyless") as without_key,
        connected(sfu, name, "agent", options=keyed()) as agent_room,
    ):
        heard: dict[str, list[int]] = {"keyed": [], "keyless": []}

        async def listen(room: rtc.Room, who: str) -> None:
            async for frame_pitch in pitches_of(room, "agent"):
                heard[who].append(frame_pitch)

        listeners = [
            asyncio.create_task(listen(with_key, "keyed")),
            asyncio.create_task(listen(without_key, "keyless")),
        ]
        session = VoiceSession(
            llm=FakeLLM(),
            tts=FakeTTS(seconds=2, tone=440),
            vad=None,
            turn_handling={"turn_detection": None},
            aec_warmup_duration=None,
        )
        try:
            async with session:
                await session.start(Agent(instructions="Speak."), room=agent_room)
                speech = session.say("Hello.")
                await speech.wait_for_playout()
        finally:
            for listener in listeners:
                listener.cancel()
            await asyncio.gather(*listeners, return_exceptions=True)

    with_key_pitches, without_key_pitches = heard["keyed"], heard["keyless"]
    assert len(with_key_pitches) > 10
    assert with_key_pitches.count(440) > len(with_key_pitches) * 3 // 4
    assert without_key_pitches and without_key_pitches.count(440) < len(without_key_pitches) // 4


async def test_room_text_still_drives_replies_in_an_encrypted_room(sfu: str) -> None:
    """A keyed peer's `lk.chat` message is answered and the reply arrives as `lk.transcription`:
    encryption does not break room text."""
    name = "e2ee-" + uuid.uuid4().hex
    async with (
        connected(sfu, name, "person", options=keyed()) as person,
        connected(sfu, name, "agent", options=keyed()) as agent_room,
    ):
        replies: asyncio.Queue[str] = asyncio.Queue()
        readers: list[asyncio.Task[None]] = []

        def transcription(reader: rtc.TextStreamReader, identity: str) -> None:
            async def read() -> None:
                replies.put_nowait(await reader.read_all())

            readers.append(asyncio.create_task(read()))

        person.register_text_stream_handler("lk.transcription", transcription)
        session = VoiceSession(
            llm=FakeLLM({"hello": "Hi there."}), vad=None, turn_handling={"turn_detection": None}
        )
        room_options = RoomOptions(audio_input=False, audio_output=False)
        try:
            async with session:
                await session.start(
                    Agent(instructions="Answer."), room=agent_room, room_options=room_options
                )

                await person.local_participant.send_text("hello", topic="lk.chat")

                while (reply := await replies.get()) != "Hi there.":
                    pass
        finally:
            for reader in readers:
                reader.cancel()
            await asyncio.gather(*readers, return_exceptions=True)
    assert reply == "Hi there."


def early_words(sample_rate: int) -> bytes:
    """Half a second of loud speech, then quiet: one utterance."""
    n = sample_rate // 2
    loud = np.where(np.arange(n) % 2 == 0, 10000, -10000).astype(np.int16)
    return loud.tobytes() + bytes(2 * n)


async def test_a_keyed_callers_preconnect_audio_reaches_the_transcript(sfu: str) -> None:
    """Words recorded before the caller connected, sent with its encrypted microphone, are
    heard and transcribed by the agent."""
    name = "e2ee-" + uuid.uuid4().hex
    async with (
        connected(sfu, name, "person", options=keyed()) as person,
        connected(sfu, name, "agent", options=keyed()) as agent_room,
    ):
        session = VoiceSession(
            stt=FakeSTT(["words before connecting"]),
            llm=FakeLLM(["Welcome."]),
            vad=FakeVAD(),
            turn_handling={"turn_detection": "stt"},
            aec_warmup_duration=None,
        )

        def said() -> list[tuple[str, str | None]]:
            return [(i.role, i.text_content) for i in session.history.items if i.type == "message"]

        async with session:
            await session.start(Agent(instructions="Answer."), room=agent_room)
            microphone, publication = await publish_microphone(person, preconnect=True)

            await send_preconnect_buffer("android", person, "agent", publication.sid, early_words)

            await until(lambda: ("user", "words before connecting") in said(), f"heard: {said()}")
            await microphone.aclose()


async def test_a_caller_with_the_wrong_key_is_heard_as_silence_and_the_session_goes_on(
    sfu: str,
) -> None:
    """The agent cannot decrypt a caller holding another key: its STT gets silence, not noise
    or an error, and the session keeps answering.

    The first undecryptable packet can still reach the decoder and fade out as up to ~0.4 s of
    noise (livekit rtc; never with the right key), so silence is asserted after half a second."""
    name = "e2ee-" + uuid.uuid4().hex
    async with (
        connected(sfu, name, "person", options=keyed(b"another-key")) as person,
        connected(sfu, name, "agent", options=keyed()) as agent_room,
    ):
        heard: list[rtc.AudioFrame] = []
        session = listening_session(heard, llm=FakeLLM({"hello": "Hi there."}))
        microphone, _ = await publish_microphone(person)
        speaking = asyncio.create_task(play(microphone, tone(1000)))
        try:
            async with session:
                await session.start(Agent(instructions="Answer."), room=agent_room)
                await until(lambda: sum(f.duration for f in heard) > 1.5, "the STT got no audio")

                await session.generate_reply(user_input="hello")
        finally:
            speaking.cancel()
            await asyncio.gather(speaking, return_exceptions=True)
            await microphone.aclose()

    after_the_first_packet = heard[round(0.5 / heard[0].duration) :]
    assert max(loudness(f) for f in after_the_first_packet) == 0
    said = [(i.role, i.text_content) for i in session.history.items if i.type == "message"]
    assert said == [("user", "hello"), ("assistant", "Hi there.")]


@pytest.mark.xfail(
    strict=True,
    reason="decryption failure is silent: the agent's STT gets silence from a caller it cannot "
    "decrypt and neither the session nor the room (no e2ee_state_changed DECRYPTION_FAILED "
    "for the remote caller) reports it; voice/room_io.py and the runtime's room input have "
    "no E2EE handling at all",
)
async def test_a_caller_the_agent_cannot_decrypt_is_reported(sfu: str) -> None:
    """When the agent cannot decrypt the caller's audio, the session says so."""
    name = "e2ee-" + uuid.uuid4().hex
    async with (
        connected(sfu, name, "person", options=keyed(b"another-key")) as person,
        connected(sfu, name, "agent", options=keyed()) as agent_room,
    ):
        heard: list[rtc.AudioFrame] = []
        errors: list[ErrorEvent] = []
        session = listening_session(heard)
        session.on("error", errors.append)
        microphone, _ = await publish_microphone(person)
        speaking = asyncio.create_task(play(microphone, tone(1000)))
        try:
            async with session:
                await session.start(Agent(instructions="Listen."), room=agent_room)
                await until(lambda: sum(f.duration for f in heard) > 1, "the STT got no audio")

                await until(lambda: errors, "the session reported nothing", timeout=5)
        finally:
            speaking.cancel()
            await asyncio.gather(speaking, return_exceptions=True)
            await microphone.aclose()
