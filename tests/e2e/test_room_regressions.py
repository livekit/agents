"""Room contracts on an isolated local SFU and real job processes."""

from __future__ import annotations

import asyncio
import contextlib
import gc
import json
import multiprocessing as mp
import uuid
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Literal

import numpy as np
import psutil
import pytest

from livekit import api, rtc
from livekit.agents import Agent, AgentServer, ModelSettings, RunContext, VoiceSession, _core
from livekit.agents.job import JobContext
from livekit.agents.utils import wait_for_participant
from livekit.agents.voice.background_audio import BackgroundAudioPlayer
from livekit.agents.voice.room_io import RoomOptions
from livekit.protocol import agent, models
from tests.fakes import FakeTTS, until

from .processes import job_runner, running_job
from .sfu import KEY, SECRET, connected, read_marker, room_token, tone, write_marker


async def _room_job(ctx: JobContext) -> None:
    marker = Path(ctx.job.metadata)

    async def shutdown(reason: str) -> None:
        write_marker(marker.with_suffix(".closed"), reason)

    ctx.add_shutdown_callback(shutdown)
    await ctx.connect()
    if ctx.proc.user_arguments == "initializing":
        desk = Agent(instructions="Initialize the call.", name="initializing")

        @desk.on_enter
        async def enter(session: VoiceSession) -> None:
            write_marker(marker, "initializing")
            await asyncio.Future()

        session = VoiceSession(vad=None, turn_handling={"turn_detection": None})
        await session.start(
            desk,
            room_options=RoomOptions(
                audio_input=False, audio_output=False, text_input=False, transcription_output=False
            ),
        )
    write_marker(marker, ctx.room.local_participant.identity)


async def _start_room_job(
    own_executor: Callable[[_core.JobSupervisor], _core.JobSupervisor],
    url: str,
    name: str,
    marker: Path,
    phase: str = "connected",
) -> _core.JobSupervisor:
    process = own_executor(
        job_runner(_room_job, mp_ctx=mp.get_context("spawn"), user_arguments=phase)
    )
    await process.start()
    await process.initialize()
    info = running_job(
        agent.Job(id="AJ_" + uuid.uuid4().hex, room=models.Room(name=name), metadata=str(marker)),
        fake_job=False,
    )
    info.url, info.token = url, room_token(name, "agent")
    await process.launch_job(info)
    return process


@pytest.mark.parametrize(
    "publishing",
    [
        pytest.param(
            False,
            marks=pytest.mark.xfail(
                strict=True,
                reason="https://github.com/livekit/agents/issues/6485 reproduces: subscriber activity arrives after a 3s batch",
            ),
        ),
        True,
    ],
)
async def test_participant_wait_returns_active_without_a_broadcast_batch_delay(
    sfu: str, publishing: bool
) -> None:
    """Participant waits return active peers promptly. https://github.com/livekit/agents/pull/5271"""
    name = "regression-" + uuid.uuid4().hex
    async with connected(sfu, name, "observer") as observer:
        for index in range(3):
            identity = f"peer-{index}"
            waiting = asyncio.create_task(wait_for_participant(observer, identity=identity))
            source = None
            try:
                async with connected(sfu, name, identity) as peer:
                    joined = asyncio.get_running_loop().time()
                    if publishing:
                        source = rtc.AudioSource(48000, 1)
                        track = rtc.LocalAudioTrack.create_audio_track("microphone", source)
                        await peer.local_participant.publish_track(track)
                    participant = await waiting
                    assert participant.identity == identity
                    assert participant.state == rtc.ParticipantState.PARTICIPANT_STATE_ACTIVE
                    assert asyncio.get_running_loop().time() - joined < 1.5
            finally:
                if not waiting.done():
                    waiting.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await waiting
                if source is not None:
                    await source.aclose()


async def _wait_for_the_person(ctx: JobContext) -> None:
    participant = await ctx.wait_for_participant()
    write_marker(Path(ctx.job.metadata), participant.identity)
    ctx.shutdown()


async def test_a_jobs_participant_wait_passes_over_an_egress_already_in_the_room(
    sfu: str, own_executor: Callable[[_core.JobSupervisor], _core.JobSupervisor], tmp_path: Path
) -> None:
    """`ctx.wait_for_participant()` skips a recording egress and returns the person who joins.

    https://github.com/livekit/agents/pull/1005
    """
    name = "regression-" + uuid.uuid4().hex
    egress = rtc.Room()
    grants = api.VideoGrants(room_join=True, room=name)
    token = api.AccessToken(KEY, SECRET).with_identity("EG_recorder").with_kind("egress")
    await egress.connect(sfu, token.with_grants(grants).to_jwt())
    try:
        runner = own_executor(job_runner(_wait_for_the_person, thread=True))
        await runner.start()
        await runner.initialize()
        job = agent.Job(id="AJ_egress", room=models.Room(name=name), metadata=str(tmp_path / "w"))
        info = running_job(job, fake_job=False)
        info.url, info.token = sfu, room_token(name, "agent")
        await runner.launch_job(info)
        await wait_for_participant(egress, identity="agent")

        async with connected(sfu, name, "person"):
            assert await read_marker(tmp_path / "w") == "person"
    finally:
        await egress.disconnect()


async def test_concurrent_job_connections_leave_existing_jobs_alive(
    sfu: str,
    own_executor: Callable[[_core.JobSupervisor], _core.JobSupervisor],
    tmp_path: Path,
) -> None:
    """Concurrent native connections preserve existing jobs. https://github.com/livekit/agents/issues/7545"""
    names = ["regression-" + uuid.uuid4().hex for _ in range(3)]
    first = await _start_room_job(own_executor, sfu, names[0], tmp_path / "job0")
    assert await read_marker(tmp_path / "job0") == "agent"
    rest = await asyncio.gather(
        *(
            _start_room_job(own_executor, sfu, name, tmp_path / f"job{index}")
            for index, name in enumerate(names[1:], 1)
        )
    )
    assert await asyncio.gather(*(read_marker(tmp_path / f"job{i}") for i in (1, 2))) == [
        "agent",
        "agent",
    ]
    async with api.LiveKitAPI(sfu, KEY, SECRET) as client:
        for name in names:
            participants = await client.room.list_participants(
                api.ListParticipantsRequest(room=name)
            )
            assert [p.identity for p in participants.participants] == ["agent"]
    assert first.exitcode is None
    for process in [first, *rest]:
        await process.aclose()
        assert process.exitcode == 0 and not process.killed


@pytest.mark.parametrize("phase", ["connected", "initializing", "concurrent_shutdown"])
async def test_room_deletion_delivers_shutdown_callbacks_and_reaps_the_job(
    sfu: str,
    own_executor: Callable[[_core.JobSupervisor], _core.JobSupervisor],
    tmp_path: Path,
    phase: str,
) -> None:
    """Deleting a room settles its job and callbacks, though nobody else ever joined.

    https://github.com/livekit/agents/pull/254 https://github.com/livekit/agents/issues/4152
    """
    names = ["regression-" + uuid.uuid4().hex for _ in range(3)]
    processes = await asyncio.gather(
        *(
            _start_room_job(own_executor, sfu, name, tmp_path / f"job{i}", phase)
            for i, name in enumerate(names)
        )
    )
    await asyncio.gather(*(read_marker(tmp_path / f"job{i}") for i in range(3)))
    async with api.LiveKitAPI(sfu, KEY, SECRET) as client:
        actions: list[Awaitable[object]] = [
            client.room.delete_room(api.DeleteRoomRequest(room=n)) for n in names
        ]
        if phase == "concurrent_shutdown":
            actions.extend(p.aclose() for p in processes)
        await asyncio.gather(*actions)
    await asyncio.gather(*(p.join() for p in processes))
    for i, process in enumerate(processes):
        assert await read_marker(tmp_path / f"job{i}.closed")
        assert process.exitcode == 0 and not process.killed


async def test_agent_shutdown_keeps_the_human_in_the_room(
    sfu: str,
    own_executor: Callable[[_core.JobSupervisor], _core.JobSupervisor],
    tmp_path: Path,
) -> None:
    """Agent shutdown leaves human participants connected. https://github.com/livekit/agents/issues/3597"""
    name = "regression-" + uuid.uuid4().hex
    async with connected(sfu, name, "human") as human:
        process = await _start_room_job(own_executor, sfu, name, tmp_path / "agent")
        assert await read_marker(tmp_path / "agent") == "agent"
        await process.aclose()
        assert human.isconnected()
        async with api.LiveKitAPI(sfu, KEY, SECRET) as client:
            participants = await client.room.list_participants(
                api.ListParticipantsRequest(room=name)
            )
            assert [p.identity for p in participants.participants] == ["human"]
        assert process.exitcode == 0


async def _dispatch_job(ctx: JobContext) -> None:
    metadata = json.loads(ctx.job.room.metadata)
    write_marker(Path(metadata["marker"]), [ctx.job.room.metadata, ctx.job.metadata])
    ctx.shutdown("metadata delivered")


async def test_self_hosted_dispatch_preserves_room_and_dispatch_metadata(
    sfu: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Self-hosted dispatch delivers both metadata payloads. https://github.com/livekit/agents/issues/713"""
    marker = tmp_path / "metadata"
    name = "regression-" + uuid.uuid4().hex
    room_metadata = json.dumps({"marker": str(marker), "room": "a different payload"})
    dispatch_metadata = '{"summoner_identity":"person", "unicode":"cafe\\u00e9"}'
    monkeypatch.setenv("LIVEKIT_WORKER_INSECURE", "1")
    monkeypatch.setenv("LIVEKIT_AGENTS_PRELOAD_LOCAL_INFERENCE", "0")
    server = AgentServer(
        ws_url=sfu,
        api_key=KEY,
        api_secret=SECRET,
        agent_name=name,
        num_idle_processes=0,
        listeners={"health": {"port": 0}},
    )
    server.rtc_session()(_dispatch_job)
    ready = asyncio.Event()
    server._on_started.append(ready.set)
    run = asyncio.create_task(server._run(devmode=True))
    async with api.LiveKitAPI(sfu, KEY, SECRET) as client:
        try:
            await ready.wait()
            await until(lambda: server.id not in ("", "unregistered"), "the worker registered")
            await client.room.create_room(api.CreateRoomRequest(name=name, metadata=room_metadata))
            await client.agent_dispatch.create_dispatch(
                api.CreateAgentDispatchRequest(
                    room=name, agent_name=name, metadata=dispatch_metadata
                )
            )
            assert await read_marker(marker) == [room_metadata, dispatch_metadata]
        finally:
            await server.aclose()
            await run
            await client.room.delete_room(api.DeleteRoomRequest(room=name))


# No hidden-publisher case: this SFU forwards no track of a hidden participant, even a plain
# rtc one, so issue 4556 is SFU semantics rather than an agents bug.
@pytest.mark.parametrize("background_first", [False, True])
async def test_background_and_speech_tracks_are_both_audible(
    sfu: str, background_first: bool
) -> None:
    """Speech and background audio both reach a listener. https://github.com/livekit/agents/issues/2560"""
    name = "regression-" + uuid.uuid4().hex
    async with (
        connected(sfu, name, "listener") as listener,
        connected(sfu, name, "agent") as publisher,
    ):
        heard: dict[str, set[int]] = {}
        streams: list[rtc.AudioStream] = []
        readers: list[asyncio.Task[None]] = []

        async def listen(stream: rtc.AudioStream, sid: str) -> None:
            async for event in stream:
                samples = np.frombuffer(event.frame.data, dtype=np.int16).astype(float)
                if np.sqrt(np.mean(samples**2)) > 50:
                    frequencies = np.fft.rfftfreq(len(samples), 1 / event.frame.sample_rate)
                    peak = frequencies[np.argmax(abs(np.fft.rfft(samples)))]
                    heard.setdefault(sid, set()).add(round(float(peak) / 100) * 100)

        @listener.on("track_subscribed")
        def subscribed(
            track: rtc.Track,
            publication: rtc.RemoteTrackPublication,
            participant: rtc.RemoteParticipant,
        ) -> None:
            if track.kind == rtc.TrackKind.KIND_AUDIO:
                stream = rtc.AudioStream(track, sample_rate=48000, num_channels=1)
                streams.append(stream)
                readers.append(asyncio.create_task(listen(stream, publication.sid)))

        session = VoiceSession(tts=FakeTTS(), vad=None, turn_handling={"turn_detection": None})

        @session.tts_node
        async def voice(
            ctx: RunContext, text: AsyncIterable[str], settings: ModelSettings
        ) -> AsyncIterator[rtc.AudioFrame]:
            async for _ in text:
                pass
            async for frame in tone(1000, 100):
                yield frame

        background = BackgroundAudioPlayer(ambient_sound=tone(400))
        try:
            if background_first:
                await background.start(room=publisher)
            await session.start(
                Agent(instructions="Speak supplied audio."),
                room=publisher,
                room_options=RoomOptions(
                    audio_input=False, text_input=False, transcription_output=False
                ),
            )
            if not background_first:
                await background.start(room=publisher)
            speech = session.say("speech marker")
            await speech.wait_for_playout()
            assert speech.exception() is None, speech.exception()
            await until(
                lambda: (
                    len(heard) == 2
                    and any(400 in frequencies for frequencies in heard.values())
                    and any(1000 in frequencies for frequencies in heard.values())
                ),
                f"both tones heard on two tracks: {heard}",
            )
            assert background.publication is not None
            assert 400 in heard[background.publication.sid]
            assert any(
                1000 in frequencies
                for sid, frequencies in heard.items()
                if sid != background.publication.sid
            )
        finally:
            await session.aclose()
            await background.aclose()
            await asyncio.gather(*(stream.aclose() for stream in streams))
            for task in readers:
                task.cancel()
            await asyncio.gather(*readers, return_exceptions=True)


@pytest.mark.parametrize("event", ["data_received", "participant_attributes_changed"])
async def test_room_callbacks_can_publish_audio_and_video(
    sfu: str, event: Literal["data_received", "participant_attributes_changed"]
) -> None:
    """Room callbacks can publish both media tracks without deadlock. https://github.com/livekit/agents/issues/1602"""
    name = "regression-" + uuid.uuid4().hex
    async with (
        connected(sfu, name, "agent") as publisher,
        connected(sfu, name, "trigger") as trigger,
    ):
        await wait_for_participant(publisher, identity="trigger")
        audio = rtc.AudioSource(48000, 1)
        video = rtc.VideoSource(16, 16)
        tasks: list[asyncio.Task[list[rtc.LocalTrackPublication]]] = []

        async def publish() -> list[rtc.LocalTrackPublication]:
            camera = await publisher.local_participant.publish_track(
                rtc.LocalVideoTrack.create_video_track("camera", video),
                rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA),
            )
            microphone = await publisher.local_participant.publish_track(
                rtc.LocalAudioTrack.create_audio_track("microphone", audio),
                rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
            )
            return [camera, microphone]

        @publisher.on(event)
        def on_event(*args: object) -> None:
            if not tasks:
                tasks.append(asyncio.create_task(publish()))

        try:
            if event == "data_received":
                await trigger.local_participant.publish_data(b"publish both tracks")
            else:
                await trigger.local_participant.set_attributes({"regression": "publish"})
            await until(lambda: bool(tasks), f"the {event} callback ran")
            publications = await tasks[0]
            expected = {p.sid for p in publications}
            await until(
                lambda: expected.issubset(trigger.remote_participants["agent"].track_publications),
                "the trigger sees both tracks",
            )
            assert {p.kind for p in publications} == {
                rtc.TrackKind.KIND_AUDIO,
                rtc.TrackKind.KIND_VIDEO,
            }
        finally:
            publisher.off(event, on_event)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await audio.aclose()
            await video.aclose()


async def test_repeated_subscribe_consume_disconnect_stabilizes_native_memory(sfu: str) -> None:
    """Repeated RTC subscriptions release native allocations. https://github.com/livekit/agents/issues/2166"""

    async def cycle() -> None:
        name = "regression-" + uuid.uuid4().hex
        async with (
            connected(
                sfu, name, "subscriber", options=rtc.RoomOptions(auto_subscribe=False)
            ) as receiver,
            connected(sfu, name, "publisher") as sender,
        ):
            subscribed: asyncio.Future[rtc.Track] = asyncio.get_running_loop().create_future()

            @receiver.on("track_published")
            def publication(
                publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant
            ) -> None:
                publication.set_subscribed(True)

            @receiver.on("track_subscribed")
            def track(track: rtc.Track, *args: object) -> None:
                if not subscribed.done():
                    subscribed.set_result(track)

            source = rtc.AudioSource(48000, 1)
            stream = None
            try:
                await sender.local_participant.publish_track(
                    rtc.LocalAudioTrack.create_audio_track("microphone", source)
                )
                remote = await subscribed
                stream = rtc.AudioStream(remote, sample_rate=48000, num_channels=1)
                async for frame in tone(1000, 10):
                    await source.capture_frame(frame)
                async with asyncio.timeout(5):
                    async for event in stream:
                        if max(abs(sample) for sample in event.frame.data) > 100:
                            break
                await source.wait_for_playout()
            finally:
                if stream is not None:
                    await stream.aclose()
                await source.aclose()

    process = psutil.Process()
    for _ in range(3):
        await cycle()
    gc.collect()
    baseline = process.memory_info().rss
    samples = []
    for _ in range(12):
        await cycle()
        gc.collect()
        samples.append(process.memory_info().rss)
    assert max(samples) - baseline < 8 * 1024 * 1024, (baseline, samples)


async def test_initial_pcm_greeting_keeps_its_prefix_for_a_late_subscriber(sfu: str) -> None:
    """Initial PCM speech retains its prefix for late subscribers. https://github.com/livekit/agents/issues/4587"""
    name = "regression-" + uuid.uuid4().hex
    async with (
        connected(sfu, name, "listener", options=rtc.RoomOptions(auto_subscribe=False)) as listener,
        connected(sfu, name, "agent") as publisher,
    ):
        loop = asyncio.get_running_loop()
        published: asyncio.Future[rtc.RemoteTrackPublication] = loop.create_future()
        subscribed: asyncio.Future[rtc.Track] = loop.create_future()

        @listener.on("track_published")
        def publication(
            publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant
        ) -> None:
            if publication.kind == rtc.TrackKind.KIND_AUDIO and not published.done():
                published.set_result(publication)

        @listener.on("track_subscribed")
        def track(track: rtc.Track, *args: object) -> None:
            if not subscribed.done():
                subscribed.set_result(track)

        async def greeting() -> AsyncIterator[rtc.AudioFrame]:
            async for frame in tone(1600, 5):
                yield frame
            async for frame in tone(1000, 45):
                yield frame

        session = VoiceSession(tts=FakeTTS(), vad=None, turn_handling={"turn_detection": None})
        desk = Agent(instructions="Greet the caller.", name="greeter")
        speeches = []

        @desk.on_enter
        async def greet(active: VoiceSession) -> None:
            speeches.append(active.say("greeting marker", audio=greeting()))

        stream = None
        try:
            await session.start(
                desk,
                room=publisher,
                room_options=RoomOptions(
                    audio_input=False, text_input=False, transcription_output=False
                ),
            )
            publication = await published
            # This exceeds the 100 ms prefix: capturing before subscription loses that prefix.
            await asyncio.sleep(0.25)
            publication.set_subscribed(True)
            stream = rtc.AudioStream(await subscribed, sample_rate=48000, num_channels=1)
            heard: list[int] = []
            async with asyncio.timeout(5):
                async for event in stream:
                    samples = np.frombuffer(event.frame.data, dtype=np.int16).astype(float)
                    if np.sqrt(np.mean(samples**2)) < 50:
                        continue
                    frequencies = np.fft.rfftfreq(len(samples), 1 / event.frame.sample_rate)
                    heard.append(
                        round(float(frequencies[np.argmax(abs(np.fft.rfft(samples)))]) / 100) * 100
                    )
                    if heard[-1] == 1000:
                        break
            assert heard[0] == 1600, f"the prefix was lost: {heard}"
            assert len(speeches) == 1
            await speeches[0].wait_for_playout()
            assert speeches[0].exception() is None
        finally:
            if stream is not None:
                await stream.aclose()
            await session.aclose()


async def test_the_persons_camera_reaches_the_sessions_video_input(sfu: str) -> None:
    """With `video_input`, the frames a person's camera publishes reach `session.input.video`:
    the core opens the stream off the event loop, and the SDK stream must still bind the loop."""
    name = "regression-" + uuid.uuid4().hex
    async with connected(sfu, name, "person") as person, connected(sfu, name, "agent") as room:
        camera = rtc.VideoSource(16, 16)
        await person.local_participant.publish_track(
            rtc.LocalVideoTrack.create_video_track("camera", camera),
            rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA),
        )

        async def film() -> None:
            frame = rtc.VideoFrame(16, 16, rtc.VideoBufferType.RGBA, bytes(16 * 16 * 4))
            while True:
                camera.capture_frame(frame)
                await asyncio.sleep(0.05)  # 20 fps

        filming = asyncio.create_task(film())
        session = VoiceSession(vad=None, aec_warmup_duration=None)
        try:
            await session.start(
                Agent(instructions="Watch."),
                room=room,
                room_options=RoomOptions(
                    video_input=True, audio_output=False, transcription_output=False
                ),
            )
            video = session.input.video
            assert isinstance(video, _core.RoomVideoInput)

            frame = await video.next_frame()
        finally:
            await session.aclose()
            filming.cancel()
            await asyncio.gather(filming, return_exceptions=True)
            await camera.aclose()

    assert frame is not None
    assert (frame.width, frame.height) == (16, 16)
