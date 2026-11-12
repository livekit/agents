"""Room text, transcription and teardown contracts on an isolated local SFU."""

from __future__ import annotations

import asyncio
import json
import multiprocessing as mp
import uuid
from collections.abc import Callable
from pathlib import Path

import pytest
from aiohttp import web

from livekit import rtc
from livekit.agents import Agent, VoiceSession, _core
from livekit.agents.beta import send_dtmf_events
from livekit.agents.job import JobContext
from livekit.agents.voice.avatar import DataStreamAudioOutput
from livekit.agents.voice.room_io import RoomOptions
from livekit.protocol import agent, models
from tests.fakes import FakeAudioStage, FakeLLM, FakeTTS, tool_call, until

from .conftest import JoinAvatar
from .processes import job_runner, running_job
from .sfu import connected


@pytest.mark.parametrize("hidden", [False, True])
async def test_room_text_drives_replies_transcribed_by_each_new_session(
    sfu: str, hidden: bool
) -> None:
    """Text sent over the room drives a reply delivered as a transcription, also from a second
    session started with the same agent on a new room I/O; a visible agent signs it with its
    identity (a hidden sender is anonymous by the server's design).
    https://github.com/livekit/agents/issues/1897"""
    name = "regression-" + uuid.uuid4().hex
    async with (
        connected(sfu, name, "person") as person,
        connected(sfu, name, "agent", hidden=hidden) as publisher,
    ):
        loop = asyncio.get_running_loop()
        transcripts: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
        readers: list[asyncio.Task[None]] = []

        def transcription(reader: rtc.TextStreamReader, identity: str) -> None:
            async def read() -> None:
                transcripts.put_nowait((identity, await reader.read_all()))

            readers.append(loop.create_task(read()))

        person.register_text_stream_handler("lk.transcription", transcription)
        desk = Agent(instructions="Answer typed messages.")
        try:
            for turn in ("first", "second"):
                session = VoiceSession(
                    llm=FakeLLM({f"{turn} message": f"{turn} reply"}),
                    vad=None,
                    turn_handling={"turn_detection": None},
                )
                try:
                    await session.start(
                        desk,
                        room=publisher,
                        room_options=RoomOptions(audio_input=False, audio_output=False),
                    )
                    await person.local_participant.send_text(f"{turn} message", topic="lk.chat")
                    async with asyncio.timeout(10):
                        while (delivered := await transcripts.get())[1] != f"{turn} reply":
                            pass
                    assert delivered[0] == ("" if hidden else "agent")
                    said = [
                        (item.role, item.text_content)
                        for item in session.history.items
                        if item.type == "message"
                    ]
                    assert said == [("user", f"{turn} message"), ("assistant", f"{turn} reply")]
                finally:
                    await session.aclose()
        finally:
            for task in readers:
                task.cancel()
            await asyncio.gather(*readers, return_exceptions=True)


async def test_the_dtmf_tool_presses_keys_in_the_room_its_session_runs_in(sfu: str) -> None:
    """`send_dtmf_events` publishes on the executing session's room, not the job's.

    https://github.com/livekit/agents/issues/6357
    """
    name = "regression-" + uuid.uuid4().hex
    pressed: list[str] = []
    async with (
        connected(sfu, name, "person") as person,
        connected(sfu, name, "agent") as publisher,
    ):
        person.on("sip_dtmf_received", lambda dtmf: pressed.append(dtmf.digit))
        session = VoiceSession(
            llm=FakeLLM([tool_call("send_dtmf_events", events=["1", "2"]), "Dialed."]),
            vad=None,
            turn_handling={"turn_detection": None},
        )
        async with session:
            await session.start(
                Agent(instructions="Dial.", tools=[send_dtmf_events]),
                room=publisher,
                room_options=RoomOptions(audio_input=False, audio_output=False),
            )
            await session.generate_reply(user_input="dial one two")

        await until(lambda: pressed == ["1", "2"], "the person heard no keypad tones")


async def _hung_deletion(ctx: JobContext) -> None:
    marker = Path(ctx.job.metadata)

    def settled(deletion: asyncio.Future[object]) -> None:
        if deletion.cancelled():
            outcome = "cancelled"
        elif (error := deletion.exception()) is not None:
            outcome = type(error).__name__
        else:
            outcome = "answered"
        marker.write_text(json.dumps(outcome))

    async def closed(reason: str) -> None:
        marker.with_suffix(".closed").write_text(json.dumps(reason))

    ctx.add_shutdown_callback(closed)
    ctx.delete_room().add_done_callback(settled)
    async with asyncio.timeout(10):
        while not marker.with_suffix(".requested").exists():
            await asyncio.sleep(0.01)
    ctx.shutdown("leaving with the deletion unanswered")


async def test_an_unanswered_room_deletion_does_not_hold_the_job_open(
    own_executor: Callable[[_core.JobSupervisor], _core.JobSupervisor],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A room deletion still unanswered when the entrypoint returns is cancelled at shutdown;
    the job exits cleanly and promptly. https://github.com/livekit/agents/pull/4638"""
    monkeypatch.setenv("LIVEKIT_API_KEY", "APIregression")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "regression-secret-regression-secret")
    marker = tmp_path / "deletion"
    hold = asyncio.Event()

    async def delete_room(request: web.Request) -> web.Response:
        marker.with_suffix(".requested").touch()
        await hold.wait()
        return web.Response(status=500)

    app = web.Application()
    app.router.add_post("/twirp/livekit.RoomService/DeleteRoom", delete_room)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]
    monkeypatch.setenv("LIVEKIT_URL", f"ws://127.0.0.1:{port}")
    try:
        process = own_executor(
            job_runner(_hung_deletion, mp_ctx=mp.get_context("spawn"), close_timeout=10)
        )
        await process.start()
        await process.initialize()
        info = running_job(
            agent.Job(id="AJ_hung_delete", room=models.Room(name="hung"), metadata=str(marker)),
            fake_job=False,
        )
        info.url = f"ws://127.0.0.1:{port}"
        await process.launch_job(info)
        await process.join()
        assert not process.killed and process.exitcode == 0
        assert json.loads(marker.with_suffix(".closed").read_text()) == (
            "leaving with the deletion unanswered"
        )
        assert json.loads(marker.read_text()) in ("cancelled", "CancelledError")
    finally:
        hold.set()
        await runner.cleanup()


async def test_a_speech_through_a_python_stage_ends_when_the_avatar_reports_it_played(
    sfu: str, join_avatar: JoinAvatar
) -> None:
    """A user's stage in front of the avatar's data-stream output streams the speech to the
    avatar worker; the speech ends only once the worker reports the segment played, and the
    stage hears that report."""
    name = "regression-" + uuid.uuid4().hex
    async with connected(sfu, name, "agent") as room, join_avatar(sfu, name) as avatar:
        output = DataStreamAudioOutput(room, destination_identity="avatar")
        stage = FakeAudioStage(output)
        session = VoiceSession(vad=None, tts=FakeTTS(seconds=0.3))
        session.output.audio = stage
        try:
            async with session:
                await session.start(Agent(instructions="Help."))
                speech = session.say("Hello.")
                segment, audio = await avatar.segments.get()
                ended_before_the_report = speech.done()
                await avatar.played(segment, 0.3)
                await speech
        finally:
            await output.aclose()

    assert audio and not ended_before_the_report
    assert not speech.interrupted
    assert [(f.playback_position, f.interrupted) for f in stage.finished] == [(0.3, False)]


async def test_interrupting_a_speech_clears_the_avatars_buffer(
    sfu: str, join_avatar: JoinAvatar
) -> None:
    """An interrupted speech asks the avatar worker to clear its buffer and ends interrupted at
    the position the worker reports."""
    name = "regression-" + uuid.uuid4().hex
    async with connected(sfu, name, "agent") as room, join_avatar(sfu, name) as avatar:
        output = DataStreamAudioOutput(room, destination_identity="avatar")
        stage = FakeAudioStage(output)
        session = VoiceSession(vad=None, tts=FakeTTS(seconds=0.3))
        session.output.audio = stage
        try:
            async with session:
                await session.start(Agent(instructions="Help."))
                speech = session.say("A much longer reply the caller talks over.")
                segment, _ = await avatar.segments.get()
                speech.interrupt()
                await until(lambda: avatar.clears == 1, "the avatar was not asked to clear")
                await avatar.played(segment, 0.05, interrupted=True)
                await speech
        finally:
            await output.aclose()

    assert speech.interrupted
    assert [(f.playback_position, f.interrupted) for f in stage.finished] == [(0.05, True)]


async def test_an_interrupt_before_the_avatar_joins_leaves_the_output_to_the_next_speech(
    sfu: str, join_avatar: JoinAvatar
) -> None:
    """A speech interrupted while its output waits for the avatar to join sends the avatar
    nothing; once the avatar joins, the next speech plays through to its report."""
    name = "regression-" + uuid.uuid4().hex
    async with connected(sfu, name, "agent") as room:
        output = DataStreamAudioOutput(room, destination_identity="avatar")
        stage = FakeAudioStage(output)
        session = VoiceSession(vad=None, tts=FakeTTS(seconds=0.3))
        session.output.audio = stage
        try:
            async with session:
                await session.start(Agent(instructions="Help."))
                waiting = session.say("Are you there?")
                await until(lambda: stage.frames > 0, "the speech never reached the stage")
                waiting.interrupt()
                await waiting
                async with join_avatar(sfu, name) as avatar:
                    speech = session.say("Hello.")
                    segment, _ = await avatar.segments.get()
                    await avatar.played(segment, 0.3)
                    await speech
                    stale = avatar.segments.qsize()
        finally:
            await output.aclose()

    assert waiting.interrupted
    assert not speech.interrupted and stale == 0
