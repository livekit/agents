"""Session report events respect content consent and stay with the job that owns them."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from livekit import rtc
from livekit.agents import VoiceSession, _core, job as job_module
from livekit.agents.job import (
    JobAcceptArguments,
    JobContext,
    JobExecutorType,
    JobProcess,
    RunningJobInfo,
)
from livekit.agents.telemetry import set_capture_content
from livekit.agents.voice.events import (
    AgentStateChangedEvent,
    CloseEvent,
    CloseReason,
    UserInputTranscribedEvent,
    event_dict,
)
from livekit.protocol import agent as job_protocol

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]


def _job_context(job_id: str, agent_name: str = "", enable_recording: bool = False) -> JobContext:
    return JobContext(
        proc=JobProcess(executor_type=JobExecutorType.THREAD, user_arguments=None, http_proxy=None),
        info=RunningJobInfo(
            JobAcceptArguments("", "", ""),
            job_protocol.Job(id=job_id, agent_name=agent_name, enable_recording=enable_recording),
            "",
            "",
            "",
            True,
        ),
        room=rtc.Room(),
        run=_core.JobRun(fake_job=True, session_end_timeout=5, redaction=False),
        server_env={},
    )


@pytest.fixture(autouse=True)
def restore_capture_policy() -> Iterator[None]:
    try:
        yield
    finally:
        set_capture_content(True)


def test_session_report_events_keep_every_snapshot_whatever_message_capture() -> None:
    set_capture_content(False)
    session = VoiceSession(turn_handling={"turn_detection": None})
    delivered = []
    session.on("user_input_transcribed", delivered.append)
    content = UserInputTranscribedEvent(transcript="kept", is_final=True)
    session.emit("user_input_transcribed", content)
    assert delivered == [content]
    # GenAI message capture gates span content only, never the session report
    assert session._recorded_events == [event_dict(content, json=True)]
    state = AgentStateChangedEvent(old_state="listening", new_state="thinking")
    for _ in range(1100):
        session.emit("agent_state_changed", state)
    # every event is kept: a long session only logs a warning past a count
    states = session._recorded_events[1:]
    assert len(states) == 1100
    assert all(
        event["type"] == "agent_state_changed" and event["new_state"] == "thinking"
        for event in states
    )


# A redacted session keeping only its errors and close is the owner's (owner_tests.rs).
def test_large_events_enter_reports() -> None:
    session = VoiceSession(turn_handling={"turn_detection": None})
    small = UserInputTranscribedEvent(transcript="captured with consent", is_final=True)
    session.emit("user_input_transcribed", small)
    large = UserInputTranscribedEvent(transcript="large" * 100_000, is_final=True)
    session.emit("user_input_transcribed", large)
    assert session._recorded_events == [event_dict(small, json=True), event_dict(large, json=True)]


async def test_session_report_access_stays_with_the_job_that_started_the_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from livekit.agents import Agent
    from livekit.agents.voice import _session_bindings as _session_host

    from .fake_llm import FakeLLM

    set_capture_content(True)
    owner, other = _job_context("owner"), _job_context("other")
    monkeypatch.setattr(_session_host, "get_job_context", lambda **_: owner)
    monkeypatch.setattr(job_module, "get_job_context", lambda **_: owner)
    session = VoiceSession(llm=FakeLLM(), turn_handling={"turn_detection": None})
    try:
        await session.start(Agent(instructions="desk"), room=None)
        session.emit(
            "user_input_transcribed",
            UserInputTranscribedEvent(transcript="retained", is_final=True),
        )
        report = owner.make_session_report(session)
        assert report.job_id == "owner"
        [retained] = [e for e in report.events if e["type"] == "user_input_transcribed"]
        assert retained["transcript"] == "retained"
        monkeypatch.setattr(_session_host, "get_job_context", lambda **_: other)
        monkeypatch.setattr(job_module, "get_job_context", lambda **_: other)
        await session.start(Agent(instructions="unused"))
        with pytest.raises(PermissionError, match="owning job"):
            other.make_session_report(session)
    finally:
        await session.aclose()


async def test_the_session_snapshot_is_taken_once_off_the_event_loop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import asyncio
    import json
    import threading

    from livekit.agents import Agent
    from livekit.agents.voice._session_bindings import ConsoleCall

    from .fake_llm import FakeLLM

    session = VoiceSession(llm=FakeLLM(), turn_handling={"turn_detection": None})
    await session.start(Agent(instructions="desk"))
    # committed by the core, whose snapshot the report takes its history from
    await session.generate_reply(user_input="recorded message")
    owner = _job_context("report-worker", agent_name="desk")
    owner._core_claim_primary(session)
    entered, release = threading.Event(), threading.Event()
    loop_thread = threading.get_ident()
    threads: list[int] = []
    make_report = JobContext._build_session_report

    def prepare(self, current, core, events):
        threads.append(threading.get_ident())
        entered.set()
        release.wait(2)
        return make_report(self, current, core, events)

    monkeypatch.setattr(JobContext, "_build_session_report", prepare)
    monkeypatch.setattr(ConsoleCall, "record", True)
    monkeypatch.chdir(tmp_path)
    snapshot = asyncio.create_task(owner._snapshot_session())
    try:
        async with asyncio.timeout(3):
            while not entered.is_set() and not snapshot.done():
                await asyncio.sleep(0)
        assert entered.is_set()
        for _ in range(3):
            await asyncio.sleep(0)
        assert not snapshot.done(), "the loop must progress while the snapshot is taken"
    finally:
        release.set()
        async with asyncio.timeout(3):
            await snapshot
        await session.aclose()
    assert threads and all(thread != loop_thread for thread in threads)

    # Every later report reads the snapshot; the tagger is read at each call.
    session.history.add_message(role="user", content="after the session closed")
    owner.tagger.success(reason="booked")
    report = owner.make_session_report()
    texts = [item.text_content for item in report.chat_history if item.type == "message"]
    assert "recorded message" in texts and "after the session closed" not in texts
    assert (report.job_id, report.agent_name) == ("report-worker", "desk")
    assert [tag.name for tag in report.tags] == ["lk.success"]
    assert report.outcome is not None and report.outcome.reason == "booked"
    await owner._snapshot_session()
    assert len(threads) == 1, "the session is snapshotted once"

    [artifact] = list(tmp_path.glob("console-recordings/*/session_report.json"))
    console = json.loads(artifact.read_text())
    assert console["job_id"] == "report-worker"
    items = console["chat_history"]["items"]
    assert ["recorded message"] in [i["content"] for i in items if i["type"] == "message"]


# Catches: a second session of the job taking the job's recording, an audio recording opened
# outside the job's session directory, or a session of the job left open at its shutdown.
async def test_the_jobs_first_session_records_what_the_job_records_and_every_one_closes_with_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from livekit.agents import Agent
    from livekit.agents.job import _JobContextVar

    from .fake_llm import FakeLLM

    ctx = _job_context("recorded", enable_recording=True)

    async def connect(self: JobContext) -> None:
        pass

    closing: list[VoiceSession] = []
    on_shutdown = JobContext._core_on_shutdown

    def register(self: JobContext, session: VoiceSession) -> None:
        closing.append(session)
        on_shutdown(self, session)

    monkeypatch.setattr(JobContext, "_core_connect", connect)
    monkeypatch.setattr(JobContext, "_core_on_shutdown", register)
    token = _JobContextVar.set(ctx)
    first = VoiceSession(llm=FakeLLM(), turn_handling={"turn_detection": None})
    second = VoiceSession(llm=FakeLLM(), turn_handling={"turn_detection": None})
    try:
        await first.start(Agent(instructions="desk"))
        await second.start(Agent(instructions="back office"), room=ctx.room)
        assert ctx._sessions == [first, second] and closing == [first, second]
        assert first._owner.recording_options.audio and first._owner.started_at is not None
        assert first._owner.recording_path == str(ctx.session_directory / "audio.ogg")
        assert not second._owner.recording_options.audio
        assert second._owner.recording_path is None
        assert ctx.make_session_report().audio_recording_path == first._owner.recording_path
    finally:
        _JobContextVar.reset(token)
        await first.aclose()
        await second.aclose()


# Catches: the job's shutdown closing its session through a Python wrapper of its own, or with a
# reason other than the job's shutdown.
async def test_the_jobs_shutdown_closes_the_session_natively_in_the_shutdowns_span() -> None:
    from types import SimpleNamespace

    registered: list[tuple[object, bool, bool]] = []

    def add(callback: object, *, takes_reason: bool, framework: bool, name: str) -> None:
        registered.append((callback, takes_reason, framework))

    from livekit.agents import Agent

    from .fake_llm import FakeLLM

    session = VoiceSession(llm=FakeLLM(), turn_handling={"turn_detection": None})
    closed: list[CloseEvent] = []
    session.on("close", closed.append)
    await session.start(Agent(instructions="desk"))
    job = SimpleNamespace(_run=SimpleNamespace(add_shutdown_callback=add))
    JobContext._core_on_shutdown(job, session)  # type: ignore[arg-type]
    [(callback, takes_reason, framework)] = registered
    assert (takes_reason, framework) == (False, True)
    await callback()  # type: ignore[operator]
    assert [event.reason for event in closed] == [CloseReason.JOB_SHUTDOWN]
