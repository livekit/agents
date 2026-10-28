"""Production session event snapshots respect consent, bounded retention and expiry."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from livekit.agents import VoiceSession
from livekit.agents.telemetry import capture_content_enabled, set_capture_content
from livekit.agents.voice import agent_session
from livekit.agents.voice.events import AgentStateChangedEvent, UserInputTranscribedEvent

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]


@pytest.fixture(autouse=True)
def restore_capture_policy() -> Iterator[None]:
    previous = capture_content_enabled()
    try:
        yield
    finally:
        set_capture_content(previous)


def test_session_report_events_are_bounded_snapshots_without_implicitly_captured_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(agent_session.time, "monotonic_ns", lambda: 1)
    set_capture_content(False)
    session = VoiceSession(turn_handling={"turn_detection": None})
    delivered = []
    session.on("user_input_transcribed", delivered.append)
    content = UserInputTranscribedEvent(transcript="private", is_final=True)
    session.emit("user_input_transcribed", content)
    assert delivered == [content], "retention policy must not suppress application delivery"
    assert session._recorded_events == []
    state = AgentStateChangedEvent(old_state="listening", new_state="thinking")
    for _ in range(1100):
        session.emit("agent_state_changed", state)
    state.new_state = "speaking"
    assert len(session._recorded_events) == 1024
    assert all(
        isinstance(event, AgentStateChangedEvent) and event.new_state == "thinking"
        for event in session._recorded_events
    )
    monkeypatch.setattr(agent_session.time, "monotonic_ns", lambda: 3600_000_000_001)
    assert session._recorded_events == []


def test_session_event_consent_is_captured_and_oversized_events_do_not_enter_reports() -> None:
    set_capture_content(True)
    session = VoiceSession(turn_handling={"turn_detection": None})
    set_capture_content(False)
    small = UserInputTranscribedEvent(transcript="captured with consent", is_final=True)
    session.emit("user_input_transcribed", small)
    session.emit(
        "user_input_transcribed",
        UserInputTranscribedEvent(transcript="large" * 100_000, is_final=True),
    )
    hidden_payload = UserInputTranscribedEvent(transcript="small visible fields", is_final=True)
    hidden_payload.__pydantic_private__ = {"cache": "large" * 100_000}
    session.emit("user_input_transcribed", hidden_payload)
    assert session._recorded_events == [small]
    session._recording_options["redaction"] = True
    session.emit("user_input_transcribed", small)
    assert session._recorded_events == []


async def test_session_report_access_stays_with_the_job_that_started_the_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace
    from typing import cast

    from livekit.agents import Agent, _core
    from livekit.agents.job import JobContext
    from livekit.protocol import agent as job_protocol

    from .fake_llm import FakeLLM

    set_capture_content(True)
    owner = SimpleNamespace(job=job_protocol.Job(id="owner"), room=object())
    other = SimpleNamespace(job=job_protocol.Job(id="other"), room=object())
    monkeypatch.setattr(agent_session, "get_job_context", lambda **_: owner)
    session = VoiceSession(llm=FakeLLM(), turn_handling={"turn_detection": None})
    try:
        await session.start(Agent(instructions="desk"), room=_core.OfflineRoom("retention"))
        session.emit(
            "user_input_transcribed",
            UserInputTranscribedEvent(transcript="retained within budget", is_final=True),
        )
        report = JobContext.make_session_report(cast(JobContext, owner), session)
        assert report.job_id == "owner"
        [reported] = [
            event for event in report.events if isinstance(event, UserInputTranscribedEvent)
        ]
        reported.transcript = "large" * 100_000
        [retained] = [
            event
            for event in JobContext.make_session_report(cast(JobContext, owner), session).events
            if isinstance(event, UserInputTranscribedEvent)
        ]
        assert retained.transcript == "retained within budget"
        monkeypatch.setattr(agent_session, "get_job_context", lambda **_: other)
        await session.start(Agent(instructions="unused"))
        with pytest.raises(PermissionError, match="owning job"):
            JobContext.make_session_report(cast(JobContext, other), session)
    finally:
        await session.aclose()


async def test_session_end_prepares_reports_metrics_and_console_files_off_the_event_loop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import asyncio
    import threading
    from types import SimpleNamespace

    from livekit.agents import Agent, job as job_module
    from livekit.agents.job import JobContext
    from livekit.agents.voice.remote_session import ConsoleCall
    from livekit.protocol import agent as job_protocol

    from .fake_llm import FakeLLM

    session = VoiceSession(llm=FakeLLM(), turn_handling={"turn_detection": None})
    await session.start(Agent(instructions="desk"))
    for _ in range(1100):
        session.emit(
            "agent_state_changed",
            AgentStateChangedEvent(old_state="listening", new_state="thinking"),
        )
    owner = object.__new__(JobContext)
    owner._info = SimpleNamespace(job=job_protocol.Job(id="report-worker"))
    owner._run = SimpleNamespace(primary_session=session, observability_url=None)
    owner._tagger = SimpleNamespace(has_evals=lambda: False)
    entered, release = threading.Event(), threading.Event()
    loop_thread = threading.get_ident()
    threads: list[int] = []
    make_report = JobContext.make_session_report
    flush_metrics = job_module.flush_turn_metrics
    write_text = Path.write_text

    def prepare(self, current=None):
        threads.append(threading.get_ident())
        entered.set()
        release.wait(2)
        return make_report(self, current)

    def flush(history):
        threads.append(threading.get_ident())
        flush_metrics(history)

    def write_report(
        path: Path,
        data: str,
        encoding=None,
        errors=None,
        newline=None,
    ) -> int:
        threads.append(threading.get_ident())
        return write_text(path, data, encoding=encoding, errors=errors, newline=newline)

    monkeypatch.setattr(Path, "write_text", write_report)
    monkeypatch.setattr(JobContext, "make_session_report", prepare)
    monkeypatch.setattr(job_module, "flush_turn_metrics", flush)
    monkeypatch.setattr(ConsoleCall, "record", True)
    monkeypatch.chdir(tmp_path)
    cleanup = asyncio.create_task(owner._on_session_end())
    try:
        async with asyncio.timeout(3):
            while not entered.is_set() and not cleanup.done():
                await asyncio.sleep(0)
        assert entered.is_set()
        assert all(thread != loop_thread for thread in threads)
        for _ in range(3):
            await asyncio.sleep(0)
        assert not cleanup.done(), "the loop must progress while report preparation is blocked"
    finally:
        release.set()
        async with asyncio.timeout(3):
            await cleanup
        await session.aclose()
    assert len(threads) == 3
    assert all(thread != loop_thread for thread in threads)
    [artifact] = list(tmp_path.glob("console-recordings/*/session_report.json"))
    assert '"job_id": "report-worker"' in artifact.read_text()


async def test_bounded_host_callbacks_keep_late_ownership_and_propagate_the_original_error() -> (
    None
):
    import asyncio
    import contextvars
    import threading

    from livekit.agents.telemetry._export import call_bounded, worker

    entered, release = threading.Event(), threading.Event()
    identity = contextvars.ContextVar("bounded-callback-job", default="outside")
    identity.set("captured")
    key = f"late-report-test:{id(release):x}"

    def stalled():
        assert identity.get() == "captured"
        entered.set()
        release.wait(3)
        return "late result"

    pending = asyncio.create_task(call_bounded(key, stalled, 1, 0.05))
    try:
        async with asyncio.timeout(3):
            while not entered.is_set():
                await asyncio.sleep(0)
        with pytest.raises(TimeoutError, match="deadline"):
            await pending
        with pytest.raises(RuntimeError, match="admission"):
            await call_bounded(key, lambda: "replacement", 1, 1)
    finally:
        release.set()
        await worker().wait_async(key, 3)
        if not pending.done():
            pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)

    failure = ValueError("callback error")

    def fail():
        raise failure

    with pytest.raises(ValueError) as caught:
        await call_bounded(f"failed-report-test:{id(failure):x}", fail, 1, 1)
    assert caught.value is failure
