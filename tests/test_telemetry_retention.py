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
    owner = object.__new__(JobContext)
    owner._info = SimpleNamespace(job=job_protocol.Job(id="owner"))
    owner._room = object()
    other = object.__new__(JobContext)
    other._info = SimpleNamespace(job=job_protocol.Job(id="other"))
    other._room = object()
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
    session.history.add_message(role="user", content="recorded message")
    for _ in range(1100):
        session.emit(
            "agent_state_changed",
            AgentStateChangedEvent(old_state="listening", new_state="thinking"),
        )
    owner = object.__new__(JobContext)
    owner._info = SimpleNamespace(job=job_protocol.Job(id="report-worker"))
    owner._run = SimpleNamespace(primary_session=session, observability_url=None, redaction=False)
    owner._tagger = SimpleNamespace(has_evals=lambda: False)
    entered, release = threading.Event(), threading.Event()
    loop_thread = threading.get_ident()
    threads: list[int] = []
    make_report = JobContext._build_session_report
    flush_metrics = job_module.flush_turn_metrics

    def prepare(self, current, history, events):
        threads.append(threading.get_ident())
        entered.set()
        release.wait(2)
        return make_report(self, current, history, events)

    def flush(history):
        threads.append(threading.get_ident())
        flush_metrics(history)

    monkeypatch.setattr(JobContext, "_build_session_report", prepare)
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
    assert len(threads) == 2
    assert all(thread != loop_thread for thread in threads)
    [artifact] = list(tmp_path.glob("console-recordings/*/session_report.json"))
    import json

    assert json.loads(artifact.read_text())["job_id"] == "report-worker"


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


@pytest.mark.parametrize(
    ("creation_consent", "later_consent", "project_redaction", "content_present"),
    [(False, True, False, False), (True, False, False, True), (True, True, True, False)],
)
@pytest.mark.parametrize("requested_formats", [False, True])
async def test_console_report_cannot_bypass_creation_consent_or_project_redaction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    creation_consent: bool,
    later_consent: bool,
    project_redaction: bool,
    content_present: bool,
    requested_formats: bool,
) -> None:
    import json
    from collections.abc import Sequence
    from types import SimpleNamespace

    from opentelemetry.sdk._logs import ReadableLogRecord
    from opentelemetry.sdk._logs.export import LogRecordExportResult
    from scripts.bench.reports import _CountingExporter, capture_reports

    from livekit.agents import _core
    from livekit.agents.job import JobContext
    from livekit.agents.metrics import AgentSessionUsage, LLMModelUsage
    from livekit.agents.telemetry import traces
    from livekit.agents.types import ATTRIBUTE_REDACTION_ENABLED
    from livekit.agents.voice.events import SessionUsageUpdatedEvent
    from livekit.agents.voice.remote_session import ConsoleCall
    from livekit.protocol import agent as job_protocol

    set_capture_content(creation_consent)
    session = VoiceSession(turn_handling={"turn_detection": None})
    session.history.add_message(role="user", content="private console conversation")
    session._recording_path = tmp_path / "private-customer-audio.ogg"
    session.emit(
        "agent_state_changed",
        AgentStateChangedEvent(old_state="listening", new_state="thinking"),
    )
    session.emit(
        "session_usage_updated",
        SessionUsageUpdatedEvent(
            usage=AgentSessionUsage(
                model_usage=[LLMModelUsage(provider="provider", model="model", input_tokens=7)]
            )
        ),
    )
    session._recording_options["logs"] = True
    session._recording_options["transcript"] = requested_formats
    session._recording_options["audio"] = requested_formats
    set_capture_content(later_consent)
    owner = object.__new__(JobContext)
    owner._info = SimpleNamespace(job=job_protocol.Job(id="console-policy"))
    owner._run = SimpleNamespace(
        primary_session=session, observability_url=None, redaction=project_redaction
    )
    owner._tagger = SimpleNamespace(has_evals=lambda: False)
    monkeypatch.setattr(ConsoleCall, "record", True)
    monkeypatch.chdir(tmp_path)

    await owner._on_session_end()

    [artifact] = list(tmp_path.glob("console-recordings/*/session_report.json"))
    snapshot = json.loads(artifact.read_text())
    assert snapshot["job_id"] == "console-policy"
    assert ("private console conversation" in artifact.read_text()) is (
        content_present and requested_formats
    )
    audio_present = requested_formats and not project_redaction
    assert ("private-customer-audio.ogg" in artifact.read_text()) is audio_present
    assert snapshot["events"][0]["type"] == "agent_state_changed"
    assert snapshot["events"][0]["new_state"] == "thinking"
    assert (snapshot["options"] is not None) is content_present
    assert (snapshot["audio_recording_path"] is not None) is audio_present
    if not (content_present and requested_formats):
        assert snapshot["chat_history"] == {"items": []}
    if not audio_present:
        assert snapshot["audio_recording_started_at"] is None
    assert snapshot["events"][1]["type"] == "session_usage_updated"
    assert snapshot["events"][1]["usage"]["model_usage"][0]["input_tokens"] == 7
    assert session.history.items[0].text_content == "private console conversation"

    records: list[ReadableLogRecord] = []
    export = _CountingExporter.export

    def collect(
        exporter: _CountingExporter, batch: Sequence[ReadableLogRecord]
    ) -> LogRecordExportResult:
        records.extend(batch)
        return export(exporter, batch)

    with capture_reports(), monkeypatch.context() as patches:
        patches.setattr(_CountingExporter, "export", collect)
        await traces._upload_session_report(
            agent_name="report-agent",
            observability_url="https://unused.invalid",
            report=owner.make_session_report(),
            tagger=_core.Tagger(),
            http_session=SimpleNamespace(),
            metadata={ATTRIBUTE_REDACTION_ENABLED: project_redaction},
        )
    cloud_events = [
        json.loads(record.log_record.attributes["session.event"])
        for record in records
        if record.log_record.body == "session event"
    ]
    assert cloud_events == snapshot["events"]
    [summary] = [record for record in records if record.log_record.body == "session report"]
    cloud_options = json.loads(json.dumps(summary.log_record.attributes["session.options"]))
    assert cloud_options == snapshot["options"]


@pytest.mark.parametrize("creation_consent", [False, True])
@pytest.mark.parametrize("source", ["session", "explicit"])
async def test_report_upload_keeps_creation_consent(creation_consent: bool, source: str) -> None:
    from types import SimpleNamespace
    from typing import cast

    from scripts.bench.reports import capture_reports, prepare

    from livekit.agents.job import JobContext, SessionReport
    from livekit.protocol import agent as job_protocol

    set_capture_content(creation_consent)
    session = VoiceSession(turn_handling={"turn_detection": None})
    session.history.add_message(role="user", content="private before consent changed")
    session._recording_options["transcript"] = True
    owner = object.__new__(JobContext)
    owner._info = SimpleNamespace(job=job_protocol.Job(id="report-job"))
    if source == "explicit":
        report = SessionReport(
            job_id="report-job",
            room_id="report-room",
            room="report-room-name",
            options={"recording_options": session._recording_options},
            events=[],
            chat_history=session.history,
        )
    set_capture_content(not creation_consent)
    if source == "session":
        report = JobContext.make_session_report(cast(JobContext, owner), session)

    with capture_reports() as capture:
        await prepare(report)

    assert bool(capture.uploads) is creation_consent
    assert bool(capture.items) is creation_consent
    if creation_consent:
        assert "private before consent changed" in capture.uploads["report-job"]


def test_safe_usage_events_reach_reports_without_conversation_capture() -> None:
    from livekit.agents.metrics import AgentSessionUsage, LLMModelUsage
    from livekit.agents.voice.events import SessionUsageUpdatedEvent

    set_capture_content(False)
    session = VoiceSession(turn_handling={"turn_detection": None})
    event = SessionUsageUpdatedEvent(
        usage=AgentSessionUsage(
            model_usage=[LLMModelUsage(provider="provider", model="model", input_tokens=7)]
        )
    )
    session.emit("session_usage_updated", event)
    oversized = event.model_copy(deep=True)
    oversized.usage.__dict__["hidden_cache"] = b"x" * (2 * 1024 * 1024)
    session.emit("session_usage_updated", oversized)
    assert session._recorded_events == [event]


async def test_explicit_report_options_do_not_implicitly_enable_recording() -> None:
    from scripts.bench.reports import capture_reports, prepare

    from livekit.agents import llm
    from livekit.agents.job import SessionReport

    set_capture_content(True)
    report = SessionReport(
        job_id="report-job",
        room_id="room",
        room="room",
        options={"custom": "value"},
        events=[],
        chat_history=llm.ChatContext.empty(),
    )
    assert report.to_dict()["options"] == {"custom": "value"}
    with capture_reports() as capture:
        await prepare(report)
    assert capture.uploads == {}
    assert capture.items == {}
