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
    assert session._recorded_events == [small]
    session._recording_options["redaction"] = True
    session.emit("user_input_transcribed", small)
    assert session._recorded_events == []


async def test_session_report_access_stays_with_the_job_that_started_the_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace
    from typing import cast

    from livekit.agents import Agent
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
        await session.start(Agent(instructions="desk"), room=None)
        session.emit(
            "user_input_transcribed",
            UserInputTranscribedEvent(transcript="retained within budget", is_final=True),
        )
        report = JobContext.make_session_report(cast(JobContext, owner), session)
        assert report.job_id == "owner"
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
    owner._run = SimpleNamespace(observability_url=None, redaction=False)
    owner._primary_session = session
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

    from livekit.agents import _core
    from livekit.agents.job import JobContext
    from livekit.agents.telemetry import traces
    from livekit.agents.types import ATTRIBUTE_REDACTION_ENABLED
    from livekit.agents.voice.events import UserStateChangedEvent, UserTranscriptionTimeoutEvent
    from livekit.agents.voice.remote_session import ConsoleCall
    from livekit.protocol import agent as job_protocol
    from scripts.bench.reports import _CountingExporter, capture_reports

    set_capture_content(creation_consent)
    session = VoiceSession(turn_handling={"turn_detection": None})
    session.history.add_message(role="user", content="private console conversation")
    session._recording_path = tmp_path / "private-customer-audio.ogg"
    session.emit(
        "agent_state_changed",
        AgentStateChangedEvent(old_state="listening", new_state="thinking"),
    )
    session.emit(
        "user_state_changed",
        UserStateChangedEvent(old_state="listening", new_state="speaking"),
    )
    session.emit(
        "user_transcription_timeout",
        UserTranscriptionTimeoutEvent(speech_duration=0.5, vad_speech_started_at=1.0),
    )
    session._recording_options["logs"] = True
    session._recording_options["transcript"] = requested_formats
    session._recording_options["audio"] = requested_formats
    set_capture_content(later_consent)
    owner = object.__new__(JobContext)
    owner._info = SimpleNamespace(job=job_protocol.Job(id="console-policy"))
    owner._run = SimpleNamespace(observability_url=None, redaction=project_redaction)
    owner._primary_session = session
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
    assert [event["type"] for event in snapshot["events"][1:]] == [
        "user_state_changed",
        "user_transcription_timeout",
    ]
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

    from livekit.agents.job import JobContext, SessionReport
    from livekit.protocol import agent as job_protocol
    from scripts.bench.reports import capture_reports, prepare

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


async def test_explicit_report_options_do_not_implicitly_enable_recording() -> None:
    from livekit.agents import llm
    from livekit.agents.job import SessionReport
    from scripts.bench.reports import capture_reports, prepare

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
    report.chat_history.add_message(role="user", content="enabled after construction")
    report.options["recording_options"] = {"transcript": True}
    with capture_reports() as capture:
        await prepare(report)
    assert capture.items["report-job"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("overflow", ["items", "bytes"])
@pytest.mark.parametrize("destination", ["cloud", "console"])
async def test_event_overflow_preserves_the_complete_transcript(
    overflow: str, destination: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import json
    from types import SimpleNamespace

    from livekit.agents import _core, llm
    from livekit.agents.job import JobContext
    from livekit.agents.voice.remote_session import ConsoleCall
    from livekit.protocol import agent as job_protocol
    from scripts.bench.reports import _CountingExporter, capture_reports, make_report, prepare

    previous = capture_content_enabled()
    set_capture_content(True)
    try:
        items = [llm.ChatMessage(role="user", content=["hello"], id=str(i)) for i in range(1000)]
        report = make_report(items)
        if overflow == "items":
            report.events = [
                AgentStateChangedEvent(old_state="listening", new_state="thinking")
                for _ in range(1100)
            ]
        else:
            report.events = [
                UserInputTranscribedEvent(transcript="x" * 3000, is_final=True) for _ in range(120)
            ]
        if destination == "cloud":
            exported_events = 0
            exported_bytes = 2
            omitted = 0
            original = _CountingExporter.export

            def collect(exporter, batch):
                nonlocal exported_events, exported_bytes, omitted
                for record in batch:
                    if record.log_record.body == "session event":
                        exported_events += 1
                        attributes = record.log_record.attributes
                        payload = (
                            attributes.get("session.event") or attributes["lk.pii.session_event"]
                        )
                        exported_bytes += len(payload.encode("utf-8")) + 1
                    if record.log_record.body == "session report":
                        omitted += int(
                            record.log_record.attributes.get(
                                "lk.telemetry.omitted_report_entries", 0
                            )
                        )
                return original(exporter, batch)

            with capture_reports() as capture, monkeypatch.context() as patches:
                patches.setattr(_CountingExporter, "export", collect)
                await prepare(report)
            assert capture.items["report-job"] == len(items)
            assert len(json.loads(capture.uploads["report-job"])["items"]) == len(items)
            assert omitted > 0
            assert 0 < exported_events <= 1024
            assert exported_events + omitted == len(report.events)
            assert exported_bytes <= 1024 * 1024
        else:
            session = VoiceSession(turn_handling={"turn_detection": None})
            owner = object.__new__(JobContext)
            owner._info = SimpleNamespace(job=job_protocol.Job(id="report-job"))
            owner._run = SimpleNamespace(observability_url=None, redaction=False)
            owner._primary_session = session
            owner._tagger = _core.Tagger()
            monkeypatch.setattr(JobContext, "_build_session_report", lambda *args: report)
            monkeypatch.setattr(ConsoleCall, "record", True)
            monkeypatch.chdir(tmp_path)
            await owner._on_session_end()
            [artifact] = list(tmp_path.glob("console-recordings/*/session_report.json"))
            output = json.loads(artifact.read_text())
            assert len(output["chat_history"]["items"]) == len(items)
            assert output["omitted_events"] > 0
            assert 0 < len(output["events"]) <= 1024
            assert len(output["events"]) + output["omitted_events"] == len(report.events)
            assert (
                len(
                    json.dumps(output["events"], ensure_ascii=False, separators=(",", ":")).encode(
                        "utf-8"
                    )
                )
                <= 1024 * 1024
            )
    finally:
        set_capture_content(previous)


async def test_report_usage_reaches_the_sdk_without_a_display_shape_roundtrip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from livekit.agents import LLMUsage, TokenUsage, TTSUsage, Usage
    from scripts.bench.reports import _CountingExporter, capture_reports, make_report, prepare

    report = make_report([])
    report.usage = Usage(
        models=(
            LLMUsage(provider="provider", model="llm", tokens=TokenUsage(input_tokens=7)),
            TTSUsage(provider="provider", model="tts", characters_count=11),
            # JSON escaping exceeds the wire bound while the admitted host representation fits.
            LLMUsage(provider="\0" * 12_000, model="oversize", tokens=TokenUsage(input_tokens=99)),
        )
    )
    summaries = []
    original = _CountingExporter.export

    def collect(exporter, batch):
        for record in batch:
            if record.log_record.body == "session report":
                summaries.append(record.log_record.attributes["usage"])
        return original(exporter, batch)

    with capture_reports(), monkeypatch.context() as patches:
        patches.setattr(_CountingExporter, "export", collect)
        await prepare(report)
    [usage] = summaries
    assert len(usage) == 2
    assert usage[0]["type"] == "llm" and usage[0]["tokens"]["input_tokens"] == 7
    assert usage[1]["type"] == "tts" and usage[1]["characters_count"] == 11
