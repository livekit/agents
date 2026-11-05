from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from livekit.agents import _hosted_logs
from livekit.agents._hosted_logs import (
    HOSTED_LOGS_ENABLED,
    HOSTED_LOGS_ENV,
    HOSTED_LOGS_HANDOFF_FILE_ENV,
    HOSTED_LOGS_JOBS_DIR_ENV,
    hand_off,
    logs_handed_off,
)
from livekit.agents.job import JobContext
from livekit.agents.version import __version__

pytestmark = pytest.mark.unit


@pytest.fixture
def hosted_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv(HOSTED_LOGS_ENV, HOSTED_LOGS_ENABLED)
    monkeypatch.setenv(HOSTED_LOGS_HANDOFF_FILE_ENV, str(tmp_path / "sdk-handoff"))
    monkeypatch.setenv(HOSTED_LOGS_JOBS_DIR_ENV, str(tmp_path / "jobs"))
    monkeypatch.setattr(_hosted_logs, "_handed_off", False)
    return tmp_path


def _jobs(tmp_path: Path) -> list[str]:
    return sorted(p.name for p in (tmp_path / "jobs").iterdir())


def test_hand_off_writes_the_sdk_and_version(hosted_env: Path) -> None:
    assert not logs_handed_off()
    hand_off()
    assert logs_handed_off()
    assert json.loads((hosted_env / "sdk-handoff").read_text()) == {
        "sdk": "python",
        "version": __version__,
    }
    assert sorted(p.name for p in hosted_env.iterdir()) == ["jobs", "sdk-handoff"]


def test_no_hand_off_when_hosted_logs_are_off(
    hosted_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(HOSTED_LOGS_ENV, "something-else")
    hand_off()
    assert not (hosted_env / "sdk-handoff").exists()
    assert not logs_handed_off()


def test_no_hand_off_without_a_jobs_dir(hosted_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(HOSTED_LOGS_JOBS_DIR_ENV)
    hand_off()
    assert not (hosted_env / "sdk-handoff").exists()
    assert not logs_handed_off()


def test_sdk_keeps_exporting_when_the_file_cannot_be_written(
    hosted_env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(HOSTED_LOGS_HANDOFF_FILE_ENV, str(hosted_env / "missing" / "sdk-handoff"))
    hand_off()
    assert not logs_handed_off()
    assert capsys.readouterr() == ("", "")


def test_importing_livekit_agents_writes_the_file(hosted_env: Path) -> None:
    subprocess.run([sys.executable, "-c", "import livekit.agents"], check=True)
    assert json.loads((hosted_env / "sdk-handoff").read_text())["sdk"] == "python"


def test_job_process_output_goes_to_its_own_file(hosted_env: Path) -> None:
    script = textwrap.dedent(
        """
        import os, sys
        from livekit.agents import _hosted_logs
        print("prewarm", flush=True)
        out = _hosted_logs.redirect_job_output("AJ_1", {})
        print("before session", flush=True)
        os.write(2, b"native\\n")
        out.decide(logs=True, redaction=False)
        print("after session")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], check=True, capture_output=True, text=True
    )
    assert result.stdout == "prewarm\n" and result.stderr == ""
    assert _jobs(hosted_env) == ["job-AJ_1.export.log"]
    assert (hosted_env / "jobs" / "job-AJ_1.export.log").read_text().splitlines() == [
        "before session",
        "native",
        "after session",
    ]


def test_job_process_keeps_inherited_output_without_a_hand_off(hosted_env: Path) -> None:
    assert _hosted_logs.redirect_job_output("AJ_1", {}) is None


@pytest.mark.parametrize(
    ("logs", "redaction", "name"),
    [
        (True, False, "job-AJ_1.export.log"),
        (True, True, "job-AJ_1.redact.log"),
        (False, False, "job-AJ_1.private.log"),
        (False, True, "job-AJ_1.private.log"),
    ],
)
def test_the_session_decides_the_file_name(
    hosted_env: Path, logs: bool, redaction: bool, name: str
) -> None:
    hand_off()
    path = hosted_env / "jobs" / "job-AJ_1.log"
    path.touch()
    _hosted_logs._ProcessOutput(str(path), "AJ_1", {}).decide(logs=logs, redaction=redaction)
    assert _jobs(hosted_env) == [name]


def test_the_first_decision_wins(hosted_env: Path) -> None:
    hand_off()
    path = hosted_env / "jobs" / "job-AJ_1.log"
    path.touch()
    out = _hosted_logs._ProcessOutput(str(path), "AJ_1", {})
    out.decide(logs=False, redaction=False)
    out.decide(logs=True, redaction=False)
    assert _jobs(hosted_env) == ["job-AJ_1.private.log"]


@pytest.mark.parametrize(
    ("existing", "done"),
    [
        ("job-AJ_1.log", "job-AJ_1.done.log"),
        ("job-AJ_1.export.log", "job-AJ_1.export.done.log"),
        ("job-AJ_1.redact.log", "job-AJ_1.redact.done.log"),
        ("job-AJ_1.private.log", "job-AJ_1.private.done.log"),
    ],
)
def test_mark_process_done(hosted_env: Path, existing: str, done: str) -> None:
    hand_off()
    (hosted_env / "jobs" / existing).touch()
    _hosted_logs.mark_process_done("AJ_1")
    assert _jobs(hosted_env) == [done]


def test_thread_job_output_routes_job_records(hosted_env: Path) -> None:
    hand_off()
    out = _hosted_logs.thread_job_output("AJ_1", {"job_id": "AJ_1", "room_id": "RM_1"})
    assert out is not None
    router = _hosted_logs._ThreadJobRouter()
    router.setFormatter(logging.Formatter("%(message)s %(room_id)s"))
    record = logging.LogRecord("t", logging.INFO, __file__, 1, "outside", None, None)
    router.handle(record)
    assert _hosted_logs._not_routed(record)

    token = _hosted_logs.current_job_output.set(out)
    try:
        record = logging.LogRecord("t", logging.INFO, __file__, 1, "inside", None, None)
        assert not _hosted_logs._not_routed(record)
        router.handle(record)
    finally:
        _hosted_logs.current_job_output.reset(token)

    out.decide(logs=True, redaction=False)
    out.close()
    assert _jobs(hosted_env) == ["job-AJ_1.export.done.log"]
    assert (hosted_env / "jobs" / "job-AJ_1.export.done.log").read_text() == "inside RM_1\n"


def test_thread_job_output_closes_and_warns_when_a_rename_fails(
    hosted_env: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    hand_off()
    out = _hosted_logs.thread_job_output("AJ_1", {})
    assert isinstance(out, _hosted_logs._ThreadJobOutput)

    def fail(src: str, dst: str) -> None:
        raise OSError("read-only")

    monkeypatch.setattr(_hosted_logs.os, "rename", fail)
    token = _hosted_logs.current_job_output.set(out)
    try:
        out.decide(logs=True, redaction=False)
    finally:
        _hosted_logs.current_job_output.reset(token)
    assert out._stream.closed
    out.write("dropped")
    assert "failed to rename the hosted log file" in caplog.text
    assert _jobs(hosted_env) == ["job-AJ_1.log"]


def test_install_thread_router_filters_every_root_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = logging.getLogger()
    custom = logging.StreamHandler()
    monkeypatch.setattr(root, "handlers", [custom])
    _hosted_logs.install_thread_router()
    assert _hosted_logs._not_routed in custom.filters
    assert isinstance(root.handlers[-1], _hosted_logs._ThreadJobRouter)


def test_thread_job_output_without_a_decision(hosted_env: Path) -> None:
    hand_off()
    out = _hosted_logs.thread_job_output("AJ_1", {})
    assert out is not None
    out.close()
    assert _jobs(hosted_env) == ["job-AJ_1.done.log"]


def test_init_recording_decides_the_job_output() -> None:
    ctx = MagicMock()
    ctx._recording_initialized = False
    ctx.job.enable_redaction = False
    ctx._info.url = "ws://localhost:7880"
    JobContext.init_recording(ctx, {"logs": False, "redaction": True})
    ctx._hosted_output.decide.assert_called_once_with(logs=False, redaction=True)


def test_init_recording_decides_only_once() -> None:
    ctx = MagicMock()
    ctx._recording_initialized = True
    ctx.job.enable_redaction = False
    JobContext.init_recording(ctx, {"logs": True})
    ctx._hosted_output.decide.assert_not_called()


async def test_cleanup_decides_export_when_no_session_decided() -> None:
    ctx = MagicMock()
    ctx._recording_initialized = False
    ctx._early_log_handler = None
    ctx._telemetry_state = None
    ctx._handlers_with_filter = []
    ctx.job.enable_redaction = True
    await JobContext._on_cleanup(ctx)
    ctx._hosted_output.decide.assert_called_once_with(logs=True, redaction=True)


async def test_cleanup_keeps_the_session_decision() -> None:
    ctx = MagicMock()
    ctx._recording_initialized = True
    ctx._early_log_handler = None
    ctx._telemetry_state = None
    ctx._handlers_with_filter = []
    await JobContext._on_cleanup(ctx)
    ctx._hosted_output.decide.assert_not_called()


def test_job_process_log_lines_carry_the_job_fields(hosted_env: Path) -> None:
    script = textwrap.dedent(
        """
        import logging, os
        from livekit.agents import _hosted_logs
        from livekit.agents.ipc.log_queue import LogQueueHandler
        root = logging.getLogger()
        root.setLevel(logging.INFO)
        queue = LogQueueHandler.__new__(LogQueueHandler)
        logging.Handler.__init__(queue)
        queue.close = lambda: None
        root.addHandler(queue)
        _hosted_logs.redirect_job_output("AJ_1", {"job_id": "AJ_1", "room_id": "RM_1"})
        assert queue not in root.handlers
        logging.getLogger("t").info("hello")
        print(os.getpid(), file=open(os.environ["PID_FILE"], "w"))
        """
    )
    pid_file = hosted_env / "pid"
    subprocess.run(
        [sys.executable, "-c", script], check=True, env={**os.environ, "PID_FILE": str(pid_file)}
    )
    line = json.loads((hosted_env / "jobs" / "job-AJ_1.log").read_text())
    assert (line["message"], line["job_id"], line["room_id"], line["pid"]) == (
        "hello",
        "AJ_1",
        "RM_1",
        int(pid_file.read_text()),
    )


def _record_with_pii() -> logging.LogRecord:
    try:
        raise ValueError("secret user text")
    except ValueError:
        exc_info = sys.exc_info()
    record = logging.LogRecord("t", logging.ERROR, __file__, 1, "failed", None, exc_info)
    record.__dict__["lk.pii.transcript"] = "hello my name is"
    return record


def test_job_files_redact_pii_when_redaction_is_on(
    hosted_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from livekit.agents.cli.log import JsonFormatter
    from livekit.agents.telemetry import utils as telemetry_utils

    monkeypatch.setattr(telemetry_utils, "redaction_enabled", lambda *_: True)
    hand_off()
    out = _hosted_logs.thread_job_output("AJ_1", {"job_id": "AJ_1"})
    assert out is not None
    router = _hosted_logs._ThreadJobRouter()
    router.setFormatter(JsonFormatter())
    record = _record_with_pii()
    token = _hosted_logs.current_job_output.set(out)
    try:
        router.handle(record)
    finally:
        _hosted_logs.current_job_output.reset(token)
    out.close()

    line = json.loads((hosted_env / "jobs" / "job-AJ_1.done.log").read_text())
    assert "lk.pii.transcript" not in line
    assert line["exc_info"] == "ValueError: exception details redacted"
    assert "secret user text" not in json.dumps(line)
    assert line["job_id"] == "AJ_1"
    assert record.__dict__["lk.pii.transcript"] == "hello my name is"


def test_job_files_keep_pii_when_redaction_is_off(monkeypatch: pytest.MonkeyPatch) -> None:
    from livekit.agents.telemetry import utils as telemetry_utils

    monkeypatch.setattr(telemetry_utils, "redaction_enabled", lambda *_: False)
    record = _record_with_pii()
    assert _hosted_logs._redacted(record) is record
