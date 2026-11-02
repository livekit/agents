from __future__ import annotations

import json
import logging
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
    monkeypatch.setattr(_hosted_logs, "_process_output", None)
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


def test_revoke_removes_the_file_but_the_sdk_stays_handed_off(hosted_env: Path) -> None:
    hand_off()
    _hosted_logs.revoke_hand_off()
    assert not (hosted_env / "sdk-handoff").exists()
    assert logs_handed_off()


def test_importing_livekit_agents_writes_the_file(hosted_env: Path) -> None:
    subprocess.run([sys.executable, "-c", "import livekit.agents"], check=True)
    assert json.loads((hosted_env / "sdk-handoff").read_text())["sdk"] == "python"


def test_job_process_output_goes_to_its_own_file(hosted_env: Path) -> None:
    script = textwrap.dedent(
        """
        import os, sys
        from livekit.agents import _hosted_logs
        out = _hosted_logs.redirect_process_output()
        print("prewarm")
        out.name_job("AJ_1", {})
        print("before session", flush=True)
        os.write(2, b"native\\n")
        out.decide(logs=True, redaction=False)
        print("after session")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], check=True, capture_output=True, text=True
    )
    assert result.stdout == "" and result.stderr == ""
    assert _jobs(hosted_env) == ["job-AJ_1.export.log"]
    assert (hosted_env / "jobs" / "job-AJ_1.export.log").read_text().splitlines() == [
        "prewarm",
        "before session",
        "native",
        "after session",
    ]


def test_job_process_keeps_inherited_output_without_a_hand_off(hosted_env: Path) -> None:
    assert _hosted_logs.redirect_process_output() is None


@pytest.mark.parametrize(
    ("logs", "redaction", "name"),
    [
        (True, False, "job-AJ_1.export.log"),
        (True, True, "job-AJ_1.export.redact.log"),
        (False, False, "job-AJ_1.private.log"),
        (False, True, "job-AJ_1.private.log"),
    ],
)
def test_the_session_decides_the_file_name(
    hosted_env: Path, logs: bool, redaction: bool, name: str
) -> None:
    hand_off()
    path = hosted_env / "jobs" / "proc-1.log"
    path.touch()
    out = _hosted_logs._ProcessOutput(str(path))
    out.name_job("AJ_1", {})
    assert _jobs(hosted_env) == ["job-AJ_1.log"]
    out.decide(logs=logs, redaction=redaction)
    assert _jobs(hosted_env) == [name]


def test_the_first_decision_wins(hosted_env: Path) -> None:
    hand_off()
    path = hosted_env / "jobs" / "proc-1.log"
    path.touch()
    out = _hosted_logs._ProcessOutput(str(path))
    out.name_job("AJ_1", {})
    out.decide(logs=False, redaction=False)
    out.decide(logs=True, redaction=False)
    assert _jobs(hosted_env) == ["job-AJ_1.private.log"]


def test_no_decision_before_a_job_arrives(hosted_env: Path) -> None:
    hand_off()
    path = hosted_env / "jobs" / "proc-1.log"
    path.touch()
    _hosted_logs._ProcessOutput(str(path)).decide(logs=True, redaction=False)
    assert _jobs(hosted_env) == ["proc-1.log"]


@pytest.mark.parametrize(
    ("existing", "job_id", "done"),
    [
        ("proc-7.log", None, "proc-7.done.log"),
        ("proc-7.log", "AJ_1", "proc-7.done.log"),
        ("job-AJ_1.log", "AJ_1", "job-AJ_1.done.log"),
        ("job-AJ_1.export.log", "AJ_1", "job-AJ_1.export.done.log"),
        ("job-AJ_1.private.log", "AJ_1", "job-AJ_1.private.done.log"),
    ],
)
def test_mark_process_done(hosted_env: Path, existing: str, job_id: str | None, done: str) -> None:
    hand_off()
    (hosted_env / "jobs" / existing).touch()
    _hosted_logs.mark_process_done(7, job_id)
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


def test_job_process_log_lines_carry_the_job_fields(
    hosted_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    hand_off()
    path = hosted_env / "jobs" / "proc-1.log"
    path.touch()
    out = _hosted_logs._ProcessOutput(str(path))
    _hosted_logs._process_output = out
    out.name_job("AJ_1", {"job_id": "AJ_1", "room_id": "RM_1"})
    handler = _hosted_logs.process_output_log_handler()
    handler.handle(logging.LogRecord("t", logging.INFO, __file__, 1, "hello", None, None))
    line = json.loads(capsys.readouterr().out)
    assert (line["message"], line["job_id"], line["room_id"]) == ("hello", "AJ_1", "RM_1")
