from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import multiprocessing as mp
import os
import subprocess
import sys
import textwrap
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from livekit.agents import JobProcess, _hosted_logs, ipc, job
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
from livekit.protocol import agent

pytestmark = pytest.mark.unit


@pytest.fixture
def hosted_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv(HOSTED_LOGS_ENV, HOSTED_LOGS_ENABLED)
    monkeypatch.setenv(HOSTED_LOGS_HANDOFF_FILE_ENV, str(tmp_path / "sdk-handoff"))
    monkeypatch.setenv(HOSTED_LOGS_JOBS_DIR_ENV, str(tmp_path / "jobs"))
    monkeypatch.setattr(_hosted_logs, "_handed_off", False)
    monkeypatch.delenv(_hosted_logs._HANDED_OFF_ENV, raising=False)
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
        out = _hosted_logs.redirect_job_output("AJ_1", {}, redaction=False)
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
    assert _hosted_logs.redirect_job_output("AJ_1", {}, redaction=False) is None


@pytest.mark.parametrize(
    ("logs", "redaction", "name"),
    [
        (True, False, "job-AJ_1.export.log"),
        (True, True, "job-AJ_1.export.log"),
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
    _hosted_logs._ProcessOutput(str(path), "AJ_1", {}, redaction=False).decide(
        logs=logs, redaction=redaction
    )
    assert _jobs(hosted_env) == [name]


def test_the_first_decision_wins(hosted_env: Path) -> None:
    hand_off()
    path = hosted_env / "jobs" / "job-AJ_1.log"
    path.touch()
    out = _hosted_logs._ProcessOutput(str(path), "AJ_1", {}, redaction=False)
    out.decide(logs=False, redaction=False)
    out.decide(logs=True, redaction=False)
    assert _jobs(hosted_env) == ["job-AJ_1.private.log"]


@pytest.mark.parametrize(
    ("existing", "done"),
    [
        ("job-AJ_1.log", "job-AJ_1.done.log"),
        ("job-AJ_1.export.log", "job-AJ_1.export.done.log"),
        ("job-AJ_1.private.log", "job-AJ_1.private.done.log"),
    ],
)
def test_mark_process_done(hosted_env: Path, existing: str, done: str) -> None:
    (hosted_env / "jobs").mkdir()
    (hosted_env / "jobs" / existing).touch()
    _hosted_logs.mark_process_done("AJ_1")
    assert _jobs(hosted_env) == [done]


def test_thread_job_output_routes_job_records(hosted_env: Path) -> None:
    hand_off()
    out = _hosted_logs.thread_job_output(
        "AJ_1", {"job_id": "AJ_1", "room_id": "RM_1"}, redaction=False
    )
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
    out = _hosted_logs.thread_job_output("AJ_1", {}, redaction=False)
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
    out = _hosted_logs.thread_job_output("AJ_1", {}, redaction=False)
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


@pytest.mark.parametrize("recording", [True, False])
async def test_cleanup_follows_the_job_when_no_session_decided(recording: bool) -> None:
    ctx = MagicMock()
    ctx._recording_initialized = False
    ctx._rejected_recording = None
    ctx._early_log_handler = None
    ctx._telemetry_state = None
    ctx._handlers_with_filter = []
    ctx.job.enable_recording = recording
    ctx.job.enable_redaction = True
    await JobContext._on_cleanup(ctx)
    ctx._hosted_output.decide.assert_called_once_with(logs=recording, redaction=True)


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
        out = _hosted_logs.redirect_job_output(
            "AJ_1", {"job_id": "AJ_1", "room_id": "RM_1"}, redaction=False
        )
        assert queue not in root.handlers
        logging.getLogger("t").info("hello")
        out.decide(logs=True, redaction=False)
        print(os.getpid(), file=open(os.environ["PID_FILE"], "w"))
        """
    )
    pid_file = hosted_env / "pid"
    subprocess.run(
        [sys.executable, "-c", script], check=True, env={**os.environ, "PID_FILE": str(pid_file)}
    )
    line = json.loads((hosted_env / "jobs" / "job-AJ_1.export.log").read_text())
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


def _route(out: _hosted_logs.JobOutput, record: logging.LogRecord) -> None:
    from livekit.agents.cli.log import JsonFormatter

    router = _hosted_logs._ThreadJobRouter()
    router.setFormatter(JsonFormatter())
    token = _hosted_logs.current_job_output.set(out)
    try:
        router.handle(record)
    finally:
        _hosted_logs.current_job_output.reset(token)


@pytest.mark.parametrize(("project", "session"), [(True, False), (False, True)])
def test_thread_job_files_redact_pii_when_redaction_is_on(
    hosted_env: Path, project: bool, session: bool
) -> None:
    hand_off()
    out = _hosted_logs.thread_job_output(
        "AJ_1", {"job_id": "AJ_1", "room": "alice-room"}, redaction=project
    )
    assert out is not None
    record = _record_with_pii()
    _route(out, record)
    assert (hosted_env / "jobs" / "job-AJ_1.log").read_text() == ""
    out.decide(logs=True, redaction=session)
    out.close()

    line = json.loads((hosted_env / "jobs" / "job-AJ_1.export.done.log").read_text())
    assert "lk.pii.transcript" not in line
    assert line["room"] == "alice-room"
    assert line["exc_info"] == "ValueError: exception details redacted"
    assert "secret user text" not in json.dumps(line)
    assert line["lk.redaction.enabled"] is True
    assert line["job_id"] == "AJ_1"
    assert record.__dict__["lk.pii.transcript"] == "hello my name is"


def test_thread_job_files_keep_pii_when_redaction_is_off(hosted_env: Path) -> None:
    hand_off()
    out = _hosted_logs.thread_job_output("AJ_1", {}, redaction=False)
    assert out is not None
    _route(out, _record_with_pii())
    out.decide(logs=True, redaction=False)
    out.close()
    line = json.loads((hosted_env / "jobs" / "job-AJ_1.export.done.log").read_text())
    assert line["lk.pii.transcript"] == "hello my name is"
    assert "lk.redaction.enabled" not in line


def test_thread_job_records_without_a_decision_are_never_written(hosted_env: Path) -> None:
    hand_off()
    out = _hosted_logs.thread_job_output("AJ_1", {}, redaction=False)
    assert out is not None
    _route(out, _record_with_pii())
    out.close()
    assert (hosted_env / "jobs" / "job-AJ_1.done.log").read_text() == ""


@pytest.mark.parametrize(("project", "session"), [(True, False), (False, True)])
def test_job_process_redacts_every_thread(hosted_env: Path, project: bool, session: bool) -> None:
    script = textwrap.dedent(
        f"""
        import logging, threading
        from livekit.agents import _hosted_logs
        logging.getLogger().setLevel(logging.INFO)
        out = _hosted_logs.redirect_job_output(
            "AJ_1", {{"job_id": "AJ_1", "room": "alice-room"}}, redaction={project}
        )
        log = lambda: logging.getLogger("t").info("x", extra={{"lk.pii.event": "secret"}})
        log()
        assert open("{hosted_env}/jobs/job-AJ_1.log").read() == ""
        out.decide(logs=True, redaction={session})
        t = threading.Thread(target=log)
        t.start()
        t.join()
        """
    )
    subprocess.run([sys.executable, "-c", script], check=True)
    lines = (hosted_env / "jobs" / "job-AJ_1.export.log").read_text().splitlines()
    assert len(lines) == 2
    for line in map(json.loads, lines):
        assert "lk.pii.event" not in line
        assert line["room"] == "alice-room"
        assert line["lk.redaction.enabled"] is True
        assert line["job_id"] == "AJ_1"


def test_job_process_records_without_a_decision_are_never_written(hosted_env: Path) -> None:
    script = textwrap.dedent(
        """
        import logging
        from livekit.agents import _hosted_logs
        logging.getLogger().setLevel(logging.INFO)
        _hosted_logs.redirect_job_output("AJ_1", {}, redaction=False)
        logging.getLogger("t").info("held")
        print("raw", flush=True)
        """
    )
    subprocess.run([sys.executable, "-c", script], check=True)
    assert (hosted_env / "jobs" / "job-AJ_1.log").read_text() == "raw\n"


def test_job_output_is_dropped_when_its_file_cannot_be_opened(hosted_env: Path) -> None:
    script = textwrap.dedent(
        """
        import logging, os, shutil
        from livekit.agents import _hosted_logs
        from livekit.agents.ipc.log_queue import LogQueueHandler
        root = logging.getLogger()
        queue = LogQueueHandler.__new__(LogQueueHandler)
        logging.Handler.__init__(queue)
        queue.close = lambda: None
        queue.emit = lambda record: None
        root.addHandler(queue)
        shutil.rmtree(os.environ["LIVEKIT_HOSTED_LOGS_JOBS_DIR"])
        assert _hosted_logs.redirect_job_output("AJ_1", {}, redaction=False) is None
        assert queue not in root.handlers
        print("dropped", flush=True)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], check=True, capture_output=True, text=True
    )
    assert result.stdout == ""


def test_process_output_falls_back_when_the_rename_fails(hosted_env: Path) -> None:
    script = textwrap.dedent(
        """
        import os
        from livekit.agents import _hosted_logs
        out = _hosted_logs.redirect_job_output("AJ_1", {}, redaction=False)
        print("before", flush=True)
        def fail(src, dst):
            raise OSError("busy")
        os.rename = fail
        out.decide(logs=True, redaction=False)
        print("after", flush=True)
        """
    )
    subprocess.run([sys.executable, "-c", script], check=True)
    jobs = hosted_env / "jobs"
    assert (jobs / "job-AJ_1.log").read_text() == "before\n"
    assert (jobs / "job-AJ_1.export.log").read_text() == "after\n"


def test_hand_off_sets_the_env_for_child_processes(hosted_env: Path) -> None:
    hand_off()
    assert os.environ[_hosted_logs._HANDED_OFF_ENV] == "1"


def test_child_trusts_the_parent_hand_off(
    hosted_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(_hosted_logs._HANDED_OFF_ENV, "1")
    hand_off()
    assert logs_handed_off()
    assert not (hosted_env / "sdk-handoff").exists()


def test_install_thread_router_runs_once(monkeypatch: pytest.MonkeyPatch) -> None:
    root = logging.getLogger()
    monkeypatch.setattr(root, "handlers", [])
    _hosted_logs.install_thread_router()
    _hosted_logs.install_thread_router()
    assert sum(isinstance(h, _hosted_logs._ThreadJobRouter) for h in root.handlers) == 1


_REJECTED = {"logs": False, "redaction": True, "transcript": False}


def _cleanup_ctx() -> MagicMock:
    ctx = MagicMock()
    ctx._recording_initialized = False
    ctx._rejected_recording = None
    ctx._early_log_handler = None
    ctx._telemetry_state = None
    ctx._handlers_with_filter = []
    ctx.job.enable_redaction = False
    return ctx


async def test_cleanup_uses_rejected_options_when_no_session_decided() -> None:
    ctx = _cleanup_ctx()
    with pytest.raises(ValueError):
        JobContext.init_recording(ctx, _REJECTED)
    ctx._hosted_output.decide.assert_not_called()
    await JobContext._on_cleanup(ctx)
    ctx._hosted_output.decide.assert_called_once_with(logs=False, redaction=True)


def test_a_valid_retry_decides_after_rejected_options() -> None:
    ctx = _cleanup_ctx()
    with pytest.raises(ValueError):
        JobContext.init_recording(ctx, _REJECTED)
    with contextlib.suppress(Exception):
        JobContext.init_recording(ctx, {"logs": True, "redaction": True, "transcript": True})
    ctx._hosted_output.decide.assert_called_once_with(logs=True, redaction=True)


@pytest.mark.parametrize(("exitcode", "marked"), [(0, True), (None, False)])
async def test_job_files_are_marked_done_only_after_exit(
    monkeypatch: pytest.MonkeyPatch, exitcode: int | None, marked: bool
) -> None:
    from livekit.agents.ipc import job_proc_executor
    from livekit.agents.ipc.supervised_proc import SupervisedProc

    async def supervise(self: object) -> None:
        pass

    monkeypatch.setattr(SupervisedProc, "_supervise_task", supervise)
    monkeypatch.setattr(job_proc_executor.metrics, "job_ended", lambda: None)
    done = MagicMock()
    monkeypatch.setattr(job_proc_executor._hosted_logs, "mark_process_done", done)
    executor = object.__new__(job_proc_executor.ProcJobExecutor)
    executor._running_job = MagicMock()
    executor._running_job.job.id = "AJ_1"
    executor._exitcode = exitcode
    await job_proc_executor.ProcJobExecutor._supervise_task(executor)
    assert done.called == marked


async def _hosted_entrypoint(job_ctx: JobContext) -> None:
    print("raw print", flush=True)
    logging.getLogger("hosted-test").warning("hello", extra={"lk.pii.secret": "s"})
    job_ctx.shutdown("done")


def _hosted_job(*, redaction: bool) -> job.RunningJobInfo:
    return job.RunningJobInfo(
        job=agent.Job(
            id=f"AJ_{uuid.uuid4().hex[:12]}",
            type=agent.JobType.JT_ROOM,
            enable_recording=True,
            enable_redaction=redaction,
        ),
        url="fake_url",
        token="fake_token",
        accept_arguments=job.JobAcceptArguments(name="", identity="", metadata=""),
        worker_id="fake_id",
        fake_job=True,
    )


def _noop_initialize(proc: JobProcess) -> None:
    pass


def _job_lines(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    records, raw = [], []
    for line in path.read_text().splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            raw.append(line)
    return records, raw


@pytest.mark.parametrize("redaction", [False, True])
async def test_job_process_writes_its_file_end_to_end(hosted_env: Path, redaction: bool) -> None:
    hand_off()
    executor = ipc.job_proc_executor.ProcJobExecutor(
        initialize_process_fnc=_noop_initialize,
        job_entrypoint_fnc=_hosted_entrypoint,
        session_end_fnc=None,
        simulation_end_fnc=None,
        initialize_timeout=20.0,
        close_timeout=10.0,
        session_end_timeout=300.0,
        memory_warn_mb=0,
        memory_limit_mb=0,
        ping_interval=2.5,
        ping_timeout=10.0,
        high_ping_threshold=1.0,
        inference_executor=None,
        http_proxy=None,
        mp_ctx=mp.get_context("spawn"),
        loop=asyncio.get_running_loop(),
    )
    info = _hosted_job(redaction=redaction)
    await executor.start()
    await executor.initialize()
    await executor.launch_job(info)
    await asyncio.wait_for(executor.join(), timeout=30.0)
    await executor.aclose()

    path = hosted_env / "jobs" / f"job-{info.job.id}.export.done.log"
    assert _jobs(hosted_env) == [path.name]
    records, raw = _job_lines(path)
    assert "raw print" in raw
    hello = next(r for r in records if r["message"] == "hello")
    assert hello["job_id"] == info.job.id
    assert ("lk.pii.secret" in hello) is not redaction
    assert hello.get("lk.redaction.enabled", False) is redaction


async def test_thread_job_writes_its_file_end_to_end(
    hosted_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hand_off()
    monkeypatch.setattr(logging.getLogger(), "handlers", [])
    _hosted_logs.install_thread_router()
    executor = ipc.job_thread_executor.ThreadJobExecutor(
        initialize_process_fnc=_noop_initialize,
        job_entrypoint_fnc=_hosted_entrypoint,
        session_end_fnc=None,
        simulation_end_fnc=None,
        inference_executor=None,
        initialize_timeout=20.0,
        close_timeout=10.0,
        session_end_timeout=300.0,
        ping_interval=2.5,
        high_ping_threshold=1.0,
        http_proxy=None,
        loop=asyncio.get_running_loop(),
    )
    info = _hosted_job(redaction=False)
    await executor.start()
    await executor.initialize()
    await executor.launch_job(info)
    await asyncio.wait_for(executor.join(), timeout=30.0)
    await asyncio.wait_for(executor.aclose(), timeout=30.0)

    path = hosted_env / "jobs" / f"job-{info.job.id}.export.done.log"
    assert _jobs(hosted_env) == [path.name]
    records, _ = _job_lines(path)
    hello = next(r for r in records if r["message"] == "hello")
    assert hello["job_id"] == info.job.id
