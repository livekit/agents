"""Hands log collection to LiveKit Cloud, with one output file per job."""

from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import os
import sys
import threading
from typing import Any, Protocol, TextIO

from .version import __version__

HOSTED_LOGS_ENV = "LIVEKIT_HOSTED_LOGS"
HOSTED_LOGS_ENABLED = "true"
HOSTED_LOGS_HANDOFF_FILE_ENV = "LIVEKIT_HOSTED_LOGS_HANDOFF_FILE"
HOSTED_LOGS_JOBS_DIR_ENV = "LIVEKIT_HOSTED_LOGS_JOBS_DIR"

_DECISION_SUFFIXES = ("", ".export", ".export.redact", ".private")

_handed_off = False

current_job_output: contextvars.ContextVar[JobOutput | None] = contextvars.ContextVar(
    "lk_hosted_job_output", default=None
)


def _enabled() -> bool:
    """Whether LiveKit Cloud collects this agent's logs."""
    return os.environ.get(HOSTED_LOGS_ENV) == HOSTED_LOGS_ENABLED


def _handoff_file() -> str | None:
    """The hand-off file path, or None when hosted logs are off."""
    if not _enabled():
        return None
    return os.environ.get(HOSTED_LOGS_HANDOFF_FILE_ENV) or None


def _jobs_dir() -> str | None:
    """The per-job log file directory, or None when hosted logs are off."""
    if not _enabled():
        return None
    return os.environ.get(HOSTED_LOGS_JOBS_DIR_ENV) or None


def hand_off() -> None:
    """Write the hand-off file, which tells LiveKit Cloud to collect logs and stops OTLP export."""
    global _handed_off
    path = _handoff_file()
    jobs_dir = _jobs_dir()
    if path is None or jobs_dir is None:
        return
    tmp = f"{path}.{os.getpid()}.tmp"
    try:
        os.makedirs(jobs_dir, exist_ok=True)
        with open(tmp, "w") as f:
            json.dump({"sdk": "python", "version": __version__}, f)
        os.replace(tmp, path)
    except OSError:
        with contextlib.suppress(OSError):
            os.remove(tmp)
        return
    _handed_off = True


def logs_handed_off() -> bool:
    """Whether this process handed log collection to LiveKit Cloud."""
    return _handed_off


def _decision(*, logs: bool, redaction: bool) -> str:
    """The file name decision: ``private`` (not collected), ``export`` or ``export.redact``."""
    if not logs:
        return "private"
    return "export.redact" if redaction else "export"


class JobOutput(Protocol):
    """A job's log file, renamed to record the session's logs decision."""

    fields: dict[str, Any]

    def decide(self, *, logs: bool, redaction: bool) -> None: ...

    def close(self) -> None: ...


def _dup_onto_std(path: str) -> bool:
    """Point fd 1 and fd 2 at ``path``; False if it can't be opened."""
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    except OSError:
        return False
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            stream.flush()
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.close(fd)
    return True


class _ProcessOutput:
    """The log file of a job process, which writes to it through fd 1 and fd 2."""

    def __init__(self, path: str, job_id: str, fields: dict[str, Any]) -> None:
        self._path = path
        self._job_id = job_id
        self._decided = False
        self.fields = fields

    def decide(self, *, logs: bool, redaction: bool) -> None:
        if self._decided:
            return
        self._decided = True
        new_path = os.path.join(
            os.path.dirname(self._path),
            f"job-{self._job_id}.{_decision(logs=logs, redaction=redaction)}.log",
        )
        try:
            os.rename(self._path, new_path)
        except OSError:
            if not _dup_onto_std(new_path):
                _dup_onto_std(os.devnull)
        self._path = new_path

    def close(self) -> None:
        pass


def _redacted(record: logging.LogRecord) -> logging.LogRecord:
    """A copy without PII fields or exception details when redaction is on, else ``record``."""
    from .telemetry import pii
    from .telemetry.utils import REDACTED_EXCEPTION_MESSAGE, redaction_enabled

    if not redaction_enabled():
        return record
    copy = logging.makeLogRecord(
        {k: v for k, v in record.__dict__.items() if not pii.is_pii_attribute(k)}
    )
    if record.exc_info and record.exc_info[0] is not None:
        copy.exc_info = None
        copy.exc_text = f"{record.exc_info[0].__name__}: {REDACTED_EXCEPTION_MESSAGE}"
    elif record.exc_text:
        copy.exc_text = REDACTED_EXCEPTION_MESSAGE
    return copy


class _JobStreamHandler(logging.StreamHandler):
    """Writes redacted records to stdout, which is the job file."""

    def format(self, record: logging.LogRecord) -> str:
        return super().format(_redacted(record))


class _JobFieldsFilter(logging.Filter):
    """Adds the job fields to each record."""

    def __init__(self, output: JobOutput) -> None:
        super().__init__()
        self._output = output

    def filter(self, record: logging.LogRecord) -> bool:
        _add_job_fields(record, self._output)
        return True


def redirect_job_output(job_id: str, fields: dict[str, Any]) -> JobOutput | None:
    """Send this job process's output and log records to the job's file."""
    from .cli.log import JsonFormatter
    from .ipc.log_queue import LogQueueHandler
    from .log import _add_global_log_fields

    jobs_dir = _jobs_dir()
    if not _handed_off or jobs_dir is None:
        return None
    path = os.path.join(jobs_dir, f"job-{job_id}.log")
    if not _dup_onto_std(path):
        return None
    output = _ProcessOutput(path, job_id, {**fields, "pid": os.getpid()})

    handler = _JobStreamHandler(sys.stdout)
    handler.addFilter(_JobFieldsFilter(output))
    handler.setFormatter(JsonFormatter())
    _add_global_log_fields(handler)
    root = logging.getLogger()
    for h in list(root.handlers):
        if isinstance(h, LogQueueHandler):
            root.removeHandler(h)
    root.addHandler(handler)
    return output


def mark_process_done(job_id: str) -> None:
    """Add ``.done`` to a job's file after its process exits."""
    jobs_dir = _jobs_dir()
    if not _handed_off or jobs_dir is None:
        return
    for suffix in _DECISION_SUFFIXES:
        stem = f"job-{job_id}{suffix}"
        with contextlib.suppress(OSError):
            os.rename(
                os.path.join(jobs_dir, f"{stem}.log"), os.path.join(jobs_dir, f"{stem}.done.log")
            )


class _ThreadJobOutput:
    """The log file of a thread job, written by the thread router."""

    def __init__(self, path: str, stream: TextIO, job_id: str, fields: dict[str, Any]) -> None:
        self._path = path
        self._stream = stream
        self._job_id = job_id
        self._decided = False
        self._closed = False
        self._lock = threading.Lock()
        self.fields = fields

    def write(self, line: str) -> None:
        with self._lock:
            if self._closed:
                return
            self._stream.write(line + "\n")
            self._stream.flush()

    def decide(self, *, logs: bool, redaction: bool) -> None:
        with self._lock:
            if self._decided or self._closed:
                return
            self._decided = True
            renamed = self._rename(
                f"job-{self._job_id}.{_decision(logs=logs, redaction=redaction)}"
            )
        if not renamed:
            self._warn_rename_failed()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            renamed = self._rename(f"{os.path.basename(self._path)[: -len('.log')]}.done")
            self._closed = True
            with contextlib.suppress(OSError):
                self._stream.close()
        if not renamed:
            self._warn_rename_failed()

    def _rename(self, stem: str) -> bool:
        new_path = os.path.join(os.path.dirname(self._path), f"{stem}.log")
        try:
            os.rename(self._path, new_path)
        except OSError:
            self._closed = True
            with contextlib.suppress(OSError):
                self._stream.close()
            return False
        self._path = new_path
        return True

    def _warn_rename_failed(self) -> None:
        from .log import logger

        token = current_job_output.set(None)
        try:
            logger.warning(
                "failed to rename the hosted log file, dropping the rest of this job's output",
                extra={"job_id": self._job_id},
            )
        finally:
            current_job_output.reset(token)


def thread_job_output(job_id: str, fields: dict[str, Any]) -> JobOutput | None:
    """Open the log file of a thread job."""
    jobs_dir = _jobs_dir()
    if not _handed_off or jobs_dir is None:
        return None
    path = os.path.join(jobs_dir, f"job-{job_id}.log")
    try:
        stream = open(path, "a", encoding="utf-8")
    except OSError:
        return None
    return _ThreadJobOutput(path, stream, job_id, fields)


def _add_job_fields(record: logging.LogRecord, output: JobOutput) -> None:
    """Set the job fields that the record doesn't already have."""
    for key, value in output.fields.items():
        if not hasattr(record, key):
            setattr(record, key, value)


class _ThreadJobRouter(logging.Handler):
    """Writes each record to the current thread job's file."""

    def emit(self, record: logging.LogRecord) -> None:
        output = current_job_output.get()
        if not isinstance(output, _ThreadJobOutput):
            return
        try:
            _add_job_fields(record, output)
            output.write(self.format(_redacted(record)))
        except Exception:
            self.handleError(record)


def _not_routed(record: logging.LogRecord) -> bool:
    """Filter for other handlers: skip records that the thread router writes."""
    return not isinstance(current_job_output.get(), _ThreadJobOutput)


def install_thread_router() -> None:
    """Route thread job records to their files, and keep them out of other handlers."""
    from .cli.log import JsonFormatter
    from .log import _add_global_log_fields

    root = logging.getLogger()
    for handler in root.handlers:
        handler.addFilter(_not_routed)
    router = _ThreadJobRouter()
    router.setFormatter(JsonFormatter())
    _add_global_log_fields(router)
    root.addHandler(router)


hand_off()
