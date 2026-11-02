from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import os
import sys
import threading
from typing import Any, TextIO

from .version import __version__

HOSTED_LOGS_ENV = "LIVEKIT_HOSTED_LOGS"
HOSTED_LOGS_ENABLED = "true"
HOSTED_LOGS_HANDOFF_FILE_ENV = "LIVEKIT_HOSTED_LOGS_HANDOFF_FILE"
HOSTED_LOGS_JOBS_DIR_ENV = "LIVEKIT_HOSTED_LOGS_JOBS_DIR"

_DECISION_SUFFIXES = ("", ".export", ".export.redact", ".private")

_handed_off = False
_process_output: _ProcessOutput | None = None

current_job_output: contextvars.ContextVar[JobOutput | None] = contextvars.ContextVar(
    "lk_hosted_job_output", default=None
)


def _enabled() -> bool:
    return os.environ.get(HOSTED_LOGS_ENV) == HOSTED_LOGS_ENABLED


def _handoff_file() -> str | None:
    if not _enabled():
        return None
    return os.environ.get(HOSTED_LOGS_HANDOFF_FILE_ENV) or None


def _jobs_dir() -> str | None:
    if not _enabled():
        return None
    return os.environ.get(HOSTED_LOGS_JOBS_DIR_ENV) or None


def hand_off() -> None:
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
    return _handed_off


def revoke_hand_off() -> None:
    path = _handoff_file()
    if path is not None:
        with contextlib.suppress(OSError):
            os.remove(path)


def _decision(*, logs: bool, redaction: bool) -> str:
    if not logs:
        return "private"
    return "export.redact" if redaction else "export"


class JobOutput:
    fields: dict[str, Any]

    def name_job(self, job_id: str, fields: dict[str, Any]) -> None:
        raise NotImplementedError

    def decide(self, *, logs: bool, redaction: bool) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


def _dup_onto_std(path: str) -> bool:
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


class _ProcessOutput(JobOutput):
    def __init__(self, path: str) -> None:
        self._path = path
        self._job_id: str | None = None
        self._decided = False
        self.fields = {}

    def name_job(self, job_id: str, fields: dict[str, Any]) -> None:
        if self._job_id is not None:
            return
        self._job_id = job_id
        self.fields = fields
        self._move(f"job-{job_id}.log")

    def decide(self, *, logs: bool, redaction: bool) -> None:
        if self._job_id is None or self._decided:
            return
        self._decided = True
        self._move(f"job-{self._job_id}.{_decision(logs=logs, redaction=redaction)}.log")

    def close(self) -> None:
        pass

    def _move(self, name: str) -> None:
        new_path = os.path.join(os.path.dirname(self._path), name)
        try:
            os.rename(self._path, new_path)
        except OSError:
            if not _dup_onto_std(new_path):
                _dup_onto_std(os.devnull)
        self._path = new_path


def redirect_process_output() -> JobOutput | None:
    global _process_output
    jobs_dir = _jobs_dir()
    if not _handed_off or jobs_dir is None:
        return None
    path = os.path.join(jobs_dir, f"proc-{os.getpid()}.log")
    if not _dup_onto_std(path):
        revoke_hand_off()
        return None
    _process_output = _ProcessOutput(path)
    return _process_output


def process_output() -> JobOutput | None:
    return _process_output


def mark_process_done(pid: int, job_id: str | None) -> None:
    jobs_dir = _jobs_dir()
    if not _handed_off or jobs_dir is None:
        return
    stems = [f"proc-{pid}"]
    if job_id is not None:
        stems += [f"job-{job_id}{suffix}" for suffix in _DECISION_SUFFIXES]
    for stem in stems:
        with contextlib.suppress(OSError):
            os.rename(
                os.path.join(jobs_dir, f"{stem}.log"), os.path.join(jobs_dir, f"{stem}.done.log")
            )


class _ThreadJobOutput(JobOutput):
    def __init__(self, path: str, stream: TextIO, job_id: str, fields: dict[str, Any]) -> None:
        self._path = path
        self._stream = stream
        self._job_id = job_id
        self._decided = False
        self._closed = False
        self._lock = threading.Lock()
        self.fields = fields

    def name_job(self, job_id: str, fields: dict[str, Any]) -> None:
        pass

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
            self._rename(f"job-{self._job_id}.{_decision(logs=logs, redaction=redaction)}")

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            with contextlib.suppress(OSError):
                self._stream.close()
            self._rename(f"{os.path.basename(self._path)[: -len('.log')]}.done")

    def _rename(self, stem: str) -> None:
        new_path = os.path.join(os.path.dirname(self._path), f"{stem}.log")
        try:
            os.rename(self._path, new_path)
        except OSError:
            self._closed = True
            return
        self._path = new_path


def thread_job_output(job_id: str, fields: dict[str, Any]) -> JobOutput | None:
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
    for key, value in output.fields.items():
        if not hasattr(record, key):
            setattr(record, key, value)


def _add_process_job_fields(record: logging.LogRecord) -> bool:
    if _process_output is not None:
        _add_job_fields(record, _process_output)
    return True


def process_output_log_handler() -> logging.Handler:
    from .cli.log import JsonFormatter

    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(_add_process_job_fields)
    handler.setFormatter(JsonFormatter())
    return handler


class _ThreadJobRouter(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        output = current_job_output.get()
        if not isinstance(output, _ThreadJobOutput):
            return
        try:
            _add_job_fields(record, output)
            output.write(self.format(record))
        except Exception:
            self.handleError(record)


def _not_routed(record: logging.LogRecord) -> bool:
    return not isinstance(current_job_output.get(), _ThreadJobOutput)


def install_thread_router() -> None:
    from .cli import log as cli_log
    from .log import _add_global_log_fields

    router = _ThreadJobRouter()
    router.setFormatter(cli_log.JsonFormatter())
    _add_global_log_fields(router)
    logging.getLogger().addHandler(router)
    if cli_log._stdout_handler is not None:
        cli_log._stdout_handler.addFilter(_not_routed)


hand_off()
