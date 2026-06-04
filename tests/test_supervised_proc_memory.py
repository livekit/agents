from __future__ import annotations

import asyncio
import multiprocessing as mp
import socket

import pytest

from livekit.agents.ipc.supervised_proc import (
    _MEMORY_WARN_COOLDOWN,
    _MEMORY_WARN_RESET_DELTA_MB,
    SupervisedProc,
    SupervisedProcKind,
)

pytestmark = pytest.mark.unit


class _FakeProc(SupervisedProc):
    """Minimal concrete SupervisedProc for testing the memory-warning bookkeeping.

    The process is never actually started; we only exercise the pure helpers that
    decide whether/what to log.
    """

    @property
    def process_kind(self) -> SupervisedProcKind:
        return SupervisedProcKind.JOB

    def _create_process(self, cch: socket.socket, log_cch: socket.socket):  # pragma: no cover
        raise NotImplementedError

    async def _main_task(self, ipc_ch):  # pragma: no cover
        raise NotImplementedError


def _make_proc(*, memory_warn_mb: float = 500, memory_limit_mb: float = 0) -> _FakeProc:
    return _FakeProc(
        initialize_timeout=10.0,
        close_timeout=10.0,
        memory_warn_mb=memory_warn_mb,
        memory_limit_mb=memory_limit_mb,
        ping_interval=2.5,
        ping_timeout=60.0,
        high_ping_threshold=0.5,
        http_proxy=None,
        mp_ctx=mp.get_context("spawn"),
        loop=asyncio.get_event_loop(),
    )


async def test_memory_warning_is_rate_limited() -> None:
    proc = _make_proc()

    # first time over the threshold fires immediately
    assert proc._should_emit_memory_warning(520.0, now=1000.0) is True
    # a sample right after, still above the threshold, is suppressed
    assert proc._should_emit_memory_warning(521.0, now=1005.0) is False
    assert proc._should_emit_memory_warning(522.0, now=1010.0) is False
    # once the cooldown elapses it fires again
    assert proc._should_emit_memory_warning(522.0, now=1000.0 + _MEMORY_WARN_COOLDOWN + 1) is True


async def test_memory_warning_reemits_on_significant_growth() -> None:
    proc = _make_proc()

    assert proc._should_emit_memory_warning(520.0, now=1000.0) is True
    # well within the cooldown, but usage jumped: re-emit so a real leak surfaces
    grown = 520.0 + _MEMORY_WARN_RESET_DELTA_MB + 1
    assert proc._should_emit_memory_warning(grown, now=1005.0) is True


async def test_memory_logging_extra_reports_baseline_and_growth() -> None:
    proc = _make_proc()

    # before a baseline is captured, only the basic fields are present
    extra = proc._memory_logging_extra(520.0)
    assert extra["memory_usage_mb"] == 520.0
    assert extra["memory_warn_mb"] == 500
    assert extra["has_running_job"] is False
    assert "uptime" in extra
    assert "baseline_memory_mb" not in extra

    # once a baseline is set, growth-since-startup is reported
    proc._memory_baseline_mb = 300.0
    extra = proc._memory_logging_extra(520.0)
    assert extra["baseline_memory_mb"] == 300.0
    assert extra["growth_memory_mb"] == 220.0


async def test_uptime_is_zero_before_start() -> None:
    proc = _make_proc()
    assert proc.uptime == 0.0


def test_process_kind_renders_as_plain_string() -> None:
    # the log message interpolates `f"{self.process_kind} process …"`, so the enum
    # must stringify to its value (not "SupervisedProcKind.JOB")
    assert f"{SupervisedProcKind.JOB} process" == "job process"
    assert f"{SupervisedProcKind.INFERENCE} process" == "inference process"


def test_subclassing_without_process_kind_is_rejected() -> None:
    class _MissingKind(SupervisedProc):
        def _create_process(self, cch, log_cch):  # pragma: no cover
            raise NotImplementedError

        async def _main_task(self, ipc_ch):  # pragma: no cover
            raise NotImplementedError

    with pytest.raises(TypeError, match="process_kind"):
        _MissingKind(
            initialize_timeout=1.0,
            close_timeout=1.0,
            memory_warn_mb=0.0,
            memory_limit_mb=0.0,
            ping_interval=1.0,
            ping_timeout=1.0,
            high_ping_threshold=1.0,
            http_proxy=None,
            mp_ctx=mp.get_context("spawn"),
            loop=asyncio.get_event_loop(),
        )
