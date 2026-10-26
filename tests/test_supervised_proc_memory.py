from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
import socket
import sys

import psutil
import pytest

from livekit.agents.ipc.supervised_proc import SupervisedProc, SupervisedProcKind

pytestmark = [pytest.mark.unit, pytest.mark.concurrent]


class _FakeProc(SupervisedProc):
    """Minimal concrete SupervisedProc; the process is never started."""

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


def test_sample_memory_picks_metric_psutil_actually_exposes() -> None:
    proc = _make_proc()
    proc._pid = os.getpid()

    value_mb, metric = proc._sample_memory_mb()

    assert metric in ("pss", "uss", "rss")
    assert value_mb > 0

    # must select the highest-priority field psutil reports nonzero on this host
    info = psutil.Process(os.getpid()).memory_full_info()
    expected_metric = next(m for m in ("pss", "uss", "rss") if getattr(info, m, 0))
    assert metric == expected_metric


@pytest.mark.skipif(sys.platform != "linux", reason="PSS is only exposed on Linux")
def test_sample_memory_reports_pss_on_linux() -> None:
    proc = _make_proc()
    proc._pid = os.getpid()

    value_mb, metric = proc._sample_memory_mb()

    assert metric == "pss"
    assert value_mb > 0


@pytest.mark.skipif(
    sys.platform not in ("darwin", "win32"),
    reason="USS-without-PSS path applies to macOS and Windows",
)
def test_sample_memory_reports_uss_on_macos_and_windows() -> None:
    proc = _make_proc()
    proc._pid = os.getpid()

    value_mb, metric = proc._sample_memory_mb()

    assert metric == "uss"
    assert value_mb > 0


def test_sample_memory_falls_back_to_rss_when_full_info_unavailable() -> None:
    # AccessDenied/NotImplementedError can't be provoked on one's own process, so
    # only the trigger is faked: memory_full_info is shadowed on a real Process
    # instance (instance attribute, so no global/class state leaks across the
    # concurrent suite) while memory_info().rss stays a genuine psutil read.
    proc = _make_proc()
    real = psutil.Process(os.getpid())

    def _unavailable():
        raise NotImplementedError

    real.memory_full_info = _unavailable  # type: ignore[method-assign]
    proc._psutil_process = real

    value_mb, metric = proc._sample_memory_mb()

    assert metric == "rss"
    assert value_mb > 0
