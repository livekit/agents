"""Regression tests for #7596: the startup cleanup of PROMETHEUS_MULTIPROC_DIR must keep the
metric files that the server process already has open."""

from __future__ import annotations

import os
from pathlib import Path

import psutil
import pytest

from livekit.agents.worker import _clean_prometheus_multiproc_dir

pytestmark = pytest.mark.unit


def test_keeps_files_this_process_has_open(tmp_path: Path) -> None:
    metric_file = tmp_path / f"gauge_all_{os.getpid()}.db"
    # prometheus_client holds its multiprocess files open like this for the process lifetime
    with open(metric_file, "a+b"):
        _clean_prometheus_multiproc_dir(str(tmp_path))

        assert metric_file.exists()


def test_removes_files_this_process_does_not_have_open(tmp_path: Path) -> None:
    # files left by earlier processes; a restarted container often reuses this process's pid
    stale_files = [tmp_path / "counter_4194303.db", tmp_path / f"gauge_all_{os.getpid()}.db"]
    for stale_file in stale_files:
        stale_file.write_bytes(b"")

    _clean_prometheus_multiproc_dir(str(tmp_path))

    assert list(tmp_path.iterdir()) == []


def test_skips_cleanup_when_open_files_cannot_be_listed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metric_file = tmp_path / f"gauge_all_{os.getpid()}.db"
    metric_file.write_bytes(b"")

    def open_files_denied(self: psutil.Process) -> list[object]:
        raise psutil.AccessDenied(self.pid)

    monkeypatch.setattr(psutil.Process, "open_files", open_files_denied)

    _clean_prometheus_multiproc_dir(str(tmp_path))

    assert metric_file.exists()
