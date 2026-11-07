"""The startup cleanup of PROMETHEUS_MULTIPROC_DIR must not delete files a live process still writes."""

from __future__ import annotations

import os

import pytest

from livekit.agents import worker

pytestmark = pytest.mark.unit


def _touch(path, name: str) -> str:
    file_path = path / name
    file_path.write_bytes(b"")
    return name


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("gauge_all_1234.db", 1234),
        ("counter_42.db", 42),
        ("histogram_1234567890.db", 1234567890),
        ("gauge_all.db", None),
        ("README", None),
        ("gauge_all_12.db.tmp", None),
    ],
)
def test_multiproc_file_pid(filename: str, expected: int | None) -> None:
    assert worker._multiproc_file_pid(filename) == expected


def test_cleanup_keeps_files_of_live_processes_and_removes_dead_ones(tmp_path, monkeypatch) -> None:
    """A file whose process still runs must survive, otherwise that process's metrics vanish."""
    own = _touch(tmp_path, f"gauge_all_{os.getpid()}.db")
    live = _touch(tmp_path, "gauge_all_4242.db")
    dead = _touch(tmp_path, "gauge_all_4243.db")
    unknown = _touch(tmp_path, "counter_custom.db")

    monkeypatch.setattr(worker, "_process_is_alive", lambda pid: pid in (os.getpid(), 4242))

    worker._clean_prometheus_multiproc_dir(str(tmp_path))

    remaining = set(os.listdir(tmp_path))
    assert own in remaining
    assert live in remaining
    assert dead not in remaining
    # unknown names cannot be attributed to a process, so they are cleaned like before
    assert unknown not in remaining


def test_cleanup_removes_everything_when_no_process_is_alive(tmp_path, monkeypatch) -> None:
    for name in ("gauge_all_1.db", "counter_2.db"):
        _touch(tmp_path, name)

    monkeypatch.setattr(worker, "_process_is_alive", lambda pid: False)

    worker._clean_prometheus_multiproc_dir(str(tmp_path))

    assert os.listdir(tmp_path) == []


def test_current_process_is_always_alive() -> None:
    assert worker._process_is_alive(os.getpid()) is True
