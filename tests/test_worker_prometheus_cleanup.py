"""The multiprocess cleanup at worker startup must not delete this process's own files.

`prometheus_client` caches one open mmap per file prefix, so a file this process
already holds open stays writable after being unlinked: the samples keep going to
the unlinked inode, while `MultiProcessCollector` globs the directory and never
sees them again. See https://github.com/livekit/agents/issues/7596.

The lost-samples half of this only reproduces on POSIX, where unlinking an open
mmap succeeds. On Windows the unlink is refused and the helper logs it, so the
file survives for an unrelated reason -- hence the first test asserts which paths
the cleanup *passes to* `os.unlink`, which discriminates on every platform.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from prometheus_client import CollectorRegistry, multiprocess, values

from livekit.agents.worker import _clean_stale_prometheus_files

pytestmark = pytest.mark.unit

STALE = ("gauge_all_999999.db", "counter_999999.db", "histogram_12345.db")


def _own_gauge(name: str):
    """A default-mode gauge value, as `lk_agents_worker_load` is.

    A fresh `MultiProcessValue` gives this test its own file cache, so it does
    not disturb the process-wide one.
    """
    value_class = values.MultiProcessValue()
    return value_class("gauge", name, name, (), (), "help", multiprocess_mode="all")


def _collected(path: Path, name: str) -> float | None:
    """What /metrics would report for this process's sample of `name`.

    Default-mode gauges keep one sample per process, labelled with the pid, so
    the lookup carries that label -- the same shape `lk_agents_worker_load` has.
    """
    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry, path=str(path))
    return registry.get_sample_value(name, {"pid": str(os.getpid())})


def test_the_cleanup_never_unlinks_a_file_belonging_to_this_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    own = f"gauge_all_{os.getpid()}.db"
    for name in (own, *STALE):
        (tmp_path / name).write_bytes(b"")

    attempted: list[str] = []
    real_unlink = os.unlink

    def spy(path, *args, **kwargs):  # type: ignore[no-untyped-def]
        attempted.append(os.path.basename(path))
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", spy)

    _clean_stale_prometheus_files(str(tmp_path))

    assert own not in attempted, "the cleanup tried to delete this process's own file"
    assert sorted(attempted) == sorted(STALE), f"unexpected deletions: {attempted}"


def test_files_from_earlier_runs_are_still_removed(tmp_path: Path) -> None:
    own = f"gauge_all_{os.getpid()}.db"
    for name in (own, *STALE):
        (tmp_path / name).write_bytes(b"")

    _clean_stale_prometheus_files(str(tmp_path))

    remaining = {p.name for p in tmp_path.iterdir()}
    assert remaining == {own}, f"expected only this process's file to survive, got {remaining}"


@pytest.mark.skipif(os.name == "nt", reason="unlinking an open mmap is refused on Windows")
def test_a_gauge_written_before_the_cleanup_is_still_collected_after(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))
    gauge = _own_gauge("app_gauge")
    gauge.set(1.0)
    assert _collected(tmp_path, "app_gauge") == 1.0

    _clean_stale_prometheus_files(str(tmp_path))

    # the process keeps writing through the mmap it opened before the cleanup
    gauge.set(2.0)
    assert _collected(tmp_path, "app_gauge") == 2.0, (
        "the sample is going to an unlinked file; /metrics will never report it again"
    )
