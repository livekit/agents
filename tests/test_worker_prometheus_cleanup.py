"""The multiprocess cleanup at worker startup must not delete this process's own files.

`prometheus_client` caches one open mmap per file prefix, so a file this process
already holds open stays writable after being unlinked -- the samples go to a
dead inode, and `MultiProcessCollector`, which globs the directory, stops seeing
them. See https://github.com/livekit/agents/issues/7596.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from prometheus_client import CollectorRegistry, multiprocess, values

from livekit.agents.worker import _clean_stale_prometheus_files

pytestmark = pytest.mark.unit


def _gauge(name: str):
    """A default-mode gauge value, as `lk_agents_worker_load` is."""
    # a fresh MultiProcessValue gives this test its own file cache
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


def test_a_gauge_written_before_cleanup_is_still_collected_after(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))
    gauge = _gauge("app_gauge")
    gauge.set(1.0)
    assert _collected(tmp_path, "app_gauge") == 1.0

    _clean_stale_prometheus_files(str(tmp_path))

    # the process keeps using the mmap it opened before the cleanup
    gauge.set(2.0)
    assert _collected(tmp_path, "app_gauge") == 2.0, (
        "the sample is being written to an unlinked file"
    )


def test_files_from_earlier_runs_are_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))
    gauge = _gauge("own_gauge")
    gauge.set(1.0)

    own = f"gauge_all_{os.getpid()}.db"
    stale = ["gauge_all_999999.db", "counter_999999.db", "histogram_12345.db"]
    for name in stale:
        (tmp_path / name).write_bytes(b"")

    _clean_stale_prometheus_files(str(tmp_path))

    remaining = {p.name for p in tmp_path.iterdir()}
    assert own in remaining, "this process's own file was deleted"
    assert not remaining & set(stale), f"stale files left behind: {remaining & set(stale)}"
