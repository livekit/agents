from __future__ import annotations

import json
from pathlib import Path

import pytest

from livekit.agents.debug import _proc_manifest, memory

pytestmark = pytest.mark.unit


def _touch(p: Path, *, size: int = 1024) -> Path:
    p.write_bytes(b"x" * size)
    return p


def _write_manifest(directory: Path, **rec: object) -> Path:
    name = f"livekit-proc-{rec['parent_pid']}-{rec['pid']}.json"
    p = directory / name
    p.write_text(json.dumps(rec))
    return p


# --- manifest write/read --------------------------------------------------- #


def test_write_manifest_round_trip(tmp_path: Path) -> None:
    _proc_manifest.write_manifest(
        pid=101, parent_pid=100, kind="job", directory=tmp_path, job_id="AJ_x", room_id="RM_y"
    )
    info = _proc_manifest.load_manifests(tmp_path)
    assert info[101].kind == "job"
    assert info[101].job_id == "AJ_x" and info[101].room_id == "RM_y"
    assert info[101].started_at is not None


def test_write_manifest_filename_pins_parent_and_pid(tmp_path: Path) -> None:
    _proc_manifest.write_manifest(pid=42, parent_pid=7, kind="worker", directory=tmp_path)
    files = sorted(p.name for p in tmp_path.glob("livekit-proc-*.json"))
    assert files == ["livekit-proc-7-42.json"]


def test_manifest_dir_respects_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LK_DEBUG_DIR", str(tmp_path / "custom"))
    assert _proc_manifest.manifest_dir() == tmp_path / "custom"


def test_load_manifests_ignores_corrupt_files(tmp_path: Path) -> None:
    (tmp_path / "livekit-proc-1-1.json").write_text("{ not json")
    _write_manifest(tmp_path, pid=2, parent_pid=1, kind="job")
    info = _proc_manifest.load_manifests(tmp_path)
    assert set(info) == {2}


def test_load_manifests_skips_required_fields_missing(tmp_path: Path) -> None:
    (tmp_path / "livekit-proc-1-1.json").write_text(json.dumps({"pid": 1}))  # no kind/parent
    assert _proc_manifest.load_manifests(tmp_path) == {}


# --- capture discovery / labelling ----------------------------------------- #


def test_parent_filename_parses_as_worker_without_manifest(tmp_path: Path) -> None:
    _touch(tmp_path / "memray-agent.py.69962.bin")
    c = memory.discover(tmp_path)[0]
    assert c.parent_pid == 69962 and c.child_pid is None and c.pid == 69962
    # parent always reads as "worker" even without a manifest
    assert c.kind == "worker" and c.label == "worker pid=69962"


def test_child_capture_is_unknown_without_manifest(tmp_path: Path) -> None:
    _touch(tmp_path / "memray-agent.py.69962.bin.69964")
    c = memory.discover(tmp_path)[0]
    assert c.pid == 69964 and c.kind == "unknown"


def test_unrelated_files_are_ignored(tmp_path: Path) -> None:
    _touch(tmp_path / "notes.txt")
    _touch(tmp_path / "memray-agent.py.bin")  # missing pid
    _touch(tmp_path / "memray-agent.py.42.bin")  # valid
    caps = memory.discover(tmp_path)
    assert [c.path.name for c in caps] == ["memray-agent.py.42.bin"]


def test_discover_groups_children_after_parent_per_worker(tmp_path: Path) -> None:
    for name in (
        "memray-agent.py.100.bin",
        "memray-agent.py.100.bin.101",
        "memray-agent.py.100.bin.102",
        "memray-agent.py.200.bin",
        "memray-agent.py.200.bin.201",
    ):
        _touch(tmp_path / name)
    caps = memory.discover(tmp_path)
    assert [(c.parent_pid, c.child_pid) for c in caps] == [
        (100, None),
        (100, 101),
        (100, 102),
        (200, None),
        (200, 201),
    ]


def test_discover_labels_captures_from_manifests(tmp_path: Path) -> None:
    _touch(tmp_path / "memray-agent.py.100.bin")
    _touch(tmp_path / "memray-agent.py.100.bin.101")
    _touch(tmp_path / "memray-agent.py.100.bin.102")
    _write_manifest(tmp_path, pid=100, parent_pid=100, kind="worker")
    _write_manifest(tmp_path, pid=101, parent_pid=100, kind="job", job_id="AJ_x")
    _write_manifest(tmp_path, pid=102, parent_pid=100, kind="inference")

    caps = {c.pid: c for c in memory.discover(tmp_path)}
    assert caps[100].kind == "worker"
    assert caps[101].kind == "job" and caps[101].label == "job pid=101  job=AJ_x"
    assert caps[102].kind == "inference"


def test_write_index_emits_group_headers_and_kinds(tmp_path: Path) -> None:
    parent_a = _touch(tmp_path / "memray-agent.py.100.bin")
    _touch(tmp_path / "memray-agent.py.100.bin.101")
    _touch(tmp_path / "memray-agent.py.200.bin")
    _write_manifest(tmp_path, pid=100, parent_pid=100, kind="worker")
    _write_manifest(tmp_path, pid=101, parent_pid=100, kind="job", job_id="AJ_x")
    _write_manifest(tmp_path, pid=200, parent_pid=200, kind="worker")

    caps = memory.discover(tmp_path)
    rendered = [(c, c.path.with_name(c.path.name + ".flamegraph.html")) for c in caps]
    rendered[0][1].write_text("<html></html>")

    index = memory._write_index(tmp_path, rendered)
    html = index.read_text()
    assert "<!--CAPTURES-->" not in html
    assert html.count('class="group"') == 2
    assert "job pid=101" in html and "job=AJ_x" in html
    assert parent_a.with_name(parent_a.name + ".flamegraph.html").name in html


# --- CLI surface ----------------------------------------------------------- #


def test_main_list_labels_kinds_from_manifests(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _touch(tmp_path / "memray-agent.py.100.bin")
    _touch(tmp_path / "memray-agent.py.100.bin.101")
    _write_manifest(tmp_path, pid=100, parent_pid=100, kind="worker")
    _write_manifest(tmp_path, pid=101, parent_pid=100, kind="job")

    rc = memory.main(["list", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "worker pid=100" in out and "job pid=101" in out
    assert "unknown" not in out


def test_main_list_warns_about_missing_manifests(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _touch(tmp_path / "memray-agent.py.100.bin.101")  # child without a manifest
    rc = memory.main(["list", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "without a livekit-proc-*.json manifest" in out
    assert "unknown pid=101" in out


def test_main_list_empty(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = memory.main(["list", str(tmp_path)])
    assert rc == 0 and "no memray-" in capsys.readouterr().out


def test_main_no_command_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    rc = memory.main([])
    assert rc == 0 and "report" in capsys.readouterr().out


def test_report_complains_when_memray_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _touch(tmp_path / "memray-agent.py.100.bin")
    monkeypatch.setattr(memory, "_load_memray", lambda: None)
    rc = memory.main(["report", str(tmp_path)])
    assert rc == 1
    assert "memray is not installed" in capsys.readouterr().err
