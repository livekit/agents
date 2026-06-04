from __future__ import annotations

import json
from pathlib import Path

import pytest

from livekit.agents.debug import memory

pytestmark = pytest.mark.unit


def _touch(p: Path, *, size: int = 1024) -> Path:
    p.write_bytes(b"x" * size)
    return p


def test_parent_filename_parses_as_worker(tmp_path: Path) -> None:
    _touch(tmp_path / "memray-agent.py.69962.bin")
    caps = memory.discover(tmp_path)
    assert len(caps) == 1
    c = caps[0]
    assert c.script == "agent.py" and c.parent_pid == 69962
    assert c.is_parent is True and c.child_pid is None
    assert c.pid == 69962
    # without logs, a parent capture is assumed to be the worker
    assert c.kind == "worker"
    assert c.label == "worker pid=69962"


def test_child_filename_parses_with_parent_pid(tmp_path: Path) -> None:
    _touch(tmp_path / "memray-agent.py.69962.bin.69964")
    caps = memory.discover(tmp_path)
    c = caps[0]
    assert c.parent_pid == 69962 and c.child_pid == 69964
    assert c.is_parent is False and c.pid == 69964
    # without logs the kind of a child is unknown
    assert c.kind == "unknown"
    assert c.label == "unknown pid=69964"


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


# --- log ingestion / kind labelling ---------------------------------------- #


def _write_log(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records))


def test_load_proc_info_maps_pid_to_kind(tmp_path: Path) -> None:
    log = tmp_path / "agent.log"
    _write_log(
        log,
        [
            {"message": "unrelated", "pid": 100},  # ignored: no memory_event
            {"memory_event": "started", "pid": 100, "process_kind": "worker"},
            {
                "memory_event": "periodic",
                "pid": 101,
                "process_kind": "job",
                "job_id": "AJ_x",
                "room_id": "RM_y",
                "memory_usage_mb": 522.9,
            },
            {"memory_event": "started", "pid": 102, "process_kind": "inference"},
        ],
    )
    info = memory.load_proc_info([log])
    assert info[100].kind == "worker"
    assert info[101].kind == "job" and info[101].job_id == "AJ_x" and info[101].room_id == "RM_y"
    assert info[101].last_memory_mb == 522.9
    assert info[102].kind == "inference"


def test_load_proc_info_ignores_non_json_lines(tmp_path: Path) -> None:
    log = tmp_path / "plain.log"
    log.write_text(
        "INFO some plain text log line\n"
        '{"memory_event": "started", "pid": 5, "process_kind": "job"}\n'
        "another non-json line\n"
    )
    info = memory.load_proc_info([log])
    assert set(info) == {5} and info[5].kind == "job"


def test_discover_labels_captures_from_logs(tmp_path: Path) -> None:
    _touch(tmp_path / "memray-agent.py.100.bin")
    _touch(tmp_path / "memray-agent.py.100.bin.101")
    _touch(tmp_path / "memray-agent.py.100.bin.102")
    log = tmp_path / "agent.log"
    _write_log(
        log,
        [
            {"memory_event": "started", "pid": 100, "process_kind": "worker"},
            {"memory_event": "started", "pid": 101, "process_kind": "job", "job_id": "AJ_x"},
            {"memory_event": "started", "pid": 102, "process_kind": "inference"},
        ],
    )
    info = memory.load_proc_info([log])
    caps = {c.pid: c for c in memory.discover(tmp_path, info)}
    assert caps[100].kind == "worker"
    assert caps[101].kind == "job" and caps[101].label == "job pid=101  job=AJ_x"
    assert caps[102].kind == "inference"


def test_write_index_emits_group_headers_and_kinds(tmp_path: Path) -> None:
    parent_a = _touch(tmp_path / "memray-agent.py.100.bin")
    _touch(tmp_path / "memray-agent.py.100.bin.101")
    _touch(tmp_path / "memray-agent.py.200.bin")
    log = tmp_path / "agent.log"
    _write_log(
        log,
        [
            {"memory_event": "started", "pid": 100, "process_kind": "worker"},
            {"memory_event": "started", "pid": 101, "process_kind": "job", "job_id": "AJ_x"},
            {"memory_event": "started", "pid": 200, "process_kind": "worker"},
        ],
    )
    caps = memory.discover(tmp_path, memory.load_proc_info([log]))
    rendered = [(c, c.path.with_name(c.path.name + ".flamegraph.html")) for c in caps]
    rendered[0][1].write_text("<html></html>")

    index = memory._write_index(tmp_path, rendered)
    html = index.read_text()
    assert "<!--CAPTURES-->" not in html
    assert html.count('class="group"') == 2  # one per parent pid
    assert "job pid=101" in html and "job=AJ_x" in html
    assert parent_a.with_name(parent_a.name + ".flamegraph.html").name in html


# --- CLI surface ------------------------------------------------------------ #


def test_main_list_with_logs_labels_kinds(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _touch(tmp_path / "memray-agent.py.100.bin")
    _touch(tmp_path / "memray-agent.py.100.bin.101")
    log = tmp_path / "agent.log"
    _write_log(
        log,
        [
            {"memory_event": "started", "pid": 100, "process_kind": "worker"},
            {"memory_event": "started", "pid": 101, "process_kind": "job"},
        ],
    )
    rc = memory.main(["list", str(tmp_path), "--logs", str(log)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "worker pid=100" in out and "job pid=101" in out


def test_main_list_without_logs_warns(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _touch(tmp_path / "memray-agent.py.100.bin.101")
    rc = memory.main(["list", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "no --logs given" in out
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
