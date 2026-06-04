from __future__ import annotations

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
    assert c.label == "worker pid=69962"


def test_child_filename_parses_with_parent_pid(tmp_path: Path) -> None:
    _touch(tmp_path / "memray-agent.py.69962.bin.69964")
    caps = memory.discover(tmp_path)
    assert len(caps) == 1
    c = caps[0]
    assert c.parent_pid == 69962 and c.child_pid == 69964
    assert c.is_parent is False
    assert "parent=69962" in c.label


def test_unrelated_files_are_ignored(tmp_path: Path) -> None:
    _touch(tmp_path / "notes.txt")
    _touch(tmp_path / "memray-agent.py.bin")  # missing pid
    _touch(tmp_path / "memray-agent.py.42.bin")  # valid
    caps = memory.discover(tmp_path)
    assert [c.path.name for c in caps] == ["memray-agent.py.42.bin"]


def test_discover_groups_children_after_parent_per_worker(tmp_path: Path) -> None:
    # two workers, each with two children — children must follow their parent
    _touch(tmp_path / "memray-agent.py.100.bin")
    _touch(tmp_path / "memray-agent.py.100.bin.101")
    _touch(tmp_path / "memray-agent.py.100.bin.102")
    _touch(tmp_path / "memray-agent.py.200.bin")
    _touch(tmp_path / "memray-agent.py.200.bin.201")

    caps = memory.discover(tmp_path)
    # worker 100 first, then its children, then worker 200, then its child
    assert [(c.parent_pid, c.child_pid) for c in caps] == [
        (100, None),
        (100, 101),
        (100, 102),
        (200, None),
        (200, 201),
    ]


def test_write_index_emits_group_headers_per_worker(tmp_path: Path) -> None:
    parent_a = _touch(tmp_path / "memray-agent.py.100.bin")
    _touch(tmp_path / "memray-agent.py.100.bin.101")
    _touch(tmp_path / "memray-agent.py.200.bin")
    caps = memory.discover(tmp_path)
    rendered = [(c, c.path.with_name(c.path.name + ".flamegraph.html")) for c in caps]
    rendered[0][1].write_text("<html></html>")  # pretend one render exists

    index = memory._write_index(tmp_path, rendered)
    html = index.read_text()
    assert index == tmp_path / "index.html"
    assert "<!--CAPTURES-->" not in html
    # one group header per distinct parent pid
    assert html.count('class="group"') == 2
    assert "worker pid=100" in html and "worker pid=200" in html
    # the rendered link is present
    assert parent_a.with_name(parent_a.name + ".flamegraph.html").name in html


def test_main_list_groups_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _touch(tmp_path / "memray-agent.py.100.bin")
    _touch(tmp_path / "memray-agent.py.100.bin.101")
    rc = memory.main(["list", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "2 capture" in out
    assert "worker pid=100" in out and "child pid=101" in out


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
