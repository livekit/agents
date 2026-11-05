from __future__ import annotations

import json
import multiprocessing
import os
import runpy
import subprocess
import sys
import types
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_IPC = Path(__file__).resolve().parents[1] / "livekit-agents-next" / "livekit" / "agents" / "ipc"


@pytest.fixture
def launch() -> types.ModuleType:
    import livekit.agents.ipc

    return livekit.agents.ipc


@pytest.mark.parametrize("inspection_failure", [False, True])
def test_forkserver_refuses_to_freeze_an_unsafe_or_uninspectable_image(
    monkeypatch: pytest.MonkeyPatch, inspection_failure: bool
) -> None:
    frozen = []
    monkeypatch.setattr("gc.freeze", lambda: frozen.append(True))
    monkeypatch.setattr("gc.collect", lambda: 0)
    if inspection_failure:

        def unavailable(*_: object) -> None:
            raise OSError("thread inventory unavailable")

        monkeypatch.setattr("os.listdir", unavailable)
        monkeypatch.setattr("psutil.Process.num_threads", unavailable)
        expected = "cannot inspect forkserver threads"
    else:
        monkeypatch.setattr("os.listdir", lambda _: ["main", "background"])

        def disappeared(*_: object) -> None:
            raise OSError("thread name disappeared")

        monkeypatch.setattr("builtins.open", disappeared)
        expected = "background threads"

    # the bootstrap's preload phase is a no-op here: only its final image check runs
    monkeypatch.setenv("LIVEKIT_AGENTS_FORKSERVER_PRELOAD_MODULES", "")
    monkeypatch.setattr("livekit.agents.ipc.load_preload_modules", lambda _: None)
    with pytest.raises(RuntimeError, match=expected):
        runpy.run_module("livekit.agents.ipc._preload_forkserver", run_name="__forkserver__")
    assert frozen == []


def test_preload_import_failure_preserves_module_and_cause(
    launch: types.ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "preload_first.py").write_text(
        "import builtins\nbuiltins.preload_order.append('first')"
    )
    (tmp_path / "preload_second.py").write_text(
        "import builtins\nbuiltins.preload_order.append('second')\n"
        "raise ImportError('missing weights')"
    )
    (tmp_path / "preload_third.py").write_text(
        "raise AssertionError('must not import after failure')"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    order: list[str] = []
    monkeypatch.setattr("builtins.preload_order", order, raising=False)
    monkeypatch.delitem(sys.modules, "preload_first", raising=False)
    try:
        with pytest.raises(RuntimeError, match="preload_second.*missing weights") as caught:
            launch.load_preload_modules(["preload_first", "preload_second", "preload_third"])
        assert isinstance(caught.value.__cause__, ImportError)
        assert order == ["first", "second"]
    finally:
        sys.modules.pop("preload_first", None)


@pytest.mark.skipif(sys.platform == "win32", reason="Windows uses spawn")
def test_running_forkserver_refuses_different_preloads_before_mutating_environment(
    launch: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    from multiprocessing.forkserver import _forkserver

    monkeypatch.setattr(_forkserver, "_forkserver_pid", None)
    monkeypatch.setattr(_forkserver, "_preload_modules", [])
    monkeypatch.setenv(launch._FORKSERVER_PRELOAD_MODULES, "")
    context = multiprocessing.get_context("spawn")
    launch.configure_forkserver(context, ["application.catalog"])
    monkeypatch.setattr(_forkserver, "_forkserver_pid", 1)
    launch.configure_forkserver(context, ["application.catalog"])
    with pytest.raises(RuntimeError, match="running forkserver has different preload_modules"):
        launch.configure_forkserver(context, ["application.other_catalog"])
    with pytest.raises(ValueError, match="invalid preload module name"):
        launch.configure_forkserver(context, ["application.catalog:injected"])
    assert _forkserver._preload_modules == [
        "livekit.agents.ipc._preload_forkserver",
        "application.catalog",
    ]
    assert os.environ[launch._FORKSERVER_PRELOAD_MODULES] == "application.catalog"


@pytest.mark.skipif(sys.platform != "linux", reason="the forkserver is used on Linux only")
def test_forkserver_freezes_its_preloads_before_forking_jobs() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "tests.utils.forkserver_preload"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    outcome = json.loads(result.stdout)
    assert outcome["freeze_count"] > 0
    assert outcome["child_cycle_collected"]
