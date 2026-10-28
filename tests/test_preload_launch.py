from __future__ import annotations

import importlib.util
import multiprocessing
import os
import runpy
import sys
import types
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_IPC = Path(__file__).resolve().parents[1] / "livekit-agents-next" / "livekit" / "agents" / "ipc"


@pytest.fixture
def launch(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    # Exercise the OS launch adapter without importing unrelated provider bindings.
    spec = importlib.util.spec_from_file_location("preload_launch", _IPC / "launch.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


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

    with pytest.raises(RuntimeError, match=expected):
        runpy.run_path(str(_IPC / "_preload_freeze.py"))
    assert frozen == []


@pytest.mark.parametrize("exception_type", [ImportError, SystemExit])
def test_preload_import_failure_preserves_module_and_cause(
    launch: types.ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exception_type: type[BaseException],
) -> None:
    (tmp_path / "preload_first.py").write_text(
        "import builtins\nbuiltins.preload_order.append('first')"
    )
    (tmp_path / "preload_second.py").write_text(
        "import builtins\nbuiltins.preload_order.append('second')\n"
        f"raise {exception_type.__name__}('missing weights')"
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
        assert isinstance(caught.value.__cause__, exception_type)
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


@pytest.mark.parametrize("callable_object", [False, True])
def test_async_setup_is_rejected_before_a_child_is_allocated(
    launch: types.ModuleType, callable_object: bool
) -> None:
    async def setup(_: object) -> None:
        raise AssertionError("asynchronous setup must not run")

    class AsyncSetup:
        async def __call__(self, _: object) -> None:
            raise AssertionError("asynchronous setup must not run")

    with pytest.raises(TypeError, match="setup_fnc must be a synchronous callable"):
        launch.validate_setup_fnc(AsyncSetup() if callable_object else setup)
