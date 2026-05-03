from __future__ import annotations

import asyncio
import os
import pathlib
import pickle
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import watchfiles
from typer.testing import CliRunner

from livekit.agents.cli import proto
from livekit.agents.cli.cli import _build_cli
from livekit.agents.cli.watcher import WatchServer
from livekit.agents.worker import AgentServer, ServerOptions


def _make_server() -> AgentServer:
    async def _fake_entrypoint(ctx: Any) -> None:
        pass

    opts = ServerOptions(entrypoint_fnc=_fake_entrypoint)
    return AgentServer.from_server_options(opts)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch) -> Any:
    # Remove iTerm2 carve-out for parsing-focused tests; the carve-out has its own tests.
    monkeypatch.delenv("TERM_PROGRAM", raising=False)
    return _build_cli(_make_server())


# ---------------------------------------------------------------------------
# Unit tests — new logic
# ---------------------------------------------------------------------------


class TestReloadDirParsing:
    """Scenarios covering how `--reload-dir` and LIVEKIT_RELOAD_DIRS land on CliArgs."""

    @patch("livekit.agents.cli.cli._run_worker")
    def test_default_no_flag_no_env(
        self, mock_run_worker: MagicMock, runner: CliRunner, app: Any
    ) -> None:
        result = runner.invoke(app, ["dev", "--no-reload"])
        assert result.exit_code == 0, result.output
        args: proto.CliArgs = mock_run_worker.call_args.kwargs["args"]
        assert args.reload_dirs == []

    @patch("livekit.agents.cli.cli._run_worker")
    def test_repeated_flag_accumulates(
        self,
        mock_run_worker: MagicMock,
        runner: CliRunner,
        app: Any,
        tmp_path: pathlib.Path,
    ) -> None:
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir()
        b.mkdir()
        result = runner.invoke(
            app,
            ["dev", "--no-reload", "--reload-dir", str(a), "--reload-dir", str(b)],
        )
        assert result.exit_code == 0, result.output
        args: proto.CliArgs = mock_run_worker.call_args.kwargs["args"]
        assert args.reload_dirs == [a.resolve(), b.resolve()]

    @patch("livekit.agents.cli.cli._run_worker")
    def test_env_var_pathsep_split(
        self,
        mock_run_worker: MagicMock,
        runner: CliRunner,
        app: Any,
        tmp_path: pathlib.Path,
    ) -> None:
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir()
        b.mkdir()
        env_value = os.pathsep.join([str(a), str(b)])
        result = runner.invoke(
            app,
            ["dev", "--no-reload"],
            env={"LIVEKIT_RELOAD_DIRS": env_value},
        )
        assert result.exit_code == 0, result.output
        args: proto.CliArgs = mock_run_worker.call_args.kwargs["args"]
        assert args.reload_dirs == [a.resolve(), b.resolve()]

    @patch("livekit.agents.cli.cli._run_worker")
    def test_flag_overrides_env(
        self,
        mock_run_worker: MagicMock,
        runner: CliRunner,
        app: Any,
        tmp_path: pathlib.Path,
    ) -> None:
        flag_dir = tmp_path / "flag"
        env_dir = tmp_path / "env"
        flag_dir.mkdir()
        env_dir.mkdir()
        result = runner.invoke(
            app,
            ["dev", "--no-reload", "--reload-dir", str(flag_dir)],
            env={"LIVEKIT_RELOAD_DIRS": str(env_dir)},
        )
        assert result.exit_code == 0, result.output
        args: proto.CliArgs = mock_run_worker.call_args.kwargs["args"]
        assert args.reload_dirs == [flag_dir.resolve()]

    @patch("livekit.agents.cli.cli._run_worker")
    def test_relative_path_resolved_to_absolute(
        self,
        mock_run_worker: MagicMock,
        runner: CliRunner,
        app: Any,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        sub = tmp_path / "sub"
        sub.mkdir()
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["dev", "--no-reload", "--reload-dir", "sub"])
        assert result.exit_code == 0, result.output
        args: proto.CliArgs = mock_run_worker.call_args.kwargs["args"]
        assert args.reload_dirs == [sub.resolve()]
        assert args.reload_dirs[0].is_absolute()

    @patch("livekit.agents.cli.cli._run_worker")
    def test_symlink_resolved(
        self,
        mock_run_worker: MagicMock,
        runner: CliRunner,
        app: Any,
        tmp_path: pathlib.Path,
    ) -> None:
        real = tmp_path / "real"
        link = tmp_path / "link"
        real.mkdir()
        link.symlink_to(real)
        result = runner.invoke(app, ["dev", "--no-reload", "--reload-dir", str(link)])
        assert result.exit_code == 0, result.output
        args: proto.CliArgs = mock_run_worker.call_args.kwargs["args"]
        # .resolve() collapses the symlink to the real path
        assert args.reload_dirs == [real.resolve()]

    @patch("livekit.agents.cli.cli._run_worker")
    def test_missing_path_flag_fails_fast(
        self,
        mock_run_worker: MagicMock,
        runner: CliRunner,
        app: Any,
        tmp_path: pathlib.Path,
    ) -> None:
        bogus = tmp_path / "does_not_exist"
        result = runner.invoke(app, ["dev", "--no-reload", "--reload-dir", str(bogus)])
        assert result.exit_code == 1
        assert "does not exist" in result.output
        mock_run_worker.assert_not_called()

    @patch("livekit.agents.cli.cli._run_worker")
    def test_missing_path_env_fails_fast(
        self,
        mock_run_worker: MagicMock,
        runner: CliRunner,
        app: Any,
        tmp_path: pathlib.Path,
    ) -> None:
        bogus = tmp_path / "does_not_exist"
        result = runner.invoke(
            app,
            ["dev", "--no-reload"],
            env={"LIVEKIT_RELOAD_DIRS": str(bogus)},
        )
        assert result.exit_code == 1
        assert "does not exist" in result.output
        mock_run_worker.assert_not_called()


class TestCliArgsPickle:
    """CliArgs crosses the watchfiles fork boundary; pickling must round-trip."""

    def test_default_round_trip(self) -> None:
        args = proto.CliArgs(log_level="DEBUG", url=None)
        restored = pickle.loads(pickle.dumps(args))
        assert restored.reload_dirs == []

    def test_non_empty_round_trip(self, tmp_path: pathlib.Path) -> None:
        paths = [tmp_path / "a", tmp_path / "b"]
        args = proto.CliArgs(log_level="DEBUG", url=None, reload_dirs=paths)
        restored = pickle.loads(pickle.dumps(args))
        assert restored.reload_dirs == paths


# ---------------------------------------------------------------------------
# Regression tests — old logic affected by the change
# ---------------------------------------------------------------------------


class TestRegressionDefaults:
    """Pin existing behavior so the additive change stays additive."""

    @patch("livekit.agents.cli.cli._run_worker")
    def test_dev_no_reload_carries_no_reload_dirs(
        self, mock_run_worker: MagicMock, runner: CliRunner, app: Any
    ) -> None:
        # No flag, no env -> reload_dirs stays empty (BC contract).
        result = runner.invoke(app, ["dev", "--no-reload"])
        assert result.exit_code == 0
        args: proto.CliArgs = mock_run_worker.call_args.kwargs["args"]
        assert args.reload_dirs == []
        assert args.reload is False
        assert args.devmode is True


class TestRegressionITerm2Carveout:
    """The iTerm2 force-disable path is unaffected and now warns when --reload-dir was set."""

    @patch("livekit.agents.cli.cli._run_worker")
    def test_iterm2_disables_reload(
        self,
        mock_run_worker: MagicMock,
        runner: CliRunner,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("TERM_PROGRAM", "iTerm.app")
        app = _build_cli(_make_server())
        result = runner.invoke(app, ["dev"])  # default --reload=True
        assert result.exit_code == 0
        args: proto.CliArgs = mock_run_worker.call_args.kwargs["args"]
        assert args.reload is False
        # The "Auto-reload is not supported" message goes through the AgentsConsole;
        # CliRunner's captured stdout does not include rich-styled markup, so we just
        # assert the worker was called single-process (no WatchServer constructed).
        mock_run_worker.assert_called_once()

    @patch("livekit.agents.cli.cli._run_worker")
    def test_iterm2_with_reload_dir_warns_and_runs_single_process(
        self,
        mock_run_worker: MagicMock,
        runner: CliRunner,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: pathlib.Path,
    ) -> None:
        monkeypatch.setenv("TERM_PROGRAM", "iTerm.app")
        extra = tmp_path / "extra"
        extra.mkdir()
        app = _build_cli(_make_server())
        result = runner.invoke(app, ["dev", "--reload-dir", str(extra)])
        assert result.exit_code == 0
        args: proto.CliArgs = mock_run_worker.call_args.kwargs["args"]
        # The reload_dirs still travel on CliArgs (so they survive a hypothetical
        # later re-enable), but reload is force-disabled and no WatchServer runs.
        assert args.reload is False
        assert args.reload_dirs == [extra.resolve()]
        mock_run_worker.assert_called_once()

    @patch("livekit.agents.cli.cli._run_worker")
    def test_no_reload_with_reload_dir_runs_single_process(
        self,
        mock_run_worker: MagicMock,
        runner: CliRunner,
        app: Any,
        tmp_path: pathlib.Path,
    ) -> None:
        extra = tmp_path / "extra"
        extra.mkdir()
        result = runner.invoke(app, ["dev", "--no-reload", "--reload-dir", str(extra)])
        assert result.exit_code == 0
        args: proto.CliArgs = mock_run_worker.call_args.kwargs["args"]
        assert args.reload is False
        assert args.reload_dirs == [extra.resolve()]
        mock_run_worker.assert_called_once()


# ---------------------------------------------------------------------------
# Watcher integration — path union and PythonFilter still in effect
# ---------------------------------------------------------------------------


class TestWatchServerPathUnion:
    """Pin the watcher's contract: auto-discovered + user paths, PythonFilter intact."""

    def test_run_unions_user_paths_and_keeps_python_filter(self, tmp_path: pathlib.Path) -> None:
        captured: dict[str, Any] = {}

        async def _fake_arun_process(*paths: Any, **kwargs: Any) -> int:
            captured["paths"] = paths
            captured["watch_filter"] = kwargs.get("watch_filter")
            captured["target"] = kwargs.get("target")
            return 0

        extra = tmp_path / "extra"
        extra.mkdir()
        cli_args = proto.CliArgs(
            log_level="DEBUG",
            url=None,
            devmode=True,
            reload=True,
            reload_dirs=[extra],
        )

        async def _go() -> None:
            loop = asyncio.get_running_loop()
            ws = WatchServer(
                worker_runner=lambda *a, **kw: None,
                server=_make_server(),
                main_file=tmp_path,
                cli_args=cli_args,
                loop=loop,
            )

            with patch("watchfiles.arun_process", side_effect=_fake_arun_process):
                run_task = loop.create_task(ws.run())
                # Wait for the fake arun_process to be called (it returns immediately,
                # which then triggers ws.aclose() inside the run-task).
                for _ in range(50):
                    await asyncio.sleep(0)
                    if "paths" in captured:
                        break
                # Make sure the run task settles cleanly within the test timeout.
                await asyncio.wait_for(run_task, timeout=2.0)

        asyncio.run(_go())

        assert "paths" in captured, "watchfiles.arun_process was never called"
        # User-supplied dir must appear in the path list passed to watchfiles.
        assert extra in captured["paths"], f"expected user dir {extra!r} in {captured['paths']!r}"
        # The Python filter must still be in effect — pin this so a future cleanup
        # cannot silently drop it and re-enable restart-storms on .git churn etc.
        assert isinstance(captured["watch_filter"], watchfiles.filters.PythonFilter)


class TestPythonFilterCoversIgnoredDirs:
    """Sanity-pin the negative case: changes that must not trigger reloads."""

    def test_filter_drops_non_python_and_ignored_dirs(self, tmp_path: pathlib.Path) -> None:
        f = watchfiles.filters.PythonFilter()
        change = watchfiles.Change.modified

        # Positive: a regular .py file inside a user-supplied dir is observed.
        assert f(change, str(tmp_path / "foo.py"))

        # Negative: non-.py files are dropped.
        assert not f(change, str(tmp_path / "notes.txt"))

        # Negative: ignored dirs (the source of restart storms) are dropped.
        assert not f(change, str(tmp_path / ".git" / "HEAD"))
        assert not f(change, str(tmp_path / ".venv" / "x.py"))
        assert not f(change, str(tmp_path / "__pycache__" / "x.cpython-311.pyc"))
        assert not f(change, str(tmp_path / "node_modules" / "x.py"))
