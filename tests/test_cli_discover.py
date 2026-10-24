from __future__ import annotations

import importlib
import sys
import textwrap
from pathlib import Path

import pytest

from livekit.agents._exceptions import CLIError
from livekit.agents.cli.discover import (
    ModuleData,
    get_app_name,
    get_default_path,
    get_import_data,
    get_module_data_from_path,
)

pytestmark = pytest.mark.unit

_SERVER_SRC = """
from livekit.agents import AgentServer

{names} = AgentServer()
"""


def _write(path: Path, content: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content))
    # a FileFinder cached for this directory may not see the new file otherwise
    importlib.invalidate_caches()
    return path


def _server_module(path: Path, *names: str) -> Path:
    return _write(path, _SERVER_SRC.format(names=" = ".join(names)))


@pytest.fixture
def isolated_imports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Restore sys.path / sys.modules so modules written under tmp_path don't leak."""
    monkeypatch.setattr(sys, "path", list(sys.path))
    before = set(sys.modules)
    yield
    root = tmp_path.resolve()
    for name in set(sys.modules) - before:
        # only drop the modules this test wrote, other lazy imports must stay cached
        file = getattr(sys.modules[name], "__file__", None)
        if file and Path(file).resolve().is_relative_to(root):
            del sys.modules[name]


class TestGetDefaultPath:
    @pytest.mark.parametrize(
        "relpath",
        ["main.py", "app.py", "agent.py", "app/main.py", "app/app.py", "app/agent.py"],
    )
    def test_finds_each_default_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relpath: str
    ):
        monkeypatch.chdir(tmp_path)
        _write(tmp_path / relpath)

        assert get_default_path() == Path(relpath)

    def test_prefers_earlier_candidates(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.chdir(tmp_path)
        _write(tmp_path / "agent.py")
        _write(tmp_path / "app" / "main.py")
        assert get_default_path() == Path("agent.py")

        _write(tmp_path / "main.py")
        assert get_default_path() == Path("main.py")

    def test_ignores_directories_with_default_names(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "main.py").mkdir()

        with pytest.raises(CLIError, match="default file"):
            get_default_path()

    def test_raises_when_nothing_found(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.chdir(tmp_path)
        _write(tmp_path / "other.py")

        with pytest.raises(CLIError, match="default file"):
            get_default_path()


class TestGetModuleDataFromPath:
    def test_standalone_file(self, tmp_path: Path):
        agent = _write(tmp_path / "myagent.py")

        data = get_module_data_from_path(agent)

        assert data.module_import_str == "myagent"
        assert data.extra_sys_path == tmp_path.resolve()
        assert data.module_paths == [agent.resolve()]

    def test_file_inside_package(self, tmp_path: Path):
        _write(tmp_path / "pkg" / "__init__.py")
        agent = _write(tmp_path / "pkg" / "agent.py")

        data = get_module_data_from_path(agent)

        assert data.module_import_str == "pkg.agent"
        assert data.extra_sys_path == tmp_path.resolve()
        assert data.module_paths == [(tmp_path / "pkg").resolve(), agent.resolve()]

    def test_file_inside_nested_packages(self, tmp_path: Path):
        _write(tmp_path / "outer" / "__init__.py")
        _write(tmp_path / "outer" / "inner" / "__init__.py")
        agent = _write(tmp_path / "outer" / "inner" / "agent.py")

        data = get_module_data_from_path(agent)

        assert data.module_import_str == "outer.inner.agent"
        assert data.extra_sys_path == tmp_path.resolve()
        assert data.module_paths == [
            (tmp_path / "outer").resolve(),
            (tmp_path / "outer" / "inner").resolve(),
            agent.resolve(),
        ]

    def test_stops_at_first_directory_without_init(self, tmp_path: Path):
        # "top" has no __init__.py, so it must not become part of the import path
        _write(tmp_path / "top" / "pkg" / "__init__.py")
        agent = _write(tmp_path / "top" / "pkg" / "agent.py")

        data = get_module_data_from_path(agent)

        assert data.module_import_str == "pkg.agent"
        assert data.extra_sys_path == (tmp_path / "top").resolve()

    def test_init_file_resolves_to_its_package(self, tmp_path: Path):
        init = _write(tmp_path / "pkg" / "__init__.py")

        data = get_module_data_from_path(init)

        assert data.module_import_str == "pkg"
        assert data.extra_sys_path == tmp_path.resolve()
        assert data.module_paths == [(tmp_path / "pkg").resolve()]

    def test_package_directory(self, tmp_path: Path):
        _write(tmp_path / "pkg" / "__init__.py")

        data = get_module_data_from_path(tmp_path / "pkg")

        assert data.module_import_str == "pkg"
        assert data.extra_sys_path == tmp_path.resolve()
        assert data.module_paths == [(tmp_path / "pkg").resolve()]


class TestGetAppName:
    @staticmethod
    def _mod_data(module: str, root: Path) -> ModuleData:
        return ModuleData(module_import_str=module, extra_sys_path=root, module_paths=[])

    @pytest.mark.parametrize("name", ["app", "server", "agent"])
    def test_finds_preferred_names(
        self, tmp_path: Path, isolated_imports, monkeypatch: pytest.MonkeyPatch, name: str
    ):
        monkeypatch.syspath_prepend(str(tmp_path))
        _server_module(tmp_path / "disc_preferred.py", name)

        assert get_app_name(mod_data=self._mod_data("disc_preferred", tmp_path)) == name

    def test_preferred_name_order_is_app_server_agent(
        self, tmp_path: Path, isolated_imports, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.syspath_prepend(str(tmp_path))
        _server_module(tmp_path / "disc_order.py", "agent", "server", "app")

        assert get_app_name(mod_data=self._mod_data("disc_order", tmp_path)) == "app"

    def test_preferred_name_wins_over_alphabetically_earlier_name(
        self, tmp_path: Path, isolated_imports, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.syspath_prepend(str(tmp_path))
        _write(
            tmp_path / "disc_pref_wins.py",
            """
            from livekit.agents import AgentServer

            alpha = AgentServer()
            server = AgentServer()
            """,
        )

        assert get_app_name(mod_data=self._mod_data("disc_pref_wins", tmp_path)) == "server"

    def test_falls_back_to_any_agent_server(
        self, tmp_path: Path, isolated_imports, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.syspath_prepend(str(tmp_path))
        _server_module(tmp_path / "disc_fallback.py", "my_custom_server")

        assert (
            get_app_name(mod_data=self._mod_data("disc_fallback", tmp_path)) == "my_custom_server"
        )

    def test_ignores_preferred_name_that_is_not_an_agent_server(
        self, tmp_path: Path, isolated_imports, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.syspath_prepend(str(tmp_path))
        _write(
            tmp_path / "disc_not_server.py",
            """
            from livekit.agents import AgentServer

            app = "not a server"
            real = AgentServer()
            """,
        )

        assert get_app_name(mod_data=self._mod_data("disc_not_server", tmp_path)) == "real"

    def test_raises_when_no_agent_server(
        self, tmp_path: Path, isolated_imports, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.syspath_prepend(str(tmp_path))
        _write(tmp_path / "disc_none.py", "app = object()\n")

        with pytest.raises(CLIError, match="AgentServer"):
            get_app_name(mod_data=self._mod_data("disc_none", tmp_path))

    def test_reraises_import_errors(
        self, tmp_path: Path, isolated_imports, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.syspath_prepend(str(tmp_path))

        with pytest.raises(ImportError):
            get_app_name(mod_data=self._mod_data("disc_does_not_exist", tmp_path))

    def test_reraises_value_errors(self, tmp_path: Path, isolated_imports):
        # importlib rejects an empty module name with ValueError, not ImportError
        with pytest.raises(ValueError):
            get_app_name(mod_data=self._mod_data("", tmp_path))


class TestGetImportData:
    def test_raises_for_missing_path(self, tmp_path: Path):
        missing = tmp_path / "nope.py"

        with pytest.raises(CLIError, match="does not exist"):
            get_import_data(path=missing)

    def test_resolves_explicit_path(self, tmp_path: Path, isolated_imports):
        agent = _server_module(tmp_path / "disc_explicit.py", "server")

        data = get_import_data(path=agent)

        assert data.app_name == "server"
        assert data.import_string == "disc_explicit:server"
        assert data.module_data.module_import_str == "disc_explicit"

    def test_adds_extra_sys_path_first(self, tmp_path: Path, isolated_imports):
        agent = _server_module(tmp_path / "disc_syspath.py", "server")

        get_import_data(path=agent)

        assert sys.path[0] == str(tmp_path.resolve())

    def test_package_module_import_string(self, tmp_path: Path, isolated_imports):
        _write(tmp_path / "disc_pkg" / "__init__.py")
        agent = _server_module(tmp_path / "disc_pkg" / "agent.py", "app")

        data = get_import_data(path=agent)

        assert data.import_string == "disc_pkg.agent:app"

    def test_uses_default_path_when_none_given(
        self, tmp_path: Path, isolated_imports, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.chdir(tmp_path)
        # "agent" is a common module name, a cached one would shadow tmp_path/agent.py
        monkeypatch.delitem(sys.modules, "agent", raising=False)
        _server_module(tmp_path / "agent.py", "server")

        data = get_import_data()

        assert data.import_string == "agent:server"

    def test_default_app_dir_without_init(
        self, tmp_path: Path, isolated_imports, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delitem(sys.modules, "main", raising=False)
        _server_module(tmp_path / "app" / "main.py", "server")

        data = get_import_data()

        assert data.import_string == "main:server"
        assert sys.path[0] == str((tmp_path / "app").resolve())

    def test_default_app_dir_as_package(
        self, tmp_path: Path, isolated_imports, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delitem(sys.modules, "app", raising=False)
        monkeypatch.delitem(sys.modules, "app.main", raising=False)
        _write(tmp_path / "app" / "__init__.py")
        _server_module(tmp_path / "app" / "main.py", "server")

        data = get_import_data()

        assert data.import_string == "app.main:server"
        assert sys.path[0] == str(tmp_path.resolve())

    def test_no_default_file_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.chdir(tmp_path)

        with pytest.raises(CLIError, match="default file"):
            get_import_data()
