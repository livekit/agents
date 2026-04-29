from __future__ import annotations

import builtins
import json
from collections.abc import Iterator
from typing import Any

import pytest
import typer
from typer.testing import CliRunner

from livekit.agents.cli.cli import _apply_cli_server_options, _build_cli, _run_preflight
from livekit.agents.diagnostics import (
    DiagnosticCategory,
    DiagnosticCheck,
    DiagnosticContext,
    DiagnosticResult,
    DiagnosticSeverity,
    PluginCapability,
    PluginDiagnosticInfo,
)
from livekit.agents.plugin import Plugin
from livekit.agents.worker import AgentServer


async def _entrypoint(ctx: Any) -> None:
    pass


def _make_server(*, registered: bool = True) -> AgentServer:
    server = AgentServer()
    if registered:
        server.rtc_session(_entrypoint)
    return server


class MissingEnvPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(
            title="Missing Env Plugin",
            version="1.2.3",
            package="livekit.plugins.missing_env",
        )

    def diagnostic_info(self) -> PluginDiagnosticInfo:
        return PluginDiagnosticInfo(
            capabilities=[PluginCapability.STT],
            required_env_vars=["MISSING_ENV_PLUGIN_API_KEY"],
        )


class DeepCheckPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(
            title="Deep Check Plugin",
            version="1.2.3",
            package="livekit.plugins.deep_check",
        )

    def diagnostic_info(self) -> PluginDiagnosticInfo:
        return PluginDiagnosticInfo()

    def diagnostics(self) -> list[DiagnosticCheck]:
        def run(ctx: DiagnosticContext) -> DiagnosticResult:
            assert ctx.deep
            return DiagnosticResult(
                id="plugins.deep_check.self_test",
                category=DiagnosticCategory.PLUGIN,
                severity=DiagnosticSeverity.OK,
                message="deep check ran",
                details={"plugin": self.package},
            )

        return [
            DiagnosticCheck(
                id="plugins.deep_check.self_test",
                category=DiagnosticCategory.PLUGIN,
                run=run,
            )
        ]


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture(autouse=True)
def restore_plugins() -> Iterator[None]:
    previous = list(Plugin.registered_plugins)
    Plugin.registered_plugins.clear()
    yield
    Plugin.registered_plugins[:] = previous


@pytest.fixture(autouse=True)
def clear_livekit_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"):
        monkeypatch.delenv(name, raising=False)


def _valid_env() -> dict[str, str]:
    return {
        "LIVEKIT_URL": "wss://diagnostics-test.livekit.cloud",
        "LIVEKIT_API_KEY": "api-key",
        "LIVEKIT_API_SECRET": "api-secret",
    }


def test_doctor_reports_missing_env(runner: CliRunner) -> None:
    app = _build_cli(_make_server())

    result = runner.invoke(app, ["doctor", "--json"], env={})

    assert result.exit_code == 1
    report = json.loads(result.output)
    assert any(
        result["id"] == "livekit.credentials"
        and set(result["details"]["missing"])
        == {"LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"}
        for result in report["results"]
    )


def test_doctor_reports_invalid_url(runner: CliRunner) -> None:
    app = _build_cli(_make_server())

    result = runner.invoke(
        app,
        ["doctor", "--json"],
        env={
            **_valid_env(),
            "LIVEKIT_URL": "not-a-url",
        },
    )

    assert result.exit_code == 1
    report = json.loads(result.output)
    assert any(
        result["id"] == "livekit.url"
        and result["severity"] == "error"
        and "ws:// or wss://" in result["message"]
        for result in report["results"]
    )


def test_doctor_reports_unregistered_entrypoint(runner: CliRunner) -> None:
    app = _build_cli(_make_server(registered=False))

    result = runner.invoke(app, ["doctor"], env=_valid_env())

    assert result.exit_code == 1
    assert "entrypoint" in result.output.lower()
    assert "registered" in result.output.lower()


def test_doctor_json_outputs_report_schema_and_success_exit_code(runner: CliRunner) -> None:
    app = _build_cli(_make_server())

    result = runner.invoke(app, ["doctor", "--json"], env=_valid_env())

    assert result.exit_code == 0
    report = json.loads(result.output)
    assert report["schema_version"] == 1
    assert report["mode"] == "doctor"
    assert report["exit_code"] == 0
    assert report["summary"]["errors"] == 0
    assert isinstance(report["plugins"], list)
    assert isinstance(report["results"], list)
    assert all("id" in result for result in report["results"])


def test_doctor_strict_treats_warnings_as_failures(runner: CliRunner) -> None:
    Plugin.register_plugin(MissingEnvPlugin())
    app = _build_cli(_make_server())

    non_strict = runner.invoke(app, ["doctor"], env=_valid_env())
    strict = runner.invoke(app, ["doctor", "--strict"], env=_valid_env())

    assert non_strict.exit_code == 0
    assert strict.exit_code == 1
    assert "plugin" in strict.output


def test_doctor_online_runs_safe_livekit_tcp_check(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Socket:
        def __enter__(self) -> _Socket:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    calls: list[tuple[tuple[str, int], float]] = []

    def create_connection(address: tuple[str, int], timeout: float) -> _Socket:
        calls.append((address, timeout))
        return _Socket()

    monkeypatch.setattr(
        "livekit.agents.diagnostics.socket.create_connection",
        create_connection,
    )
    app = _build_cli(_make_server())

    result = runner.invoke(app, ["doctor", "--online", "--json"], env=_valid_env())

    assert result.exit_code == 0
    assert calls == [(("diagnostics-test.livekit.cloud", 443), 3)]
    report = json.loads(result.output)
    assert any(
        item["id"] == "network.livekit_tcp" and item["severity"] == "ok"
        for item in report["results"]
    )


def test_doctor_deep_runs_plugin_declared_checks(runner: CliRunner) -> None:
    Plugin.register_plugin(DeepCheckPlugin())
    app = _build_cli(_make_server())

    result = runner.invoke(app, ["doctor", "--deep", "--json"], env=_valid_env())

    assert result.exit_code == 0
    report = json.loads(result.output)
    assert any(
        item["id"] == "plugins.deep_check.self_test" and item["severity"] == "ok"
        for item in report["results"]
    )


def test_preflight_text_console_skips_audio_import(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in _valid_env().items():
        monkeypatch.setenv(name, value)

    original_import = builtins.__import__

    def import_guard(name: str, *args: object, **kwargs: object) -> object:
        if name == "sounddevice":
            raise AssertionError("text console preflight should not import sounddevice")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_guard)

    _run_preflight(
        server=_make_server(),
        mode="console",
        console=None,
        console_audio=False,
    )


def test_preflight_audio_console_blocks_missing_sounddevice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name, value in _valid_env().items():
        monkeypatch.setenv(name, value)

    original_import = builtins.__import__

    def import_guard(name: str, *args: object, **kwargs: object) -> object:
        if name == "sounddevice":
            raise ImportError("missing sounddevice")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_guard)

    with pytest.raises(typer.Exit) as exc_info:
        _run_preflight(
            server=_make_server(),
            mode="console",
            console=None,
            console_audio=True,
        )

    assert exc_info.value.exit_code == 1


def test_cli_server_options_ignore_empty_strings() -> None:
    server = AgentServer(
        ws_url="wss://configured.livekit.cloud",
        api_key="configured-key",
        api_secret="configured-secret",
    )

    _apply_cli_server_options(server, url="", api_key="", api_secret="")

    assert server._ws_url == "wss://configured.livekit.cloud"
    assert server._api_key == "configured-key"
    assert server._api_secret == "configured-secret"
