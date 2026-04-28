from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest
from typer.testing import CliRunner

from livekit.agents.cli.cli import _build_cli
from livekit.agents.diagnostics import PluginCapability, PluginDiagnosticInfo
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
