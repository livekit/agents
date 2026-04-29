from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest

from livekit.agents.diagnostics import (
    DiagnosticCategory,
    DiagnosticCheck,
    DiagnosticContext,
    DiagnosticMode,
    DiagnosticResult,
    DiagnosticSeverity,
    PluginCapability,
    PluginDiagnosticInfo,
    collect_diagnostics,
    diagnostic_report_to_json,
    report_exit_code,
)
from livekit.agents.plugin import Plugin
from livekit.agents.worker import AgentServer


async def _entrypoint(ctx: Any) -> None:
    pass


def _make_server() -> AgentServer:
    server = AgentServer(
        ws_url="wss://diagnostics-test.livekit.cloud",
        api_key="api-key",
        api_secret="api-secret",
    )
    server.rtc_session(_entrypoint)
    return server


def _context(
    *,
    mode: DiagnosticMode = "doctor",
    strict: bool = False,
    deep: bool = False,
    env: dict[str, str] | None = None,
) -> DiagnosticContext:
    return DiagnosticContext(
        mode=mode,
        strict=strict,
        deep=deep,
        env=env or {},
        registered_plugins=tuple(Plugin.registered_plugins),
        server=_make_server(),
    )


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

    def diagnostics(self) -> list[DiagnosticCheck]:
        def run(ctx: DiagnosticContext) -> DiagnosticResult:
            assert isinstance(ctx, DiagnosticContext)
            return DiagnosticResult(
                id="plugins.missing_env.self_test",
                category=DiagnosticCategory.PLUGIN,
                severity=DiagnosticSeverity.WARN,
                message="optional provider token is not configured",
                details={"plugin": self.package},
            )

        return [
            DiagnosticCheck(
                id="plugins.missing_env.self_test",
                category=DiagnosticCategory.PLUGIN,
                run=run,
            )
        ]


class ExplodingPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(
            title="Exploding Plugin",
            version="9.9.9",
            package="livekit.plugins.exploding",
        )

    def diagnostic_info(self) -> PluginDiagnosticInfo:
        return PluginDiagnosticInfo()

    def diagnostics(self) -> list[DiagnosticCheck]:
        raise RuntimeError("provider self-test exploded")


class LegacyPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(
            title="Legacy Plugin",
            version="0.1.0",
            package="livekit.plugins.legacy",
        )


@pytest.fixture(autouse=True)
def restore_plugins() -> Iterator[None]:
    previous = list(Plugin.registered_plugins)
    Plugin.registered_plugins.clear()
    yield
    Plugin.registered_plugins[:] = previous


def _report_dict(report: Any) -> dict[str, Any]:
    return json.loads(diagnostic_report_to_json(report))


def test_diagnostic_report_to_json_schema_and_exit_code() -> None:
    Plugin.register_plugin(MissingEnvPlugin())

    report = collect_diagnostics(_context())
    payload = _report_dict(report)
    serialized = json.dumps(payload)

    assert payload["schema_version"] == 1
    assert payload["mode"] == "doctor"
    assert payload["exit_code"] == 0
    assert isinstance(payload["results"], list)
    assert all("id" in result for result in payload["results"])
    assert report_exit_code(report) == 0
    assert "livekit.plugins.missing_env" in serialized
    assert "MISSING_ENV_PLUGIN_API_KEY" in serialized


def test_strict_mode_makes_warnings_fail() -> None:
    Plugin.register_plugin(MissingEnvPlugin())

    report = collect_diagnostics(_context(strict=True))
    payload = _report_dict(report)

    assert any(result["severity"] == "warn" for result in payload["results"])
    assert payload["exit_code"] == 1
    assert report_exit_code(report) == 1


def test_plugin_diagnostics_exception_becomes_plugin_error_without_crashing() -> None:
    Plugin.register_plugin(ExplodingPlugin())

    report = collect_diagnostics(_context(deep=True))
    payload = _report_dict(report)
    serialized = json.dumps(payload)

    assert any(result["severity"] == "error" for result in payload["results"])
    assert report_exit_code(report) == 1
    assert "livekit.plugins.exploding" in serialized
    assert "provider self-test exploded" in serialized


def test_plugin_without_diagnostic_metadata_does_not_warn_or_fail_strict() -> None:
    Plugin.register_plugin(LegacyPlugin())

    report = collect_diagnostics(_context(strict=True))
    payload = _report_dict(report)

    assert report_exit_code(report) == 0
    assert all(result["severity"] != "warn" for result in payload["results"])
    assert all("legacy.metadata" not in result["id"] for result in payload["results"])
    assert any(plugin["package"] == "livekit.plugins.legacy" for plugin in payload["plugins"])


def test_console_mode_missing_livekit_credentials_is_non_blocking_warning() -> None:
    server = AgentServer()
    server.rtc_session(_entrypoint)

    report = collect_diagnostics(
        DiagnosticContext(
            mode="console",
            env={},
            registered_plugins=(),
            server=server,
        )
    )
    payload = _report_dict(report)

    credentials = next(
        result for result in payload["results"] if result["id"] == "livekit.credentials"
    )
    assert credentials["severity"] == "warn"


def test_worker_modes_missing_livekit_credentials_remain_fatal() -> None:
    server = AgentServer()
    server.rtc_session(_entrypoint)

    report = collect_diagnostics(
        DiagnosticContext(
            mode="dev",
            env={},
            registered_plugins=(),
            server=server,
        )
    )
    payload = _report_dict(report)

    credentials = next(
        result for result in payload["results"] if result["id"] == "livekit.credentials"
    )
    assert credentials["severity"] == "fatal"
    assert report_exit_code(report) == 1
