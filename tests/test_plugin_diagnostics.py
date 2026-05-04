from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest

import livekit.agents.diagnostics as diagnostics_mod
from livekit.agents.diagnostics import (
    DiagnosticCategory,
    DiagnosticCheck,
    DiagnosticContext,
    DiagnosticMode,
    DiagnosticReport,
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


class _FakeVersionInfo(tuple):
    def __new__(cls, major: int, minor: int, micro: int) -> _FakeVersionInfo:
        return super().__new__(cls, (major, minor, micro, "final", 0))

    @property
    def major(self) -> int:
        return self[0]

    @property
    def minor(self) -> int:
        return self[1]

    @property
    def micro(self) -> int:
        return self[2]


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


def test_diagnostic_report_plugins_are_schema_state_not_result_side_effect() -> None:
    report = DiagnosticReport(
        results=[],
        plugins=[
            {
                "title": "Schema Plugin",
                "version": "1.0.0",
                "package": "livekit.plugins.schema",
            }
        ],
    )

    payload = _report_dict(report)

    assert payload["plugins"] == [
        {
            "title": "Schema Plugin",
            "version": "1.0.0",
            "package": "livekit.plugins.schema",
        }
    ]


def test_diagnostic_result_details_redact_credential_like_keys() -> None:
    report = DiagnosticReport(
        results=[
            DiagnosticResult(
                id="test.redaction",
                category=DiagnosticCategory.ENVIRONMENT,
                severity=DiagnosticSeverity.ERROR,
                message="redaction test",
                details={
                    "api_key": "secret-key",
                    "authorization": "Bearer secret-token",
                    "credential": "secret-credential",
                    "connection_string": "postgres://user:pass@example.com/db",
                    "nested": {"password": "secret-password"},
                    "safe": "visible",
                },
            )
        ]
    )

    payload = _report_dict(report)
    details = payload["results"][0]["details"]

    assert details["api_key"] == "***"
    assert details["authorization"] == "***"
    assert details["credential"] == "***"
    assert details["connection_string"] == "***"
    assert details["nested"]["password"] == "***"
    assert details["safe"] == "visible"


def test_python_newer_than_tested_range_warns_without_blocking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(diagnostics_mod.sys, "version_info", _FakeVersionInfo(3, 15, 0))

    report = collect_diagnostics(_context())
    payload = _report_dict(report)

    python_result = next(
        result for result in payload["results"] if result["id"] == "python.version"
    )
    assert python_result["severity"] == "warn"
    assert report_exit_code(report) == 0


def test_python_below_supported_range_is_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(diagnostics_mod.sys, "version_info", _FakeVersionInfo(3, 9, 18))

    report = collect_diagnostics(_context())
    payload = _report_dict(report)

    python_result = next(
        result for result in payload["results"] if result["id"] == "python.version"
    )
    assert python_result["severity"] == "fatal"
    assert report_exit_code(report) == 1


def test_livekit_url_redacts_userinfo_credentials() -> None:
    report = collect_diagnostics(
        DiagnosticContext(
            env={
                "LIVEKIT_URL": "wss://api-key:api-secret@example.livekit.cloud",
                "LIVEKIT_API_KEY": "api-key",
                "LIVEKIT_API_SECRET": "api-secret",
            },
        )
    )
    payload = _report_dict(report)

    livekit_url = next(result for result in payload["results"] if result["id"] == "livekit.url")
    assert livekit_url["details"]["url"] == "wss://***:***@example.livekit.cloud"


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


def test_console_mode_invalid_livekit_url_is_non_blocking_warning() -> None:
    server = AgentServer()
    server.rtc_session(_entrypoint)

    report = collect_diagnostics(
        DiagnosticContext(
            mode="console",
            env={"LIVEKIT_URL": "not-a-url"},
            registered_plugins=(),
            server=server,
        )
    )
    payload = _report_dict(report)

    livekit_url = next(result for result in payload["results"] if result["id"] == "livekit.url")
    assert livekit_url["severity"] == "warn"


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


def test_start_mode_warns_on_float_non_positive_drain_timeout() -> None:
    server = _make_server()
    server._drain_timeout = 0.0

    report = collect_diagnostics(
        DiagnosticContext(
            mode="start",
            env={},
            registered_plugins=(),
            server=server,
        )
    )
    payload = _report_dict(report)

    assert any(
        result["id"] == "deployment.drain_timeout" and result["severity"] == "warn"
        for result in payload["results"]
    )
