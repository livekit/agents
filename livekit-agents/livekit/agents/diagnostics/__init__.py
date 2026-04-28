from __future__ import annotations

import json
import os
import socket
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal
from urllib.parse import urlparse

from rich.table import Table

DiagnosticMode = Literal["doctor", "console", "dev", "start", "connect"]


class DiagnosticSeverity(str, Enum):
    """Severity levels returned by first-run diagnostics."""

    OK = "ok"
    WARN = "warn"
    ERROR = "error"
    FATAL = "fatal"


class DiagnosticCategory(str, Enum):
    """High-level area covered by a diagnostic result."""

    ENVIRONMENT = "environment"
    LIVEKIT = "livekit"
    AUDIO = "audio"
    PLUGIN = "plugin"
    DOWNLOADS = "downloads"
    NETWORK = "network"
    DEPLOYMENT = "deployment"


class PluginCapability(str, Enum):
    """Capabilities that a LiveKit Agents plugin can advertise to diagnostics."""

    LLM = "llm"
    STT = "stt"
    TTS = "tts"
    REALTIME = "realtime"
    VAD = "vad"
    AVATAR = "avatar"
    NOISE_CANCELLATION = "noise_cancellation"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class PluginDiagnosticInfo:
    """Static diagnostic metadata exposed by a plugin without side effects."""

    required_env_vars: Sequence[str] = ()
    optional_env_vars: Sequence[str] = ()
    capabilities: Sequence[PluginCapability | str] = ()
    downloadable_files: Sequence[str] = ()
    docs_url: str | None = None
    notes: str | None = None


@dataclass(frozen=True)
class DiagnosticContext:
    """Inputs available to diagnostics for one CLI mode or preflight run."""

    mode: DiagnosticMode = "doctor"
    online: bool = False
    deep: bool = False
    strict: bool = False
    env: Mapping[str, str] = field(default_factory=lambda: os.environ.copy())
    registered_plugins: Sequence[Any] = ()
    server: Any | None = None
    input_device: str | int | None = None
    output_device: str | int | None = None


@dataclass(frozen=True)
class DiagnosticResult:
    """Single diagnostic finding with a user-facing message and optional fix."""

    id: str
    category: DiagnosticCategory | str
    severity: DiagnosticSeverity | str
    message: str
    fix_hint: str | None = None
    docs_url: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        category = self.category.value if isinstance(self.category, Enum) else self.category
        severity = self.severity.value if isinstance(self.severity, Enum) else self.severity
        return {
            "id": self.id,
            "category": category,
            "severity": severity,
            "message": self.message,
            "fix_hint": self.fix_hint,
            "docs_url": self.docs_url,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class DiagnosticCheck:
    """Callable deep diagnostic check provided by a plugin."""

    id: str
    category: DiagnosticCategory
    run: Callable[[DiagnosticContext], DiagnosticResult]


@dataclass(frozen=True)
class DiagnosticReport:
    """Collected diagnostics and rendering metadata for a single run."""

    results: Sequence[DiagnosticResult]
    mode: DiagnosticMode = "doctor"
    strict: bool = False
    online: bool = False
    deep: bool = False

    def to_dict(self) -> dict[str, Any]:
        result_dicts = [result.to_dict() for result in self.results]
        counts = {
            severity.value: sum(
                1 for result in self.results if _severity_value(result.severity) == severity.value
            )
            for severity in DiagnosticSeverity
        }
        plugins: list[dict[str, Any]] = []
        for result in self.results:
            if result.id == "plugins.registered":
                raw_plugins = result.details.get("plugins", [])
                if isinstance(raw_plugins, list):
                    plugins = [plugin for plugin in raw_plugins if isinstance(plugin, dict)]
                break

        return {
            "schema_version": 1,
            "mode": self.mode,
            "strict": self.strict,
            "online": self.online,
            "deep": self.deep,
            "exit_code": report_exit_code(self),
            "summary": {
                "ok": counts[DiagnosticSeverity.OK.value],
                "warnings": counts[DiagnosticSeverity.WARN.value],
                "errors": counts[DiagnosticSeverity.ERROR.value],
                "fatal": counts[DiagnosticSeverity.FATAL.value],
                "exit_code": report_exit_code(self),
            },
            "plugins": plugins,
            "results": result_dicts,
        }


_LIVEKIT_REQUIRED_MODES = {"console", "dev", "start", "connect"}


def collect_diagnostics(context: DiagnosticContext) -> DiagnosticReport:
    results = [
        _check_python_version(),
        _check_entrypoint(context),
        *_check_livekit_credentials(context),
        *_check_plugins(context),
    ]

    if context.mode == "console":
        results.extend(_check_audio_devices(context))

    if context.mode == "start":
        results.extend(_check_deployment(context))

    if context.online:
        results.append(_check_livekit_tcp_connectivity(context))

    return DiagnosticReport(
        results=results,
        mode=context.mode,
        strict=context.strict,
        online=context.online,
        deep=context.deep,
    )


def diagnostic_report_to_json(report: DiagnosticReport) -> str:
    return json.dumps(report.to_dict(), default=str, sort_keys=True)


def report_exit_code(report: DiagnosticReport) -> int:
    severities = {_severity_value(result.severity) for result in report.results}
    if DiagnosticSeverity.FATAL.value in severities or DiagnosticSeverity.ERROR.value in severities:
        return 1
    if report.strict and DiagnosticSeverity.WARN.value in severities:
        return 1
    return 0


def build_diagnostics_table(report: DiagnosticReport) -> Table:
    table = Table(title="LiveKit Agents diagnostics")
    table.add_column("Severity", no_wrap=True)
    table.add_column("Category", no_wrap=True)
    table.add_column("Check")
    table.add_column("Message")
    table.add_column("Fix")

    for result in report.results:
        table.add_row(
            _severity_value(result.severity),
            _category_value(result.category),
            result.id,
            result.message,
            result.fix_hint or "",
        )

    return table


def _check_python_version() -> DiagnosticResult:
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    supported = (3, 10) <= sys.version_info[:2] < (3, 15)
    if supported:
        return DiagnosticResult(
            id="python.version",
            category=DiagnosticCategory.ENVIRONMENT,
            severity=DiagnosticSeverity.OK,
            message=f"Python {version} is supported",
            details={"version": version},
        )

    return DiagnosticResult(
        id="python.version",
        category=DiagnosticCategory.ENVIRONMENT,
        severity=DiagnosticSeverity.FATAL,
        message=f"Python {version} is outside the supported range >=3.10,<3.15",
        fix_hint="Use Python 3.10, 3.11, 3.12, 3.13, or 3.14.",
        details={"version": version},
    )


def _check_entrypoint(context: DiagnosticContext) -> DiagnosticResult:
    server = context.server
    has_entrypoint = bool(getattr(server, "_entrypoint_fnc", None))
    if has_entrypoint:
        return DiagnosticResult(
            id="agent.entrypoint",
            category=DiagnosticCategory.ENVIRONMENT,
            severity=DiagnosticSeverity.OK,
            message="RTC session entrypoint is registered",
        )

    return DiagnosticResult(
        id="agent.entrypoint",
        category=DiagnosticCategory.ENVIRONMENT,
        severity=DiagnosticSeverity.FATAL,
        message="No RTC session entrypoint has been registered",
        fix_hint="Register one with @server.rtc_session(...) before calling cli.run_app(server).",
        docs_url="https://docs.livekit.io/agents/start/voice-ai/",
    )


def _check_livekit_credentials(context: DiagnosticContext) -> list[DiagnosticResult]:
    required = context.mode in _LIVEKIT_REQUIRED_MODES
    missing_severity = DiagnosticSeverity.FATAL if required else DiagnosticSeverity.ERROR
    results: list[DiagnosticResult] = []

    credentials = {
        "LIVEKIT_URL": _resolved_value(context, "_ws_url", "LIVEKIT_URL"),
        "LIVEKIT_API_KEY": _resolved_value(context, "_api_key", "LIVEKIT_API_KEY"),
        "LIVEKIT_API_SECRET": _resolved_value(context, "_api_secret", "LIVEKIT_API_SECRET"),
    }

    missing = [name for name, value in credentials.items() if not value]
    if missing:
        results.append(
            DiagnosticResult(
                id="livekit.credentials",
                category=DiagnosticCategory.LIVEKIT,
                severity=missing_severity,
                message=f"Missing LiveKit credential environment variables: {', '.join(missing)}",
                fix_hint="Set LIVEKIT_URL, LIVEKIT_API_KEY, and LIVEKIT_API_SECRET, or pass CLI options.",
                docs_url="https://docs.livekit.io/agents/start/voice-ai/",
                details={"missing": missing},
            )
        )
    else:
        results.append(
            DiagnosticResult(
                id="livekit.credentials",
                category=DiagnosticCategory.LIVEKIT,
                severity=DiagnosticSeverity.OK,
                message="LiveKit credentials are configured",
                details={"env_vars": list(credentials.keys())},
            )
        )

    url = credentials["LIVEKIT_URL"]
    if url:
        valid = _valid_ws_url(url)
        results.append(
            DiagnosticResult(
                id="livekit.url",
                category=DiagnosticCategory.LIVEKIT,
                severity=DiagnosticSeverity.OK if valid else missing_severity,
                message=(
                    "LiveKit URL uses a WebSocket scheme"
                    if valid
                    else "LiveKit URL must start with ws:// or wss://"
                ),
                fix_hint=None
                if valid
                else "Use the WebSocket URL for your LiveKit server or Cloud project.",
                docs_url=None
                if valid
                else "https://docs.livekit.io/home/get-started/api-primitives/",
                details={"url": _redact_url(url)},
            )
        )

    return results


def _check_plugins(context: DiagnosticContext) -> list[DiagnosticResult]:
    results: list[DiagnosticResult] = [
        DiagnosticResult(
            id="plugins.registered",
            category=DiagnosticCategory.PLUGIN,
            severity=DiagnosticSeverity.OK,
            message=f"{len(context.registered_plugins)} plugin(s) registered",
            details={
                "plugins": [
                    {
                        "title": getattr(plugin, "title", type(plugin).__name__),
                        "version": getattr(plugin, "version", ""),
                        "package": getattr(plugin, "package", ""),
                    }
                    for plugin in context.registered_plugins
                ]
            },
        )
    ]

    for plugin in context.registered_plugins:
        results.extend(_plugin_metadata_results(context, plugin))

        if context.deep:
            results.extend(_plugin_deep_results(context, plugin))

    return results


def _plugin_metadata_results(context: DiagnosticContext, plugin: Any) -> list[DiagnosticResult]:
    plugin_id = _plugin_id(plugin)
    try:
        info = plugin.diagnostic_info()
    except Exception as exc:
        return [
            DiagnosticResult(
                id=f"plugins.{plugin_id}.metadata",
                category=DiagnosticCategory.PLUGIN,
                severity=DiagnosticSeverity.ERROR,
                message=f"Plugin diagnostic metadata failed: {exc}",
                fix_hint="Check the plugin diagnostic_info() implementation.",
                details={"plugin": plugin_id},
            )
        ]

    if info is None:
        return [
            DiagnosticResult(
                id=f"plugins.{plugin_id}.metadata",
                category=DiagnosticCategory.PLUGIN,
                severity=DiagnosticSeverity.WARN,
                message="Plugin does not expose diagnostic metadata",
                fix_hint="Update the plugin to return PluginDiagnosticInfo from diagnostic_info().",
                details={"plugin": plugin_id},
            )
        ]

    results = [
        DiagnosticResult(
            id=f"plugins.{plugin_id}.capabilities",
            category=DiagnosticCategory.PLUGIN,
            severity=DiagnosticSeverity.OK,
            message="Plugin diagnostic metadata is available",
            docs_url=info.docs_url,
            details={
                "plugin": plugin_id,
                "capabilities": [
                    capability.value if isinstance(capability, Enum) else capability
                    for capability in info.capabilities
                ],
                "optional_env_vars": list(info.optional_env_vars),
                "notes": info.notes,
            },
        )
    ]

    missing_env = [name for name in info.required_env_vars if not context.env.get(name)]
    if missing_env:
        results.append(
            DiagnosticResult(
                id=f"plugins.{plugin_id}.env",
                category=DiagnosticCategory.PLUGIN,
                severity=DiagnosticSeverity.WARN,
                message=f"Plugin may need missing environment variables: {', '.join(missing_env)}",
                fix_hint="Set the provider key or pass credentials explicitly when constructing the plugin client.",
                docs_url=info.docs_url,
                details={"plugin": plugin_id, "missing": missing_env},
            )
        )
    elif info.required_env_vars:
        results.append(
            DiagnosticResult(
                id=f"plugins.{plugin_id}.env",
                category=DiagnosticCategory.PLUGIN,
                severity=DiagnosticSeverity.OK,
                message="Required plugin environment variables are set",
                docs_url=info.docs_url,
                details={"plugin": plugin_id, "env_vars": list(info.required_env_vars)},
            )
        )

    if info.downloadable_files:
        results.append(
            DiagnosticResult(
                id=f"plugins.{plugin_id}.downloads",
                category=DiagnosticCategory.DOWNLOADS,
                severity=DiagnosticSeverity.WARN,
                message="Plugin may require local model or asset downloads",
                fix_hint="Run `python your_agent.py download-files` before the first run.",
                docs_url=info.docs_url,
                details={"plugin": plugin_id, "files": list(info.downloadable_files)},
            )
        )

    return results


def _plugin_deep_results(context: DiagnosticContext, plugin: Any) -> list[DiagnosticResult]:
    plugin_id = _plugin_id(plugin)
    try:
        checks = plugin.diagnostics()
    except Exception as exc:
        return [
            DiagnosticResult(
                id=f"plugins.{plugin_id}.diagnostics",
                category=DiagnosticCategory.PLUGIN,
                severity=DiagnosticSeverity.ERROR,
                message=f"Plugin diagnostics failed: {exc}",
                fix_hint="Check the plugin diagnostics() implementation.",
                details={"plugin": plugin_id},
            )
        ]

    results: list[DiagnosticResult] = []
    for check in checks:
        try:
            results.append(check.run(context))
        except Exception as exc:
            results.append(
                DiagnosticResult(
                    id=check.id,
                    category=check.category,
                    severity=DiagnosticSeverity.ERROR,
                    message=f"Diagnostic check failed: {exc}",
                    fix_hint="Check the plugin diagnostic check implementation.",
                    details={"plugin": plugin_id},
                )
            )

    return results


def _check_audio_devices(context: DiagnosticContext) -> list[DiagnosticResult]:
    try:
        import sounddevice as sd  # type: ignore
    except Exception as exc:
        return [
            DiagnosticResult(
                id="audio.sounddevice",
                category=DiagnosticCategory.AUDIO,
                severity=DiagnosticSeverity.FATAL,
                message=f"Unable to import sounddevice: {exc}",
                fix_hint="Install the audio dependencies for livekit-agents.",
            )
        ]

    checks: list[tuple[str, str | int | None, str]] = [
        ("audio.input_device", context.input_device, "input"),
        ("audio.output_device", context.output_device, "output"),
    ]
    results: list[DiagnosticResult] = []
    for check_id, device, kind in checks:
        if device is None:
            continue
        try:
            sd.query_devices(device, kind=kind)
        except Exception:
            results.append(
                DiagnosticResult(
                    id=check_id,
                    category=DiagnosticCategory.AUDIO,
                    severity=DiagnosticSeverity.FATAL,
                    message=f"Unable to access the {kind} audio device",
                    fix_hint="Run `python your_agent.py console --list-devices` to inspect devices.",
                    details={"device": device},
                )
            )
        else:
            results.append(
                DiagnosticResult(
                    id=check_id,
                    category=DiagnosticCategory.AUDIO,
                    severity=DiagnosticSeverity.OK,
                    message=f"{kind.capitalize()} audio device is accessible",
                    details={"device": device},
                )
            )

    return results


def _check_deployment(context: DiagnosticContext) -> list[DiagnosticResult]:
    server = context.server
    results: list[DiagnosticResult] = []
    drain_timeout = getattr(server, "_drain_timeout", None)
    if isinstance(drain_timeout, int) and drain_timeout <= 0:
        results.append(
            DiagnosticResult(
                id="deployment.drain_timeout",
                category=DiagnosticCategory.DEPLOYMENT,
                severity=DiagnosticSeverity.WARN,
                message="drain_timeout is disabled or non-positive",
                fix_hint="Use a positive drain timeout so active jobs can finish during shutdown.",
                details={"drain_timeout": drain_timeout},
            )
        )

    prometheus_dir = getattr(server, "_prometheus_multiproc_dir", None)
    if prometheus_dir and not os.path.isdir(prometheus_dir):
        results.append(
            DiagnosticResult(
                id="deployment.prometheus_multiproc_dir",
                category=DiagnosticCategory.DEPLOYMENT,
                severity=DiagnosticSeverity.WARN,
                message="Prometheus multiprocess directory does not exist yet",
                fix_hint="Ensure the process can create and clean this directory on startup.",
                details={"path": prometheus_dir},
            )
        )

    return results


def _check_livekit_tcp_connectivity(context: DiagnosticContext) -> DiagnosticResult:
    url = _resolved_value(context, "_ws_url", "LIVEKIT_URL")
    if not url or not _valid_ws_url(url):
        return DiagnosticResult(
            id="network.livekit_tcp",
            category=DiagnosticCategory.NETWORK,
            severity=DiagnosticSeverity.WARN,
            message="Skipping online connectivity check because LIVEKIT_URL is missing or invalid",
            fix_hint="Set LIVEKIT_URL to a ws:// or wss:// URL before using --online.",
        )

    parsed = urlparse(url)
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "wss" else 80)
    if not host:
        return DiagnosticResult(
            id="network.livekit_tcp",
            category=DiagnosticCategory.NETWORK,
            severity=DiagnosticSeverity.ERROR,
            message="Unable to resolve a hostname from LIVEKIT_URL",
            details={"url": _redact_url(url)},
        )

    try:
        with socket.create_connection((host, port), timeout=3):
            pass
    except OSError as exc:
        return DiagnosticResult(
            id="network.livekit_tcp",
            category=DiagnosticCategory.NETWORK,
            severity=DiagnosticSeverity.ERROR,
            message=f"Unable to open a TCP connection to {host}:{port}: {exc}",
            fix_hint="Check network access, DNS, proxy settings, and the LiveKit URL.",
            details={"host": host, "port": port},
        )

    return DiagnosticResult(
        id="network.livekit_tcp",
        category=DiagnosticCategory.NETWORK,
        severity=DiagnosticSeverity.OK,
        message=f"TCP connectivity to {host}:{port} succeeded",
        details={"host": host, "port": port},
    )


def _resolved_value(context: DiagnosticContext, server_attr: str, env_key: str) -> str:
    server_value = getattr(context.server, server_attr, None)
    if server_value:
        return str(server_value)
    return context.env.get(env_key, "")


def _valid_ws_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"ws", "wss"} and bool(parsed.netloc)


def _redact_url(url: str) -> str:
    parsed = urlparse(url)
    if not parsed.password:
        return url
    netloc = parsed.netloc.replace(parsed.password, "***")
    return parsed._replace(netloc=netloc).geturl()


def _plugin_id(plugin: Any) -> str:
    package = getattr(plugin, "package", "") or getattr(plugin, "title", "")
    return str(package or type(plugin).__name__).replace(".", "_").replace("-", "_")


def _severity_value(severity: DiagnosticSeverity | str) -> str:
    return severity.value if isinstance(severity, Enum) else severity


def _category_value(category: DiagnosticCategory | str) -> str:
    return category.value if isinstance(category, Enum) else category


__all__ = [
    "DiagnosticCategory",
    "DiagnosticCheck",
    "DiagnosticContext",
    "DiagnosticMode",
    "DiagnosticReport",
    "DiagnosticResult",
    "DiagnosticSeverity",
    "PluginCapability",
    "PluginDiagnosticInfo",
    "build_diagnostics_table",
    "collect_diagnostics",
    "diagnostic_report_to_json",
    "report_exit_code",
]
