from __future__ import annotations

import json
import logging

import pytest

from livekit.agents.cli.log import JsonFormatter
from livekit.agents.telemetry.hosted_logs import (
    HOSTED_LOGS_ATTR,
    HOSTED_LOGS_ENV,
    HOSTED_LOGS_SIDECAR,
    announce_sidecar_log_handoff,
    logs_handled_by_sidecar,
)
from livekit.agents.version import __version__

pytestmark = pytest.mark.unit


def test_logs_handled_by_sidecar_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(HOSTED_LOGS_ENV, raising=False)
    assert not logs_handled_by_sidecar()
    monkeypatch.setenv(HOSTED_LOGS_ENV, "something-else")
    assert not logs_handled_by_sidecar()
    monkeypatch.setenv(HOSTED_LOGS_ENV, HOSTED_LOGS_SIDECAR)
    assert logs_handled_by_sidecar()


def test_announce_prints_nothing_without_sidecar(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv(HOSTED_LOGS_ENV, raising=False)
    announce_sidecar_log_handoff()
    assert capsys.readouterr().out == ""


def test_announce_prints_one_json_marker_line(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(HOSTED_LOGS_ENV, HOSTED_LOGS_SIDECAR)
    announce_sidecar_log_handoff()
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    marker = json.loads(lines[0])
    assert marker[HOSTED_LOGS_ATTR] == HOSTED_LOGS_SIDECAR
    assert marker["sdk"] == "python"
    assert marker["sdk_version"] == __version__
    assert marker["level"] == "INFO"
    assert marker["message"]


def test_json_formatter_adds_code_attributes() -> None:
    record = logging.LogRecord(
        name="sidecartest",
        level=logging.INFO,
        pathname="/app/agent.py",
        lineno=42,
        msg="hello",
        args=None,
        exc_info=None,
        func="entrypoint",
    )
    line = json.loads(JsonFormatter().format(record))
    assert line["code.file.path"] == "/app/agent.py"
    assert line["code.function.name"] == "entrypoint"
    assert line["code.line.number"] == 42
