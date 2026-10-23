from __future__ import annotations

from pathlib import Path

import pytest

from livekit.agents.telemetry.hosted_logs import (
    HOSTED_LOGS_ENV,
    HOSTED_LOGS_HANDOFF_FILE_ENV,
    HOSTED_LOGS_SIDECAR,
    hand_off_logs_to_sidecar,
    logs_handled_by_sidecar,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def handoff_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    path = tmp_path / "sdk-handoff"
    monkeypatch.setenv(HOSTED_LOGS_ENV, HOSTED_LOGS_SIDECAR)
    monkeypatch.setenv(HOSTED_LOGS_HANDOFF_FILE_ENV, str(path))
    return path


def test_hand_off_creates_the_file(handoff_file: Path) -> None:
    assert not logs_handled_by_sidecar()
    hand_off_logs_to_sidecar()
    assert handoff_file.exists()
    assert logs_handled_by_sidecar()


def test_hand_off_keeps_an_existing_file(handoff_file: Path) -> None:
    handoff_file.write_text("x")
    hand_off_logs_to_sidecar()
    assert handoff_file.read_text() == "x"
    assert logs_handled_by_sidecar()


def test_no_hand_off_without_sidecar_mode(
    handoff_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(HOSTED_LOGS_ENV, "something-else")
    hand_off_logs_to_sidecar()
    assert not handoff_file.exists()
    assert not logs_handled_by_sidecar()


def test_no_hand_off_without_a_file_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(HOSTED_LOGS_ENV, HOSTED_LOGS_SIDECAR)
    monkeypatch.delenv(HOSTED_LOGS_HANDOFF_FILE_ENV, raising=False)
    hand_off_logs_to_sidecar()
    assert not logs_handled_by_sidecar()


def test_sdk_keeps_exporting_when_the_file_cannot_be_created(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(HOSTED_LOGS_ENV, HOSTED_LOGS_SIDECAR)
    monkeypatch.setenv(HOSTED_LOGS_HANDOFF_FILE_ENV, str(tmp_path / "missing-dir" / "sdk-handoff"))
    hand_off_logs_to_sidecar()
    assert not logs_handled_by_sidecar()
