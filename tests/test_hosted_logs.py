from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from livekit.agents._hosted_logs import (
    HOSTED_LOGS_ENABLED,
    HOSTED_LOGS_ENV,
    HOSTED_LOGS_HANDOFF_FILE_ENV,
    hand_off,
    logs_handed_off,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def handoff_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    path = tmp_path / "sdk-handoff"
    monkeypatch.setenv(HOSTED_LOGS_ENV, HOSTED_LOGS_ENABLED)
    monkeypatch.setenv(HOSTED_LOGS_HANDOFF_FILE_ENV, str(path))
    return path


def test_hand_off_creates_the_file(handoff_file: Path) -> None:
    assert not logs_handed_off()
    hand_off()
    assert handoff_file.exists()
    assert logs_handed_off()


def test_hand_off_keeps_an_existing_file(handoff_file: Path) -> None:
    handoff_file.write_text("x")
    hand_off()
    assert handoff_file.read_text() == "x"
    assert logs_handed_off()


def test_no_hand_off_when_hosted_logs_are_off(
    handoff_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(HOSTED_LOGS_ENV, "something-else")
    hand_off()
    assert not handoff_file.exists()
    assert not logs_handed_off()


def test_no_hand_off_without_a_file_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(HOSTED_LOGS_ENV, HOSTED_LOGS_ENABLED)
    monkeypatch.delenv(HOSTED_LOGS_HANDOFF_FILE_ENV, raising=False)
    hand_off()
    assert not logs_handed_off()


def test_sdk_keeps_exporting_when_the_file_cannot_be_created(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(HOSTED_LOGS_ENV, HOSTED_LOGS_ENABLED)
    monkeypatch.setenv(HOSTED_LOGS_HANDOFF_FILE_ENV, str(tmp_path / "missing-dir" / "sdk-handoff"))
    hand_off()
    assert not logs_handed_off()
    assert capsys.readouterr() == ("", "")


def test_importing_livekit_agents_creates_the_file(handoff_file: Path) -> None:
    subprocess.run([sys.executable, "-c", "import livekit.agents"], check=True)
    assert handoff_file.exists()
