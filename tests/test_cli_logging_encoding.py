from __future__ import annotations

import io
import json
import logging
import sys

import pytest

from livekit.agents.cli._legacy import _configure_logger
from livekit.agents.cli.log import setup_logging

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]


def test_production_cli_logging_preserves_unicode_on_cp1252_stdout(monkeypatch):
    output = io.BytesIO()
    stdout = io.TextIOWrapper(output, encoding="cp1252")
    root = logging.getLogger()
    monkeypatch.setattr(root, "handlers", [])
    monkeypatch.setattr(root, "level", logging.NOTSET)
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(logging, "raiseExceptions", False)

    setup_logging("INFO", devmode=False, console=False)
    logging.getLogger().info("こんにちは 🗣️")
    stdout.flush()

    record = json.loads(output.getvalue().decode("utf-8"))
    assert record["message"] == "こんにちは 🗣️"


def test_development_cli_logging_preserves_unicode_on_cp1252_stdout(monkeypatch):
    output = io.BytesIO()
    stdout = io.TextIOWrapper(output, encoding="cp1252")
    root = logging.getLogger()
    monkeypatch.setattr(root, "handlers", [])
    monkeypatch.setattr(root, "level", logging.NOTSET)
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(logging, "raiseExceptions", False)

    setup_logging("INFO", devmode=True, console=False)
    logging.getLogger().info("こんにちは 🗣️")
    stdout.flush()

    assert "こんにちは 🗣️" in output.getvalue().decode("utf-8")


def test_legacy_cli_logging_preserves_unicode_on_cp1252_stdout(monkeypatch):
    output = io.BytesIO()
    stdout = io.TextIOWrapper(output, encoding="cp1252")
    root = logging.getLogger()
    monkeypatch.setattr(root, "handlers", [])
    monkeypatch.setattr(root, "level", logging.NOTSET)
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(logging, "raiseExceptions", False)

    _configure_logger(None, "INFO")
    logging.getLogger().info("こんにちは 🗣️")
    stdout.flush()

    record = json.loads(output.getvalue().decode("utf-8"))
    assert record["message"] == "こんにちは 🗣️"
