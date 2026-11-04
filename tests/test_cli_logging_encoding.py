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
    logging.getLogger().info("café عاوز 🗣️")
    stdout.flush()

    record = json.loads(output.getvalue().decode("cp1252", errors="replace"))
    assert record["message"] == "café عاوز 🗣️"


def test_development_cli_logging_preserves_unicode_on_cp1252_stdout(monkeypatch):
    output = io.BytesIO()
    stdout = io.TextIOWrapper(output, encoding="cp1252")
    root = logging.getLogger()
    monkeypatch.setattr(root, "handlers", [])
    monkeypatch.setattr(root, "level", logging.NOTSET)
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(logging, "raiseExceptions", False)

    setup_logging("INFO", devmode=True, console=False)
    logging.getLogger().info("café عاوز 🗣️")
    stdout.flush()

    message = output.getvalue().decode("cp1252", errors="replace")
    assert "café" in message
    assert r"\u0639" in message
    assert r"\U0001f5e3" in message


def test_legacy_cli_logging_preserves_unicode_on_cp1252_stdout(monkeypatch):
    output = io.BytesIO()
    stdout = io.TextIOWrapper(output, encoding="cp1252")
    root = logging.getLogger()
    monkeypatch.setattr(root, "handlers", [])
    monkeypatch.setattr(root, "level", logging.NOTSET)
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(logging, "raiseExceptions", False)

    _configure_logger(None, "INFO")
    logging.getLogger().info("café عاوز 🗣️")
    stdout.flush()

    record = json.loads(output.getvalue().decode("cp1252", errors="replace"))
    assert record["message"] == "café عاوز 🗣️"


def test_cli_logging_setup_preserves_locale_encoded_stdout_for_print(monkeypatch):
    output = io.BytesIO()
    stdout = io.TextIOWrapper(output, encoding="iso8859-1")
    root = logging.getLogger()
    monkeypatch.setattr(root, "handlers", [])
    monkeypatch.setattr(root, "level", logging.NOTSET)
    monkeypatch.setattr(sys, "stdout", stdout)

    setup_logging("INFO", devmode=False, console=False)
    print("café")
    stdout.flush()

    assert output.getvalue() == b"caf\xe9\n"


def test_production_cli_logging_keeps_raw_unicode_on_utf8_stdout(monkeypatch):
    output = io.BytesIO()
    stdout = io.TextIOWrapper(output, encoding="utf-8")
    root = logging.getLogger()
    monkeypatch.setattr(root, "handlers", [])
    monkeypatch.setattr(root, "level", logging.NOTSET)
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(logging, "raiseExceptions", False)

    setup_logging("INFO", devmode=False, console=False)
    logging.getLogger().info("café")
    stdout.flush()

    assert '"message": "café"' in output.getvalue().decode("utf-8")
