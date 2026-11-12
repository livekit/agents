"""Logging preserves user fields and Unicode without dumping local credentials."""

from __future__ import annotations

import io
import json
import logging
import sys
from typing import Literal

import pytest

from livekit.agents import cli, log as agent_log


@pytest.mark.parametrize("style", ["json", "dev", "console"])
def test_exception_tracebacks_exclude_local_credentials(
    style: Literal["json", "dev", "console"], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tracebacks omit local credentials; https://github.com/livekit/agents/pull/5300."""
    output = io.StringIO()
    monkeypatch.setattr(sys, "stdout", output)
    handler = cli.setup_logging("debug", style)
    logger = logging.getLogger("customer.failure")
    api_key = "key-must-never-appear"
    api_secret = "secret-must-never-appear"
    try:
        try:
            raise RuntimeError("application failed")
        except RuntimeError:
            logger.exception("could not start")
        line = output.getvalue()
        assert "application failed" in line and "RuntimeError" in line
        assert api_key not in line and api_secret not in line
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()


def test_logging_keeps_scope_user_fields_and_style_without_changing_record_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Logging preserves user fields and rendering; https://github.com/livekit/agents/pull/4040 https://github.com/livekit/agents/pull/4142 https://github.com/livekit/agents/pull/4146 https://github.com/livekit/agents/pull/4169."""
    output = io.StringIO()
    monkeypatch.setattr(sys, "stdout", output)
    monkeypatch.setattr(agent_log, "_deployed_region", "cloud-region")
    factory = logging.getLogRecordFactory()
    handler = cli.setup_logging("info", "json")
    try:
        logging.getLogger("customer.scope").info(
            "hello %s", "Ada", extra={"region": "user-region", "scope": "custom", "request": 17}
        )
        record = json.loads(output.getvalue())
        assert record["message"] == "hello Ada"
        assert record["name"] == "customer.scope"
        assert record["region"] == "user-region"
        assert record["scope"] == "custom" and record["request"] == 17
        assert logging.getLogRecordFactory() is factory
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()


class _Customer:
    def __str__(self) -> str:
        return "customer #42"


def test_a_json_log_field_json_cannot_hold_is_written_as_its_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A log field that is not JSON (an application object) is written as its `str()`, and the
    line is still logged."""
    output = io.StringIO()
    monkeypatch.setattr(sys, "stdout", output)
    handler = cli.setup_logging("info", "json")
    try:
        logging.getLogger("customer.fields").info("served", extra={"lk.pii.customer": _Customer()})
        record = json.loads(output.getvalue())
        assert record["message"] == "served"
        assert record["lk.pii.customer"] == "customer #42"
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()


@pytest.mark.parametrize("style", ["json", "dev", "console"])
def test_cp1252_console_keeps_unicode_messages_and_exception_text(
    style: Literal["json", "dev", "console"], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Legacy consoles preserve Unicode logs; https://github.com/livekit/agents/pull/7594."""
    buffer = io.BytesIO()
    output = io.TextIOWrapper(buffer, encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", output)
    handler = cli.setup_logging("info", style)
    words = "hello 日本語 🦀"
    try:
        try:
            raise ValueError(words)
        except ValueError:
            logging.getLogger("customer.unicode").exception(words)
        output.flush()
        line = buffer.getvalue().decode("cp1252")
        if style == "json":
            record = json.loads(line)
            assert record["message"] == words
            assert words in record["exception"]
        else:
            assert words.encode("cp1252", "backslashreplace").decode("cp1252") in line
            assert "ValueError" in line
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()
        output.detach()
