from __future__ import annotations

import json
import logging

import pytest

from livekit.agents.cli.log import JsonFormatter

pytestmark = pytest.mark.unit


def test_json_formatter_adds_code_attributes() -> None:
    record = logging.LogRecord(
        name="formattertest",
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
