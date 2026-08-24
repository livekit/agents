from __future__ import annotations

import logging
from typing import Any

import pytest
from rich.table import Table
from rich.text import Text

from livekit.agents.cli._legacy import RichLoggingHandler

pytestmark = pytest.mark.unit


class _StubConsole:
    width = 120

    def print(self, *args: Any, **kwargs: Any) -> None:
        pass


class _StubAgentsConsole:
    def __init__(self) -> None:
        self.console = _StubConsole()
        self.rendered: Table | None = None

    def _render_tag(self, output: Table, *, tag_width: int) -> Table:
        self.rendered = output
        return output


def test_default_millisecond_timestamp_has_fixed_width() -> None:
    console = _StubAgentsConsole()
    handler = RichLoggingHandler(console)  # type: ignore[arg-type]
    record = logging.LogRecord("axum", logging.INFO, "", 0, "request complete", (), None)
    record.created = 0.001

    handler.emit(record)

    assert console.rendered is not None
    time_column = console.rendered.columns[0]
    assert time_column.width == len("00:00:00.000")
    assert time_column.no_wrap is True

    timestamp = time_column._cells[0]
    assert isinstance(timestamp, Text)
    assert timestamp.cell_len == time_column.width
