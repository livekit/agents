from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
import os
import sys
import time

import pytest

from livekit.agents.ipc import stdio_capture
from livekit.agents.ipc.stdio_capture import (
    ChildStdio,
    StdioReader,
    create_stdio_pairs,
    redirect_stdio,
)

pytestmark = [pytest.mark.unit]


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def capture():
    handler = _Capture()
    lg = logging.getLogger("livekit.agents.stdio")
    prev_level = lg.level
    lg.addHandler(handler)
    lg.setLevel(logging.INFO)
    yield handler
    lg.removeHandler(handler)
    lg.setLevel(prev_level)


def test_capture_enabled_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LIVEKIT_CAPTURE_JOB_STDIO", raising=False)
    monkeypatch.delenv("LIVEKIT_AGENT_ID", raising=False)
    assert not stdio_capture.capture_enabled()
    monkeypatch.setenv("LIVEKIT_AGENT_ID", "CA_test")
    assert stdio_capture.capture_enabled()
    monkeypatch.setenv("LIVEKIT_CAPTURE_JOB_STDIO", "0")
    assert not stdio_capture.capture_enabled()
    monkeypatch.delenv("LIVEKIT_AGENT_ID")
    monkeypatch.setenv("LIVEKIT_CAPTURE_JOB_STDIO", "true")
    assert stdio_capture.capture_enabled()


@pytest.mark.asyncio
async def test_reader_splits_lines_and_attributes(capture: _Capture) -> None:
    parent, child = create_stdio_pairs()
    loop = asyncio.get_running_loop()
    reader = StdioReader(parent.stdout, "stdout", lambda: {"job_id": "AJ_1"}, loop)
    reader.start()

    before = time.time()
    os.write(child.stdout.fileno(), b"hello\nwor")
    await asyncio.sleep(0.05)
    os.write(child.stdout.fileno(), b"ld\n\n  \n")
    os.write(child.stdout.fileno(), b"tail without newline")
    child.close()
    parent.stderr.close()
    await reader.aclose()

    assert [r.getMessage() for r in capture.records] == ["hello", "world", "tail without newline"]
    for r in capture.records:
        assert r.stream == "stdout"  # type: ignore[attr-defined]
        assert r.job_id == "AJ_1"  # type: ignore[attr-defined]
        assert before - 1 <= r.created <= time.time() + 1
    if sys.platform == "linux":
        assert all(r.pid == os.getpid() for r in capture.records)  # type: ignore[attr-defined]


def _child_main(stdio: ChildStdio) -> None:
    redirect_stdio(stdio)
    print("printed line")
    os.write(2, b"raw stderr\n")
    sys.stdout.flush()


@pytest.mark.asyncio
async def test_redirect_in_spawned_child(capture: _Capture) -> None:
    parent, child = create_stdio_pairs()
    loop = asyncio.get_running_loop()
    out = StdioReader(parent.stdout, "stdout", dict, loop)
    err = StdioReader(parent.stderr, "stderr", dict, loop)
    out.start()
    err.start()

    proc = mp.get_context("spawn").Process(target=_child_main, args=(child,))
    proc.start()
    child.close()
    await loop.run_in_executor(None, proc.join, 30)
    await out.aclose()
    await err.aclose()

    seen = {(r.stream, r.getMessage()) for r in capture.records}  # type: ignore[attr-defined]
    assert ("stdout", "printed line") in seen
    assert ("stderr", "raw stderr") in seen
    if sys.platform == "linux":
        assert {r.pid for r in capture.records} == {proc.pid}  # type: ignore[attr-defined]
