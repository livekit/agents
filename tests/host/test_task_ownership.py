"""Every task the framework starts has an owner: `utils.aio.TaskSet` and the objects built on it."""

from __future__ import annotations

import asyncio
import logging
import re
import subprocess
from pathlib import Path

import pytest

from livekit.agents.utils.aio import TaskSet

ROOT = Path(__file__).resolve().parents[2]


async def _fail() -> None:
    raise ValueError("boom")


async def test_a_failure_nobody_awaits_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    """Regression: a background task's failure vanished, or surfaced only when GC ran."""

    def failures() -> list[logging.LogRecord]:
        return [r for r in caplog.records if "unobserved" in r.getMessage()]

    async with TaskSet() as tasks:
        tasks.spawn(_fail(), name="unobserved")
        while not failures():
            await asyncio.sleep(0)

    assert [r.levelno for r in failures()] == [logging.ERROR]
    assert failures()[0].exc_info is not None and isinstance(failures()[0].exc_info[1], ValueError)


async def test_a_failure_is_raised_to_its_awaiter_and_not_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async with TaskSet() as tasks:
        with pytest.raises(ValueError):
            await asyncio.gather(tasks.spawn(_fail(), name="awaited"))
        await asyncio.sleep(0)

    assert not [r for r in caplog.records if "awaited" in r.getMessage()]


async def test_closing_cancels_and_joins_what_is_still_running() -> None:
    released = False

    async def hold() -> None:
        nonlocal released
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            released = True

    tasks = TaskSet()
    task = tasks.spawn(hold())
    await asyncio.sleep(0)
    await tasks.aclose()

    assert task.cancelled() and released
    with pytest.raises(RuntimeError):
        tasks.spawn(hold())


async def test_first_completed_returns_the_winner_and_cancels_the_losers() -> None:
    gate = asyncio.Event()
    async with TaskSet() as tasks:
        loser = tasks.spawn(asyncio.Event().wait())
        winner = tasks.spawn(gate.wait())
        gate.set()
        assert await tasks.first_completed(loser, winner) is winner
        assert loser.cancelled()

        with pytest.raises(ValueError):
            await tasks.first_completed(_fail(), asyncio.Event().wait())


def test_tasks_are_never_spawned_around_the_task_set() -> None:
    """`loop.create_task` and `ctx.run(asyncio.create_task, ...)` are invisible to ruff's TID251."""
    pattern = re.compile(r"(?<!\basyncio)\.create_task\(|\.run\(\s*asyncio\.create_task\b")
    sources = subprocess.run(
        ["git", "ls-files", "python/livekit-agents/livekit", "python/livekit-plugins"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    offenders = [
        f"{path}:{number}"
        for path in sources
        if path.endswith(".py") and "/_generated/" not in path and not path.endswith("utils/aio.py")
        for number, line in enumerate((ROOT / path).read_text().splitlines(), 1)
        if pattern.search(line) and "noqa: TID251" not in line
    ]
    assert not offenders, "spawn on a utils.aio.TaskSet instead:\n" + "\n".join(offenders)
