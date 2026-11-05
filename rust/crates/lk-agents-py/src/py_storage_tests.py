"""Local storage, and the contract of a handler the native turn runs (docs/server.md
"Progress, checkpoints and cancellation")."""

import asyncio
import contextvars
import pickle
from collections.abc import Awaitable, Callable
from datetime import timedelta
from pathlib import Path

import pytest

from livekit.agents import _core
from livekit.agents._generated.storage_turn import HandlerOutput, TurnContext, TurnResult
from livekit.agents.storage import SQLiteStorage, StorageUnavailable, ThreadNotFound

pytestmark = pytest.mark.unit


async def test_default_local_directory_is_project_isolated_and_survives_pickle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.delenv("LIVEKIT_PROJECT_ID", raising=False)
    storage = SQLiteStorage()
    other = SQLiteStorage(project_id="other")
    restored = pickle.loads(pickle.dumps(storage))
    try:
        assert storage.threads.location == str(tmp_path / ".livekit/agent-threads/default")
        assert other.threads.location == str(tmp_path / ".livekit/agent-threads/other")
        created = await storage.threads.create_thread(scope="customer")
        assert (await restored.threads.get_thread(created.thread_id)).scope == "customer"
        with pytest.raises(ThreadNotFound):
            await other.threads.get_thread(created.thread_id)
    finally:
        await restored.aclose()
        await other.aclose()
        await storage.aclose()


@pytest.mark.parametrize("project_id", ["", "p_", "../other", "/tmp/other", "a/b", "a\\b"])
def test_local_project_id_cannot_escape_its_directory(project_id: str) -> None:
    with pytest.raises(ValueError, match="project_id"):
        SQLiteStorage(project_id=project_id)


@pytest.mark.parametrize("interval", [0, 5, 10, 11])
def test_storage_rejects_unsafe_renewal_timing(tmp_path: Path, interval: int) -> None:
    with pytest.raises(ValueError, match="renew_interval"):
        SQLiteStorage(
            directory=tmp_path,
            lease_ttl=timedelta(seconds=10),
            renew_interval=timedelta(seconds=interval),
        )


def _output(text: str = "done") -> HandlerOutput:
    return HandlerOutput(
        output=text,
        new_items=[],
        state=None,
        scope_writes=[],
        app_writes=[],
        usage=[],
        input_required=None,
        pending_call=None,
        interrupted=False,
        metadata={},
        error=None,
        tool_executions=[],
    )


_Run = Callable[[TurnContext, _core._core_HandlerTurn], Awaitable[HandlerOutput]]


class _Host:
    def __init__(self, run: _Run) -> None:
        self._run = run

    async def _core_handle(
        self, ctx: TurnContext, reports: _core._core_HandlerTurn
    ) -> HandlerOutput:
        return await self._run(ctx, reports)


async def _turn(tmp_path: Path, run: _Run) -> TurnResult:
    storage = SQLiteStorage(directory=tmp_path)
    try:
        thread_id = (await storage.threads.create_thread()).thread_id
        core = await storage._open(thread_id)
        try:
            message = {"messageId": "m1", "role": "ROLE_USER", "parts": [{"text": "hi"}]}
            return await _core.Turn(core, "desk", _Host(run)).run(message)
        finally:
            core.close()
    finally:
        await storage.aclose()


_caller: contextvars.ContextVar[str] = contextvars.ContextVar("caller", default="none")


async def test_a_handler_runs_in_the_callers_context_past_a_full_report_queue(
    tmp_path: Path,
) -> None:
    async def run(ctx: TurnContext, reports: _core._core_HandlerTurn) -> HandlerOutput:
        for n in range(200):
            await reports.progress(f"step {n}")
            await reports.token(f"{n} ")
        assert not await reports.checkpoint()
        return _output(_caller.get())

    _caller.set("request")
    async with asyncio.timeout(10):
        result = await _turn(tmp_path, run)
    assert result.task.status == "completed" and result.output == "request"


async def test_a_storage_error_the_handler_raises_fails_its_task_durably(tmp_path: Path) -> None:
    async def run(ctx: TurnContext, reports: _core._core_HandlerTurn) -> HandlerOutput:
        await reports.progress("working")
        raise StorageUnavailable("the database is down")

    async with asyncio.timeout(10):
        result = await _turn(tmp_path, run)
    assert result.task.status == "failed" and result.output is None


async def test_cancelling_the_turn_cancels_the_waiting_handler_and_runs_its_cleanup(
    tmp_path: Path,
) -> None:
    entered, cleaned = asyncio.Event(), asyncio.Event()

    async def run(ctx: TurnContext, reports: _core._core_HandlerTurn) -> HandlerOutput:
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()
        pytest.fail("a cancelled handler resumed")

    async with asyncio.timeout(10):
        turn = asyncio.create_task(_turn(tmp_path, run))
        await entered.wait()
        turn.cancel()
        with pytest.raises(asyncio.CancelledError):
            await turn
        await cleaned.wait()


async def test_reports_after_the_answer_are_dropped_and_checkpoints_yield(
    tmp_path: Path,
) -> None:
    kept: list[_core._core_HandlerTurn] = []

    async def run(ctx: TurnContext, reports: _core._core_HandlerTurn) -> HandlerOutput:
        kept.append(reports)
        return _output()

    async with asyncio.timeout(10):
        assert (await _turn(tmp_path, run)).output == "done"
        (reports,) = kept
        for n in range(100):
            await reports.progress(f"late {n}")
        assert await reports.checkpoint()
