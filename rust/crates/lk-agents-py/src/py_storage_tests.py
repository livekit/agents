"""The host handler stream's terminal states and checkpoint cancellation contract."""

import asyncio
import pickle
from collections.abc import Awaitable, Callable
from datetime import timedelta
from pathlib import Path

import pytest

from livekit.agents._generated.storage_turn import Checkpoint, HandlerOutput, Output
from livekit.agents.storage import SQLiteStorage
from livekit.agents.storage.context import _TurnStream
from livekit.agents.storage.errors import ThreadNotFound

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


def _output() -> HandlerOutput:
    return HandlerOutput(
        output="done",
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


async def test_close_before_read_never_starts_handler() -> None:
    async def run(checkpoint: Callable[[], Awaitable[bool]]) -> HandlerOutput:
        pytest.fail("closed stream started its handler")

    stream = _TurnStream(run)
    await stream.aclose()
    async with asyncio.timeout(5):
        with pytest.raises(StopAsyncIteration):
            await anext(stream)


@pytest.mark.parametrize("fail", [False, True])
async def test_handler_completion_is_terminal_on_repeated_reads(fail: bool) -> None:
    async def run(checkpoint: Callable[[], Awaitable[bool]]) -> HandlerOutput:
        if fail:
            raise ValueError("handler storage failure")
        return _output()

    stream = _TurnStream(run)
    async with asyncio.timeout(5):
        try:
            if fail:
                with pytest.raises(ValueError, match="handler storage failure"):
                    await anext(stream)
            else:
                event = await anext(stream)
                assert isinstance(event, Output) and event.output.output == "done"
            for _ in range(2):
                with pytest.raises(StopAsyncIteration):
                    await anext(stream)
        finally:
            await stream.aclose()


async def test_close_at_checkpoint_settles_handler_cleanup() -> None:
    settled = asyncio.Event()

    async def run(checkpoint: Callable[[], Awaitable[bool]]) -> HandlerOutput:
        try:
            await checkpoint()
            pytest.fail("closed checkpoint resumed handler")
        finally:
            settled.set()

    stream = _TurnStream(run)
    async with asyncio.timeout(5):
        assert isinstance(await anext(stream), Checkpoint)
        await stream.aclose()
        assert settled.is_set()
        with pytest.raises(StopAsyncIteration):
            await anext(stream)


async def test_cancelled_close_waiter_can_rejoin_without_cancelling_cleanup_again() -> None:
    cleaning, release, settled = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def run(checkpoint: Callable[[], Awaitable[bool]]) -> HandlerOutput:
        try:
            await checkpoint()
            return _output()
        finally:
            cleaning.set()
            await release.wait()
            settled.set()

    stream = _TurnStream(run)
    async with asyncio.timeout(5):
        assert isinstance(await anext(stream), Checkpoint)
        closing = asyncio.create_task(stream.aclose())
        try:
            await cleaning.wait()
            closing.cancel()
            with pytest.raises(asyncio.CancelledError):
                await closing
            assert not settled.is_set()
        finally:
            release.set()
            await stream.aclose()
        assert settled.is_set()
