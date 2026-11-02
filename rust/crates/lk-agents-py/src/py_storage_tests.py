"""The host handler stream's terminal states and checkpoint cancellation contract."""

import asyncio
from collections.abc import Awaitable, Callable

import pytest

from livekit.agents._generated.storage_turn import Checkpoint, HandlerOutput, Output
from livekit.agents.storage.context import _TurnStream

pytestmark = pytest.mark.unit


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
