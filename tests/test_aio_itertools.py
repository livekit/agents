import asyncio

import pytest

from livekit.agents.utils.aio.itertools import Tee


async def _async_iter(items):
    for item in items:
        yield item


async def _failing_iter(items, error):
    for item in items:
        yield item
    raise error


@pytest.mark.asyncio
async def test_tee_basic():
    tee = Tee(_async_iter([1, 2, 3]), n=3)
    for peer in tee:
        result = [item async for item in peer]
        assert result == [1, 2, 3]


@pytest.mark.asyncio
async def test_tee_exception_propagation():
    err = ValueError("upstream error")
    tee = Tee(_failing_iter([1, 2], err), n=2)

    for peer in tee:
        result = []
        with pytest.raises(ValueError, match="upstream error"):
            async for item in peer:
                result.append(item)
        assert result == [1, 2]


@pytest.mark.asyncio
async def test_tee_exception_concurrent():
    err = ValueError("concurrent error")
    tee = Tee(_failing_iter([1], err), n=2)

    results: list[list[int]] = [[], []]
    errors: list[BaseException | None] = [None, None]

    async def consume(idx: int, peer):
        try:
            async for item in peer:
                results[idx].append(item)
        except ValueError as e:
            errors[idx] = e

    await asyncio.gather(consume(0, tee[0]), consume(1, tee[1]))

    for idx in range(2):
        assert results[idx] == [1]
        assert isinstance(errors[idx], ValueError)
        assert str(errors[idx]) == "concurrent error"


@pytest.mark.asyncio
async def test_tee_cancelled_error_does_not_propagate_to_peers():
    """CancelledError in one peer's task should not cascade to other peers."""
    items = [1, 2, 3]
    tee = Tee(_async_iter(items), n=2)

    peer1_results: list[int] = []

    async def consume_peer0():
        async for _item in tee[0]:
            # Block forever after first item so we can cancel this task
            await asyncio.sleep(1000)

    async def consume_peer1():
        async for item in tee[1]:
            peer1_results.append(item)

    task0 = asyncio.create_task(consume_peer0())
    task1 = asyncio.create_task(consume_peer1())

    # Give tasks a chance to start
    await asyncio.sleep(0.01)

    # Cancel task0 — this should NOT affect task1
    task0.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task0

    # task1 should complete normally without CancelledError
    await asyncio.wait_for(task1, timeout=2.0)
    assert peer1_results == items


@pytest.mark.asyncio
async def test_tee_empty_iterator():
    tee = Tee(_async_iter([]), n=2)
    for peer in tee:
        result = [item async for item in peer]
        assert result == []
