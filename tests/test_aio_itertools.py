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
async def test_tee_empty_iterator():
    tee = Tee(_async_iter([]), n=2)
    for peer in tee:
        result = [item async for item in peer]
        assert result == []
