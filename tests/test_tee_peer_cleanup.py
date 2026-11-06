import asyncio
import gc
import weakref

import pytest

from livekit.agents.utils.aio.itertools import Tee

pytestmark = pytest.mark.unit


class Payload:
    pass


@pytest.mark.asyncio
async def test_closing_an_unstarted_peer_releases_future_buffered_items():
    references = []

    async def source():
        for _ in range(100):
            item = Payload()
            references.append(weakref.ref(item))
            yield item

    tee = Tee(source())
    active, unused = tee
    try:
        await unused.aclose()
        async for item in active:
            del item
        gc.collect()
        assert all(reference() is None for reference in references)
    finally:
        await tee.aclose()


class CloseableSource:
    def __init__(self):
        self.close_count = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration

    async def aclose(self):
        self.close_count += 1


@pytest.mark.asyncio
@pytest.mark.parametrize("n", [0, 1, 2])
async def test_tee_close_closes_unstarted_upstream_only_once(n):
    source = CloseableSource()
    tee = Tee(source, n=n)
    await tee.aclose()
    await tee.aclose()
    assert source.close_count == 1


@pytest.mark.asyncio
async def test_tee_close_retries_after_a_peer_fails_to_close_upstream():
    class Source(CloseableSource):
        async def aclose(self):
            await super().aclose()
            if self.close_count == 1:
                raise ValueError("first close failed")

    source = Source()
    tee = Tee(source)
    await tee.aclose()
    assert source.close_count == 2
    await tee.aclose()
    assert source.close_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("n", [0, 2])
async def test_concurrent_close_waits_and_retries_after_the_first_caller_is_cancelled(n):
    started = asyncio.Event()
    release = asyncio.Event()

    class Source(CloseableSource):
        async def aclose(self):
            started.set()
            await release.wait()
            await super().aclose()

    source = Source()
    tee = Tee(source, n=n)
    first = asyncio.create_task(tee.aclose())
    await started.wait()
    second = asyncio.create_task(tee.aclose())
    try:
        await asyncio.sleep(0)
        assert not second.done()
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        release.set()
        await second
        assert source.close_count == 1
    finally:
        release.set()
        first.cancel()
        await asyncio.gather(first, second, return_exceptions=True)
    await tee.aclose()
    assert source.close_count == 1


@pytest.mark.asyncio
async def test_closing_all_unstarted_peers_closes_the_upstream_once():
    source = CloseableSource()
    tee = Tee(source)
    first, second = tee
    await first.aclose()
    assert source.close_count == 0
    await second.aclose()
    assert source.close_count == 1
    await first.aclose()
    await second.aclose()
    assert source.close_count == 1
