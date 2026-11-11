import asyncio
import contextvars
import subprocess
import sys
import textwrap

import pytest

from livekit.agents.utils.aio.itertools import Tee

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]


def test_event_loop_shutdown_finishes_source_cleanup():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import asyncio
                from livekit.agents.utils.aio.itertools import Tee

                async def main():
                    entered = asyncio.Event()

                    async def source():
                        try:
                            entered.set()
                            await asyncio.Event().wait()
                            yield 1
                        finally:
                            await asyncio.sleep(0.01)
                            print("source cleanup completed", flush=True)

                    tee = Tee(source(), n=2)
                    reader = asyncio.create_task(anext(tee[0]))
                    await entered.wait()
                    assert not reader.done()
                    # Leave the second peer unstarted and let asyncio.run
                    # cancel all pending tasks without an explicit Tee close.

                asyncio.run(main())
                print("event loop shutdown completed", flush=True)
                """
            ),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "source cleanup completed",
        "event loop shutdown completed",
    ]


@pytest.mark.parametrize("producer_first", [True, False])
async def test_shutdown_cancellation_order_preserves_async_cleanup(producer_first):
    entered = asyncio.Event()
    closing = asyncio.Event()
    release = asyncio.Event()
    cleanup_count = 0

    async def source():
        nonlocal cleanup_count
        try:
            entered.set()
            await asyncio.Event().wait()
            yield 1
        finally:
            closing.set()
            await release.wait()
            cleanup_count += 1

    tee = Tee(source(), n=2)
    reader = asyncio.create_task(anext(tee[0]))
    await entered.wait()
    producer = tee._source._task
    supervisor = tee._source._supervisor
    assert producer is not None and supervisor is not None
    try:
        if producer_first:
            producer.cancel()
            await closing.wait()
            supervisor.cancel()
        else:
            supervisor.cancel()
            await asyncio.sleep(0)
            producer.cancel()
            await closing.wait()
        reader.cancel()
        await asyncio.sleep(0)
        assert not producer.done()
        assert not supervisor.done()
        release.set()
        await asyncio.gather(reader, producer, supervisor, return_exceptions=True)
        assert cleanup_count == 1
    finally:
        release.set()
        await tee.aclose()
        await asyncio.gather(reader, producer, supervisor, return_exceptions=True)


async def test_upstream_cancellation_does_not_end_surviving_peers():
    context = contextvars.ContextVar("source-context", default="caller")

    class Source:
        pulls = 0
        closes = 0
        token = None
        tasks = []

        def __aiter__(self):
            return self

        async def __anext__(self):
            self.pulls += 1
            self.tasks.append(asyncio.current_task())
            if self.pulls == 1:
                self.token = context.set("source")
                raise asyncio.CancelledError
            assert context.get() == "source"
            if self.pulls == 2:
                return "alive"
            raise StopAsyncIteration

        async def aclose(self):
            context.reset(self.token)
            self.closes += 1

    source = Source()
    tee = Tee(source, n=3)
    try:
        with pytest.raises(asyncio.CancelledError):
            await anext(tee[0])
        assert source.closes == 0
        assert source.pulls == 1
        assert [item async for item in tee[1]] == ["alive"]
        assert [item async for item in tee[2]] == ["alive"]
        assert source.pulls == 3
        assert source.closes == 1
        assert all(task is source.tasks[0] for task in source.tasks)
    finally:
        await tee.aclose()


async def test_upstream_cancellation_is_retained_after_reader_cancellation():
    entered = asyncio.Event()
    release = asyncio.Event()

    class Source:
        pulls = 0
        closes = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            self.pulls += 1
            if self.pulls == 1:
                entered.set()
                await release.wait()
                raise asyncio.CancelledError
            if self.pulls == 2:
                return "alive"
            raise StopAsyncIteration

        async def aclose(self):
            self.closes += 1

    source = Source()
    tee = Tee(source, n=3)
    reader = asyncio.create_task(anext(tee[0]))
    await entered.wait()
    try:
        reader.cancel()
        with pytest.raises(asyncio.CancelledError):
            await reader
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await anext(tee[1])
        assert source.closes == 0
        assert [item async for item in tee[2]] == ["alive"]
        assert source.pulls == 3
        assert source.closes == 1
    finally:
        release.set()
        reader.cancel()
        await asyncio.gather(reader, return_exceptions=True)
        await tee.aclose()


async def test_cancelled_upstream_pull_is_reused_by_surviving_peers():
    entered = asyncio.Event()
    release = asyncio.Event()
    pulls = []

    async def source():
        entered.set()
        await release.wait()
        for item in [1, 2, 3]:
            pulls.append(item)
            yield item

    tee = Tee(source(), n=3)
    first = asyncio.create_task(anext(tee[0]))
    await entered.wait()
    second = asyncio.create_task(anext(tee[1]))
    try:
        await asyncio.sleep(0)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        release.set()
        assert await second == 1
        assert [item async for item in tee[1]] == [2, 3]
        assert [item async for item in tee[2]] == [1, 2, 3]
        assert pulls == [1, 2, 3]
    finally:
        release.set()
        first.cancel()
        second.cancel()
        await asyncio.gather(first, second, return_exceptions=True)
        await tee.aclose()


async def test_source_error_after_reader_cancellation_reaches_all_remaining_peers():
    entered = asyncio.Event()
    release = asyncio.Event()
    error = ValueError("upstream failure")

    async def source():
        entered.set()
        await release.wait()
        yield 1
        raise error

    tee = Tee(source(), n=3)
    reader = asyncio.create_task(anext(tee[0]))
    await entered.wait()
    try:
        reader.cancel()
        with pytest.raises(asyncio.CancelledError):
            await reader
        release.set()
        for peer in tee[1:]:
            assert await anext(peer) == 1
            with pytest.raises(ValueError) as caught:
                await anext(peer)
            assert caught.value is error
    finally:
        release.set()
        reader.cancel()
        await asyncio.gather(reader, return_exceptions=True)
        await tee.aclose()


async def test_last_reader_cancellation_closes_pending_source():
    entered = asyncio.Event()
    closed = asyncio.Event()

    async def source():
        try:
            entered.set()
            await asyncio.Future()
            yield 1
        finally:
            closed.set()

    tee = Tee(source(), n=1)
    reader = asyncio.create_task(anext(tee[0]))
    await entered.wait()
    try:
        reader.cancel()
        with pytest.raises(asyncio.CancelledError):
            await reader
        assert closed.is_set()
    finally:
        reader.cancel()
        await asyncio.gather(reader, return_exceptions=True)
        await tee.aclose()


async def test_parent_close_cancels_and_awaits_pending_source():
    entered = asyncio.Event()
    closed = asyncio.Event()

    async def source():
        try:
            entered.set()
            await asyncio.Future()
            yield 1
        finally:
            closed.set()

    tee = Tee(source(), n=2)
    readers = [asyncio.create_task(anext(peer)) for peer in tee]
    await entered.wait()
    try:
        await tee.aclose()
        assert closed.is_set()
        await asyncio.gather(*readers, return_exceptions=True)
    finally:
        for reader in readers:
            reader.cancel()
        await asyncio.gather(*readers, return_exceptions=True)
        await tee.aclose()


async def test_upstream_context_survives_yields_and_early_close():
    context = contextvars.ContextVar("upstream-context", default="caller")

    async def run_case(early_close):
        closed = False
        seen = []

        async def source():
            nonlocal closed
            token = context.set("source")
            try:
                seen.append(context.get())
                yield 1
                seen.append(context.get())
                yield 2
            finally:
                context.reset(token)
                closed = True

        tee = Tee(source(), n=1)
        try:
            assert await anext(tee[0]) == 1
            if not early_close:
                assert [item async for item in tee[0]] == [2]
            await tee.aclose()
            assert closed
            assert seen == (["source"] if early_close else ["source", "source"])
        finally:
            await tee.aclose()

    await run_case(False)
    await run_case(True)


async def test_source_does_not_prefetch_and_closes_in_its_own_context():
    pulls = []
    context = contextvars.ContextVar("source-token", default="caller")
    closed = False

    async def source():
        nonlocal closed
        token = context.set("source")
        try:
            for item in [1, 2, 3]:
                pulls.append(item)
                yield item
        finally:
            context.reset(token)
            closed = True

    tee = Tee(source(), n=2)
    try:
        assert await anext(tee[0]) == 1
        await asyncio.sleep(0)
        assert pulls == [1]
        assert await anext(tee[1]) == 1
        await asyncio.sleep(0)
        assert pulls == [1]
        await tee.aclose()
        assert closed
    finally:
        await tee.aclose()


async def test_completed_pull_is_retained_when_its_reader_is_cancelled():
    pulls = []

    async def source():
        pulls.append(1)
        # Cancel the reader after this pull finishes but before it can observe
        # the completed response and broadcast its item.
        asyncio.get_running_loop().call_soon(reader.cancel)
        yield 1
        pulls.append(2)
        yield 2

    tee = Tee(source(), n=2)
    reader = asyncio.create_task(anext(tee[0]))
    try:
        with pytest.raises(asyncio.CancelledError):
            await reader
        assert [item async for item in tee[1]] == [1, 2]
        assert pulls == [1, 2]
    finally:
        reader.cancel()
        await asyncio.gather(reader, return_exceptions=True)
        await tee.aclose()


async def test_cancelled_close_does_not_interrupt_shared_cleanup():
    entered = asyncio.Event()
    closing = asyncio.Event()
    finish_close = asyncio.Event()
    closed = False

    async def source():
        nonlocal closed
        try:
            entered.set()
            await asyncio.Future()
            yield 1
        finally:
            closing.set()
            await finish_close.wait()
            closed = True

    tee = Tee(source(), n=2)
    reader = asyncio.create_task(anext(tee[0]))
    await entered.wait()
    first_close = asyncio.create_task(tee.aclose())
    second_close = None
    try:
        await closing.wait()
        second_close = asyncio.create_task(tee.aclose())
        first_close.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first_close
        await asyncio.sleep(0)
        assert not second_close.done()
        finish_close.set()
        await second_close
        assert closed
    finally:
        finish_close.set()
        reader.cancel()
        await asyncio.gather(reader, first_close, return_exceptions=True)
        if second_close is not None:
            await second_close
        await tee.aclose()


async def test_parent_close_before_producer_starts_closes_source():
    class Source:
        closed = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            await asyncio.Future()

        async def aclose(self):
            self.closed = True

    source = Source()
    tee = Tee(source, n=2)
    reader = asyncio.create_task(anext(tee[0]))
    try:
        # The reader schedules the producer after this task's next step.
        await asyncio.sleep(0)
        await tee.aclose()
        assert source.closed
    finally:
        reader.cancel()
        await asyncio.gather(reader, return_exceptions=True)
        await tee.aclose()


@pytest.mark.parametrize("num_peers", [1, 2])
async def test_terminal_peer_waits_for_cleanup_after_source_terminates(num_peers):
    async def run_case(error):
        entered_close = asyncio.Event()
        finish_close = asyncio.Event()

        class Source:
            closed = False

            def __aiter__(self):
                return self

            async def __anext__(self):
                raise error

            async def aclose(self):
                entered_close.set()
                await finish_close.wait()
                self.closed = True

        source = Source()
        tee = Tee(source, n=num_peers)
        reader = asyncio.create_task(anext(tee[0]))
        try:
            await entered_close.wait()
            await asyncio.sleep(0)
            assert not reader.done()
            finish_close.set()
            with pytest.raises(type(error)):
                await reader
            assert source.closed
        finally:
            finish_close.set()
            await asyncio.gather(reader, return_exceptions=True)
            await tee.aclose()

    await run_case(StopAsyncIteration())
    await run_case(ValueError("upstream failure"))


@pytest.mark.parametrize("error", [StopAsyncIteration(), ValueError("upstream failure")])
async def test_cancelled_terminal_reader_does_not_cancel_cleanup(error):
    entered = asyncio.Event()
    release = asyncio.Event()
    close_count = 0

    class Source:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise error

        async def aclose(self):
            nonlocal close_count
            entered.set()
            await release.wait()
            close_count += 1

    tee = Tee(Source(), n=2)
    reader = asyncio.create_task(anext(tee[0]))
    try:
        await entered.wait()
        await asyncio.sleep(0)
        reader.cancel()
        with pytest.raises(asyncio.CancelledError):
            await reader
        assert close_count == 0
        release.set()
        with pytest.raises(type(error)) as caught:
            await anext(tee[1])
        if not isinstance(error, StopAsyncIteration):
            assert caught.value is error
        assert close_count == 1
    finally:
        release.set()
        reader.cancel()
        await asyncio.gather(reader, return_exceptions=True)
        await tee.aclose()


async def test_completed_error_is_retained_when_its_reader_is_cancelled():
    error = ValueError("upstream failure")

    async def source():
        asyncio.get_running_loop().call_soon(reader.cancel)
        raise error
        yield

    tee = Tee(source(), n=3)
    reader = asyncio.create_task(anext(tee[0]))
    try:
        with pytest.raises(asyncio.CancelledError):
            await reader
        for peer in tee[1:]:
            with pytest.raises(ValueError) as caught:
                await anext(peer)
            assert caught.value is error
    finally:
        reader.cancel()
        await asyncio.gather(reader, return_exceptions=True)
        await tee.aclose()


async def test_cancelling_peer_waiting_for_lock_preserves_active_pull():
    entered = asyncio.Event()
    release = asyncio.Event()

    async def source():
        entered.set()
        await release.wait()
        yield 1

    tee = Tee(source(), n=3)
    reader = asyncio.create_task(anext(tee[0]))
    await entered.wait()
    waiter = asyncio.create_task(anext(tee[1]))
    try:
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        release.set()
        assert await reader == 1
        assert [item async for item in tee[0]] == []
        assert [item async for item in tee[2]] == [1]
    finally:
        release.set()
        reader.cancel()
        waiter.cancel()
        await asyncio.gather(reader, waiter, return_exceptions=True)
        await tee.aclose()


async def test_source_cleanup_error_reaches_last_peer():
    error = ValueError("cleanup failure")

    class Source:
        yielded = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.yielded:
                raise StopAsyncIteration
            self.yielded = True
            return 1

        async def aclose(self):
            raise error

    tee = Tee(Source(), n=2)
    try:
        assert [item async for item in tee[0]] == [1]
        assert await anext(tee[1]) == 1
        with pytest.raises(ValueError) as caught:
            await anext(tee[1])
        assert caught.value is error
    finally:
        await tee.aclose()
