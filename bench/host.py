"""The Python side of `bench/host.rs`: the loop host calls run on, and what they call."""

import asyncio
import threading
from types import SimpleNamespace


def host():
    """A loop thread and the callables the benchmark times."""
    import contextvars
    import time

    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def run():
        asyncio.set_event_loop(loop)
        loop.call_soon(ready.set)
        loop.run_forever()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    ready.wait()

    variables = [contextvars.ContextVar(f"bench_{i}") for i in range(64)]

    def capture(depth):
        """The (loop, context) pair a call runs in, with `depth` context variables set."""

        def enter():
            for var in variables[:depth]:
                var.set(depth)
            return contextvars.copy_context()

        return (loop, contextvars.Context().run(enter))

    def sync_noop():
        return None

    async def async_noop():
        return None

    async def async_yield():
        await asyncio.sleep(0)

    def threadsafe_baseline(iterations, kind):
        """Nanoseconds per call of the pure-Python crossing: a foreign thread runs one call on
        the loop and waits for its result, as a Rust caller does."""
        if kind == "sync":

            async def call():
                return sync_noop()

        else:
            call = {"async": async_noop, "yield": async_yield}[kind]
        started = time.perf_counter_ns()
        for _ in range(iterations):
            asyncio.run_coroutine_threadsafe(call(), loop).result()
        return (time.perf_counter_ns() - started) / iterations

    async def items(count, size):
        item = b"\0" * size if size else "token"
        for _ in range(count):
            yield item

    def stream(count, size):
        return items(count, size)

    class Sink:
        def __init__(self):
            self.received = 0
            self.ended = asyncio.Event()

        def __aiter__(self):
            return self._output()

        async def _output(self):
            await self.ended.wait()
            yield str(self.received)

        def _core_push(self, item):
            self.received += 1

        def _core_flush(self):
            pass

        def _core_end_input(self):
            self.ended.set()

    cancelled = threading.Event()
    entered = threading.Event()

    async def blocked():
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    class Lag:
        """Absolute-deadline heartbeat on the loop: how late each 1ms wake ran."""

        def __init__(self):
            self.samples = []
            self.task = None

        def start(self):
            def begin():
                self.samples.clear()
                self.task = loop.create_task(self._run())

            loop.call_soon_threadsafe(begin)

        async def _run(self):
            period = 0.001
            deadline = loop.time() + period
            while True:
                await asyncio.sleep(max(0, deadline - loop.time()))
                now = loop.time()
                self.samples.append(max(0, int((now - deadline) * 1e9)))
                deadline += (max(0, int((now - deadline) / period)) + 1) * period

        def stop(self):
            done = threading.Event()

            def end():
                self.task.cancel()
                done.set()

            loop.call_soon_threadsafe(end)
            done.wait(2)
            return list(self.samples)

    def close():
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2)

    return SimpleNamespace(
        loop=loop,
        capture=capture,
        sync_noop=sync_noop,
        async_noop=async_noop,
        async_yield=async_yield,
        threadsafe_baseline=threadsafe_baseline,
        stream=stream,
        Sink=Sink,
        blocked=blocked,
        entered=entered,
        cancelled=cancelled,
        lag=Lag(),
        close=close,
    )
