"""A virtual timer must respect physical work without abandoning deadlines."""

from __future__ import annotations

import asyncio
import threading
import time
import warnings
from collections.abc import Callable
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from typing import ParamSpec, TypeVar

import pytest

from .virtual_time import _VirtualTimeLoop

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]
_P = ParamSpec("_P")
_R = TypeVar("_R")


def current_loop() -> _VirtualTimeLoop:
    loop = asyncio.get_running_loop()
    assert isinstance(loop, _VirtualTimeLoop)
    return loop


def gate(entered: threading.Event, release: threading.Event) -> int:
    entered.set()
    if not release.wait(1):
        raise TimeoutError("The bounded offload gate was not released")
    return 17


async def entered(event: threading.Event) -> None:
    assert await asyncio.to_thread(event.wait, 0.5)


async def test_virtual_timers_still_autojump_without_real_work() -> None:
    loop = current_loop()
    start = time.monotonic()
    clock = loop.time()
    await asyncio.sleep(3600)
    assert loop.time() - clock >= 3600
    assert time.monotonic() - start < 0.2


async def test_borrowed_offload_can_depend_on_a_virtual_timer() -> None:
    loop = current_loop()
    ready, release = threading.Event(), threading.Event()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = loop.run_in_executor(executor, gate, ready, release)
        try:
            await entered(ready)
            start = time.monotonic()
            loop.call_later(0.03, release.set)
            assert await asyncio.wait_for(future, 0.5) == 17
            assert time.monotonic() - start >= 0.02
            assert not executor._shutdown
        finally:
            release.set()


async def test_cancelling_a_running_wrapper_keeps_physical_work_observed() -> None:
    loop = current_loop()
    ready, release = threading.Event(), threading.Event()
    future = loop.run_in_executor(None, gate, ready, release)
    try:
        await entered(ready)
        future.cancel()
        await asyncio.sleep(0)
        assert future.cancelled()
        assert loop._virtual_clock._real_work == 1
        start = time.monotonic()
        loop.call_later(0.03, release.set)
        assert await asyncio.to_thread(release.wait, 0.5)
        await asyncio.sleep(0)
        assert time.monotonic() - start >= 0.02
    finally:
        release.set()


async def test_cancelled_queued_work_releases_its_clock_observation() -> None:
    loop = current_loop()
    ready, release = threading.Event(), threading.Event()
    called: list[bool] = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        running = loop.run_in_executor(executor, gate, ready, release)
        try:
            await entered(ready)
            queued = loop.run_in_executor(executor, called.append, True)
            queued.cancel()
            await asyncio.sleep(0.002)
            assert queued.cancelled()
            assert not called
            assert loop._virtual_clock._real_work == 1
            release.set()
            assert await running == 17
        finally:
            release.set()


async def test_offload_timeout_advances_by_real_elapsed_time() -> None:
    loop = current_loop()
    ready, release = threading.Event(), threading.Event()
    future = loop.run_in_executor(None, gate, ready, release)
    try:
        await entered(ready)
        start = time.monotonic()
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(future, 0.03)
        assert time.monotonic() - start >= 0.02
        assert loop._virtual_clock._real_work == 1
    finally:
        release.set()


class RejectedExecutor(Executor):
    def submit(self, fn: Callable[_P, _R], /, *args: _P.args, **kwargs: _P.kwargs) -> Future[_R]:
        raise RuntimeError("Owned executor rejects submission")


class DeliveryGapFuture(Future[_R]):
    def __init__(self, release: threading.Event) -> None:
        super().__init__()
        self._release = release
        self._observed = False

    def add_done_callback(self, fn: Callable[[Future[_R]], object]) -> None:
        super().add_done_callback(fn)
        if not self._observed:
            self._observed = True

            def delay(_: Future[_R]) -> None:
                if not self._release.wait(1):
                    raise TimeoutError("The bounded completion delivery was not released")

            # Hold delivery between the physical observer and asyncio's chain.
            super().add_done_callback(delay)


class DeliveryGapExecutor(Executor):
    def __init__(self, release: threading.Event) -> None:
        self._release = release
        self._threads: list[threading.Thread] = []

    def submit(self, fn: Callable[_P, _R], /, *args: _P.args, **kwargs: _P.kwargs) -> Future[_R]:
        future: DeliveryGapFuture[_R] = DeliveryGapFuture(self._release)

        def execute() -> None:
            if future.set_running_or_notify_cancel():
                try:
                    result = fn(*args, **kwargs)
                except BaseException as error:
                    future.set_exception(error)
                else:
                    future.set_result(result)

        thread = threading.Thread(target=execute)
        self._threads.append(thread)
        thread.start()
        return future

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        if wait:
            for thread in self._threads:
                thread.join()


async def test_physical_completion_does_not_autojump_before_result_delivery() -> None:
    loop = current_loop()
    work, deliver = threading.Event(), threading.Event()
    executor = DeliveryGapExecutor(deliver)
    try:
        future = loop.run_in_executor(executor, gate, threading.Event(), work)
        start = time.monotonic()
        loop.call_later(0.03, deliver.set)
        work.set()
        assert await asyncio.wait_for(future, 0.5) == 17
        assert time.monotonic() - start >= 0.02
    finally:
        work.set()
        deliver.set()
        executor.shutdown(wait=True)


async def test_offload_deadline_includes_busy_ready_callback_time() -> None:
    loop = current_loop()
    ready, release = threading.Event(), threading.Event()
    future = loop.run_in_executor(None, gate, ready, release)

    async def busy() -> None:
        until = time.monotonic() + 0.09
        while time.monotonic() < until:
            tick = time.monotonic() + 0.001
            while time.monotonic() < tick:
                pass
            await asyncio.sleep(0)

    pump: asyncio.Task[None] | None = None
    try:
        await entered(ready)
        start = time.monotonic()
        pump = asyncio.create_task(busy())
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(future, 0.03)
        assert 0.02 <= time.monotonic() - start < 0.08
        assert not pump.done()
    finally:
        release.set()
        if pump is not None:
            pump.cancel()
            try:
                await pump
            except asyncio.CancelledError:
                pass


async def test_rejected_submission_does_not_disable_autojump() -> None:
    loop = current_loop()
    with pytest.raises(TypeError, match="ThreadPoolExecutor"):
        loop.set_default_executor(RejectedExecutor())
    with pytest.raises(RuntimeError, match="rejects submission"):
        loop.run_in_executor(RejectedExecutor(), lambda: None)
    assert loop._virtual_clock._real_work == 0
    start = time.monotonic()
    await asyncio.sleep(3600)
    assert time.monotonic() - start < 0.2


async def test_default_executor_shutdown_waits_for_its_physical_joiner() -> None:
    loop = current_loop()
    ready, release = threading.Event(), threading.Event()
    future = loop.run_in_executor(None, gate, ready, release)
    try:
        await entered(ready)
        future.cancel()
        start = time.monotonic()
        loop.call_later(0.03, release.set)
        with warnings.catch_warnings(record=True) as observed:
            await loop.shutdown_default_executor(timeout=0.5)
        assert not observed
        assert time.monotonic() - start >= 0.02
        assert loop._virtual_clock._real_work == 0
        with pytest.raises(RuntimeError, match="shutdown"):
            loop.run_in_executor(None, lambda: None)
    finally:
        release.set()


class DelayedShutdown(ThreadPoolExecutor):
    def __init__(self) -> None:
        super().__init__(max_workers=1)
        self.entered = threading.Event()
        self.release = threading.Event()

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        super().shutdown(wait=wait, cancel_futures=cancel_futures)
        if wait:
            self.entered.set()
            if not self.release.wait(1):
                raise TimeoutError("The bounded executor joiner was not released")


async def wait_for_event(event: threading.Event) -> None:
    while not event.is_set():
        await asyncio.sleep(0.001)


@pytest.mark.parametrize("cancel", [False, True], ids=["timeout", "cancelled"])
async def test_shutdown_observation_survives_its_cancelled_or_timed_out_waiter(
    cancel: bool,
) -> None:
    loop = current_loop()
    executor = DelayedShutdown()
    loop.set_default_executor(executor)
    try:
        # The submitted call is already finished. Only physical shutdown remains.
        await loop.run_in_executor(None, lambda: None)
        if cancel:
            closing = asyncio.create_task(loop.shutdown_default_executor(timeout=0.5))
            await asyncio.wait_for(wait_for_event(executor.entered), 0.5)
            closing.cancel()
            with pytest.raises(asyncio.CancelledError):
                await closing
        else:
            start = time.monotonic()
            with pytest.warns(RuntimeWarning, match="threads"):
                await loop.shutdown_default_executor(timeout=0.03)
            assert time.monotonic() - start >= 0.02
        assert loop._virtual_clock._real_work == 1
        start = time.monotonic()
        loop.call_later(0.03, executor.release.set)
        await asyncio.sleep(0.04)
        assert time.monotonic() - start >= 0.02
        assert loop._virtual_clock._real_work == 0
    finally:
        executor.release.set()
        executor.shutdown(wait=True)
