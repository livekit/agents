"""Regression tests for #5856 / #6724: SIGTERM must reliably shut the worker down.

``signal.signal`` handlers run on the main thread at an arbitrary bytecode
boundary. The old handler raised ``_ExitCli`` directly, so the raise could land
inside whatever asyncio task or callback the loop happened to be executing —
asyncio stores any non-SystemExit/KeyboardInterrupt exception on the task
("Task exception was never retrieved") and the loop keeps running, so the
worker never drains and keeps accepting jobs.

The fix has three parts, each covered here:
  * ``_ExitCli`` derives from ``SystemExit``, the only class (besides
    KeyboardInterrupt) that ``Task.__step`` / ``Handle._run`` re-raise out of
    the event loop;
  * the first signal only *schedules* the exit on the loop instead of raising,
    so a healthy loop shuts down without any task being preempted;
  * a watchdog escalates when the loop is blocked by synchronous code (the
    #6724 repro: a ``request_fnc`` calling ``time.sleep``): it re-signals the
    main thread so the raise lands in the frame that is blocking the loop.
"""

from __future__ import annotations

import asyncio
import contextlib
import signal
import threading
import time
from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

import pytest

from livekit.agents.cli import _legacy, proto
from livekit.agents.cli.cli import _ExitCli, _run_worker

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]


# ---------------------------------------------------------------------------
# _ExitCli propagation through asyncio — the SystemExit base class, adopted
# from PR #6624, covering the #5856 (callback) and #6724 (task) landing cases
# ---------------------------------------------------------------------------


def test_exit_cli_is_systemexit() -> None:
    # Task.__step / Handle._run re-raise only SystemExit and KeyboardInterrupt;
    # anything else is swallowed by the loop when the raise lands inside a task
    # or callback
    assert issubclass(_ExitCli, SystemExit)
    assert issubclass(_legacy._ExitCli, SystemExit)


def test_exit_cli_escapes_a_loop_callback() -> None:
    # 5856: the raise lands inside Handle._run (a loop callback)
    loop = asyncio.new_event_loop()
    try:

        def _raise_exit() -> None:
            raise _ExitCli()

        swallowed: list[dict[str, Any]] = []
        loop.set_exception_handler(lambda _loop, ctx: swallowed.append(ctx))
        loop.call_soon(_raise_exit)
        loop.call_later(0.5, loop.stop)  # safety stop for the broken case

        with pytest.raises(_ExitCli):
            loop.run_forever()

        assert not swallowed, "exit was reported to the exception handler instead of raised"
    finally:
        loop.close()


def test_exit_cli_escapes_a_task() -> None:
    # 6724: the raise lands inside a coroutine (e.g. _job_request_task while a
    # request_fnc blocks); Task.__step must re-raise it out of the loop
    loop = asyncio.new_event_loop()
    try:

        async def _job_request_task() -> None:
            raise _ExitCli()

        async def _main() -> None:
            await asyncio.sleep(30)  # the worker's main task, still running

        main_task = loop.create_task(_main())
        request_tasks: list[asyncio.Task[None]] = []
        loop.call_soon(lambda: request_tasks.append(loop.create_task(_job_request_task())))
        loop.call_later(0.5, loop.stop)  # safety stop for the broken case

        with pytest.raises(_ExitCli):
            loop.run_until_complete(main_task)
    finally:
        # drain the cancellation and retrieve the request task's exception so
        # closing the loop doesn't emit destroyed-pending / never-retrieved noise
        main_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, _ExitCli):
            loop.run_until_complete(main_task)
        for task in request_tasks:
            if task.done() and not task.cancelled():
                task.exception()
        loop.close()


# ---------------------------------------------------------------------------
# _run_worker end-to-end signal handling (stub server, real signals)
# ---------------------------------------------------------------------------


class _StubServer:
    """Stands in for AgentServer: run() until aclose(), with an optional task
    that blocks the event loop the way a synchronous request_fnc does."""

    def __init__(self, *, block_loop_for: float = 0.0) -> None:
        self._block_loop_for = block_loop_for
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stopped: asyncio.Event | None = None
        self.calls: list[str] = []
        self.run_active_during_drain = False

    async def run(self, *, devmode: bool, unregistered: bool) -> None:
        self._loop = asyncio.get_running_loop()
        self._stopped = asyncio.Event()
        if self._block_loop_for > 0:

            async def _blocking_job_request() -> None:
                time.sleep(self._block_loop_for)  # noqa: ASYNC251

            self._loop.create_task(_blocking_job_request())
        await self._stopped.wait()

    async def drain(self) -> None:
        self.calls.append("drain")
        # drain must run while run() is still active: jobs are joined by drain,
        # and aclose() is what ends run()
        self.run_active_during_drain = self._stopped is not None and not self._stopped.is_set()

    async def aclose(self) -> None:
        self.calls.append("aclose")
        if self._stopped is not None:
            self._stopped.set()

    def force_finish_threadsafe(self) -> None:
        """Test safety net: end run() so a regression fails by timing instead of
        hanging pytest forever."""
        if self._loop is not None and self._stopped is not None:
            with contextlib.suppress(RuntimeError):
                self._loop.call_soon_threadsafe(self._stopped.set)


@contextlib.contextmanager
def _run_worker_test_env() -> Iterator[None]:
    """_run_worker installs signal handlers (and leaves force-exit ones behind on
    purpose); restore the pytest process state afterwards."""
    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        with patch("livekit.agents.cli.cli.setup_logging"):
            yield
    finally:
        for sig, handler in saved.items():
            signal.signal(sig, handler)
        asyncio.set_event_loop(None)


def _sigterm_main_thread_after(delay: float) -> threading.Timer:
    main_id = threading.main_thread().ident
    assert main_id is not None
    timer = threading.Timer(delay, lambda: signal.pthread_kill(main_id, signal.SIGTERM))
    timer.daemon = True
    timer.start()
    return timer


@pytest.mark.skipif(not hasattr(signal, "pthread_kill"), reason="needs signal.pthread_kill (POSIX)")
def test_sigterm_triggers_graceful_shutdown() -> None:
    """One SIGTERM on a healthy loop: drain + aclose run and _run_worker returns."""
    server = _StubServer()
    args = proto.CliArgs(log_level="DEBUG", dev=False)

    with _run_worker_test_env():
        _sigterm_main_thread_after(0.2)
        safety = threading.Timer(8.0, server.force_finish_threadsafe)
        safety.daemon = True
        safety.start()

        start = time.monotonic()
        _run_worker(server, args)  # type: ignore[arg-type]
        elapsed = time.monotonic() - start
        safety.cancel()

    assert server.calls == ["drain", "aclose"]
    assert server.run_active_during_drain
    assert elapsed < 5.0, f"shutdown took {elapsed:.1f}s — the exit signal was lost"


@pytest.mark.skipif(not hasattr(signal, "pthread_kill"), reason="needs signal.pthread_kill (POSIX)")
def test_sigterm_with_blocked_event_loop_escalates() -> None:
    """The #6724 repro: SIGTERM arrives while a job-request task blocks the event
    loop with synchronous work. The watchdog must preempt the blocking frame so
    drain starts long before the blocking call would have finished."""
    block_for = 30.0
    server = _StubServer(block_loop_for=block_for)
    args = proto.CliArgs(log_level="DEBUG", dev=False)

    with (
        _run_worker_test_env(),
        patch("livekit.agents.cli.cli._EXIT_ESCALATION_TIMEOUT", 0.25),
    ):
        # deliver while time.sleep() holds the loop; the handler must not lose it
        _sigterm_main_thread_after(0.3)
        safety = threading.Timer(8.0, server.force_finish_threadsafe)
        safety.daemon = True
        safety.start()

        start = time.monotonic()
        _run_worker(server, args)  # type: ignore[arg-type]
        elapsed = time.monotonic() - start
        safety.cancel()

    assert server.calls == ["drain", "aclose"]
    assert server.run_active_during_drain
    assert elapsed < block_for / 2, (
        f"shutdown took {elapsed:.1f}s — the blocked loop was never preempted"
    )
