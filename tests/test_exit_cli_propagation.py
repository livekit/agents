"""Regression tests for #5856: _ExitCli must escape asyncio's Handle._run.

``_handle_exit`` raises ``_ExitCli`` synchronously from a signal handler; the
frame that raise lands in is whatever the event loop happened to be executing.
When that frame is a loop callback, CPython's ``Handle._run`` re-raises only
``SystemExit``/``KeyboardInterrupt`` and reports every other ``BaseException``
to the loop exception handler — logging it and leaving the loop running, so the
CLI's drain/shutdown path is never reached.
"""

import asyncio

import pytest

from livekit.agents.cli import _legacy
from livekit.agents.cli.cli import _ExitCli

pytestmark = pytest.mark.unit


class TestExitCliPropagation:
    def test_is_systemexit(self) -> None:
        # the only exception types Handle._run re-raises are SystemExit and
        # KeyboardInterrupt; _ExitCli must be one of them to survive a raise
        # that lands inside a loop callback
        assert issubclass(_ExitCli, SystemExit)
        assert issubclass(_legacy._ExitCli, SystemExit)

    def test_propagates_out_of_a_loop_callback(self) -> None:
        # simulates SIGTERM landing while the loop is mid-callback: the raise
        # happens inside Handle._run. Before the fix the loop swallowed it and
        # kept running until the 0.5s stop; now run_forever raises immediately.
        loop = asyncio.new_event_loop()
        try:

            def _raise_exit() -> None:
                raise _ExitCli()

            swallowed: list[dict] = []
            loop.set_exception_handler(lambda _loop, ctx: swallowed.append(ctx))
            loop.call_soon(_raise_exit)
            loop.call_later(0.5, loop.stop)  # safety stop for the broken case

            with pytest.raises(_ExitCli):
                loop.run_forever()

            assert not swallowed, "exit was reported to the exception handler instead of raised"
        finally:
            loop.close()

    def test_still_caught_by_except_exit_cli(self) -> None:
        # _run_worker's `except _ExitCli:` handling must keep working
        try:
            raise _ExitCli()
        except _ExitCli:
            caught = True
        assert caught

    def test_propagates_out_of_a_task(self) -> None:
        # #6724: SIGTERM landing while the main thread is inside a job-request task
        # (a request_fnc doing blocking work) puts the raise in a coroutine, not a
        # callback. Task.__step re-raises only SystemExit/KeyboardInterrupt out of the
        # loop; any other BaseException is stored on the task ("Task exception was never
        # retrieved") and the loop keeps running, so the worker never drains.
        loop = asyncio.new_event_loop()
        try:

            async def _job_request_task() -> None:
                raise _ExitCli()

            async def _main() -> None:
                await asyncio.sleep(30)  # the worker's main task, still running

            main_task = loop.create_task(_main())
            loop.call_soon(lambda: loop.create_task(_job_request_task()))

            swallowed: list[dict] = []
            loop.set_exception_handler(lambda _loop, ctx: swallowed.append(ctx))
            loop.call_later(0.5, loop.stop)  # safety stop for the broken case

            with pytest.raises(_ExitCli):
                loop.run_until_complete(main_task)

            assert not swallowed, "exit was left on the task instead of reaching the loop"
        finally:
            main_task.cancel()
            loop.close()
