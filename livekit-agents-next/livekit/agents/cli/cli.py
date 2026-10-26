from __future__ import annotations

import asyncio
import os
import signal
import threading
from types import FrameType
from typing import Any

from ..log import logger
from ..telemetry import loop_monitor
from ..worker import AgentServer, WorkerOptions
from . import proto
from .log import setup_logging

HANDLED_SIGNALS = (
    signal.SIGINT,
    signal.SIGTERM,
)

# how long the exit scheduled on the event loop may go unserved before the
# watchdog concludes the loop is blocked by synchronous code and preempts it
# with a raise (see _run_worker._handle_exit). Kept well under the ~30s
# SIGTERM→SIGKILL budget of common orchestrators so the drain still gets time.
_EXIT_ESCALATION_TIMEOUT = 3.0


class _ExitCli(SystemExit):
    # SystemExit rather than BaseException: a raise from the signal path can land
    # at an arbitrary bytecode boundary of the main thread, i.e. inside whatever
    # asyncio task or callback the loop happened to be executing. Task.__step and
    # Handle._run re-raise only SystemExit/KeyboardInterrupt out of the event
    # loop; any other BaseException is stored on the task / reported to the loop
    # exception handler, silently swallowing the exit (#5856, #6724).
    pass


def _run_worker(server: AgentServer, args: proto.CliArgs) -> None:
    kwargs: dict = {}
    if args.url:
        kwargs["ws_url"] = args.url
    if args.api_key:
        kwargs["api_key"] = args.api_key
    if args.api_secret:
        kwargs["api_secret"] = args.api_secret
    if kwargs:
        server.update_options(**kwargs)

    if args.simulation:
        server._simulation = True

    if args.cli_addr and not args.dev:
        raise ValueError("--cli-addr requires --dev")

    devmode = args.dev
    colored_logs = devmode or args.log_format == "colored"

    setup_logging(args.log_level, devmode=colored_logs, console=False, compact=args.simulation)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    loop.slow_callback_duration = 0.1  # 100ms
    # a blocked worker loop delays job dispatch; there is no session to attach a span to here,
    # so this one logs (and records the metric) only
    loop_monitor.start_monitoring(loop, name="worker", emit_spans=False)

    # exit signalling. A plain `signal.signal` handler runs on the main thread at
    # an arbitrary bytecode boundary — raising from it can land inside whatever
    # asyncio task or callback the loop is executing, killing an unrelated task
    # (bad mid-drain) or never reaching run_until_complete at all (#5856, #6724).
    # So the first signal only *schedules* the exit on the loop. If the scheduled
    # callback doesn't run within _EXIT_ESCALATION_TIMEOUT, the loop is blocked by
    # synchronous code (e.g. a request_fnc doing blocking I/O) and a raise is the
    # only thing that can interrupt it: the watchdog re-signals with the handler
    # switched to raise mode, so the raise lands in the frame that is actually
    # blocking the loop.
    exit_fut: asyncio.Future[None] = loop.create_future()
    exit_raised = False
    escalating = False
    escalation_watchdog: threading.Timer | None = None

    def _trigger_exit() -> None:
        # runs on the event loop: the loop is healthy, no preemption needed
        if escalation_watchdog is not None:
            escalation_watchdog.cancel()
        if not exit_fut.done():
            exit_fut.set_result(None)

    def _escalate_blocked_loop() -> None:
        # watchdog thread: _trigger_exit never ran within the timeout
        nonlocal escalating
        escalating = True
        main_thread_id = threading.main_thread().ident
        if hasattr(signal, "pthread_kill") and main_thread_id is not None:
            # deliver to the main thread at the OS level, so its blocking call
            # (e.g. time.sleep) is interrupted and the handler runs right away
            signal.pthread_kill(main_thread_id, signal.SIGTERM)
        else:
            # Windows has no pthread_kill; re-raise SIGINT rather than SIGTERM:
            # CPython's C-level handler sets the SIGINT event that time.sleep()
            # and other sigint-aware waits block on, so those are interrupted
            # immediately. Other blocking calls only observe the handler at the
            # main thread's next bytecode boundary — best effort there.
            signal.raise_signal(signal.SIGINT)

    def _handle_exit(sig: int, frame: FrameType | None) -> None:
        nonlocal exit_raised, escalating, escalation_watchdog
        if escalating:
            escalating = False
            # raise only while the scheduled exit is still unserved (the loop is
            # provably blocked): the raise then lands inside the blocking frame,
            # and _ExitCli (a SystemExit) is re-raised out of whatever task or
            # callback that is, so run_until_complete sees it. If exit_fut is
            # already resolved, the loop resumed and won the race with the
            # watchdog — raising here could land after run_until_complete
            # returned and skip the drain, so the stray re-signal is dropped.
            if not exit_fut.done():
                raise _ExitCli()
            return
        if exit_raised:
            os._exit(1)
        exit_raised = True
        loop.call_soon_threadsafe(_trigger_exit)
        escalation_watchdog = threading.Timer(_EXIT_ESCALATION_TIMEOUT, _escalate_blocked_loop)
        escalation_watchdog.daemon = True
        escalation_watchdog.start()

    for sig in HANDLED_SIGNALS:
        signal.signal(sig, _handle_exit)

    def _loop_exception_handler(loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
        if isinstance(context.get("exception"), _ExitCli):
            # a task preempted by the exit escalation; the shutdown path is
            # already running, don't log it as an unretrieved task exception
            return
        loop.default_exception_handler(context)

    loop.set_exception_handler(_loop_exception_handler)

    async def _worker_run(worker: AgentServer) -> None:
        try:
            await server.run(devmode=devmode, unregistered=False)
        except Exception:
            logger.exception("worker failed")

    watch_client = None
    if args.cli_addr:
        from .watcher import WatchClient

        watch_client = WatchClient(server, args.cli_addr, loop=loop)
        watch_client.start()

    try:
        main_task = loop.create_task(_worker_run(server), name="worker_main_task_cli")

        async def _wait_for_exit_or_main() -> None:
            # the worker keeps running during drain: an exit must end this wait
            # without cancelling main_task
            await asyncio.wait({main_task, exit_fut}, return_when=asyncio.FIRST_COMPLETED)

        try:
            loop.run_until_complete(_wait_for_exit_or_main())
        except _ExitCli:
            pass  # the escalation raise, surfaced through the event loop

        if escalation_watchdog is not None:
            escalation_watchdog.cancel()

        # Second Ctrl+C force-exits.
        def _force_exit(sig: int, frame: FrameType | None) -> None:
            nonlocal escalating
            if escalating:
                # a stray watchdog re-signal that lost the race with the loop
                # resuming — not the operator asking for a force exit
                escalating = False
                return
            logger.warning("exiting forcefully", extra={"signal": sig})
            os._exit(1)

        for sig in HANDLED_SIGNALS:
            signal.signal(sig, _force_exit)

        try:
            if not devmode:
                try:
                    loop.run_until_complete(server.drain())
                except asyncio.TimeoutError:
                    logger.warning("drain timed out, forcing shutdown")
            loop.run_until_complete(server.aclose())

            if watch_client:
                loop.run_until_complete(watch_client.aclose())
        except _ExitCli:
            pass  # stray from first signal — ignore
    finally:
        # Re-enable force exit for the final cleanup phase
        for sig in HANDLED_SIGNALS:
            signal.signal(sig, lambda *_: os._exit(1))

        try:
            tasks = asyncio.all_tasks(loop)
            for task in tasks:
                task.cancel()

            loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
        except Exception:
            pass
        finally:
            loop_monitor.stop_monitoring(loop)
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.run_until_complete(loop.shutdown_default_executor())
            except Exception:
                pass
            loop.close()


def run_app(server: AgentServer | WorkerOptions) -> None:
    """``python myagent.py start [--dev]``: the interface ``lk agent`` drives."""
    from ..__main__ import _dispatch, _parser

    if isinstance(server, WorkerOptions):
        server = AgentServer.from_server_options(server)
    _dispatch(server, _parser().parse_args())
