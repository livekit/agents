from __future__ import annotations  # noqa: I001

import asyncio
import pathlib
import signal
import sys
import threading


from .. import utils
from ..log import logger
from ..worker import Worker
from . import proto
from .log import setup_logging


def run_dev(args: proto.CliArgs) -> None:
    if args.watch:
        from .watcher import WatchServer

        setup_logging(args.log_level, args.devmode, args.console)
        main_file = pathlib.Path(sys.argv[0]).parent

        async def _run_loop() -> None:
            server = WatchServer(run_worker, main_file, args, loop=asyncio.get_event_loop())
            await server.run()

        try:
            asyncio.run(_run_loop())
        except KeyboardInterrupt:
            pass
    else:
        run_worker(args)


def _esc(*codes: int) -> str:
    return "\033[" + ";".join(str(c) for c in codes) + "m"


def run_worker(args: proto.CliArgs, *, jupyter: bool = False) -> None:
    setup_logging(args.log_level, args.devmode, args.console)
    args.opts.validate_config(args.devmode)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    if args.console:
        print(_esc(34) + "=" * 50 + _esc(0))
        print(_esc(34) + "     Livekit Agents - Console" + _esc(0))
        print(_esc(34) + "=" * 50 + _esc(0))
        print("Press [Ctrl+B] to toggle between Text/Audio mode, [Q] to quit.\n")

    worker = Worker(args.opts, devmode=args.devmode, register=args.register, loop=loop)

    loop.set_debug(args.asyncio_debug)
    loop.slow_callback_duration = 0.1  # 100ms
    utils.aio.debug.hook_slow_callbacks(2)

    @worker.once("worker_started")
    def _worker_started() -> None:
        if args.simulate_job and args.reload_count == 0:
            loop.create_task(worker.simulate_job(args.simulate_job))

        if args.devmode:
            logger.info(
                f"{_esc(1)}see tracing information at http://localhost:{worker.worker_info.http_port}/debug{_esc(0)}"
            )
        else:
            logger.info(
                f"see tracing information at http://localhost:{worker.worker_info.http_port}/debug"
            )

    try:

        def _signal_handler() -> None:
            raise KeyboardInterrupt

        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, _signal_handler)

    except NotImplementedError:
        # TODO(theomonnom): add_signal_handler is not implemented on win
        pass

    async def _worker_run(worker: Worker) -> None:
        try:
            await worker.run()
        except Exception:
            logger.exception("worker failed")

    watch_client = None
    if args.watch:
        from .watcher import WatchClient

        watch_client = WatchClient(worker, args, loop=loop)
        watch_client.start()

    try:
        main_task = loop.create_task(_worker_run(worker), name="agent_runner")
        try:
            loop.run_until_complete(main_task)
        except KeyboardInterrupt:
            pass

        try:
            if not args.devmode:
                loop.run_until_complete(worker.drain(timeout=args.opts.drain_timeout))

            loop.run_until_complete(worker.aclose())

            if watch_client:
                loop.run_until_complete(watch_client.aclose())
        except KeyboardInterrupt:
            if not jupyter:
                logger.warning("exiting forcefully")
                import os

                os._exit(1)  # TODO(theomonnom): add aclose(force=True) in worker
    finally:
        if jupyter:
            loop.close()  # close can only be called from the main thread
            return  # noqa: B012

        try:
            tasks = asyncio.all_tasks(loop)
            for task in tasks:
                task.cancel()

            loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.run_until_complete(loop.shutdown_default_executor())
        finally:
            loop.close()
