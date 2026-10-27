from __future__ import annotations

from multiprocessing import current_process

if current_process().name == "agents_inference_process":
    import signal

    # ignore signals in the jobs process (the parent process will handle them)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

    if hasattr(signal, "SIGUSR1"):
        from .proc_client import _dump_stack_traces

        signal.signal(signal.SIGUSR1, _dump_stack_traces)

import asyncio
import math
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .. import _core
from ..inference_runner import _RunnersDict
from ..log import logger
from .proc_client import _dump_stack_traces_impl, serve_parent


@dataclass
class ProcStartArgs:
    log_cch: socket.socket
    mp_cch: socket.socket
    runners: _RunnersDict


def proc_main(args: ProcStartArgs) -> None:
    serve_parent(args.mp_cch, _InferenceProc(args.runners), job=False)


class _InferenceProc:
    def __init__(self, runners: _RunnersDict) -> None:
        # create an instance of each runner (the ctor must not requires any argument)
        self._runners = {name: runner() for name, runner in runners.items()}
        self._executor = ThreadPoolExecutor(max_workers=math.ceil(_core.cpu_count()))

    def initialize(self, http_proxy: str | None) -> None:
        for runner in self._runners.values():
            try:
                logger.debug(
                    "initializing inference runner",
                    extra={"runner": runner.__class__.INFERENCE_METHOD},
                )
                start_time = time.perf_counter()
                runner.initialize()
                logger.debug(
                    "inference runner initialized",
                    extra={
                        "runner": runner.__class__.INFERENCE_METHOD,
                        "elapsed_time": time.perf_counter() - start_time,
                    },
                )
            except Exception:
                logger.exception(
                    "error initializing inference runner",
                    extra={"runner": runner.__class__.INFERENCE_METHOD},
                )

    # --- the parent's messages, as the core's IpcClient serves them -------------------------

    async def _core_dump_stacks(self) -> None:
        _dump_stack_traces_impl()

    async def _core_inference(self, method: str, data: bytes) -> bytes | None:
        if method not in self._runners:
            logger.warning("unknown inference method", extra={"method": method})
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._runners[method].run, data)

    async def aclose(self) -> None:
        pass
