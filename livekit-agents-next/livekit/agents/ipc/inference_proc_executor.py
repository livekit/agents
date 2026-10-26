from __future__ import annotations

import asyncio
import multiprocessing as mp
import socket
from multiprocessing.context import BaseContext
from typing import Any

from ..inference_runner import _RunnersDict
from .inference_proc_lazy_main import ProcStartArgs, proc_main
from .supervised_proc import SupervisedProc, SupervisedProcKind


class InferenceProcExecutor(SupervisedProc):
    def __init__(
        self,
        *,
        runners: _RunnersDict,
        initialize_timeout: float,
        close_timeout: float,
        memory_warn_mb: float,
        memory_limit_mb: float,
        ping_interval: float,
        ping_timeout: float,
        high_ping_threshold: float,
        mp_ctx: BaseContext,
        loop: asyncio.AbstractEventLoop,
        http_proxy: str | None,
    ) -> None:
        super().__init__(
            initialize_timeout=initialize_timeout,
            close_timeout=close_timeout,
            memory_warn_mb=memory_warn_mb,
            memory_limit_mb=memory_limit_mb,
            ping_interval=ping_interval,
            ping_timeout=ping_timeout,
            high_ping_threshold=high_ping_threshold,
            mp_ctx=mp_ctx,
            loop=loop,
            http_proxy=http_proxy,
        )

        self._runners = runners

    @property
    def process_kind(self) -> SupervisedProcKind:
        return SupervisedProcKind.INFERENCE

    def _create_process(self, cch: socket.socket, log_cch: socket.socket) -> mp.Process:
        proc_args = ProcStartArgs(
            log_cch=log_cch,
            mp_cch=cch,
            runners=self._runners,
        )

        return self._mp_ctx.Process(  # type: ignore
            target=proc_main,
            args=(proc_args,),
            name="agents_inference_process",
        )

    async def do_inference(self, method: str, data: bytes) -> bytes | None:
        if not self.started:
            raise RuntimeError("process not started")

        return await self._pch.do_inference(method, data)  # type: ignore[no-any-return]

    def logging_extra(self) -> dict[str, Any]:
        extra = super().logging_extra()
        extra["inference"] = True
        return extra

    def is_alive(self) -> bool:
        try:
            return self._proc.is_alive()
        except ValueError:
            # the process object is closed after _supervise_task exits
            return False
