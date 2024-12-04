from __future__ import annotations

import asyncio
import contextlib
import multiprocessing as mp
import socket
from multiprocessing.context import BaseContext

from ..inference_runner import _RunnersDict
from ..log import logger
from ..utils import aio, log_exceptions, shortuuid
from . import channel, proto
from .inference_proc_lazy_main import ProcStartArgs, proc_main
from .supervised_proc import SupervisedProc


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
        )

        self._runners = runners
        self._active_requests: dict[str, asyncio.Future[proto.InferenceResponse]] = {}

    def _create_process(self, cch: socket.socket, log_cch: socket.socket) -> mp.Process:
        proc_args = ProcStartArgs(
            log_cch=log_cch,
            mp_cch=cch,
            runners=self._runners,
        )

        return self._mp_ctx.Process(  # type: ignore
            target=proc_main,
            args=(proc_args,),
            name="inference_proc",
        )

    @log_exceptions(logger=logger)
    async def _main_task(self, ipc_ch: aio.ChanReceiver[channel.Message]) -> None:
        async for msg in ipc_ch:
            if isinstance(msg, proto.InferenceResponse):
                fut = self._active_requests.pop(msg.request_id, None)
                if fut is None:
                    logger.warning(
                        "received unexpected inference response",
                        extra={"request_id": msg.request_id},
                    )
                    return

                with contextlib.suppress(asyncio.InvalidStateError):
                    fut.set_result(msg)

    async def do_inference(self, method: str, data: bytes) -> bytes | None:
        if not self.started:
            raise RuntimeError("process not started")

        request_id = shortuuid("inference_req_")
        fut = asyncio.Future[proto.InferenceResponse]()

        await channel.asend_message(
            self._pch,
            proto.InferenceRequest(request_id=request_id, method=method, data=data),
        )

        self._active_requests[request_id] = fut

        inf_resp = await fut
        if inf_resp.error:
            raise RuntimeError(f"inference of {method} failed: {inf_resp.error}")

        return inf_resp.data

    def logging_extra(self):
        extra = super().logging_extra()
        extra["inference"] = True
        return extra
