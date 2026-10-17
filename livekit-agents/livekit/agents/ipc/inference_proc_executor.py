from __future__ import annotations

import asyncio
import contextlib
import multiprocessing as mp
import socket
from multiprocessing.context import BaseContext
from typing import Any

from ..inference_runner import _RunnersDict
from ..log import logger
from ..utils import aio, log_exceptions, shortuuid
from . import channel, proto
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
        self._active_requests: dict[str, asyncio.Future[proto.InferenceResponse]] = {}
        # set once the IPC read loop has ended: the process is gone (or being torn
        # down) and no inference response can ever arrive anymore
        self._requests_closed = False

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

    @log_exceptions(logger=logger)
    async def _main_task(self, ipc_ch: aio.ChanReceiver[channel.Message]) -> None:
        try:
            async for msg in ipc_ch:
                if isinstance(msg, proto.InferenceResponse):
                    fut = self._active_requests.pop(msg.request_id, None)
                    if fut is None:
                        logger.warning(
                            "received unexpected inference response",
                            extra={"request_id": msg.request_id},
                        )
                        continue

                    with contextlib.suppress(asyncio.InvalidStateError):
                        fut.set_result(msg)
        finally:
            # the read task is the sole reader of the IPC channel; once it ends, the
            # inference process is gone (channel closed) or the executor is being torn
            # down. Reject the pending requests so do_inference() callers don't wait
            # forever for a response that can never arrive.
            self._fail_pending_requests()

    def _fail_pending_requests(self) -> None:
        self._requests_closed = True

        pending_requests, self._active_requests = self._active_requests, {}
        for fut in pending_requests.values():
            with contextlib.suppress(asyncio.InvalidStateError):
                fut.set_exception(RuntimeError("the inference process exited before responding"))

    async def do_inference(self, method: str, data: bytes) -> bytes | None:
        if not self.started:
            raise RuntimeError("process not started")

        if self._requests_closed:
            # the process is gone (or shutting down), a response would never arrive
            raise RuntimeError("the inference process exited before responding")

        request_id = shortuuid("inference_req_")
        fut = asyncio.Future[proto.InferenceResponse]()
        self._active_requests[request_id] = fut

        try:
            await channel.asend_message(
                self._pch,
                proto.InferenceRequest(request_id=request_id, method=method, data=data),
            )
        except Exception:
            if not fut.done():
                fut.cancel()
            self._active_requests.pop(request_id, None)
            raise

        inf_resp = await fut
        if inf_resp.error:
            raise RuntimeError(f"inference of {method} failed: {inf_resp.error}")

        return inf_resp.data

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
