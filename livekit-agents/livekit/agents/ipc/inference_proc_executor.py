from __future__ import annotations

import asyncio
import contextlib
import logging
import socket
import sys
import threading
from dataclasses import dataclass
from multiprocessing.context import BaseContext
from typing import Any

from .. import utils
from ..log import logger
from ..utils.aio import duplex_unix
from . import channel, inference_proc_lazy_main, proto
from .log_queue import LogQueueListener

from ..inference_runner import _InferenceRunner
from .inference_executor import InferenceExecutor, _InferenceRunnerClient


@dataclass
class _ProcOpts:
    mp_ctx: BaseContext
    initialize_timeout: float
    close_timeout: float


class InferenceProcExecutor(InferenceExecutor):
    def __init__(
        self,
        *,
        mp_ctx: BaseContext,
        loop: asyncio.AbstractEventLoop,
        initialize_timeout: float = 60.0,
        close_timeout: float = 2.5,
    ) -> None:
        self._loop = loop
        self._opts = _ProcOpts(
            initialize_timeout=initialize_timeout,
            close_timeout=close_timeout,
            mp_ctx=mp_ctx,
        )

        self._exitcode: int | None = None
        self._pid: int | None = None

        self._main_atask: asyncio.Task[None] | None = None
        self._closing = False
        self._kill_sent = False
        self._initialize_fut = asyncio.Future[None]()

        self._lock = asyncio.Lock()

    @property
    def exitcode(self) -> int | None:
        return self._exitcode

    @property
    def killed(self) -> bool:
        return self._kill_sent

    @property
    def pid(self) -> int | None:
        return self._pid

    @property
    def started(self) -> bool:
        return self._main_atask is not None

    async def start(self) -> None:
        if self.started:
            raise RuntimeError("process already started")

        if self._closing:
            raise RuntimeError("process is closed")

        await asyncio.shield(self._start())

    async def _start(self) -> None:
        def _add_proc_ctx_log(record: logging.LogRecord) -> None:
            extra = self.logging_extra()
            for key, value in extra.items():
                setattr(record, key, value)

        async with self._lock:
            self._pong_timeout = utils.aio.sleep(proto.PING_TIMEOUT)

            mp_pch, mp_cch = socket.socketpair()
            mp_log_pch, mp_log_cch = socket.socketpair()

            self._pch = await duplex_unix._AsyncDuplex.open(mp_pch)

            log_pch = duplex_unix._Duplex.open(mp_log_pch)
            log_listener = LogQueueListener(log_pch, _add_proc_ctx_log)
            log_listener.start()

            self._proc_args = inference_proc_lazy_main.ProcStartArgs(
                log_cch=mp_log_cch,
                mp_cch=mp_cch,
                asyncio_debug=self._loop.get_debug(),
                runners=_InferenceRunner.registered_runners,
            )

            self._inf_client = _InferenceRunnerClient(cch=self._pch)

            self._proc = self._opts.mp_ctx.Process(  # type: ignore
                target=inference_proc_lazy_main.proc_main,
                args=(self._proc_args,),
                name="inference_proc",
            )

            self._proc.start()
            mp_log_cch.close()
            mp_cch.close()

            self._pid = self._proc.pid
            self._join_fut = asyncio.Future[None]()

            def _sync_run():
                self._proc.join()
                log_listener.stop()
                try:
                    self._loop.call_soon_threadsafe(self._join_fut.set_result, None)
                except RuntimeError:
                    pass

            thread = threading.Thread(target=_sync_run, name="proc_join_thread")
            thread.start()
            self._main_atask = asyncio.create_task(self._main_task())

    async def join(self) -> None:
        if not self.started:
            raise RuntimeError("process not started")

        async with self._lock:
            if self._main_atask:
                await asyncio.shield(self._main_atask)

    async def do_inference(self, method: str, data: bytes) -> bytes | None:
        if not self.started:
            raise RuntimeError("process not started")

        return await self._inf_client.do_inference(method, data)

    async def initialize(self) -> None:
        await channel.asend_message(self._pch, proto.InitializeRequest())

        # wait for the process to become ready
        try:
            init_res = await asyncio.wait_for(
                channel.arecv_message(self._pch, proto.IPC_MESSAGES),
                timeout=self._opts.initialize_timeout,
            )
            assert isinstance(
                init_res, proto.InitializeResponse
            ), "first message must be InitializeResponse"
        except asyncio.TimeoutError:
            self._initialize_fut.set_exception(
                asyncio.TimeoutError("process initialization timed out")
            )
            logger.error(
                "initialization timed out, killing process", extra=self.logging_extra()
            )
            self._send_kill_signal()
            raise
        except Exception as e:  # should be channel.ChannelClosed most of the time
            self._initialize_fut.set_exception(e)
            raise
        else:
            self._initialize_fut.set_result(None)

    async def aclose(self) -> None:
        if not self.started:
            return

        self._closing = True
        with contextlib.suppress(utils.aio.duplex_unix.DuplexClosed):
            await channel.asend_message(self._pch, proto.ShutdownRequest())

        try:
            if self._main_atask:
                await asyncio.wait_for(
                    asyncio.shield(self._main_atask), timeout=self._opts.close_timeout
                )
        except asyncio.TimeoutError:
            logger.error(
                "process did not exit in time, killing process",
                extra=self.logging_extra(),
            )
            self._send_kill_signal()

        async with self._lock:
            if self._main_atask:
                await asyncio.shield(self._main_atask)

    async def kill(self) -> None:
        if not self.started:
            raise RuntimeError("process not started")

        self._closing = True
        self._send_kill_signal()

        async with self._lock:
            if self._main_atask:
                await asyncio.shield(self._main_atask)

    def _send_kill_signal(self) -> None:
        try:
            if not self._proc.is_alive():
                return
        except ValueError:
            return

        logger.info("killing process", extra=self.logging_extra())
        if sys.platform == "win32":
            self._proc.terminate()
        else:
            self._proc.kill()

        self._kill_sent = True

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        try:
            await self._initialize_fut
        except asyncio.TimeoutError:
            pass  # this happens when the initialization takes longer than self._initialize_timeout
        except Exception:
            pass  # initialization failed

        # the process is killed if it doesn't respond to ping requests
        ping_task = asyncio.create_task(self._ping_pong_task())
        monitor_task = asyncio.create_task(self._monitor_task())

        await self._join_fut
        self._exitcode = self._proc.exitcode
        self._proc.close()
        await utils.aio.gracefully_cancel(ping_task, monitor_task)

        with contextlib.suppress(duplex_unix.DuplexClosed):
            await self._pch.aclose()

        if self._exitcode != 0 and not self._kill_sent:
            logger.error(
                f"inference process exited with non-zero exit code {self.exitcode}",
                extra=self.logging_extra(),
            )

    @utils.log_exceptions(logger=logger)
    async def _monitor_task(self) -> None:
        while True:
            try:
                msg = await channel.arecv_message(self._pch, proto.IPC_MESSAGES)
            except utils.aio.duplex_unix.DuplexClosed:
                break

            if isinstance(msg, proto.PongResponse):
                delay = utils.time_ms() - msg.timestamp
                if delay > proto.HIGH_PING_THRESHOLD * 1000:
                    logger.warning(
                        "inference process is unresponsive",
                        extra={"delay": delay, **self.logging_extra()},
                    )

                with contextlib.suppress(utils.aio.SleepFinished):
                    self._pong_timeout.reset()

            if isinstance(msg, proto.InferenceResponse):
                self._inf_client._on_inference_response(msg)

    @utils.log_exceptions(logger=logger)
    async def _ping_pong_task(self) -> None:
        ping_interval = utils.aio.interval(proto.PING_INTERVAL)

        async def _send_ping_co():
            while True:
                await ping_interval.tick()
                try:
                    await channel.asend_message(
                        self._pch, proto.PingRequest(timestamp=utils.time_ms())
                    )
                except utils.aio.duplex_unix.DuplexClosed:
                    break

        async def _pong_timeout_co():
            while True:
                await self._pong_timeout
                logger.error(
                    "inference process is unresponsive, killing proc",
                    extra=self.logging_extra(),
                )
                self._pong_timeout = utils.aio.sleep(proto.PING_TIMEOUT)

        tasks = [
            asyncio.create_task(_send_ping_co()),
            asyncio.create_task(_pong_timeout_co()),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            await utils.aio.gracefully_cancel(*tasks)

    def logging_extra(self):
        extra: dict[str, Any] = {
            "pid": self.pid,
            "inference_proc": True,
        }
        return extra
