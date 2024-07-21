from __future__ import annotations

import asyncio
import logging
import contextlib
import multiprocessing as mp
import sys
import threading
from multiprocessing.context import ForkServerContext, SpawnContext
from typing import Any, Callable, Coroutine

from .. import utils
from ..job import JobContext, JobProcess, RunningJobInfo
from ..log import logger
from . import channel, proc_main, proto


class LogQueueListener:
    _sentinel = None

    def __init__(self, queue: mp.Queue) -> None:
        self._thread: threading.Thread | None = None
        self._q = queue

    def start(self) -> None:
        self._thread = t = threading.Thread(target=self._monitor, daemon=True)
        t.start()

    def stop(self) -> None:
        self._q.put_nowait(self._sentinel)
        self._thread.join()
        self._thread = None

    def handle(self, record: logging.LogRecord) -> None:
        handlers = logging.root.handlers
        for handler in handlers:
            if record.levelno >= handler.level:
                handler.handle(record)

    def _monitor(self):
        while True:
            record = self._q.get()
            if record is self._sentinel:
                break

            self.handle(record)


class SupervisedProc:
    def __init__(
        self,
        *,
        initialize_process_fnc: Callable[[JobProcess], Any],
        job_entrypoint_fnc: Callable[[JobContext], Coroutine],
        job_shutdown_fnc: Callable[[JobContext], Coroutine],
        initialize_timeout: float,
        close_timeout: float,
        mp_ctx: SpawnContext | ForkServerContext,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._loop = loop
        log_q = mp.Queue()
        mp_pch, mp_cch = mp_ctx.Pipe(duplex=True)

        self._pch = channel.ProcChannel(
            conn=mp_pch, loop=self._loop, messages=proto.IPC_MESSAGES
        )

        self._initialize_timeout = initialize_timeout
        self._close_timeout = close_timeout
        self._proc_args = proto.ProcStartArgs(
            initialize_process_fnc=initialize_process_fnc,
            job_entrypoint_fnc=job_entrypoint_fnc,
            job_shutdown_fnc=job_shutdown_fnc,
            log_q=log_q,
            mp_cch=mp_cch,
            asyncio_debug=loop.get_debug(),
        )

        self._proc = mp_ctx.Process(
            target=proc_main.main, args=(self._proc_args,), name="job_proc"
        )
        self._running_job: RunningJobInfo | None = None

        self._exitcode: int | None = None
        self._pid: int | None = self._proc.pid

        self._main_atask: asyncio.Task[None] | None = None
        self._closing = False
        self._kill_sent = False
        self._initialize_fut = asyncio.Future()

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

    @property
    def start_arguments(self) -> Any | None:
        return self._proc_args.user_arguments

    @start_arguments.setter
    def start_arguments(self, value: Any | None) -> None:
        self._proc_args.user_arguments = value

    def start(self) -> None:
        """start the job process"""
        if self.started:
            raise RuntimeError("process already started")

        if self._closing:
            raise RuntimeError("process is closed")

        log_listener = LogQueueListener(self._proc_args.log_q)
        log_listener.start()

        self._proc.start()
        self._pid = self._proc.pid
        self._join_fut = asyncio.Future()

        def _sync_run():
            self._proc.join()
            log_listener.stop()
            self._loop.call_soon_threadsafe(self._join_fut.set_result, None)

        thread = threading.Thread(target=_sync_run)
        thread.start()
        self._main_atask = asyncio.create_task(self._main_task())

    async def join(self) -> None:
        """wait for the job process to finish"""
        if not self.started:
            raise RuntimeError("process not started")

        await asyncio.shield(self._main_atask)

    async def initialize(self) -> None:
        """initialize the job process, this is calling the user provided initialize_process_fnc
        raise asyncio.TimeoutError if initialization times out"""
        await self._pch.asend(proto.InitializeRequest())

        # wait for the process to become ready
        try:
            init_res = await asyncio.wait_for(
                self._pch.arecv(), timeout=self._initialize_timeout
            )
            assert isinstance(
                init_res, proto.InitializeResponse
            ), "first message must be InitializeResponse"
        except asyncio.TimeoutError:
            self._initialize_fut.set_exception(
                asyncio.TimeoutError("process initialization timed out")
            )
            logger.error(
                "initialization timed out, killing job", extra=self.logging_extra()
            )
            self._send_kill_signal()
            raise
        else:
            self._initialize_fut.set_result(None)

    async def aclose(self) -> None:
        """attempt to gracefully close the job process"""
        if not self.started:
            raise RuntimeError("process not started")

        self._closing = True
        with contextlib.suppress(channel.ChannelClosed):
            await self._pch.asend(proto.ShutdownRequest())

        try:
            await asyncio.wait_for(
                asyncio.shield(self._main_atask), timeout=self._close_timeout
            )
        except asyncio.TimeoutError:
            logger.error(
                "process did not exit in time, killing job", extra=self.logging_extra()
            )
            self._send_kill_signal()

        await asyncio.shield(self._main_atask)

    async def kill(self) -> None:
        """forcefully kill the job process"""
        if not self.started:
            raise RuntimeError("process not started")

        self._closing = True
        self._send_kill_signal()
        await asyncio.shield(self._main_atask)

    async def launch_job(self, info: RunningJobInfo) -> None:
        """start/assign a job to the process"""
        if self._running_job is not None:
            raise RuntimeError("process already has a running job")

        self._running_job = info
        start_req = proto.StartJobRequest(job=info.job, url=info.url, token=info.token)
        start_req.accept_args = info.accept_args
        await self._pch.asend(start_req)

    def _send_kill_signal(self) -> None:
        """forcefully kill the job process"""
        if not self._proc.is_alive():
            return

        logger.debug("killing job process", extra=self.logging_extra())
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
            # this happens when the initialization takes longer than self._initialize_timeout
            pass

        # the process is killed if it doesn't respond to pings within this time
        pong_timeout = utils.aio.sleep(proto.PING_TIMEOUT)
        ping_task = asyncio.create_task(self._ping_pong_task(pong_timeout))
        monitor_task = asyncio.create_task(self._monitor_task(pong_timeout))

        await self._join_fut
        self._exitcode = self._proc.exitcode
        self._proc.close()
        await utils.aio.gracefully_cancel(ping_task, monitor_task)

        await self._pch.aclose()

        if self._exitcode != 0 and not self._kill_sent:
            logger.error(
                f"job process exited with non-zero exit code {self.exitcode}",
                extra=self.logging_extra(),
            )

    @utils.log_exceptions(logger=logger)
    async def _monitor_task(self, pong_timeout: utils.aio.Sleep) -> None:
        while True:
            msg = await self._pch.arecv()

            if isinstance(msg, proto.PongResponse):
                delay = utils.time_ms() - msg.timestamp
                if delay > proto.HIGH_PING_THRESHOLD * 1000:
                    logger.warning(
                        "job process is unresponsive",
                        extra={"delay": delay, **self.logging_extra()},
                    )

                with contextlib.suppress(utils.aio.SleepFinished):
                    pong_timeout.reset()

            if isinstance(msg, proto.Exiting):
                logger.debug(
                    "job exiting", extra={"reason": msg.reason, **self.logging_extra()}
                )

    @utils.log_exceptions(logger=logger)
    async def _ping_pong_task(self, pong_timeout: utils.aio.Sleep) -> None:
        ping_interval = utils.aio.interval(proto.PING_INTERVAL)

        async def _send_ping_co():
            while True:
                await ping_interval.tick()
                try:
                    await self._pch.asend(proto.PingRequest(timestamp=utils.time_ms()))
                except channel.ChannelClosed:
                    break

        async def _pong_timeout_co():
            await pong_timeout
            logger.error("job is unresponsive, killing job", extra=self.logging_extra())
            self._send_kill_signal()

        await asyncio.gather(_send_ping_co(), _pong_timeout_co())

    def logging_extra(self) -> dict:
        return {"pid": self.pid}
