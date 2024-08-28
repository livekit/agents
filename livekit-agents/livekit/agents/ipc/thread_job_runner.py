from __future__ import annotations

import socket
import threading
import asyncio
import contextlib
import sys
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from .. import utils
from ..utils.aio import duplex_unix
from ..job import JobContext, JobProcess, RunningJobInfo
from ..log import logger
from . import channel, proto, thread_main


@dataclass
class _ProcOpts:
    initialize_process_fnc: Callable[[JobProcess], Any]
    job_entrypoint_fnc: Callable[[JobContext], Awaitable[None]]


class SupervisedProc:
    def __init__(
        self,
        *,
        initialize_process_fnc: Callable[[JobProcess], Any],
        job_entrypoint_fnc: Callable[[JobContext], Awaitable[None]],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._loop = loop
        self._opts = _ProcOpts(
            initialize_process_fnc=initialize_process_fnc,
            job_entrypoint_fnc=job_entrypoint_fnc,
        )

        self._user_args: Any | None = None
        self._running_job: RunningJobInfo | None = None

        self._main_atask: asyncio.Task[None] | None = None
        self._closing = False
        self._initialize_fut = asyncio.Future[None]()

        self._lock = asyncio.Lock()

    @property
    def started(self) -> bool:
        return self._main_atask is not None

    @property
    def start_arguments(self) -> Any | None:
        return self._user_args

    @start_arguments.setter
    def start_arguments(self, value: Any | None) -> None:
        self._user_args = value

    @property
    def running_job(self) -> RunningJobInfo | None:
        return self._running_job

    async def start(self) -> None:
        """start the job process"""
        if self.started:
            raise RuntimeError("process already started")

        if self._closing:
            raise RuntimeError("process is closed")

        await asyncio.shield(self._start())

    async def _start(self) -> None:
        async with self._lock:
            # to simplify the runners implementation, also use a duplex in single process mode
            # (ThreadedRunners), so we can use the same protocol
            mp_pch, mp_cch = socket.socketpair()
            self._pch = await duplex_unix._AsyncDuplex.open(mp_pch)

            self._thread = t = threading.Thread(
                target=thread_main.main,
                args=(
                    mp_cch,
                    self._opts.initialize_process_fnc,
                    self._opts.job_entrypoint_fnc,
                    self._user_args,
                    self._loop.get_debug(),
                ),
                name="job_thread_runner",
            )
            t.start()

    async def join(self) -> None:
        """wait for the job process to finish"""
        if not self.started:
            raise RuntimeError("runner not started")

        async with self._lock:
            if self._main_atask:
                await asyncio.shield(self._main_atask)

    async def initialize(self) -> None:
        """initialize the job process, this is calling the user provided initialize_process_fnc
        raise asyncio.TimeoutError if initialization times out"""
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
                "initialization timed out, killing job", extra=self.logging_extra()
            )
            self._send_kill_signal()
            raise
        except Exception as e:  # should be channel.ChannelClosed most of the time
            self._initialize_fut.set_exception(e)
            raise
        else:
            self._initialize_fut.set_result(None)

    async def aclose(self) -> None:
        """attempt to gracefully close the job process"""
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
                "process did not exit in time, killing job", extra=self.logging_extra()
            )
            self._send_kill_signal()

        async with self._lock:
            if self._main_atask:
                await asyncio.shield(self._main_atask)

    async def kill(self) -> None:
        """forcefully kill the job process"""
        if not self.started:
            raise RuntimeError("process not started")

        self._closing = True
        self._send_kill_signal()

        async with self._lock:
            if self._main_atask:
                await asyncio.shield(self._main_atask)

    async def launch_job(self, info: RunningJobInfo) -> None:
        """start/assign a job to the process"""
        if self._running_job is not None:
            raise RuntimeError("process already has a running job")

        self._running_job = info
        start_req = proto.StartJobRequest()
        start_req.running_job = info
        await channel.asend_message(self._pch, start_req)

    def _send_kill_signal(self) -> None:
        """forcefully kill the job process"""
        try:
            if not self._proc.is_alive():
                return
        except ValueError:
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
            pass  # this happens when the initialization takes longer than self._initialize_timeout
        except Exception:
            pass  # initialization failed

        # the process is killed if it doesn't respond to ping requests
        pong_timeout = utils.aio.sleep(proto.PING_TIMEOUT)
        ping_task = asyncio.create_task(self._ping_pong_task(pong_timeout))
        monitor_task = asyncio.create_task(self._monitor_task(pong_timeout))

        await self._join_fut
        self._exitcode = self._proc.exitcode
        self._proc.close()
        await utils.aio.gracefully_cancel(ping_task, monitor_task)

        with contextlib.suppress(duplex_unix.DuplexClosed):
            await self._pch.aclose()

        if self._exitcode != 0 and not self._kill_sent:
            logger.error(
                f"job process exited with non-zero exit code {self.exitcode}",
                extra=self.logging_extra(),
            )

    @utils.log_exceptions(logger=logger)
    async def _monitor_task(self, pong_timeout: utils.aio.Sleep) -> None:
        while True:
            try:
                msg = await channel.arecv_message(self._pch, proto.IPC_MESSAGES)
            except utils.aio.duplex_unix.DuplexClosed:
                break

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
                    await channel.asend_message(
                        self._pch, proto.PingRequest(timestamp=utils.time_ms())
                    )
                except utils.aio.duplex_unix.DuplexClosed:
                    break

        async def _pong_timeout_co():
            await pong_timeout
            logger.error("job is unresponsive, killing job", extra=self.logging_extra())
            self._send_kill_signal()

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
        }
        if self._running_job:
            extra["job_id"] = self._running_job.job.id

        return extra
