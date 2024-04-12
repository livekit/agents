import asyncio
import contextlib
import multiprocessing
import sys
import threading

from livekit.protocol import agent

from .. import aio, apipe
from ..job_request import AcceptData
from ..log import logger
from ..utils import time_ms
from . import consts, protocol
from .job_main import _run_job


class JobProcess:
    def __init__(
        self,
        job: agent.Job,
        url: str,
        token: str,
        accept_data: AcceptData,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self._loop = loop or asyncio.get_event_loop()
        self._job = job
        pch, cch = multiprocessing.Pipe(duplex=True)
        asyncio_debug = self._loop.get_debug()
        args = (
            cch,
            protocol.JobMainArgs(job.id, url, token, accept_data, asyncio_debug),
        )
        self._process = multiprocessing.Process(
            target=_run_job, args=args, daemon=True
        )  # daemon=True to avoid unresponsive process in production
        self._pipe = apipe.AsyncPipe(
            pch, loop=self._loop, messages=protocol.IPC_MESSAGES
        )
        self._close_future = asyncio.Future()

    async def run(self) -> None:
        self._process.start()

        start_timeout = asyncio.sleep(consts.START_TIMEOUT)
        ping_interval = aio.interval(consts.PING_INTERVAL)
        pong_timeout = aio.sleep(consts.PING_TIMEOUT)

        start_res: protocol.StartJobResponse | None = None

        await self._pipe.write(protocol.StartJobRequest(job=self._job))
        async with contextlib.aclosing(
            aio.select([self._pipe, start_timeout, ping_interval, pong_timeout])
        ) as select:
            while True:
                s = await select()
                if s.selected is start_timeout and start_res is None:
                    logger.error(
                        "process start timed out, killing job",
                        extra=self.logging_extra(),
                    )
                    self._sig_kill()
                    break

                if s.selected is pong_timeout:
                    logger.error(
                        "job ping timeout, killing job",
                        extra=self.logging_extra(),
                    )
                    self._sig_kill()
                    break

                if s.selected is ping_interval:
                    ping = protocol.Ping(timestamp=time_ms())
                    await self._pipe.write(ping)  # send ping each consts.PING_INTERVAL
                    continue

                try:
                    res = s.result()
                except aio.ChanClosed:
                    logger.error(
                        "pipe closed, exiting job",
                        extra=self.logging_extra(),
                    )
                    break

                if isinstance(res, protocol.StartJobResponse):
                    start_res = res
                if isinstance(res, protocol.Log):
                    logger.log(res.level, res.message, extra=self.logging_extra())
                if isinstance(res, protocol.Pong):
                    delay = time_ms() - res.timestamp
                    if delay > consts.HIGH_PING_THRESHOLD * 1000:
                        logger.warning(
                            "job is unresponsive",
                            extra={"delay": delay, **self.logging_extra()},
                        )

                    with contextlib.suppress(aio.SleepFinished):
                        pong_timeout.reset()

                if isinstance(res, protocol.UserExit) or isinstance(
                    res, protocol.ShutdownResponse
                ):
                    logger.info("job exiting", extra=self.logging_extra())
                    break

        def _join_process():
            try:
                self._process.join()
            except Exception as e:
                logger.error(
                    "error joining job process",
                    exc_info=e,
                    extra=self.logging_extra(),
                )

            self._loop.call_soon_threadsafe(self._close_future.set_result, None)

        join_t = threading.Thread(target=_join_process, daemon=True)
        join_t.start()
        await self._close_future

        logger.info("job process closed", extra=self.logging_extra())

    def _sig_kill(self) -> None:
        if not self._process.is_alive():
            return

        logger.info("killing job process", extra=self.logging_extra())
        if sys.platform == "win32":
            self._process.terminate()
        else:
            self._process.kill()

    async def aclose(self) -> None:
        logger.info("closing job process", extra=self.logging_extra())
        await self._pipe.write(protocol.ShutdownRequest())
        await self._close_future
        self._pipe.close()

    @property
    def job(self) -> agent.Job:
        return self._job

    def logging_extra(self) -> dict:
        return {"job_id": self._job.id, "pid": self._process.pid}
