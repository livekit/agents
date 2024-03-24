import asyncio
import contextlib
import multiprocessing
from typing import Callable

from livekit.protocol import agent

from .. import aio
from ..log import logger
from ..utils import time_ms
from . import apipe, consts, protocol
from .job_main import _run_job


class JobProcess:
    def __init__(
        self,
        job: agent.Job,
        url: str,
        token: str,
        target: Callable,  # must be serializable by pickle
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self._loop = loop or asyncio.get_event_loop()
        self._job = job
        pch, cch = multiprocessing.Pipe(duplex=True)
        args = (cch, protocol.JobMainArgs(job.id, url, token, target))
        self._process = multiprocessing.Process(target=_run_job, args=args)
        self._pipe = apipe.AsyncPipe(pch, loop=self._loop)

    def start(self) -> None:
        self._process.start()
        self._task = asyncio.create_task(self._proc_task())

        def _log_exc(task: asyncio.Task):
            if not task.cancelled() and task.exception():
                logger.error(
                    "error in job process",
                    exc_info=task.exception(),
                    extra=self.logging_extra(),
                )

        self._task.add_done_callback(_log_exc)


    async def _proc_task(self) -> None:
        start_timeout = asyncio.sleep(consts.START_TIMEOUT)
        ping_interval = aio.interval(consts.PING_INTERVAL)
        pong_timeout = aio.sleep(consts.PING_TIMEOUT)

        start_res: protocol.StartJobResponse | None = None

        async with contextlib.aclosing(
            aio.select([self._pipe, start_timeout, ping_interval, pong_timeout])
        ) as select:
            async for s in select:
                # self._job_rx has been closed, break the loop
                if isinstance(s.exc, aio.ChanClosed):
                    break

                # timeout the beginning of the process
                # this make sure to avoid zombie if something failed
                if s.selected == start_timeout:
                    if start_res is None:
                        logger.error(
                            "process start timed out", extra=self.logging_extra()
                        )
                        break

                if s.selected == pong_timeout:
                    # process main thread is unresponsive?
                    break

                # send ping each consts.PING_INTERVAL
                if s.selected == ping_interval:
                    await self._pipe.write(protocol.Ping(timestamp=time_ms()))
                    continue

                res = s.result()
                if isinstance(res, protocol.StartJobResponse):
                    start_res = res
                if isinstance(res, protocol.Log):
                    logger.log(res.level, res.message)
                if isinstance(res, protocol.Pong):
                    logger.log(res.level, res.message)
                if isinstance(res, protocol.Pong):
                    print("pong")
                if isinstance(res, protocol.Pong):
                    with contextlib.suppress(aio.SleepFinished):
                        pong_timeout.reset()
                    last_ping = time_ms()

    async def aclose(self) -> None:
        self._pipe.close()
        await self._task

    @property
    def job_id(self) -> str:
        return self._job.id

    def logging_extra(self) -> dict:
        return {"job_id": self._job.id, "pid": self._process.pid}
