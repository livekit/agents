import asyncio
import contextlib
import multiprocessing
import sys
import threading
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
        self._process = multiprocessing.Process(
            target=_run_job, args=args, daemon=True
        )  # daemon=True to avoid unresponsive process in production
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

                res = s.result()
                if isinstance(res, protocol.StartJobResponse):
                    start_res = res
                if isinstance(res, protocol.Log):
                    logger.log(res.level, res.message)
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

        f = asyncio.Future()

        def _join_process():
            self._process.join()
            self._loop.call_soon_threadsafe(f.set_result, None)

        join_t = threading.Thread(target=self._process.join, daemon=True)
        join_t.start()
        await f

    def _sig_kill(self) -> None:
        if self._process.is_alive():
            if sys.platform == "win32":
                self._process.terminate()
            else:
                self._process.kill()

    async def aclose(self) -> None:
        await self._pipe.write(protocol.ShutdownRequest())
        await self._task
        self._pipe.close()

    @property
    def job_id(self) -> str:
        return self._job.id

    def logging_extra(self) -> dict:
        return {"job_id": self._job.id, "pid": self._process.pid}
