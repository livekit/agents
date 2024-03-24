import asyncio
import contextlib
import multiprocessing
import queue
import threading
from typing import Callable

from livekit.protocol import agent

from .. import aio
<<<<<<< Updated upstream
from ..log import worker_logger
from ..utils import time_ms
from . import consts, protocol
from .job_main import _run_job


async def _awrite(q: queue.Queue, msg: protocol.Message) -> None:
    await asyncio.to_thread(q.put, msg)
=======
from ..log import logger
from ..utils import time_ms
from . import consts, protocol
from .job_main import _run_job
from . import apipe
>>>>>>> Stashed changes


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
<<<<<<< Updated upstream
        self._pch, self._cch = multiprocessing.Pipe(duplex=True)
        args = (self._cch, protocol.JobMainArgs(job.id, url, token, target))
        self._process = multiprocessing.Process(target=_run_job, args=args)
=======
        pch, cch = multiprocessing.Pipe(duplex=True)
        args = (cch, protocol.JobMainArgs(job.id, url, token, target))
        self._process = multiprocessing.Process(target=_run_job, args=args)
        self._pipe = apipe.AsyncPipe(pch, loop=self._loop)
>>>>>>> Stashed changes

    def start(self) -> None:
        self._process.start()
        self._task = asyncio.create_task(self._proc_task())

        def _log_exc(task: asyncio.Task):
            if not task.cancelled() and task.exception():
<<<<<<< Updated upstream
                worker_logger.error(
=======
                logger.error(
>>>>>>> Stashed changes
                    "error in job process",
                    exc_info=task.exception(),
                    extra=self.logging_extra(),
                )

        self._task.add_done_callback(_log_exc)

<<<<<<< Updated upstream
        job_tx, self._job_rx = aio.channel(32, loop=self._loop)
        self._write_queue = queue.Queue(32)

        self._exit_ev = threading.Event()
        self._read_t = threading.Thread(target=self._read_thread, args=(job_tx,))
        self._write_t = threading.Thread(target=self._write_thread, args=())
        self._read_t.start()
        self._write_t.start()

    def _read_thread(self, tx: aio.ChanSender[protocol.Message]) -> None:
        while not self._exit_ev.is_set():
            msg = protocol.read_msg(self._pch)

            def _put_msg(msg):
                _ = asyncio.ensure_future(tx.send(msg))

            self._loop.call_soon_threadsafe(_put_msg, msg)

        tx.close()

    def _write_thread(self) -> None:
        while not self._exit_ev.is_set():
            msg = self._write_queue.get()
            protocol.write_msg(self._pch, msg)

=======
>>>>>>> Stashed changes
    async def _proc_task(self) -> None:
        start_timeout = asyncio.sleep(consts.START_TIMEOUT)
        ping_interval = aio.interval(consts.PING_INTERVAL)
        pong_timeout = aio.sleep(consts.PING_TIMEOUT)

        start_res: protocol.StartJobResponse | None = None

        async with contextlib.aclosing(
<<<<<<< Updated upstream
            aio.select([self._job_rx, start_timeout, ping_interval, pong_timeout])
=======
            aio.select([self._pipe, start_timeout, ping_interval, pong_timeout])
>>>>>>> Stashed changes
        ) as select:
            async for s in select:
                # self._job_rx has been closed, break the loop
                if isinstance(s.exc, aio.ChanClosed):
                    break

                # timeout the beginning of the process
                # this make sure to avoid zombie if something failed
                if s.selected == start_timeout:
                    if start_res is None:
<<<<<<< Updated upstream
                        worker_logger.error(
=======
                        logger.error(
>>>>>>> Stashed changes
                            "process start timed out", extra=self.logging_extra()
                        )
                        break

                if s.selected == pong_timeout:
                    # process main thread is unresponsive?
                    break

                # send ping each consts.PING_INTERVAL
                if s.selected == ping_interval:
<<<<<<< Updated upstream
                    await _awrite(self._write_queue, protocol.Ping(timestamp=time_ms()))
=======
                    await self._pipe.write(protocol.Ping(timestamp=time_ms()))
>>>>>>> Stashed changes
                    continue

                res = s.result()
                if isinstance(res, protocol.StartJobResponse):
                    start_res = res
                if isinstance(res, protocol.Log):
<<<<<<< Updated upstream
                    worker_logger.log(res.level, res.message)
                if isinstance(res, protocol.Pong):
=======
                    logger.log(res.level, res.message)
                if isinstance(res, protocol.Pong):
                    print("pong")
>>>>>>> Stashed changes
                    with contextlib.suppress(aio.SleepFinished):
                        pong_timeout.reset()
                    last_ping = time_ms()

    async def aclose(self) -> None:
<<<<<<< Updated upstream
        self._exit_ev.set()
=======
        self._pipe.close()
>>>>>>> Stashed changes
        await self._task

    @property
    def job_id(self) -> str:
        return self._job.id

    def logging_extra(self) -> dict:
        return {"job_id": self._job.id, "pid": self._process.pid}
