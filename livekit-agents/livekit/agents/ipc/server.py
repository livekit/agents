from livekit.protocol import worker, agent
from typing import Callable, Dict, List
import asyncio
import sys
import threading
import contextlib
import multiprocessing
import logging
from ..log import worker_logger, job_logger
from ..aio import ChanSender, ChanReceiver, channel
from .job_main import _run_job
from . import consts


class JobProcess:
    def __init__(
        self,
        job: agent.Job,
        url: str,
        token: str,
        target: Callable,  # must be serializable by pickle
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        args = consts.JobMainArgs(job.id, url, token, target)
        self._process = multiprocessing.Process(target=_run_job, args=(args,))
        self._loop = loop
        self._job = job
        self._started = False

    def logging_extra(self) -> dict:
        return {"job_id": self._job.id, "pid": self._process.pid}

    @property
    def job_id(self) -> str:
        return self._job.id

    def start(self, rx: asyncio.StreamReader, tx: asyncio.StreamWriter) -> None:
        self._started = True
        self._main_task = asyncio.create_task(self._run(rx, tx))
        self._tx = tx

    async def _run(self, rx: asyncio.StreamReader, tx: asyncio.StreamWriter) -> None:
        try:
            start_req = worker.StartJobRequest()
            start_req.job.CopyFrom(self._job)
            await _write_msg(tx, worker.IPCWorkerMessage(start_job=start_req))

            # handle messages from the process
            try:
                while True:
                    msg = await _recv_msg(rx)

                    if msg.HasField("log"):
                        _log_process(self, msg.log)
                    elif msg.HasField("start_job"):
                        if msg.start_job.error:
                            break # starting job failed
                    elif msg.HasField("exit"):
                        break
                    elif msg.HasField("shutdown"):
                        break

            except asyncio.IncompleteReadError:
                pass  # EOF

            # join the process & avoid zombies on exit
            join_ev = asyncio.Event()

            def _join_proc():
                self._process.join()
                self._loop.call_soon_threadsafe(join_ev.set)

            join_t = threading.Thread(target=_join_proc)
            join_t.start()

            await join_ev.wait()

            exitcode = self._process.exitcode
            if exitcode != 0:
                worker_logger.error(f"unexpected process exit with code {exitcode}")

        except asyncio.CancelledError:
            pass
        except Exception:
            worker_logger.exception("error running process", extra=self.logging_extra())
        finally:
            self._tx.close()
            if self._process.is_alive():
                self._process.kill()

    async def aclose(self) -> None:
        """Gracefully shutdown the process"""
        if not self._started:
            return

        #await _write_msg(self._tx, worker.IPCWorkerMessage(exit=worker.ExitRequest()))

        self._main_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._main_task


class IPCServer:
    def __init__(self, loop: asyncio.AbstractEventLoop | None = None):
        self._loop = loop or asyncio.get_event_loop()
        self._processes: Dict[str, JobProcess] = {}

    def start(self) -> None:
        """Start the IPC server listening on the IPC_PORT"""
        self._main_task = asyncio.create_task(self._run())

    def new_process(
        self,
        job: agent.Job,
        url: str,
        token: str,
        target: Callable,
    ) -> JobProcess:
        proc = JobProcess(job, url, token, target, self._loop)
        self._processes[job.id] = proc
        proc._process.start()
        return proc

    async def _run(self) -> None:
        self._server = await asyncio.start_server(
            self._handle_client, "127.0.0.1", consts.IPC_PORT
        )

        worker_logger.debug(f"ipc server listening on port {consts.IPC_PORT}")
        try:
            async with self._server:
                await self._server.serve_forever()
        except asyncio.CancelledError:
            pass
        except BaseException:
            worker_logger.exception("error in ipc server")

    async def _handle_client(
        self, rx: asyncio.StreamReader, tx: asyncio.StreamWriter
    ) -> None:
        """Called when a subprocess connects to the server"""
        try:
            join_msg = await _recv_msg(rx)
            if not join_msg.HasField("hello"):
                worker_logger.warning("expected hello message")
                return

            job_id = join_msg.hello.job_id
            if job_id not in self._processes:
                worker_logger.error(f"no pending process for job {job_id}")
                return

            proc = self._processes[job_id]
            proc.start(rx, tx)
        except Exception:
            worker_logger.exception(f"error handling client")

    async def aclose(self) -> None:
        for proc in self._processes.values():
            await proc.aclose()

        self._server.close()
        self._main_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._main_task




async def _recv_msg(rx: asyncio.StreamReader) -> worker.IPCJobMessage:
    len = await rx.readexactly(4)
    len = int.from_bytes(len, byteorder=sys.byteorder)
    if len >= consts.MAX_PACKET_SIZE:
        raise ValueError("packet too large")

    data = await rx.readexactly(len)
    msg = worker.IPCJobMessage()
    msg.ParseFromString(data)
    return msg


async def _write_msg(tx: asyncio.StreamWriter, msg: worker.IPCWorkerMessage):
    data = msg.SerializeToString()
    tx.write(len(data).to_bytes(4, byteorder=sys.byteorder))
    tx.write(data)
    await tx.drain()


def _log_process(process: JobProcess, msg: worker.IPCLog) -> None:
    level = logging.NOTSET
    if msg.level == worker.IPCLogLevel.IL_CRITICAL:
        level = logging.CRITICAL
    elif msg.level == worker.IPCLogLevel.IL_ERROR:
        level = logging.ERROR
    elif msg.level == worker.IPCLogLevel.IL_WARNING:
        level = logging.WARNING
    elif msg.level == worker.IPCLogLevel.IL_INFO:
        level = logging.INFO
    elif msg.level == worker.IPCLogLevel.IL_DEBUG:
        level = logging.DEBUG

    logging.log(level, msg.message, extra=process.logging_extra())
