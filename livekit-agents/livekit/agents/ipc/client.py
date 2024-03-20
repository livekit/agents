from __future__ import annotations

from livekit.protocol import worker
from typing import Callable, Tuple

from .consts import IPC_PORT
from ..log import process_logger

import asyncio
import multiprocessing
import sys
import logging


class LogHandler(logging.Handler):
    def __init__(self, tx: asyncio.Queue, loop: asyncio.AbstractEventLoop) -> None:
        super().__init__(logging.NOTSET)  # forward all logs
        self._tx = tx
        self._loop = loop

    def emit(self, record: logging.LogRecord) -> None:
        level = worker.IPCLogLevel.IL_NOTSET
        if record.levelno == logging.CRITICAL:
            level = worker.IPCLogLevel.IL_CRITICAL
        elif record.levelno == logging.ERROR:
            level = worker.IPCLogLevel.IL_ERROR
        elif record.levelno == logging.WARNING:
            level = worker.IPCLogLevel.IL_WARNING
        elif record.levelno == logging.INFO:
            level = worker.IPCLogLevel.IL_INFO
        elif record.levelno == logging.DEBUG:
            level = worker.IPCLogLevel.IL_DEBUG

        msg = worker.IPCJobMessage(
            log=worker.IPCLog(
                level=level,
                message=record.getMessage(),
            )
        )

        try:
            self._tx.put_nowait(msg)
        except asyncio.QueueFull:

            async def _send_log() -> None:
                await self._tx.put(msg)

            asyncio.run_coroutine_threadsafe(_send_log(), loop=self._loop)


async def _recv_msg(rx: asyncio.StreamReader) -> worker.IPCWorkerMessage:
    len = await rx.readexactly(4)
    len = int.from_bytes(len, byteorder=sys.byteorder)
    data = await rx.readexactly(len)
    msg = worker.IPCWorkerMessage()
    msg.ParseFromString(data)
    return msg


async def _write_msg(tx: asyncio.StreamWriter, msg: worker.IPCJobMessage):
    data = msg.SerializeToString()
    tx.write(len(data).to_bytes(4, byteorder=sys.byteorder))
    tx.write(data)
    await tx.drain()


class IPCClient:
    def __init__(
        self,
        job_id: str,
        recv_queue: asyncio.Queue[worker.IPCWorkerMessage],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._job_id = job_id
        self._send_queue = asyncio.Queue[worker.IPCJobMessage | None]()
        self._recv_queue = recv_queue
        self._loop = loop
        self._log_handler = LogHandler(self._send_queue, self._loop)

        self._main_task = self._loop.create_task(self._run_conn())

        def log_exception(task: asyncio.Task) -> None:
            if not task.cancelled() and task.exception():
                print(f"ipc task failed: {task.exception()}")

        self._main_task.add_done_callback(log_exception)
        self._closed = False

    @staticmethod
    def create(
        job_id: str,
        loop: asyncio.AbstractEventLoop,
    ) -> Tuple[IPCClient, asyncio.Queue[worker.IPCWorkerMessage]]:
        recv_queue = asyncio.Queue[worker.IPCWorkerMessage]()
        return IPCClient(job_id, recv_queue, loop), recv_queue

    @property
    def log_handler(self) -> LogHandler:
        return self._log_handler

    async def send(self, msg: worker.IPCJobMessage) -> None:
        if self._closed:
            raise Exception("ipc client is closed")

        await self._send_queue.put(msg)

    async def _run_conn(self) -> None:
        process_logger.debug("connecting to ipc server")
        rx, tx = await asyncio.open_connection("127.0.0.1", IPC_PORT)
        await _write_msg(
            tx, worker.IPCJobMessage(hello=worker.HelloWorker(job_id=self._job_id))
        )

        try:

            async def read_task():
                while True:
                    msg = await _recv_msg(rx)
                    await self._recv_queue.put(msg)

            async def write_task():
                while True:
                    msg = await self._send_queue.get()
                    if msg is None:
                        tx.close()
                        return
                    await _write_msg(tx, msg)

            await asyncio.gather(read_task(), write_task())
        except Exception:
            process_logger.exception("ipc client failed")

    async def aclose(self) -> None:
        self._closed = True
        await self._send_queue.put(None)
        await self._main_task
