from livekit.protocol import worker
from .job_process import JobProcess
from typing import Dict, List
from .consts import IPC_PORT, MAX_PACKET_SIZE
import asyncio
import sys
from ..log import worker_logger, job_logger


async def _recv_msg(rx: asyncio.StreamReader) -> worker.IPCJobMessage:
    len = await rx.readexactly(4)
    len = int.from_bytes(len, byteorder=sys.byteorder)
    if len >= MAX_PACKET_SIZE:
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


class IPCServer:
    def __init__(self):
        self._pending_processes: Dict[str, JobProcess] = {}

    async def run(self) -> None:
        """Start the IPC server listening on the IPC_PORT"""
        self._server = await asyncio.start_server(
            self._handle_client, "127.0.0.1", IPC_PORT
        )

        worker_logger.debug(f"ipc server listening on port {IPC_PORT}")
        async with self._server:
            await self._server.serve_forever()

    async def _handle_client(
        self, rx: asyncio.StreamReader, tx: asyncio.StreamWriter
    ) -> None:
        """Called when a subprocess connects to the server"""

        # First received message is always HelloWorker
        join_msg = await _recv_msg(rx)
        if not join_msg.HasField("hello"):
            worker_logger.error("expected hello message")
            return

        job_id = join_msg.hello.job_id
        process = self._pending_processes.pop(job_id)
        if process is None:
            worker_logger.error(f"no pending process for job {job_id}")
            return

        try:

            async def read_task():
                while True:
                    msg = await _recv_msg(rx)
                    if msg.HasField("log"):
                        # log message
                        continue

                    await process._recv_queue.put(msg)

            async def write_task():
                while True:
                    msg = await process._send_queue.get()
                    if msg is None:
                        tx.close()
                        return
                    await _write_msg(tx, msg)

            await asyncio.gather(read_task(), write_task())
        except Exception as e:
            logging.error(f"error handling client: {e}")

    async def close(self) -> None:
        self._server.close()
        await self._server.wait_closed()
