from livekit.protocol import worker
from .job_process import JobProcess
from typing import Dict, List
import asyncio
import logging
import sys

IPC_PORT = 2003
MAX_PACKET_SIZE = 1024 * 16

async def _recv_msg(rx: asyncio.StreamReader) -> worker.JobMessage:
    len = await rx.readexactly(4)
    len = int.from_bytes(len, byteorder=sys.byteorder)
    if len >= MAX_PACKET_SIZE:
        raise ValueError("packet too large")

    data = await rx.readexactly(len)
    msg = worker.JobMessage()
    msg.ParseFromString(data)
    return msg


async def _write_msg(tx: asyncio.StreamWriter, msg: worker.WorkerMessage):
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

        async with self._server:
            await self._server.serve_forever()

    async def _handle_client(self, rx: asyncio.StreamReader, tx: asyncio.StreamWriter) -> None:
        """Called when a subprocess connects to the server"""

        join_msg = await _recv_msg(rx)
        if not join_msg.HasField("hello"):
            logging.error("expected hello message")
            return

        job_id = join_msg.hello.job_id
        process = self._pending_processes.pop(job_id)
        if process is None:
            logging.error(f"no pending process for job {job_id}")
            return

        try:

            async def read_task():
                while True:
                    msg = await _recv_msg(rx)
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

    async def stop(self) -> None:
        self._server.close()
        await self._server.wait_closed()


