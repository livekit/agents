from livekit.protocol import worker
from .server import IPC_PORT

import asyncio
import sys

async def _recv_msg(rx: asyncio.StreamReader) -> worker.WorkerMessage:
    len = await rx.readexactly(4)
    len = int.from_bytes(len, byteorder=sys.byteorder)
    data = await rx.readexactly(len)
    msg = worker.WorkerMessage()
    msg.ParseFromString(data)
    return msg

async def _write_msg(tx: asyncio.StreamWriter, msg: worker.JobMessage):
    data = msg.SerializeToString()
    tx.write(len(data).to_bytes(4, byteorder=sys.byteorder))
    tx.write(data)
    await tx.drain()

class IPCClient:
    def __init__(self, job_id: str) -> None:
        self.job_id = job_id

    async def run(self) -> None:
        # connect to the IPC server
        rx, tx = await asyncio.open_connection('127.0.0.1', IPC_PORT)
        await _write_msg(tx, worker.JobMessage(hello=worker.Hello(job_id=self.job_id)))

        msg = await _recv_msg(rx)
        
        while True:
            msg = await _recv_msg(rx)
            print(f"received message: {msg}")
