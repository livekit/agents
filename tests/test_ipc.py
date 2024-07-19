import asyncio
import io
import multiprocessing
import multiprocessing as mp
from time import sleep
from livekit.agents import ipc, JobProcess, JobContext
from typing import ClassVar
from dataclasses import dataclass


@dataclass
class EmptyMessage:
    MSG_ID: ClassVar[int] = 0


@dataclass
class SomeDataMessage:
    MSG_ID: ClassVar[int] = 1
    string: str = ""
    number: int = 0
    double: float = 0.0
    data: bytes = b""

    def write(self, b: io.BytesIO) -> None:
        ipc.channel.write_string(b, self.string)
        ipc.channel.write_int(b, self.number)
        ipc.channel.write_double(b, self.double)
        ipc.channel.write_bytes(b, self.data)

    def read(self, b: io.BytesIO) -> None:
        self.string = ipc.channel.read_string(b)
        self.number = ipc.channel.read_int(b)
        self.double = ipc.channel.read_double(b)
        self.data = ipc.channel.read_bytes(b)


IPC_MESSAGES = {
    EmptyMessage.MSG_ID: EmptyMessage,
    SomeDataMessage.MSG_ID: SomeDataMessage,
}


def _ping_pong_main(mp_cch):
    async def _pong():
        loop = asyncio.get_event_loop()
        cch = ipc.channel.ProcChannel(conn=mp_cch, loop=loop, messages=IPC_MESSAGES)
        while True:
            try:
                msg = await cch.arecv()
                await cch.asend(msg)
            except ipc.channel.ChannelClosed:
                break

    asyncio.run(_pong())


async def test_async_channel():
    loop = asyncio.get_event_loop()
    mp_pch, mp_cch = mp.Pipe(duplex=True)
    pch = ipc.channel.ProcChannel(conn=mp_pch, loop=loop, messages=IPC_MESSAGES)
    proc = mp.Process(target=_ping_pong_main, args=(mp_cch,))
    proc.start()

    await pch.asend(EmptyMessage())
    assert await pch.arecv() == EmptyMessage()

    await pch.asend(
        SomeDataMessage(string="hello", number=42, double=3.14, data=b"world")
    )
    assert await pch.arecv() == SomeDataMessage(
        string="hello", number=42, double=3.14, data=b"world"
    )

    await pch.aclose()

    proc.kill()
    proc.join()


def _initialize_proc_main(proc: JobProcess) -> None:
    pass


async def _job_entrypoint(proc: JobContext) -> None:
    pass


async def _job_shutdown(proc: JobContext) -> None:
    pass


async def test_proc_pool():
    loop = asyncio.get_event_loop()
    num_idle_processes = 3
    pool = ipc.proc_pool.ProcPool(
        initialize_process_fnc=_initialize_proc_main,
        job_entrypoint_fnc=_job_entrypoint,
        job_shutdown_fnc=_job_shutdown,
        num_idle_processes=num_idle_processes,
        loop=loop,
    )

    start_q = asyncio.Queue()
    ready_q = asyncio.Queue()
    close_q = asyncio.Queue()

    @pool.on("process_started")
    def _process_started(proc: ipc.proc_pool.SupervisedProc):
        start_q.put_nowait(None)

    @pool.on("process_ready")
    def _process_ready(proc: ipc.proc_pool.SupervisedProc):
        ready_q.put_nowait(None)

    @pool.on("process_closed")
    def _process_closed(proc: ipc.proc_pool.SupervisedProc):
        close_q.put_nowait(None)

    pool.start()
    await asyncio.wait_for(
        asyncio.gather(*(start_q.get() for _ in range(num_idle_processes))), timeout=2.0
    )

    await asyncio.wait_for(
        asyncio.gather(*(ready_q.get() for _ in range(num_idle_processes))), timeout=2.0
    )

    await pool.aclose()
