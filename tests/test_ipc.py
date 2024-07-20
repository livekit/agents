import asyncio
import ctypes
import io
import multiprocessing as mp
import uuid
from dataclasses import dataclass
from typing import ClassVar

import psutil
from livekit.agents import JobContext, JobProcess, ipc, job
from livekit.protocol import agent


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
    (initialize_fnc_v, _, _) = proc.start_arguments

    # incrementing isn't atomic (the lock should be reentrant by default)
    with initialize_fnc_v.get_lock():
        initialize_fnc_v.value += 1


async def _job_entrypoint(job_ctx: JobContext) -> None:
    (_, entrypoint_fnc_v, _) = job_ctx.proc.start_arguments
    with entrypoint_fnc_v.get_lock():
        entrypoint_fnc_v.value += 1

    job_ctx.shutdown(
        "calling shutdown inside the test to avoid a warning when neither shutdown nor connect is called."
    )


async def _job_shutdown(job_ctx: JobContext) -> None:
    (_, _, shutdown_fnc_v) = job_ctx.proc.start_arguments

    with shutdown_fnc_v.get_lock():
        shutdown_fnc_v.value += 1


def _generate_fake_job() -> job.RunningJobInfo:
    return job.RunningJobInfo(
        job=agent.Job(
            id="fake_job_" + str(uuid.uuid4().hex), type=agent.JobType.JT_ROOM
        ),
        url="fake_url",
        token="fake_token",
        accept_args=job.JobAcceptArguments(name="", identity="", metadata=""),
    )


async def test_proc_pool():
    loop = asyncio.get_running_loop()
    num_idle_processes = 3
    pool = ipc.proc_pool.ProcPool(
        initialize_process_fnc=_initialize_proc_main,
        job_entrypoint_fnc=_job_entrypoint,
        job_shutdown_fnc=_job_shutdown,
        num_idle_processes=num_idle_processes,
        close_timeout=20.0,
        loop=loop,
    )

    initialize_fnc_v = mp.Value(ctypes.c_uint)
    entrypoint_fnc_v = mp.Value(ctypes.c_uint)
    shutdown_fnc_v = mp.Value(ctypes.c_uint)

    created_q = asyncio.Queue()
    start_q = asyncio.Queue()
    ready_q = asyncio.Queue()
    close_q = asyncio.Queue()

    pids = []
    exitcodes = []

    @pool.on("process_created")
    def _process_created(proc: ipc.proc_pool.SupervisedProc):
        created_q.put_nowait(None)
        proc.start_arguments = (initialize_fnc_v, entrypoint_fnc_v, shutdown_fnc_v)

    @pool.on("process_started")
    def _process_started(proc: ipc.proc_pool.SupervisedProc):
        start_q.put_nowait(None)
        pids.append(proc.pid)

    @pool.on("process_ready")
    def _process_ready(proc: ipc.proc_pool.SupervisedProc):
        ready_q.put_nowait(None)

    @pool.on("process_closed")
    def _process_closed(proc: ipc.proc_pool.SupervisedProc):
        close_q.put_nowait(None)
        exitcodes.append(proc.exitcode)

    pool.start()

    async def wait_for_q(q, n):
        await asyncio.wait_for(
            asyncio.gather(*(q.get() for _ in range(n))), timeout=2.0
        )

    await wait_for_q(created_q, num_idle_processes)
    await wait_for_q(start_q, num_idle_processes)
    await wait_for_q(ready_q, num_idle_processes)

    assert initialize_fnc_v.value == num_idle_processes

    jobs_to_start = 2

    for _ in range(jobs_to_start):
        await pool.launch_job(_generate_fake_job())

    await wait_for_q(created_q, jobs_to_start)
    await wait_for_q(start_q, jobs_to_start)
    await wait_for_q(ready_q, jobs_to_start)

    await pool.aclose()

    assert entrypoint_fnc_v.value == jobs_to_start
    assert shutdown_fnc_v.value == jobs_to_start

    await wait_for_q(close_q, jobs_to_start + num_idle_processes)

    # the way we check that a process doesn't exist anymore isn't technically reliable (pid recycle could happen)
    for pid in pids:
        assert not psutil.pid_exists(pid)

    for exitcode in exitcodes:
        # this test expects graceful shutdown, kill is tested on another test
        assert exitcode == 0, f"process did not exit cleanly: {exitcode}"
