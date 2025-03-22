from __future__ import annotations

import asyncio
import ctypes
import io
import multiprocessing as mp
import socket
import time
import uuid
from dataclasses import dataclass
from multiprocessing.context import BaseContext
from typing import ClassVar

import psutil

from livekit.agents import JobContext, JobProcess, ipc, job, utils
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


def _echo_main(mp_cch):
    async def _pong():
        cch = await utils.aio.duplex_unix._AsyncDuplex.open(mp_cch)
        while True:
            try:
                msg = await ipc.channel.arecv_message(cch, IPC_MESSAGES)
                await ipc.channel.asend_message(cch, msg)
            except utils.aio.duplex_unix.DuplexClosed:
                print("_echo_main, duplex closed..")
                break

    asyncio.run(_pong())


async def test_async_channel():
    mp_pch, mp_cch = socket.socketpair()
    pch = await utils.aio.duplex_unix._AsyncDuplex.open(mp_pch)
    proc = mp.get_context("spawn").Process(target=_echo_main, args=(mp_cch,))
    proc.start()
    mp_cch.close()

    await ipc.channel.asend_message(pch, EmptyMessage())
    assert await ipc.channel.arecv_message(pch, IPC_MESSAGES) == EmptyMessage()

    await ipc.channel.asend_message(
        pch, SomeDataMessage(string="hello", number=42, double=3.14, data=b"world")
    )
    assert await ipc.channel.arecv_message(pch, IPC_MESSAGES) == SomeDataMessage(
        string="hello", number=42, double=3.14, data=b"world"
    )

    await pch.aclose()
    await asyncio.sleep(0.5)
    proc.terminate()
    proc.join()


def test_sync_channel():
    mp_pch, mp_cch = socket.socketpair()
    pch = utils.aio.duplex_unix._Duplex.open(mp_pch)

    proc = mp.get_context("spawn").Process(target=_echo_main, args=(mp_cch,))
    proc.start()
    mp_cch.close()

    ipc.channel.send_message(pch, EmptyMessage())
    assert ipc.channel.recv_message(pch, IPC_MESSAGES) == EmptyMessage()

    ipc.channel.send_message(
        pch, SomeDataMessage(string="hello", number=42, double=3.14, data=b"world")
    )
    assert ipc.channel.recv_message(pch, IPC_MESSAGES) == SomeDataMessage(
        string="hello", number=42, double=3.14, data=b"world"
    )

    pch.close()


def _generate_fake_job() -> job.RunningJobInfo:
    return job.RunningJobInfo(
        job=agent.Job(
            id="fake_job_" + str(uuid.uuid4().hex), type=agent.JobType.JT_ROOM
        ),
        url="fake_url",
        token="fake_token",
        accept_arguments=job.JobAcceptArguments(name="", identity="", metadata=""),
        worker_id="fake_id",
    )


@dataclass
class _StartArgs:
    initialize_counter: mp.Value
    entrypoint_counter: mp.Value
    shutdown_counter: mp.Value
    initialize_simulate_work_time: float
    entrypoint_simulate_work_time: float
    shutdown_simulate_work_time: float
    update_ev: mp.Condition


def _new_start_args(mp_ctx: BaseContext) -> _StartArgs:
    return _StartArgs(
        initialize_counter=mp_ctx.Value(ctypes.c_uint),
        entrypoint_counter=mp_ctx.Value(ctypes.c_uint),
        shutdown_counter=mp_ctx.Value(ctypes.c_uint),
        initialize_simulate_work_time=0.0,
        entrypoint_simulate_work_time=0.0,
        shutdown_simulate_work_time=0.0,
        update_ev=mp_ctx.Condition(),
    )


def _initialize_proc(proc: JobProcess) -> None:
    start_args: _StartArgs = proc.user_arguments

    # incrementing isn't atomic (the lock should be reentrant by default)
    with start_args.initialize_counter.get_lock():
        start_args.initialize_counter.value += 1

    time.sleep(start_args.initialize_simulate_work_time)

    with start_args.update_ev:
        start_args.update_ev.notify()


async def _job_entrypoint(job_ctx: JobContext) -> None:
    start_args: _StartArgs = job_ctx.proc.user_arguments

    async def _job_shutdown() -> None:
        with start_args.shutdown_counter.get_lock():
            start_args.shutdown_counter.value += 1

        await asyncio.sleep(start_args.shutdown_simulate_work_time)

        with start_args.update_ev:
            start_args.update_ev.notify()

    job_ctx.add_shutdown_callback(_job_shutdown)

    with start_args.entrypoint_counter.get_lock():
        start_args.entrypoint_counter.value += 1

    await asyncio.sleep(start_args.entrypoint_simulate_work_time)

    job_ctx.shutdown(
        "calling shutdown inside the test to avoid a warning when neither shutdown nor connect is called."
    )

    with start_args.update_ev:
        start_args.update_ev.notify()


async def _wait_for_elements(q: asyncio.Queue, num_elements: int) -> None:
    for _ in range(num_elements):
        await q.get()


async def test_proc_pool():
    mp_ctx = mp.get_context("spawn")
    loop = asyncio.get_running_loop()
    num_idle_processes = 3
    pool = ipc.proc_pool.ProcPool(
        initialize_process_fnc=_initialize_proc,
        job_entrypoint_fnc=_job_entrypoint,
        num_idle_processes=num_idle_processes,
        job_executor_type=job.JobExecutorType.PROCESS,
        initialize_timeout=20.0,
        close_timeout=20.0,
        inference_executor=None,
        memory_warn_mb=0,
        memory_limit_mb=0,
        mp_ctx=mp_ctx,
        loop=loop,
    )

    start_args = _new_start_args(mp_ctx)
    created_q = asyncio.Queue()
    start_q = asyncio.Queue()
    ready_q = asyncio.Queue()
    close_q = asyncio.Queue()

    pids = []
    exitcodes = []

    @pool.on("process_created")
    def _process_created(proc: ipc.job_proc_executor.ProcJobExecutor):
        created_q.put_nowait(None)
        proc.user_arguments = start_args

    @pool.on("process_started")
    def _process_started(proc: ipc.job_proc_executor.ProcJobExecutor):
        start_q.put_nowait(None)
        pids.append(proc.pid)

    @pool.on("process_ready")
    def _process_ready(proc: ipc.job_proc_executor.ProcJobExecutor):
        ready_q.put_nowait(None)

    @pool.on("process_closed")
    def _process_closed(proc: ipc.job_proc_executor.ProcJobExecutor):
        close_q.put_nowait(None)
        exitcodes.append(proc.exitcode)

    pool.start()

    await _wait_for_elements(created_q, num_idle_processes)
    await _wait_for_elements(start_q, num_idle_processes)
    await _wait_for_elements(ready_q, num_idle_processes)

    assert start_args.initialize_counter.value == num_idle_processes

    jobs_to_start = 2

    for _ in range(jobs_to_start):
        await pool.launch_job(_generate_fake_job())

    await _wait_for_elements(created_q, jobs_to_start)
    await _wait_for_elements(start_q, jobs_to_start)
    await _wait_for_elements(ready_q, jobs_to_start)

    await pool.aclose()

    assert start_args.entrypoint_counter.value == jobs_to_start
    assert start_args.shutdown_counter.value == jobs_to_start

    await _wait_for_elements(close_q, num_idle_processes + jobs_to_start)

    # the way we check that a process doesn't exist anymore isn't technically reliable (pid recycle could happen)
    for pid in pids:
        assert not psutil.pid_exists(pid)

    for exitcode in exitcodes:
        # this test expects graceful shutdown, kill is tested on another test
        assert exitcode == 0, f"process did not exit cleanly: {exitcode}"


async def test_slow_initialization():
    mp_ctx = mp.get_context("spawn")
    loop = asyncio.get_running_loop()
    num_idle_processes = 2
    pool = ipc.proc_pool.ProcPool(
        job_executor_type=job.JobExecutorType.PROCESS,
        initialize_process_fnc=_initialize_proc,
        job_entrypoint_fnc=_job_entrypoint,
        num_idle_processes=num_idle_processes,
        initialize_timeout=1.0,
        close_timeout=20.0,
        inference_executor=None,
        memory_warn_mb=0,
        memory_limit_mb=0,
        mp_ctx=mp_ctx,
        loop=loop,
    )

    start_args = _new_start_args(mp_ctx)
    start_args.initialize_simulate_work_time = 2.0
    start_q = asyncio.Queue()
    close_q = asyncio.Queue()

    pids = []
    exitcodes = []

    @pool.on("process_created")
    def _process_created(proc: ipc.job_proc_executor.ProcJobExecutor):
        proc.user_arguments = start_args
        start_q.put_nowait(None)

    @pool.on("process_closed")
    def _process_closed(proc: ipc.job_proc_executor.ProcJobExecutor):
        close_q.put_nowait(None)
        pids.append(proc.pid)
        exitcodes.append(proc.exitcode)

    pool.start()

    await _wait_for_elements(start_q, num_idle_processes)
    await _wait_for_elements(close_q, num_idle_processes)

    # after initialization failure, warmup should be retried
    await _wait_for_elements(start_q, num_idle_processes)
    await pool.aclose()

    for pid in pids:
        assert not psutil.pid_exists(pid)

    for exitcode in exitcodes:
        assert exitcode != 0, "process should have been killed"


def _create_proc(
    *,
    close_timeout: float,
    mp_ctx: BaseContext,
    initialize_timeout: float = 20.0,
) -> tuple[ipc.job_proc_executor.ProcJobExecutor, _StartArgs]:
    start_args = _new_start_args(mp_ctx)
    loop = asyncio.get_running_loop()
    proc = ipc.job_proc_executor.ProcJobExecutor(
        initialize_process_fnc=_initialize_proc,
        job_entrypoint_fnc=_job_entrypoint,
        initialize_timeout=initialize_timeout,
        close_timeout=close_timeout,
        memory_warn_mb=0,
        memory_limit_mb=0,
        ping_interval=2.5,
        ping_timeout=10.0,
        high_ping_threshold=1.0,
        inference_executor=None,
        mp_ctx=mp_ctx,
        loop=loop,
    )
    proc.user_arguments = start_args
    return proc, start_args


async def test_shutdown_no_job():
    mp_ctx = mp.get_context("spawn")
    proc, start_args = _create_proc(close_timeout=10.0, mp_ctx=mp_ctx)
    await proc.start()
    await proc.initialize()
    await asyncio.sleep(1.0)
    await proc.aclose()

    assert proc.exitcode == 0
    assert not proc.killed
    assert start_args.shutdown_counter.value == 0, (
        "shutdown_cb isn't called when there is no job"
    )


async def test_job_slow_shutdown():
    mp_ctx = mp.get_context("spawn")
    proc, start_args = _create_proc(close_timeout=1.0, mp_ctx=mp_ctx)
    start_args.shutdown_simulate_work_time = 10.0

    await proc.start()
    await proc.initialize()
    await asyncio.sleep(1.0)

    fake_job = _generate_fake_job()
    await proc.launch_job(fake_job)
    await asyncio.sleep(1.0)
    await proc.aclose()

    # process is killed when there is a job with slow timeout
    assert proc.exitcode != 0, "process should have been killed"
    assert proc.killed


async def test_job_graceful_shutdown():
    mp_ctx = mp.get_context("spawn")
    proc, start_args = _create_proc(close_timeout=10.0, mp_ctx=mp_ctx)
    start_args.shutdown_simulate_work_time = 1.0
    await proc.start()
    await proc.initialize()
    await asyncio.sleep(1.0)

    fake_job = _generate_fake_job()
    await proc.launch_job(fake_job)
    await asyncio.sleep(1.0)
    await proc.aclose()

    assert proc.exitcode == 0, "process should have exited cleanly"
    assert not proc.killed
    assert start_args.shutdown_counter.value == 1
