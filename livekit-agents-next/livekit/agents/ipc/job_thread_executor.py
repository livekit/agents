from __future__ import annotations

import asyncio
import multiprocessing as mp
import socket
import threading
from collections.abc import Awaitable, Callable
from typing import Any

from ..job import JobContext
from ..utils import shortuuid
from .inference_executor import InferenceExecutor
from .job_proc_executor import ProcJobExecutor
from .job_proc_lazy_main import ThreadStartArgs, thread_main


class _JobThread:
    """A job thread as ``SupervisedProc`` drives a process: started, joined, never killed."""

    pid = None
    exitcode = 0

    def __init__(self, args: ThreadStartArgs) -> None:
        self._thread = threading.Thread(target=thread_main, args=(args,), name="job_thread_runner")

    @property
    def native_id(self) -> int | None:
        return self._thread.native_id

    def start(self) -> None:
        self._thread.start()

    def join(self) -> None:
        self._thread.join()

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def kill(self) -> None:
        pass

    def close(self) -> None:
        pass


class ThreadJobExecutor(ProcJobExecutor):
    """A job run on a thread of this process, supervised as a process is (the core's
    ``ProcSupervisor`` over a socket pair) except that it cannot be killed: a late
    initialization, ack or exit is reported and waited for."""

    _killable = False
    # thread jobs are not counted in the job metrics
    _metered = False

    def __init__(
        self,
        *,
        job_entrypoint_fnc: Callable[[JobContext], Awaitable[None]],
        session_end_fnc: Callable[[JobContext], Awaitable[None]] | None,
        simulation_end_fnc: Callable[[Any], Any] | None,
        inference_executor: InferenceExecutor | None,
        initialize_timeout: float,
        close_timeout: float,
        session_end_timeout: float,
        ping_interval: float,
        high_ping_threshold: float,
        http_proxy: str | None,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        super().__init__(
            job_entrypoint_fnc=job_entrypoint_fnc,
            session_end_fnc=session_end_fnc,
            simulation_end_fnc=simulation_end_fnc,
            inference_executor=inference_executor,
            initialize_timeout=initialize_timeout,
            close_timeout=close_timeout,
            session_end_timeout=session_end_timeout,
            memory_warn_mb=0,
            memory_limit_mb=0,
            ping_interval=ping_interval,
            # no pong deadline: a thread cannot be killed for being late
            ping_timeout=0,
            high_ping_threshold=high_ping_threshold,
            http_proxy=http_proxy,
            mp_ctx=mp.get_context(),
            loop=loop,
        )
        self._id = shortuuid("THEXEC_")
        self._thread: _JobThread | None = None

    def _create_process(self, cch: socket.socket) -> Any:
        # the thread shares this process's logging, and keeps its own copy of the socket the
        # parent closes once started
        self._thread = _JobThread(
            ThreadStartArgs(
                job_entrypoint_fnc=self._job_entrypoint_fnc,
                session_end_fnc=self._session_end_fnc,
                simulation_end_fnc=self._simulation_end_fnc,
                session_end_timeout=self._session_end_timeout,
                mp_cch=cch.dup(),
                user_arguments=self._user_args,
            )
        )
        return self._thread

    def logging_extra(self) -> dict[str, Any]:
        extra = super().logging_extra()
        del extra["pid"]
        return {"tid": self._thread.native_id if self._thread else None, **extra}
