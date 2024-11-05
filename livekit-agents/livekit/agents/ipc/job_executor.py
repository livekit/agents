from __future__ import annotations

from enum import Enum
from typing import Any, Protocol

from ..job import RunningJobInfo


class JobExecutor(Protocol):
    @property
    def started(self) -> bool: ...

    @property
    def start_arguments(self) -> Any | None: ...

    @start_arguments.setter
    def start_arguments(self, value: Any | None) -> None: ...

    @property
    def running_job(self) -> RunningJobInfo | None: ...

    @property
    def run_status(self) -> RunStatus: ...

    @property
    def exception(self) -> Exception | None: ...

    async def start(self) -> None: ...

    async def join(self) -> None: ...

    async def initialize(self) -> None: ...

    async def aclose(self) -> None: ...

    async def launch_job(self, info: RunningJobInfo) -> None: ...


class RunStatus(Enum):
    STARTING = "STARTING"
    WAITING_FOR_JOB = "WAITING_FOR_JOB"
    RUNNING_JOB = "RUNNING_JOB"
    FINISHED_FAILED = "FINISHED_FAILED"
    FINISHED_CLEAN = "FINISHED_CLEAN"


class JobExecutorError(Exception):
    pass


class JobExecutorError_ShutdownTimeout(JobExecutorError):
    pass


class JobExecutorError_Unresponsive(JobExecutorError):
    pass


class JobExecutorError_Runtime(JobExecutorError):
    pass
