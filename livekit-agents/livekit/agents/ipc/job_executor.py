from __future__ import annotations

from enum import Enum
from typing import Any, Protocol

from ..job import RunningJobInfo


class JobExecutor(Protocol):
    @property
    def id(self) -> str: ...

    @property
    def started(self) -> bool: ...

    @property
    def user_arguments(self) -> Any | None: ...

    @user_arguments.setter
    def user_arguments(self, value: Any | None) -> None: ...

    @property
    def running_job(self) -> RunningJobInfo | None: ...

    @property
    def status(self) -> JobStatus: ...

    @property
    def jobs_completed(self) -> int: ...

    @property
    def baseline_memory_mb(self) -> float | None: ...

    @property
    def current_memory_mb(self) -> float | None: ...

    @property
    def memory_growth_mb(self) -> float: ...

    async def start(self) -> None: ...

    async def join(self) -> None: ...

    async def initialize(self) -> None: ...

    async def aclose(self) -> None: ...

    async def launch_job(self, info: RunningJobInfo) -> None: ...

    def clear_running_job(self) -> None: ...

    def logging_extra(self) -> dict[str, Any]: ...


class JobStatus(Enum):
    RUNNING = "running"
    FAILED = "failed"
    SUCCESS = "success"
