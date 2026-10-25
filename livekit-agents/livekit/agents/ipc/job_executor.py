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

    async def start(self) -> None: ...

    async def join(self) -> None: ...

    async def initialize(self) -> None: ...

    async def aclose(self) -> None: ...

    async def launch_job(self, info: RunningJobInfo) -> None: ...

    async def signal_lane(self, thread_id: str, lane: str, kind: str) -> None:
        """A message or a cancel for a lane the running job holds for its whole call."""
        ...

    def logging_extra(self) -> dict[str, Any]: ...


class JobStatus(Enum):
    RUNNING = "running"
    FAILED = "failed"
    SUCCESS = "success"
