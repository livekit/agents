from __future__ import annotations

import io
import pickle
from typing import ClassVar

from attrs import Factory, define
from livekit.protocol import agent

from .. import ipc_enc
from ..worker import ActiveJob, WorkerOptions


@define(kw_only=True)
class CliArgs:
    opts: WorkerOptions
    log_level: str
    production: bool
    asyncio_debug: bool
    watch: bool
    cch: ipc_enc.ProcessPipe | None = None  # None when watch is disabled


@define(kw_only=True)
class ActiveJobsRequest:
    MSG_ID: ClassVar[int] = 1

    def write(self, b: io.BytesIO) -> None:
        pass

    def read(self, b: io.BytesIO) -> None:
        pass


@define(kw_only=True)
class ActiveJobsResponse:
    MSG_ID: ClassVar[int] = 2
    jobs: list[ActiveJob] = Factory(list)

    def write(self, b: io.BytesIO) -> None:
        b.write(len(self.jobs).to_bytes(4, "big"))
        for aj in self.jobs:
            job_s = aj.job.SerializeToString()
            b.write(len(job_s).to_bytes(4, "big"))
            b.write(job_s)
            accept_s = pickle.dumps(aj.accept_data)
            b.write(len(accept_s).to_bytes(4, "big"))
            b.write(accept_s)

    def read(self, b: io.BytesIO) -> None:
        job_count = int.from_bytes(b.read(4), "big")
        for _ in range(job_count):
            job_len = int.from_bytes(b.read(4), "big")
            job = agent.Job()
            job.ParseFromString(b.read(job_len))
            accept_len = int.from_bytes(b.read(4), "big")
            accept_data = pickle.loads(b.read(accept_len))
            self.jobs.append(ActiveJob(job=job, accept_data=accept_data))


@define(kw_only=True)
class ReloadJobsRequest:
    MSG_ID: ClassVar[int] = 3

    def write(self, b: io.BytesIO) -> None:
        pass

    def read(self, b: io.BytesIO) -> None:
        pass


@define(kw_only=True)
class ReloadJobsResponse:
    MSG_ID: ClassVar[int] = 4
    jobs: list[ActiveJob] = Factory(list)

    def write(self, b: io.BytesIO) -> None:
        b.write(len(self.jobs).to_bytes(4, "big"))
        for aj in self.jobs:
            job_s = aj.job.SerializeToString()
            b.write(len(job_s).to_bytes(4, "big"))
            b.write(job_s)
            accept_s = pickle.dumps(aj.accept_data)
            b.write(len(accept_s).to_bytes(4, "big"))
            b.write(accept_s)

    def read(self, b: io.BytesIO) -> None:
        job_count = int.from_bytes(b.read(4), "big")
        for _ in range(job_count):
            job_len = int.from_bytes(b.read(4), "big")
            job = agent.Job()
            job.ParseFromString(b.read(job_len))
            accept_len = int.from_bytes(b.read(4), "big")
            accept_data = pickle.loads(b.read(accept_len))
            self.jobs.append(ActiveJob(job=job, accept_data=accept_data))


@define(kw_only=True)
class Reloaded:
    MSG_ID: ClassVar[int] = 5

    def write(self, b: io.BytesIO) -> None:
        pass

    def read(self, b: io.BytesIO) -> None:
        pass


IPC_MESSAGES = {
    ActiveJobsRequest.MSG_ID: ActiveJobsRequest,
    ActiveJobsResponse.MSG_ID: ActiveJobsResponse,
    ReloadJobsRequest.MSG_ID: ReloadJobsRequest,
    ReloadJobsResponse.MSG_ID: ReloadJobsResponse,
    Reloaded.MSG_ID: Reloaded,
}
