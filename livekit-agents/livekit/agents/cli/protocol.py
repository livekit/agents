from __future__ import annotations

import io
import pickle
from dataclasses import dataclass, field
from typing import ClassVar

from livekit.protocol import agent

from .. import ipc_enc
from ..worker import ActiveJob, WorkerOptions


@dataclass
class CliArgs:
    opts: WorkerOptions
    log_level: str
    production: bool
    asyncio_debug: bool
    watch: bool
    drain_timeout: int
    room: str = ""
    participant_identity: str = ""
    cch: ipc_enc.ProcessPipe | None = None  # None when watch is disabled


@dataclass
class ActiveJobsRequest:
    MSG_ID: ClassVar[int] = 1

    def write(self, b: io.BytesIO) -> None:
        pass

    def read(self, b: io.BytesIO) -> None:
        pass


@dataclass
class ActiveJobsResponse:
    MSG_ID: ClassVar[int] = 2
    jobs: list[ActiveJob] = field(default_factory=list)

    def write(self, b: io.BytesIO) -> None:
        ipc_enc._write_int(b, len(self.jobs))
        for aj in self.jobs:
            job_s = aj.job.SerializeToString()
            ipc_enc._write_bytes(b, job_s)
            accept_s = pickle.dumps(aj.accept_data)
            ipc_enc._write_bytes(b, accept_s)

    def read(self, b: io.BytesIO) -> None:
        job_count = ipc_enc._read_int(b)
        for _ in range(job_count):
            job_s = ipc_enc._read_bytes(b)
            job = agent.Job()
            job.ParseFromString(job_s)
            accept_s = ipc_enc._read_bytes(b)
            accept_data = pickle.loads(accept_s)
            self.jobs.append(ActiveJob(job=job, accept_data=accept_data))


@dataclass
class ReloadJobsRequest:
    MSG_ID: ClassVar[int] = 3

    def write(self, b: io.BytesIO) -> None:
        pass

    def read(self, b: io.BytesIO) -> None:
        pass


@dataclass
class ReloadJobsResponse:
    MSG_ID: ClassVar[int] = 4
    jobs: list[ActiveJob] = field(default_factory=list)

    def write(self, b: io.BytesIO) -> None:
        ipc_enc._write_int(b, len(self.jobs))
        for aj in self.jobs:
            job_s = aj.job.SerializeToString()
            ipc_enc._write_bytes(b, job_s)
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


@dataclass
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
