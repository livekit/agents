from __future__ import annotations  # noqa: I001

import io
import socket
from dataclasses import dataclass, field
from typing import ClassVar

from livekit.protocol import agent

from ..ipc import channel
from ..job import JobAcceptArguments, RunningJobInfo
from ..worker import WorkerOptions, SimulateJobInfo


@dataclass
class CliArgs:
    opts: WorkerOptions
    log_level: str
    devmode: bool
    asyncio_debug: bool
    watch: bool

    console: bool = False
    # whether to run the worker in console mode (console subcommand
    record: bool = False

    # register the worker to the worker pool
    register: bool = True

    simulate_job: SimulateJobInfo | str | None = None

    # amount of time this worker has been reloaded
    reload_count: int = 0

    # pipe used for the communication between the watch server and the watch client
    # when reload/dev mode is enabled
    mp_cch: socket.socket | None = None


@dataclass
class ActiveJobsRequest:
    MSG_ID: ClassVar[int] = 1


@dataclass
class ActiveJobsResponse:
    MSG_ID: ClassVar[int] = 2
    jobs: list[RunningJobInfo] = field(default_factory=list)
    reload_count: int = 0

    def write(self, b: io.BytesIO) -> None:
        channel.write_int(b, len(self.jobs))
        for running_job in self.jobs:
            accept_args = running_job.accept_arguments
            channel.write_bytes(b, running_job.job.SerializeToString())
            channel.write_string(b, accept_args.name)
            channel.write_string(b, accept_args.identity)
            channel.write_string(b, accept_args.metadata)
            channel.write_string(b, running_job.url)
            channel.write_string(b, running_job.token)
            channel.write_string(b, running_job.worker_id)

        channel.write_int(b, self.reload_count)

    def read(self, b: io.BytesIO) -> None:
        for _ in range(channel.read_int(b)):
            job = agent.Job()
            job.ParseFromString(channel.read_bytes(b))
            self.jobs.append(
                RunningJobInfo(
                    accept_arguments=JobAcceptArguments(
                        name=channel.read_string(b),
                        identity=channel.read_string(b),
                        metadata=channel.read_string(b),
                    ),
                    job=job,
                    url=channel.read_string(b),
                    token=channel.read_string(b),
                    worker_id=channel.read_string(b),
                )
            )

        self.reload_count = channel.read_int(b)


@dataclass
class ReloadJobsRequest:
    MSG_ID: ClassVar[int] = 3


@dataclass
class ReloadJobsResponse(ActiveJobsResponse):
    MSG_ID: ClassVar[int] = 4


@dataclass
class Reloaded:
    MSG_ID: ClassVar[int] = 5


IPC_MESSAGES = {
    ActiveJobsRequest.MSG_ID: ActiveJobsRequest,
    ActiveJobsResponse.MSG_ID: ActiveJobsResponse,
    ReloadJobsRequest.MSG_ID: ReloadJobsRequest,
    ReloadJobsResponse.MSG_ID: ReloadJobsResponse,
    Reloaded.MSG_ID: Reloaded,
}
