from __future__ import annotations

import io
import pickle
from dataclasses import dataclass, field
from typing import Any, ClassVar

from livekit.protocol import agent

from ..job import JobAcceptArguments, RunningJobInfo
from . import channel


@dataclass
class InitializeRequest:
    """sent by the main process to the subprocess to initialize it. this is going to call initialize_process_fnc"""  # noqa: E501

    MSG_ID: ClassVar[int] = 0

    asyncio_debug: bool = False
    ping_interval: float = 0
    ping_timeout: float = 0  # if no response, process is considered dead
    # if ping is higher than this, process is considered unresponsive
    high_ping_threshold: float = 0
    http_proxy: str = ""  # empty = None

    def write(self, b: io.BytesIO) -> None:
        channel.write_bool(b, self.asyncio_debug)
        channel.write_float(b, self.ping_interval)
        channel.write_float(b, self.ping_timeout)
        channel.write_float(b, self.high_ping_threshold)
        channel.write_string(b, self.http_proxy)

    def read(self, b: io.BytesIO) -> None:
        self.asyncio_debug = channel.read_bool(b)
        self.ping_interval = channel.read_float(b)
        self.ping_timeout = channel.read_float(b)
        self.high_ping_threshold = channel.read_float(b)
        self.http_proxy = channel.read_string(b)


@dataclass
class InitializeResponse:
    """mark the process as initialized"""

    MSG_ID: ClassVar[int] = 1
    error: str = ""

    def write(self, b: io.BytesIO) -> None:
        channel.write_string(b, self.error)

    def read(self, b: io.BytesIO) -> None:
        self.error = channel.read_string(b)


@dataclass
class PingRequest:
    """sent by the main process to the subprocess to check if it is still alive"""

    MSG_ID: ClassVar[int] = 2
    timestamp: int = 0

    def write(self, b: io.BytesIO) -> None:
        channel.write_long(b, self.timestamp)

    def read(self, b: io.BytesIO) -> None:
        self.timestamp = channel.read_long(b)


@dataclass
class PongResponse:
    """response to a PingRequest"""

    MSG_ID: ClassVar[int] = 3
    last_timestamp: int = 0
    timestamp: int = 0

    def write(self, b: io.BytesIO) -> None:
        channel.write_long(b, self.last_timestamp)
        channel.write_long(b, self.timestamp)

    def read(self, b: io.BytesIO) -> None:
        self.last_timestamp = channel.read_long(b)
        self.timestamp = channel.read_long(b)


@dataclass
class StartJobRequest:
    """sent by the main process to the subprocess to start a job, the subprocess will only
    receive this message if the process is fully initialized (after sending a InitializeResponse)."""  # noqa: E501

    MSG_ID: ClassVar[int] = 4
    running_job: RunningJobInfo = field(init=False)

    def write(self, b: io.BytesIO) -> None:
        accept_args = self.running_job.accept_arguments
        channel.write_bytes(b, self.running_job.job.SerializeToString())
        channel.write_string(b, accept_args.name)
        channel.write_string(b, accept_args.identity)
        channel.write_string(b, accept_args.metadata)
        channel.write_string(b, self.running_job.url)
        channel.write_string(b, self.running_job.token)
        channel.write_string(b, self.running_job.worker_id)

    def read(self, b: io.BytesIO) -> None:
        job = agent.Job()
        job.ParseFromString(channel.read_bytes(b))
        self.running_job = RunningJobInfo(
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


@dataclass
class ShutdownRequest:
    """sent by the main process to the subprocess to indicate that it should shut down
    gracefully. the subprocess will follow with a ExitInfo message"""

    MSG_ID: ClassVar[int] = 5
    reason: str = ""

    def write(self, b: io.BytesIO) -> None:
        channel.write_string(b, self.reason)

    def read(self, b: io.BytesIO) -> None:
        self.reason = channel.read_string(b)


@dataclass
class Exiting:
    """sent by the subprocess to the main process to indicate that it is exiting"""

    MSG_ID: ClassVar[int] = 6
    reason: str = ""

    def write(self, b: io.BytesIO) -> None:
        channel.write_string(b, self.reason)

    def read(self, b: io.BytesIO) -> None:
        self.reason = channel.read_string(b)


@dataclass
class InferenceRequest:
    """sent by a subprocess to the main process to request inference"""

    MSG_ID: ClassVar[int] = 7
    method: str = ""
    request_id: str = ""
    data: bytes = b""

    def write(self, b: io.BytesIO) -> None:
        channel.write_string(b, self.method)
        channel.write_string(b, self.request_id)
        channel.write_bytes(b, self.data)

    def read(self, b: io.BytesIO) -> None:
        self.method = channel.read_string(b)
        self.request_id = channel.read_string(b)
        self.data = channel.read_bytes(b)


@dataclass
class InferenceResponse:
    """response to an InferenceRequest"""

    MSG_ID: ClassVar[int] = 8
    request_id: str = ""
    data: bytes | None = None
    error: str = ""

    def write(self, b: io.BytesIO) -> None:
        channel.write_string(b, self.request_id)
        channel.write_bool(b, self.data is not None)
        if self.data is not None:
            channel.write_bytes(b, self.data)
        channel.write_string(b, self.error)

    def read(self, b: io.BytesIO) -> None:
        self.request_id = channel.read_string(b)
        has_data = channel.read_bool(b)
        if has_data:
            self.data = channel.read_bytes(b)
        self.error = channel.read_string(b)


@dataclass
class TracingRequest:
    MSG_ID: ClassVar[int] = 9
    request_id: str = ""

    def write(self, b: io.BytesIO) -> None:
        channel.write_string(b, self.request_id)

    def read(self, b: io.BytesIO) -> None:
        self.request_id = channel.read_string(b)


@dataclass
class TracingResponse:
    MSG_ID: ClassVar[int] = 10
    request_id: str = ""
    info: dict[str, Any] = field(default_factory=dict)

    def write(self, b: io.BytesIO) -> None:
        channel.write_string(b, self.request_id)
        channel.write_bytes(b, pickle.dumps(self.info))

    def read(self, b: io.BytesIO) -> None:
        self.request_id = channel.read_string(b)
        self.info = pickle.loads(channel.read_bytes(b))


IPC_MESSAGES = {
    InitializeRequest.MSG_ID: InitializeRequest,
    InitializeResponse.MSG_ID: InitializeResponse,
    PingRequest.MSG_ID: PingRequest,
    PongResponse.MSG_ID: PongResponse,
    StartJobRequest.MSG_ID: StartJobRequest,
    ShutdownRequest.MSG_ID: ShutdownRequest,
    Exiting.MSG_ID: Exiting,
    InferenceRequest.MSG_ID: InferenceRequest,
    InferenceResponse.MSG_ID: InferenceResponse,
    TracingRequest.MSG_ID: TracingRequest,
    TracingResponse.MSG_ID: TracingResponse,
}
