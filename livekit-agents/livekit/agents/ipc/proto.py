from __future__ import annotations

import copy
import threading
import logging
import io
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Coroutine

from livekit.protocol import agent

from ..job import JobAcceptArguments, JobContext, JobProcess
from . import channel

PING_INTERVAL = 5
PING_TIMEOUT = 90
HIGH_PING_THRESHOLD = 0.02  # 20ms


@dataclass
class ProcStartArgs:
    initialize_process_fnc: Callable[[JobProcess], Any]
    job_entrypoint_fnc: Callable[[JobContext], Coroutine]
    job_shutdown_fnc: Callable[[JobContext], Coroutine]
    log_q: mp.Queue
    mp_cch: channel.ProcessConn
    asyncio_debug: bool
    user_arguments: Any | None = None


@dataclass
class InitializeRequest:
    """sent by the main process to the subprocess to initialize it. this is going to call initialize_process_fnc"""

    MSG_ID: ClassVar[int] = 0


@dataclass
class InitializeResponse:
    """mark the process as initialized"""

    MSG_ID: ClassVar[int] = 1


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
    receive this message if the process is fully initialized (after sending a InitializeResponse)."""

    MSG_ID: ClassVar[int] = 4
    job: agent.Job = field(default_factory=agent.Job)
    accept_args: JobAcceptArguments = field(init=False)
    url: str = ""
    token: str = ""

    def write(self, b: io.BytesIO) -> None:
        channel.write_bytes(b, self.job.SerializeToString())
        channel.write_string(b, self.accept_args.name)
        channel.write_string(b, self.accept_args.identity)
        channel.write_string(b, self.accept_args.metadata)
        channel.write_string(b, self.url)
        channel.write_string(b, self.token)

    def read(self, b: io.BytesIO) -> None:
        self.job.ParseFromString(channel.read_bytes(b))
        self.accept_args = JobAcceptArguments(
            name=channel.read_string(b),
            identity=channel.read_string(b),
            metadata=channel.read_string(b),
        )
        self.url = channel.read_string(b)
        self.token = channel.read_string(b)


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


IPC_MESSAGES = {
    InitializeRequest.MSG_ID: InitializeRequest,
    InitializeResponse.MSG_ID: InitializeResponse,
    PingRequest.MSG_ID: PingRequest,
    PongResponse.MSG_ID: PongResponse,
    StartJobRequest.MSG_ID: StartJobRequest,
    ShutdownRequest.MSG_ID: ShutdownRequest,
    Exiting.MSG_ID: Exiting,
}
