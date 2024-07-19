import io
from dataclasses import dataclass
from typing import Callable, Any, Coroutine, ClassVar

from ..job import JobProcess, JobContext
from . import channel

import multiprocessing as mp


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


@dataclass
class InitializeRequest:
    MSG_ID: ClassVar[int] = 0


@dataclass
class InitializeResponse:
    MSG_ID: ClassVar[int] = 1


@dataclass
class PingRequest:
    MSG_ID: ClassVar[int] = 2
    timestamp: int = 0

    def write(self, b: io.BytesIO) -> None:
        channel.write_long(b, self.timestamp)

    def read(self, b: io.BytesIO) -> None:
        self.timestamp = channel.read_long(b)


@dataclass
class PongResponse:
    MSG_ID: ClassVar[int] = 3
    last_timestamp: int = 0
    timestamp: int = 0

    def write(self, b: io.BytesIO) -> None:
        channel.write_long(b, self.last_timestamp)
        channel.write_long(b, self.timestamp)

    def read(self, b: io.BytesIO) -> None:
        self.last_timestamp = channel.read_long(b)
        self.timestamp = channel.read_long(b)


IPC_MESSAGES = {
    InitializeRequest.MSG_ID: InitializeRequest,
    InitializeResponse.MSG_ID: InitializeResponse,
    PingRequest.MSG_ID: PingRequest,
    PongResponse.MSG_ID: PongResponse,
}
