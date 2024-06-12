from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import ClassVar

from livekit.protocol import agent

from .. import ipc_enc
from ..job_request import AcceptData


@dataclass
class JobMainArgs:
    job_id: str
    url: str
    token: str
    accept_data: AcceptData
    asyncio_debug: bool


@dataclass
class StartJobRequest:
    MSG_ID: ClassVar[int] = 0
    job: agent.Job = field(default_factory=agent.Job)

    def write(self, b: io.BytesIO) -> None:
        ipc_enc._write_bytes(b, self.job.SerializeToString())

    def read(self, b: io.BytesIO) -> None:
        self.job.ParseFromString(ipc_enc._read_bytes(b))


@dataclass
class StartJobResponse:
    MSG_ID: ClassVar[int] = 1
    error: str = ""

    def write(self, b: io.BytesIO) -> None:
        ipc_enc._write_string(b, self.error)

    def read(self, b: io.BytesIO) -> None:
        self.error = ipc_enc._read_string(b)


@dataclass
class Log:
    MSG_ID: ClassVar[int] = 2
    level: int = 0  # logging._Level
    logger_name: str = ""
    message: str = ""

    def write(self, b: io.BytesIO) -> None:
        ipc_enc._write_int(b, self.level)
        ipc_enc._write_string(b, self.logger_name)
        ipc_enc._write_string(b, self.message)

    def read(self, b: io.BytesIO) -> None:
        self.level = ipc_enc._read_int(b)
        self.logger_name = ipc_enc._read_string(b)
        self.message = ipc_enc._read_string(b)


@dataclass
class Ping:
    MSG_ID: ClassVar[int] = 3
    timestamp: int = 0

    def write(self, b: io.BytesIO) -> None:
        b.write(self.timestamp.to_bytes(8, "big"))

    def read(self, b: io.BytesIO) -> None:
        self.timestamp = int.from_bytes(b.read(8), "big")


@dataclass
class Pong:
    MSG_ID: ClassVar[int] = 4
    last_timestamp: int = 0
    timestamp: int = 0

    def write(self, b: io.BytesIO) -> None:
        ipc_enc._write_long(b, self.last_timestamp)
        ipc_enc._write_long(b, self.timestamp)

    def read(self, b: io.BytesIO) -> None:
        self.last_timestamp = ipc_enc._read_long(b)
        self.timestamp = ipc_enc._read_long(b)


@dataclass
class ShutdownRequest:
    MSG_ID: ClassVar[int] = 5

    def write(self, b: io.BytesIO) -> None:
        pass

    def read(self, b: io.BytesIO) -> None:
        pass


@dataclass
class ShutdownResponse:
    MSG_ID: ClassVar[int] = 6

    def write(self, b: io.BytesIO) -> None:
        pass

    def read(self, b: io.BytesIO) -> None:
        pass


@dataclass
class UserExit:
    MSG_ID: ClassVar[int] = 7
    reason: str = ""

    def write(self, b: io.BytesIO) -> None:
        ipc_enc._write_string(b, self.reason)

    def read(self, b: io.BytesIO) -> None:
        self.reason = ipc_enc._read_string(b)


IPC_MESSAGES = {
    StartJobRequest.MSG_ID: StartJobRequest,
    StartJobResponse.MSG_ID: StartJobResponse,
    Log.MSG_ID: Log,
    Ping.MSG_ID: Ping,
    Pong.MSG_ID: Pong,
    ShutdownRequest.MSG_ID: ShutdownRequest,
    ShutdownResponse.MSG_ID: ShutdownResponse,
    UserExit.MSG_ID: UserExit,
}
