from __future__ import annotations

import io
import pickle
from typing import Callable, ClassVar, Protocol

from attr import define
from livekit.protocol import agent


@define
class JobMainArgs:
    job_id: str
    url: str
    token: str
    target: Callable
    asyncio_debug: bool


class ProcessPipeReader(Protocol):
    def recv_bytes(self, maxlength: int | None = None) -> bytes:
        ...

    def close(self) -> None:
        ...


class ProcessPipeWriter(Protocol):
    def send_bytes(
        self,
        buf: bytes | bytearray | memoryview,
        offset: int = 0,
        size: int | None = None,
    ):
        ...

    def close(self) -> None:
        ...


class ProcessPipe(ProcessPipeReader, ProcessPipeWriter, Protocol):
    ...


class Message(Protocol):
    MSG_ID: ClassVar[int]

    def write(self, b: io.BytesIO) -> None:
        ...

    def read(self, b: io.BytesIO) -> None:
        ...


@staticmethod
def read_msg(p: ProcessPipeReader) -> "Message":
    b = io.BytesIO(p.recv_bytes())
    msg_id = int.from_bytes(b.read(4))
    msg = MESSAGES[msg_id]()
    msg.read(b)
    return msg


@staticmethod
def write_msg(p: ProcessPipeWriter, msg: "Message") -> None:
    b = io.BytesIO()
    b.write(msg.MSG_ID.to_bytes(4))
    msg.write(b)
    p.send_bytes(b.getvalue())


@define(kw_only=True)
class StartJobRequest:
    MSG_ID: ClassVar[int] = 0
    job: agent.Job = agent.Job()

    def write(self, b: io.BytesIO) -> None:
        job_s = self.job.SerializeToString()
        b.write(len(job_s).to_bytes(4))
        b.write(job_s)

    def read(self, b: io.BytesIO) -> None:
        job_len = int.from_bytes(b.read(4))
        self.job = agent.Job()
        self.job.ParseFromString(b.read(job_len))


@define(kw_only=True)
class StartJobResponse:
    MSG_ID: ClassVar[int] = 1
    exc: BaseException | None = None

    def write(self, b: io.BytesIO) -> None:
        if self.exc is None:
            b.write(bytes(4))
        else:
            exc_s = pickle.dumps(self.exc)
            b.write(len(exc_s).to_bytes(4))
            b.write(pickle.dumps(self.exc))

    def read(self, b: io.BytesIO) -> None:
        exc_len = int.from_bytes(b.read(4))
        if exc_len == 0:
            self.exc = None
        else:
            self.exc = pickle.loads(b.read(exc_len))


@define(kw_only=True)
class Log:
    MSG_ID: ClassVar[int] = 2
    level: int = 0  # logging._Level
    message: str = ""

    def write(self, b: io.BytesIO) -> None:
        b.write(self.level.to_bytes(4))
        message_s = self.message.encode()
        b.write(len(message_s).to_bytes(4))
        b.write(message_s)

    def read(self, b: io.BytesIO) -> None:
        self.level = int.from_bytes(b.read(4))
        message_len = int.from_bytes(b.read(4))
        self.message = b.read(message_len).decode()


@define(kw_only=True)
class Ping:
    MSG_ID: ClassVar[int] = 3
    timestamp: int = 0

    def write(self, b: io.BytesIO) -> None:
        b.write(self.timestamp.to_bytes(8))

    def read(self, b: io.BytesIO) -> None:
        self.timestamp = int.from_bytes(b.read(8))


@define(kw_only=True)
class Pong:
    MSG_ID: ClassVar[int] = 4
    last_timestamp: int = 0
    timestamp: int = 0

    def write(self, b: io.BytesIO) -> None:
        b.write(self.last_timestamp.to_bytes(8))
        b.write(self.timestamp.to_bytes(8))

    def read(self, b: io.BytesIO) -> None:
        self.last_timestamp = int.from_bytes(b.read(8))
        self.timestamp = int.from_bytes(b.read(8))


@define(kw_only=True)
class ShutdownRequest:
    MSG_ID: ClassVar[int] = 5

    def write(self, b: io.BytesIO) -> None:
        pass

    def read(self, b: io.BytesIO) -> None:
        pass


@define(kw_only=True)
class ShutdownResponse:
    MSG_ID: ClassVar[int] = 6

    def write(self, b: io.BytesIO) -> None:
        pass

    def read(self, b: io.BytesIO) -> None:
        pass


@define(kw_only=True)
class UserExit:
    MSG_ID: ClassVar[int] = 7

    def write(self, b: io.BytesIO) -> None:
        pass

    def read(self, b: io.BytesIO) -> None:
        pass


MESSAGES = {
    StartJobRequest.MSG_ID: StartJobRequest,
    StartJobResponse.MSG_ID: StartJobResponse,
    Log.MSG_ID: Log,
    Ping.MSG_ID: Ping,
    Pong.MSG_ID: Pong,
    ShutdownRequest.MSG_ID: ShutdownRequest,
    ShutdownResponse.MSG_ID: ShutdownResponse,
    UserExit.MSG_ID: UserExit,
}
