from __future__ import annotations

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


class ProcessPipeReader(Protocol):
    def recv_bytes(self, maxlength: int) -> bytes:
        ...


class ProcessPipeWriter(Protocol):
    def send_bytes(
        self,
        buf: bytes | bytearray | memoryview,
        offset: int = 0,
        size: int | None = None,
    ):
        ...


class ProcessPipe(ProcessPipeReader, ProcessPipeWriter, Protocol):
    ...


class Message(Protocol):
    MSG_ID: ClassVar[int]

    def write(self, p: ProcessPipeWriter) -> None:
        ...

    def read(self, p: ProcessPipeReader) -> None:
        ...


@staticmethod
def read_msg(p: ProcessPipeReader) -> "Message":
    msg_id = int.from_bytes(p.recv_bytes(4))
    msg = MESSAGES[msg_id]()
    msg.read(p)
    return msg


@staticmethod
def write_msg(p: ProcessPipeWriter, msg: "Message") -> None:
    p.send_bytes(msg.MSG_ID.to_bytes(4))
    msg.write(p)


@define(kw_only=True)
class StartJobRequest:
    MSG_ID: ClassVar[int] = 0
    job: agent.Job = agent.Job()

    def write(self, p: ProcessPipeWriter) -> None:
        job_s = self.job.SerializeToString()
        p.send_bytes(len(job_s).to_bytes(4))
        p.send_bytes(job_s)

    def read(self, p: ProcessPipeReader) -> None:
        job_len = int.from_bytes(p.recv_bytes(4))
        self.job = agent.Job()
        self.job.ParseFromString(p.recv_bytes(job_len))


@define(kw_only=True)
class StartJobResponse:
    MSG_ID: ClassVar[int] = 1
    exc: BaseException | None = None

    def write(self, p: ProcessPipeWriter) -> None:
        if self.exc is None:
            p.send_bytes(bytes(4))
        else:
            exc_s = pickle.dumps(self.exc)
            p.send_bytes(len(exc_s).to_bytes(4))
            p.send_bytes(pickle.dumps(self.exc))

    def read(self, p: ProcessPipeReader) -> None:
        exc_len = int.from_bytes(p.recv_bytes(4))
        if exc_len == 0:
            self.exc = None
        else:
            self.exc = pickle.loads(p.recv_bytes(exc_len))


@define(kw_only=True)
class Log:
    MSG_ID: ClassVar[int] = 2
    level: int = 0  # logging._Level
    message: str = ""

    def write(self, p: ProcessPipeWriter) -> None:
        p.send_bytes(self.level.to_bytes(4))
        message_s = self.message.encode()
        p.send_bytes(len(message_s).to_bytes(4))
        p.send_bytes(message_s)

    def read(self, p: ProcessPipeReader) -> None:
        self.level = int.from_bytes(p.recv_bytes(4))
        message_len = int.from_bytes(p.recv_bytes(4))
        self.message = p.recv_bytes(message_len).decode()


@define(kw_only=True)
class Ping:
    MSG_ID: ClassVar[int] = 3
    timestamp: int = 0

    def write(self, p: ProcessPipeWriter) -> None:
        p.send_bytes(self.timestamp.to_bytes(8))

    def read(self, p: ProcessPipeReader) -> None:
        self.timestamp = int.from_bytes(p.recv_bytes(8))


@define(kw_only=True)
class Pong:
    MSG_ID: ClassVar[int] = 4
    last_timestamp: int = 0
    timestamp: int = 0

    def write(self, p: ProcessPipeWriter) -> None:
        p.send_bytes(self.last_timestamp.to_bytes(8))
        p.send_bytes(self.timestamp.to_bytes(8))

    def read(self, p: ProcessPipeReader) -> None:
        self.last_timestamp = int.from_bytes(p.recv_bytes(8))
        self.timestamp = int.from_bytes(p.recv_bytes(8))


@define(kw_only=True)
class ShutdownRequest:
    MSG_ID: ClassVar[int] = 5

    def write(self, p: ProcessPipeWriter) -> None:
        pass

    def read(self, p: ProcessPipeReader) -> None:
        pass


@define(kw_only=True)
class ShutdownResponse:
    MSG_ID: ClassVar[int] = 6

    def write(self, p: ProcessPipeWriter) -> None:
        pass

    def read(self, p: ProcessPipeReader) -> None:
        pass


@define(kw_only=True)
class UserExit:
    MSG_ID: ClassVar[int] = 7

    def write(self, p: ProcessPipeWriter) -> None:
        pass

    def read(self, p: ProcessPipeReader) -> None:
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
