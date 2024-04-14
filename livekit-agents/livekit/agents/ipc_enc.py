import io
from typing import ClassVar, Protocol, Type


class ProcessPipeReader(Protocol):
    def recv_bytes(self, maxlength: int | None = None) -> bytes: ...

    def close(self) -> None: ...


class ProcessPipeWriter(Protocol):
    def send_bytes(
        self,
        buf: bytes | bytearray | memoryview,
        offset: int = 0,
        size: int | None = None,
    ): ...

    def close(self) -> None: ...


class ProcessPipe(ProcessPipeReader, ProcessPipeWriter, Protocol): ...


class Message(Protocol):
    MSG_ID: ClassVar[int]

    def write(self, b: io.BytesIO) -> None: ...

    def read(self, b: io.BytesIO) -> None: ...


@staticmethod
def read_msg(p: ProcessPipeReader, messages: dict[int, Type[Message]]) -> "Message":
    b = io.BytesIO(p.recv_bytes())
    msg_id = int.from_bytes(b.read(4), "big")
    msg = messages[msg_id]()
    msg.read(b)
    return msg


@staticmethod
def write_msg(p: ProcessPipeWriter, msg: "Message") -> None:
    b = io.BytesIO()
    b.write(msg.MSG_ID.to_bytes(4, "big"))
    msg.write(b)
    p.send_bytes(b.getvalue())
