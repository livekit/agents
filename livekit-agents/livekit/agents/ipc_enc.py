import io
import struct
from typing import ClassVar, Protocol, Type


class ProcessPipeReader(Protocol):
    def recv_bytes(self, maxlength: int | None = None) -> bytes: ...

    def poll(self, timeout: float = 0.0) -> bool: ...

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


def read_msg(p: ProcessPipeReader, messages: dict[int, Type[Message]]) -> "Message":
    b = io.BytesIO(p.recv_bytes())
    msg_id = int.from_bytes(b.read(4), "big")
    msg = messages[msg_id]()
    msg.read(b)
    return msg


def write_msg(p: ProcessPipeWriter, msg: "Message") -> None:
    b = io.BytesIO()
    b.write(msg.MSG_ID.to_bytes(4, "big"))
    msg.write(b)
    p.send_bytes(b.getvalue())


# some utils for cleaner proto code


def _write_bytes(b: io.BytesIO, buf: bytes) -> None:
    b.write(len(buf).to_bytes(4, "big"))
    b.write(buf)


def _read_bytes(b: io.BytesIO) -> bytes:
    length = int.from_bytes(b.read(4), "big")
    return b.read(length)


def _write_string(b: io.BytesIO, s: str) -> None:
    encoded = s.encode("utf-8")
    b.write(len(encoded).to_bytes(4, "big"))
    b.write(encoded)


def _read_string(b: io.BytesIO) -> str:
    length = int.from_bytes(b.read(4), "big")
    return b.read(length).decode("utf-8")


def _write_int(b: io.BytesIO, i: int) -> None:
    b.write(i.to_bytes(4, "big"))


def _read_int(b: io.BytesIO) -> int:
    return int.from_bytes(b.read(4), "big")


def _write_bool(b: io.BytesIO, bi: bool) -> None:
    b.write(bi.to_bytes(1, "big"))


def _read_bool(b: io.BytesIO) -> bool:
    return bool.from_bytes(b.read(1), "big")


def _write_float(b: io.BytesIO, f: float) -> None:
    b.write(struct.pack("f", f))


def _read_float(b: io.BytesIO) -> float:
    return struct.unpack("f", b.read(4))[0]


def _write_double(b: io.BytesIO, d: float) -> None:
    b.write(struct.pack("d", d))


def _read_double(b: io.BytesIO) -> float:
    return struct.unpack("d", b.read(8))[0]


def _write_long(b: io.BytesIO, long: int) -> None:
    b.write(long.to_bytes(8, "big"))


def _read_long(b: io.BytesIO) -> int:
    return int.from_bytes(b.read(8), "big")
