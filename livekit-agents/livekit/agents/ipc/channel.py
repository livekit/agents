from __future__ import annotations

import asyncio
import concurrent.futures
import io
import struct
from typing import ClassVar, Protocol, Type, runtime_checkable


class ProcessConn(Protocol):
    def recv_bytes(self, maxlength: int | None = None) -> bytes: ...

    def send_bytes(
        self,
        buf: bytes | bytearray | memoryview,
        offset: int = 0,
        size: int | None = None,
    ): ...

    def close(self) -> None: ...


class Message(Protocol):
    MSG_ID: ClassVar[int]


@runtime_checkable
class DataMessage(Message, Protocol):
    def write(self, b: io.BytesIO) -> None: ...

    def read(self, b: io.BytesIO) -> None: ...


class ChannelClosed(Exception):
    pass


class ProcChannel:
    def __init__(
        self,
        *,
        conn: ProcessConn,
        loop: asyncio.AbstractEventLoop,
        messages: dict[int, Type[Message]],
    ) -> None:
        self._conn = conn
        self._loop = loop
        self._messages = messages
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    @property
    def conn(self) -> ProcessConn:
        return self._conn

    async def asend(self, msg: Message) -> None:
        await self._loop.run_in_executor(self._executor, self.send, msg)

    async def arecv(self) -> Message:
        return await self._loop.run_in_executor(self._executor, self.recv)

    async def aclose(self) -> None:
        await self._loop.run_in_executor(self._executor, self._conn.close)

    def send(self, msg: Message) -> None:
        b = io.BytesIO()
        write_int(b, msg.MSG_ID)

        if isinstance(msg, DataMessage):
            msg.write(b)

        try:
            self._conn.send_bytes(b.getvalue())
        except (OSError, ValueError):
            raise ChannelClosed()

    def recv(self) -> Message:
        try:
            b = io.BytesIO(self._conn.recv_bytes())
        except (OSError, EOFError):
            raise ChannelClosed()

        msg_id = read_int(b)
        msg = self._messages[msg_id]()

        if isinstance(msg, DataMessage):
            msg.read(b)

        return msg

    def close(self) -> None:
        self._conn.close()


def write_bytes(b: io.BytesIO, buf: bytes) -> None:
    b.write(len(buf).to_bytes(4, "big"))
    b.write(buf)


def read_bytes(b: io.BytesIO) -> bytes:
    length = int.from_bytes(b.read(4), "big")
    return b.read(length)


def write_string(b: io.BytesIO, s: str) -> None:
    encoded = s.encode("utf-8")
    b.write(len(encoded).to_bytes(4, "big"))
    b.write(encoded)


def read_string(b: io.BytesIO) -> str:
    length = int.from_bytes(b.read(4), "big")
    return b.read(length).decode("utf-8")


def write_int(b: io.BytesIO, i: int) -> None:
    b.write(i.to_bytes(4, "big"))


def read_int(b: io.BytesIO) -> int:
    return int.from_bytes(b.read(4), "big")


def write_bool(b: io.BytesIO, bi: bool) -> None:
    b.write(bi.to_bytes(1, "big"))


def read_bool(b: io.BytesIO) -> bool:
    return bool.from_bytes(b.read(1), "big")


def write_float(b: io.BytesIO, f: float) -> None:
    b.write(struct.pack("f", f))


def read_float(b: io.BytesIO) -> float:
    return struct.unpack("f", b.read(4))[0]


def write_double(b: io.BytesIO, d: float) -> None:
    b.write(struct.pack("d", d))


def read_double(b: io.BytesIO) -> float:
    return struct.unpack("d", b.read(8))[0]


def write_long(b: io.BytesIO, long: int) -> None:
    b.write(long.to_bytes(8, "big"))


def read_long(b: io.BytesIO) -> int:
    return int.from_bytes(b.read(8), "big")
