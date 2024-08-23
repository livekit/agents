from __future__ import annotations

import io
import struct
from typing import ClassVar, Protocol, runtime_checkable

from .. import utils


class Message(Protocol):
    MSG_ID: ClassVar[int]


@runtime_checkable
class DataMessage(Message, Protocol):
    def write(self, b: io.BytesIO) -> None: ...

    def read(self, b: io.BytesIO) -> None: ...


MessagesDict = dict[int, type[Message]]


def _read_message(data: bytes, messages: MessagesDict) -> Message:
    bio = io.BytesIO(data)
    msg_id = read_int(bio)
    msg = messages[msg_id]()
    if isinstance(msg, DataMessage):
        msg.read(bio)

    return msg


def _write_message(msg: Message) -> bytes:
    bio = io.BytesIO()
    write_int(bio, msg.MSG_ID)

    if isinstance(msg, DataMessage):
        msg.write(bio)

    return bio.getvalue()


async def arecv_message(
    dplx: utils.aio.duplex_unix._AsyncDuplex, messages: MessagesDict
) -> Message:
    return _read_message(await dplx.recv_bytes(), messages)


async def asend_message(dplx: utils.aio.duplex_unix._AsyncDuplex, msg: Message) -> None:
    await dplx.send_bytes(_write_message(msg))


def recv_message(
    dplx: utils.aio.duplex_unix._Duplex, messages: MessagesDict
) -> Message:
    return _read_message(dplx.recv_bytes(), messages)


def send_message(dplx: utils.aio.duplex_unix._Duplex, msg: Message) -> None:
    dplx.send_bytes(_write_message(msg))


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
