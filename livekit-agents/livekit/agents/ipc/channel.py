from __future__ import annotations

import asyncio
import threading
import queue
import io
import struct
import contextlib
from typing import ClassVar, Protocol, Type, runtime_checkable, Optional


class ProcessConn(Protocol):
    def recv_bytes(self) -> bytes: ...

    def send_bytes(
        self,
        buf: bytes | bytearray | memoryview,
        offset: int = 0,
        size: int | None = None,
    ): ...

    def poll(self, timeout: float) -> bool: ...

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
        messages: dict[int, type[Message]],
    ) -> None:
        self._loop = loop
        self._conn = conn
        self._messages = messages
        self._closed = False

        self._read_q = asyncio.Queue[Optional[Message]]()
        self._write_q = queue.Queue[Optional[Message]]()
        self._exit_fut = asyncio.Future()

        self._read_t = threading.Thread(
            target=self._read_thread, daemon=True, name="proc_channel_read"
        )
        self._write_t = threading.Thread(
            target=self._write_thread, daemon=True, name="proc_channel_write"
        )
        self._read_t.start()
        self._write_t.start()

    def _read_thread(self) -> None:
        while True:
            try:
                b = io.BytesIO(self._conn.recv_bytes())
            except (OSError, EOFError):
                break

            msg_id = read_int(b)
            msg = self._messages[msg_id]()

            if isinstance(msg, DataMessage):
                msg.read(b)

            try:
                self._loop.call_soon_threadsafe(self._read_q.put_nowait, msg)
            except RuntimeError:
                break

        with contextlib.suppress(RuntimeError):

            def _close():
                self._exit_fut.set_result(None)
                self._read_q.put_nowait(None)
                self._send_close()

            self._loop.call_soon_threadsafe(_close)

    def _write_thread(self) -> None:
        while True:
            msg = self._write_q.get()
            if msg is None:
                break

            b = io.BytesIO()
            write_int(b, msg.MSG_ID)

            if isinstance(msg, DataMessage):
                msg.write(b)

            try:
                self._conn.send_bytes(b.getvalue())
            except (OSError, ValueError):
                break

        self._conn.close()

    async def arecv(self) -> Message:
        if self._closed:
            raise ChannelClosed()

        msg = await self._read_q.get()
        if msg is None:
            raise ChannelClosed()

        return msg

    async def asend(self, msg: Message) -> None:
        if self._closed:
            raise ChannelClosed()

        self._write_q.put_nowait(msg)

    async def aclose(self) -> None:
        self._send_close()
        # it seems like the conn close fnc could deadlock if the child process
        # crashed or was killed?
        await asyncio.wait_for(self._exit_fut, timeout=10.0)

    def _send_close(self) -> None:
        if self._closed:
            return

        self._closed = True
        self._write_q.put_nowait(None)


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
