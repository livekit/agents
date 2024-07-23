from __future__ import annotations

import asyncio
import contextlib
import io
import queue
import struct
import threading
from typing import ClassVar, Optional, Protocol, runtime_checkable


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
    _sentinel = None

    def __init__(
        self,
        *,
        conn: ProcessConn,
        messages: dict[int, type[Message]],
    ) -> None:
        self._conn = conn
        self._messages = messages
        self._closed = False

    def recv(self) -> Message:
        if self._closed:
            raise ChannelClosed()

        try:
            b = io.BytesIO(self._conn.recv_bytes())
        except (OSError, EOFError, ValueError):
            raise ChannelClosed()

        msg_id = read_int(b)
        msg = self._messages[msg_id]()

        if isinstance(msg, DataMessage):
            msg.read(b)

        return msg

    def send(self, msg: Message) -> None:
        if self._closed:
            raise ChannelClosed()

        b = io.BytesIO()
        write_int(b, msg.MSG_ID)

        if isinstance(msg, DataMessage):
            msg.write(b)

        try:
            self._conn.send_bytes(b.getvalue())
        except (OSError, ValueError):
            raise ChannelClosed()

    def close(self) -> None:
        if self._closed:
            return

        self._closed = True
        self._conn.close()


class AsyncProcChannel(ProcChannel):
    def __init__(
        self,
        *,
        conn: ProcessConn,
        messages: dict[int, type[Message]],
        loop: asyncio.AbstractEventLoop,
    ):
        super().__init__(conn=conn, messages=messages)
        self._loop = loop

        self._read_q = asyncio.Queue[Optional[Message]]()
        self._write_q = queue.Queue[Optional[Message]]()
        self._exit_fut = asyncio.Future[None]()

        self._read_t = threading.Thread(
            target=self._read_thread, daemon=True, name="proc_channel_read"
        )
        self._write_t = threading.Thread(
            target=self._write_thread, daemon=True, name="proc_channel_write"
        )
        self._read_t.start()
        self._write_t.start()
        self._closed = False

    async def arecv(self) -> Message:
        if self._closed:
            raise ChannelClosed()

        msg = await self._read_q.get()
        if msg is self._sentinel:
            raise ChannelClosed()

        return msg

    async def asend(self, msg: Message) -> None:
        if self._closed:
            raise ChannelClosed()

        self._write_q.put_nowait(msg)

    async def aclose(self) -> None:
        self.close()
        await self._exit_fut

    def _read_thread(self) -> None:
        while True:
            try:
                if self._conn.poll(1.0):
                    msg = self.recv()
                    try:
                        self._loop.call_soon_threadsafe(self._read_q.put_nowait, msg)
                    except RuntimeError:
                        break
            except (OSError, EOFError, ValueError):
                break
            except ChannelClosed:
                break

        with contextlib.suppress(RuntimeError):

            def _close():
                self._exit_fut.set_result(None)
                self._read_q.put_nowait(self._sentinel)
                self._write_q.put_nowait(self._sentinel)
                self.close()

            self._loop.call_soon_threadsafe(_close)

    def _write_thread(self) -> None:
        while True:
            msg = self._write_q.get()
            if msg is self._sentinel:
                break

            try:
                self.send(msg)
            except ChannelClosed:
                break


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
