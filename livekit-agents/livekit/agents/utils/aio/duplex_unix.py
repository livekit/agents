from __future__ import annotations

import asyncio
import socket
import struct


class DuplexClosed(Exception):
    """Exception raised when the duplex connection is closed."""

    pass


class _AsyncDuplex:
    def __init__(
        self,
        sock: socket.socket,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self._loop = loop
        self._sock = sock
        self._reader = reader
        self._writer = writer

    @staticmethod
    async def open(sock: socket.socket) -> _AsyncDuplex:
        loop = asyncio.get_running_loop()
        reader, writer = await asyncio.open_connection(sock=sock)
        return _AsyncDuplex(sock, reader, writer, loop)

    async def recv_bytes(self) -> bytes:
        try:
            len_bytes = await self._reader.readexactly(4)
            len = struct.unpack("!I", len_bytes)[0]
            return await self._reader.readexactly(len)
        except (
            OSError,
            EOFError,
            asyncio.IncompleteReadError,
        ):
            raise DuplexClosed()

    async def send_bytes(self, data: bytes) -> None:
        try:
            len_bytes = struct.pack("!I", len(data))
            self._writer.write(len_bytes)
            self._writer.write(data)
            await self._writer.drain()
        except OSError:
            raise DuplexClosed()

    async def aclose(self) -> None:
        try:
            self._writer.close()
            await self._writer.wait_closed()
            self._sock.close()
        except OSError:
            raise DuplexClosed()


def _read_exactly(sock: socket.socket, num_bytes: int) -> bytes:
    data = bytearray()
    while len(data) < num_bytes:
        packet = sock.recv(num_bytes - len(data))
        if not packet:
            raise EOFError()
        data.extend(packet)
    return bytes(data)


class _Duplex:
    def __init__(self, sock: socket.socket) -> None:
        self._sock: socket.socket | None = sock

    @staticmethod
    def open(sock: socket.socket) -> _Duplex:
        return _Duplex(sock)

    def recv_bytes(self) -> bytes:
        if self._sock is None:
            raise DuplexClosed()

        try:
            len_bytes = _read_exactly(self._sock, 4)
            len = struct.unpack("!I", len_bytes)[0]
            return _read_exactly(self._sock, len)
        except (OSError, EOFError):
            raise DuplexClosed()

    def send_bytes(self, data: bytes) -> None:
        if self._sock is None:
            raise DuplexClosed()

        try:
            len_bytes = struct.pack("!I", len(data))
            self._sock.sendall(len_bytes)
            self._sock.sendall(data)
        except OSError:
            raise DuplexClosed()

    def detach(self) -> socket.socket:
        if self._sock is None:
            raise DuplexClosed()

        sock = self._sock
        self._sock = None
        return sock

    def close(self) -> None:
        try:
            if self._sock is not None:
                self._sock.close()
                self._sock = None
        except OSError:
            raise DuplexClosed()
