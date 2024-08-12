from __future__ import annotations

import asyncio
import socket
import struct


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
        except asyncio.IncompleteReadError:
            raise EOFError()

    async def send_bytes(self, data: bytes) -> None:
        len_bytes = struct.pack("!I", len(data))
        self._writer.write(len_bytes)
        self._writer.write(data)
        await self._writer.drain()

    async def aclose(self) -> None:
        self._writer.close()
        await self._writer.wait_closed()
        self._sock.close()


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
        self._sock = sock

    @staticmethod
    def open(sock: socket.socket) -> _Duplex:
        return _Duplex(sock)

    def recv_bytes(self) -> bytes:
        len_bytes = _read_exactly(self._sock, 4)
        len = struct.unpack("!I", len_bytes)[0]
        return _read_exactly(self._sock, len)

    def send_bytes(self, data: bytes) -> None:
        len_bytes = struct.pack("!I", len(data))
        self._sock.sendall(len_bytes)
        self._sock.sendall(data)

    def detach(self) -> socket.socket:
        sock = self._sock
        self._sock = None
        return sock

    def close(self) -> None:
        self._sock.close()
