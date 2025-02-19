"""
Unix socket duplex communication utilities for async message passing.

Provides length-prefixed message framing over socket connections.
"""

from __future__ import annotations

import asyncio
import socket
import struct


class DuplexClosed(Exception):
    """Exception raised when attempting to use a closed duplex connection."""

    pass


class _AsyncDuplex:
    """Async duplex communication channel over sockets.
    
    Features:
    - Length-prefixed message framing
    - Async/await compatible
    - Automatic connection management
    
    Usage:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect("/tmp/socket.sock")
        duplex = await _AsyncDuplex.open(sock)
        await duplex.send_bytes(b"hello")
        response = await duplex.recv_bytes()
    """

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
        """Create async duplex from existing socket.
        
        Args:
            sock: Connected socket instance
            
        Returns:
            Configured _AsyncDuplex instance
        """
        loop = asyncio.get_running_loop()
        reader, writer = await asyncio.open_connection(sock=sock)
        return _AsyncDuplex(sock, reader, writer, loop)

    async def recv_bytes(self) -> bytes:
        """Receive length-prefixed message.
        
        Returns:
            bytes: Received message payload
            
        Raises:
            DuplexClosed: If connection is closed during receive
        """
        try:
            len_bytes = await self._reader.readexactly(4)
            msg_len = struct.unpack("!I", len_bytes)[0]
            return await self._reader.readexactly(msg_len)
        except (
            OSError,
            EOFError,
            asyncio.IncompleteReadError,
        ):
            raise DuplexClosed()

    async def send_bytes(self, data: bytes) -> None:
        """Send length-prefixed message.
        
        Args:
            data: Bytes to send
            
        Raises:
            DuplexClosed: If connection is closed during send
        """
        try:
            len_bytes = struct.pack("!I", len(data))
            self._writer.write(len_bytes)
            self._writer.write(data)
            await self._writer.drain()
        except OSError:
            raise DuplexClosed()

    async def aclose(self) -> None:
        """Close connection and release resources."""
        try:
            self._writer.close()
            await self._writer.wait_closed()
            self._sock.close()
        except OSError:
            raise DuplexClosed()


def _read_exactly(sock: socket.socket, num_bytes: int) -> bytes:
    """Blocking read of exact number of bytes from socket.
    
    Args:
        sock: Connected socket
        num_bytes: Number of bytes to read
        
    Returns:
        bytes: Read data
        
    Raises:
        EOFError: If socket closes before reading all bytes
    """
    data = bytearray()
    while len(data) < num_bytes:
        packet = sock.recv(num_bytes - len(data))
        if not packet:
            raise EOFError()
        data.extend(packet)
    return bytes(data)


class _Duplex:
    """Synchronous duplex communication channel over sockets.
    
    Provides blocking I/O with same message framing as async version.
    """

    def __init__(self, sock: socket.socket) -> None:
        self._sock: socket.socket | None = sock

    @staticmethod
    def open(sock: socket.socket) -> _Duplex:
        """Create sync duplex from existing socket.
        
        Args:
            sock: Connected socket instance
            
        Returns:
            Configured _Duplex instance
        """
        return _Duplex(sock)

    def recv_bytes(self) -> bytes:
        """Receive length-prefixed message (blocking).
        
        Returns:
            bytes: Received message payload
            
        Raises:
            DuplexClosed: If connection is closed
        """
        if self._sock is None:
            raise DuplexClosed()

        try:
            len_bytes = _read_exactly(self._sock, 4)
            msg_len = struct.unpack("!I", len_bytes)[0]
            return _read_exactly(self._sock, msg_len)
        except (OSError, EOFError):
            raise DuplexClosed()

    def send_bytes(self, data: bytes) -> None:
        """Send length-prefixed message (blocking).
        
        Args:
            data: Bytes to send
            
        Raises:
            DuplexClosed: If connection is closed
        """
        if self._sock is None:
            raise DuplexClosed()

        try:
            len_bytes = struct.pack("!I", len(data))
            self._sock.sendall(len_bytes)
            self._sock.sendall(data)
        except OSError:
            raise DuplexClosed()

    def detach(self) -> socket.socket:
        """Return underlying socket without closing it.
        
        Returns:
            socket.socket: Detached socket instance
            
        Raises:
            DuplexClosed: If already closed
        """
        if self._sock is None:
            raise DuplexClosed()

        sock = self._sock
        self._sock = None
        return sock

    def close(self) -> None:
        """Close connection and release resources."""
        try:
            if self._sock is not None:
                self._sock.close()
                self._sock = None
        except OSError:
            raise DuplexClosed()
