from __future__ import annotations

import asyncio
import logging
import struct

from livekit.protocol.agent_pb import agent_session as agent_pb

from .session_transport import SessionTransport

logger = logging.getLogger(__name__)

# Wire format: [4 bytes BE length][protobuf bytes]
_HEADER_SIZE = 4
_MAX_MESSAGE_SIZE = 1 << 20  # 1MB


class TcpSessionTransport(SessionTransport):
    """TCP transport using [4B len][protobuf] framing."""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        self._reader = reader
        self._writer = writer
        self._closed = False
        self._loop = asyncio.get_running_loop()

    @classmethod
    async def connect(cls, host: str, port: int) -> TcpSessionTransport:
        """Connect to a TCP server and return a transport."""
        reader, writer = await asyncio.open_connection(host, port)
        # Disable Nagle's algorithm for low-latency audio frame delivery
        sock = writer.transport.get_extra_info("socket")
        if sock is not None:
            import socket

            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return cls(reader, writer)

    async def send_message(self, msg: agent_pb.AgentSessionMessage) -> None:
        """Send a message from the transport's own event loop."""
        if self._closed:
            return
        data = msg.SerializeToString()
        header = struct.pack(">I", len(data))
        self._writer.write(header + data)
        if self._writer.transport.get_write_buffer_size() > 64 * 1024:
            await self._writer.drain()

    def send_message_threadsafe(self, msg: agent_pb.AgentSessionMessage) -> None:
        """Send a message from any thread. Schedules the write on the transport's loop."""
        if self._closed:
            return
        data = msg.SerializeToString()
        payload = struct.pack(">I", len(data)) + data
        self._loop.call_soon_threadsafe(self._writer.write, payload)

    async def recv_message(self) -> agent_pb.AgentSessionMessage | None:
        try:
            header = await self._reader.readexactly(_HEADER_SIZE)
        except (asyncio.IncompleteReadError, ConnectionError, OSError):
            return None

        length = struct.unpack(">I", header)[0]
        if length > _MAX_MESSAGE_SIZE:
            logger.error("TCP message too large: %d bytes", length)
            return None

        try:
            data = await self._reader.readexactly(length)
        except (asyncio.IncompleteReadError, ConnectionError, OSError):
            return None

        msg = agent_pb.AgentSessionMessage()
        msg.ParseFromString(data)
        return msg

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._writer.close()
            await self._writer.wait_closed()
        except (ConnectionError, OSError):
            pass
