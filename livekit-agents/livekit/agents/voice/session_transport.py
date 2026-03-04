from __future__ import annotations

from abc import ABC, abstractmethod

from livekit.protocol.agent_pb import agent_session as agent_pb


class SessionTransport(ABC):
    """Transport-agnostic interface for agent session communication.

    Implementations handle the actual wire protocol (TCP, Room streams, etc.)
    while consumers work with protobuf SessionMessage objects.
    """

    @abstractmethod
    async def send_message(self, msg: agent_pb.SessionMessage) -> None:
        """Send a SessionMessage to the remote peer."""
        ...

    @abstractmethod
    async def recv_message(self) -> agent_pb.SessionMessage | None:
        """Receive a SessionMessage from the remote peer.

        Returns None when the connection is closed.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close the transport connection."""
        ...
