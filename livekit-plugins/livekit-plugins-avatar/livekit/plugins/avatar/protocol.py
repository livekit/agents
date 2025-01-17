from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Protocol

from livekit import rtc
from pydantic import BaseModel


class PlaybackFinishedEvent(BaseModel):
    """Event sent from worker to sink when audio playback is finished"""

    stream_id: str
    interrupted: bool


class AvatarProtocol(ABC):
    """Protocol interface for communication between sink and worker"""

    @abstractmethod
    async def interrupt_playback(self, stream_id: str) -> None:
        """Signal worker to interrupt current playback"""
        pass

    @abstractmethod
    async def notify_playback_finished(
        self, stream_id: str, playback_position: float, interrupted: bool
    ) -> None:
        """Signal sink that playback has finished"""
        pass


class RPCProtocol(AvatarProtocol):
    """LiveKit RPC implementation of the avatar protocol"""

    RPC_INTERRUPT = "avatar.interrupt_playback"
    RPC_PLAYBACK_FINISHED = "avatar.playback_finished"

    def __init__(self, local_participant: rtc.LocalParticipant, worker_identity: str):
        self._local = local_participant
        self._worker_identity = worker_identity

        # Register RPC methods
        self._local.register_rpc_method(
            self.RPC_PLAYBACK_FINISHED, self._handle_playback_finished
        )

    async def interrupt_playback(self, stream_id: str) -> None:
        await self._local.perform_rpc(
            destination_identity=self._worker_identity,
            method=self.RPC_INTERRUPT,
            payload=stream_id,
        )

    async def notify_playback_finished(
        self, stream_id: str, interrupted: bool
    ) -> None:
        event = PlaybackFinishedEvent(
            stream_id=stream_id,
            interrupted=interrupted,
        )
        await self._local.perform_rpc(
            destination_identity=self._worker_identity,
            method=self.RPC_PLAYBACK_FINISHED,
            payload=event.model_dump_json(),
        )

    async def _handle_playback_finished(self, data: rtc.RpcInvocationData) -> None:
        """Handle playback finished RPC from worker"""
        event = PlaybackFinishedEvent.model_validate_json(data.payload)
        # Notify sink about playback finished
        if self._on_playback_finished:
            await self._on_playback_finished(event)
