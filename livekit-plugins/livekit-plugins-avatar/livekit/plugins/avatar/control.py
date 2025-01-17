import json
from abc import ABC, abstractmethod
from dataclasses import asdict

from livekit import rtc
from livekit.agents.pipeline.io import PlaybackFinishedEvent


class AvatarPlaybackControl(ABC):
    """Control interface for avatar audio playback between sink and worker"""

    @abstractmethod
    async def interrupt_playback(self) -> None:
        """Signal worker to interrupt current playback"""
        pass

    @abstractmethod
    async def notify_playback_finished(self, event: PlaybackFinishedEvent) -> None:
        """Signal sink that playback has finished"""
        pass


class RPCPlaybackControl(AvatarPlaybackControl):
    """LiveKit RPC implementation of avatar playback control"""

    RPC_INTERRUPT = "avatar.interrupt_playback"
    RPC_PLAYBACK_FINISHED = "avatar.playback_finished"

    def __init__(self, local_participant: rtc.LocalParticipant, remote_identity: str):
        self._local = local_participant
        self._remote_identity = remote_identity

    async def interrupt_playback(self) -> None:
        await self._local.perform_rpc(
            destination_identity=self._remote_identity,
            method=self.RPC_INTERRUPT,
            payload="",  # No payload needed for interrupt
        )

    async def notify_playback_finished(self, event: PlaybackFinishedEvent) -> None:
        await self._local.perform_rpc(
            destination_identity=self._remote_identity,
            method=self.RPC_PLAYBACK_FINISHED,
            payload=json.dumps(asdict(event)),
        )
