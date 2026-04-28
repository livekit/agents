from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Coroutine
from typing import TYPE_CHECKING, Generic, Literal, TypeVar

from livekit import rtc

from ... import utils
from ...job import get_job_context
from ...log import logger
from ...metrics.base import AvatarMetrics, Metadata
from ..events import ConversationItemAddedEvent, MetricsCollectedEvent

if TYPE_CHECKING:
    from ..agent_session import AgentSession


class AudioSegmentEnd:
    pass


class AudioReceiver(ABC, rtc.EventEmitter[Literal["clear_buffer"]]):
    async def start(self) -> None:
        pass

    @abstractmethod
    def notify_playback_finished(
        self, playback_position: float, interrupted: bool
    ) -> None | Coroutine[None, None, None]:
        """Notify the sender that playback has finished"""

    @abstractmethod
    def notify_playback_started(self) -> None | Coroutine[None, None, None]:
        """Notify the sender that playback has started"""

    @abstractmethod
    def __aiter__(self) -> AsyncIterator[rtc.AudioFrame | AudioSegmentEnd]:
        """Continuously stream out audio frames or AudioSegmentEnd when the stream ends"""

    async def aclose(self) -> None:
        pass


class VideoGenerator(ABC):
    @abstractmethod
    async def push_audio(self, frame: rtc.AudioFrame | AudioSegmentEnd) -> None:
        """Push an audio frame to the video generator"""

    @abstractmethod
    def clear_buffer(self) -> None | Coroutine[None, None, None]:
        """Clear the audio buffer, stopping audio playback immediately"""

    @abstractmethod
    def __aiter__(
        self,
    ) -> AsyncIterator[rtc.VideoFrame | rtc.AudioFrame | AudioSegmentEnd]:
        """Continuously stream out video and audio frames, or AudioSegmentEnd when the audio segment ends"""  # noqa: E501


TEvent = TypeVar("TEvent")


class AvatarSession(ABC, rtc.EventEmitter[Literal["metrics_collected"] | TEvent], Generic[TEvent]):
    """Base class for avatar plugin sessions."""

    def __init__(self) -> None:
        super().__init__()
        self._wait_avatar_join_task: asyncio.Task[None] | None = None
        self._room: rtc.Room | None = None
        self._agent_session: AgentSession | None = None

    @property
    @abstractmethod
    def avatar_identity(self) -> str:
        """The participant identifier of the avatar"""
        ...

    @property
    def provider(self) -> str:
        """The provider of the avatar"""
        return "unknown"

    async def start(self, agent_session: AgentSession, room: rtc.Room) -> None:
        job_ctx = get_job_context(required=False)
        if job_ctx is not None:
            job_ctx.add_shutdown_callback(self.aclose)
        else:
            logger.debug(
                "AvatarSession started outside a job context; call aclose() manually to "
                "release resources when the job shuts down"
            )

        if agent_session._started and (audio_output := agent_session.output.audio) is not None:
            logger.warning(
                (
                    "AvatarSession.start() was called after AgentSession.start(); "
                    "the existing audio output may be replaced by the avatar. "
                    "Please start the avatar session before AgentSession.start() to avoid this."
                ),
                extra={"audio_output": audio_output.label},
            )

        self._room = room
        self._agent_session = agent_session
        self._agent_session.on("conversation_item_added", self._on_conversation_item_added)

        if self._room.isconnected():
            self._wait_avatar_join_task = asyncio.create_task(self._wait_avatar_join())
        else:
            self._room.on("connection_state_changed", self._on_connection_state_changed)

    async def aclose(self) -> None:
        if self._agent_session:
            self._agent_session.off("conversation_item_added", self._on_conversation_item_added)
            self._agent_session = None

        if self._room:
            self._room.off("connection_state_changed", self._on_connection_state_changed)
            self._room = None

        if self._wait_avatar_join_task:
            await utils.aio.cancel_and_wait(self._wait_avatar_join_task)
            self._wait_avatar_join_task = None

    async def _wait_avatar_join(self) -> None:
        assert self._room is not None

        started_time = time.time()
        await utils.wait_for_participant(room=self._room, identity=self.avatar_identity)
        await utils.wait_for_track_publication(
            room=self._room, identity=self.avatar_identity, kind=rtc.TrackKind.KIND_VIDEO
        )
        joined_time = time.time()
        self._emit_metrics(
            AvatarMetrics(
                timestamp=joined_time,
                session_started_time=started_time,
                avatar_joined_time=joined_time,
                metadata=Metadata(
                    model_provider=self.provider,
                ),
            )
        )

    def _on_conversation_item_added(self, ev: ConversationItemAddedEvent) -> None:
        if ev.item.type == "message" and ev.item.role == "assistant":
            playback_latency = ev.item.metrics.get("playback_latency")
            if playback_latency is not None:
                self._emit_metrics(
                    AvatarMetrics(
                        timestamp=ev.created_at,
                        playback_latency=playback_latency,
                        metadata=Metadata(
                            model_provider=self.provider,
                        ),
                    )
                )

    def _on_connection_state_changed(self, state: rtc.ConnectionState.ValueType) -> None:
        assert self._room is not None

        if state == rtc.ConnectionState.CONN_CONNECTED and not self._wait_avatar_join_task:
            self._wait_avatar_join_task = asyncio.create_task(self._wait_avatar_join())

    def _emit_metrics(self, metrics: AvatarMetrics) -> None:
        assert self._agent_session is not None

        self.emit("metrics_collected", metrics)
        self._agent_session.emit("metrics_collected", MetricsCollectedEvent(metrics=metrics))
