from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Coroutine
from typing import TYPE_CHECKING, Literal

from livekit import rtc

from ...job import get_job_context
from ...log import logger

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


class AvatarSession:
    """Base class for avatar plugin sessions."""

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

    async def aclose(self) -> None:
        pass
