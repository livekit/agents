from __future__ import annotations

import contextlib
import random
from collections.abc import Generator
from typing import Any

from livekit import rtc

from .. import _core
from .._generated import (
    facades_voice_background_audio as _facades,
    public_voice_background_audio as _public,
)
from .._generated.public_voice_background_audio import *  # noqa: F403
from ..job import get_job_context
from ..log import logger
from ..types import NOT_GIVEN, NotGivenOr
from .agent_session import VoiceSession
from .events import AgentStateChangedEvent

__all__ = ["BackgroundAudioPlayer", "PlayHandle"]
__all__ += _public.__all__


class BackgroundAudioPlayer:
    def __init__(
        self,
        *,
        ambient_sound: NotGivenOr[Sounds | None] = NOT_GIVEN,
        thinking_sound: NotGivenOr[Sounds | None] = NOT_GIVEN,
        stream_timeout_ms: int = 200,
    ) -> None:
        """Publishes an audio track playing ambient and "thinking" sounds.

        A sound is a ``BuiltinAudioClip``, a file path (loopable), an async iterator of
        ``rtc.AudioFrame``, an ``AudioConfig`` or a list of them. In a list, each sound's
        probability is its chance of being picked; a total below 1.0 may pick silence.

        Args:
            ambient_sound: Played continuously; file paths loop, iterators must not end.
            thinking_sound: Played while the agent is in the "thinking" state.
            stream_timeout_ms: How long the mixer waits for a stream's next frame.
        """
        self._ambient = ambient_sound or None
        self._thinking = thinking_sound or None
        self._player = _core.BackgroundAudioPlayer(stream_timeout_ms=stream_timeout_ms)
        self.publication: rtc.LocalTrackPublication | None = None
        self._agent_session: VoiceSession | None = None

    def play(
        self,
        audio: Sounds,
        *,
        loop: bool = False,
    ) -> PlayHandle:
        """Play a sound once or in a loop.

        Args:
            audio: A file path, an async iterator of ``rtc.AudioFrame`` (played until exhausted),
                an ``AudioConfig``, or a list of them picked by probability.
            loop: Loop the audio; only applies to file paths.

        Returns:
            A handle that can be awaited or stopped.
        """
        return PlayHandle(self._player.play(audio, random.random(), looping=loop))

    async def start(
        self,
        *,
        room: rtc.Room,
        agent_session: NotGivenOr[VoiceSession] = NOT_GIVEN,
        track_publish_options: NotGivenOr[rtc.TrackPublishOptions] = NOT_GIVEN,
    ) -> None:
        """Publish the audio track and start the ambient sound.

        Args:
            room: The room to publish the track to.
            agent_session: The session whose agent state drives ``thinking_sound``; required
                for it to play.
            track_publish_options: Options for publishing the track.
        """
        with contextlib.suppress(RuntimeError):
            if get_job_context().is_fake_job():
                logger.warning(
                    "Background audio is not supported in console mode. Audio will not be played."
                )
        options = track_publish_options or rtc.TrackPublishOptions()
        sid = await self._player.start(
            _core._core_create_room_session(room),
            random.random(),
            ambient=self._ambient,
            source=options.source,
        )
        if isinstance(room, rtc.Room):
            self.publication = room.local_participant.track_publications.get(sid)
        if agent_session:
            self._agent_session = agent_session
            agent_session.on("agent_state_changed", self._agent_state_changed)

    async def aclose(self) -> None:
        """Stop all playback and unpublish the track."""
        await self._player.aclose()
        if self._agent_session:
            self._agent_session.off("agent_state_changed", self._agent_state_changed)

    def _agent_state_changed(self, ev: AgentStateChangedEvent) -> None:
        self._player.on_agent_state(self._thinking, random.random(), ev.new_state == "thinking")


class PlayHandle(_facades.PlayHandle):
    def __await__(self) -> Generator[Any, None, PlayHandle]:
        async def _await_impl() -> PlayHandle:
            await self.wait_for_playout()
            return self

        return _await_impl().__await__()
