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
        """
        Initializes the BackgroundAudio component with optional ambient and thinking sounds.

        This component creates and publishes a continuous audio track to a LiveKit room while managing
        the playback of ambient and agent “thinking” sounds. It supports three types of audio sources:
        - A BuiltinAudioClip enum value, which will use a pre-defined sound from the package resources
        - A file path (string) pointing to an audio file, which can be looped.
        - An AsyncIterator that yields rtc.AudioFrame

        When a list (or AudioConfig) is supplied, the component considers each sound’s volume and probability:
        - The probability value determines the chance that a particular sound is selected for playback.
        - A total probability below 1.0 means there is a chance no sound will be selected (resulting in silence).

        Args:
            ambient_sound (NotGivenOr[Sounds | None], optional):
                The ambient sound to be played continuously. For file paths, the sound will be looped.
                For AsyncIterator sources, ensure the iterator is infinite or looped.

            thinking_sound (NotGivenOr[Sounds | None], optional):
                The sound to be played when the associated agent enters a “thinking” state. This can be a single
                sound source or a list of AudioConfig objects (with volume and probability settings).

        """  # noqa: E501

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
        """
        Plays an audio once or in a loop.

        Args:
            audio (Sounds):
                The audio to play. Can be:
                - A string pointing to a file path
                - An AsyncIterator that yields `rtc.AudioFrame`
                - An AudioConfig object with volume and probability
                - A list of AudioConfig objects, where one will be selected based on probability

                If a string is provided and `loop` is True, the sound will be looped.
                If an AsyncIterator is provided, it is played until exhaustion (and cannot be looped
                automatically).
            loop (bool, optional):
                Whether to loop the audio. Only applicable if `audio` is a string or contains strings.
                Defaults to False.

        Returns:
            PlayHandle: An object representing the playback handle. This can be
            awaited or stopped manually.
        """  # noqa: E501
        return PlayHandle(self._player.play(audio, random.random(), looping=loop))

    async def start(
        self,
        *,
        room: rtc.Room,
        agent_session: NotGivenOr[VoiceSession] = NOT_GIVEN,
        track_publish_options: NotGivenOr[rtc.TrackPublishOptions] = NOT_GIVEN,
    ) -> None:
        """
        Starts the background audio system, publishing the audio track
        and beginning playback of any configured ambient sound.

        If `ambient_sound` is provided (and contains file paths), they will loop
        automatically. If `ambient_sound` contains AsyncIterators, they are assumed
        to be already infinite or looped.

        Args:
            room (rtc.Room):
                The LiveKit Room object where the audio track will be published.
            agent_session (NotGivenOr[VoiceSession], optional):
                The session object used to track the agent's state (e.g., "thinking").
                Required if `thinking_sound` is provided.
            track_publish_options (NotGivenOr[rtc.TrackPublishOptions], optional):
                Options used when publishing the audio track. If not given, defaults will
                be used.
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
        """
        Gracefully closes the background audio system, canceling all ongoing
        playback tasks and unpublishing the audio track.
        """
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
