from __future__ import annotations

import asyncio
import atexit
import contextlib
import enum
import random
from collections.abc import AsyncIterator, Generator
from importlib.resources import as_file, files
from typing import Any, NamedTuple

from livekit import rtc

from .. import _core
from .._generated.background_audio_wire import Sound, SoundSourceFile, SoundSourceStream
from ..job import get_job_context
from ..log import logger
from ..types import NOT_GIVEN, NotGivenOr
from .agent_session import VoiceSession
from .events import AgentStateChangedEvent
from .room_io import _SOURCES, Room as _RtcRoom, _name

_resource_stack = contextlib.ExitStack()
atexit.register(_resource_stack.close)


class BuiltinAudioClip(enum.Enum):
    CITY_AMBIENCE = "city-ambience.ogg"
    FOREST_AMBIENCE = "forest-ambience.ogg"
    OFFICE_AMBIENCE = "office-ambience.ogg"
    CROWDED_ROOM = "crowded-room.ogg"
    KEYBOARD_TYPING = "keyboard-typing.ogg"
    KEYBOARD_TYPING2 = "keyboard-typing2.ogg"
    HOLD_MUSIC = "hold_music.ogg"

    def path(self) -> str:
        file_path = files("livekit.agents.resources") / self.value
        return str(_resource_stack.enter_context(as_file(file_path)))


AudioSource = AsyncIterator[rtc.AudioFrame] | str | BuiltinAudioClip


class AudioConfig(NamedTuple):
    """
    Definition for the audio to be played in the background

    Args:
        volume: The volume of the audio (0.0-1.0)
        probability: The probability of the audio being played, when multiple
            AudioConfigs are provided (0.0-1.0)
        fade_in: Duration in seconds to ramp the volume from 0 up to ``volume``
            when playback starts. ``0`` (default) starts at full volume.
        fade_out: Duration in seconds to ramp the volume back down to 0 when
            ``PlayHandle.stop()`` is called. ``0`` (default) cuts immediately,
            preserving the previous behaviour.
    """

    source: AudioSource
    volume: float = 1.0
    probability: float = 1.0
    fade_in: float = 0.0
    fade_out: float = 0.0


class _FrameStream:
    """An ``AsyncIterator[rtc.AudioFrame]`` as the core's ``FrameStream``."""

    def __init__(self, frames: AsyncIterator[rtc.AudioFrame]) -> None:
        self._frames = frames
        self._pending: asyncio.Future[rtc.AudioFrame] | None = None

    async def _core_next(self) -> rtc.AudioFrame | None:
        self._pending = asyncio.ensure_future(self._frames.__anext__())
        try:
            return await self._pending
        except (StopAsyncIteration, asyncio.CancelledError):
            # exhausted, or cancelled by the core's `cancel_pending` as it drops the sound
            return None
        finally:
            self._pending = None

    def _core_cancel_pending(self) -> None:
        if self._pending is not None:
            self._pending.cancel()

    async def _core_close(self) -> None:
        self._core_cancel_pending()


def _sounds(
    source: NotGivenOr[AudioSource | AudioConfig | list[AudioConfig] | None],
) -> tuple[list[Sound], bool]:
    """The core's candidates (none for an unset sound), and whether to draw one by probability
    (a list)."""
    configs = source if isinstance(source, list) else [source] if source else []
    sounds = []
    for cfg in configs:
        if not isinstance(cfg, AudioConfig):
            cfg = AudioConfig(cfg)
        audio = cfg.source.path() if isinstance(cfg.source, BuiltinAudioClip) else cfg.source
        sounds.append(
            Sound(
                source=SoundSourceFile(path=audio)
                if isinstance(audio, str)
                else SoundSourceStream(stream=_FrameStream(audio)),
                volume=cfg.volume,
                probability=cfg.probability,
                fade_in=cfg.fade_in,
                fade_out=cfg.fade_out,
            )
        )
    return sounds, isinstance(source, list)


class BackgroundAudioPlayer:
    def __init__(
        self,
        *,
        ambient_sound: NotGivenOr[AudioSource | AudioConfig | list[AudioConfig] | None] = NOT_GIVEN,
        thinking_sound: NotGivenOr[
            AudioSource | AudioConfig | list[AudioConfig] | None
        ] = NOT_GIVEN,
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
            ambient_sound (NotGivenOr[Union[AudioSource, AudioConfig, List[AudioConfig], None]], optional):
                The ambient sound to be played continuously. For file paths, the sound will be looped.
                For AsyncIterator sources, ensure the iterator is infinite or looped.

            thinking_sound (NotGivenOr[Union[AudioSource, AudioConfig, List[AudioConfig], None]], optional):
                The sound to be played when the associated agent enters a “thinking” state. This can be a single
                sound source or a list of AudioConfig objects (with volume and probability settings).

        """  # noqa: E501

        self._ambient = _sounds(ambient_sound)
        self._thinking = _sounds(thinking_sound)
        self._player = _core.BackgroundAudioPlayer(stream_timeout_ms=stream_timeout_ms)
        self.publication: rtc.LocalTrackPublication | None = None
        self._agent_session: VoiceSession | None = None

    def play(
        self,
        audio: AudioSource | AudioConfig | list[AudioConfig],
        *,
        loop: bool = False,
    ) -> PlayHandle:
        """
        Plays an audio once or in a loop.

        Args:
            audio (Union[AudioSource, AudioConfig, List[AudioConfig]]):
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
        sounds, choose = _sounds(audio)
        return PlayHandle(
            _state=self._player.play(sounds, random.random(), looping=loop, choose=choose)
        )

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
        ambient, choose = self._ambient
        sid = await self._player.start(
            _core._core_create_room_session(_RtcRoom.of(room)),
            random.random(),
            ambient=ambient,
            choose=choose,
            source=_name(_SOURCES, options.source, "unknown"),
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
        sounds, choose = self._thinking
        self._player.on_agent_state(
            sounds, random.random(), ev.new_state == "thinking", choose=choose
        )


class PlayHandle:
    def __init__(self, fade_out: float = 0.0, *, _state: _core.PlayState | None = None) -> None:
        self._state = _state if _state is not None else _core.PlayState(fade_out)

    def done(self) -> bool:
        """
        Returns True if the sound has finished playing.
        """
        return self._state.done

    def stop(self) -> None:
        """
        Stops the sound from playing.

        If the source was started with a ``fade_out`` duration, this
        triggers the fade-out and the handle stays "not done" until the
        generator has finished tailing out. With ``fade_out=0`` (the
        default), playback is cut immediately as before.
        """
        self._state.stop()

    async def wait_for_playout(self) -> None:
        """
        Waits for the sound to finish playing.
        """
        await self._state.wait_for_playout()

    def __await__(self) -> Generator[Any, None, PlayHandle]:
        async def _await_impl() -> PlayHandle:
            await self.wait_for_playout()
            return self

        return _await_impl().__await__()

    def _mark_playout_done(self) -> None:
        self._state.mark_playout_done()
