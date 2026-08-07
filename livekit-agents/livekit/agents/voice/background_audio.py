from __future__ import annotations

import asyncio
import atexit
import contextlib
import enum
import random
from collections.abc import AsyncGenerator, AsyncIterator, Generator
from importlib.resources import as_file, files
from typing import Any, NamedTuple

import numpy as np

from livekit import rtc

from ..job import get_job_context
from ..log import logger
from ..types import NOT_GIVEN, NotGivenOr
from ..utils import is_given, log_exceptions
from ..utils.aio import cancel_and_wait
from ..utils.audio import audio_frames_from_file
from .agent_session import AgentSession
from .events import AgentStateChangedEvent

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


def _frame_gain(
    t: int,
    n: int,
    stop_t: int | None,
    fade_in: float,
    fade_out: float,
    sample_rate: int,
    volume: float,
) -> np.ndarray | None:
    """Combined gain to apply to ``n`` samples starting at sample ``t``.

    Returns ``None`` when no modification is needed (volume == 1 and no fade
    is active for this frame); the caller passes the frame through untouched.
    Otherwise returns an ``np.ndarray`` of length ``n`` with the volume scalar
    and the equal-power fade envelope baked in.

    Equal-power (``sin(phase * pi/2)``) keeps a gentle slope at the silent end
    of each ramp where a linear ramp would knee audibly.
    """
    fade_in_n = int(fade_in * sample_rate) if fade_in > 0 else 0
    fade_out_n = int(fade_out * sample_rate) if fade_out > 0 else 0
    needs_fade_in = fade_in_n > 0 and t < fade_in_n
    needs_fade_out = fade_out_n > 0 and stop_t is not None
    if not needs_fade_in and not needs_fade_out and volume == 1.0:
        return None

    gain = np.full(n, volume, dtype=np.float32)
    if needs_fade_in:
        idx = t + np.arange(n)
        phase = np.clip(idx / fade_in_n, 0.0, 1.0)
        gain *= np.sin(phase * (np.pi / 2)).astype(np.float32)
    if stop_t is not None and fade_out_n > 0:
        idx = t + np.arange(n)
        phase = np.clip((idx - stop_t) / fade_out_n, 0.0, 1.0)
        gain *= np.cos(phase * (np.pi / 2)).astype(np.float32)
    return gain


def _apply_gain(frame: rtc.AudioFrame, gain: np.ndarray | None) -> rtc.AudioFrame:
    """Return ``frame`` with ``gain`` applied, or the frame unchanged when
    ``gain`` is ``None`` (single source of truth for the no-op fast path).
    """
    if gain is None:
        return frame

    data = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32)
    if frame.num_channels > 1:
        gain = np.repeat(gain, frame.num_channels)
    data *= gain
    np.clip(data, -32768, 32767, out=data)
    return rtc.AudioFrame(
        data=data.astype(np.int16).tobytes(),
        sample_rate=frame.sample_rate,
        num_channels=frame.num_channels,
        samples_per_channel=frame.samples_per_channel,
    )


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


# The queue size is set to 400ms, which determines how much audio Rust will buffer.
# We intentionally keep this small within BackgroundAudio because calling
# AudioSource.clear_queue() would abruptly cut off ambient sounds.
# Instead, we remove the sound from the mixer, and it will get removed 400ms later.
_AUDIO_SOURCE_BUFFER_MS = 400
_TRACK_NAME = "background_audio"


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

        self._ambient_sound = ambient_sound if is_given(ambient_sound) else None
        self._thinking_sound = thinking_sound if is_given(thinking_sound) else None

        self._audio_source = rtc.AudioSource(48000, 1, queue_size_ms=_AUDIO_SOURCE_BUFFER_MS)
        self._audio_mixer = rtc.AudioMixer(
            48000, 1, blocksize=4800, capacity=1, stream_timeout_ms=stream_timeout_ms
        )
        self.publication: rtc.LocalTrackPublication | None = None
        self._lock = asyncio.Lock()

        self._mixer_atask: asyncio.Task[None] | None = None

        self._play_tasks: list[asyncio.Task[None]] = []

        self._ambient_handle: PlayHandle | None = None
        self._thinking_handle: PlayHandle | None = None

    def _select_sound_from_list(self, sounds: list[AudioConfig]) -> AudioConfig | None:
        """
        Selects a sound from a list of BackgroundSound based on their probabilities.
        Returns None if no sound is selected (when sum of probabilities < 1.0).
        """
        total_probability = sum(sound.probability for sound in sounds)
        if total_probability <= 0:
            return None

        if total_probability < 1.0 and random.random() > total_probability:
            return None

        normalize_factor = 1.0 if total_probability <= 1.0 else total_probability
        r = random.random() * min(total_probability, 1.0)
        cumulative = 0.0

        for sound in sounds:
            if sound.probability <= 0:
                continue

            norm_prob = sound.probability / normalize_factor
            cumulative += norm_prob

            if r <= cumulative:
                return sound

        return sounds[-1]

    def _normalize_sound_source(
        self, source: AudioSource | AudioConfig | list[AudioConfig] | None
    ) -> AudioConfig | None:
        if source is None:
            return None

        if isinstance(source, list):
            source = self._select_sound_from_list(source)
            if source is None:
                return None

        if isinstance(source, AudioConfig):
            # `_replace` returns a new NamedTuple; the caller's instance is untouched.
            return source._replace(source=self._normalize_builtin_audio(source.source))
        return AudioConfig(self._normalize_builtin_audio(source))

    def _normalize_builtin_audio(self, source: AudioSource) -> AsyncIterator[rtc.AudioFrame] | str:
        if isinstance(source, BuiltinAudioClip):
            return source.path()
        else:
            return source

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
        if not self._mixer_atask:
            raise RuntimeError("BackgroundAudio is not started")

        cfg = self._normalize_sound_source(audio)
        if cfg is None:
            play_handle = PlayHandle()
            play_handle._mark_playout_done()
            return play_handle

        if loop and isinstance(cfg.source, AsyncIterator):
            raise ValueError(
                "Looping sound via AsyncIterator is not supported. Use a string file path or your own 'infinite' AsyncIterator with loop=False"  # noqa: E501
            )

        play_handle = PlayHandle(fade_out=cfg.fade_out)
        task = asyncio.create_task(self._play_task(play_handle, cfg, loop))
        task.add_done_callback(lambda _: self._play_tasks.remove(task))
        task.add_done_callback(lambda _: play_handle._mark_playout_done())
        self._play_tasks.append(task)
        return play_handle

    async def start(
        self,
        *,
        room: rtc.Room,
        agent_session: NotGivenOr[AgentSession] = NOT_GIVEN,
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
            agent_session (NotGivenOr[AgentSession], optional):
                The session object used to track the agent's state (e.g., "thinking").
                Required if `thinking_sound` is provided.
            track_publish_options (NotGivenOr[rtc.TrackPublishOptions], optional):
                Options used when publishing the audio track. If not given, defaults will
                be used.
        """
        async with self._lock:
            self._room = room
            self._agent_session = agent_session or None
            self._track_publish_options = track_publish_options or None

            try:
                job_ctx = get_job_context()
                if job_ctx.is_fake_job():
                    logger.warning(
                        "Background audio is not supported in console mode. Audio will not be played."
                    )
            except RuntimeError:
                pass

            await self._publish_track()

            self._mixer_atask = asyncio.create_task(self._run_mixer_task())

            if self._agent_session:
                self._agent_session.on("agent_state_changed", self._agent_state_changed)

            if self._ambient_sound:
                cfg = self._normalize_sound_source(self._ambient_sound)
                if cfg is not None:
                    loop_ambient = isinstance(cfg.source, str)
                    self._ambient_handle = self.play(cfg, loop=loop_ambient)

    async def aclose(self) -> None:
        """
        Gracefully closes the background audio system, canceling all ongoing
        playback tasks and unpublishing the audio track.
        """
        async with self._lock:
            if not self._mixer_atask:
                return  # not started

            await cancel_and_wait(*self._play_tasks)

            await cancel_and_wait(self._mixer_atask)
            self._mixer_atask = None

            await self._audio_mixer.aclose()
            await self._audio_source.aclose()

            if self._agent_session:
                self._agent_session.off("agent_state_changed", self._agent_state_changed)

            with contextlib.suppress(Exception):
                # The cached publication SID may be stale if the SDK
                # republished it during a full reconnect; resolve the current
                # publication by track name before unpublishing.
                current = self._find_publication_by_name(_TRACK_NAME)
                if current is not None:
                    await self._room.local_participant.unpublish_track(current.sid)

    def _find_publication_by_name(self, name: str) -> rtc.LocalTrackPublication | None:
        for pub in self._room.local_participant.track_publications.values():
            if pub.name == name:
                return pub
        return None

    def _agent_state_changed(self, ev: AgentStateChangedEvent) -> None:
        if not self._thinking_sound:
            return

        if ev.new_state == "thinking":
            if self._thinking_handle and not self._thinking_handle.done():
                return

            assert self._thinking_sound is not None
            self._thinking_handle = self.play(self._thinking_sound)

        elif self._thinking_handle:
            self._thinking_handle.stop()

    @log_exceptions(logger=logger)
    async def _play_task(self, play_handle: PlayHandle, cfg: AudioConfig, loop: bool) -> None:
        sound, volume, fade_in, fade_out = cfg.source, cfg.volume, cfg.fade_in, cfg.fade_out

        if isinstance(sound, BuiltinAudioClip):
            sound = sound.path()
        if isinstance(sound, str):
            sound = _loop_audio_frames(sound) if loop else audio_frames_from_file(sound)

        stopped = False

        async def _gen_wrapper() -> AsyncGenerator[rtc.AudioFrame, None]:
            t = 0  # cumulative samples (per channel) emitted so far
            stop_t: int | None = None  # sample index when stop was requested
            try:
                async for frame in sound:
                    if stopped:
                        break
                    if stop_t is None and fade_out > 0 and play_handle._stop_fut.done():
                        stop_t = t

                    n = frame.samples_per_channel
                    gain = _frame_gain(t, n, stop_t, fade_in, fade_out, frame.sample_rate, volume)
                    yield _apply_gain(frame, gain)

                    t += n
                    if stop_t is not None and (t - stop_t) >= int(fade_out * frame.sample_rate):
                        break
            finally:
                # use try/finally because the mixer's asyncio.wait_for can cancel
                # __anext__, which finalizes the generator and skips code after
                # the async for loop
                play_handle._mark_playout_done()

        gen = _gen_wrapper()
        try:
            self._audio_mixer.add_stream(gen)
            await play_handle.wait_for_playout()
        finally:
            self._audio_mixer.remove_stream(gen)
            play_handle._mark_playout_done()

            await asyncio.sleep(0)
            if play_handle._stop_fut.done():
                stopped = True
                with contextlib.suppress(RuntimeError):
                    # ignore error caused by race condition between aclose() and gen.__anext__()
                    await gen.aclose()

    @log_exceptions(logger=logger)
    async def _run_mixer_task(self) -> None:
        async for frame in self._audio_mixer:
            await self._audio_source.capture_frame(frame)

    async def _publish_track(self) -> None:
        if self.publication is not None:
            return

        track = rtc.LocalAudioTrack.create_audio_track(_TRACK_NAME, self._audio_source)
        self.publication = await self._room.local_participant.publish_track(
            track, self._track_publish_options or rtc.TrackPublishOptions()
        )


class PlayHandle:
    def __init__(self, fade_out: float = 0.0) -> None:
        self._done_fut = asyncio.Future[None]()
        self._stop_fut = asyncio.Future[None]()
        self._fade_out = fade_out

    def done(self) -> bool:
        """
        Returns True if the sound has finished playing.
        """
        return self._done_fut.done()

    def stop(self) -> None:
        """
        Stops the sound from playing.

        If the source was started with a ``fade_out`` duration, this
        triggers the fade-out and the handle stays "not done" until the
        generator has finished tailing out. With ``fade_out=0`` (the
        default), playback is cut immediately as before.
        """
        if self.done():
            return

        with contextlib.suppress(asyncio.InvalidStateError):
            self._stop_fut.set_result(None)
            # Mark done immediately only when there's no fade-out to
            # honour; otherwise the play task's wait_for_playout would
            # return before the generator has tailed out, and the
            # finally block would aclose() the generator mid-fade.
            if self._fade_out <= 0:
                self._mark_playout_done()

    async def wait_for_playout(self) -> None:
        """
        Waits for the sound to finish playing.
        """
        await asyncio.shield(self._done_fut)

    def __await__(self) -> Generator[Any, None, PlayHandle]:
        async def _await_impl() -> PlayHandle:
            await self.wait_for_playout()
            return self

        return _await_impl().__await__()

    def _mark_playout_done(self) -> None:
        with contextlib.suppress(asyncio.InvalidStateError):
            self._done_fut.set_result(None)


async def _loop_audio_frames(file_path: str) -> AsyncGenerator[rtc.AudioFrame, None]:
    while True:
        async for frame in audio_frames_from_file(file_path):
            yield frame
