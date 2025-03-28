import asyncio
import atexit
import contextlib
import random
from collections.abc import AsyncGenerator, AsyncIterator
from importlib.resources import as_file, files
from typing import NamedTuple, Union, cast

import numpy as np

from livekit import rtc

from ..log import logger
from ..types import NOT_GIVEN, NotGivenOr
from ..utils import is_given, log_exceptions
from ..utils.aio import cancel_and_wait
from ..utils.audio import audio_frames_from_file
from .agent_session import AgentSession
from .events import AgentStateChangedEvent

SoundSource = Union[AsyncIterator[rtc.AudioFrame], str]


class BackgroundSound(NamedTuple):
    sound: SoundSource
    volume: float = 1.0
    probability: float = 1.0


# The queue size is set to 400ms, which determines how much audio Rust will buffer.
# We intentionally keep this small within BackgroundAudio because calling
# AudioSource.clear_queue() would abruptly cut off ambient sounds.
# Instead, we remove the sound from the mixer, and it will get removed 400ms later.
_AUDIO_SOURCE_BUFFER_MS = 400

_resource_stack = contextlib.ExitStack()
atexit.register(_resource_stack.close)


class BackgroundAudio:
    def __init__(
        self,
        *,
        ambient_sound: NotGivenOr[
            Union[SoundSource, BackgroundSound, list[BackgroundSound], None]
        ] = NOT_GIVEN,
        thinking_sound: NotGivenOr[
            Union[SoundSource, BackgroundSound, list[BackgroundSound], None]
        ] = NOT_GIVEN,
    ) -> None:
        """
        Initializes the BackgroundAudio component with optional ambient and thinking sounds.

        This component creates and publishes a continuous audio track to a LiveKit room while managing
        the playback of ambient and agent “thinking” sounds. It supports two types of audio sources:
        - A file path (string) pointing to an audio file, which can be looped.
        - An AsyncIterator that yields rtc.AudioFrame

        If no ambient sound is provided, a default low‑volume office ambience sound
        (from "office-ambience.ogg" in the package resources) is used. Similarly, if no thinking sound is provided,
        a default list of two keyboard typing sounds (from "keyboard-typing.ogg" and "keyboard-typing2.ogg")
        with higher volume and associated selection probabilities is used.

        When a list (or BackgroundSound) is supplied, the component considers each sound’s volume and probability:
        - The probability value determines the chance that a particular sound is selected for playback.
        - A total probability below 1.0 means there is a chance no sound will be selected (resulting in silence).

        Args:
            ambient_sound (NotGivenOr[Union[SoundSource, BackgroundSound, List[BackgroundSound], None]], optional):
                The ambient sound to be played continuously. For file paths, the sound will be looped.
                For AsyncIterator sources, ensure the iterator is infinite or looped. Defaults to a low‑volume
                office ambience sound.
            thinking_sound (NotGivenOr[Union[SoundSource, BackgroundSound, List[BackgroundSound], None]], optional):
                The sound to be played when the associated agent enters a “thinking” state. This can be a single
                sound source or a list of BackgroundSound objects (with volume and probability settings). Defaults
                to a set of keyboard typing sounds.
        """  # noqa: E501
        default_office = files("livekit.agents.resources") / "office-ambience.ogg"
        default_keyboard = files("livekit.agents.resources") / "keyboard-typing.ogg"
        default_keyboard2 = files("livekit.agents.resources") / "keyboard-typing2.ogg"

        office_path = _resource_stack.enter_context(as_file(default_office))
        keyboard_path = _resource_stack.enter_context(as_file(default_keyboard))
        keyboard_path2 = _resource_stack.enter_context(as_file(default_keyboard2))

        self._ambient_sound = (
            ambient_sound
            if is_given(ambient_sound)
            else BackgroundSound(str(office_path), volume=0.2)
        )
        self._thinking_sound = (
            thinking_sound
            if is_given(thinking_sound)
            else [
                BackgroundSound(str(keyboard_path), volume=0.8, probability=0.4),
                BackgroundSound(str(keyboard_path2), volume=0.8, probability=0.6),
            ]
        )

        self._audio_source = rtc.AudioSource(48000, 1, queue_size_ms=_AUDIO_SOURCE_BUFFER_MS)
        self._audio_mixer = rtc.AudioMixer(48000, 1, blocksize=4800, capacity=1)
        self._publication: rtc.LocalTrackPublication | None = None
        self._lock = asyncio.Lock()

        self._republish_task: asyncio.Task | None = None  # republish the task on reconnect
        self._mixer_atask: asyncio.Task | None = None

        self._play_tasks: list[asyncio.Task] = []

        self._ambient_handle: PlayHandle | None = None
        self._thinking_handle: PlayHandle | None = None

    def _select_sound_from_list(
        self, sounds: list[BackgroundSound]
    ) -> Union[BackgroundSound, None]:
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
        self, source: Union[SoundSource, BackgroundSound, list[BackgroundSound], None]
    ) -> Union[tuple[SoundSource, float], None]:
        if source is None:
            return None

        if isinstance(source, list):
            selected = self._select_sound_from_list(cast(list[BackgroundSound], source))
            if selected is None:
                return None
            return selected.sound, selected.volume

        if isinstance(source, BackgroundSound):
            return source.sound, source.volume

        return source, 1.0

    def play(
        self,
        sound: Union[SoundSource, BackgroundSound, list[BackgroundSound]],
        *,
        loop: bool = False,
    ) -> "PlayHandle":
        """
        Plays a sound once or in a loop.

        Args:
            sound (Union[SoundSource, BackgroundSound, List[BackgroundSound]]):
                The sound to play. Can be:
                - A string pointing to a file path
                - An AsyncIterator that yields `rtc.AudioFrame`
                - A BackgroundSound object with volume and probability
                - A list of BackgroundSound objects, where one will be selected based on probability

                If a string is provided and `loop` is True, the sound will be looped.
                If an AsyncIterator is provided, it is played until exhaustion (and cannot be looped
                automatically).
            loop (bool, optional):
                Whether to loop the sound. Only applicable if `sound` is a string or contains strings.
                Defaults to False.

        Returns:
            PlayHandle: An object representing the playback handle. This can be
            awaited or stopped manually.
        """  # noqa: E501
        if not self._mixer_atask:
            raise RuntimeError("BackgroundAudio is not started")

        normalized = self._normalize_sound_source(sound)
        if normalized is None:
            play_handle = PlayHandle()
            play_handle._mark_playout_done()
            return play_handle

        sound_source, volume = normalized

        if loop and isinstance(sound_source, AsyncIterator):
            raise ValueError(
                "Looping sound via AsyncIterator is not supported. Use a string file path or your own 'infinite' AsyncIterator with loop=False"  # noqa: E501
            )

        play_handle = PlayHandle()
        task = asyncio.create_task(self._play_task(play_handle, sound_source, volume, loop))
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

            await self._publish_track()

            self._mixer_atask = asyncio.create_task(self._run_mixer_task())
            self._room.on("reconnected", self._on_reconnected)

            if self._agent_session:
                self._agent_session.on("agent_state_changed", self._agent_state_changed)

            if self._ambient_sound:
                normalized = self._normalize_sound_source(self._ambient_sound)
                if normalized:
                    sound_source, volume = normalized
                    selected_sound = BackgroundSound(sound_source, volume)
                    if isinstance(sound_source, str):
                        self._ambient_handle = self.play(selected_sound, loop=True)
                    else:
                        self._ambient_handle = self.play(selected_sound)

    async def aclose(self) -> None:
        """
        Gracefully closes the background audio system, canceling all ongoing
        playback tasks and unpublishing the audio track.
        """
        async with self._lock:
            if not self._mixer_atask:
                return  # not started

            await cancel_and_wait(*self._play_tasks)

            if self._republish_task:
                await cancel_and_wait(self._republish_task)

            await cancel_and_wait(self._mixer_atask)

            await self._audio_source.aclose()
            await self._audio_mixer.aclose()

            if self._agent_session:
                self._agent_session.off("agent_state_changed", self._agent_state_changed)

            self._room.off("reconnected", self._on_reconnected)

            with contextlib.suppress(Exception):
                if self._publication is not None:
                    await self._room.local_participant.unpublish_track(self._publication.sid)

    def _on_reconnected(self) -> None:
        if self._republish_task:
            self._republish_task.cancel()

        self._publication = None
        self._republish_task = asyncio.create_task(self._republish_track_task())

    def _agent_state_changed(self, ev: AgentStateChangedEvent) -> None:
        if not self._thinking_sound:
            return

        if ev.state == "thinking":
            if self._thinking_handle and not self._thinking_handle.done():
                return

            self._thinking_handle = self.play(self._thinking_sound)

        elif self._thinking_handle:
            self._thinking_handle.stop()

    @log_exceptions(logger=logger)
    async def _play_task(
        self, play_handle: "PlayHandle", sound: SoundSource, volume: float, loop: bool
    ) -> None:
        if isinstance(sound, str):
            if loop:
                sound = _loop_audio_frames(sound)
            else:
                sound = audio_frames_from_file(sound)

        async def _gen_wrapper() -> AsyncGenerator[rtc.AudioFrame, None]:
            async for frame in sound:
                if volume != 1.0:
                    data = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32)
                    data *= 10 ** (np.log10(volume))
                    np.clip(data, -32768, 32767, out=data)
                    yield rtc.AudioFrame(
                        data=data.astype(np.int16).tobytes(),
                        sample_rate=frame.sample_rate,
                        num_channels=frame.num_channels,
                        samples_per_channel=frame.samples_per_channel,
                    )
                else:
                    yield frame

            # TODO(theomonnom): the wait_for_playout() may be innaccurate by 400ms
            play_handle._mark_playout_done()

        gen = _gen_wrapper()
        try:
            self._audio_mixer.add_stream(gen)
            await play_handle.wait_for_playout()  # wait for playout or interruption
        finally:
            if play_handle._stop_fut.done():
                self._audio_mixer.remove_stream(gen)
                await gen.aclose()

            play_handle._mark_playout_done()  # the task could be cancelled

    @log_exceptions(logger=logger)
    async def _run_mixer_task(self) -> None:
        async for frame in self._audio_mixer:
            await self._audio_source.capture_frame(frame)

    async def _publish_track(self) -> None:
        if self._publication is not None:
            return

        track = rtc.LocalAudioTrack.create_audio_track("background_audio", self._audio_source)
        self._publication = await self._room.local_participant.publish_track(
            track, self._track_publish_options or rtc.TrackPublishOptions()
        )

    @log_exceptions(logger=logger)
    async def _republish_track_task(self) -> None:
        # used to republish the track on agent reconnect
        async with self._lock:
            await self._publish_track()


class PlayHandle:
    def __init__(self) -> None:
        self._done_fut = asyncio.Future()
        self._stop_fut = asyncio.Future()

    def done(self) -> bool:
        """
        Returns True if the sound has finished playing.
        """
        return self._done_fut.done()

    def stop(self) -> None:
        """
        Stops the sound from playing.
        """
        if self.done():
            return

        with contextlib.suppress(asyncio.InvalidStateError):
            self._stop_fut.set_result(None)
            self._mark_playout_done()  # TODO(theomonnom): move this to _play_task

    async def wait_for_playout(self) -> None:
        """
        Waits for the sound to finish playing.
        """
        await asyncio.shield(self._done_fut)

    def __await__(self):
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
