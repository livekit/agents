import asyncio
import atexit
import contextlib
import random
from importlib.resources import files, as_file
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Union, Dict, List, Tuple, Optional, TypeVar, cast

from livekit import rtc

from ..log import logger
from ..types import NOT_GIVEN, NotGivenOr
from ..utils import is_given, log_exceptions
from ..utils.aio import cancel_and_wait
from ..utils.audio import audio_frames_from_file
from .agent_session import AgentSession
from .events import AgentStateChangedEvent

T = TypeVar("T")
SoundSource = Union[AsyncIterator[rtc.AudioFrame], str]
SoundWithWeight = Union[SoundSource, Tuple[SoundSource, float]]
SoundDistribution = Union[SoundSource, List[SoundWithWeight]]

# The queue size is set to 500ms, which determines how much audio Rust will buffer.
# We intentionally keep this small within BackgroundAudio because calling
# AudioSource.clear_queue() would abruptly cut off ambient sounds.
# Instead, we remove the sound from the mixer, and it will get removed 500ms later.
_AUDIO_SOURCE_BUFFER_MS = 500

_resource_stack = contextlib.ExitStack()
atexit.register(_resource_stack.close)


class BackgroundAudio:
    def __init__(
        self,
        *,
        ambient_sound: NotGivenOr[SoundDistribution | None] = NOT_GIVEN,
        thinking_sound: NotGivenOr[SoundDistribution | None] = NOT_GIVEN,
        ambient_volume: float = 1.0,
        thinking_volume: float = 1.0,
        track_publish_options: NotGivenOr[rtc.TrackPublishOptions] = NOT_GIVEN,
    ) -> None:
        """
        Initializes the background audio system.

        Args:
            ambient_sound (NotGivenOr[SoundDistribution | None], optional):
                Ambient sound(s) to be played. You can provide:
                - A single string (file path) that can be looped
                - A single AsyncIterator of audio frames
                - A list of (sound, weight) tuples where weight determines the probability of selection
                - A list of sounds (equal weights will be assigned)
            thinking_sound (NotGivenOr[SoundDistribution | None], optional):
                Sound(s) to be played when the associated agent enters a "thinking" state.
                Accepts the same formats as ambient_sound.
            ambient_volume (float, optional):
                Volume level for ambient sounds (0.0 to 1.0). Defaults to 1.0.
            thinking_volume (float, optional):
                Volume level for thinking sounds (0.0 to 1.0). Defaults to 1.0.
            track_publish_options (NotGivenOr[rtc.TrackPublishOptions], optional):
                Options used when publishing the audio track. If not given, defaults will
                be used.
        """
        default_office = files("livekit.agents.resources") / "office-ambience.ogg"
        default_keyboard = files("livekit.agents.resources") / "keyboard-typing.ogg"
        default_keyboard2 = files("livekit.agents.resources") / "keyboard-typing2.ogg"

        office_path = _resource_stack.enter_context(as_file(default_office))
        keyboard_path = _resource_stack.enter_context(as_file(default_keyboard))
        keyboard2_path = _resource_stack.enter_context(as_file(default_keyboard2))

        self._ambient_sound = ambient_sound if is_given(ambient_sound) else str(office_path)
        self._thinking_sound = (
            thinking_sound
            if is_given(thinking_sound)
            else [(str(keyboard_path), 0.5), (str(keyboard2_path), 0.5)]
        )
        self._ambient_volume = max(0.0, min(1.0, ambient_volume))
        self._thinking_volume = max(0.0, min(1.0, thinking_volume))
        self._track_publish_options = track_publish_options or None

        self._audio_source = rtc.AudioSource(48000, 1, queue_size_ms=_AUDIO_SOURCE_BUFFER_MS)
        self._audio_mixer = rtc.AudioMixer(48000, 1, blocksize=4800, capacity=1)
        self._publication: rtc.LocalTrackPublication | None = None
        self._lock = asyncio.Lock()

        self._republish_task: asyncio.Task | None = None  # republish the task on reconnect
        self._mixer_atask: asyncio.Task | None = None

        self._play_tasks: list[asyncio.Task] = []

        self._ambient_handle: PlayHandle | None = None
        self._thinking_handle: PlayHandle | None = None

        self._room: Optional[rtc.Room] = None
        self._agent_session: Optional[AgentSession] = None

    def set_volume(
        self, ambient_volume: Optional[float] = None, thinking_volume: Optional[float] = None
    ) -> None:
        """
        Sets the volume levels for ambient and thinking sounds.

        Args:
            ambient_volume (Optional[float], optional):
                New volume level for ambient sounds (0.0 to 1.0). If None, keeps current value.
            thinking_volume (Optional[float], optional):
                New volume level for thinking sounds (0.0 to 1.0). If None, keeps current value.
        """
        if ambient_volume is not None:
            self._ambient_volume = max(0.0, min(1.0, ambient_volume))
            # Update currently playing ambient sound if any
            if self._ambient_handle and not self._ambient_handle.done():
                self._ambient_handle.set_volume(self._ambient_volume)

        if thinking_volume is not None:
            self._thinking_volume = max(0.0, min(1.0, thinking_volume))
            # Update currently playing thinking sound if any
            if self._thinking_handle and not self._thinking_handle.done():
                self._thinking_handle.set_volume(self._thinking_volume)

    def play(
        self, sound: SoundDistribution, *, volume: float = 1.0, loop: bool = False
    ) -> "PlayHandle":
        """
        Plays a sound or randomly selects from multiple sounds based on weights.

        Args:
            sound (SoundDistribution):
                Can be:
                - A string pointing to a file path
                - An AsyncIterator that yields `rtc.AudioFrame`
                - A list of (sound, weight) tuples where weight determines selection probability
                - A list of sounds (equal weights will be assigned)

                If a string is provided and `loop` is True, the sound will be looped.
                If an AsyncIterator is provided, it is played until exhaustion
                (and cannot be looped automatically).
            volume (float, optional):
                Volume level for the sound (0.0 to 1.0). Defaults to 1.0.
            loop (bool, optional):
                Whether to loop the sound. Only applicable if `sound` is a string or list of strings.
                Defaults to False.

        Returns:
            PlayHandle: An object representing the playback handle. This can be
            awaited or stopped manually.
        """
        if loop and isinstance(sound, AsyncIterator):
            raise ValueError(
                "Looping sound via AsyncIterator is not supported. Use a string file path or your own 'infinite' AsyncIterator with loop=False"  # noqa: E501
            )

        # Normalize volume
        volume = max(0.0, min(1.0, volume))

        # Select a sound from the distribution if multiple sounds are provided
        selected_sound = self._select_sound(sound)

        play_handle = PlayHandle(volume=volume)
        task = asyncio.create_task(self._play_task(play_handle, selected_sound, loop))
        task.add_done_callback(lambda _: self._play_tasks.remove(task))
        task.add_done_callback(lambda _: play_handle._mark_playout_done())
        self._play_tasks.append(task)
        return play_handle

    def _select_sound(self, sound_distribution: SoundDistribution) -> SoundSource:
        """
        Selects a sound from a distribution according to specified weights.

        If a single sound is provided, it's returned directly.
        If a list of sounds (or sound-weight tuples) is provided, one is randomly selected.
        """
        if isinstance(sound_distribution, (str, AsyncIterator)):
            return sound_distribution

        # Handle list of sounds
        if not sound_distribution:
            raise ValueError("Empty sound distribution provided")

        # Check if we have a list of (sound, weight) tuples or just sounds
        has_weights = isinstance(sound_distribution[0], tuple)

        if has_weights:
            # Extract sounds and weights
            sounds = []
            weights = []
            for item in sound_distribution:
                s, w = cast(Tuple[SoundSource, float], item)
                sounds.append(s)
                weights.append(max(0.0, w))  # Ensure weights are positive

            # Normalize weights if they don't sum to 0
            weight_sum = sum(weights)
            if weight_sum == 0:
                # If all weights are 0, assign equal probabilities
                weights = [1.0 / len(sounds)] * len(sounds)
            else:
                weights = [w / weight_sum for w in weights]

            # Select sound based on weights
            return random.choices(sounds, weights=weights, k=1)[0]
        else:
            # Equal weights for all sounds
            return random.choice(sound_distribution)

    async def start(
        self, *, room: rtc.Room, agent_session: NotGivenOr[AgentSession] = NOT_GIVEN
    ) -> None:
        """
        Starts the background audio system, publishing the audio track
        and beginning playback of any configured ambient sound.

        If `ambient_sound` is provided (and is a file path), it will loop
        automatically. If `ambient_sound` is an AsyncIterator, it is assumed
        to be already infinite or looped.


        Args:
            room (rtc.Room):
                The LiveKit Room object where the audio track will be published.
            agent_session (NotGivenOr[AgentSession], optional):
                The session object used to track the agent's state (e.g., "thinking").
                Required if `thinking_sound` is provided.
        """
        async with self._lock:
            self._room = room
            self._agent_session = agent_session or None

            await self._publish_track()

            if self._ambient_sound:
                selected_sound = self._select_sound(self._ambient_sound)

                if isinstance(selected_sound, str):
                    self._ambient_handle = self.play(
                        selected_sound, volume=self._ambient_volume, loop=True
                    )
                elif isinstance(selected_sound, AsyncIterator):
                    # assume the AsyncIterator is already looped
                    self._ambient_handle = self.play(selected_sound, volume=self._ambient_volume)

            self._mixer_atask = asyncio.create_task(self._run_mixer_task())
            self._room.on("reconnected", self._on_reconnected)

            if self._agent_session:
                self._agent_session.on("agent_state_changed", self._agent_state_changed)

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
            if self._thinking_handle:
                return

            self._thinking_handle = self.play(self._thinking_sound, volume=self._thinking_volume)
        elif self._thinking_handle:
            self._thinking_handle.stop()
            self._thinking_handle = None

    @log_exceptions(logger=logger)
    async def _play_task(self, play_handle: "PlayHandle", sound: SoundSource, loop: bool) -> None:
        if isinstance(sound, str):
            if loop:
                sound = _loop_audio_frames(sound)
            else:
                sound = audio_frames_from_file(sound)

        async def _gen_wrapper() -> AsyncGenerator[rtc.AudioFrame, None]:
            async for frame in sound:
                # Apply volume to the frame
                if play_handle.volume != 1.0:
                    # Apply volume to audio data
                    # We're assuming frame.data is a buffer of audio samples
                    # This creates a copy of the data to avoid modifying the original
                    original_data = frame.data
                    scaled_data = bytearray(len(original_data))

                    # Process audio data as 16-bit signed integers (assuming that's the format)
                    for i in range(0, len(original_data), 2):
                        if i + 1 < len(original_data):
                            # Extract the sample as a 16-bit signed int
                            sample = int.from_bytes(
                                original_data[i : i + 2], byteorder="little", signed=True
                            )
                            # Apply volume scaling
                            scaled_sample = int(sample * play_handle.volume)
                            # Clamp to 16-bit range
                            scaled_sample = max(-32768, min(32767, scaled_sample))
                            # Convert back to bytes
                            scaled_data[i : i + 2] = scaled_sample.to_bytes(
                                2, byteorder="little", signed=True
                            )

                    # Create a new frame with scaled data
                    new_frame = rtc.AudioFrame(
                        samples=frame.samples,
                        sample_rate=frame.sample_rate,
                        channels=frame.channels,
                        data=bytes(scaled_data),
                    )
                    yield new_frame
                else:
                    # No volume change needed
                    yield frame

            # TODO(theomonnom): the wait_for_playout() may be innaccurate by 500ms
            play_handle._mark_playout_done()

        gen = _gen_wrapper()
        try:
            self._audio_mixer.add_stream(gen)
            await play_handle.wait_for_playout()  # wait for playout or interruption
        finally:
            self._audio_mixer.remove_stream(gen)
            play_handle._mark_playout_done()  # the task could be cancelled
            await gen.aclose()

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
    def __init__(self, *, volume: float = 1.0) -> None:
        self._done_fut = asyncio.Future()
        self._stop_fut = asyncio.Future()
        self._volume = volume

    @property
    def volume(self) -> float:
        """Get the current volume setting."""
        return self._volume

    def set_volume(self, volume: float) -> None:
        """
        Set the volume for this playing sound.

        Args:
            volume (float): Volume level between 0.0 and 1.0
        """
        self._volume = max(0.0, min(1.0, volume))

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
