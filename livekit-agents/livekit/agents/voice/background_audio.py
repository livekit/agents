import asyncio
import atexit
import contextlib
from importlib.resources import files, as_file
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Union

from livekit import rtc

from ..log import logger
from ..types import NOT_GIVEN, NotGivenOr
from ..utils import is_given, log_exceptions
from ..utils.aio import cancel_and_wait
from ..utils.audio import audio_frames_from_file
from .agent_session import AgentSession
from .events import AgentStateChangedEvent

SoundSource = Union[AsyncIterator[rtc.AudioFrame], str]

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
        ambient_sound: NotGivenOr[SoundSource | None] = NOT_GIVEN,
        thinking_sound: NotGivenOr[SoundSource | None] = NOT_GIVEN,
        track_publish_options: NotGivenOr[rtc.TrackPublishOptions] = NOT_GIVEN,
    ) -> None:
        """
        Initializes the background audio system.

        Args:
            ambient_sound (NotGivenOr[SoundSource | None], optional):
                Ambient sound to be played. If a string (file path) is provided, the sound
                can be looped. If an AsyncIterator is provided, it is assumed to be already
                infinite or looped.
            thinking_sound (NotGivenOr[SoundSource | None], optional):
                Sound to be played when the associated agent enters a "thinking" state.
            track_publish_options (NotGivenOr[rtc.TrackPublishOptions], optional):
                Options used when publishing the audio track. If not given, defaults will
                be used.
        """
        default_office = files("livekit.agents.resources") / "office-ambience.ogg"
        default_keyboard = files("livekit.agents.resources") / "keyboard-typing.ogg"
        default_keyboard2 = files("livekit.agents.resources") / "keyboard-typing2.ogg"

        office_path = _resource_stack.enter_context(as_file(default_office))
        keyboard_path = _resource_stack.enter_context(as_file(default_keyboard))

        self._ambient_sound = ambient_sound if is_given(ambient_sound) else str(office_path)
        self._thinking_sound = thinking_sound if is_given(thinking_sound) else str(keyboard_path)
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

    def play(self, sound: SoundSource, *, loop: bool = False) -> "PlayHandle":
        """
        Plays a sound once or in a loop.

        Args:
            sound (SoundSource):
                Either a string pointing to a file path or an AsyncIterator that
                yields `rtc.AudioFrame`. If a string is provided and `loop` is
                True, the sound will be looped. If an AsyncIterator is provided,
                it is played until exhaustion (and cannot be looped automatically).
            loop (bool, optional):
                Whether to loop the sound. Only applicable if `sound` is a string.
                Defaults to False.

        Returns:
            PlayHandle: An object representing the playback handle. This can be
            awaited or stopped manually.
        """
        if loop and isinstance(sound, AsyncIterator):
            raise ValueError(
                "Looping sound via AsyncIterator is not supported. Use a string file path or your own 'infinite' AsyncIterator with loop=False"  # noqa: E501
            )

        play_handle = PlayHandle()
        task = asyncio.create_task(self._play_task(play_handle, sound, loop))
        task.add_done_callback(lambda _: self._play_tasks.remove(task))
        task.add_done_callback(lambda _: play_handle._mark_playout_done())
        self._play_tasks.append(task)
        return play_handle

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
                if isinstance(self._ambient_sound, str):
                    self._ambient_handle = self.play(self._ambient_sound, loop=True)
                elif isinstance(self._ambient_sound, AsyncIterator):
                    # assume the AsyncIterator is already looped
                    self._ambient_handle = self.play(self._ambient_sound)

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

            self._thinking_handle = self.play(self._thinking_sound)
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
