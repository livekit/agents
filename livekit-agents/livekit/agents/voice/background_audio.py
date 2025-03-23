import asyncio
from collections.abc import AsyncIterable
from livekit import rtc
from .agent_session import AgentSession
from ..types import NotGiven, NOT_GIVEN, NotGivenOr
from ..utils.audio import audio_frames_from_file
from ..utils.aio import cancel_and_wait
from ..utils import log_exceptions
from ..log import logger

from .events import AgentStateChangedEvent


class BackgroundAudio:
    def __init__(
        self,
        room: rtc.Room,
        *,
        ambiant_sound: NotGivenOr[str | None] = NOT_GIVEN,
        thinking_sound: NotGivenOr[str | None] = NOT_GIVEN,
        agent_session: NotGivenOr[AgentSession] = NOT_GIVEN,
        track_publish_options: NotGivenOr[rtc.TrackPublishOptions] = NOT_GIVEN,
    ) -> None:
        self._room = room
        self._agent_session = agent_session or None
        self._ambiant_sound = ambiant_sound or None
        self._thinking_sound = thinking_sound or None
        self._track_publish_options = track_publish_options or None
        self._audio_source = rtc.AudioSource(48000, 1)
        self._audio_mixer = rtc.AudioMixer(48000, 1, blocksize=4800, capacity=1)
        self._publication: rtc.LocalTrackPublication | None = None
        self._republish_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        self._mixer_task: asyncio.Task | None = None
        self._thinking_task: asyncio.Task | None = None

    async def _publish_track(self) -> None:
        async with self._lock:
            track = rtc.LocalAudioTrack.create_audio_track("background_audio", self._audio_source)
            self._publication = await self._room.local_participant.publish_track(
                track, self._track_publish_options or rtc.TrackPublishOptions()
            )

    async def start(self) -> None:
        await self._publish_track()

        if self._ambiant_sound:
            ambient_stream = self._loop_audio_frames(self._ambiant_sound)
            self._audio_mixer.add_stream(ambient_stream)

        self._mixer_task = asyncio.create_task(self._mixer())

        def _on_reconnected() -> None:
            if self._republish_task:
                self._republish_task.cancel()
            self._republish_task = asyncio.create_task(self._publish_track())

        self._room.on("reconnected", _on_reconnected)

        if self._agent_session:
            self._agent_session.on("agent_state_changed", self._agent_state_changed)

    async def _agent_state_changed(self, ev: AgentStateChangedEvent) -> None:
        if ev.state == "thinking" and self._thinking_sound:
            self._audio_mixer.add_stream(audio_frames_from_file(self._thinking_sound))

    @log_exceptions(logger=logger)
    async def _mixer(self) -> None:
        async for frame in self._audio_mixer:
            await self._audio_source.capture_frame(frame)

    async def aclose(self) -> None:
        async with self._lock:
            assert self._mixer_task
            await cancel_and_wait(self._mixer_task)
            await self._audio_source.aclose()
            await self._audio_mixer.aclose()

            try:
                if self._publication:
                    await self._room.local_participant.unpublish_track(self._publication.sid)
            except Exception:
                pass

    async def _loop_audio_frames(self, file_path: str) -> AsyncIterable[rtc.AudioFrame]:
        while True:
            async for frame in audio_frames_from_file(file_path):
                yield frame
