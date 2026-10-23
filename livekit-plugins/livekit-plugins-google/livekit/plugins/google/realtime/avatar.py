from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import TYPE_CHECKING, Any

import numpy as np

from livekit import rtc
from livekit.agents import utils
from livekit.agents.voice.avatar import AvatarSession as _BaseAvatarSession
from livekit.agents.voice.io import AudioOutput, AudioOutputCapabilities

from ..log import logger
from ._avatar_stream import AVATAR_AUDIO_CHANNELS, AVATAR_AUDIO_SAMPLE_RATE
from .realtime_api import RealtimeModel

if TYPE_CHECKING:
    from livekit.agents.voice.agent_session import AgentSession

AVATAR_VIDEO_FPS = 24
_MAX_IDLE_BACKLOG = 4
_IDLE_LOOP_FRAMES = AVATAR_VIDEO_FPS
_QUIET_VIDEO_FRAMES = 6
_SPEECH_PEAK = 300


class AvatarSession(_BaseAvatarSession[Any]):
    """Publishes the Live Avatar that Gemini Live renders for a `RealtimeModel`.

    With `avatar_config` set, the model streams a lip-synced talking-head video with its
    speech, including idle frames while it listens. This session decodes that stream and
    publishes it from the agent participant as a camera track and a microphone track.

    Example:
        ```python
        model = google.realtime.RealtimeModel(
            model="gemini-3.8-live",
            vertexai=True,
            avatar_config=types.AvatarConfig(avatar_name="Kira"),
        )
        session = AgentSession(llm=model)
        avatar = google.realtime.AvatarSession(model)
        await avatar.start(session, room=ctx.room)
        await session.start(agent=MyAgent(), room=ctx.room)
        ```
    """

    def __init__(self, model: RealtimeModel, *, video_queue_size_ms: int = 100) -> None:
        """
        Args:
            model: The realtime model whose avatar to publish. It must have `avatar_config` set.
            video_queue_size_ms: Size of the video queue of the audio/video synchronizer.
        """
        super().__init__()
        if not utils.is_given(model._opts.avatar_config):
            raise ValueError("the RealtimeModel has no avatar_config")

        self._model = model
        self._video_queue_size_ms = video_queue_size_ms
        self._frame_ch = utils.aio.Chan[tuple[rtc.VideoFrame | rtc.AudioFrame, bool]]()
        self._forward_atask: asyncio.Task[None] | None = None
        self._audio_source: rtc.AudioSource | None = None
        self._video_source: rtc.VideoSource | None = None
        self._av_sync: rtc.AVSynchronizer | None = None
        self._audio_muted = False
        self._output_detached = False
        self._quiet_frames = _QUIET_VIDEO_FRAMES
        self._recent_quiet: deque[rtc.VideoFrame] = deque(maxlen=_QUIET_VIDEO_FRAMES)
        self._idle_frames: deque[rtc.VideoFrame] = deque(maxlen=_IDLE_LOOP_FRAMES)
        self._held_frames: list[rtc.VideoFrame] | None = None
        self._held_index = 0

    @property
    def avatar_identity(self) -> str:
        # the avatar is published by the agent participant itself
        if self._room is None:
            raise RuntimeError("AvatarSession has not been started")
        return self._room.local_participant.identity

    @property
    def provider(self) -> str:
        return "google"

    async def start(self, agent_session: AgentSession, room: rtc.Room) -> None:
        """Attach to the model and route the session's audio output through the avatar.

        Call it before `AgentSession.start()`. The tracks are published when the first video
        frame arrives.

        Args:
            agent_session: The session that runs the model passed to the constructor.
            room: The room to publish the avatar to.
        """
        if self._model._avatar_session not in (None, self):
            raise RuntimeError("another AvatarSession is already publishing this model")

        await super().start(agent_session, room)
        self._model._avatar_session = self
        self._forward_atask = asyncio.create_task(
            self._forward_frames(room), name="gemini-avatar-forward"
        )
        agent_session.output.replace_audio_tail(_AvatarPlayoutOutput(self))

    async def aclose(self) -> None:
        """Detach from the model and stop publishing the avatar."""
        if self._model._avatar_session is self:
            self._model._avatar_session = None
        self._frame_ch.close()
        if self._forward_atask:
            await utils.aio.cancel_and_wait(self._forward_atask)
            self._forward_atask = None
        if self._av_sync:
            await self._av_sync.aclose()
        if self._audio_source:
            await self._audio_source.aclose()
        if self._video_source:
            await self._video_source.aclose()
        # the base class removes avatar_identity from the room, which here is the agent itself
        room, self._room = self._room, None
        if room:
            room.off("connection_state_changed", self._on_connection_state_changed)
        await super().aclose()

    def _push_frame(self, frame: rtc.VideoFrame | rtc.AudioFrame) -> None:
        if self._frame_ch.closed:
            return
        silenced = self._audio_muted or self._output_detached
        if isinstance(frame, rtc.AudioFrame):
            idle = not _is_speech(frame)
            if not idle:
                self._quiet_frames = 0
                self._recent_quiet.clear()
            if silenced:
                frame = rtc.AudioFrame(
                    data=bytes(len(frame.data) * 2),
                    sample_rate=frame.sample_rate,
                    num_channels=frame.num_channels,
                    samples_per_channel=frame.samples_per_channel,
                )
        else:
            self._quiet_frames += 1
            idle = self._quiet_frames > _QUIET_VIDEO_FRAMES
            if idle:
                # a frame joins the idle loop only once quiet frames follow it too, so the
                # first frames of a reply never do
                if len(self._recent_quiet) == _QUIET_VIDEO_FRAMES:
                    self._idle_frames.append(self._recent_quiet[0])
                self._recent_quiet.append(frame)
            if silenced and not idle and self._idle_frames:
                # the server keeps rendering speech it was asked to stop, so show the
                # avatar listening instead of mouthing words nobody can hear
                frame = self._next_held_frame()
            else:
                self._held_frames = None
        self._frame_ch.send_nowait((frame, idle))

    def _next_held_frame(self) -> rtc.VideoFrame:
        if self._held_frames is None:
            self._held_frames = list(self._idle_frames)
            self._held_index = 0
        frames = self._held_frames
        period = max(2 * len(frames) - 2, 1)
        position = self._held_index % period
        self._held_index += 1
        return frames[position if position < len(frames) else period - position]

    def _set_audio_muted(self, muted: bool) -> None:
        self._audio_muted = muted

    def _set_output_detached(self, detached: bool) -> None:
        self._output_detached = detached

    @utils.log_exceptions(logger=logger)
    async def _forward_frames(self, room: rtc.Room) -> None:
        async for frame, idle in self._frame_ch:
            if self._av_sync is None:
                if not isinstance(frame, rtc.VideoFrame):
                    # nothing is published before the first video frame sets the size
                    continue
                await self._publish_tracks(room, frame.width, frame.height)
            elif idle and self._frame_ch.qsize() > _MAX_IDLE_BACKLOG:
                # the synchronizer plays in real time, so a burst (such as the decoder's
                # probe window) would otherwise remain as a permanent delay
                continue
            assert self._av_sync is not None
            await self._av_sync.push(frame)

    async def _publish_tracks(self, room: rtc.Room, width: int, height: int) -> None:
        self._audio_source = rtc.AudioSource(
            sample_rate=AVATAR_AUDIO_SAMPLE_RATE,
            num_channels=AVATAR_AUDIO_CHANNELS,
            queue_size_ms=self._video_queue_size_ms,
        )
        self._video_source = rtc.VideoSource(width=width, height=height)
        self._av_sync = rtc.AVSynchronizer(
            audio_source=self._audio_source,
            video_source=self._video_source,
            video_fps=AVATAR_VIDEO_FPS,
            video_queue_size_ms=self._video_queue_size_ms,
        )

        audio_track = rtc.LocalAudioTrack.create_audio_track("avatar_audio", self._audio_source)
        await room.local_participant.publish_track(
            audio_track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        )
        video_track = rtc.LocalVideoTrack.create_video_track("avatar_video", self._video_source)
        await room.local_participant.publish_track(
            video_track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA)
        )


class _AvatarPlayoutOutput(AudioOutput):
    """Tracks playout of the model's speech for the AgentSession.

    The speech is already part of the avatar stream, which `AvatarSession` publishes as it
    arrives. This output only reports when each segment plays and how far it got, so that
    transcripts, agent state, and interruptions keep working.
    """

    def __init__(self, avatar: AvatarSession) -> None:
        super().__init__(
            label="GeminiLiveAvatar",
            capabilities=AudioOutputCapabilities(pause=False),
            sample_rate=AVATAR_AUDIO_SAMPLE_RATE,
        )
        self._avatar = avatar
        self._started_at: float | None = None
        self._pushed_duration = 0.0
        self._finish_handle: asyncio.TimerHandle | None = None

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await super().capture_frame(frame)
        if self._started_at is None:
            self._avatar._set_audio_muted(False)
            self._started_at = time.time()
            self.on_playback_started(created_at=self._started_at)
        self._pushed_duration += frame.duration

    def flush(self) -> None:
        super().flush()
        if self._started_at is None:
            return
        remaining = self._started_at + self._pushed_duration - time.time()
        self._finish_handle = asyncio.get_running_loop().call_later(
            max(remaining, 0.0), self._finish, self._pushed_duration, False
        )

    def clear_buffer(self) -> None:
        if self._started_at is None:
            return
        position = min(time.time() - self._started_at, self._pushed_duration)
        # the server keeps rendering the reply until it detects the interruption itself,
        # so silence it locally until the next reply plays
        self._avatar._set_audio_muted(True)
        self._finish(position, True)

    def on_attached(self) -> None:
        super().on_attached()
        self._avatar._set_output_detached(False)

    def on_detached(self) -> None:
        super().on_detached()
        self._avatar._set_output_detached(True)

    def _finish(self, position: float, interrupted: bool) -> None:
        if self._finish_handle:
            self._finish_handle.cancel()
            self._finish_handle = None
        self._started_at = None
        self._pushed_duration = 0.0
        self.on_playback_finished(playback_position=position, interrupted=interrupted)


def _is_speech(frame: rtc.AudioFrame) -> bool:
    samples = np.frombuffer(frame.data, dtype=np.int16)
    return samples.size > 0 and int(np.abs(samples.astype(np.int32)).max()) > _SPEECH_PEAK
