from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Coroutine, Mapping
from dataclasses import dataclass, field
from datetime import timedelta
from typing import TYPE_CHECKING, Generic, Literal, Never

from typing_extensions import TypeVar

from livekit import api, rtc

from .. import _core, http, utils
from .._bindings import APIError
from .._generated import public_voice_avatar as _public
from .._generated.avatar_queue import QueueEffects
from .._generated.io_events import AvatarPlaybackSegment
from .._generated.public_voice_avatar import *  # noqa: F403
from ..job import _server_credentials, get_job_context
from ..log import logger
from ..types import (
    ATTRIBUTE_PUBLISH_ON_BEHALF,
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from .io import AudioOutput, AudioOutputCapabilities

if TYPE_CHECKING:
    from .agent_session import VoiceSession

# Agent <-> avatar worker protocol over DataStream: one byte stream per segment on
# `lk.audio_stream`, `lk.clear_buffer` from the agent, `lk.playback_started` and
# `lk.playback_finished` back. Both ends and the runner live in the core.

__all__ = [
    "AvatarRunner",
    "AvatarSession",
    "AvatarStart",
    "ForwardingAvatarSession",
    "VideoGenerator",
    "AudioReceiver",
    "QueueAudioOutput",
    "DataStreamAudioReceiver",
]
__all__ += _public.__all__


_ReceiverEventT = TypeVar("_ReceiverEventT", default=Never, contravariant=True)


class AudioReceiver(
    ABC, rtc.EventEmitter[Literal["clear_buffer"] | _ReceiverEventT], Generic[_ReceiverEventT]
):
    """Audio the avatar worker receives from the agent, fed by ``AvatarRunner`` to its generator.

    A subclass yields frames, then ``AudioSegmentEnd`` at each segment's end. Emitting
    ``"clear_buffer"`` cuts the current segment. ``notify_playback_started`` and
    ``notify_playback_finished`` receive the segment's key.
    """

    async def start(self) -> None:
        pass

    @abstractmethod
    def notify_playback_finished(
        self, playback_position: float, interrupted: bool, *, segment: AvatarPlaybackSegment
    ) -> None | Coroutine[None, None, None]:
        """Notify the sender that playback has finished"""

    @abstractmethod
    def notify_playback_started(
        self, *, segment: AvatarPlaybackSegment
    ) -> None | Coroutine[None, None, None]:
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


TEvent = TypeVar("TEvent", default=Never, contravariant=True)


@dataclass(frozen=True)
class AvatarStart:
    """What ``create_session`` builds the provider's request from."""

    room: rtc.Room
    agent_session: VoiceSession
    agent_identity: str
    """The agent participant the avatar publishes on behalf of."""
    livekit_url: str
    """The URL the avatar worker joins the room through."""
    conn_options: APIConnectOptions
    avatar_identity: str
    avatar_name: str
    _api_key: str = field(repr=False)
    _api_secret: str = field(repr=False)

    def livekit_token(
        self,
        *,
        ttl: timedelta = timedelta(hours=1),
        attributes: Mapping[str, str] | None = None,
        grants: api.VideoGrants | None = None,
    ) -> str:
        """A token for the avatar participant: kind agent, its identity and name, joining this
        room and publishing on behalf of ``agent_identity``. ``grants`` replaces the default
        room join; ``attributes`` are added to the publish-on-behalf one."""
        return (
            api.AccessToken(api_key=self._api_key, api_secret=self._api_secret)
            .with_kind("agent")
            .with_identity(self.avatar_identity)
            .with_name(self.avatar_name)
            .with_ttl(ttl)
            .with_grants(grants or api.VideoGrants(room_join=True, room=self.room.name))
            .with_attributes(
                {ATTRIBUTE_PUBLISH_ON_BEHALF: self.agent_identity, **(attributes or {})}
            )
            .to_jwt()
        )


class AvatarSession(ABC, rtc.EventEmitter[Literal["metrics_collected"] | TEvent], Generic[TEvent]):
    """Base class for avatar plugin sessions.

    A plugin writes ``create_session``: one attempt at its provider's session-create request(s)
    and the parsing of the answer, returning the provider's session id; optionally
    ``end_session`` and ``audio_output``. ``start()`` is the framework's:

    - The LiveKit credentials resolve from the constructor, else the server settings.
    - The avatar participant (``avatar_identity``) is awaited in the background until it joins
      and publishes its video; ``wait_for_join()`` awaits it.
    - ``create_session`` runs under ``conn_options`` through ``http.retry_request``: a 5xx, 429
      or connection failure is retried after at least the provider's ``Retry-After``, nothing is
      resent once a read timed out. A failure that is not an ``APIError`` becomes a final ``protocol`` one.
    - The agent's audio goes to ``audio_output``: DataStream to the avatar worker by default.
    - Inside a job, the job's shutdown runs ``aclose()``, which runs ``end_session`` once.
    - ``metrics_collected`` emits ``AvatarMetrics`` once when the avatar joins, then for each
      assistant message with a ``playback_latency``, until ``aclose()``.
    """

    sample_rate: int = 24000
    """The rate of the audio the default output sends the avatar worker."""

    def __init__(
        self,
        *,
        avatar_participant_identity: NotGivenOr[str] = NOT_GIVEN,
        avatar_participant_name: NotGivenOr[str] = NOT_GIVEN,
        livekit_url: NotGivenOr[str] = NOT_GIVEN,
        livekit_api_key: NotGivenOr[str] = NOT_GIVEN,
        livekit_api_secret: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        """
        Args:
            avatar_participant_identity: The avatar's room identity; ``"<provider>-avatar-agent"``
                by default.
            avatar_participant_name: Its display name; its identity by default.
            livekit_url: The URL the avatar worker joins through; ``LIVEKIT_URL`` by default.
            livekit_api_key: Signs the avatar's token; ``LIVEKIT_API_KEY`` by default.
            livekit_api_secret: Signs the avatar's token; ``LIVEKIT_API_SECRET`` by default.
            conn_options: Retries and phase limits of the session-create request.
        """
        super().__init__()
        self._identity = avatar_participant_identity or None
        self._name = avatar_participant_name or None
        self._credentials = (livekit_url, livekit_api_key, livekit_api_secret)
        self._conn_options = conn_options
        self._created: str | None = None
        self._tasks = utils.aio.TaskSet()
        self._ended: asyncio.Task[None] | None = None
        self._lifecycle: _core._core_AvatarLifecycle | None = None
        self._room: rtc.Room | None = None
        self._agent_session: VoiceSession | None = None
        self._audio_output: DataStreamAudioOutput | QueueAudioOutput | None = None

    @property
    def avatar_identity(self) -> str:
        """The participant identifier of the avatar"""
        return self._identity or f"{self.provider}-avatar-agent"

    @property
    def provider(self) -> str:
        """The provider of the avatar"""
        return "unknown"

    async def create_session(self, ctx: AvatarStart) -> str | None:
        """One attempt at creating the provider's session: its request(s) and the parsing of the
        answer. Returns the provider's session id. Raise ``APIError`` (by its kind) for a
        failure the provider reported; any other exception is final."""
        raise NotImplementedError

    async def end_session(self, session_id: str) -> None:
        """One attempt at ending the provider's session, run once by ``aclose()``."""

    def audio_output(self, ctx: AvatarStart) -> DataStreamAudioOutput | QueueAudioOutput:
        """Where the agent's audio goes: DataStream to the avatar worker at ``sample_rate``,
        ready once it publishes its video."""
        return DataStreamAudioOutput(
            room=ctx.room,
            destination_identity=self.avatar_identity,
            sample_rate=self.sample_rate,
            wait_remote_track=rtc.TrackKind.KIND_VIDEO,
        )

    async def start(self, agent_session: VoiceSession, room: rtc.Room) -> None:
        """Create the provider's session and send it the agent's audio. Raises ``APIError``
        when the session could not be created; a second call while one runs raises
        ``RuntimeError``."""
        if self._lifecycle is not None:
            raise RuntimeError("the avatar is already started")
        self._room = room
        self._agent_session = agent_session
        self._lifecycle = _core._core_AvatarLifecycle(
            agent_session._native,
            room,
            identity=self.avatar_identity,
            provider=self.provider,
            avatar=self,
            job=get_job_context(required=False),
        )
        if type(self).create_session is AvatarSession.create_session:
            # a plugin that still runs its own start: the lifecycle is all it takes from here
            return
        try:
            ctx = self._start_context(agent_session, room)
            self._created = await http.retry_request(
                lambda: self._attempt(ctx), conn_options=self._conn_options
            )
            self._attach_audio_output(agent_session, self.audio_output(ctx))
        except BaseException:
            await self.aclose()
            raise

    def _start_context(self, agent_session: VoiceSession, room: rtc.Room) -> AvatarStart:
        url, key, secret = _server_credentials(*self._credentials)
        if not url or not key or not secret:
            raise ValueError(
                "livekit_url, livekit_api_key and livekit_api_secret must be set by arguments or "
                "environment variables"
            )
        job = get_job_context(required=False)
        if job is not None:
            agent_identity = job.local_participant_identity
        elif room.isconnected():
            agent_identity = room.local_participant.identity
        else:
            raise RuntimeError("the avatar starts in a job or with a connected room")
        return AvatarStart(
            room=room,
            agent_session=agent_session,
            agent_identity=agent_identity,
            livekit_url=url,
            conn_options=self._conn_options,
            avatar_identity=self.avatar_identity,
            avatar_name=self._name or self.avatar_identity,
            _api_key=key,
            _api_secret=secret,
        )

    async def _attempt(self, ctx: AvatarStart) -> str | None:
        try:
            return await self.create_session(ctx)
        except APIError:
            raise
        except Exception as e:
            # the answer broke the provider's contract: final, since a retry could create a
            # second session
            raise APIError(
                f"{self.provider} avatar session failed to start", kind="protocol"
            ) from e

    def _attach_audio_output(
        self, agent_session: VoiceSession, output: DataStreamAudioOutput | QueueAudioOutput
    ) -> None:
        if self._audio_output is not None and self._audio_output is not output:
            raise RuntimeError("the avatar already owns an audio output")
        self._audio_output = output
        if self._lifecycle is not None and isinstance(output, DataStreamAudioOutput):
            # past the join timeout, its audio fails at once instead of waiting for the avatar
            self._lifecycle.attach_output(output)
        agent_session.output.replace_audio_tail(output)

    async def wait_for_join(self, *, timeout: float | None = 30.0) -> None:
        """Wait until the avatar participant has joined the room and published its video track.

        Returns at once before ``start()`` or after ``aclose()``.

        Raises:
            TimeoutError: Not joined within ``timeout`` seconds (``None`` waits indefinitely);
                the background wait continues.
            asyncio.CancelledError: The avatar was closed before it joined.
        """
        if self._lifecycle is None:
            return
        await self._lifecycle.wait_for_join(timeout)

    async def aclose(self) -> None:
        """End the provider's session, close the avatar's audio output and lifecycle; inside a
        job, also remove the avatar participant from the room. Idempotent: concurrent closes
        share one."""
        try:
            if self._created is not None and self._ended is None:
                self._ended = self._tasks.spawn(self._end(self._created), name="end_session")
            if self._ended is not None:
                await asyncio.shield(self._ended)
            output = self._audio_output
            if output is not None:
                await output.aclose()
                if self._audio_output is output:
                    self._audio_output = None
        finally:
            lifecycle = self._lifecycle
            if lifecycle is not None:
                # a concurrent close waits for the same one
                await lifecycle.aclose()
            if self._lifecycle is lifecycle:
                self._lifecycle = None
                self._agent_session = None
                self._room = None

    async def _end(self, session_id: str) -> None:
        limit = self._conn_options.read_timeout or self._conn_options.connect_timeout or 10.0
        try:
            async with asyncio.timeout(limit):
                await self.end_session(session_id)
        except Exception as e:
            # the close completes whatever the provider answered
            logger.warning(
                "ending the avatar session failed",
                extra={"provider": self.provider, "session_id": session_id, "error": repr(e)},
            )


class ForwardingAvatarSession(AvatarSession[TEvent]):
    """An avatar whose audio the agent sends to the provider itself (an SDK, its own socket).

    The plugin writes ``forward`` (each frame, then ``AudioSegmentEnd`` at a segment's end) and
    ``clear`` (the agent was interrupted), and reports its provider's playback with
    ``playback_started`` and ``playback_finished``. The framework owns the loop: frames go out
    in order, a clear runs once at a time, and a failure of either is the session's
    ``avatar_error`` and closes the avatar.
    """

    wait_playback_start: bool = True
    """A segment's playback starts when the provider says so, not when its audio is sent."""

    @abstractmethod
    async def forward(self, frame: rtc.AudioFrame | AudioSegmentEnd) -> None:
        """Send one frame of the agent's audio, or the end of its segment."""

    @abstractmethod
    async def clear(self) -> None:
        """Drop the audio not yet played: the agent was interrupted."""

    def playback_started(self) -> None:
        """The provider started playing the current segment."""
        if isinstance(output := self._audio_output, QueueAudioOutput):
            output.notify_playback_started()

    def playback_finished(self, playback_position: float, *, interrupted: bool) -> None:
        """The provider finished (or cut) the current segment at ``playback_position`` seconds."""
        if isinstance(output := self._audio_output, QueueAudioOutput):
            output.notify_playback_finished(
                playback_position=max(0.0, playback_position), interrupted=interrupted
            )

    def audio_output(self, ctx: AvatarStart) -> DataStreamAudioOutput | QueueAudioOutput:
        output = QueueAudioOutput(
            sample_rate=self.sample_rate, wait_playback_start=self.wait_playback_start
        )
        # listening from the start: an interruption before the loop runs is not missed
        cleared = asyncio.Event()
        output.on("clear_buffer", cleared.set)
        self._forwarding = self._tasks.spawn(
            self._forward_loop(output, cleared), name="avatar_forward"
        )
        return output

    async def aclose(self) -> None:
        forwarding: asyncio.Future[None] | None = getattr(self, "_forwarding", None)
        if forwarding is not None and forwarding is not asyncio.current_task():
            await utils.aio.cancel_and_wait(forwarding)
        await super().aclose()

    async def _forward_loop(self, output: QueueAudioOutput, cleared: asyncio.Event) -> None:
        async def clears() -> None:
            while True:
                await cleared.wait()
                cleared.clear()
                await self.clear()

        try:
            async with utils.aio.TaskSet() as tasks:
                await tasks.first_completed(self._forward_frames(output), clears())
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if (lifecycle := self._lifecycle) is not None:
                lifecycle.fail(error)
            await self.aclose()
        finally:
            output.off("clear_buffer", cleared.set)

    async def _forward_frames(self, output: QueueAudioOutput) -> None:
        async for item in output:
            await self.forward(item)


class DataStreamAudioReceiver(AudioReceiver):
    """Receives the agent's audio over LiveKit DataStream.

    Subscribes to ``sender_identity`` if given, else to the first agent participant in the room.
    """

    def __init__(
        self,
        room: rtc.Room,
        *,
        sender_identity: str | None = None,
        frame_size_ms: NotGivenOr[int] = NOT_GIVEN,
        rpc_max_retries: int = 3,
    ):
        super().__init__()
        self._native = _core.DataStreamAudioReceiver(
            room,
            sender_identity=sender_identity,
            frame_size_ms=frame_size_ms if utils.is_given(frame_size_ms) else 100,
            rpc_max_retries=rpc_max_retries,
            on_clear_buffer=self,
        )

    async def start(self) -> None:
        await self._native.start()

    def notify_playback_finished(
        self, playback_position: float, interrupted: bool, *, segment: AvatarPlaybackSegment
    ) -> None:
        self._native.notify_playback_finished(segment, playback_position, interrupted)

    def notify_playback_started(self, *, segment: AvatarPlaybackSegment) -> None:
        self._native.notify_playback_started(segment)

    def __aiter__(self) -> AsyncIterator[rtc.AudioFrame | AudioSegmentEnd]:
        return self

    async def __anext__(self) -> rtc.AudioFrame | AudioSegmentEnd:
        if (item := (await self._native.next_audio()).frame) is None:
            raise StopAsyncIteration
        return item

    async def aclose(self) -> None:
        self._native.aclose()


class QueueAudioOutput(
    AudioOutput[Literal["clear_buffer"]],
    AudioReceiver[Literal["playback_finished", "playback_started", "playback_progressed"]],
):
    """An in-process audio output that is also the avatar's ``AudioReceiver``."""

    def __init__(
        self, *, sample_rate: int | None = None, wait_playback_start: bool = False
    ) -> None:
        super().__init__(
            label="DebugQueueIO",
            next_in_chain=None,
            sample_rate=sample_rate,
            capabilities=AudioOutputCapabilities(pause=False),
        )
        self._native = _core._core_QueueAudio(wait_playback_start)

    def _apply(self, effects: QueueEffects) -> None:
        if effects.flushed:
            self._tracker.note_flush()
        for segment in effects.started:
            self.on_playback_started(segment=segment, created_at=time.time())
        for finished in effects.finished:
            self.on_playback_finished(
                segment=finished.segment,
                playback_position=finished.playback_position,
                interrupted=finished.interrupted,
            )
        if effects.clear:
            self.emit("clear_buffer")
        if effects.reset:
            self._tracker.reset()

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        self._native.check_open()
        await super().capture_frame(frame)
        key = self.playback_segment
        assert key is not None
        self._apply(self._native.capture(frame, key))

    def flush(self) -> None:
        super().flush()
        self._native.flush()

    def clear_buffer(self) -> None:
        self._apply(self._native.clear())

    def notify_playback_finished(
        self,
        playback_position: float,
        interrupted: bool,
        *,
        segment: AvatarPlaybackSegment | None = None,
    ) -> None:
        self._apply(self._native.playback_finished(playback_position, interrupted, segment))

    def notify_playback_started(self, *, segment: AvatarPlaybackSegment | None = None) -> None:
        self._apply(self._native.playback_started(segment))

    def __aiter__(self) -> AsyncIterator[rtc.AudioFrame | AudioSegmentEnd]:
        return self

    async def __anext__(self) -> rtc.AudioFrame | AudioSegmentEnd:
        if (item := (await self._native.next_audio()).frame) is None:
            raise StopAsyncIteration
        return item

    async def aclose(self) -> None:
        self._apply(self._native.close())


class AvatarRunner:
    """Worker that generates synchronized avatar video from received audio."""

    def __init__(
        self,
        room: rtc.Room,
        *,
        audio_recv: AudioReceiver,
        video_gen: VideoGenerator,
        options: AvatarOptions,
        # queue size of the AV synchronizer
        _queue_size_ms: int = 100,
        # defer publishing the tracks until the first frame is pushed
        _lazy_publish: bool = True,
    ) -> None:
        self._audio_recv = audio_recv
        self._host_receiver: _core.HostAudioReceiver | None = None
        receiver: _core.DataStreamAudioReceiver | _core.HostAudioReceiver
        if isinstance(audio_recv, DataStreamAudioReceiver):
            receiver = audio_recv._native
        elif isinstance(audio_recv, QueueAudioOutput):
            # the queue stamps and reports its own clears
            receiver = _core.HostAudioReceiver(audio_recv, audio=audio_recv._native)
        else:
            receiver = self._host_receiver = _core.HostAudioReceiver(audio_recv)
        self._core = _core.AvatarRunner(
            _core._core_create_room_session(room),
            receiver=receiver,
            generator=video_gen,
            options=options,
            queue_size_ms=_queue_size_ms,
            lazy_publish=_lazy_publish,
        )

    @property
    def av_sync(self) -> _core.AVSynchronizer:
        return self._core.av_sync

    async def start(self) -> None:
        await self._audio_recv.start()
        if self._host_receiver is not None:
            self._audio_recv.on("clear_buffer", self._host_receiver.clear_buffer)
        await self._core.start()

    async def wait_for_complete(self) -> None:
        await self._core.wait_for_complete()

    async def aclose(self) -> None:
        try:
            await self._core.aclose()
        finally:
            if self._host_receiver is not None:
                self._audio_recv.off("clear_buffer", self._host_receiver.clear_buffer)
