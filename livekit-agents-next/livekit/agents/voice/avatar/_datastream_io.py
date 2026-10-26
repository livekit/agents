"""The agent and an avatar worker exchanging audio over LiveKit DataStream: one byte stream per
segment on ``lk.audio_stream``, ``lk.clear_buffer`` from the agent, ``lk.playback_started`` and
``lk.playback_finished`` back. Both ends run in the core (``_core.DataStreamAudioOutput``,
``_core.DataStreamAudioReceiver``)."""

from __future__ import annotations

from collections.abc import AsyncIterator

from livekit import rtc

from ... import _core
from ...types import NOT_GIVEN, NotGivenOr
from ._types import AudioReceiver, AudioSegmentEnd

RPC_CLEAR_BUFFER = "lk.clear_buffer"
RPC_PLAYBACK_FINISHED = "lk.playback_finished"
RPC_PLAYBACK_STARTED = "lk.playback_started"
AUDIO_STREAM_TOPIC = "lk.audio_stream"

# The agent's side: the session's audio sink, streaming to the avatar and paced by its reports.
DataStreamAudioOutput = _core.DataStreamAudioOutput


class DataStreamAudioReceiver(AudioReceiver):
    """
    Audio receiver that receives streamed audio from a sender participant using LiveKit DataStream.
    If the sender_identity is provided, subscribe to the specified participant. If not provided,
    subscribe to the first agent participant in the room.
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
            frame_size_ms=frame_size_ms or 100,
            rpc_max_retries=rpc_max_retries,
            on_clear_buffer=lambda: self.emit("clear_buffer"),
        )

    async def start(self) -> None:
        await self._native.start()

    def notify_playback_finished(self, playback_position: float, interrupted: bool) -> None:
        self._native.notify_playback_finished(playback_position, interrupted)

    def notify_playback_started(self) -> None:
        self._native.notify_playback_started()

    def __aiter__(self) -> AsyncIterator[rtc.AudioFrame | AudioSegmentEnd]:
        return self

    async def __anext__(self) -> rtc.AudioFrame | AudioSegmentEnd:
        frame = await self._native.recv()
        return AudioSegmentEnd() if frame is None else frame

    async def aclose(self) -> None:
        await self._native.aclose()
