from __future__ import annotations

import asyncio
import json
import math
from collections.abc import AsyncIterator
from dataclasses import asdict
from typing import Any, Callable, Union

from livekit import rtc

from ... import utils
from ...log import logger
from ...types import NOT_GIVEN, NotGivenOr
from ..io import AudioOutput, PlaybackFinishedEvent
from ._types import AudioReceiver, AudioSegmentEnd

RPC_CLEAR_BUFFER = "lk.clear_buffer"
RPC_PLAYBACK_FINISHED = "lk.playback_finished"
AUDIO_STREAM_TOPIC = "lk.audio_stream"


class DataStreamAudioOutput(AudioOutput):
    """
    AudioOutput implementation that streams audio to a remote avatar worker using LiveKit DataStream.
    """  # noqa: E501

    _playback_finished_handlers: dict[str, Callable[[rtc.RpcInvocationData], str]] = {}
    _playback_finished_rpc_registered: bool = False

    def __init__(
        self,
        room: rtc.Room,
        *,
        destination_identity: str,
        sample_rate: int | None = None,
        wait_remote_track: rtc.TrackKind.ValueType | None = None,
    ):
        super().__init__(label="DataStreamIO", next_in_chain=None, sample_rate=sample_rate)
        self._room = room
        self._destination_identity = destination_identity
        self._wait_remote_track = wait_remote_track
        self._stream_writer: rtc.ByteStreamWriter | None = None
        self._pushed_duration: float = 0.0
        self._tasks: set[asyncio.Task[Any]] = set()

        self._room_connected_fut = asyncio.Future[None]()
        self._room.on("connection_state_changed", self._handle_connection_state_changed)
        if self._room.isconnected():
            self._room_connected_fut.set_result(None)
        self._started = False
        self._lock = asyncio.Lock()

        self._start_atask: asyncio.Task | None = None

    @utils.log_exceptions(logger=logger)
    async def _start_task(self) -> None:
        async with self._lock:
            if self._started:
                return

            await self._room_connected_fut

            self._register_playback_finished_rpc(
                self._room,
                caller_identity=self._destination_identity,
                handler=self._handle_playback_finished,
            )
            logger.debug(
                "waiting for the remote participant",
                extra={"identity": self._destination_identity},
            )
            await utils.wait_for_participant(room=self._room, identity=self._destination_identity)
            if self._wait_remote_track:
                logger.debug(
                    "waiting for the remote track",
                    extra={
                        "identity": self._destination_identity,
                        "kind": rtc.TrackKind.Name(self._wait_remote_track),
                    },
                )
                await utils.wait_for_track_publication(
                    room=self._room,
                    identity=self._destination_identity,
                    kind=self._wait_remote_track,
                )
            logger.debug("remote participant ready", extra={"identity": self._destination_identity})

            self._started = True

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        """Capture and stream audio frame to remote worker"""
        # TODO(theomonnom): this class should be encapsuled somewhere else
        # to allow for a clean close
        if self._start_atask is None:
            self._start_atask = asyncio.create_task(self._start_task())

        # TODO(theomonnom): what to do if start takes a while?
        # we want to avoid OOM & outdated speech?
        await asyncio.shield(self._start_atask)

        await super().capture_frame(frame)

        if not self._stream_writer:
            self._stream_writer = await self._room.local_participant.stream_bytes(
                name=utils.shortuuid("AUDIO_"),
                topic=AUDIO_STREAM_TOPIC,
                destination_identities=[self._destination_identity],
                attributes={
                    "sample_rate": str(frame.sample_rate),
                    "num_channels": str(frame.num_channels),
                },
            )
            self._pushed_duration = 0.0
        await self._stream_writer.write(bytes(frame.data))
        self._pushed_duration += frame.duration

    def flush(self) -> None:
        """Mark end of current audio segment"""
        super().flush()
        if self._stream_writer is None or not self._started:
            return

        # close the stream marking the end of the segment
        task = asyncio.create_task(self._stream_writer.aclose())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

        self._stream_writer = None

    def clear_buffer(self) -> None:
        if not self._started:
            return

        task = asyncio.create_task(
            self._room.local_participant.perform_rpc(
                destination_identity=self._destination_identity,
                method=RPC_CLEAR_BUFFER,
                payload="",
            )
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _handle_playback_finished(self, data: rtc.RpcInvocationData) -> str:
        if data.caller_identity != self._destination_identity:
            logger.warning(
                "playback finished event received from unexpected participant",
                extra={
                    "caller_identity": data.caller_identity,
                    "expected_identity": self._destination_identity,
                },
            )
            return "reject"

        logger.info(
            "playback finished event received",
            extra={"caller_identity": data.caller_identity},
        )

        event = PlaybackFinishedEvent(**json.loads(data.payload))
        self.on_playback_finished(
            playback_position=event.playback_position,
            interrupted=event.interrupted,
        )
        return "ok"

    def _handle_connection_state_changed(self, state: rtc.ConnectionState) -> None:
        if self._room.isconnected() and not self._room_connected_fut.done():
            self._room_connected_fut.set_result(None)

    @classmethod
    def _register_playback_finished_rpc(
        cls,
        room: rtc.Room,
        *,
        caller_identity: str,
        handler: Callable[[rtc.RpcInvocationData], str],
    ) -> None:
        cls._playback_finished_handlers[caller_identity] = handler

        if cls._playback_finished_rpc_registered:
            return

        def _handler(data: rtc.RpcInvocationData) -> str:
            if data.caller_identity not in cls._playback_finished_handlers:
                logger.warning(
                    "playback finished event received from unexpected participant",
                    extra={
                        "caller_identity": data.caller_identity,
                        "expected_identities": list(cls._playback_finished_handlers.keys()),
                    },
                )
                return "reject"
            return cls._playback_finished_handlers[data.caller_identity](data)

        room.local_participant.register_rpc_method(RPC_PLAYBACK_FINISHED, _handler)
        cls._playback_finished_rpc_registered = True


class DataStreamAudioReceiver(AudioReceiver):
    """
    Audio receiver that receives streamed audio from a sender participant using LiveKit DataStream.
    If the sender_identity is provided, subscribe to the specified participant. If not provided,
    subscribe to the first agent participant in the room.
    """

    _clear_buffer_rpc_registered: bool = False
    _clear_buffer_handlers: dict[str, Callable[[rtc.RpcInvocationData], str]] = {}

    def __init__(
        self,
        room: rtc.Room,
        *,
        sender_identity: str | None = None,
        frame_size_ms: NotGivenOr[int] = NOT_GIVEN,
        rpc_max_retries: int = 3,
    ):
        super().__init__()
        self._room = room
        self._sender_identity = sender_identity
        self._remote_participant: rtc.RemoteParticipant | None = None
        self._frame_size_ms = frame_size_ms or 100

        self._stream_readers: list[rtc.ByteStreamReader] = []
        self._stream_reader_changed: asyncio.Event = asyncio.Event()
        self._data_ch = utils.aio.Chan[Union[rtc.AudioFrame, AudioSegmentEnd]]()

        self._current_reader: rtc.ByteStreamReader | None = None
        self._current_reader_cleared: bool = False

        self._playback_finished_ch = utils.aio.Chan[PlaybackFinishedEvent]()
        self._rpc_max_retries = rpc_max_retries

        self._main_atask: asyncio.Task | None = None
        self._exception: Exception | None = None
        self._closing: bool = False

    async def start(self) -> None:
        # wait for the target participant or first agent participant to join
        self._remote_participant = await utils.wait_for_participant(
            room=self._room,
            identity=self._sender_identity,
            kind=rtc.ParticipantKind.PARTICIPANT_KIND_AGENT if not self._sender_identity else None,
        )
        self._main_atask = asyncio.create_task(self._main_task())

        def _handle_clear_buffer(data: rtc.RpcInvocationData) -> str:
            assert self._remote_participant is not None
            if data.caller_identity != self._remote_participant.identity:
                logger.warning(
                    "clear buffer event received from unexpected participant",
                    extra={
                        "caller_identity": data.caller_identity,
                        "expected_identity": self._remote_participant.identity,
                    },
                )
                return "reject"

            if self._current_reader:
                self._current_reader_cleared = True
            self.emit("clear_buffer")
            return "ok"

        def _handle_stream_received(
            reader: rtc.ByteStreamReader, remote_participant_id: str
        ) -> None:
            if (
                not self._remote_participant
                or remote_participant_id != self._remote_participant.identity
            ):
                return

            self._stream_readers.append(reader)
            self._stream_reader_changed.set()

        self._register_clear_buffer_rpc(
            self._room,
            caller_identity=self._remote_participant.identity,
            handler=_handle_clear_buffer,
        )
        self._room.register_byte_stream_handler(AUDIO_STREAM_TOPIC, _handle_stream_received)

    def notify_playback_finished(self, playback_position: float, interrupted: bool) -> None:
        self._playback_finished_ch.send_nowait(
            PlaybackFinishedEvent(playback_position=playback_position, interrupted=interrupted)
        )

    async def _main_task(self) -> None:
        tasks = [
            asyncio.create_task(self._recv_task()),
            asyncio.create_task(self._send_task()),
        ]
        try:
            await asyncio.gather(*tasks)
        except Exception as error:
            self._exception = error
        finally:
            self._playback_finished_ch.close()
            self._data_ch.close()
            await utils.aio.cancel_and_wait(*tasks)

    @utils.log_exceptions(logger=logger)
    async def _send_task(self) -> None:
        async for event in self._playback_finished_ch:
            assert self._remote_participant is not None

            retry_count = 0  # TODO: use retry logic in rust
            while retry_count < self._rpc_max_retries:
                logger.debug(
                    f"notifying playback finished: {event.playback_position:.3f}s, "
                    f"interrupted: {event.interrupted}"
                )
                try:
                    await self._room.local_participant.perform_rpc(
                        destination_identity=self._remote_participant.identity,
                        method=RPC_PLAYBACK_FINISHED,
                        payload=json.dumps(asdict(event)),
                    )
                    break
                except rtc.RpcError as e:
                    if retry_count == self._rpc_max_retries - 1:
                        logger.error(
                            f"failed to notify playback finished after {retry_count + 1} retries",
                            exc_info=e,
                        )
                        raise
                    retry_count += 1
                    logger.warning("failed to notify the agent playback finished, retrying...")
                    await asyncio.sleep(0.1)

    @utils.log_exceptions(logger=logger)
    async def _recv_task(self) -> None:
        while not self._data_ch.closed:
            await self._stream_reader_changed.wait()

            while self._stream_readers:
                self._current_reader = self._stream_readers.pop(0)

                if (
                    not (attrs := self._current_reader.info.attributes)
                    or "sample_rate" not in attrs
                    or "num_channels" not in attrs
                ):
                    raise ValueError("sample_rate or num_channels not found in byte stream")

                sample_rate = int(attrs["sample_rate"])
                num_channels = int(attrs["num_channels"])
                bstream = utils.audio.AudioByteStream(
                    sample_rate=sample_rate,
                    num_channels=num_channels,
                    samples_per_channel=int(math.ceil(sample_rate * self._frame_size_ms / 1000)),
                )

                try:
                    async for data in self._current_reader:
                        if self._current_reader_cleared:
                            # ignore the rest data of the current reader if clear_buffer was called
                            break
                        for frame in bstream.push(data):
                            self._data_ch.send_nowait(frame)

                    if not self._current_reader_cleared:
                        for frame in bstream.flush():
                            self._data_ch.send_nowait(frame)

                    self._current_reader = None
                    self._current_reader_cleared = False
                    self._data_ch.send_nowait(AudioSegmentEnd())

                except utils.aio.ChanClosed:
                    if self._closing:
                        return
                    raise

            self._stream_reader_changed.clear()

    def __aiter__(self) -> AsyncIterator[rtc.AudioFrame | AudioSegmentEnd]:
        return self

    async def __anext__(self) -> rtc.AudioFrame | AudioSegmentEnd:
        try:
            return await self._data_ch.recv()
        except utils.aio.ChanClosed as e:
            if self._exception:
                raise self._exception from e

            raise StopAsyncIteration from None

    async def aclose(self) -> None:
        self._closing = True
        self._playback_finished_ch.close()
        self._data_ch.close()
        self._stream_reader_changed.set()
        if self._main_atask:
            await utils.aio.cancel_and_wait(self._main_atask)

    @classmethod
    def _register_clear_buffer_rpc(
        cls,
        room: rtc.Room,
        *,
        caller_identity: str,
        handler: Callable[[rtc.RpcInvocationData], str],
    ) -> None:
        cls._clear_buffer_handlers[caller_identity] = handler

        if cls._clear_buffer_rpc_registered:
            return

        def _handler(data: rtc.RpcInvocationData) -> str:
            if data.caller_identity not in cls._clear_buffer_handlers:
                logger.warning(
                    "clear buffer event received from unexpected participant",
                    extra={
                        "caller_identity": data.caller_identity,
                        "expected_identities": list(cls._clear_buffer_handlers.keys()),
                    },
                )
                return "reject"
            return cls._clear_buffer_handlers[data.caller_identity](data)

        room.local_participant.register_rpc_method(RPC_CLEAR_BUFFER, _handler)
        cls._clear_buffer_rpc_registered = True
