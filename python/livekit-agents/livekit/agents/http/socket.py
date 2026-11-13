"""Provider sockets: the plugin declares its protocol's messages and writes how its input is
encoded and each message read; the framework runs the socket (``_core.SocketSession``)."""

from __future__ import annotations

import asyncio
import collections
import itertools
import json
import types
import typing
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Annotated, Any, ClassVar, Generic, Literal, TypeAlias, TypeVar, Union

import pydantic_core
from pydantic import TypeAdapter, ValidationError

from livekit import rtc

from .. import _core, utils
from .._bindings import APIError
from .._generated.socket import (
    KeepaliveFrame,
    SocketConnection,
    SocketSend,
    SocketTarget,
    WsFrame,
)
from ..log import logger
from ..types import APIConnectOptions

__all__ = [
    "DONE",
    "READY",
    "protocol_error",
    "Done",
    "Encoded",
    "EndOfInput",
    "Keepalive",
    "Multiplexer",
    "Ready",
    "SocketProtocol",
]

ItemT = TypeVar("ItemT")
InboundT = TypeVar("InboundT")
OutboundT = TypeVar("OutboundT")


class EndOfInput:
    """The last item ``encode`` receives: the input ended."""


class Done:
    """``on_message``'s answer once the provider finished: the stream ends."""


DONE = Done()


class Ready:
    """``on_message``'s answer once the provider is ready for input (see ``awaits_ready``)."""


READY = Ready()

Encoded: TypeAlias = Union[  # noqa: UP007
    bytes, str, OutboundT, Iterable[Union[bytes, str, OutboundT]], None  # noqa: UP007
]
"""What ``encode`` returns: frames as ``bytes`` (binary), ``str`` (raw text) or a command the
framework sends as JSON text; ``None`` sends nothing."""


@dataclass(frozen=True)
class Keepalive(Generic[OutboundT]):
    """``frame``, sent once nothing else was sent for ``interval`` seconds."""

    interval: float
    frame: bytes | str | OutboundT


def _frame(value: object) -> WsFrame:
    if isinstance(value, bytes | str):
        return value
    if isinstance(value, dict):
        return json.dumps(value)
    raise TypeError(f"a socket frame is bytes, str or a command dict, not {type(value).__name__}")


def _frames(encoded: object) -> list[WsFrame]:
    if encoded is None:
        return []
    if isinstance(encoded, bytes | str | dict):
        return [_frame(encoded)]
    if isinstance(encoded, Iterable):
        return [_frame(value) for value in encoded]
    return [_frame(encoded)]


class _Decoder:
    """Reads a frame as the declared ``Inbound`` type. A raw ``bytes`` or ``str`` member takes
    its frames as they are; every other member is a message read from JSON, once.

    Lenient where the plugin does not read: a message's undeclared keys are ignored, and a
    message whose tag (the union's discriminator) names no declared type is skipped, since
    providers add event types. Strict where it reads: a declared message missing a required
    field or holding a wrong-typed one fails with ``protocol``.
    """

    def __init__(self, inbound: Any) -> None:
        discriminator: str | None = None
        if typing.get_origin(inbound) is Annotated:
            inbound, *metadata = typing.get_args(inbound)
            discriminator = next(
                (
                    m.discriminator
                    for m in metadata
                    if isinstance(getattr(m, "discriminator", None), str)
                ),
                None,
            )
        union = typing.get_origin(inbound) in (Union, types.UnionType)
        members = typing.get_args(inbound) if union else (inbound,)
        self.raw_bytes = bytes in members
        self.raw_text = str in members
        parsed = [member for member in members if member not in (bytes, str)]
        if discriminator is None:
            discriminator = _common_tag(parsed)
        self._discriminator = discriminator
        # by tag when the union names its discriminator, else one adapter for all of them
        self._by_tag: dict[object, TypeAdapter[Any]] = {}
        self._adapter: TypeAdapter[Any] | None = None
        if discriminator is not None:
            for member in parsed:
                tag = typing.get_type_hints(member)[discriminator]
                adapter: TypeAdapter[Any] = TypeAdapter(member)
                for value in typing.get_args(tag):
                    self._by_tag[value] = adapter
        elif parsed:
            body: Any = parsed[0] if len(parsed) == 1 else Union[tuple(parsed)]  # noqa: UP007
            self._adapter = TypeAdapter(body)
        self._skipped: set[object] = set()

    def decode(self, frame: WsFrame, label: str) -> Any:
        """The message, or ``None`` for one of a type the protocol does not read."""
        if isinstance(frame, bytes) and self.raw_bytes:
            return frame
        if isinstance(frame, str) and self.raw_text:
            return frame
        if self._adapter is None and not self._by_tag:
            raise _protocol_error(label, f"an unexpected {type(frame).__name__} frame")
        try:
            if self._adapter is not None:
                return self._adapter.validate_json(frame)
            value = pydantic_core.from_json(frame)
            tag = value.get(self._discriminator) if isinstance(value, dict) else None
            adapter = self._by_tag.get(tag)
            if adapter is None:
                if tag not in self._skipped:
                    self._skipped.add(tag)
                    logger.debug("%s: skipping a message type it does not read: %r", label, tag)
                return None
            return adapter.validate_python(value)
        except (ValidationError, ValueError) as e:
            raise _protocol_error(label, f"{e}; frame: {_shown(frame)}") from e


def _common_tag(members: list[Any]) -> str | None:
    """The one key every member declares as a ``Literal``: the messages' tag, when the union
    names none."""
    if not members or not all(typing.is_typeddict(member) for member in members):
        return None
    keys = [
        {
            key
            for key, hint in typing.get_type_hints(member).items()
            if typing.get_origin(hint) is Literal
        }
        for member in members
    ]
    common = set.intersection(*keys)
    return common.pop() if len(common) == 1 else None


def _shown(frame: WsFrame) -> str:
    return frame if isinstance(frame, str) else frame.decode("utf-8", errors="replace")


def _protocol_error(label: str, detail: str) -> APIError:
    # the provider broke its contract: a new connection would read the same message
    return APIError(
        f"{label} sent a message that breaks its protocol", kind="protocol", detail=detail
    )


def protocol_error(
    protocol: SocketProtocol[Any, Any, Any] | Multiplexer[Any, Any], detail: str
) -> APIError:
    """The error a decoder or ``on_message`` raises for a message that breaks the protocol
    (``kind="protocol"``, final); ``detail`` is the provider's own text, logged once and never
    shown."""
    label = (
        protocol._socket_label()
        if isinstance(protocol, SocketProtocol)
        else type(protocol).__name__
    )
    return _protocol_error(label, detail)


def _inbound(cls: type, base: type) -> Any | None:
    """The ``Inbound`` type ``cls`` declared on a generic subclass of ``base``."""
    for klass in cls.__mro__:
        for orig in getattr(klass, "__orig_bases__", ()):
            origin = typing.get_origin(orig)
            if not (isinstance(origin, type) and issubclass(origin, base)):
                continue
            params = getattr(origin, "__parameters__", ())
            args = typing.get_args(orig)
            if InboundT in params and len(args) == len(params):
                found = args[params.index(InboundT)]
                if not isinstance(found, TypeVar):
                    return found
    return None


def _job_proxy() -> str | None:
    from ..job import get_job_context

    job = get_job_context(required=False)
    return job.proc.http_proxy if job is not None else None


class SocketProtocol(ABC, Generic[ItemT, InboundT, OutboundT]):
    """A provider's socket protocol. A kind's base (``stt.SocketStream``, ``tts.SocketStream``,
    ``llm.RealtimeSocketSession``) fixes the items; the plugin declares ``Inbound`` (its
    provider's messages: TypedDicts as one union on their tag, plus ``bytes`` for binary frames)
    and ``Outbound`` (its commands, sent as JSON).

    The framework connects through ``connect`` (or a ``multiplexer``), sends each input item as
    ``encode`` frames it, decodes each received frame once and hands it to ``on_message``, sends
    the keepalive, the goodbye and rotates the connection; a failure, from the socket or a hook,
    fails the stream.
    """

    keepalive: Keepalive[OutboundT] | None = None
    goodbye: bytes | str | OutboundT | None = None
    """Sent once the input ended and before a rotation; the provider then finishes and closes."""
    max_session_duration: float | None = None
    """Seconds after which the connection is rotated, at the next input boundary."""
    ping_interval: float | None = 30.0
    """Seconds between websocket pings; a ping unanswered for half of it fails the connection
    as lost. ``None`` sends none."""
    frame_ms: int | None = None
    """Audio items are re-chunked to this many milliseconds before ``encode``."""
    multiplexer: Multiplexer[InboundT, OutboundT] | None = None
    """Set to share one provider socket across streams; ``connect`` then is not called."""

    awaits_ready: ClassVar[bool] = False
    """Each connection opens a provider session that says when it is ready: nothing but the
    ``setup`` is sent until ``on_message`` returns ``READY``, which is due within the connect
    limit."""

    _decoder: ClassVar[_Decoder | None] = None
    _socket_commands_outlive_connection: ClassVar[bool] = True
    """A command taken for a connection that ended before writing it is sent on the next one;
    a kind whose commands belong to their provider session drops it instead."""
    _socket_ends_settled: ClassVar[bool] = False
    """The kind hands ``EndOfInput`` only once the provider owes nothing: the stream then ends
    with no goodbye or drain."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if (inbound := _inbound(cls, SocketProtocol)) is not None:
            cls._decoder = _Decoder(inbound)

    async def connect(self, conn_options: APIConnectOptions) -> SocketTarget:
        """One connection, from the current options: called again at each reconnect, so a signed
        URL or a token is fresh. It may run HTTP requests first (a session-create call)."""
        raise NotImplementedError("a socket protocol implements connect() or sets a multiplexer")

    @abstractmethod
    def encode(self, item: ItemT | EndOfInput) -> Encoded[OutboundT]:
        """The frames that carry an input item; ``Flush`` maps to the provider's own segment-end
        command, ``EndOfInput`` to its end-of-input one (the ``goodbye`` follows)."""

    def decode(self, frame: bytes | str) -> InboundT | None:
        """A received frame as ``Inbound``, or ``None`` for a message the protocol does not
        read (skipped). Override for a binary codec or a framing the declared types do not
        cover; raise ``protocol_error`` for a frame that breaks the protocol."""
        if self._decoder is None:
            raise TypeError(f"{type(self).__name__} declares no Inbound type and no decode()")
        decoded: InboundT | None = self._decoder.decode(frame, self._socket_label())
        return decoded

    def setup(self) -> Encoded[OutboundT]:
        """The frames each connection sends first, ahead of every command and input item: the
        provider session's setup, built after ``connect``."""
        return None

    def command_sent(self, command: bytes | str | OutboundT) -> None:
        """``command`` was written to the connection it was queued for."""

    def send_command(self, command: bytes | str | OutboundT, *, first: bool = False) -> None:
        """Queue a command, sent ahead of the input not yet taken: a provider command, or an
        answer to a protocol-level frame (a ping). ``first`` puts it ahead of the commands
        already queued: state ``connect`` rebuilds for a session the queued ones assume."""
        if first:
            self._socket_commands.appendleft(command)
        else:
            self._socket_commands.append(command)
        self._socket_wake.set()

    def reconnect(self, reason: str) -> None:
        """Rotate the connection now: goodbye, drain, close, connect again; nothing queued is
        lost and no retry is spent."""
        if (session := self._socket_session) is not None:
            session.reconnect(reason)

    @property
    def context_id(self) -> str | None:
        """This stream's context on its multiplexer's socket."""
        return self._socket_context

    # framework

    _socket_session: _core.SocketSession | None = None
    _socket_context: str | None = None

    def _socket_init(self) -> None:
        # what the multiplexer decoded for this stream, so a frame is decoded once
        self._socket_routed: dict[WsFrame, Any] = {}
        self._socket_commands: collections.deque[bytes | str | OutboundT] = collections.deque()
        self._socket_wake = asyncio.Event()
        self._socket_audio: utils.media.AudioByteStream | None = None
        # connections made, and each command taken but not yet written, by its id: the command
        # and the connection it was taken for
        self._socket_connections = 0
        self._socket_ids = itertools.count(1)
        self._socket_taken: dict[int, tuple[bytes | str | OutboundT, int]] = {}
        # input went out that the provider has not answered yet
        self._socket_owed = False

    def _socket_answered(self) -> None:
        """The provider answered everything sent so far: an input that ends now owes nothing."""
        self._socket_owed = False

    def _socket_label(self) -> str:
        return type(self).__name__

    async def _socket_input(self) -> ItemT | EndOfInput:
        """The next input item; ``EndOfInput`` once the input ended."""
        await asyncio.Future()
        raise AssertionError

    def _socket_boundary(self, item: object) -> bool:
        """Whether ``item`` ends a segment."""
        return False

    async def _socket_run(self, conn_options: APIConnectOptions) -> None:
        """Runs the socket to the end of the stream, its failure raised."""
        mux = self.multiplexer
        context = self._socket_context = utils.shortuuid() if mux is not None else None
        session = _core.SocketSession(
            self,
            conn_options,
            self._socket_label(),
            mux=mux._socket_native() if mux is not None else None,
            context_id=context,
        )
        self._socket_session = session
        if mux is not None and context is not None:
            mux._socket_streams[context] = self
        try:
            await session.run()
        finally:
            await session.aclose()
            if mux is not None and context is not None:
                mux._socket_streams.pop(context, None)
            self._socket_routed.clear()

    def _socket_segments(self) -> _core.TtsSegments | None:
        """A TTS stream's segments: its contexts on a multiplexed provider."""
        return None

    async def _next_input(self) -> ItemT | EndOfInput | None:
        """The next input item, or ``None`` once a command was queued first."""
        self._socket_wake.clear()
        if self._socket_commands:
            return None
        async with utils.aio.TaskSet() as tasks:
            item = tasks.spawn(self._socket_input())
            await tasks.first_completed(item, self._socket_wake.wait())
        # taken: never dropped, even when a command arrived with it
        return item.result() if item.done() and not item.cancelled() else None

    def _rechunk(self, frame: rtc.AudioFrame) -> list[rtc.AudioFrame]:
        assert self.frame_ms is not None
        if self._socket_audio is None:
            self._socket_audio = utils.media.AudioByteStream(
                frame.sample_rate,
                frame.num_channels,
                samples_per_channel=frame.sample_rate * self.frame_ms // 1000,
            )
        return list(self._socket_audio.push(frame.data))

    def _flush_audio(self) -> list[rtc.AudioFrame]:
        return list(self._socket_audio.flush()) if self._socket_audio is not None else []

    async def _core_connect(self, conn_options: APIConnectOptions) -> SocketConnection:
        self._socket_connections += 1
        target = await self.connect(conn_options)
        keepalive = self.keepalive
        return SocketConnection(
            target=target,
            keepalive=(
                KeepaliveFrame(interval=keepalive.interval, frame=_frame(keepalive.frame))
                if keepalive is not None
                else None
            ),
            goodbye=_frame(self.goodbye) if self.goodbye is not None else None,
            max_session_duration=self.max_session_duration,
            proxy=_job_proxy(),
            ping_interval=self.ping_interval,
            setup=_frames(self.setup()),
            awaits_ready=self.awaits_ready,
        )

    async def _core_next(self) -> SocketSend:
        while True:
            if self._socket_commands:
                command = self._socket_commands.popleft()
                command_id = next(self._socket_ids)
                self._socket_taken[command_id] = (command, self._socket_connections)
                return SocketSend(
                    frames=_frames(command),
                    boundary=True,
                    end=False,
                    done=False,
                    command=command_id,
                )
            item = await self._next_input()
            if item is None:
                continue
            end = isinstance(item, EndOfInput)
            boundary = end or self._socket_boundary(item)
            frames: list[WsFrame] = []
            if isinstance(item, rtc.AudioFrame) and self.frame_ms is not None:
                for chunk in self._rechunk(item):
                    frames += _frames(self.encode(typing.cast(ItemT, chunk)))
            else:
                if boundary:
                    for chunk in self._flush_audio():
                        frames += _frames(self.encode(typing.cast(ItemT, chunk)))
                frames += _frames(self.encode(item))
            if frames and not end:
                self._socket_owed = True
            if frames or boundary:
                # an input that ends with nothing owed closes at once, without the drain
                done = end and (self._socket_ends_settled or not self._socket_owed)
                return SocketSend(
                    frames=frames, boundary=boundary, end=end, done=done, command=None
                )

    async def _core_on_message(self, frame: WsFrame) -> bool:
        mux = self.multiplexer
        message = self._socket_routed.pop(frame, None) if mux is not None else None
        if message is None:
            message = self.decode(frame)
        if message is None:
            return False
        answer = self.on_message(message)
        if isinstance(answer, Done):
            return True
        if isinstance(answer, Ready):
            if self._socket_session is not None:
                self._socket_session.ready()
            return False
        self._socket_deliver(answer)
        return False

    def _core_retrying(self, error: Exception) -> None:
        pass

    def _core_sent(self, command_id: int) -> None:
        if (taken := self._socket_taken.pop(command_id, None)) is not None:
            self.command_sent(taken[0])

    def _core_returned(self, command_id: int) -> None:
        if (taken := self._socket_taken.pop(command_id, None)) is None:
            return
        command, connection = taken
        if self._socket_commands_outlive_connection or connection == self._socket_connections:
            self._socket_commands.appendleft(command)
            self._socket_wake.set()

    @abstractmethod
    def on_message(self, message: InboundT) -> Any:
        """One received message, decoded. Raising fails the stream: an ``APIError`` by its
        kind, anything else as it is."""

    @abstractmethod
    def _socket_deliver(self, answer: Any) -> None:
        """Hands what ``on_message`` returned to the stream's reader."""


class Multiplexer(ABC, Generic[InboundT, OutboundT]):
    """One provider socket carrying many streams, each its own context. Streams that set it as
    their ``multiplexer`` share its sockets, opened on demand; ``route`` sends each received
    message to its stream."""

    keepalive: Keepalive[OutboundT] | None = None
    max_session_duration: float | None = None
    """Seconds after which a socket takes no new stream; it closes once its streams ended."""
    ping_interval: float | None = 30.0
    """Seconds between websocket pings; a ping unanswered for half of it fails the socket and
    every stream on it as lost. ``None`` sends none."""
    max_contexts: int | None = None
    """Streams one socket carries at most; another socket opens for the next."""

    _decoder: ClassVar[_Decoder | None] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if (inbound := _inbound(cls, Multiplexer)) is not None:
            cls._decoder = _Decoder(inbound)

    @abstractmethod
    async def connect(self, conn_options: APIConnectOptions) -> SocketTarget:
        """One shared socket, from the current options."""

    @abstractmethod
    def route(self, message: InboundT) -> str | None:
        """The context ``message`` belongs to: a stream's ``context_id``, or a segment's id for a
        TTS stream; ``None`` for a socket-level message. Raising fails the socket and every
        stream on it."""

    def open_context(self, context_id: str) -> Encoded[OutboundT]:
        """Frames that open a context, if the provider needs any."""
        return None

    def close_context(self, context_id: str) -> Encoded[OutboundT]:
        """Frames that end a context its stream left before the provider finished it: each of a
        TTS stream's open segments."""
        return None

    def decode(self, frame: bytes | str) -> InboundT | None:
        """A received frame as ``Inbound``, ``None`` to skip it; see ``SocketProtocol.decode``."""
        if self._decoder is None:
            raise TypeError(f"{type(self).__name__} declares no Inbound type and no decode()")
        decoded: InboundT | None = self._decoder.decode(frame, type(self).__name__)
        return decoded

    def prewarm(self) -> None:
        """Open a shared socket now, unless one has room: the first stream finds it open."""
        self._socket_native().prewarm()

    def invalidate(self) -> None:
        """Options changed: idle sockets close now, busy ones once their streams ended."""
        if (native := getattr(self, "_native", None)) is not None:
            native.invalidate()

    async def aclose(self) -> None:
        if (native := getattr(self, "_native", None)) is not None:
            await native.aclose()

    def _socket_native(self) -> _core.SocketMux:
        native: _core.SocketMux | None = getattr(self, "_native", None)
        if native is None:
            native = self._native = _core.SocketMux(self, max_contexts=self.max_contexts)
            self._socket_streams: dict[str, SocketProtocol[Any, InboundT, OutboundT]] = {}
        return native

    def _socket_stream(self, context: str) -> SocketProtocol[Any, InboundT, OutboundT] | None:
        """The stream a context belongs to: its own, or one of a TTS stream's segments."""
        if (stream := self._socket_streams.get(context)) is not None:
            return stream
        for stream in self._socket_streams.values():
            if (segments := stream._socket_segments()) is not None and segments.holds(context):
                return stream
        return None

    async def _core_connect(self, conn_options: APIConnectOptions) -> SocketConnection:
        target = await self.connect(conn_options)
        keepalive = self.keepalive
        return SocketConnection(
            target=target,
            keepalive=(
                KeepaliveFrame(interval=keepalive.interval, frame=_frame(keepalive.frame))
                if keepalive is not None
                else None
            ),
            goodbye=None,
            max_session_duration=self.max_session_duration,
            proxy=_job_proxy(),
            ping_interval=self.ping_interval,
            setup=[],
            awaits_ready=False,
        )

    async def _core_route(self, frame: WsFrame) -> str | None:
        message = self.decode(frame)
        if message is None or (context := self.route(message)) is None:
            return None
        # a frame no stream reads (a late one of a closed context) is dropped, never kept
        if (stream := self._socket_stream(context)) is None:
            return None
        stream._socket_routed[frame] = message
        return stream.context_id

    async def _core_open_context(self, context_id: str) -> list[WsFrame]:
        return _frames(self.open_context(context_id))

    async def _core_close_context(self, context_id: str) -> list[WsFrame]:
        stream = self._socket_streams.get(context_id)
        segments = stream._socket_segments() if stream is not None else None
        contexts = [context_id] if segments is None else segments.open
        return [frame for c in contexts for frame in _frames(self.close_context(c))]


def _single_retry_owner(conn_options: APIConnectOptions) -> APIConnectOptions:
    """The view of ``conn_options`` a socket gets when its stream's attempts own the retries."""
    return replace(conn_options, max_retry=0)
