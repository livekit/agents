"""The provider WebSocket plumbing the kind bases share, run by ``_core.WebSocketSession``:
connection, decoding, control messages, readiness, pacing and request ids. It speaks no kind's
vocabulary: each kind base gives its stream a ``WebSocketHost`` built from its own
``WebSocketKind``, and re-exports the names its plugins write. Plugins never import it."""

from __future__ import annotations

import asyncio
import collections
import json
import types
import typing
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Annotated, Any, ClassVar, Generic, Literal, NamedTuple, TypeVar, Union

import pydantic_core
from pydantic import TypeAdapter, ValidationError

from . import _core, utils
from ._bindings import APIError
from ._core import WebSocketPacing
from ._generated.websocket import (
    WebSocketConnection,
    WebSocketContextsCancel,
    WebSocketInput,
    WebSocketKeepaliveFrame,
    WebSocketMuxOptions,
    WebSocketMuxRoute,
    WebSocketRead,
    WebSocketTarget,
    WsFrame,
)
from .log import logger
from .types import APIConnectOptions

__all__ = [
    "Keepalive",
    "WebSocketMultiplexer",
    "WebSocketPacing",
    "WebSocketTarget",
    "log_skipped",
    "protocol_error",
]

ItemT = TypeVar("ItemT")
InboundT = TypeVar("InboundT")
OutboundT = TypeVar("OutboundT")
# a keepalive only carries its frame: one of a narrower message type fits a protocol's wider one
FrameT_co = TypeVar("FrameT_co", covariant=True)

Signal = Literal["ready", "session_finished"]
Completion = Literal["settled", "session_finished", "peer_closed"]


@dataclass(frozen=True, kw_only=True)
class Keepalive(Generic[FrameT_co]):
    """``frame``, sent once nothing else was sent for ``every`` seconds. ``only_while_unused``:
    a multiplexer's, sent only while no stream holds the socket (a parked socket kept off the
    provider's idle timeout)."""

    every: float
    frame: bytes | str | FrameT_co
    only_while_unused: bool = False


def _keepalive_frame(keepalive: Keepalive[object] | None) -> WebSocketKeepaliveFrame | None:
    if keepalive is None:
        return None
    return WebSocketKeepaliveFrame(
        interval=keepalive.every,
        frame=_frame(keepalive.frame),
        only_while_unused=keepalive.only_while_unused,
    )


def _frame(value: object) -> WsFrame:
    if isinstance(value, bytes | str):
        return value
    if isinstance(value, dict):
        return json.dumps(value)
    raise TypeError(f"a socket frame is bytes, str or a message dict, not {type(value).__name__}")


def frames_of(value: object) -> list[WsFrame]:
    """The frame a hook's value sends (``bytes`` binary, ``str`` text, a message as JSON);
    ``None`` sends none."""
    return [] if value is None else [_frame(value)]


class _Decoder:
    """Reads a frame as the declared ``Inbound`` type. A raw ``bytes`` or ``str`` member takes
    its frames as they are; every other member is a message read from JSON, once.

    Lenient where the plugin does not read: a message's undeclared keys are ignored, and a
    message whose tag (the union's discriminator) names no declared type, or is no scalar, is
    skipped, since providers add event types. Strict where it reads: a declared message missing a
    required field or holding a wrong-typed one fails with ``protocol``.

    A message type with no tag cannot tell a new shape from a malformed one, so every field it
    declares is optional (refused with ``TypeError`` otherwise): an object of a shape it does not
    read decodes with none of its fields, a JSON value that is no object is skipped, and only a
    read field of the wrong type fails.
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
            for member in parsed:
                try:
                    TypeAdapter(member).validate_python({})
                except ValidationError as e:
                    raise TypeError(
                        f"{member!r} has no tag, so every field it declares is NotRequired: {e}"
                    ) from e
            body: Any = parsed[0] if len(parsed) == 1 else Union[tuple(parsed)]  # noqa: UP007
            self._adapter = TypeAdapter(body)

    def decode(self, frame: WsFrame, label: str) -> Any:
        """The message, or ``None`` for one of a type the protocol does not read."""
        if isinstance(frame, bytes) and self.raw_bytes:
            return frame
        if isinstance(frame, str) and self.raw_text:
            return frame
        if self._adapter is None and not self._by_tag:
            raise _protocol_error(label, f"an unexpected {type(frame).__name__} frame")
        try:
            value = pydantic_core.from_json(frame)
            if self._adapter is not None:
                if not isinstance(value, dict):
                    # every untagged type is an object: another frame is no message it reads
                    log_skipped(label, f"a JSON {type(value).__name__}")
                    return None
                return self._adapter.validate_python(value, strict=True)
            tag = value.get(self._discriminator) if isinstance(value, dict) else None
            # a list or an object is no tag (nor hashable): skipped like an unknown one; an absent
            # or null tag is the key of a member whose tag is ``Literal[None]``
            scalar = tag is None or isinstance(tag, str | int | float)
            adapter = self._by_tag.get(tag) if scalar else None
            if adapter is None:
                shown = repr(tag) if scalar else type(tag).__name__
                log_skipped(label, f"{self._discriminator}={shown}")
                return None
            return adapter.validate_python(value, strict=True)
        except (ValidationError, ValueError) as e:
            raise _protocol_error(label, f"{e}; frame: {_shown(frame)}") from e


_logged_skips: set[tuple[str, str]] = set()


def log_skipped(label: str, kind: str) -> None:
    """A received message the protocol does not read (a type the provider added since), skipped:
    logged once per provider ``label`` and ``kind``. For a plugin's own ``decode()``."""
    if (label, kind) in _logged_skips:
        return
    _logged_skips.add((label, kind))
    logger.debug("%s: skipping a message it does not read: %s", label, kind)


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
    protocol: WebSocketProtocol[Any, Any, Any] | WebSocketMultiplexer[Any, Any], detail: str
) -> APIError:
    """The error a decoder or ``on_message`` raises for a message that breaks the protocol
    (``kind="protocol"``, final); ``detail`` is the provider's own text, logged once and never
    shown."""
    label = (
        protocol._websocket.label
        if isinstance(protocol, WebSocketProtocol)
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


_decoders: weakref.WeakKeyDictionary[type, _Decoder] = weakref.WeakKeyDictionary()


def _class_decoder(cls: type, base: type) -> _Decoder:
    """The validator of the ``Inbound`` ``cls`` declared, built when a frame first needs it: a
    protocol that decodes its own frames, or whose multiplexer does, never builds one."""
    if (decoder := _decoders.get(cls)) is None:
        if (inbound := _inbound(cls, base)) is None:
            raise TypeError(f"{cls.__name__} declares no Inbound type and no decode()")
        decoder = _decoders[cls] = _Decoder(inbound)
    return decoder


class WebSocketProtocol(ABC, Generic[ItemT, InboundT, OutboundT]):
    """A provider's WebSocket protocol. A kind's base fixes the items; the plugin declares
    ``Inbound`` (its provider's messages: TypedDicts as one union on their tag, plus ``bytes`` for
    binary frames) and ``Outbound`` (its messages, sent as JSON), sets a few class attributes and
    writes ``connect``, ``encode`` and ``on_message``.

    The framework connects through ``connect`` (or a ``multiplexer``), sends each input item as
    ``encode`` frames it, decodes each received frame once and hands it to ``on_message``, sends
    the keepalive and replaces the connection; a failure, from the socket or a hook, fails the
    stream and carries the provider's ``request_id``.
    """

    keepalive: Keepalive[OutboundT] | None = None
    max_session_duration: float | None = None
    """Seconds after which the connection is replaced, once the stream has nothing in progress a
    new connection would cut."""
    ping_interval: float | None = 30.0
    """Seconds between websocket pings; a ping unanswered for half of it fails the connection
    as lost. ``None`` sends none."""
    multiplexer: WebSocketMultiplexer[InboundT, OutboundT] | None = None
    """Set to share one provider socket across streams; ``connect`` then is not called."""
    pacing: WebSocketPacing | None = None
    """How fast frames go: a minimum interval between two, or no faster than the input plays.
    One ``WebSocketPacing`` set on several streams (one the model holds) paces them together."""
    completion: ClassVar[Completion] = "settled"
    """The documented session completion: settled transcripts, a terminal message reported as
    ``SessionFinished``, or a clean peer close after it flushes its last responses."""

    awaits_ready: ClassVar[bool] = False
    """Each connection opens a provider session that says when it is ready: nothing but the
    ``setup`` goes until ``on_message`` reports it, within the connect limit."""

    _websocket: WebSocketHost
    """The stream's side of the session, which its kind base builds."""

    async def connect(self, conn_options: APIConnectOptions) -> WebSocketTarget:
        """One connection, from the current options: called again at each connection, so a signed
        URL or a token is fresh. It may run HTTP requests first (a session-create call)."""
        raise NotImplementedError("a socket protocol implements connect() or sets a multiplexer")

    @abstractmethod
    def encode(self, item: ItemT) -> bytes | OutboundT | None:
        """The frame that carries an input item, the kind's end item included (the provider's
        goodbye); ``None`` where the item sends nothing."""

    @abstractmethod
    def on_message(self, message: InboundT) -> None:
        """One received message, decoded, reported through the kind's ``report``. Raising fails
        the stream: an ``APIError`` by its kind, anything else as it is."""

    def decode(self, frame: bytes | str) -> InboundT | None:
        """A received frame as ``Inbound``, or ``None`` for a message the protocol does not
        read (skipped). Override for a binary codec or a framing the declared types do not
        cover; raise ``protocol_error`` for a frame that breaks the protocol."""
        decoder = _class_decoder(type(self), WebSocketProtocol)
        decoded: InboundT | None = decoder.decode(frame, self._websocket.label)
        return decoded

    def _skip_unknown_message(self, message: object) -> None:
        # Custom decoders may accept more types than the handler knows. Provider additions
        # must not terminate an otherwise usable stream; do not log their payloads.
        log_skipped(self._websocket.label, "an unhandled message")

    def setup(self) -> bytes | OutboundT | None:
        """The frame each connection sends first, ahead of every other: the provider session's
        setup, built after ``connect``."""
        return None

    def send_now(self, message: bytes | OutboundT, *, first: bool = False) -> None:
        """Send ``message`` on the open connection, ahead of the input not yet sent: a control
        message, or an answer to a protocol-level frame (a ping). One sent while no connection is
        open goes on the next only if sent from its ``connect``: ``connect`` rebuilds the session's
        state. ``first`` puts it ahead of the ones not yet sent."""
        self._websocket.send_now(message, first=first)

    def sent_now(self, message: bytes | OutboundT) -> None:
        """``message``, given to ``send_now``, went to the open connection."""

    def reconnect(self, reason: str) -> None:
        """Replace the connection now: close it and connect again; no retry is spent."""
        self._websocket.reconnect(reason)

    @property
    def context_id(self) -> str | None:
        """This stream's context on its multiplexer's socket."""
        return self._websocket.context_id

    @property
    def request_id(self) -> str | None:
        """The provider's id for what the stream carries now: the last one a message named, else
        its connection's (``WebSocketTarget.request_id_header``)."""
        return self._websocket.request_id


class Batch(NamedTuple):
    """What one input item gives ``encode``, in order, and the seconds it plays for, which
    real-time pacing spaces its frames by."""

    items: list[Any]
    pace: float = 0.0


class WebSocketKind(ABC, Generic[ItemT]):
    """A kind's terms, as the shared layer asks for them: its input, what it owes the provider,
    and when its connection can be replaced. Each kind base gives its stream's
    ``WebSocketHost`` one; the host sets ``host``."""

    host: WebSocketHost

    @property
    @abstractmethod
    def label(self) -> str:
        """The provider's label, in retries, logs and protocol errors."""

    @abstractmethod
    async def take(self) -> Batch | None:
        """The next input item for ``encode``, waiting while there is none: the kind's end item
        once the input ended, then ``None``."""

    @abstractmethod
    def owes_reply(self) -> bool:
        """The input ended: whether the provider owes a reply to what the stream sent."""

    @abstractmethod
    def can_reconnect(self) -> bool:
        """Whether the stream has nothing in progress a new connection would cut."""

    @abstractmethod
    def retrying(self, error: Exception) -> None:
        """A failed attempt the session retries: the provider's recoverable error event."""

    def nothing_pending(self) -> bool:
        """The wait for the provider's reply ran out with it silent: whether nothing it was sent
        is left unanswered. A kind that cannot tell owes its answer."""
        return False

    def expects_close(self) -> bool:
        """The provider closed cleanly: whether the stream asked it to end its session (its end
        item went, or an item that ends the session)."""
        return False

    def goodbye(self) -> list[WsFrame]:
        """The frames that end the provider session when a replacement is due and when the input
        ends, read at each connection: the connection is replaced, or ends with the stream, once
        the provider finished its session or closed, bounded by the read limit. Empty: left at
        once."""
        return []

    def connecting(self, first: bool) -> None:
        """A connection opens; ``first``: the first of the session's run."""

    def opened(self, request_id: str | None) -> None:
        """A connection opened; ``request_id`` is the provider's id from its handshake."""

    def session_finished(self) -> None:
        """The provider finished a session; the next connection opens for the next input."""

    def contexts(self) -> list[str] | None:
        """The stream's open contexts on its multiplexer's socket; ``None``: its own."""
        return None

    def input_idle_after(self) -> float | None:
        """Seconds without input after which ``input_idle`` frames go, once per silence."""
        return None

    def input_idle(self) -> list[WsFrame]:
        """The frames sent once the input stayed silent, which are not input."""
        return []

    def response_idle_after(self) -> float | None:
        """Seconds of silence after response data and the latest input; no first-response timer."""
        return None

    def response_idle(self) -> list[WsFrame]:
        """The response went silent for ``response_idle_after`` seconds."""
        return []


@dataclass
class _Queued:
    message: object
    connection: int
    """The connection it was sent for."""


class WebSocketHost:
    """One stream's side of ``_core.WebSocketSession``: the plugin's protocol hooks and its
    kind's terms, behind an object of their own, so the session's host interface never shares
    names with the stream's own."""

    def __init__(
        self, protocol: WebSocketProtocol[Any, Any, Any], kind: WebSocketKind[Any]
    ) -> None:
        self._protocol = protocol
        self._kind = kind
        kind.host = self
        self.request_id: str | None = None
        self.context_id: str | None = None
        self.reading = False
        """Whether a received message is being handled: what the stream reports now answers it."""
        self._session: _core.WebSocketSession | None = None
        self._controls: collections.deque[_Queued] = collections.deque()
        self._control_sent = asyncio.Event()
        self._control_taken = asyncio.Event()
        # the provider finished its session and no connection opened since: no connection takes
        # what is sent, the next one's ``connect`` rebuilding the state
        self._between_sessions = False
        self._reconnecting = False
        # the take of the open connection's control messages; one connection each
        self._taking_control: asyncio.Task[list[WsFrame]] | None = None
        # connections opened by the stream, and by the current run
        self._connections = 0
        self._run_connections = 0
        # what the message being handled signals
        self._signal: Signal | None = None
        self._named_request: str | None = None
        self.response_received = False

    @property
    def label(self) -> str:
        return self._kind.label

    def send_now(self, message: object, *, first: bool) -> None:
        if self._between_sessions:
            return
        connection = self._connections + int(self._reconnecting)
        queued = _Queued(message, max(connection, 1))
        if first:
            self._controls.appendleft(queued)
        else:
            self._controls.append(queued)
        self._control_sent.set()

    def signal(self, signal: Signal) -> None:
        """What the provider's message tells the connection, reported while it is handled."""
        self._signal = signal

    def name_request(self, request_id: str) -> None:
        """The provider named its id for what the stream carries now."""
        self._named_request = request_id

    def reconnect(self, reason: str) -> None:
        if (session := self._session) is not None:
            # Commands issued after the replacement was requested belong to its new session.
            self._reconnecting = True
            session.reconnect(reason)

    def end_session(self) -> None:
        """The provider's session is over, as the stream read it: the next item goes on a new
        connection."""
        if (session := self._session) is not None:
            session.end_session()

    async def run(self, conn_options: APIConnectOptions) -> None:
        """Runs the socket to the end of the stream, its failure raised."""
        mux = self._protocol.multiplexer
        context = self.context_id = utils.shortuuid() if mux is not None else None
        self._run_connections = 0
        session = self._session = _core.WebSocketSession(
            self,
            conn_options,
            self._kind.label,
            mux=mux._native() if mux is not None else None,
            context_id=context,
            pacing=self._protocol.pacing,
        )
        if mux is not None and context is not None:
            mux._streams[context] = self
        try:
            await session.run()
        finally:
            await session.aclose()
            if mux is not None and context is not None:
                mux._streams.pop(context, None)

    async def aclose(self) -> None:
        if (session := self._session) is not None:
            await session.aclose()

    def contexts(self) -> list[str]:
        """The stream's open contexts on its multiplexer's socket."""
        contexts = self._kind.contexts()
        if contexts is not None:
            return contexts
        return [self.context_id] if self.context_id is not None else []

    # the session's host interface

    async def _core_connect(self, conn_options: APIConnectOptions) -> WebSocketConnection:
        self._between_sessions = False
        self._reconnecting = False
        # the last connection's take, dropped with it, must not take what this one is sent; a
        # take that already ran took the last connection's message, which went with it
        if (taking := self._taking_control) is not None:
            taking.cancel()
            self._taking_control = None
        self._kind.connecting(first=self._run_connections == 0)
        self._connections += 1
        self._run_connections += 1
        protocol = self._protocol
        target = await protocol.connect(conn_options)
        return WebSocketConnection(
            target=target,
            keepalive=_keepalive_frame(protocol.keepalive),
            max_session_duration=protocol.max_session_duration,
            ping_interval=protocol.ping_interval,
            setup=frames_of(protocol.setup()),
            awaits_ready=protocol.awaits_ready,
            goodbye=self._kind.goodbye(),
        )

    async def _core_next_input(self) -> WebSocketInput | None:
        while (batch := await self._kind.take()) is not None:
            frames = [f for item in batch.items for f in frames_of(self._protocol.encode(item))]
            # a control message ``encode`` sent goes ahead of the frames it returned
            while self._controls:
                self._control_taken.clear()
                await self._control_taken.wait()
            if frames:
                return WebSocketInput(frames=frames, pace=batch.pace)
        return None

    async def _core_next_control(self) -> list[WsFrame]:
        self._taking_control = asyncio.current_task()
        while True:
            while self._controls:
                if self._controls[0].connection > max(self._connections, 1):
                    # The new connection cancels this take and starts its own; do not hand a
                    # new session's command to the connection being replaced.
                    self._control_sent.clear()
                    await self._control_sent.wait()
                    continue
                queued = self._controls.popleft()
                self._control_taken.set()
                # one sent for an earlier connection: ``connect`` rebuilt the session since
                if queued.connection < self._connections:
                    continue
                self._protocol.sent_now(queued.message)
                return frames_of(queued.message)
            self._control_sent.clear()
            await self._control_sent.wait()

    def _core_input_idle_after(self) -> float | None:
        return self._kind.input_idle_after()

    async def _core_input_idle(self) -> list[WsFrame]:
        return self._kind.input_idle()

    def _core_response_idle_after(self) -> float | None:
        return self._kind.response_idle_after()

    async def _core_response_idle(self) -> list[WsFrame]:
        return self._kind.response_idle()

    async def _core_on_message(self, frame: WsFrame, message: object) -> WebSocketRead:
        self._signal = self._named_request = None
        self.response_received = False
        self.reading = True
        try:
            if message is None:
                message = self._protocol.decode(frame)
            if message is not None:
                self._protocol.on_message(message)
        except APIError as error:
            if error.request_id is None:
                error.request_id = self._named_request or self.request_id
            raise
        except Exception as error:
            failure = protocol_error(self._protocol, f"{type(error).__name__}: {error}")
            failure.request_id = self._named_request or self.request_id
            raise failure from error
        finally:
            self.reading = False
        if self._named_request is not None:
            self.request_id = self._named_request
        return WebSocketRead(
            signal=self._signal, request_id=self._named_request, response=self.response_received
        )

    def _core_opened(self, request_id: str | None) -> None:
        self.request_id = request_id
        self._kind.opened(request_id)

    def _core_session_finished(self) -> None:
        # what is queued was for the finished session: the input it held back goes on the next
        self._between_sessions = True
        self._controls.clear()
        self._control_taken.set()
        self._kind.session_finished()

    async def _core_close_context(self) -> WebSocketContextsCancel:
        cancel = WebSocketContextsCancel(contexts=[], frames=[])
        if (mux := self._protocol.multiplexer) is None:
            return cancel
        for context in self.contexts():
            if frames := frames_of(mux.close_context(context)):
                cancel.contexts.append(context)
                cancel.frames.extend(frames)
        return cancel

    def _core_retrying(self, error: Exception) -> None:
        self._kind.retrying(error)

    def _core_completion(self) -> Completion:
        return self._protocol.completion

    def _core_owes_reply(self) -> bool:
        return self._kind.owes_reply()

    def _core_nothing_pending(self) -> bool:
        return self._kind.nothing_pending()

    def _core_expects_close(self) -> bool:
        return self._kind.expects_close()

    def _core_can_reconnect(self) -> bool:
        return self._kind.can_reconnect()


class WebSocketMultiplexer(ABC, Generic[InboundT, OutboundT]):
    """The provider sockets streams share, opened on demand: several streams on one socket, each
    its own context that ``route`` sends each received message to, or a pool (``routed =
    False``) of sockets carrying one stream each, a socket reused by the next stream once the
    one before ended cleanly."""

    keepalive: Keepalive[OutboundT] | None = None
    max_session_duration: float | None = None
    """Seconds after which a socket takes no new stream; it closes once its streams ended."""
    ping_interval: float | None = 30.0
    """Seconds between websocket pings; a ping unanswered for half of it fails the socket and
    every stream on it as lost. ``None`` sends none."""
    max_contexts: int | None = None
    """Streams one socket carries at most; another socket opens for the next."""
    max_sockets: int | None = None
    """Sockets open at most; a stream finding none with room waits for one, within
    ``conn_options.pool_timeout``."""
    idle_timeout: float | None = None
    """Seconds a socket no stream holds stays open."""
    keep_warm: bool = False
    """A socket with room is kept open as a standby: one opens when a stream takes the last, and
    an idle one the provider closed is opened again, so the next stream finds one open."""
    reuses_sockets: ClassVar[bool] = True
    """A socket is reused by the next stream once its stream left it. ``False``: each stream takes
    a fresh socket, which closes after it; with ``keep_warm`` the next one is already open."""
    routed: ClassVar[bool] = True
    """Each received message goes to the context ``route`` names. A pool (``False``) carries
    one stream a socket and hands it every frame once the socket is ready; ``route`` then sees
    only the socket's own frames (before it is ready, or with no stream on it)."""
    awaits_ready: ClassVar[bool] = False
    """Each socket opens a provider session that says when it is ready: no stream takes it until
    ``ready`` reads a message saying so, within the connect limit."""

    @abstractmethod
    async def connect(self, conn_options: APIConnectOptions) -> WebSocketTarget:
        """One shared socket, from the current options."""

    def setup(self) -> bytes | OutboundT | None:
        """The frame each socket sends first: the provider session's setup."""
        return None

    def route(self, message: InboundT) -> str | None:
        """The context ``message`` belongs to (one of a stream's contexts); ``None`` for a
        socket-level message. Raising fails the socket and every stream on it."""
        return None

    def ready(self, message: InboundT) -> bool:
        """Whether ``message`` says the socket's provider session is ready (with
        ``awaits_ready``)."""
        return False

    def open_context(self, context_id: str) -> bytes | OutboundT | None:
        """The frame that opens a context, if the provider needs one."""
        return None

    def close_context(self, context_id: str) -> bytes | OutboundT | None:
        """The frame that ends a context its stream left before the provider finished it."""
        return None

    def context_ended(self, message: InboundT) -> str | None:
        """The context ``message`` ends, if it is the provider's last word on one. A pool
        (``routed = False``) that reads it cancels a stream left mid-answer with
        ``close_context`` and reuses the socket once the provider ended the cancelled contexts;
        one that does not closes that socket."""
        return None

    def goodbye(self) -> bytes | OutboundT | None:
        """The frame sent on a socket the multiplexer closes (idle, replaced, spent, or at its
        close) before the close: the provider session's own end."""
        return None

    def decode(self, frame: bytes | str) -> InboundT | None:
        """A received frame as ``Inbound``, ``None`` to skip it; see ``WebSocketProtocol.decode``."""
        decoder = _class_decoder(type(self), WebSocketMultiplexer)
        decoded: InboundT | None = decoder.decode(frame, type(self).__name__)
        return decoded

    def prewarm(self) -> None:
        """Open a shared socket now, unless one has room: the first stream finds it open."""
        self._native().prewarm()

    def invalidate(self) -> None:
        """Options changed: idle sockets close now, busy ones once their streams ended."""
        if (native := getattr(self, "_mux", None)) is not None:
            native.invalidate()

    async def aclose(self) -> None:
        if (native := getattr(self, "_mux", None)) is not None:
            await native.aclose()

    def _native(self) -> _core.WebSocketMux:
        native: _core.WebSocketMux | None = getattr(self, "_mux", None)
        if native is None:
            options = WebSocketMuxOptions(
                max_contexts=self.max_contexts,
                max_sockets=self.max_sockets,
                idle_timeout=self.idle_timeout,
                keep_warm=self.keep_warm,
                routed=self.routed,
                reuses_sockets=self.reuses_sockets,
                reads_cancel_end=self._reads_cancel_end(),
            )
            native = self._mux = _core.WebSocketMux(self, options)
            self._streams: dict[str, WebSocketHost] = {}
        return native

    def _reads_cancel_end(self) -> bool:
        return type(self).context_ended is not WebSocketMultiplexer.context_ended

    def _stream(self, context: str) -> WebSocketHost | None:
        """The stream holding a context."""
        if (stream := self._streams.get(context)) is not None:
            return stream
        return next((s for s in self._streams.values() if context in s.contexts()), None)

    async def _core_connect(self, conn_options: APIConnectOptions) -> WebSocketConnection:
        target = await self.connect(conn_options)
        return WebSocketConnection(
            target=target,
            keepalive=_keepalive_frame(self.keepalive),
            max_session_duration=self.max_session_duration,
            ping_interval=self.ping_interval,
            setup=frames_of(self.setup()),
            awaits_ready=self.awaits_ready,
            # a shared socket says goodbye through ``goodbye()`` whenever it closes one
            goodbye=[],
        )

    async def _core_route(self, frame: WsFrame) -> WebSocketMuxRoute:
        nowhere = WebSocketMuxRoute(context=None, message=None, ready=False, ended_context=None)
        try:
            message = self.decode(frame)
            if message is None:
                return nowhere
            if self.awaits_ready and self.ready(message):
                return replace(nowhere, ready=True)
            context = self.route(message)
            # A late frame may still acknowledge a context's cancellation.
            if context is None or (stream := self._stream(context)) is None:
                return replace(nowhere, ended_context=self.context_ended(message))
            # The stream reads the message decoded here, without decoding it again.
            return replace(nowhere, context=stream.context_id, message=message)
        except APIError:
            raise
        except Exception as error:
            raise protocol_error(self, f"{type(error).__name__}: {error}") from error

    async def _core_open_context(self, context_id: str) -> list[WsFrame]:
        return frames_of(self.open_context(context_id))

    async def _core_goodbye(self) -> list[WsFrame]:
        return frames_of(self.goodbye())
