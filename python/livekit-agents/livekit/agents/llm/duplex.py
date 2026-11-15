from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterable, Callable, Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Generic, Literal, Never, Protocol, assert_never

from typing_extensions import TypeVar

from livekit import rtc

from .. import _bindings, _core, utils
from .._generated import facades_llm_duplex as _facades
from .._generated.public_llm_duplex import *  # noqa: F403
from .._generated.websocket import WsFrame
from .._websocket import (
    Batch,
    InboundT,
    Keepalive as Keepalive,
    OutboundT,
    WebSocketHost,
    WebSocketKind,
    WebSocketProtocol,
    WebSocketTarget as WebSocketTarget,
    frames_of,
    log_skipped as log_skipped,
    protocol_error as protocol_error,
)
from ..types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, APIConnectOptions, NotGivenOr
from ..utils import is_given
from .chat_context import ChatContext, ChatItem
from .realtime import RealtimeError, RealtimeModelError, RealtimeSessionReconnectedEvent
from .tool_context import Tool, ToolChoice, ToolContext, Toolset

DuplexEventTypes = Literal[
    "transcript_delta",  # the model's transcript of its own speech
    "function_call",
    "input_speech_started",  # the user's turns, as the plugin detects them
    "input_speech_stopped",
    "input_audio_transcription_completed",
    "session_reconnected",
    "metrics_collected",
    "error",
]

TEvent = TypeVar("TEvent", default=Never, contravariant=True)


class DuplexModel(ABC):
    """A speech model that listens and speaks at the same time, handling interruptions itself.

    Run one inside a ``VoiceSession`` with :class:`~livekit.agents.llm.DuplexRealtimeAdapter`.
    """

    def __init__(self, *, capabilities: DuplexCapabilities) -> None:
        self._capabilities = capabilities
        self._label = f"{type(self).__module__}.{type(self).__name__}"

    @property
    def model(self) -> str:
        return "unknown"

    @property
    def provider(self) -> str:
        return "unknown"

    @property
    def capabilities(self) -> DuplexCapabilities:
        return self._capabilities

    @property
    def label(self) -> str:
        return self._label

    def audio_gate(self) -> AudioGate | None:
        """The gate for this model's output, or None to let the adapter infer one."""
        return None

    @abstractmethod
    def session(self) -> DuplexSession:
        """Open a session; the adapter configures it with ``_update_session`` before use."""

    @abstractmethod
    async def aclose(self) -> None: ...

    async def _core_session(
        self, tools: Callable[[], Sequence[Tool | Toolset]] | None
    ) -> _bindings._DuplexSession:
        return _bindings._DuplexSession(self.session(), tools or list)

    async def __aenter__(self) -> DuplexModel:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.aclose()


class DuplexSession(ABC, rtc.EventEmitter[DuplexEventTypes | TEvent], Generic[TEvent]):
    def __init__(self, duplex_model: DuplexModel) -> None:
        super().__init__()
        self._duplex_model = duplex_model
        self._closing = False
        # set once configured; a model whose configuration is immutable waits on it to connect
        self._configured = asyncio.Event()
        # the caller's audio, gaps and mute changes; a plugin more than 250 ms behind skips to
        # the newest audio after a gap
        self._input = _core.DuplexInput()

    @property
    def duplex_model(self) -> DuplexModel:
        return self._duplex_model

    @property
    def capabilities(self) -> DuplexCapabilities:
        return self._duplex_model.capabilities

    @property
    @abstractmethod
    def audio_stream(self) -> AsyncIterable[DuplexAudioFrame]:
        """The model's output audio, its own silence included; it may stop between bursts."""

    @property
    @abstractmethod
    def tools(self) -> ToolContext: ...

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        """Queue the caller's audio; the plugin pulls it from ``_input``."""
        self._input.push(frame)

    def push_video(self, frame: rtc.VideoFrame) -> None:
        """Feed a video frame to the model; ignored by models without video input."""
        pass

    async def aclose(self) -> None:
        """Close the session; an override calls ``super().aclose()`` first.

        Releases a model waiting on ``_configured``; ``_closing`` tells it the configuration was
        abandoned.
        """
        self._closing = True
        self._configured.set()

    @abstractmethod
    async def _update_instructions(self, instructions: str) -> None: ...

    @abstractmethod
    async def _append_items(self, items: list[ChatItem]) -> None:
        """Tell the model about new chat items; the adapter owns the context and sends only these."""

    @abstractmethod
    async def _update_tools(self, tools: list[Tool]) -> None: ...

    @abstractmethod
    def _update_options(
        self, *, tool_choice: NotGivenOr[ToolChoice | None] = NOT_GIVEN
    ) -> None: ...

    def _generate_reply(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        tools: NotGivenOr[list[Tool]] = NOT_GIVEN,
    ) -> None:
        """Ask the model to speak now; its next speech is taken as the reply."""
        raise RealtimeError(f"{type(self).__name__} decides for itself when to speak")

    async def _update_session(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        chat_ctx: NotGivenOr[ChatContext] = NOT_GIVEN,
        tools: NotGivenOr[list[Tool]] = NOT_GIVEN,
    ) -> None:
        """Apply the whole configuration at once, right after the session is created."""
        if is_given(instructions):
            await self._update_instructions(instructions)

        if is_given(chat_ctx):
            await self._append_items(chat_ctx.items)

        if is_given(tools):
            await self._update_tools(tools)

        # only a complete configuration releases a model that cannot be reconfigured later
        self._configured.set()

    def _report_connection_acquired(self, acquire_time: float) -> None:
        """Report connection timing as a RealtimeModelMetrics event with zero usage."""
        model = self._duplex_model
        self.emit(
            "metrics_collected",
            _core._core_realtime_connection_metrics(model.model, model.provider, acquire_time),
        )


@dataclass(frozen=True)
class Ready:
    """Reported once the provider's session is ready for input after the connection's setup
    (with ``awaits_ready``)."""


@dataclass(frozen=True)
class SessionFinished:
    """Reported when the provider ended its session, its answer to ``goodbye`` among them: the
    connection is replaced, or, while the session closes, the session ends."""


class DuplexWebSocketSession(
    WebSocketProtocol[rtc.AudioFrame | DuplexInputGap | DuplexInputMute, InboundT, OutboundT],
    DuplexSession[TEvent],
):
    """A duplex session over the provider's socket. The plugin writes ``connect`` (from the
    configuration ``_update_session`` set, which the socket waits for), ``encode`` (the caller's
    audio, re-chunked to ``frame_ms``, and the input's gaps and mute changes, in order) and
    ``on_message``, which emits the session's events and reports ``Ready`` and
    ``SessionFinished``; its commands go through ``send_now``.

    A protocol with a ``goodbye`` drains its provider session before leaving it: at a due
    replacement (``max_session_duration``) and at ``aclose()``, the goodbye goes and the connection
    ends once the provider reports ``SessionFinished`` or closes, bounded by the read limit at the
    close. Each new connection starts from the newest input and emits ``session_reconnected``. Its
    failures retry under ``conn_options``, each one a recoverable ``error`` event, and the one that
    ends it a final one."""

    frame_ms: int | None = None
    """The caller's audio is re-chunked to frames of this many milliseconds before ``encode``."""

    def __init__(
        self,
        duplex_model: DuplexModel,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        input_sample_rate: int | None = None,
        input_num_channels: int = 1,
    ) -> None:
        """
        Args:
            conn_options: The socket's connection limits and retries.
            input_sample_rate: The audio format ``encode`` takes, converted from what is pushed
                (resampled, channels mixed); ``None`` takes it as pushed.
            input_num_channels: The channel count ``encode`` takes, with ``input_sample_rate``.
        """
        super().__init__(duplex_model)
        self._socket: _DuplexSocket[InboundT, OutboundT] = _DuplexSocket(
            self, input_sample_rate, input_num_channels
        )
        self._websocket = WebSocketHost(self, self._socket)
        self._websocket_tasks = utils.aio.TaskSet()
        self._websocket_run = self._websocket_tasks.spawn(
            self._websocket_main(conn_options), name="DuplexWebSocketSession"
        )

    def goodbye(self) -> bytes | OutboundT | None:
        """The message that asks the provider to end its session, which it answers with
        ``SessionFinished`` (or its close); ``None``: the connection is left without one."""
        return None

    def report(self, report: Ready | SessionFinished) -> None:
        """What the provider's message says about its session."""
        match report:
            case Ready():
                self._websocket.signal("ready")
                self._socket.ready()
            case SessionFinished():
                self._websocket.signal("session_finished")
            case _:
                assert_never(report)

    async def aclose(self) -> None:
        """Close the session: a provider session that is ready is sent the ``goodbye`` and
        drained first; an override calls ``super().aclose()`` first."""
        await super().aclose()
        if self._socket.end():
            # the run ends once the provider finished its session, or its read limit passed; its
            # failure is the session's error event
            await self._websocket_run
        await self._websocket.aclose()
        await self._websocket_tasks.aclose()

    async def _websocket_main(self, conn_options: APIConnectOptions) -> None:
        # the configuration a session starts with is complete before the first connection
        await self._configured.wait()
        if self._closing:
            return
        try:
            await self._websocket.run(conn_options)
        except Exception as error:
            self._emit_socket_error(error, recoverable=False)

    def _emit_socket_error(self, error: Exception, *, recoverable: bool) -> None:
        self.emit(
            "error",
            RealtimeModelError(
                timestamp=time.time(),
                label=self._duplex_model.label,
                error=error,
                recoverable=recoverable,
            ),
        )


class _DuplexSocket(
    WebSocketKind[rtc.AudioFrame | DuplexInputGap | DuplexInputMute], Generic[InboundT, OutboundT]
):
    """The duplex session's terms for its socket: the caller's input in order, each connection
    from the newest audio; the provider session left with its ``goodbye``, whose answer is all a
    closing session owes."""

    def __init__(
        self,
        session: DuplexWebSocketSession[InboundT, OutboundT],
        sample_rate: int | None,
        num_channels: int,
    ) -> None:
        self._session = session
        self._format = (sample_rate, num_channels)
        self._converter: utils.media.AudioConverter | None = None
        self._chunks = utils.media.AudioChunker(session.frame_ms)
        self._ending = asyncio.Event()
        self._connections = 0
        self._connect_started = 0.0
        # the open connection's provider session is ready, and the goodbye it was opened with
        self._ready = False
        self._goodbye: list[WsFrame] = []

    @property
    def label(self) -> str:
        return self._session.duplex_model.label

    async def take(self) -> Batch | None:
        while not self._ending.is_set():
            async with utils.aio.TaskSet() as race:
                taking = race.spawn(anext(self._session._input))
                await race.first_completed(taking, self._ending.wait())
            if not taking.done() or taking.cancelled():
                return None
            item = taking.result()
            if isinstance(item, DuplexInputGap | DuplexInputMute):
                # what is held goes ahead of the gap or the mute that follows it
                held = self._chunks.flush()
                return Batch([*held, item], sum(frame.duration for frame in held))
            frames = [
                chunk for converted in self._convert(item) for chunk in self._chunks.push(converted)
            ]
            if frames:
                return Batch(frames, sum(frame.duration for frame in frames))
        return None

    def _convert(self, frame: rtc.AudioFrame) -> list[rtc.AudioFrame]:
        sample_rate, num_channels = self._format
        if sample_rate is None:
            return [frame]
        if self._converter is None:
            self._converter = utils.media.AudioConverter(sample_rate, num_channels)
        return self._converter.push(frame)

    def ready(self) -> None:
        """The open connection's provider session is ready: the connection is acquired."""
        self._ready = True
        self._session._report_connection_acquired(time.perf_counter() - self._connect_started)

    def end(self) -> bool:
        """The session closes: whether its ready provider session drains first. The input ends,
        and the socket sends the goodbye and ends with its answer."""
        if not self._ready or not self._goodbye:
            return False
        self._ending.set()
        return True

    def connecting(self, first: bool) -> None:
        self._connect_started = time.perf_counter()
        self._ready = False
        if not first:
            # a new provider session: its input starts from the newest audio, nothing held over
            self._session._input.flush()
            self._converter = None
            self._chunks = utils.media.AudioChunker(self._session.frame_ms)

    def opened(self, request_id: str | None) -> None:
        self._connections += 1
        if self._connections > 1:
            self._session.emit("session_reconnected", RealtimeSessionReconnectedEvent())
        if not self._session.awaits_ready:
            self.ready()

    def owes_reply(self) -> bool:
        # the socket drains the goodbye's answer, the only reply a closing session owes
        return False

    def can_reconnect(self) -> bool:
        return True

    def goodbye(self) -> list[WsFrame]:
        self._goodbye = frames_of(self._session.goodbye())
        return self._goodbye

    def retrying(self, error: Exception) -> None:
        self._session._emit_socket_error(error, recoverable=True)


class AudioGate(Protocol):
    """Decides which frames of the model's output carry speech worth playing."""

    def update(self, frame: rtc.AudioFrame) -> bool:
        """True while the frame belongs to an open burst of output."""
        ...

    def deactivate(self) -> None:
        """End the open burst, keeping whatever has been learned about the model."""
        ...


class DuplexRealtimeAdapter(_facades.DuplexRealtimeAdapter):
    """Runs a :class:`DuplexModel` inside a ``VoiceSession``.

    The model's continuous output is gated and segmented into generations with their
    transcripts; output the model never transcribes still plays, without a chat item.
    """

    @property
    def duplex_model(self) -> DuplexModel:
        return self._duplex_model
