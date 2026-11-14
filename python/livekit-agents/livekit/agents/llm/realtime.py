from __future__ import annotations

import asyncio
import time
from abc import abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from livekit import rtc

from .. import utils
from .._generated import facades_llm_realtime as _facades, public_llm_realtime as _public
from .._generated.public_llm_realtime import *  # noqa: F403
from .._generated.realtime import EventTypes, RealtimeModel, RealtimeModelError, RealtimeSession
from .._websocket import (
    Batch,
    InboundT,
    Keepalive,
    OutboundT,
    WebSocketHost,
    WebSocketKind,
    WebSocketProtocol,
    WebSocketTarget,
    protocol_error,
)
from ..types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

if TYPE_CHECKING:
    from .tool_context import Tool, Toolset


class RealtimeModelFallbackAdapter(_facades.RealtimeModelFallbackAdapter):
    """Falls back between realtime models (or restarts one), preserving chat context and handlers.

    A failed model is skipped until its cooldown ends, the conversation is replayed onto the next
    one, and a reply cut by the swap is requested again.
    """


__all__ = [
    "EventTypes",
    "Keepalive",
    "Ready",
    "RealtimeWebSocketSession",
    "RealtimeModelError",
    "RealtimeModelFallbackAdapter",
    "WebSocketTarget",
    "protocol_error",
]
__all__ += _public.__all__


@dataclass(frozen=True)
class Ready:
    """Reported once the provider's session is ready for input after the connection's setup
    (with ``awaits_ready``)."""


class RealtimeWebSocketSession(
    WebSocketProtocol[rtc.AudioFrame, InboundT, OutboundT], RealtimeSession[Any]
):
    """A realtime session over the provider's socket. The plugin writes ``connect`` (built from
    the current instructions, context and tools), ``encode`` (input audio) and ``on_message``,
    which drives the session's events; its commands go through ``send_now``. A change the
    provider applies only to a new session calls ``reconnect``. The socket runs from construction
    to ``aclose()``; its failures retry under ``conn_options``, each one a recoverable ``error``
    event, and the one that ends it a final one."""

    frame_ms: int | None = None
    """Audio is re-chunked to frames of this many milliseconds before ``encode``."""

    def __init__(
        self,
        realtime_model: RealtimeModel,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        input_sample_rate: int | None = None,
        input_num_channels: int = 1,
    ) -> None:
        super().__init__(
            realtime_model,
            input_sample_rate=input_sample_rate,
            input_num_channels=input_num_channels,
        )
        self._audio_in = utils.aio.Chan[rtc.AudioFrame]()
        # the generations requested or started and not yet drained: a due replacement waits
        self._generations = 0
        self._websocket = WebSocketHost(self, _Conversation(self))
        self._websocket_tasks = utils.aio.TaskSet()
        self._websocket_tasks.spawn(
            self._websocket_main(conn_options), name="RealtimeWebSocketSession"
        )

    @abstractmethod
    def on_message(self, message: InboundT) -> None:
        """One received message: the plugin emits the session's events from it, and reports
        ``Ready`` once the provider session is ready (with ``awaits_ready``)."""

    def _core_attach(self, tools: Callable[[], Sequence[Tool | Toolset]]) -> None:
        super()._core_attach(tools)
        self.on("generation_created", self._generation_created)

    def _generation_created(self, ev: GenerationCreatedEvent) -> None:
        # a requested one counts from its request
        if not ev.user_initiated:
            self._generations += 1

    async def _core_generation(
        self, request: asyncio.Future[GenerationCreatedEvent]
    ) -> GenerationCreatedEvent:
        self._generations += 1
        try:
            return await super()._core_generation(request)
        except BaseException:
            self._generations -= 1
            raise

    async def _core_drain(self, ev: GenerationCreatedEvent, req_id: str | None) -> None:
        try:
            await super()._core_drain(ev, req_id)
        finally:
            self._generations -= 1

    def report(self, ready: Ready) -> None:
        """The provider's session is ready for input."""
        self._websocket.signal("ready")

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        if not self._audio_in.closed:
            self._audio_in.send_nowait(frame)

    async def aclose(self) -> None:
        self._audio_in.close()
        await self._websocket.aclose()
        await self._websocket_tasks.aclose()

    async def _websocket_main(self, conn_options: APIConnectOptions) -> None:
        try:
            await self._websocket.run(conn_options)
        except Exception as error:
            self._emit_socket_error(error, recoverable=False)

    def _emit_socket_error(self, error: Exception, *, recoverable: bool) -> None:
        self.emit(
            "error",
            RealtimeModelError(
                timestamp=time.time(),
                label=self.realtime_model.label,
                error=error,
                recoverable=recoverable,
            ),
        )


class _Conversation(WebSocketKind[rtc.AudioFrame]):
    """The realtime session's terms for its socket: audio in, re-chunked; a reply in progress is
    never cut by a replacement; nothing is owed at the close. The session retries under its own
    options."""

    def __init__(self, session: RealtimeWebSocketSession[Any, Any]) -> None:
        self._session = session
        self._chunks: utils.media.AudioChunker | None = None

    @property
    def label(self) -> str:
        return self._session.realtime_model.label

    async def take(self) -> Batch | None:
        try:
            frame = await self._session._audio_in.recv()
        except utils.aio.ChanClosed:
            return None
        if self._chunks is None:
            self._chunks = utils.media.AudioChunker(self._session.frame_ms)
        audio = self._chunks.push(frame)
        return Batch(list(audio), sum(chunk.duration for chunk in audio))

    def owes_reply(self) -> bool:
        return False

    def can_reconnect(self) -> bool:
        return self._session._generations == 0

    def retrying(self, error: Exception) -> None:
        self._session._emit_socket_error(error, recoverable=True)
