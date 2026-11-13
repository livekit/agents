from __future__ import annotations

import time
from abc import abstractmethod
from typing import Any

from livekit import rtc

from .. import utils
from .._generated import facades_llm_realtime as _facades, public_llm_realtime as _public
from .._generated.public_llm_realtime import *  # noqa: F403
from .._generated.realtime import EventTypes, RealtimeModel, RealtimeModelError, RealtimeSession
from ..http.socket import EndOfInput, InboundT, OutboundT, SocketProtocol
from ..types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions


class RealtimeModelFallbackAdapter(_facades.RealtimeModelFallbackAdapter):
    """Falls back between realtime models (or restarts one), preserving chat context and handlers.

    A failed model is skipped until its cooldown ends, the conversation is replayed onto the next
    one, and a reply cut by the swap is requested again.
    """

    def session(self, *, turn_detection_disabled: bool = False) -> RealtimeSession:
        raise NotImplementedError(
            "RealtimeModelFallbackAdapter sessions run inside a VoiceSession (the core swaps them)"
        )


__all__ = [
    "EventTypes",
    "RealtimeSocketSession",
    "RealtimeModelError",
    "RealtimeModelFallbackAdapter",
]
__all__ += _public.__all__


class RealtimeSocketSession(
    SocketProtocol[rtc.AudioFrame, InboundT, OutboundT], RealtimeSession[Any]
):
    """A realtime session over the provider's socket. The plugin writes ``connect`` (built from
    the current instructions, context and tools), ``encode`` (input audio) and ``on_message``,
    which drives the session's events; its commands go through ``send``. A change the provider
    applies only to a new session calls ``reconnect``. The socket runs from construction to
    ``aclose()``; its failures retry under ``conn_options``, each one a recoverable ``error``
    event, and the one that ends it a final one."""

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
        self._socket_init()
        self._socket_input_audio = utils.aio.Chan[rtc.AudioFrame]()
        self._socket_tasks = utils.aio.TaskSet()
        self._socket_tasks.spawn(self._socket_main(conn_options), name="RealtimeSocketSession")

    @abstractmethod
    def on_message(self, message: InboundT) -> None:
        """One received message: the plugin emits the session's events from it."""

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        if not self._socket_input_audio.closed:
            self._socket_input_audio.send_nowait(frame)

    async def aclose(self) -> None:
        self._socket_input_audio.close()
        if (session := self._socket_session) is not None:
            await session.aclose()
        await self._socket_tasks.aclose()

    async def _socket_main(self, conn_options: APIConnectOptions) -> None:
        try:
            await self._socket_run(conn_options)
        except Exception as error:
            self._emit_socket_error(error, recoverable=False)

    async def _socket_input(self) -> rtc.AudioFrame | EndOfInput:
        try:
            return await self._socket_input_audio.recv()
        except utils.aio.ChanClosed:
            return EndOfInput()

    def _socket_boundary(self, item: object) -> bool:
        # a realtime session has no segments: a due rotation runs between any two items
        return True

    def _socket_label(self) -> str:
        return self.realtime_model.label

    def _socket_deliver(self, answer: None) -> None:
        pass

    def _core_retrying(self, error: Exception) -> None:
        self._emit_socket_error(error, recoverable=True)

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
