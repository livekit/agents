from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterable
from typing import TypeAlias

from livekit import rtc

from ._generated import facades_stt as _facades, public_stt as _public
from ._generated.public_stt import *  # noqa: F403
from ._generated.public_stt import SpeechEnded, SpeechStarted, Transcript
from ._generated.socket import SocketConnection
from .http.socket import (
    Done,
    EndOfInput,
    InboundT,
    OutboundT,
    SocketProtocol,
    _single_retry_owner,
)
from .types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, APIConnectOptions, NotGivenOr
from .utils import aio

__all__ = [
    "Report",
    "SocketStream",
    "StreamAdapter",
    "FallbackAdapter",
]
__all__ += _public.__all__


class FallbackAdapter(_facades.FallbackAdapter):
    """Several STT behind one: a failed one is skipped until it recovers."""

    def _core_retries_itself(self) -> bool:
        return True

    def _update_session_keyterms(self, keyterms: list[str]) -> None:
        for stt in self._members:
            stt._update_session_keyterms(keyterms)

    def _push_conversation_item(self, role: str, text: str) -> None:
        for stt in self._members:
            stt._push_conversation_item(role, text)

    async def _recognize_impl(
        self,
        buffer: list[rtc.AudioFrame] | rtc.AudioFrame,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> SpeechEvent:
        frames = buffer if isinstance(buffer, list) else [buffer]
        return await self._core_native_recognize(
            frames, language if language else None, conn_options=conn_options
        )

    def prewarm(self) -> None:
        self._members[0].prewarm()


class StreamAdapter(_facades.StreamAdapter):
    """A recognize-only STT as a streaming one: the VAD cuts the audio into utterances, each
    recognized on its own."""

    def _core_retries_itself(self) -> bool:
        return True

    @property
    def model(self) -> str:
        return self._stt.model

    @property
    def provider(self) -> str:
        return self._stt.provider

    def _update_session_keyterms(self, keyterms: list[str]) -> None:
        self._stt._update_session_keyterms(keyterms)

    def _push_conversation_item(self, role: str, text: str) -> None:
        self._stt._push_conversation_item(role, text)

    async def _recognize_impl(
        self,
        buffer: list[rtc.AudioFrame] | rtc.AudioFrame,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechEvent:
        return await self._stt.recognize(
            buffer=buffer, language=language, conn_options=conn_options
        )

    def prewarm(self) -> None:
        self._stt.prewarm()


Report: TypeAlias = Transcript | SpeechStarted | SpeechEnded
"""What a stream reports of what its provider heard."""


class SocketStream(
    SocketProtocol[rtc.AudioFrame | RecognizeStream.Flush, InboundT, OutboundT], RecognizeStream
):
    """A recognition over the provider's socket: the plugin writes ``connect``, ``encode`` (audio
    frames, ``Flush`` and ``EndOfInput``) and ``on_message``, which reports what the provider
    heard; the framework runs the socket. Each attempt is one socket session; the stream's
    attempts own the retries, and each connection's end closes the speech it left open."""

    def __init__(
        self,
        *,
        stt: STT,
        conn_options: APIConnectOptions,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=sample_rate)
        self._socket_init()

    @abstractmethod
    def on_message(self, message: InboundT) -> Report | Iterable[Report] | Done | None:
        """One received message: what the provider heard, ``DONE`` once it finished."""

    async def _run(self) -> None:
        await self._socket_run(_single_retry_owner(self._conn_options))

    async def _core_connect(self, conn_options: APIConnectOptions) -> SocketConnection:
        self._end_speech()
        return await super()._core_connect(conn_options)

    async def _socket_input(self) -> rtc.AudioFrame | RecognizeStream.Flush | EndOfInput:
        try:
            return await self.input.recv()
        except aio.ChanClosed:
            return EndOfInput()

    def _socket_boundary(self, item: object) -> bool:
        return isinstance(item, RecognizeStream.Flush)

    def _socket_label(self) -> str:
        return self._stt.label

    # the recognition's answer, not the socket base's
    _core_settled = RecognizeStream._core_settled

    def _socket_deliver(self, answer: Report | Iterable[Report] | None) -> None:
        if answer is None:
            return
        for report in answer if isinstance(answer, Iterable) else [answer]:
            self.report(report)
