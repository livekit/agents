"""Speech to text: the base classes a provider subclasses (generated from the core's
declarations) and the adapters, whose fallback chain is the core's (``_core.FallbackStt``)."""

from __future__ import annotations

from dataclasses import replace
from typing import Literal, Protocol, runtime_checkable

from livekit import rtc

from . import _bindings, _core
from ._generated import facades_stt as _facades, public_stt as _public
from ._generated.public_stt import *  # noqa: F403
from ._generated.vad import VAD
from .types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, APIConnectOptions, NotGivenOr

__all__ = [
    "StreamAdapter",
    "FallbackAdapter",
    "SpeakerContext",
]
__all__ += _public.__all__


# A structural type, which has no core counterpart to be generated from.
@runtime_checkable
class SpeakerContext(Protocol):
    """A speaker model an STT context provides, which the expressive system turns into LLM
    instructions."""

    def to_instructions(self) -> str: ...


class FallbackAdapter(
    _facades.FallbackAdapter,
    _bindings.FallbackMembers[STT],
    STT[Literal["stt_availability_changed"]],
):
    """Several STT behind one: a failed one is skipped until it recovers."""

    _kind = "STT"

    def __init__(
        self,
        stt: list[STT],
        *,
        vad: VAD | None = None,
        attempt_timeout: float = 10.0,
        max_retry_per_stt: int = 1,
        retry_interval: float = 5,
    ) -> None:
        """
        Args:
            stt: The STT, in order of preference.
            vad: Wraps the non-streaming ones in a ``StreamAdapter``; required when any is.
            attempt_timeout: The timeout of one attempt.
            max_retry_per_stt: Retries of each before moving to the next.
            retry_interval: The wait between retries.
        """
        if not stt:
            raise ValueError("At least one STT instance must be provided.")
        if non_streaming := [t.label for t in stt if not t.capabilities.streaming]:
            if vad is None:
                raise ValueError(
                    f"STTs do not support streaming: {', '.join(non_streaming)}. "
                    "Provide a VAD to enable stt.StreamAdapter automatically "
                    "or wrap them with stt.StreamAdapter before using this adapter."
                )
            adapted: dict[int, STT] = {}
            for recognizer in stt:
                identity = id(recognizer)
                if identity not in adapted:
                    adapted[identity] = (
                        recognizer
                        if recognizer.capabilities.streaming
                        else StreamAdapter(stt=recognizer, vad=vad)
                    )
            stt = [adapted[id(recognizer)] for recognizer in stt]
        self._members = stt
        self._attempt_timeout = attempt_timeout
        self._max_retry_per_stt = max_retry_per_stt
        self._retry_interval = retry_interval
        # the chain's own merge, read before any chain is built
        super().__init__(
            capabilities=_core.stt_fallback_capabilities([t.capabilities for t in stt])
        )

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
        return await self._native.recognize(
            frames, language if language else None, conn_options=conn_options
        )

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> RecognizeStream:
        return StreamAdapterWrapper(
            stt=self, native=self._native, language=language, conn_options=conn_options
        )

    def prewarm(self) -> None:
        self._members[0].prewarm()


class StreamAdapterWrapper(RecognizeStream):
    """A recognition stream the core runs: a ``FallbackAdapter``'s chain or a
    ``StreamAdapter``'s VAD and recognitions."""

    def __init__(
        self,
        *,
        stt: STT,
        native: _core.FallbackStt | _core.SttStreamAdapter,
        language: NotGivenOr[str],
        conn_options: APIConnectOptions,
    ) -> None:
        # the core's chain reconnects: the stream does not retry on top of it
        super().__init__(stt=stt, conn_options=replace(conn_options, max_retry=0))
        self._request_conn_options = conn_options
        self._native = native
        self._language = language if language else None

    async def _run(self) -> None:
        native = await self._native.stream(
            self._language, self.start_time_offset, conn_options=self._request_conn_options
        )
        await _bindings.pump(
            native,
            self.input,
            lambda item: native.flush() if isinstance(item, self.Flush) else native.push(item),
            self.send,
        )


class StreamAdapter(_facades.StreamAdapter, STT):
    """A recognize-only STT as a streaming one: the VAD cuts the audio into utterances, each
    recognized on its own (``_core.SttStreamAdapter``)."""

    def __init__(self, *, stt: STT, vad: VAD) -> None:
        super().__init__(
            capabilities=STTCapabilities(
                streaming=True,
                interim_results=False,
                diarization=False,  # diarization requires streaming STT
                keyterms=stt.capabilities.keyterms,
                chat_context=stt.capabilities.chat_context,
            )
        )
        self._vad = vad
        self._stt = stt

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

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> RecognizeStream:
        return StreamAdapterWrapper(
            stt=self, native=self._native, language=language, conn_options=conn_options
        )

    def prewarm(self) -> None:
        self._stt.prewarm()
