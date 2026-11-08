"""Speech to text: the base classes a provider subclasses (generated from the core's
declarations) and the adapters, whose fallback chain is the core's (``_core.FallbackStt``)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from livekit import rtc

from ._generated import facades_stt as _facades, public_stt as _public
from ._generated.public_stt import *  # noqa: F403
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


class FallbackAdapter(_facades.FallbackAdapter):
    """Several STT behind one: a failed one is skipped until it recovers."""

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

    def prewarm(self) -> None:
        self._members[0].prewarm()


class StreamAdapter(_facades.StreamAdapter):
    """A recognize-only STT as a streaming one: the VAD cuts the audio into utterances, each
    recognized on its own (``_core.SttStreamAdapter``)."""

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
