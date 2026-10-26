from __future__ import annotations

import asyncio
import weakref
from typing import TYPE_CHECKING, Any

from .. import _core, stt, vad as vad_mod
from .._generated.inference_stt import (
    AssemblyAIModels,
    AssemblyaiOptions,
    CartesiaModels,
    CartesiaOptions,
    DeepgramFluxModels,
    DeepgramFluxOptions,
    DeepgramModels,
    DeepgramOptions,
    ElevenlabsModels,
    ElevenlabsOptions,
    FallbackModel,
    FallbackModelType,
    InworldModels,
    InworldOptions,
    SpeechmaticsModels,
    SpeechmaticsOptions,
    STTEncoding,
    STTLanguages,
    STTModels,
    XaiModels,
    XaiOptions,
)
from ..language import LanguageCode
from ..log import logger
from ..types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, APIConnectOptions, NotGivenOr
from ..utils import aio, is_given
from ._utils import Credentials, get_inference_headers

if TYPE_CHECKING:
    from ..voice.events import ConversationItemAddedEvent

__all__ = [
    "STT",
    "SpeechStream",
    "STTModels",
    "STTLanguages",
    "STTEncoding",
    "FallbackModel",
    "FallbackModelType",
    "AssemblyAIModels",
    "AssemblyaiOptions",
    "CartesiaModels",
    "CartesiaOptions",
    "DeepgramModels",
    "DeepgramOptions",
    "DeepgramFluxModels",
    "DeepgramFluxOptions",
    "ElevenlabsModels",
    "ElevenlabsOptions",
    "InworldModels",
    "InworldOptions",
    "SpeechmaticsModels",
    "SpeechmaticsOptions",
    "XaiModels",
    "XaiOptions",
]

DEFAULT_SAMPLE_RATE = 16000
_MAX_AGENT_CONTEXT_CHARS = 1750


class STT(stt.STT):
    """LiveKit Inference STT over the core's gateway client (``_core.InferenceStt``)."""

    def __init__(
        self,
        model: NotGivenOr[STTModels | str] = NOT_GIVEN,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        encoding: NotGivenOr[STTEncoding] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        api_secret: NotGivenOr[str] = NOT_GIVEN,
        http_session: Any = None,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
        fallback: NotGivenOr[list[FallbackModelType] | FallbackModelType] = NOT_GIVEN,
        conn_options: NotGivenOr[APIConnectOptions] = NOT_GIVEN,
        vad: NotGivenOr[vad_mod.VAD | None] = NOT_GIVEN,
    ) -> None:
        """LiveKit Inference STT.

        Args:
            model: ``"provider/model[:language]"``.
            language: The language to transcribe; overrides the model string's.
            base_url: The gateway URL; ``LIVEKIT_INFERENCE_URL`` or the cloud gateway by default.
            encoding: The audio encoding sent to the gateway.
            sample_rate: The sample rate sent to the gateway.
            api_key: ``LIVEKIT_INFERENCE_API_KEY`` or ``LIVEKIT_API_KEY`` by default.
            api_secret: ``LIVEKIT_INFERENCE_API_SECRET`` or ``LIVEKIT_API_SECRET`` by default.
            http_session: Unused; the core owns the connection.
            extra_kwargs: Provider options passed through to the gateway.
            fallback: Fallback models, names or ``FallbackModel`` entries, run by the gateway.
            conn_options: The connection timeout the gateway client uses.
            vad: A VAD whose end of speech finalizes the transcript (Speechmatics only).
        """
        if isinstance(model, str):
            model, parsed_language = _core.stt_parse_model_string(model)
            if parsed_language and not is_given(language):
                language = parsed_language
        fallbacks = fallback if isinstance(fallback, list) else [fallback] if fallback else []
        self._model: str | None = model if is_given(model) else None
        self._language = LanguageCode(language) if is_given(language) else None
        self._encoding = encoding if is_given(encoding) else "pcm_s16le"
        self._sample_rate = sample_rate if is_given(sample_rate) else DEFAULT_SAMPLE_RATE
        self._extra = dict(extra_kwargs) if is_given(extra_kwargs) else {}
        self._fallback = [
            _core.stt_normalize_fallback_str(f) if isinstance(f, str) else dict(f)
            for f in fallbacks
        ]
        self._conn = conn_options if is_given(conn_options) else DEFAULT_API_CONNECT_OPTIONS
        self._creds = Credentials.resolve(base_url, api_key, api_secret)
        self._vad = self._resolve_vad(vad if is_given(vad) else None)
        self._session_keyterms: list[str] = []
        self._streams = weakref.WeakSet[SpeechStream]()
        super().__init__(capabilities=self._fold_capabilities())

    @classmethod
    def from_model_string(cls, model: str) -> STT:
        """An STT from ``"provider/model[:language]"``."""
        return cls(model)

    @property
    def model(self) -> str:
        return self._model or "unknown"

    @property
    def provider(self) -> str:
        return "livekit"

    def _fold_capabilities(self) -> stt.STTCapabilities:
        caps = _core.stt_capabilities(
            self._model,
            fallback=[f["model"] for f in self._fallback],
            extra_kwargs=self._extra,
        )
        caps["aligned_transcript"] = "word" if caps["aligned_transcript"] else False
        return stt.STTCapabilities(**caps)

    def _resolve_vad(self, vad: vad_mod.VAD | None) -> vad_mod.VAD | None:
        match _core.stt_resolve_vad_for_model(self._model, has_vad=vad is not None):
            case "drop_and_warn":
                logger.warning(
                    "`vad` will be ignored: model %r handles endpointing server-side.", self._model
                )
                return None
            case "create_default":
                from .vad import VAD

                return VAD()
        return vad

    def _native(self, language: str | None) -> _core.InferenceStt:
        return _core.InferenceStt(
            model=self._model,
            language=language or (str(self._language) if self._language else None),
            encoding=self._encoding,
            sample_rate=self._sample_rate,
            base_url=self._creds.base_url,
            api_key=self._creds.api_key,
            api_secret=self._creds.api_secret,
            extra_kwargs=self._extra | self._keyterm_extra(),
            fallback=self._fallback or None,
            timeout=self._conn.timeout,
            vad=self._vad,
            headers=list(get_inference_headers().items()),
        )

    def _keyterm_extra(self) -> dict[str, Any]:
        extra = _core.stt_keyterms_extra_for_model(
            self._model, extra_kwargs=self._extra, session_keyterms=self._session_keyterms
        )
        return extra or {}

    async def _recognize_impl(self, buffer: Any, **_: Any) -> stt.SpeechEvent:
        raise NotImplementedError(
            "LiveKit Inference STT does not support batch recognition, use stream() instead"
        )

    def stream(
        self,
        *,
        language: NotGivenOr[STTLanguages | str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        stream = SpeechStream(
            stt=self,
            language=str(LanguageCode(language)) if is_given(language) else None,
            conn_options=conn_options,
        )
        self._streams.add(stream)
        return stream

    def update_options(
        self,
        *,
        model: NotGivenOr[STTModels | str] = NOT_GIVEN,
        language: NotGivenOr[STTLanguages | str] = NOT_GIVEN,
        extra: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> None:
        """Update the options; live streams get them as a mid-stream ``session.update``."""
        if is_given(model):
            model, parsed_language = _core.stt_parse_model_string(model)
            if parsed_language and not is_given(language):
                language = parsed_language
            self._model = model
            self._vad = self._resolve_vad(self._vad)
        if is_given(language):
            self._language = LanguageCode(language)
        if is_given(extra):
            self._extra.update(extra)
            extra = {**extra, **self._keyterm_extra()}
        self._capabilities = self._fold_capabilities()
        for stream in self._streams:
            stream.update_options(model=model, language=language, extra=extra)

    def _update_session_keyterms(self, keyterms: list[str]) -> None:
        if keyterms == self._session_keyterms:
            return
        if not self._capabilities.keyterms:
            super()._update_session_keyterms(keyterms)
            return
        self._session_keyterms = list(keyterms)
        extra = self._keyterm_extra()
        for stream in self._streams:
            stream._apply_keyterms(extra)

    def _push_conversation_item(self, ev: ConversationItemAddedEvent) -> None:
        item = ev.item
        if item.type == "message" and item.role == "assistant" and (text := item.text_content):
            self.update_options(extra={"agent_context": text[-_MAX_AGENT_CONTEXT_CHARS:]})


class SpeechStream(stt.SpeechStream):
    """One gateway recognition session; each retry opens a new one."""

    def __init__(self, *, stt: STT, language: str | None, conn_options: APIConnectOptions) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=stt._sample_rate)
        self._stt: STT = stt
        self._language = language
        self._native: _core.InferenceSttStream | None = None

    def update_options(
        self,
        *,
        model: NotGivenOr[str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        extra: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> None:
        if is_given(language):
            self._language = str(LanguageCode(language))
        if self._native is not None:
            self._native.update_options(
                model=model if is_given(model) else None,
                language=self._language if is_given(language) else None,
                extra=extra if is_given(extra) else None,
            )

    def _apply_keyterms(self, extra: dict[str, Any]) -> None:
        if self._native is not None:
            self._native.apply_keyterms(extra)

    async def _run(self) -> None:
        native = self._native = self._stt._native(self._language).stream(language=self._language)

        async def _push() -> None:
            async for item in self._input_ch:
                if isinstance(item, self._FlushSentinel):
                    native.flush()
                else:
                    native.push(item)
            native.end_input()

        push = asyncio.create_task(_push())
        try:
            while (ev := await native.recv()) is not None:
                self._event_ch.send_nowait(self._event(ev))
        finally:
            self._native = None
            await aio.cancel_and_wait(push)
            await native.aclose()

    def _event(self, ev: dict[str, Any]) -> stt.SpeechEvent:
        offset = self.start_time_offset
        alternatives = []
        for alt in ev.pop("alternatives"):
            alt["start_time"] += offset
            alt["end_time"] += offset
            for word in alt["words"] or []:
                word.start_time = (word.start_time or 0.0) + offset
                word.end_time = (word.end_time or 0.0) + offset
                word.start_time_offset = offset
            alternatives.append(stt.SpeechData(**alt))
        usage = ev.pop("recognition_usage")
        return stt.SpeechEvent(
            **ev,
            alternatives=alternatives,
            recognition_usage=stt.RecognitionUsage(**usage) if usage else None,
        )
