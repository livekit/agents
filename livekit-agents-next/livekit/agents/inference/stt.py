from __future__ import annotations

import dataclasses
from typing import Any

from .. import _bindings, _core, stt
from .._generated import (
    facades_inference_stt as _facades,
    public_inference_stt as _public,
    vad as vad_mod,
)
from .._generated.inference_stt import FallbackModel as FallbackModel
from .._generated.language import LanguageCode
from .._generated.public_inference_stt import *  # noqa: F403
from ..log import logger
from ..types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, APIConnectOptions, NotGivenOr
from ..utils import is_given
from . import _resolve_credentials, get_inference_headers

FallbackModelType = FallbackModel | str


__all__ = [
    "STT",
    "FallbackModel",
    "FallbackModelType",
]
__all__ += _public.__all__

DEFAULT_SAMPLE_RATE = 16000


class STT(_facades.STT, stt.STT):
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
            sample_rate: The sample rate sent to the gateway. Defaults to 24000 for OpenAI,
                16000 for other providers.
            api_key: ``LIVEKIT_INFERENCE_API_KEY`` or ``LIVEKIT_API_KEY`` by default.
            api_secret: ``LIVEKIT_INFERENCE_API_SECRET`` or ``LIVEKIT_API_SECRET`` by default.
            extra_kwargs: Provider options passed through to the gateway.
            fallback: Fallback models, names or ``FallbackModel`` entries, run by the gateway.
            conn_options: The connection timeout, and the retries the gateway runs server-side.
            vad: A VAD whose end of speech finalizes the transcript (Speechmatics and OpenAI).
        """
        if isinstance(model, str):
            model, parsed_language = _core.stt_parse_model_string(model)
            if parsed_language and not is_given(language):
                language = parsed_language
        fallbacks = fallback if isinstance(fallback, list) else [fallback] if fallback else []
        self._model: str | None = model if is_given(model) else None
        self._language = LanguageCode(language) if is_given(language) else None
        self._encoding = encoding if is_given(encoding) else "pcm_s16le"
        self._extra_kwargs = dict(extra_kwargs) if is_given(extra_kwargs) else {}
        self._fallback = [
            _core.stt_normalize_fallback_str(f) if isinstance(f, str) else f.copy()
            for f in fallbacks
        ]
        models = [self._model or "", *(f["model"] for f in self._fallback)]
        default_sample_rate = (
            24000
            if any(model.partition("/")[0] == "openai" for model in models)
            else DEFAULT_SAMPLE_RATE
        )
        self._sample_rate = sample_rate if is_given(sample_rate) else default_sample_rate
        self._conn_options = conn_options if is_given(conn_options) else DEFAULT_API_CONNECT_OPTIONS
        credentials = _resolve_credentials(base_url, api_key, api_secret)
        self._base_url, self._api_key, self._api_secret = dataclasses.astuple(credentials)
        self._vad = self._resolve_vad(vad if is_given(vad) else None)
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
        return _core.stt_capabilities(
            self._model,
            fallback=[f["model"] for f in self._fallback],
            extra_kwargs=self._extra_kwargs,
        )

    def _resolve_vad(self, vad: vad_mod.VAD | None) -> vad_mod.VAD | None:
        model = next(
            (
                model
                for model in [self._model, *(f["model"] for f in self._fallback)]
                if _core.stt_resolve_vad_for_model(model, has_vad=False) == "create_default"
            ),
            self._model,
        )
        match _core.stt_resolve_vad_for_model(model, has_vad=vad is not None):
            case "drop_and_warn":
                logger.warning(
                    "`vad` will be ignored: model %r handles endpointing server-side.", self._model
                )
                return None
            case "create_default":
                from . import VAD

                return VAD()
        return vad

    def _core_conn_options(self) -> APIConnectOptions | None:
        """The policy this provider was given, which a session or run that supplies none
        keeps (models.md)."""
        return self._conn_options

    def _core_native(self) -> _core.InferenceStt:
        """The client as a session of the current job holds it: its connections carry that job."""
        return self._native.identified(list(get_inference_headers().items()), self)

    async def _recognize_impl(self, buffer: Any, **_: Any) -> stt.SpeechEvent:
        raise NotImplementedError(
            "LiveKit Inference STT does not support batch recognition, use stream() instead"
        )

    def stream(
        self,
        *,
        language: NotGivenOr[STTLanguages | str] = NOT_GIVEN,
        conn_options: NotGivenOr[APIConnectOptions] = NOT_GIVEN,
    ) -> SpeechStream:
        resolved = conn_options if is_given(conn_options) else self._conn_options
        return SpeechStream(
            stt=self,
            language=str(LanguageCode(language)) if is_given(language) else None,
            conn_options=resolved,
        )

    def update_options(
        self,
        *,
        model: NotGivenOr[STTModels | str] = NOT_GIVEN,
        language: NotGivenOr[STTLanguages | str] = NOT_GIVEN,
        extra: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> None:
        """Update the options; open streams get them as a mid-stream ``session.update``."""
        if is_given(model):
            model, parsed_language = _core.stt_parse_model_string(model)
            if parsed_language and not is_given(language):
                language = parsed_language
            self._model = model
            self._vad = self._resolve_vad(self._vad)
            self._native.set_vad(self._vad)
        if is_given(language):
            self._language = LanguageCode(language)
        if is_given(extra):
            self._extra_kwargs.update(extra)
        self._capabilities = self._fold_capabilities()
        self._native.update_options(
            model=model if is_given(model) else None,
            language=str(self._language) if is_given(language) else None,
            extra=extra if is_given(extra) else None,
        )

    def _update_session_keyterms(self, keyterms: list[str]) -> None:
        self._native.update_keyterms(keyterms)

    def _push_conversation_item(self, role: str, text: str) -> None:
        self._native.push_conversation_item(role, text)


class SpeechStream(stt.RecognizeStream):
    """One gateway recognition session; the client reconnects it under its ``conn_options``."""

    def __init__(self, *, stt: STT, language: str | None, conn_options: APIConnectOptions) -> None:
        # the client retries, so this stream does not retry on top of it
        super().__init__(stt=stt, conn_options=dataclasses.replace(conn_options, max_retry=0))
        self._request_conn_options = conn_options
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

    async def _run(self) -> None:
        native = self._native = self._stt._core_native().stream(
            language=self._language,
            start_time_offset=self.start_time_offset,
            conn_options=self._request_conn_options,
        )
        try:
            await _bindings.pump(
                native,
                self.input,
                lambda item: native.flush() if isinstance(item, self.Flush) else native.push(item),
                self.send,
            )
        finally:
            self._native = None
