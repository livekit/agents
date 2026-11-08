from __future__ import annotations

import dataclasses
from typing import Any

from .. import _core, tts
from .._generated import facades_inference_tts as _facades, public_inference_tts as _public
from .._generated.inference_tts import FallbackModel as FallbackModel
from .._generated.language import LanguageCode
from .._generated.public_inference_tts import *  # noqa: F403
from ..types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, APIConnectOptions, NotGivenOr
from ..utils import is_given
from . import _resolve_credentials, get_inference_headers

FallbackModelType = FallbackModel | str


__all__ = [
    "TTS",
    "SynthesizeStream",
    "FallbackModel",
    "FallbackModelType",
]
__all__ += _public.__all__

DEFAULT_SAMPLE_RATE = 24000


class TTS(_facades.TTS, tts.TTS):
    """LiveKit Inference TTS over the core's gateway client (``_core.InferenceTts``)."""

    def __init__(
        self,
        model: TTSModels | str,
        *,
        voice: NotGivenOr[str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        encoding: NotGivenOr[TTSEncoding] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        api_secret: NotGivenOr[str] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
        fallback: NotGivenOr[list[FallbackModelType] | FallbackModelType] = NOT_GIVEN,
        conn_options: NotGivenOr[APIConnectOptions] = NOT_GIVEN,
    ) -> None:
        """LiveKit Inference TTS.

        Args:
            model: ``"provider/model[:voice]"``.
            voice: The voice; overrides the model string's.
            language: The language to synthesize.
            encoding: The audio encoding asked of the gateway.
            sample_rate: The sample rate asked of the gateway.
            base_url: The gateway URL; ``LIVEKIT_INFERENCE_URL`` or the cloud gateway by default.
            api_key: ``LIVEKIT_INFERENCE_API_KEY`` or ``LIVEKIT_API_KEY`` by default.
            api_secret: ``LIVEKIT_INFERENCE_API_SECRET`` or ``LIVEKIT_API_SECRET`` by default.
            extra_kwargs: Provider options passed through to the gateway.
            fallback: Fallback models, names or ``FallbackModel`` entries, run by the gateway.
            conn_options: The connection timeout, and the retries the gateway runs server-side.
        """
        model, parsed_voice = _core.tts_parse_model_string(model)
        if parsed_voice and not is_given(voice):
            voice = parsed_voice
        fallbacks = fallback if isinstance(fallback, list) else [fallback] if fallback else []
        self._model: str = model
        self._voice = voice if is_given(voice) else None
        self._language = LanguageCode(language) if is_given(language) else None
        self._encoding = encoding if is_given(encoding) else "pcm_s16le"
        self._extra_kwargs = dict(extra_kwargs) if is_given(extra_kwargs) else {}
        self._fallback = [
            _core.tts_normalize_fallback_str(f) if isinstance(f, str) else f.copy()
            for f in fallbacks
        ]
        self._conn_options = conn_options if is_given(conn_options) else DEFAULT_API_CONNECT_OPTIONS
        credentials = _resolve_credentials(base_url, api_key, api_secret)
        self._base_url, self._api_key, self._api_secret = dataclasses.astuple(credentials)
        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=True,
                aligned_transcript=_core.tts_has_aligned_transcript(model, self._extra_kwargs),
            ),
            sample_rate=sample_rate if is_given(sample_rate) else DEFAULT_SAMPLE_RATE,
            num_channels=1,
        )

    class Markup(tts.TTS.Markup):
        def _provider_key(self) -> str:
            assert isinstance(self._tts, TTS)
            return _core.tts_markup_provider_key(self._tts._model)

    @classmethod
    def from_model_string(cls, model: str) -> TTS:
        """A TTS from ``"provider/model[:voice]"``."""
        return cls(model)

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "livekit"

    def _core_conn_options(self) -> APIConnectOptions | None:
        """The policy this provider was given, which a session or run that supplies none
        keeps (models.md)."""
        return self._conn_options

    def _core_native(self) -> _core.InferenceTts:
        """The client as a session of the current job holds it: its connections carry that job."""
        return self._native.identified(list(get_inference_headers().items()), self)

    def update_options(
        self,
        *,
        voice: NotGivenOr[str] = NOT_GIVEN,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> None:
        """Update the options the next synthesis uses."""
        if is_given(model):
            self._model = model
        if is_given(voice):
            self._voice = voice
        if is_given(language):
            self._language = LanguageCode(language)
        if is_given(extra_kwargs):
            self._extra_kwargs.update(extra_kwargs)
        self._capabilities.aligned_transcript = _core.tts_has_aligned_transcript(
            self._model, self._extra_kwargs
        )
        self._native.update_options(
            model=model if is_given(model) else None,
            voice=voice if is_given(voice) else None,
            language=str(self._language) if is_given(language) else None,
            extra=extra_kwargs if is_given(extra_kwargs) else None,
        )

    def synthesize(
        self, text: str, *, conn_options: NotGivenOr[APIConnectOptions] = NOT_GIVEN
    ) -> tts.ChunkedStream:
        resolved = conn_options if is_given(conn_options) else self._conn_options
        return self._synthesize_with_stream(text, conn_options=resolved)

    def stream(
        self, *, conn_options: NotGivenOr[APIConnectOptions] = NOT_GIVEN
    ) -> SynthesizeStream:
        resolved = conn_options if is_given(conn_options) else self._conn_options
        return SynthesizeStream(tts=self, conn_options=resolved)


class SynthesizeStream(tts.SynthesizeStream):
    """One gateway synthesis: text in, audio and alignment out; the client splits the sentences
    and converts the markup."""

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts

    async def _core_open(self) -> _core.TtsStream:
        return await self._tts._core_native().stream(
            conn_options=self._conn_options, expressive=self._expressive
        )

    _run = tts.SynthesizeStream._core_run_native
