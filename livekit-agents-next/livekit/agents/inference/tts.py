from __future__ import annotations

import functools
from typing import Any

from .. import _bindings, _core, tts
from .._generated.inference_tts import (
    CartesiaModels,
    CartesiaOptions,
    DeepgramModels,
    DeepgramOptions,
    FallbackModel,
    FallbackModelType,
    FishAudioModels,
    FishAudioOptions,
    InworldModels,
    InworldOptions,
    RimeModels,
    RimeOptions,
    TTSEncoding,
    TTSModels,
    XaiModels,
    XaiOptions,
)
from ..language import LanguageCode
from ..types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, APIConnectOptions, NotGivenOr
from ..utils import is_given, shortuuid
from ._utils import Credentials, get_inference_headers

__all__ = [
    "TTS",
    "SynthesizeStream",
    "TTSModels",
    "TTSEncoding",
    "FallbackModel",
    "FallbackModelType",
    "CartesiaModels",
    "CartesiaOptions",
    "DeepgramModels",
    "DeepgramOptions",
    "FishAudioModels",
    "FishAudioOptions",
    "InworldModels",
    "InworldOptions",
    "RimeModels",
    "RimeOptions",
    "XaiModels",
    "XaiOptions",
]

DEFAULT_SAMPLE_RATE = 24000


class TTS(tts.TTS):
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
        http_session: Any = None,
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
            http_session: Unused; the core owns the connection.
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
        self._extra = dict(extra_kwargs) if is_given(extra_kwargs) else {}
        self._fallback = [
            _core.tts_normalize_fallback_str(f) if isinstance(f, str) else dict(f)
            for f in fallbacks
        ]
        self._conn = conn_options if is_given(conn_options) else DEFAULT_API_CONNECT_OPTIONS
        self._creds = Credentials.resolve(base_url, api_key, api_secret)
        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=True,
                aligned_transcript=_core.tts_has_aligned_transcript(model, self._extra),
            ),
            sample_rate=sample_rate if is_given(sample_rate) else DEFAULT_SAMPLE_RATE,
            num_channels=1,
        )

    class Markup(tts.TTS.Markup):
        def _provider_key(self) -> str:
            assert isinstance(self._tts, TTS)
            return _core.tts_markup_provider_key(self._tts._model)  # type: ignore[no-any-return]

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

    @functools.cached_property
    def _native(self) -> _core.InferenceTts:
        # built on first use, after a fork; the client splits, converts and aligns each synthesis
        return _core.InferenceTts(
            self._model,
            voice=self._voice,
            language=str(self._language) if self._language else None,
            encoding=self._encoding,
            sample_rate=self.sample_rate,
            base_url=self._creds.base_url,
            api_key=self._creds.api_key,
            api_secret=self._creds.api_secret,
            extra_kwargs=self._extra,
            fallback=self._fallback or None,
            timeout=self._conn.timeout,
            max_retry=self._conn.max_retry,
        )

    def _core_leg(self) -> Any:
        """The core's own client, unless a subclass replaces what it would bypass."""
        return (
            self if _bindings.overrides(self, TTS, "synthesize", "stream") else self._core_native()
        )

    def _core_native(self) -> _core.InferenceTts:
        """The client as a session of the current job holds it: its connections carry that job."""
        return self._native.identified(list(get_inference_headers().items()), self)

    def _set_expressive(self, enabled: bool) -> None:
        super()._set_expressive(enabled)
        self._native.set_expressive(enabled)

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
            self._extra.update(extra_kwargs)
        self._capabilities.aligned_transcript = _core.tts_has_aligned_transcript(
            self._model, self._extra
        )
        self._native.update_options(
            model=model if is_given(model) else None,
            voice=voice if is_given(voice) else None,
            language=str(self._language) if is_given(language) else None,
            extra=extra_kwargs if is_given(extra_kwargs) else None,
        )

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> tts.ChunkedStream:
        return self._synthesize_with_stream(text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        return SynthesizeStream(tts=self, conn_options=conn_options)


class SynthesizeStream(tts.SynthesizeStream):
    """One gateway synthesis: text in, audio and alignment out; the client splits the sentences
    and converts the markup."""

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        native = await self._tts._core_native().stream()
        output_emitter.initialize(
            request_id=shortuuid("tts_"),
            sample_rate=self._tts.sample_rate,
            num_channels=1,
            stream=True,
            mime_type="audio/pcm",
        )

        await tts._duplex(self, native, output_emitter, self.input, lambda t: t, native.push)
