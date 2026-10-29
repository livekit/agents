from __future__ import annotations

import dataclasses
import functools
from typing import Any, Literal, TypedDict

from .. import _bindings, _core, stt
from .._generated import vad as vad_mod
from .._generated.inference_stt import FallbackModel
from .._generated.language import LanguageCode
from ..log import logger
from ..types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, APIConnectOptions, NotGivenOr
from ..utils import is_given
from ._utils import Credentials, get_inference_headers

DeepgramModels = Literal[
    "deepgram/nova-3",
    "deepgram/nova-3-medical",
    "deepgram/nova-2",
    "deepgram/nova-2-medical",
    "deepgram/nova-2-conversationalai",
    "deepgram/nova-2-phonecall",
]


DeepgramFluxModels = Literal[
    "deepgram/flux-general", "deepgram/flux-general-en", "deepgram/flux-general-multi"
]


CartesiaModels = Literal["cartesia/ink-whisper", "cartesia/ink-2"]


AssemblyAIModels = Literal[
    "assemblyai/universal-streaming",
    "assemblyai/universal-streaming-multilingual",
    "assemblyai/u3-rt-pro",
    "assemblyai/universal-3-5-pro",
    "assemblyai/universal-3-6-pro",
]


XaiModels = Literal["xai/stt-1"]


SpeechmaticsModels = Literal[
    "speechmatics/enhanced", "speechmatics/standard", "speechmatics/linden-1"
]


InworldModels = Literal["inworld/inworld-stt-1"]


GoogleModels = Literal["google/gemini-3.5-transcribe-live"]


class CartesiaOptions(TypedDict, total=False):
    min_volume: float
    max_silence_duration_secs: float
    turn_start_threshold: float
    turn_eager_end_threshold: float
    turn_end_threshold: float
    turn_end_timeout_ms: int
    keyterm: str | list[str]


class DeepgramOptions(TypedDict, total=False):
    filler_words: bool
    interim_results: bool
    endpointing: int
    punctuate: bool
    smart_format: bool
    keywords: list[tuple[str, float]]
    keyterm: str | list[str]
    profanity_filter: bool
    numerals: bool
    mip_opt_out: bool
    vad_events: bool
    diarize: bool
    dictation: bool
    detect_language: bool
    no_delay: bool
    utterance_end: bool
    redact: str | list[str]
    replace: str | list[str]
    search: str | list[str]
    tag: str | list[str]
    channels: int
    version: str
    callback: str
    callback_method: str
    extra: str


class DeepgramFluxOptions(TypedDict, total=False):
    eager_eot_threshold: float
    eot_threshold: float
    eot_timeout_ms: int
    keyterm: str | list[str]
    mip_opt_out: bool
    tag: str | list[str]
    detect_language: bool


class AssemblyaiOptions(TypedDict, total=False):
    format_turns: bool
    end_of_turn_confidence_threshold: float
    min_end_of_turn_silence_when_confident: int
    max_turn_silence: int
    keyterms_prompt: list[str]
    language_detection: bool
    inactivity_timeout: float
    prompt: str
    speaker_labels: bool
    agent_context: str
    previous_context_n_turns: int
    voice_focus: Literal["near-field", "far-field"]
    voice_focus_threshold: float
    mode: Literal["min_latency", "balanced", "max_accuracy"]


class SpeechmaticsOptions(TypedDict, total=False):
    domain: str
    output_locale: str
    max_delay: float
    max_delay_mode: str
    diarization: str
    speaker_sensitivity: float
    max_speakers: int
    prefer_current_speaker: bool
    enable_partials: bool
    enable_entities: bool
    punctuation_overrides: dict[str, Any]
    additional_vocab: list[dict[str, Any]]
    end_of_utterance_silence_trigger: float
    audio_filtering_config: dict[str, Any]
    transcript_filtering_config: dict[str, Any]


class XaiOptions(TypedDict, total=False):
    diarize: bool
    endpointing: int
    format: bool
    interim_results: bool


class InworldOptions(TypedDict, total=False):
    enable_voice_profile: bool
    voice_profile_top_n: int
    include_word_timestamps: bool
    audio_encoding: Literal["LINEAR16", "AUTO_DETECT"]
    inactivity_timeout_seconds: int
    end_of_turn_confidence_threshold: float
    min_end_of_turn_silence_when_confident: int
    prompts: list[str]
    vad_threshold: float


class GoogleOptions(TypedDict, total=False):
    language_codes: list[str]
    custom_vocabulary: list[str]


STTLanguages = Literal["multi", "en", "de", "es", "fr", "ja", "pt", "zh", "hi"]


FallbackModelType = FallbackModel | str


STTModels = (
    DeepgramModels
    | DeepgramFluxModels
    | CartesiaModels
    | AssemblyAIModels
    | XaiModels
    | SpeechmaticsModels
    | InworldModels
    | GoogleModels
    | Literal["auto"]
)


STTEncoding = Literal["pcm_s16le"]


# --- the frozen public paths -----------------------------------------------------------------
#
# Assigned here rather than in each class body: pydantic resolves annotations against
# `sys.modules[cls.__module__]` as it builds the class, so setting it early sends it
# looking for `Literal` and every foreign type in the re-export module, which imports
# none of them. What this fixes is `repr`, `pickle` and the equivalence gate all
# reporting `_generated.*` for a name whose import path never moved.
CartesiaOptions.__module__ = "livekit.agents.inference.stt"
DeepgramOptions.__module__ = "livekit.agents.inference.stt"
DeepgramFluxOptions.__module__ = "livekit.agents.inference.stt"
AssemblyaiOptions.__module__ = "livekit.agents.inference.stt"
SpeechmaticsOptions.__module__ = "livekit.agents.inference.stt"
XaiOptions.__module__ = "livekit.agents.inference.stt"
InworldOptions.__module__ = "livekit.agents.inference.stt"
GoogleOptions.__module__ = "livekit.agents.inference.stt"


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
    "GoogleModels",
    "GoogleOptions",
    "InworldModels",
    "InworldOptions",
    "SpeechmaticsModels",
    "SpeechmaticsOptions",
    "XaiModels",
    "XaiOptions",
]

DEFAULT_SAMPLE_RATE = 16000


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
            conn_options: The connection timeout, and the retries the gateway runs server-side.
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
            _core.stt_normalize_fallback_str(f) if isinstance(f, str) else f.copy()
            for f in fallbacks
        ]
        self._conn = conn_options if is_given(conn_options) else DEFAULT_API_CONNECT_OPTIONS
        self._creds = Credentials.resolve(base_url, api_key, api_secret)
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
        caps = _core.stt_capabilities(
            self._model,
            fallback=[f["model"] for f in self._fallback],
            extra_kwargs=self._extra,
        )
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

    @functools.cached_property
    def _native(self) -> _core.InferenceStt:
        # built on first use, after a fork: the client tracks its open streams and takes the
        # option and keyterm changes to them
        return _core.InferenceStt(
            model=self._model,
            language=str(self._language) if self._language else None,
            encoding=self._encoding,
            sample_rate=self._sample_rate,
            base_url=self._creds.base_url,
            api_key=self._creds.api_key,
            api_secret=self._creds.api_secret,
            extra_kwargs=self._extra,
            fallback=self._fallback or None,
            conn_options=self._conn,
            vad=self._vad,
        )

    def _core_leg(self) -> Any:
        """The core's own client, unless a subclass replaces what it would bypass."""
        return self if _bindings.overrides(self, STT, "stream") else self._core_native()

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
        resolved = conn_options if is_given(conn_options) else self._conn
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
            self._extra.update(extra)
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


class SpeechStream(stt.SpeechStream):
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
                lambda ev: self.send(stt._speech_event_from_core(ev)),
            )
        finally:
            self._native = None
