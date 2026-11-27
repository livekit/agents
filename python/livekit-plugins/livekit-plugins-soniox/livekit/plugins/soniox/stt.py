# Copyright 2025 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import ClassVar, Literal, NotRequired, TypedDict, assert_never

from livekit import rtc
from livekit.agents import (
    APIConnectOptions,
    APIStatusError,
    LanguageCode,
    stt,
)
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    NotGivenOr,
)

# Base URL for Soniox Speech-to-Text API.
BASE_URL = "wss://stt-rt.soniox.com/transcribe-websocket"

# the tokens that mark an endpoint, and a finalization
END_TOKEN = "<end>"
FINALIZED_TOKEN = "<fin>"


@dataclass
class ContextGeneralItem:
    key: str
    value: str


@dataclass
class ContextTranslationTerm:
    source: str
    target: str


@dataclass
class ContextObject:
    """Context object for models with context_version 2, for Soniox stt-rt-v3-preview and higher.

    Learn more about context in the documentation:
    https://soniox.com/docs/stt/concepts/context
    """

    general: list[ContextGeneralItem] | None = None
    text: str | None = None
    terms: list[str] | None = None
    translation_terms: list[ContextTranslationTerm] | None = None


@dataclass
class TranslationConfig:
    """Translation configuration for the Soniox Speech-to-Text API.

    See: https://soniox.com/docs/stt/api-reference/websocket-api
    """

    type: Literal["one_way", "two_way"]
    target_language: str | None = None
    """Target language for one-way translation."""
    language_a: str | None = None
    """First language for two-way translation."""
    language_b: str | None = None
    """Second language for two-way translation."""

    def __post_init__(self) -> None:
        if self.type == "one_way" and not self.target_language:
            raise ValueError("target_language is required for one_way translation")
        if self.type == "two_way" and not (self.language_a and self.language_b):
            raise ValueError("language_a and language_b are both required for two_way translation")


@dataclass
class STTOptions:
    """Configuration options for Soniox Speech-to-Text service."""

    model: str = "stt-rt-v6"

    language_hints: list[str] | None = None
    language_hints_strict: bool = False
    context: ContextObject | str | None = None

    num_channels: int = 1
    sample_rate: int = 16000

    enable_speaker_diarization: bool = False
    enable_language_identification: bool = True

    max_endpoint_delay_ms: int = 2000
    """Maximum delay in milliseconds between speech cessation and endpoint detection.
    Range: 500–3000.
    See: https://soniox.com/docs/stt/rt/endpoint-detection"""

    endpoint_sensitivity: float | None = None
    """How readily the model emits speech endpoints. Range: -1.0 to 1.0.
    Higher values make endpoints more likely (finalize sooner); lower values make them
    less likely. Leave as None to use the server-side default.
    Introduced in the Soniox v5 model; earlier models reject it."""

    endpoint_latency_adjustment_level: int | None = None
    """How aggressively the model reduces endpoint latency. Range: 0 to 3.
    Higher values reduce latency but may emit more endpoints and slightly reduce accuracy.
    Leave as None to use the server-side default.
    Introduced in the Soniox v5 model; earlier models reject it."""

    client_reference_id: str | None = None
    translation: TranslationConfig | None = None

    def __post_init__(self) -> None:
        if not (500 <= self.max_endpoint_delay_ms <= 3000):
            raise ValueError("max_endpoint_delay_ms must be between 500 and 3000")
        if self.endpoint_sensitivity is not None and not (-1.0 <= self.endpoint_sensitivity <= 1.0):
            raise ValueError("endpoint_sensitivity must be between -1.0 and 1.0")
        if self.endpoint_latency_adjustment_level is not None and not (
            0 <= self.endpoint_latency_adjustment_level <= 3
        ):
            raise ValueError("endpoint_latency_adjustment_level must be between 0 and 3")


class _Token(TypedDict):
    text: str
    is_final: bool
    language: NotRequired[str]
    speaker: NotRequired[str]
    start_ms: NotRequired[float]
    end_ms: NotRequired[float]
    confidence: NotRequired[float]
    translation_status: NotRequired[str]


class _Response(TypedDict):
    """A result, the session's end (``finished``) or an error: Soniox tags none of them."""

    tokens: NotRequired[list[_Token]]
    total_audio_proc_ms: NotRequired[float]
    finished: NotRequired[bool]
    error_code: NotRequired[int]
    error_message: NotRequired[str]


class _GeneralItem(TypedDict):
    key: str
    value: str


class _TranslationTerm(TypedDict):
    source: str
    target: str


class _Context(TypedDict, total=False):
    general: list[_GeneralItem]
    text: str
    terms: list[str]
    translation_terms: list[_TranslationTerm]


class _Translation(TypedDict):
    type: Literal["one_way", "two_way"]
    target_language: NotRequired[str]
    language_a: NotRequired[str]
    language_b: NotRequired[str]


class _Config(TypedDict):
    model: str
    audio_format: Literal["pcm_s16le"]
    num_channels: int
    enable_endpoint_detection: bool
    sample_rate: int
    language_hints: list[str] | None
    language_hints_strict: bool
    enable_speaker_diarization: bool
    enable_language_identification: bool
    client_reference_id: str | None
    max_endpoint_delay_ms: int
    context: NotRequired[str | _Context]
    endpoint_sensitivity: NotRequired[float]
    endpoint_latency_adjustment_level: NotRequired[int]
    translation: NotRequired[_Translation]


class _Keepalive(TypedDict):
    type: Literal["keepalive"]


def _context(context: ContextObject) -> _Context:
    wire = _Context()
    if context.general is not None:
        wire["general"] = [_GeneralItem(key=g.key, value=g.value) for g in context.general]
    if context.text is not None:
        wire["text"] = context.text
    if context.terms is not None:
        wire["terms"] = context.terms
    if context.translation_terms is not None:
        wire["translation_terms"] = [
            _TranslationTerm(source=t.source, target=t.target) for t in context.translation_terms
        ]
    return wire


class STT(stt.STT):
    """Speech-to-Text service using Soniox Speech-to-Text API.

    This service connects to Soniox Speech-to-Text API for real-time transcription
    with support for multiple languages, custom context, speaker diarization,
    and more.

    For complete API documentation, see: https://soniox.com/docs/stt/api-reference/websocket-api
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = BASE_URL,
        params: STTOptions | None = None,
    ) -> None:
        """Initialize instance of Soniox Speech-to-Text API service.

        Args:
            api_key: Soniox API key, if not provided, will look for SONIOX_API_KEY env variable.
            base_url: Base URL for Soniox Speech-to-Text API, default to BASE_URL defined in this
                module.
            params: Additional configuration parameters, such as model, language hints, context and
                speaker diarization.
        """
        params = params or STTOptions()
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
                aligned_transcript="chunk",
                offline_recognize=False,
                diarization=params.enable_speaker_diarization,
                # usage is the total_audio_proc_ms Soniox reports with each response
                reports_usage=True,
            )
        )

        resolved_key = api_key or os.getenv("SONIOX_API_KEY")
        if not resolved_key:
            raise ValueError("Soniox API key is required. Set SONIOX_API_KEY or pass api_key")
        self._api_key = resolved_key
        self._base_url = base_url
        self._params = params

    @property
    def model(self) -> str:
        return self._params.model

    @property
    def provider(self) -> str:
        return "soniox"

    async def _recognize_impl(
        self,
        buffer: list[rtc.AudioFrame] | rtc.AudioFrame,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        """Raise error since single-frame recognition is not supported
        by Soniox Speech-to-Text API."""
        raise NotImplementedError(
            "Soniox Speech-to-Text API does not support single frame recognition"
        )

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        """Return a new LiveKit streaming speech-to-text session; ``language`` hints only it."""
        params = self._params
        if language:
            params = replace(params, language_hints=[language])
        return SpeechStream(
            stt=self,
            params=params,
            base_url=self._base_url,
            api_key=self._api_key,
            conn_options=conn_options,
        )


class SpeechStream(stt.WebSocketStream[_Response, _Config | _Keepalive]):
    completion: ClassVar[stt.Completion] = "session_finished"
    keepalive = stt.Keepalive(every=5.0, frame=_Keepalive(type="keepalive"))

    def __init__(
        self,
        stt: STT,
        *,
        params: STTOptions,
        base_url: str,
        api_key: str,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=params.sample_rate)
        self._params = params
        self._base_url = base_url
        self._api_key = api_key
        # Translation routes original-language tokens to `_final_original` and translated
        # tokens to `_final`. In non-translation mode, all tokens go to `_final` and
        # `_final_original` stays empty (so `_final` IS the source side there).
        self._final = _TokenAccumulator()
        self._final_original = _TokenAccumulator()
        self._reported_ms = 0.0

    async def connect(self, conn_options: APIConnectOptions) -> stt.WebSocketTarget:
        # a connection's session starts from nothing
        self._final.reset()
        self._final_original.reset()
        self._reported_ms = 0.0
        url = self._base_url.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
        return stt.WebSocketTarget(url, headers={"Authorization": f"Bearer {self._api_key}"})

    def setup(self) -> _Config:
        params = self._params
        config = _Config(
            model=params.model,
            audio_format="pcm_s16le",
            num_channels=params.num_channels or 1,
            enable_endpoint_detection=True,
            sample_rate=params.sample_rate,
            language_hints=params.language_hints,
            language_hints_strict=params.language_hints_strict,
            enable_speaker_diarization=params.enable_speaker_diarization,
            enable_language_identification=params.enable_language_identification,
            client_reference_id=params.client_reference_id,
            max_endpoint_delay_ms=params.max_endpoint_delay_ms,
        )
        if isinstance(params.context, ContextObject):
            config["context"] = _context(params.context)
        elif params.context is not None:
            config["context"] = params.context
        if params.endpoint_sensitivity is not None:
            config["endpoint_sensitivity"] = params.endpoint_sensitivity
        if params.endpoint_latency_adjustment_level is not None:
            config["endpoint_latency_adjustment_level"] = params.endpoint_latency_adjustment_level
        if (tr := params.translation) is not None:
            translation = _Translation(type=tr.type)
            if tr.type == "one_way" and tr.target_language:
                translation["target_language"] = tr.target_language
            elif tr.type == "two_way" and tr.language_a and tr.language_b:
                translation["language_a"] = tr.language_a
                translation["language_b"] = tr.language_b
            config["translation"] = translation
        return config

    def encode(self, item: rtc.AudioFrame | stt.Flush | stt.EndOfAudio) -> bytes | None:
        match item:
            case rtc.AudioFrame():
                return bytes(item.data)
            case stt.Flush():
                return None
            case stt.EndOfAudio():
                # an empty frame ends the audio: Soniox finalizes and answers ``finished``
                return b""
            case _:
                assert_never(item)

    def on_message(self, message: _Response) -> None:
        if "error_code" in message or "error_message" in message:
            code = message.get("error_code", -1)
            raise APIStatusError(
                f"Soniox STT error: {code}",
                status_code=code,
                body=message,
                detail=message.get("error_message"),
            )
        translating = self._params.translation is not None
        final, final_original = self._final, self._final_original
        non_final, non_final_original = _TokenAccumulator(), _TokenAccumulator()
        for token in message.get("tokens", []):
            is_end = token["text"] in (END_TOKEN, FINALIZED_TOKEN)
            if translating and not is_end and token.get("translation_status") != "translation":
                # an original-language token: the source text only
                (final_original if token["is_final"] else non_final_original).update(token)
            elif is_end:
                if token["is_final"]:
                    self._endpoint()
            elif token["is_final"]:
                final.update(token)
            else:
                non_final.update(token)

        if final.text or non_final.text:
            source = _merge_lang_segments(
                final_original.language_segments, non_final_original.language_segments
            )
            primary = _merge_lang_segments(final.language_segments, non_final.language_segments)
            # every token final: a stable interim, for preemptive generation once speaking
            self.report(
                final.transcript(
                    non_final,
                    self.start_time_offset,
                    source=source if translating else primary,
                    target=primary if translating else [],
                    preflight=not non_final.text and self.speaking,
                )
            )

        if message.get("finished", False):
            self._endpoint()
        # processed audio for every response: gating on endpoints drops what never reached one
        processed = message.get("total_audio_proc_ms", 0.0)
        if processed > self._reported_ms:
            self.send(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.RECOGNITION_USAGE,
                    recognition_usage=stt.RecognitionUsage(
                        audio_duration=(processed - self._reported_ms) / 1000
                    ),
                )
            )
            self._reported_ms = processed
        if message.get("finished", False):
            self.report(stt.SessionFinished())

    def _endpoint(self) -> None:
        """Soniox detected the end of the speech: its final tokens are the final transcript."""
        final, final_original = self._final, self._final_original
        if not final.text and final_original.text:
            # the originals' translation is still to come: the next endpoint carries both
            return
        translating = self._params.translation is not None
        # an empty final retracts an interim no final token confirmed, and ends the speech
        self.report(
            final.transcript(
                _TokenAccumulator(),
                self.start_time_offset,
                source=final_original.language_segments if translating else final.language_segments,
                target=final.language_segments if translating else [],
                final=True,
            )
        )
        final.reset()
        final_original.reset()


def _merge_lang_segments(
    a: Sequence[tuple[str, str]], b: Sequence[tuple[str, str]]
) -> list[tuple[str, str]]:
    """Merge two (language, text) segment lists, combining adjacent segments of the same language."""
    result = list(a)
    for lang, text in b:
        if result and result[-1][0] == lang:
            lang, t = result[-1]
            result[-1] = (lang, t + text)
        else:
            result.append((lang, text))
    return result


def _lang_segments_to_fields(
    segments: Sequence[tuple[str, str]],
) -> tuple[list[LanguageCode] | None, list[str] | None]:
    """Convert `(lang, text)` runs to the parallel `SpeechData` field pair,
    or `(None, None)` when empty."""
    if not segments:
        return None, None
    return (
        [LanguageCode(lang) for lang, _ in segments],
        [t for _, t in segments],
    )


class _TokenAccumulator:
    """Accumulates token metadata (text, language, speaker, timing, confidence).

    Tokens are assumed to arrive in chronological order, but individual tokens
    may omit timing keys. start_time is the earliest ``start_ms`` seen and
    end_time the latest ``end_ms`` seen, so a token that carries timing late
    (or regresses) cannot collapse the span onto itself.
    """

    def __init__(self) -> None:
        self.text: str = ""
        self.language: str = ""
        self.speaker_id: str | None = None
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self._confidence_sum: float = 0.0
        self._confidence_count: int = 0
        self._has_start_time: bool = False
        self._lang_segments: list[tuple[str, str]] = []  # (language, text) pairs
        # per language: its characters, and when (in tokens taken) it reached them
        self._lang_stats: dict[str, tuple[int, int]] = {}
        self._taken = 0

    @property
    def language_segments(self) -> Sequence[tuple[str, str]]:
        return tuple(self._lang_segments)

    def _get_language(self) -> str:
        """Language with the most characters; ties go to the one that reached the count first."""
        if not self._lang_stats:
            return ""
        most_chars = max(chars for chars, _ in self._lang_stats.values())
        tied = [
            (lang_code, reached)
            for lang_code, (chars, reached) in self._lang_stats.items()
            if chars == most_chars
        ]
        return min(tied, key=lambda t: t[1])[0]

    def update(self, token: _Token) -> None:
        text = token["text"]
        lang = token.get("language", "")
        self.text += text
        self._taken += 1
        if lang and text:
            chars, _ = self._lang_stats.get(lang, (0, 0))
            self._lang_stats[lang] = (chars + len(text), self._taken)
            self.language = self._get_language()
        if "speaker" in token and self.speaker_id is None:
            self.speaker_id = token["speaker"]
        if "start_ms" in token:
            start_ms = token["start_ms"]
            if not self._has_start_time or start_ms < self.start_time:
                self._has_start_time = True
                self.start_time = start_ms
        if "end_ms" in token:
            self.end_time = max(self.end_time, token["end_ms"])
        if "confidence" in token:
            self._confidence_sum += token["confidence"]
            self._confidence_count += 1
        if text:
            if self._lang_segments and self._lang_segments[-1][0] == lang:
                lang, t = self._lang_segments[-1]
                self._lang_segments[-1] = (lang, t + text)
            else:
                self._lang_segments.append((lang, text))

    @property
    def confidence(self) -> float:
        if self._confidence_count == 0:
            return 0.0
        return self._confidence_sum / self._confidence_count

    def reset(self) -> None:
        self.text = ""
        self.language = ""
        self.speaker_id = None
        self.start_time = 0.0
        self.end_time = 0.0
        self._confidence_sum = 0.0
        self._confidence_count = 0
        self._has_start_time = False
        self._lang_segments = []
        self._lang_stats = {}
        self._taken = 0

    def transcript(
        self,
        other: _TokenAccumulator,
        start_time_offset: float,
        *,
        source: Sequence[tuple[str, str]],
        target: Sequence[tuple[str, str]],
        final: bool = False,
        preflight: bool = False,
    ) -> stt.Transcript:
        """The transcript of self (final tokens) followed by other (non-final ones)."""
        candidates = [acc.start_time for acc in (self, other) if acc._has_start_time]
        start = min(candidates) if candidates else 0.0
        end = max(self.end_time, other.end_time)
        total_count = self._confidence_count + other._confidence_count
        total_sum = self._confidence_sum + other._confidence_sum
        source_languages, source_texts = _lang_segments_to_fields(source)
        target_languages, target_texts = _lang_segments_to_fields(target)
        return stt.Transcript(
            self.text + other.text,
            LanguageCode(self.language if self.language else other.language),
            final=final,
            preflight=preflight,
            start_time=start / 1000 + start_time_offset,
            end_time=end / 1000 + start_time_offset,
            confidence=total_sum / total_count if total_count > 0 else 0.0,
            speaker_id=self.speaker_id if self.speaker_id is not None else other.speaker_id,
            source_languages=source_languages,
            source_texts=source_texts,
            target_languages=target_languages,
            target_texts=target_texts,
        )
