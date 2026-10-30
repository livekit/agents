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

"""Speech-to-Text implementation for Gnani Vachana

This module provides an STT implementation that uses the Gnani Vachana API,
supporting both REST recognition and real-time streaming (WebSocket).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
import weakref
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, cast

import aiohttp
import numpy as np

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    LanguageCode,
    stt,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils.misc import is_given

if TYPE_CHECKING:
    from livekit.agents.utils import AudioBuffer

from .log import logger
from .models import (
    DEFAULT_STT_MODEL,
    INDIC_AUTO,
    PRISMA_V30,
    STT_MODELS,
    GnaniSTTFormats,
    GnaniSTTLanguages,
    GnaniSTTModels,
)

GnaniSTTFormat = GnaniSTTFormats

GNANI_STT_BASE_URL = "https://api.vachana.ai"

SAMPLE_RATE_16K = 16000
SAMPLE_RATE_8K = 8000
SAMPLE_RATE_44K = 44100
SAMPLE_RATE_48K = 48000
STREAM_SUPPORTED_SAMPLE_RATES = (
    SAMPLE_RATE_8K,
    SAMPLE_RATE_16K,
    SAMPLE_RATE_44K,
    SAMPLE_RATE_48K,
)
STREAM_CHUNK_BYTES = 1024

STT_FORMAT_VALUES = ("verbatim", "transcribe", "codemix", "transliterate")
_V30_ONLY_FORMATS = frozenset({"codemix", "transliterate"})

MAX_LANGUAGES = 4
MAX_KEYTERMS = 200
MAX_TOPIC_CHARS = 1600
MAX_PREVIOUS_TEXT_CHARS = 600
MAX_SPEAKER_NAMES = 4

# option name -> (WebSocket header, minimum, maximum, type). The server silently clamps
# values outside these ranges, which hides typos, so they are validated here instead.
_SESSION_TUNING: dict[str, tuple[str, float, float, type]] = {
    "idle_timeout_sec": ("x-idle-timeout-sec", 5, 900, int),
    "vad_threshold": ("x-vad-threshold", 0.0, 1.0, float),
    "min_silence_ms": ("x-min-silence-ms", 100, 2000, int),
    "min_speech_ms": ("x-min-speech-ms", 50, 2000, int),
    "min_segment_ms": ("x-min-segment-ms", 100, 1000, int),
    "max_speech_s": ("x-max-speech-s", 5.0, 120.0, float),
}

# Peak sample amplitude (of 32767) below which a frame is treated as silence. Only audio
# above it can start an utterance the server will answer, so only that audio makes the
# stream wait for a reply at the end of the input (~-48 dBFS: digital silence and dither
# stay below it, speech is far above it).
_SIGNAL_PEAK = 128
# A session that the server closes this quickly is not an idle timeout: stop reconnecting.
_QUICK_CLOSE_S = 2.0
_MAX_QUICK_CLOSES = 3


def _has_signal(pcm: bytes) -> bool:
    """True if 16-bit little-endian PCM holds anything louder than the noise floor."""
    if len(pcm) < 2:
        return False
    samples = np.frombuffer(pcm, dtype="<i2", count=len(pcm) // 2)
    return bool(np.abs(samples.astype(np.int32)).max() >= _SIGNAL_PEAK)


@dataclass
class GnaniSTTOptions:
    api_key: str
    # Comma-joined language code(s): "hi-IN", "indic-auto" or "hi-IN,en-IN".
    language: str
    sample_rate: int = SAMPLE_RATE_16K
    base_url: str = GNANI_STT_BASE_URL
    model: str | None = None
    preferred_language: str | None = None
    format: str = "verbatim"
    itn_native_numerals: bool = False
    use_streaming: bool = True
    # Prisma v3.0 context
    keyterms: tuple[str, ...] = ()
    topic: str | None = None
    previous_text: str | None = None
    speaker_names: tuple[str, ...] = ()
    # Streaming session tuning (None = server default)
    idle_timeout_sec: int | None = None
    vad_threshold: float | None = None
    min_silence_ms: int | None = None
    min_speech_ms: int | None = None
    min_segment_ms: int | None = None
    max_speech_s: float | None = None


_DEPRECATED_STT_KWARGS = frozenset(("organization_id", "user_id", "http_session"))


def _check_deprecated_args(kwargs: dict[str, Any], *, caller: str = "STT.__init__") -> None:
    """Warn about deprecated kwargs and raise on truly unknown ones."""
    for name in _DEPRECATED_STT_KWARGS:
        if name in kwargs:
            logger.warning(f"`{name}` is deprecated and no longer used")

    unknown = set(kwargs) - _DEPRECATED_STT_KWARGS
    if unknown:
        raise TypeError(
            f"{caller}() got unexpected keyword argument(s): {', '.join(sorted(unknown))}"
        )


# ---------------------------------------------------------------------------
# Option normalisation and validation
# ---------------------------------------------------------------------------


def _normalize_language(language: str | Sequence[str]) -> str:
    """Return ``language`` as a comma-joined string: ``"hi-IN"``, ``"indic-auto"``,
    ``"hi-IN,en-IN"``. Accepts a string (optionally comma-separated) or a list of codes."""
    parts = language.split(",") if isinstance(language, str) else list(language)
    codes = list(dict.fromkeys(c.strip() for c in parts if c and c.strip()))
    if not codes:
        raise ValueError("language must not be empty")
    return ",".join(codes)


def _language_codes(language: str) -> list[str]:
    return language.split(",")


def _is_multi_language(language: str) -> bool:
    codes = _language_codes(language)
    return len(codes) > 1 or codes[0] == INDIC_AUTO


def _normalize_terms(values: Sequence[str] | None, *, name: str) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        raise TypeError(f"{name} must be a list of strings, not a single string")
    terms = tuple(dict.fromkeys(v.strip() for v in values if v and v.strip()))
    for term in terms:
        if "<|" in term or "|>" in term:
            raise ValueError(f"{name} entries must not contain '<|' or '|>': {term!r}")
    return terms


def _single_line(text: str) -> str:
    """HTTP header values cannot contain newlines."""
    return " ".join(text.split())


def _validate_options(opts: GnaniSTTOptions) -> None:
    """Reject option combinations the API would ignore or answer with an error."""
    if opts.model is not None and opts.model not in STT_MODELS:
        # The API silently falls back to the default model for unknown names, which
        # would make a typo look like it works.
        raise ValueError(f"model must be one of {', '.join(STT_MODELS)}, got {opts.model!r}")
    is_v30 = opts.model == PRISMA_V30

    codes = _language_codes(opts.language)
    if INDIC_AUTO in codes and len(codes) > 1:
        raise ValueError(f"{INDIC_AUTO!r} cannot be combined with other language codes")
    if len(codes) > MAX_LANGUAGES:
        raise ValueError(f"at most {MAX_LANGUAGES} languages are supported, got {len(codes)}")
    if _is_multi_language(opts.language) and not is_v30:
        raise ValueError(
            f"language={opts.language!r} (automatic or multi-language detection) requires "
            f"model={PRISMA_V30!r}; the default model only transcribes one language per "
            "stream. Pass a single language code or set model='prisma-v3.0'."
        )

    if opts.format not in STT_FORMAT_VALUES:
        raise ValueError(
            f"format must be one of {', '.join(STT_FORMAT_VALUES)}, got {opts.format!r}"
        )
    if opts.format in _V30_ONLY_FORMATS and not is_v30:
        raise ValueError(f"format={opts.format!r} requires model={PRISMA_V30!r}")

    if opts.sample_rate not in STREAM_SUPPORTED_SAMPLE_RATES:
        allowed = ", ".join(str(r) for r in STREAM_SUPPORTED_SAMPLE_RATES)
        raise ValueError(f"sample_rate must be one of {allowed}, got {opts.sample_rate}")

    context_set = {
        "keyterms": bool(opts.keyterms),
        "topic": opts.topic is not None,
        "previous_text": opts.previous_text is not None,
        "speaker_names": bool(opts.speaker_names),
    }
    used = [name for name, is_set in context_set.items() if is_set]
    if used and not is_v30:
        raise ValueError(f"{', '.join(used)} require model={PRISMA_V30!r}")
    if len(opts.keyterms) > MAX_KEYTERMS:
        raise ValueError(f"at most {MAX_KEYTERMS} keyterms are supported, got {len(opts.keyterms)}")
    if opts.topic is not None:
        if not opts.topic.strip():
            raise ValueError("topic must not be empty")
        if len(opts.topic) > MAX_TOPIC_CHARS:
            raise ValueError(f"topic must be at most {MAX_TOPIC_CHARS} characters")
        if "<|" in opts.topic or "|>" in opts.topic:
            raise ValueError("topic must not contain '<|' or '|>'")
    if opts.previous_text is not None:
        if not opts.previous_text.strip():
            raise ValueError("previous_text must not be empty")
        if "<|" in opts.previous_text or "|>" in opts.previous_text:
            raise ValueError("previous_text must not contain '<|' or '|>'")
    if opts.speaker_names and not 1 <= len(opts.speaker_names) <= MAX_SPEAKER_NAMES:
        raise ValueError(f"speaker_names takes 1-{MAX_SPEAKER_NAMES} names")

    if opts.itn_native_numerals and is_v30:
        logger.warning(
            "itn_native_numerals has no effect with prisma-v3.0: the model writes numbers itself"
        )

    for name, (_, low, high, kind) in _SESSION_TUNING.items():
        value = getattr(opts, name)
        if value is not None and not low <= value <= high:
            shown = int if kind is int else float
            raise ValueError(f"{name} must be between {shown(low)} and {shown(high)}, got {value}")


def _merge_keyterms(user: Sequence[str], session: Sequence[str]) -> tuple[str, ...]:
    """User keyterms first, then the framework-managed ones, capped at the API limit."""
    merged = tuple(dict.fromkeys([*user, *session]))
    if len(merged) > MAX_KEYTERMS:
        logger.warning(f"too many keyterms ({len(merged)}); using the first {MAX_KEYTERMS}")
        merged = merged[:MAX_KEYTERMS]
    return merged


def _fallback_language(language: str) -> LanguageCode:
    """Language to report when the server does not say (``detected_language_code`` is only
    returned by prisma-v3.0, and not for ``format="transliterate"``)."""
    if _is_multi_language(language):
        return LanguageCode("multi")
    return LanguageCode(language)


# ---------------------------------------------------------------------------
# Request building (pure, so the wire format is easy to test)
# ---------------------------------------------------------------------------


def _build_rest_fields(opts: GnaniSTTOptions, language: str) -> list[tuple[str, str]]:
    """Multipart form fields for ``POST /stt/v3`` (everything except the audio)."""
    fields = [("language_code", language), ("format", opts.format)]
    if opts.model is not None:
        fields.append(("model", opts.model))
    if opts.keyterms:
        fields.append(("keyterms", json.dumps(list(opts.keyterms), ensure_ascii=False)))
    if opts.topic is not None:
        fields.append(("topic", opts.topic))
    if opts.previous_text is not None:
        fields.append(("previous_text", opts.previous_text[-MAX_PREVIOUS_TEXT_CHARS:]))
    if opts.speaker_names:
        fields.append(("speaker_names", json.dumps(list(opts.speaker_names), ensure_ascii=False)))
    if opts.preferred_language is not None:
        fields.append(("preferred_language", opts.preferred_language))
    if opts.itn_native_numerals:
        fields.append(("itn_native_numerals", "true"))
    return fields


def _ws_language_headers(language: str) -> dict[str, str]:
    """The streaming endpoint takes exactly one ``language_code``. A list of languages is
    expressed as ``indic-auto`` restricted to ``x-language-candidates``."""
    codes = _language_codes(language)
    if len(codes) == 1:
        return {"language_code": codes[0]}
    return {"language_code": INDIC_AUTO, "x-language-candidates": ",".join(codes)}


def _build_ws_headers(opts: GnaniSTTOptions) -> dict[str, str]:
    """Upgrade headers for ``wss://.../stt/v3/stream``. HTTP headers are bytes, so list
    values are JSON-escaped (``ensure_ascii``) as the API documents."""
    headers: dict[str, str] = {
        "x-api-key-id": opts.api_key,
        "x-sample-rate": str(opts.sample_rate),
        # Always explicit: omitting it makes prisma-v3.0 default to "transcribe" for
        # single hi-IN/en-IN, while prisma-v2.5 defaults to "verbatim".
        "format": opts.format,
        **_ws_language_headers(opts.language),
    }
    if opts.model is not None:
        headers["model"] = opts.model
    if opts.keyterms:
        headers["keyterms"] = json.dumps(list(opts.keyterms), ensure_ascii=True)
    if opts.topic is not None:
        headers["topic"] = _single_line(opts.topic)
    if opts.previous_text is not None:
        headers["previous_text"] = _single_line(opts.previous_text)[-MAX_PREVIOUS_TEXT_CHARS:]
    if opts.speaker_names:
        headers["speaker_names"] = json.dumps(
            [_single_line(n) for n in opts.speaker_names], ensure_ascii=True
        )
    if opts.preferred_language is not None:
        headers["preferred_language"] = opts.preferred_language
    if opts.itn_native_numerals:
        headers["itn_native_numerals"] = "true"
    for name, (header, *_rest) in _SESSION_TUNING.items():
        value = getattr(opts, name)
        if value is not None:
            headers[header] = str(value)
    return headers


class STT(stt.STT):
    """Gnani Vachana Speech-to-Text implementation.

    Provides speech-to-text functionality using Gnani's Vachana platform.
    Supports REST recognition and real-time streaming via WebSocket.

    Args:
        language: A BCP-47 language code (e.g. "hi-IN", "en-IN"), ``"indic-auto"`` to
            detect the language per utterance, or 2-4 codes (a list, or a comma-separated
            string such as "hi-IN,en-IN") to restrict detection. Automatic and
            multi-language detection require ``model="prisma-v3.0"``.
        model: "prisma-v3.0" or "prisma-v2.5". When unset the server default
            (prisma-v2.5) is used. Note the ``/stt/v3`` URL is the API version, not the
            model version.
        api_key: Gnani API key (falls back to GNANI_API_KEY env var).
        sample_rate: Audio sample rate for streaming (8000, 16000, 44100, or 48000).
        base_url: Vachana API base URL.
        preferred_language: Deprecated and undocumented by the current API; kept for
            backwards compatibility.
        format: "verbatim" (default), "transcribe" (number formatting), or, with
            prisma-v3.0 only, "codemix" or "transliterate".
        itn_native_numerals: prisma-v2.5 only. Render digits in native script when
            format="transcribe".
        use_streaming: When True (default), transcribe over the WebSocket stream
            (wss://.../stt/v3/stream). When False, use REST recognition
            (POST /stt/v3), which requires a local VAD — LiveKit wraps the STT
            with ``stt.StreamAdapter`` automatically.
        keyterms: prisma-v3.0 only. Up to 200 words or phrases to favour. LiveKit's
            ``stt_context_options`` keyterms are merged in automatically.
        topic: prisma-v3.0 only. What the audio is about (up to 1600 characters).
            Fixed for the life of a streaming connection.
        previous_text: prisma-v3.0 only. What was said just before (the last 600
            characters are used). Fixed for the life of a streaming connection.
        speaker_names: prisma-v3.0 only. 1-4 names that may be spoken. Fixed for the
            life of a streaming connection.
        idle_timeout_sec: Streaming only. Seconds without speech before the server ends
            the session (5-900, server default 60). The plugin reconnects transparently.
        vad_threshold, min_silence_ms, min_speech_ms, min_segment_ms, max_speech_s:
            Streaming only. Server-side voice activity detection tuning; see the
            Gnani realtime STT docs for ranges. ``min_silence_ms`` is the main latency
            knob (the server default is 500 ms).
    """

    def __init__(
        self,
        *,
        language: GnaniSTTLanguages | str | Sequence[str] = "en-IN",
        model: GnaniSTTModels | None = None,
        api_key: str | None = None,
        sample_rate: int = SAMPLE_RATE_16K,
        base_url: str = GNANI_STT_BASE_URL,
        preferred_language: str | None = None,
        format: GnaniSTTFormat = "verbatim",
        itn_native_numerals: bool = False,
        use_streaming: bool = True,
        keyterms: Sequence[str] | None = None,
        topic: str | None = None,
        previous_text: str | None = None,
        speaker_names: Sequence[str] | None = None,
        idle_timeout_sec: int | None = None,
        vad_threshold: float | None = None,
        min_silence_ms: int | None = None,
        min_speech_ms: int | None = None,
        min_segment_ms: int | None = None,
        max_speech_s: float | None = None,
        **kwargs: Any,
    ) -> None:
        _check_deprecated_args(kwargs)

        api_key = api_key or os.environ.get("GNANI_API_KEY")
        if not api_key:
            raise ValueError(
                "Gnani API key is required. "
                "Provide it directly or set GNANI_API_KEY environment variable."
            )

        user_keyterms = _normalize_terms(keyterms, name="keyterms")
        opts = GnaniSTTOptions(
            api_key=api_key,
            language=_normalize_language(language),
            sample_rate=sample_rate,
            base_url=base_url,
            model=model,
            preferred_language=preferred_language,
            format=format,
            itn_native_numerals=itn_native_numerals,
            use_streaming=use_streaming,
            keyterms=user_keyterms,
            topic=topic,
            previous_text=previous_text,
            speaker_names=_normalize_terms(speaker_names, name="speaker_names"),
            idle_timeout_sec=idle_timeout_sec,
            vad_threshold=vad_threshold,
            min_silence_ms=min_silence_ms,
            min_speech_ms=min_speech_ms,
            min_segment_ms=min_segment_ms,
            max_speech_s=max_speech_s,
        )
        _validate_options(opts)

        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=use_streaming,
                interim_results=False,
                aligned_transcript=False,
                # keyterms are a prisma-v3.0 feature; the other model ignores them
                keyterms=model == PRISMA_V30,
                # maps to the server's `commit` frame, which closes the open utterance
                # without waiting for the VAD's trailing silence
                manual_flush=use_streaming,
            )
        )

        self._api_key = api_key
        self._opts = opts
        self._user_keyterms: tuple[str, ...] = user_keyterms
        self._session_keyterms: tuple[str, ...] = ()
        self._streams = weakref.WeakSet[SpeechStream]()
        self._session: aiohttp.ClientSession | None = None

    @property
    def model(self) -> str:
        return self._opts.model or DEFAULT_STT_MODEL

    @property
    def provider(self) -> str:
        return "Gnani"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    @staticmethod
    def _single_attempt(conn_options: APIConnectOptions) -> APIConnectOptions:
        return APIConnectOptions(
            max_retry=0,
            retry_interval=conn_options.retry_interval,
            timeout=conn_options.timeout,
        )

    def update_options(
        self,
        *,
        language: NotGivenOr[GnaniSTTLanguages | str | Sequence[str]] = NOT_GIVEN,
        format: NotGivenOr[GnaniSTTFormat] = NOT_GIVEN,
        keyterms: NotGivenOr[Sequence[str]] = NOT_GIVEN,
    ) -> None:
        """Change options for new requests and for streams that are already running.

        Running streams apply ``format``, ``keyterms`` and a single ``language`` live
        (``update_settings``). A change to a *list* of languages is picked up the next
        time the stream (re)connects, because the server cannot change the candidate
        set mid-session. ``model``, ``topic``, ``previous_text`` and ``speaker_names``
        are fixed for a connection and cannot be updated.

        Nothing is changed if any value is rejected.
        """
        candidate = replace(self._opts)
        user_keyterms = self._user_keyterms
        if is_given(language):
            candidate.language = _normalize_language(language)
        if is_given(format):
            candidate.format = format
        if is_given(keyterms):
            user_keyterms = _normalize_terms(keyterms, name="keyterms")
            candidate.keyterms = _merge_keyterms(user_keyterms, self._session_keyterms)
        _validate_options(candidate)

        self._opts = candidate
        self._user_keyterms = user_keyterms
        for stream in self._streams:
            stream.update_options(
                language=candidate.language if is_given(language) else NOT_GIVEN,
                format=candidate.format if is_given(format) else NOT_GIVEN,
                keyterms=candidate.keyterms if is_given(keyterms) else NOT_GIVEN,
            )

    def _update_session_keyterms(self, keyterms: list[str]) -> None:
        if not self._capabilities.keyterms:
            super()._update_session_keyterms(keyterms)
            return
        session = _normalize_terms(keyterms, name="keyterms")
        if session == self._session_keyterms:
            return
        self._session_keyterms = session
        merged = _merge_keyterms(self._user_keyterms, session)
        self._opts = replace(self._opts, keyterms=merged)
        for stream in self._streams:
            stream.update_options(keyterms=merged)

    async def recognize(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str | Sequence[str]] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        return await super().recognize(
            buffer,
            language=language,  # type: ignore[arg-type]
            conn_options=self._single_attempt(conn_options),
        )

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str | Sequence[str]] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        opts = replace(self._opts)
        if is_given(language):
            opts.language = _normalize_language(language)
            _validate_options(opts)
        lang = opts.language

        wav_bytes = rtc.combine_audio_frames(buffer).to_wav_bytes()

        form_data = aiohttp.FormData()
        form_data.add_field("audio_file", wav_bytes, filename="audio.wav", content_type="audio/wav")
        for name, value in _build_rest_fields(opts, lang):
            form_data.add_field(name, value)

        headers: dict[str, str] = {
            "X-API-Key-ID": self._opts.api_key,
        }

        try:
            async with self._ensure_session().post(
                url=f"{self._opts.base_url}/stt/v3",
                data=form_data,
                headers=headers,
                timeout=aiohttp.ClientTimeout(
                    total=conn_options.timeout,
                    sock_connect=conn_options.timeout,
                ),
            ) as res:
                if res.status != 200:
                    error_text = await res.text()
                    logger.error(f"Gnani STT API error: {res.status} - {error_text}")
                    raise APIStatusError(
                        message=f"Gnani STT API Error ({res.status}): {error_text}",
                        status_code=res.status,
                        body=error_text,
                    )

                response_json = await res.json()
                transcript = response_json.get("transcript", "")
                request_id = response_json.get("request_id", "")
                detected = response_json.get("detected_language_code")

                return stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    request_id=request_id,
                    alternatives=[
                        stt.SpeechData(
                            language=LanguageCode(detected)
                            if detected
                            else _fallback_language(lang),
                            text=transcript,
                            confidence=1.0,
                        )
                    ],
                )

        except asyncio.TimeoutError as e:
            raise APITimeoutError("Gnani STT API request timed out") from e
        except (APIStatusError, APIConnectionError, APITimeoutError):
            raise
        except Exception as e:
            raise APIConnectionError(f"Gnani STT error: {e}") from e

    def stream(
        self,
        *,
        language: NotGivenOr[str | Sequence[str]] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        if not self._opts.use_streaming:
            return cast(
                "SpeechStream",
                super().stream(
                    language=language,  # type: ignore[arg-type]
                    conn_options=conn_options,
                ),
            )

        opts = replace(self._opts)
        if is_given(language):
            opts.language = _normalize_language(language)
            _validate_options(opts)
        stream = SpeechStream(
            stt=self,
            opts=opts,
            conn_options=self._single_attempt(conn_options),
        )
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        pass


class SpeechStream(stt.RecognizeStream):
    """WebSocket-based streaming STT for Gnani Vachana.

    Connects to wss://api.vachana.ai/stt/v3/stream and sends raw PCM audio
    in 1024-byte chunks (512 samples, 16-bit mono).

    The server detects speech itself and returns one final transcript per utterance;
    there are no interim results. The server ends a session after the idle timeout
    (60 s by default) and after 15 minutes. Both are routine: the stream reconnects
    transparently while audio input is still open.
    """

    # defaults at class level so the message handlers work on a partially built stream
    _ws: Any = None
    _speaking: bool = False
    # Reply bookkeeping for the end of the input. The server answers an utterance with one
    # transcript (or error), and answers nothing for audio without speech, so the stream
    # only waits for replies it can expect:
    #  * signal audio was sent that no transcript has answered yet and no commit covers
    _signal_unanswered: bool = False
    #  * commits that closed such audio and whose reply (transcript or error) is outstanding
    _commits_pending: int = 0
    # a language change needs a new handshake: reconnect once the current session closes
    _reconnect_requested: bool = False
    _last_error: str | None = None

    def __init__(
        self,
        *,
        stt: STT,
        opts: GnaniSTTOptions,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(
            stt=stt,
            conn_options=conn_options,
            sample_rate=opts.sample_rate,
        )
        self._opts = opts
        self._control_tasks: set[asyncio.Task[None]] = set()

    def _build_ws_url(self) -> str:
        base = self._opts.base_url
        if base.startswith("https://"):
            ws_base = "wss://" + base[len("https://") :]
        elif base.startswith("http://"):
            ws_base = "ws://" + base[len("http://") :]
        else:
            ws_base = "wss://" + base
        return f"{ws_base}/stt/v3/stream"

    def update_options(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        format: NotGivenOr[str] = NOT_GIVEN,
        keyterms: NotGivenOr[Sequence[str]] = NOT_GIVEN,
    ) -> None:
        """Apply options to this stream: live via ``update_settings`` when connected, and
        otherwise on the next (re)connect. Values are expected to be validated already
        (see :meth:`STT.update_options`).

        ``update_settings`` can only change a single ``language_code``. A language *list*
        is expressed in the handshake (``indic-auto`` plus ``x-language-candidates``), so
        any change that involves a list reconnects with the new handshake.
        """
        settings: dict[str, Any] = {}
        reconnect = False
        if is_given(language) and language != self._opts.language:
            previous = _language_codes(self._opts.language)
            self._opts.language = language
            codes = _language_codes(language)
            if len(previous) == 1 and len(codes) == 1:
                settings["language_code"] = codes[0]
            else:
                reconnect = self._ws is not None
        if is_given(format):
            self._opts.format = format
            settings["format"] = format
        if is_given(keyterms):
            self._opts.keyterms = tuple(keyterms)
            settings["bias"] = {"list": list(keyterms)}

        if settings and self._ws is not None:
            self._send_control({"type": "update_settings", **settings})
        if reconnect:
            self._reconnect_for_new_handshake()

    def _reconnect_for_new_handshake(self) -> None:
        """Close the current session so ``_run`` opens a new one with the updated options.
        An utterance in progress is committed first, so its transcript is not lost."""
        ws = self._ws
        if ws is None or self._reconnect_requested:
            return
        self._reconnect_requested = True

        async def _reconnect() -> None:
            with contextlib.suppress(Exception):
                if self._speaking or self._commits_pending:
                    if self._speaking:
                        await ws.send(json.dumps({"type": "commit"}))
                    deadline = time.monotonic() + min(self._conn_options.timeout, 5.0)
                    while (self._speaking or self._commits_pending) and (
                        time.monotonic() < deadline
                    ):
                        await asyncio.sleep(0.05)
                await ws.close()

        task = asyncio.get_running_loop().create_task(_reconnect())
        self._control_tasks.add(task)
        task.add_done_callback(self._control_tasks.discard)

    def _send_control(self, message: dict[str, Any]) -> None:
        ws = self._ws

        async def _send() -> None:
            with contextlib.suppress(Exception):
                await ws.send(json.dumps(message))

        task = asyncio.get_running_loop().create_task(_send())
        self._control_tasks.add(task)
        task.add_done_callback(self._control_tasks.discard)

    async def _run(self) -> None:
        quick_closes = 0
        while True:
            started = time.monotonic()
            input_finished = await self._run_session()
            if input_finished:
                return

            if self._reconnect_requested:
                # our own doing (new language handshake), not a failure to throttle
                self._reconnect_requested = False
                quick_closes = 0
                logger.debug("Gnani STT reconnecting with updated language options")
                continue

            # The server ended the session (idle timeout, 15 minute limit) while the
            # caller is still sending audio: open a new one. Audio pushed in the
            # meantime waits in the input channel.
            if time.monotonic() - started < _QUICK_CLOSE_S:
                quick_closes += 1
                if quick_closes >= _MAX_QUICK_CLOSES:
                    raise APIConnectionError(
                        "Gnani STT session keeps closing immediately after connecting"
                    )
                await asyncio.sleep(self._conn_options.retry_interval)
            else:
                quick_closes = 0
            logger.debug("Gnani STT session ended by the server, reconnecting")

    async def _run_session(self) -> bool:
        """Run one WebSocket session.

        Returns True when the audio input finished, and False when the server ended the
        session first.
        """
        import websockets

        ws_url = self._build_ws_url()
        headers = _build_ws_headers(self._opts)
        self._last_error = None
        self._signal_unanswered = False
        self._commits_pending = 0
        # this session is built from the current options, so earlier requests are met
        self._reconnect_requested = False

        try:
            async with websockets.connect(
                ws_url,
                additional_headers=headers,
                ping_interval=20,
                ping_timeout=20,
                close_timeout=10,
            ) as ws:
                connected_msg = await asyncio.wait_for(ws.recv(), timeout=10)
                connected_data = json.loads(connected_msg)
                if connected_data.get("type") == "error":
                    # e.g. an invalid header; the server closes the connection next
                    self._handle_error(connected_data)
                elif connected_data.get("type") != "connected":
                    logger.warning(
                        "Unexpected first message from Gnani STT",
                        extra={"lk.pii.data": connected_data},
                    )

                self._ws = ws
                send_task = asyncio.create_task(self._send_audio(ws), name="gnani-stt-send")
                recv_task = asyncio.create_task(self._recv_messages(ws), name="gnani-stt-recv")

                try:
                    done, _ = await asyncio.wait(
                        [send_task, recv_task],
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in done:
                        task.result()

                    if send_task in done:
                        await self._drain_after_input(recv_task)
                        return True
                    return False
                finally:
                    self._ws = None
                    await utils.aio.gracefully_cancel(send_task, recv_task)
                    await self._end_speech_if_open()

        except websockets.exceptions.InvalidStatus as e:
            status = e.response.status_code
            raise APIStatusError(
                f"Gnani STT WebSocket handshake rejected ({status})",
                status_code=status,
                body=e.response.body.decode(errors="replace") if e.response.body else None,
            ) from e
        except websockets.exceptions.ConnectionClosed as e:
            code = e.rcvd.code if e.rcvd is not None else None
            if code == 1008:
                # policy violation: the server rejected the request (bad language, format,
                # context, ...) and sent the reason in an `error` message just before
                reason = self._last_error or (e.rcvd.reason if e.rcvd else "") or "rejected"
                raise APIStatusError(
                    f"Gnani STT rejected the request: {reason}",
                    status_code=400,
                    body=reason,
                ) from e
            raise APIConnectionError(f"Gnani STT WebSocket closed unexpectedly: {e}") from e
        except asyncio.TimeoutError as e:
            raise APITimeoutError("Gnani STT WebSocket connection timed out") from e
        except (APIConnectionError, APIStatusError, APITimeoutError):
            raise
        except Exception as e:
            raise APIConnectionError(f"Gnani STT WebSocket error: {e}") from e

    async def _drain_after_input(self, recv_task: asyncio.Task[None]) -> None:
        """The audio input ended and its last utterance was committed: keep receiving
        until every committed utterance has been answered (a transcript or an error), the
        server ended the session, or the connection timeout runs out.

        Completion is tracked explicitly (``_commits_pending``) rather than inferred from
        speech events, because the server may take a while to answer: a short clip can
        yield ``speech_start`` and the transcript well after the ``commit`` was sent.
        """
        deadline = time.monotonic() + self._conn_options.timeout
        while self._commits_pending > 0 and not recv_task.done():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                # e.g. the last chunk held no speech, so the server never answers
                logger.debug("Gnani STT: no reply for the final utterance before timeout")
                return
            await asyncio.wait({recv_task}, timeout=min(remaining, 0.05))

    def _note_reply(self) -> None:
        """A transcript or an utterance error arrived: it answers one outstanding commit,
        or else audio the server finalised on its own (trailing-silence detection)."""
        if self._commits_pending > 0:
            self._commits_pending -= 1
        else:
            self._signal_unanswered = False

    async def _send_audio(self, ws: Any) -> None:
        audio_buffer = bytearray()

        async def commit() -> None:
            # close the open utterance now instead of waiting for trailing silence
            if self._signal_unanswered:
                # count it before sending, so a fast reply cannot be missed
                self._signal_unanswered = False
                self._commits_pending += 1
            await ws.send(json.dumps({"type": "commit"}))

        async for data in self._input_ch:
            if isinstance(data, self._FlushSentinel):
                if audio_buffer:
                    await ws.send(bytes(audio_buffer))
                    audio_buffer.clear()
                await commit()
                continue

            frame: rtc.AudioFrame = data
            raw_pcm = frame.data.tobytes()
            if _has_signal(raw_pcm):
                self._signal_unanswered = True
            audio_buffer.extend(raw_pcm)

            while len(audio_buffer) >= STREAM_CHUNK_BYTES:
                chunk = bytes(audio_buffer[:STREAM_CHUNK_BYTES])
                audio_buffer = audio_buffer[STREAM_CHUNK_BYTES:]
                await ws.send(chunk)

        if audio_buffer:
            await ws.send(bytes(audio_buffer))
        await commit()

    def _handle_error(self, data: dict[str, Any]) -> None:
        message = data.get("message", "Unknown error")
        if data.get("timeout_reason"):
            # "Session closed: no speech segment received for N seconds."; the server
            # follows up with `session_end` and a normal close. Not a failure.
            logger.debug(f"Gnani STT session timeout: {message}")
            return
        # Utterance-level problems (e.g. "Transcription failed or audio unclear") leave
        # the connection open; handshake problems are followed by a 1008 close, where
        # the remembered message becomes the error.
        logger.warning(f"Gnani STT error: {message}")
        self._last_error = message
        self._note_reply()
        self._end_speech_nowait()

    def _end_speech_nowait(self) -> None:
        if self._speaking:
            self._speaking = False
            self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH))

    async def _end_speech_if_open(self) -> None:
        self._end_speech_nowait()

    async def _recv_messages(self, ws: Any) -> None:
        async for msg in ws:
            if isinstance(msg, bytes):
                continue

            data = json.loads(msg)
            msg_type = data.get("type", "")

            if msg_type == "transcript":
                self._mark_provider_progress()
                # an utterance has been answered (even with an empty text)
                self._note_reply()
                # partials are disabled, so every transcript is final; ignore any other
                if not data.get("is_final", True):
                    continue
                text = data.get("text", "")
                if text:
                    detected = data.get("detected_language_code")
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(
                            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                            request_id=data.get("segment_id", ""),
                            alternatives=[
                                stt.SpeechData(
                                    language=LanguageCode(detected)
                                    if detected
                                    else _fallback_language(self._opts.language),
                                    text=text,
                                    confidence=1.0,
                                )
                            ],
                        )
                    )
                # The server sends no end-of-speech message: the final transcript is
                # what closes the utterance, which STT-based turn detection needs.
                self._end_speech_nowait()

            elif msg_type in ("speech_start", "vad_start"):
                self._mark_provider_progress()
                if not self._speaking:
                    self._speaking = True
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)
                    )

            elif msg_type in ("speech_end", "vad_end"):
                # not sent by the current API; handled in case it is added
                self._mark_provider_progress()
                self._end_speech_nowait()

            elif msg_type == "processing":
                self._mark_provider_progress()

            elif msg_type == "session_end":
                logger.debug(f"Gnani STT session ended: {data.get('reason')}")
                return

            elif msg_type == "error":
                self._handle_error(data)

            elif msg_type == "settings_updated":
                logger.debug("Gnani STT settings updated")

    async def aclose(self) -> None:
        await super().aclose()
        for task in list(self._control_tasks):
            task.cancel()
