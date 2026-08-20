# Copyright 2026 LiveKit, Inc.
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

import asyncio
import base64
import binascii
import codecs
import copy
import json
import math
import os
import weakref
from collections.abc import AsyncIterable, AsyncIterator, Callable, Collection
from dataclasses import dataclass, replace
from typing import Any, Literal
from urllib.parse import urlsplit

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
    tokenize,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given
from livekit.agents.voice.io import TimedString

from .log import logger
from .models import (
    COMPRESSED_AUDIO_FORMATS,
    CONTEXT_TEXT_RESOURCE_IDS,
    EMOTION_SCALE_RANGE,
    ICL_2_RESOURCE_IDS,
    LOUDNESS_RATE_RANGE,
    MIN_CUSTOM_BIT_RATE,
    MIN_DEFAULT_BIT_RATE,
    PITCH_RANGE,
    SILENCE_DURATION_RANGE_MS,
    SPEECH_RATE_RANGE,
    SUBTITLE_RESOURCE_IDS,
    SUPPORTED_AUDIO_FORMATS,
    SUPPORTED_CONTEXT_LANGUAGES,
    SUPPORTED_EXPLICIT_DIALECTS,
    SUPPORTED_EXPLICIT_LANGUAGES,
    SUPPORTED_LATEX_PARSERS,
    SUPPORTED_PARENTHESIS_FILTER_LENGTHS,
    SUPPORTED_SAMPLE_RATES,
    TIMESTAMP_RESOURCE_IDS,
    UNSUPPORTED_CHAR_RATIO_RANGE,
    AIGCMetadata,
    TTSAudioFormat,
    TTSContextLanguage,
    TTSEmotion,
    TTSExplicitDialect,
    TTSExplicitLanguage,
    TTSLatexParser,
    TTSModel,
    TTSParenthesisFilterLength,
    TTSSampleRate,
    TTSSpeakerModel,
    TTSUsage,
    TTSVoice,
)
from .version import __version__

DEFAULT_BASE_URL = "https://voice.ap-southeast-1.bytepluses.com"
DEFAULT_RESOURCE_ID: TTSModel = "seed-tts-2.0"
DEFAULT_VOICE_ID: TTSVoice = "zh_female_vv_uranus_bigtts"
# Public protocol identifier; authentication uses api_key or the legacy app/access key pair.
DEFAULT_API_APP_KEY = "aGjiRDfUWi"
NUM_CHANNELS = 1
USER_AGENT = f"livekit-plugins-byteplus/{__version__}"

_DEFAULT_PATH = "/api/v3/tts/unidirectional"
_PCM_SAMPLE_WIDTH_BYTES = 2
_MAX_STREAM_BUFFER_CHARS = 32 * 1024 * 1024
_MAX_ERROR_BODY_CHARS = 4096
_MAX_EXTRA_ADDITIONS_CHARS = 1024 * 1024
_OPTIONAL_ADDITION_FIELDS = (
    "silence_duration",
    "enable_language_detector",
    "explicit_language",
    "context_language",
    "explicit_dialect",
    "context_texts",
    "section_id",
    "disable_markdown_filter",
    "disable_emoji_filter",
    "enable_latex_tn",
    "latex_parser",
    "max_length_to_filter_parenthesis",
    "unsupported_char_ratio_thresh",
    "tone_fidelity",
    "disable_default_bit_rate",
    "aigc_watermark",
)


@dataclass(slots=True)
class _TTSOptions:
    api_key: str | None
    app_id: str | None
    access_key: str | None
    api_app_key: str
    resource_id: TTSModel | str
    base_url: str
    request_timeout: float
    voice: TTSVoice | str
    speaker_model: TTSSpeakerModel | str | None
    include_usage: bool
    audio_format: TTSAudioFormat
    sample_rate: TTSSampleRate
    bit_rate: int | None
    emotion: TTSEmotion | str | None
    emotion_scale: int | None
    speech_rate: int
    loudness_rate: int
    pitch: int
    user_id: str | None
    silence_duration: int | None
    enable_timestamp: bool | None
    enable_subtitle: bool | None
    enable_language_detector: bool | None
    explicit_language: TTSExplicitLanguage | None
    context_language: TTSContextLanguage | None
    explicit_dialect: TTSExplicitDialect | None
    context_texts: list[str] | None
    section_id: str | None
    disable_markdown_filter: bool | None
    disable_emoji_filter: bool | None
    enable_latex_tn: bool | None
    latex_parser: TTSLatexParser | None
    max_length_to_filter_parenthesis: TTSParenthesisFilterLength | None
    unsupported_char_ratio_thresh: float | None
    tone_fidelity: bool | None
    disable_default_bit_rate: bool | None
    use_cache: bool | None
    aigc_watermark: bool | None
    aigc_metadata: AIGCMetadata | None
    extra_additions: dict[str, Any] | None


class TTS(tts.TTS[Literal["usage_collected"]]):
    """LiveKit TTS provider for BytePlus HTTP streaming synthesis."""

    def __init__(
        self,
        *,
        voice: TTSVoice | str = DEFAULT_VOICE_ID,
        model: TTSModel | str = DEFAULT_RESOURCE_ID,
        api_key: str | None = None,
        app_id: str | None = None,
        access_key: str | None = None,
        api_app_key: str = DEFAULT_API_APP_KEY,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        request_timeout: float = 60.0,
        speaker_model: TTSSpeakerModel | str | None = None,
        include_usage: bool = False,
        audio_format: TTSAudioFormat = "pcm",
        sample_rate: TTSSampleRate = 24000,
        bit_rate: int | None = None,
        emotion: TTSEmotion | str | None = None,
        emotion_scale: int | None = None,
        speech_rate: int = 0,
        loudness_rate: int = 0,
        pitch: int = 0,
        user_id: str | None = None,
        silence_duration: int | None = None,
        enable_timestamp: bool | None = None,
        enable_subtitle: bool | None = None,
        enable_language_detector: bool | None = None,
        explicit_language: TTSExplicitLanguage | None = None,
        context_language: TTSContextLanguage | None = None,
        explicit_dialect: TTSExplicitDialect | None = None,
        context_texts: list[str] | None = None,
        section_id: str | None = None,
        disable_markdown_filter: bool | None = None,
        disable_emoji_filter: bool | None = None,
        enable_latex_tn: bool | None = None,
        latex_parser: TTSLatexParser | None = None,
        max_length_to_filter_parenthesis: TTSParenthesisFilterLength | None = None,
        unsupported_char_ratio_thresh: float | None = None,
        tone_fidelity: bool | None = None,
        disable_default_bit_rate: bool | None = None,
        use_cache: bool | None = None,
        aigc_watermark: bool | None = None,
        aigc_metadata: AIGCMetadata | None = None,
        extra_additions: dict[str, Any] | None = None,
        tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """Create a unidirectional HTTP streaming TTS provider.

        Args:
            voice: Provider speaker ID. Public and account-specific cloned voices are accepted.
            model: Resource ID sent in ``X-Api-Resource-Id``.
            api_key: Recommended API key. Falls back to ``BYTEPLUS_API_KEY``.
            app_id: Legacy BytePlus App ID. Must be used together with ``access_key``.
            access_key: Legacy BytePlus access key. Must be used together with ``app_id``.
            api_app_key: Public BytePlus protocol identifier. This is not an account
                credential and normally does not need to be changed.
            base_url: HTTPS API origin. Defaults to the BytePlus endpoint.
            request_timeout: Maximum duration of one HTTP synthesis request in seconds.
            speaker_model: Model used by an ICL 2.0 cloned voice.
            include_usage: Request billable character usage and emit ``usage_collected``.
            audio_format: ``pcm``, ``mp3``, or ``ogg_opus``. PCM is recommended for streaming.
            sample_rate: Output sample rate in Hz.
            bit_rate: MP3 or OGG Opus bit rate in bits per second.
            emotion: Emotion label supported by the selected voice.
            emotion_scale: Emotion strength from 1 to 5. Requires ``emotion``.
            speech_rate: Speech rate from -50 (0.5x) to 100 (2.0x).
            loudness_rate: Loudness from -50 (0.5x) to 100 (2.0x).
            pitch: Post-process pitch from -12 to 12.
            user_id: Optional provider user identifier.
            silence_duration: Silence appended to each provider request, from 0 to
                30000 ms. In LiveKit streaming mode, each tokenized sentence is a request.
            enable_timestamp: Word/phoneme timestamps for TTS/ICL 1.0.
            enable_subtitle: Word/phoneme timestamps for TTS/ICL 2.0.
            enable_language_detector: Enable automatic language detection.
            explicit_language: Restrict synthesis to one documented language.
            context_language: Reference language for Western European synthesis.
            explicit_dialect: Chinese dialect requested from a compatible voice.
            context_texts: TTS/ICL 2.0 style instructions. Empty strings are rejected.
            section_id: Identifier used to preserve semantics across related requests.
            disable_markdown_filter: Despite its provider name, ``True`` parses and removes
                Markdown syntax before synthesis.
            disable_emoji_filter: ``True`` preserves emoji; ``False`` filters emoji.
            enable_latex_tn: Enable LaTeX reading. Requires ``disable_markdown_filter=True``.
            latex_parser: Enhanced LaTeX parser. Currently only ``v2``.
            max_length_to_filter_parenthesis: ``0`` disables parenthesis filtering and
                ``100`` enables the documented filtering behavior.
            unsupported_char_ratio_thresh: Unsupported-character threshold from 0 to 1.
            tone_fidelity: Preserve ICL 2.0 training voice style. Same-language synthesis only.
            disable_default_bit_rate: Must be ``True`` for bit rates below 64000.
            use_cache: Enable the provider's one-hour text cache. Cached responses contain
                no timestamps.
            aigc_watermark: Append the audible AIGC rhythm watermark.
            aigc_metadata: Metadata watermark for MP3 or OGG Opus output.
            extra_additions: Forward-compatible fields merged into ``req_params.additions``.
                Explicit constructor options take precedence over colliding keys.
            tokenizer: Sentence tokenizer used to adapt LiveKit text streaming to the
                provider's request-per-sentence HTTP API.
            http_session: Optional caller-managed ``aiohttp.ClientSession``.

        Raises:
            ValueError: If authentication, an individual option, or an option combination
                violates the documented provider contract.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=True,
                aligned_transcript=enable_timestamp is True or enable_subtitle is True,
            ),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )

        opts = _TTSOptions(
            api_key=api_key or os.getenv("BYTEPLUS_API_KEY"),
            app_id=app_id or os.getenv("BYTEPLUS_APP_ID"),
            access_key=access_key or os.getenv("BYTEPLUS_ACCESS_KEY"),
            api_app_key=api_app_key,
            resource_id=model,
            base_url=_resolve_base_url(base_url),
            request_timeout=request_timeout,
            voice=voice,
            speaker_model=speaker_model,
            include_usage=include_usage,
            audio_format=audio_format,
            sample_rate=sample_rate,
            bit_rate=bit_rate,
            emotion=emotion,
            emotion_scale=emotion_scale,
            speech_rate=speech_rate,
            loudness_rate=loudness_rate,
            pitch=pitch,
            user_id=user_id,
            silence_duration=silence_duration,
            enable_timestamp=enable_timestamp,
            enable_subtitle=enable_subtitle,
            enable_language_detector=enable_language_detector,
            explicit_language=explicit_language,
            context_language=context_language,
            explicit_dialect=explicit_dialect,
            context_texts=copy.deepcopy(context_texts),
            section_id=section_id,
            disable_markdown_filter=disable_markdown_filter,
            disable_emoji_filter=disable_emoji_filter,
            enable_latex_tn=enable_latex_tn,
            latex_parser=latex_parser,
            max_length_to_filter_parenthesis=max_length_to_filter_parenthesis,
            unsupported_char_ratio_thresh=unsupported_char_ratio_thresh,
            tone_fidelity=tone_fidelity,
            disable_default_bit_rate=disable_default_bit_rate,
            use_cache=use_cache,
            aigc_watermark=aigc_watermark,
            aigc_metadata=aigc_metadata,
            extra_additions=copy.deepcopy(extra_additions),
        )
        self._opts = _validate_options(opts)
        self._opts.base_url = self._opts.base_url.rstrip("/")
        self._session = http_session
        self._sentence_tokenizer = (
            tokenizer if is_given(tokenizer) else tokenize.blingfire.SentenceTokenizer()
        )
        self._streams = weakref.WeakSet[SynthesizeStream]()

    @property
    def model(self) -> str:
        """Return the resource ID used for LiveKit telemetry."""
        return self._opts.resource_id

    @property
    def provider(self) -> str:
        """Return the provider name."""
        return "BytePlus"

    def _ensure_session(self) -> aiohttp.ClientSession:
        """Return the injected session or LiveKit's job-scoped shared session."""
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def update_options(
        self,
        *,
        voice: NotGivenOr[TTSVoice | str] = NOT_GIVEN,
        model: NotGivenOr[TTSModel | str] = NOT_GIVEN,
        speaker_model: NotGivenOr[TTSSpeakerModel | str | None] = NOT_GIVEN,
        include_usage: NotGivenOr[bool] = NOT_GIVEN,
        audio_format: NotGivenOr[TTSAudioFormat] = NOT_GIVEN,
        sample_rate: NotGivenOr[TTSSampleRate] = NOT_GIVEN,
        bit_rate: NotGivenOr[int | None] = NOT_GIVEN,
        emotion: NotGivenOr[TTSEmotion | str | None] = NOT_GIVEN,
        emotion_scale: NotGivenOr[int | None] = NOT_GIVEN,
        speech_rate: NotGivenOr[int] = NOT_GIVEN,
        loudness_rate: NotGivenOr[int] = NOT_GIVEN,
        pitch: NotGivenOr[int] = NOT_GIVEN,
        user_id: NotGivenOr[str | None] = NOT_GIVEN,
        silence_duration: NotGivenOr[int | None] = NOT_GIVEN,
        enable_timestamp: NotGivenOr[bool | None] = NOT_GIVEN,
        enable_subtitle: NotGivenOr[bool | None] = NOT_GIVEN,
        enable_language_detector: NotGivenOr[bool | None] = NOT_GIVEN,
        explicit_language: NotGivenOr[TTSExplicitLanguage | None] = NOT_GIVEN,
        context_language: NotGivenOr[TTSContextLanguage | None] = NOT_GIVEN,
        explicit_dialect: NotGivenOr[TTSExplicitDialect | None] = NOT_GIVEN,
        context_texts: NotGivenOr[list[str] | None] = NOT_GIVEN,
        section_id: NotGivenOr[str | None] = NOT_GIVEN,
        disable_markdown_filter: NotGivenOr[bool | None] = NOT_GIVEN,
        disable_emoji_filter: NotGivenOr[bool | None] = NOT_GIVEN,
        enable_latex_tn: NotGivenOr[bool | None] = NOT_GIVEN,
        latex_parser: NotGivenOr[TTSLatexParser | None] = NOT_GIVEN,
        max_length_to_filter_parenthesis: NotGivenOr[TTSParenthesisFilterLength | None] = NOT_GIVEN,
        unsupported_char_ratio_thresh: NotGivenOr[float | None] = NOT_GIVEN,
        tone_fidelity: NotGivenOr[bool | None] = NOT_GIVEN,
        disable_default_bit_rate: NotGivenOr[bool | None] = NOT_GIVEN,
        use_cache: NotGivenOr[bool | None] = NOT_GIVEN,
        aigc_watermark: NotGivenOr[bool | None] = NOT_GIVEN,
        aigc_metadata: NotGivenOr[AIGCMetadata | None] = NOT_GIVEN,
        extra_additions: NotGivenOr[dict[str, Any] | None] = NOT_GIVEN,
    ) -> None:
        """Atomically update synthesis options used by subsequently created streams.

        Every supplied value is applied to a new configuration snapshot and the complete
        cross-parameter contract is validated before replacing the active configuration.
        Existing streams retain the snapshot captured when they were created.

        Args:
            voice: New speaker ID.
            model: New provider resource ID.
            speaker_model: New ICL 2.0 cloned-voice model, or ``None`` to clear it.
            include_usage: Enable or disable billable-character usage events.
            audio_format: New PCM, MP3, or OGG Opus output format.
            sample_rate: New supported output sample rate.
            bit_rate: New MP3/OGG bit rate, or ``None`` to use the provider default.
            emotion: New voice-supported emotion, or ``None``.
            emotion_scale: New emotion strength from 1 to 5, or ``None``.
            speech_rate: New speech rate from -50 to 100.
            loudness_rate: New loudness from -50 to 100.
            pitch: New post-process pitch from -12 to 12.
            user_id: New provider user ID, or ``None``.
            silence_duration: New per-request trailing silence in milliseconds, or ``None``.
            enable_timestamp: Toggle TTS/ICL 1.0 aligned timestamps.
            enable_subtitle: Toggle TTS/ICL 2.0 aligned timestamps.
            enable_language_detector: Toggle automatic language detection.
            explicit_language: New language restriction, or ``None``.
            context_language: New Western-European reference language, or ``None``.
            explicit_dialect: New Chinese dialect, or ``None``.
            context_texts: New TTS/ICL 2.0 style instructions, or ``None``.
            section_id: New cross-request section ID, or ``None``.
            disable_markdown_filter: Toggle provider Markdown processing.
            disable_emoji_filter: Toggle provider emoji preservation.
            enable_latex_tn: Toggle LaTeX text normalization.
            latex_parser: Select the enhanced LaTeX parser, or ``None``.
            max_length_to_filter_parenthesis: Set documented 0/100 filtering, or ``None``.
            unsupported_char_ratio_thresh: New threshold from 0 to 1, or ``None``.
            tone_fidelity: Toggle ICL 2.0 same-language tone fidelity.
            disable_default_bit_rate: Allow or disallow bit rates below 64000.
            use_cache: Toggle the one-hour text cache.
            aigc_watermark: Toggle the audible AIGC rhythm marker.
            aigc_metadata: New MP3/OGG metadata watermark, or ``None``.
            extra_additions: New forward-compatible additions, or ``None``.

        Raises:
            ValueError: If the resulting configuration violates any constructor rule,
                including cache/timestamp incompatibility or model-specific requirements.
        """
        updates: dict[str, Any] = {
            field_name: copy.deepcopy(value)
            for field_name, value in (
                ("voice", voice),
                ("resource_id", model),
                ("speaker_model", speaker_model),
                ("include_usage", include_usage),
                ("audio_format", audio_format),
                ("sample_rate", sample_rate),
                ("bit_rate", bit_rate),
                ("emotion", emotion),
                ("emotion_scale", emotion_scale),
                ("speech_rate", speech_rate),
                ("loudness_rate", loudness_rate),
                ("pitch", pitch),
                ("user_id", user_id),
                ("silence_duration", silence_duration),
                ("enable_timestamp", enable_timestamp),
                ("enable_subtitle", enable_subtitle),
                ("enable_language_detector", enable_language_detector),
                ("explicit_language", explicit_language),
                ("context_language", context_language),
                ("explicit_dialect", explicit_dialect),
                ("context_texts", context_texts),
                ("section_id", section_id),
                ("disable_markdown_filter", disable_markdown_filter),
                ("disable_emoji_filter", disable_emoji_filter),
                ("enable_latex_tn", enable_latex_tn),
                ("latex_parser", latex_parser),
                ("max_length_to_filter_parenthesis", max_length_to_filter_parenthesis),
                ("unsupported_char_ratio_thresh", unsupported_char_ratio_thresh),
                ("tone_fidelity", tone_fidelity),
                ("disable_default_bit_rate", disable_default_bit_rate),
                ("use_cache", use_cache),
                ("aigc_watermark", aigc_watermark),
                ("aigc_metadata", aigc_metadata),
                ("extra_additions", extra_additions),
            )
            if is_given(value)
        }

        self._opts = _validate_options(replace(self._opts, **updates))
        self._sample_rate = self._opts.sample_rate
        self._capabilities.aligned_transcript = (
            self._opts.enable_timestamp is True or self._opts.enable_subtitle is True
        )

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        """Synthesize one complete input string with LiveKit retry options."""
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        """Create LiveKit text streaming adapted to sentence-level HTTP requests."""
        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        """Close every active streaming synthesis object."""
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()


class ChunkedStream(tts.ChunkedStream):
    """Single-request LiveKit stream for a complete input string."""

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=NUM_CHANNELS,
            mime_type=_mime_type(self._opts.audio_format),
        )

        text = self._input_text.strip()
        if not text:
            return

        await _request_and_emit_audio(
            provider=self._tts,
            session=self._tts._ensure_session(),
            opts=self._opts,
            text=text,
            output_emitter=output_emitter,
            conn_options=self._conn_options,
            request_id=request_id,
            transcript_offset=0.0,
        )
        output_emitter.flush()


class SynthesizeStream(tts.SynthesizeStream):
    """Sentence-tokenized adapter over BytePlus's unidirectional HTTP API."""

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        decode_compressed_audio = self._opts.audio_format != "pcm"
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm"
            if decode_compressed_audio
            else _mime_type(self._opts.audio_format),
            stream=True,
        )

        sentence_stream = self._tts._sentence_tokenizer.stream()

        async def _input_task() -> None:
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    sentence_stream.flush()
                    continue
                sentence_stream.push_text(data)
            sentence_stream.end_input()

        async def _synthesis_task() -> None:
            segment_started = False
            transcript_offset = 0.0
            async for sentence in sentence_stream:
                text = sentence.token.strip()
                if not text:
                    continue
                self._mark_started()
                if not segment_started:
                    output_emitter.start_segment(segment_id=request_id)
                    segment_started = True
                request_timeline_duration = await _request_and_emit_audio(
                    provider=self._tts,
                    session=self._tts._ensure_session(),
                    opts=self._opts,
                    text=text,
                    output_emitter=output_emitter,
                    conn_options=self._conn_options,
                    request_id=utils.shortuuid(),
                    transcript_offset=transcript_offset,
                    decode_compressed_audio=decode_compressed_audio,
                )
                transcript_offset += request_timeline_duration
                output_emitter.flush()
            output_emitter.end_input()

        tasks = [asyncio.create_task(_input_task()), asyncio.create_task(_synthesis_task())]
        try:
            await asyncio.gather(*tasks)
        finally:
            await utils.aio.gracefully_cancel(*tasks)
            await sentence_stream.aclose()


async def _request_and_emit_audio(
    *,
    provider: TTS,
    session: aiohttp.ClientSession,
    opts: _TTSOptions,
    text: str,
    output_emitter: tts.AudioEmitter,
    conn_options: APIConnectOptions,
    request_id: str,
    transcript_offset: float,
    decode_compressed_audio: bool = False,
) -> float:
    """Execute one provider request and emit audio, timestamps, and usage.

    Returns:
        The request timeline duration, measured from emitted audio and extended to the
        largest provider word timestamp when necessary.
    """
    url = f"{opts.base_url}{_DEFAULT_PATH}"
    headers = _build_headers(opts, request_id=request_id)
    payload = _build_request_payload(opts, text=text)
    audio_decoder: utils.codecs.AudioStreamDecoder | None = None
    decode_task: asyncio.Task[None] | None = None
    audio_sink: Callable[[bytes], None] = output_emitter.push
    received_compressed_bytes = 0
    decoded_audio_frames = 0
    emitted_audio_duration = 0.0

    if decode_compressed_audio:
        audio_decoder = utils.codecs.AudioStreamDecoder(
            sample_rate=opts.sample_rate,
            num_channels=NUM_CHANNELS,
            format=_mime_type(opts.audio_format),
        )

        def _push_compressed_audio(data: bytes) -> None:
            nonlocal received_compressed_bytes
            assert audio_decoder is not None
            received_compressed_bytes += len(data)
            audio_decoder.push(data)

        audio_sink = _push_compressed_audio

        async def _forward_decoded_audio() -> None:
            nonlocal decoded_audio_frames, emitted_audio_duration
            assert audio_decoder is not None
            async for frame in audio_decoder:
                decoded_audio_frames += 1
                emitted_audio_duration += frame.duration
                output_emitter.push(bytes(frame.data))

        decode_task = asyncio.create_task(_forward_decoded_audio())
    elif opts.audio_format == "pcm":

        def _push_pcm_audio(data: bytes) -> None:
            nonlocal emitted_audio_duration
            emitted_audio_duration += len(data) / (
                opts.sample_rate * NUM_CHANNELS * _PCM_SAMPLE_WIDTH_BYTES
            )
            output_emitter.push(data)

        audio_sink = _push_pcm_audio

    async def _finish_audio_decoder(*, validate_output: bool) -> None:
        nonlocal audio_decoder, decode_task
        if audio_decoder is None:
            return
        audio_decoder.end_input()
        if decode_task is not None:
            await decode_task
        await audio_decoder.aclose()
        audio_decoder = None
        decode_task = None
        if validate_output and received_compressed_bytes > 0 and decoded_audio_frames == 0:
            raise APIConnectionError(
                "BytePlus TTS returned compressed audio that could not be decoded. "
                f"Provider request ID: {request_id}"
            )

    logger.debug(
        "starting BytePlus TTS request",
        extra={
            "request_id": request_id,
            "model": opts.resource_id,
            "audio_format": opts.audio_format,
        },
    )

    try:
        async with session.post(
            url,
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(
                total=opts.request_timeout,
                sock_connect=conn_options.timeout,
            ),
            read_bufsize=10 * 1024 * 1024,
        ) as resp:
            log_id = resp.headers.get("X-Tt-Logid") if hasattr(resp, "headers") else None
            provider_request_id = log_id or request_id
            output_emitter._note_provider_request_id(provider_request_id)
            status = getattr(resp, "status", 200)
            if status >= 400:
                body = await _read_error_body(resp)
                raise APIStatusError(
                    message=(
                        f"BytePlus TTS HTTP request failed with status {status}. "
                        f"Provider request ID: {provider_request_id}"
                    ),
                    status_code=status,
                    request_id=provider_request_id,
                    body=body,
                    retryable=status in {408, 429} or status >= 500,
                )
            request_transcript_end = 0.0
            async for event in _iter_json_events(resp.content.iter_any()):
                is_terminal, event_transcript_end = _handle_response_event(
                    provider=provider,
                    event=event,
                    output_emitter=output_emitter,
                    request_id=provider_request_id,
                    transcript_offset=transcript_offset,
                    audio_sink=audio_sink,
                )
                request_transcript_end = max(request_transcript_end, event_transcript_end)
                if is_terminal:
                    await _finish_audio_decoder(validate_output=True)
                    logger.debug(
                        "completed BytePlus TTS request",
                        extra={"request_id": provider_request_id},
                    )
                    return max(emitted_audio_duration, request_transcript_end)
            raise APIConnectionError(
                "BytePlus TTS response ended before the terminal event. "
                f"Provider request ID: {provider_request_id}",
            )

    except asyncio.TimeoutError:
        raise APITimeoutError(
            f"BytePlus TTS request timed out after {opts.request_timeout:g} seconds"
        ) from None
    except APIError:
        raise
    except aiohttp.ClientResponseError as e:
        raise APIStatusError(
            message=f"BytePlus TTS HTTP {e.status}: {e.message}",
            status_code=e.status,
            request_id=request_id,
            body=None,
        ) from e
    except (aiohttp.ClientError, OSError) as e:
        raise APIConnectionError(f"BytePlus TTS connection failed: {type(e).__name__}: {e}") from e
    finally:
        await _finish_audio_decoder(validate_output=False)


async def _read_error_body(resp: Any) -> str | None:
    """Read a bounded HTTP error body without masking the original status."""
    try:
        body = await resp.text()
    except Exception:
        return None
    if not isinstance(body, str):
        return None
    return body[:_MAX_ERROR_BODY_CHARS]


def _resolve_base_url(base_url: NotGivenOr[str]) -> str:
    """Resolve the explicit or configured BytePlus endpoint."""
    if is_given(base_url):
        return base_url
    return os.getenv("BYTEPLUS_TTS_BASE_URL") or DEFAULT_BASE_URL


def _validate_options(opts: _TTSOptions) -> _TTSOptions:
    """Validate individual values and every deterministic cross-parameter rule."""
    if bool(opts.app_id) != bool(opts.access_key) and not opts.api_key:
        raise ValueError("legacy authentication requires both app_id and access_key")
    if not opts.api_key and not (opts.app_id and opts.access_key):
        raise ValueError(
            "BytePlus TTS authentication is required. Provide api_key or set "
            "BYTEPLUS_API_KEY; alternatively provide "
            "both app_id and access_key for legacy authentication."
        )

    for name, value in (
        ("api_key", opts.api_key),
        ("app_id", opts.app_id),
        ("access_key", opts.access_key),
    ):
        if value is not None:
            _validate_non_empty_string(name, value)
    _validate_non_empty_string("voice", opts.voice)
    _validate_non_empty_string("model/resource_id", opts.resource_id)
    _validate_non_empty_string("api_app_key", opts.api_app_key)
    _validate_base_url(opts.base_url)
    _validate_number_range("request_timeout", opts.request_timeout, (0.001, float("inf")))

    if opts.speaker_model is not None:
        _validate_non_empty_string("speaker_model", opts.speaker_model)
    if opts.user_id is not None:
        _validate_non_empty_string("user_id", opts.user_id)
    if opts.section_id is not None:
        _validate_non_empty_string("section_id", opts.section_id)

    _validate_choice("audio_format", opts.audio_format, SUPPORTED_AUDIO_FORMATS)
    _validate_choice("sample_rate", opts.sample_rate, SUPPORTED_SAMPLE_RATES)
    if opts.explicit_language is not None:
        _validate_choice("explicit_language", opts.explicit_language, SUPPORTED_EXPLICIT_LANGUAGES)
    if opts.context_language is not None:
        _validate_choice("context_language", opts.context_language, SUPPORTED_CONTEXT_LANGUAGES)
    if opts.explicit_dialect is not None:
        _validate_choice("explicit_dialect", opts.explicit_dialect, SUPPORTED_EXPLICIT_DIALECTS)
    if opts.latex_parser is not None:
        _validate_choice("latex_parser", opts.latex_parser, SUPPORTED_LATEX_PARSERS)
    if opts.max_length_to_filter_parenthesis is not None:
        _validate_choice(
            "max_length_to_filter_parenthesis",
            opts.max_length_to_filter_parenthesis,
            SUPPORTED_PARENTHESIS_FILTER_LENGTHS,
        )
    _validate_int_range("speech_rate", opts.speech_rate, SPEECH_RATE_RANGE)
    _validate_int_range("loudness_rate", opts.loudness_rate, LOUDNESS_RATE_RANGE)
    _validate_int_range("pitch", opts.pitch, PITCH_RANGE)
    if opts.emotion_scale is not None:
        _validate_int_range("emotion_scale", opts.emotion_scale, EMOTION_SCALE_RANGE)
    if opts.silence_duration is not None:
        _validate_int_range(
            "silence_duration",
            opts.silence_duration,
            SILENCE_DURATION_RANGE_MS,
            unit=" ms",
        )
    if opts.unsupported_char_ratio_thresh is not None:
        _validate_number_range(
            "unsupported_char_ratio_thresh",
            opts.unsupported_char_ratio_thresh,
            UNSUPPORTED_CHAR_RATIO_RANGE,
        )
    if opts.bit_rate is not None:
        _validate_int_range("bit_rate", opts.bit_rate, (MIN_CUSTOM_BIT_RATE, 2**31 - 1))

    _validate_bool("include_usage", opts.include_usage)
    for name in (
        "enable_timestamp",
        "enable_subtitle",
        "enable_language_detector",
        "disable_markdown_filter",
        "disable_emoji_filter",
        "enable_latex_tn",
        "tone_fidelity",
        "disable_default_bit_rate",
        "use_cache",
        "aigc_watermark",
    ):
        _validate_optional_bool(name, getattr(opts, name))

    if opts.context_texts is not None and (
        not isinstance(opts.context_texts, list)
        or not opts.context_texts
        or any(not isinstance(text, str) or not text.strip() for text in opts.context_texts)
    ):
        raise ValueError("context_texts must be a non-empty list of non-empty strings")
    if opts.extra_additions is not None:
        if not isinstance(opts.extra_additions, dict):
            raise ValueError("extra_additions must be a dictionary or None")
        try:
            serialized_additions = json.dumps(opts.extra_additions)
        except (TypeError, ValueError, RecursionError) as e:
            raise ValueError("extra_additions must contain JSON-serializable values") from e
        if len(serialized_additions) > _MAX_EXTRA_ADDITIONS_CHARS:
            raise ValueError(
                f"extra_additions must not exceed {_MAX_EXTRA_ADDITIONS_CHARS} serialized characters"
            )
    if opts.aigc_metadata is not None and not isinstance(opts.aigc_metadata, AIGCMetadata):
        raise ValueError("aigc_metadata must be an AIGCMetadata instance or None")
    if opts.aigc_metadata is not None:
        _validate_bool("aigc_metadata.enable", opts.aigc_metadata.enable)
        for name in (
            "content_producer",
            "produce_id",
            "content_propagator",
            "propagate_id",
        ):
            value = getattr(opts.aigc_metadata, name)
            if value is not None:
                _validate_non_empty_string(f"aigc_metadata.{name}", value)

    if opts.emotion_scale is not None and opts.emotion is None:
        raise ValueError("emotion_scale requires emotion to be set")
    if opts.bit_rate is not None and opts.audio_format not in COMPRESSED_AUDIO_FORMATS:
        raise ValueError("bit_rate is only supported with audio_format='mp3' or 'ogg_opus'")
    if (
        opts.bit_rate is not None
        and opts.bit_rate < MIN_DEFAULT_BIT_RATE
        and opts.disable_default_bit_rate is not True
    ):
        raise ValueError(
            f"bit_rate below {MIN_DEFAULT_BIT_RATE} requires disable_default_bit_rate=True"
        )
    if opts.enable_timestamp is True and opts.enable_subtitle is True:
        raise ValueError("enable_timestamp and enable_subtitle cannot both be enabled")
    if opts.enable_timestamp is True and opts.resource_id not in TIMESTAMP_RESOURCE_IDS:
        raise ValueError("enable_timestamp is only supported by TTS 1.0 and ICL 1.0 resource IDs")
    if opts.enable_subtitle is True and opts.resource_id not in SUBTITLE_RESOURCE_IDS:
        raise ValueError("enable_subtitle is only supported by TTS 2.0 and ICL 2.0 resource IDs")
    if opts.context_texts is not None and opts.resource_id not in CONTEXT_TEXT_RESOURCE_IDS:
        raise ValueError("context_texts is only supported by TTS 2.0 and ICL 2.0 resource IDs")
    if opts.context_texts is not None and opts.speaker_model == "seed-tts-2.0-standard":
        raise ValueError("context_texts is not supported by speaker_model='seed-tts-2.0-standard'")
    if opts.speaker_model is not None and opts.resource_id not in ICL_2_RESOURCE_IDS:
        raise ValueError("speaker_model is only supported with model='seed-icl-2.0'")
    if opts.tone_fidelity is True and opts.resource_id not in ICL_2_RESOURCE_IDS:
        raise ValueError("tone_fidelity is only supported with model='seed-icl-2.0'")
    if (opts.enable_latex_tn is True or opts.latex_parser is not None) and (
        opts.disable_markdown_filter is not True
    ):
        raise ValueError("enable_latex_tn and latex_parser require disable_markdown_filter=True")
    if opts.use_cache is True and (opts.enable_timestamp is True or opts.enable_subtitle is True):
        raise ValueError("cached responses do not include timestamps or subtitles")
    if opts.aigc_metadata is not None and opts.audio_format not in COMPRESSED_AUDIO_FORMATS:
        raise ValueError(
            f"aigc_metadata requires audio_format to be one of {sorted(COMPRESSED_AUDIO_FORMATS)}"
        )
    return opts


def _validate_non_empty_string(name: str, value: object) -> None:
    """Require a non-empty string and identify the offending parameter."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError(f"{name} must not contain control characters")


def _validate_base_url(value: object) -> None:
    """Require a credential-safe absolute HTTPS API origin."""
    _validate_non_empty_string("base_url", value)
    parsed = urlsplit(str(value))
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("base_url must be an absolute HTTPS URL to protect API credentials")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("base_url must not contain embedded credentials")
    if parsed.path not in {"", "/"}:
        raise ValueError("base_url must be an origin without an additional path")
    if parsed.query or parsed.fragment:
        raise ValueError("base_url must not contain a query string or fragment")


def _validate_choice(name: str, value: object, choices: Collection[Any]) -> None:
    """Require an exact-typed value from a documented finite set."""
    valid = any(type(value) is type(choice) and value == choice for choice in choices)
    if not valid:
        raise ValueError(f"{name} must be one of {sorted(choices)}")


def _validate_int_range(
    name: str, value: object, value_range: tuple[int, int], *, unit: str = ""
) -> None:
    """Validate an integer range while rejecting bool, which subclasses int."""
    minimum, maximum = value_range
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer between {minimum} and {maximum}{unit}")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}{unit}")


def _validate_number_range(name: str, value: object, value_range: tuple[float, float]) -> None:
    """Validate a finite integer or float against an inclusive range."""
    minimum, maximum = value_range
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number between {minimum} and {maximum}")
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")


def _validate_optional_bool(name: str, value: object) -> None:
    """Validate a tri-state provider switch."""
    if value is not None and not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean or None")


def _validate_bool(name: str, value: object) -> None:
    """Require a strict boolean rather than accepting truthy values."""
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")


def _build_headers(opts: _TTSOptions, *, request_id: str) -> dict[str, str]:
    """Build authentication, routing, tracing, and optional usage headers."""
    headers = {
        "X-Api-Resource-Id": opts.resource_id,
        "X-Api-App-Key": opts.api_app_key,
        "X-Api-Request-Id": request_id,
        "Content-Type": "application/json",
        "Connection": "keep-alive",
        "User-Agent": USER_AGENT,
    }
    if opts.api_key:
        headers["X-Api-Key"] = opts.api_key
    else:
        headers["X-Api-App-Id"] = opts.app_id or ""
        headers["X-Api-Access-Key"] = opts.access_key or ""
    if opts.include_usage:
        headers["X-Control-Require-Usage-Tokens-Return"] = "*"
    return headers


def _build_request_payload(opts: _TTSOptions, *, text: str) -> dict[str, Any]:
    """Build the JSON body while keeping header and body concerns separate."""
    audio_params: dict[str, Any] = {
        "format": opts.audio_format,
        "sample_rate": opts.sample_rate,
        "speech_rate": opts.speech_rate,
        "loudness_rate": opts.loudness_rate,
    }
    audio_params.update(
        {
            name: value
            for name, value in (
                ("bit_rate", opts.bit_rate),
                ("emotion", opts.emotion),
                ("emotion_scale", opts.emotion_scale),
                ("enable_timestamp", opts.enable_timestamp),
                ("enable_subtitle", opts.enable_subtitle),
            )
            if value is not None
        }
    )

    additions = _build_additions(opts)
    req_params: dict[str, Any] = {
        "text": text,
        "speaker": opts.voice,
        "audio_params": audio_params,
        # The API expects this field as a JSON string, not a nested object.
        "additions": json.dumps(additions, ensure_ascii=False, separators=(",", ":")),
    }
    if opts.speaker_model is not None:
        req_params["model"] = opts.speaker_model
    payload: dict[str, Any] = {"req_params": req_params}
    if opts.user_id:
        payload["user"] = {"uid": opts.user_id}
    return payload


def _build_additions(opts: _TTSOptions) -> dict[str, Any]:
    """Serialize advanced provider options into the required JSON-string object."""
    additions: dict[str, Any] = copy.deepcopy(opts.extra_additions) if opts.extra_additions else {}

    if opts.pitch != 0:
        post_process = additions.get("post_process")
        if not isinstance(post_process, dict):
            post_process = {}
            additions["post_process"] = post_process
        post_process["pitch"] = opts.pitch
    additions.update(
        {
            name: value
            for name in _OPTIONAL_ADDITION_FIELDS
            if (value := getattr(opts, name)) is not None
        }
    )
    if opts.use_cache is not None:
        additions["cache_config"] = {"text_type": 1, "use_cache": opts.use_cache}
    if opts.aigc_metadata is not None:
        additions["aigc_metadata"] = opts.aigc_metadata.to_dict()
    return additions


async def _iter_json_events(content: AsyncIterable[bytes]) -> AsyncIterator[dict[str, Any]]:
    """Incrementally decode concatenated JSON events across arbitrary byte chunks."""
    utf8_decoder = codecs.getincrementaldecoder("utf-8")()
    buffer = ""
    async for raw_chunk in content:
        if not raw_chunk:
            continue
        if not isinstance(raw_chunk, bytes):
            raise APIStatusError(
                "BytePlus TTS returned a non-bytes response chunk",
                status_code=-1,
                request_id=None,
                body=type(raw_chunk).__name__,
                retryable=False,
            )
        try:
            buffer += utf8_decoder.decode(raw_chunk)
        except UnicodeDecodeError as e:
            raise APIStatusError(
                "BytePlus TTS returned invalid UTF-8 streaming data",
                status_code=-1,
                request_id=None,
                body=None,
                retryable=False,
            ) from e
        if len(buffer) > _MAX_STREAM_BUFFER_CHARS:
            raise APIStatusError(
                "BytePlus TTS streaming event exceeded the maximum supported size",
                status_code=-1,
                request_id=None,
                body=None,
                retryable=False,
            )
        events, buffer = _extract_json_events(buffer)
        for event in events:
            yield event

    try:
        buffer += utf8_decoder.decode(b"", final=True)
    except UnicodeDecodeError as e:
        raise APIConnectionError(
            "BytePlus TTS response was truncated in the middle of UTF-8 data"
        ) from e
    events, buffer = _extract_json_events(buffer)
    for event in events:
        yield event

    remaining = buffer.strip()
    if remaining:
        try:
            json.JSONDecoder().raw_decode(remaining)
        except json.JSONDecodeError as e:
            if e.msg.startswith("Unterminated string") or e.pos >= len(remaining):
                raise APIConnectionError(
                    "BytePlus TTS response was truncated before a JSON event completed"
                ) from e
            raise APIStatusError(
                "BytePlus TTS returned invalid JSON streaming data",
                status_code=-1,
                request_id=None,
                body=remaining[:_MAX_ERROR_BODY_CHARS],
                retryable=False,
            ) from e
        raise APIStatusError(
            "BytePlus TTS returned invalid trailing streaming data",
            status_code=-1,
            request_id=None,
            body=remaining[:_MAX_ERROR_BODY_CHARS],
            retryable=False,
        )


def _extract_json_events(buffer: str) -> tuple[list[dict[str, Any]], str]:
    """Extract every complete object and preserve an incomplete trailing object."""
    decoder = json.JSONDecoder()
    events: list[dict[str, Any]] = []
    while True:
        buffer = buffer.lstrip()
        if buffer.startswith("data:"):
            buffer = buffer[5:].lstrip()
        if not buffer:
            return events, ""
        try:
            event, idx = decoder.raw_decode(buffer)
        except json.JSONDecodeError:
            return events, buffer
        if not isinstance(event, dict):
            raise APIStatusError(
                "BytePlus TTS returned a non-object streaming event",
                status_code=-1,
                request_id=None,
                body=event,
                retryable=False,
            )
        events.append(event)
        buffer = buffer[idx:]


def _handle_response_event(
    *,
    provider: TTS,
    event: dict[str, Any],
    output_emitter: tts.AudioEmitter,
    request_id: str,
    transcript_offset: float,
    audio_sink: Callable[[bytes], None] | None = None,
) -> tuple[bool, float]:
    """Process one provider event and return terminal state plus transcript duration."""
    code = event.get("code")
    message = event.get("message", "")
    if isinstance(code, bool) or not isinstance(code, int):
        raise APIStatusError(
            "BytePlus TTS returned an invalid response code",
            status_code=-1,
            request_id=request_id,
            body=event,
            retryable=False,
        )

    _emit_usage(provider=provider, event=event, request_id=request_id)
    timed_words = _extract_timed_words(event, time_offset=transcript_offset)
    if timed_words:
        output_emitter.push_timed_transcript(timed_words)
    transcript_end = max(
        (
            float(word.end_time) - transcript_offset
            for word in timed_words
            if is_given(word.end_time)
        ),
        default=0.0,
    )

    if code == 20000000:
        return True, transcript_end
    if code != 0:
        raise APIStatusError(
            f"BytePlus TTS request failed [{code}]: {message or 'unknown provider error'}. "
            f"Provider request ID: {request_id}",
            status_code=code,
            request_id=request_id,
            body=event,
            retryable=_is_retryable_provider_error(code=code, message=message),
        )

    data = event.get("data")
    if not isinstance(data, str) or not data:
        return False, transcript_end
    try:
        audio = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as e:
        raise APIStatusError(
            "BytePlus TTS returned invalid base64 audio data",
            status_code=-1,
            request_id=request_id,
            body=event,
            retryable=False,
        ) from e
    (audio_sink or output_emitter.push)(audio)
    return False, transcript_end


def _emit_usage(*, provider: TTS, event: dict[str, Any], request_id: str) -> None:
    """Emit validated billable-character usage through the LiveKit provider event."""
    usage = event.get("usage")
    if not isinstance(usage, dict):
        return
    text_words = usage.get("text_words")
    if isinstance(text_words, bool) or not isinstance(text_words, int) or text_words < 0:
        logger.warning(
            "ignored invalid BytePlus TTS usage payload",
            extra={"request_id": request_id},
        )
        return
    provider.emit(
        "usage_collected",
        TTSUsage(request_id=request_id, text_words=text_words),
    )


def _extract_timed_words(event: dict[str, Any], *, time_offset: float = 0.0) -> list[TimedString]:
    """Convert provider word timestamps to monotonic LiveKit ``TimedString`` values."""
    sentence = event.get("sentence")
    if not isinstance(sentence, dict):
        return []
    words = sentence.get("words")
    if not isinstance(words, list):
        return []

    timed_words: list[TimedString] = []
    for item in words:
        if not isinstance(item, dict):
            continue
        text = item.get("word")
        start_time = item.get("startTime")
        end_time = item.get("endTime")
        confidence = item.get("confidence")
        if (
            not isinstance(text, str)
            or not text
            or isinstance(start_time, bool)
            or not isinstance(start_time, (int, float))
            or isinstance(end_time, bool)
            or not isinstance(end_time, (int, float))
            or not math.isfinite(start_time)
            or not math.isfinite(end_time)
            or start_time < 0
            or end_time < start_time
        ):
            continue
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(confidence)
            or not 0 <= confidence <= 1
        ):
            confidence = NOT_GIVEN
        timed_words.append(
            TimedString(
                text,
                start_time=float(start_time) + time_offset,
                end_time=float(end_time) + time_offset,
                confidence=float(confidence) if is_given(confidence) else NOT_GIVEN,
            )
        )
    return timed_words


def _is_retryable_provider_error(*, code: int, message: object) -> bool:
    """Classify documented transient provider business failures."""
    normalized_message = message.lower() if isinstance(message, str) else ""
    if code == 55000000:
        return True
    return code == 45000000 and (
        "concurrency" in normalized_message or "quota exceeded" in normalized_message
    )


def _mime_type(audio_format: TTSAudioFormat) -> str:
    """Map BytePlus format names to MIME types accepted by ``AudioEmitter``."""
    if audio_format == "pcm":
        return "audio/pcm"
    if audio_format == "ogg_opus":
        return "audio/ogg"
    return f"audio/{audio_format}"
