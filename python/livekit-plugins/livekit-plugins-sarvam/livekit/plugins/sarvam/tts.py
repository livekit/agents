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

"""Text-to-Speech implementation for Sarvam.ai

This module provides a TTS implementation that uses the Sarvam.ai API.
"""

from __future__ import annotations

import base64
import binascii
import os
import platform
import re
import weakref
from collections import deque
from dataclasses import dataclass, replace
from typing import Literal, NotRequired, TypedDict, assert_never

from pydantic import JsonValue, TypeAdapter, ValidationError

from livekit.agents import (
    APIConnectOptions,
    APIError,
    APIStatusError,
    LanguageCode,
    __version__ as livekit_version,
    http,
    tokenize,
    tts,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr

from .log import logger
from .models import (
    MODEL_SPEAKER_COMPATIBILITY,
    SarvamTTSLanguages,
    SarvamTTSModels,
    SarvamTTSOutputAudioBitrate,
    SarvamTTSSpeakers,
)

USER_AGENT = f"Livekit/{livekit_version} Python/{platform.python_version()}"

SARVAM_TTS_BASE_URL = "https://api.sarvam.ai/text-to-speech"
SARVAM_TTS_WS_URL = "wss://api.sarvam.ai/text-to-speech/ws"

# Sarvam closes a socket idle for 60 seconds: an idle pooled socket is pinged every 30
_KEEPALIVE_INTERVAL = 30.0

# bulbul:v4-flash rides the same pipeline as v3/v3-beta: temperature and the websocket
# buffering knobs apply to all three.
_V3_PIPELINE_MODELS = ("bulbul:v3", "bulbul:v3-beta", "bulbul:v4-flash")

ALLOWED_OUTPUT_AUDIO_BITRATES: set[str] = {"32k", "64k", "96k", "128k", "192k"}
ALLOWED_OUTPUT_AUDIO_CODECS: set[str] = {
    "mp3",
    "opus",
    "flac",
    "aac",
    "wav",
    "linear16",
    "mulaw",
    "alaw",
}

_CODEC_TO_MIME: dict[str, str] = {
    "mp3": "audio/mp3",
    "wav": "audio/wav",
    "opus": "audio/opus",
    "flac": "audio/flac",
    "aac": "audio/aac",
    "linear16": "audio/pcm",
    "mulaw": "audio/pcmu",
    "alaw": "audio/pcma",
}


def _codec_to_mime_type(codec: str) -> str:
    """Map a Sarvam output_audio_codec value to the MIME type the framework decoder expects."""
    mime = _CODEC_TO_MIME.get(codec)
    if mime is None:
        raise ValueError(f"Unsupported output_audio_codec: {codec}")
    return mime


def validate_model_speaker_compatibility(model: str, speaker: str) -> bool:
    """Validate that the speaker is compatible with the model version."""
    if model not in MODEL_SPEAKER_COMPATIBILITY:
        logger.warning(f"Unknown model '{model}', skipping compatibility check")
        return True

    compatible_speakers = MODEL_SPEAKER_COMPATIBILITY[model]["all"]
    if speaker.lower() not in compatible_speakers:
        logger.error(
            f"Speaker '{speaker}' is not compatible with model '{model}'. "
            f"Compatible speakers for {model}: {', '.join(compatible_speakers)}"
        )
        return False
    return True


# Accepted [min, max] per synthesis parameter. bulbul:v3 narrows the pace; bulbul:v4-flash is
# stricter still.
_PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "pitch": (-0.75, 0.75),
    "pace": (0.3, 3.0),
    "loudness": (0.5, 2.0),
    "temperature": (0.01, 2.0),
}
_V3_PARAM_BOUNDS: dict[str, tuple[float, float]] = {**_PARAM_BOUNDS, "pace": (0.5, 2.0)}
_V4_PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "pitch": (-0.5, 0.5),
    "pace": (0.5, 2.0),
    "loudness": (0.1, 2.5),
    "temperature": (0.01, 1.0),
}


def _param_bounds(model: str, param: str) -> tuple[float, float]:
    """Accepted range for a synthesis parameter on the given model."""
    if model == "bulbul:v4-flash":
        return _V4_PARAM_BOUNDS[param]
    if model in ("bulbul:v3", "bulbul:v3-beta"):
        return _V3_PARAM_BOUNDS[param]
    return _PARAM_BOUNDS[param]


def _validate_param(model: str, param: str, value: float) -> None:
    """Raise if a synthesis parameter is outside the range the model accepts."""
    low, high = _param_bounds(model, param)
    if not low <= value <= high:
        raise ValueError(f"{param} must be between {low} and {high} for model '{model}'")


def _clamp_pitch(model: str, pitch: float) -> float:
    low, high = _param_bounds(model, "pitch")
    if not low <= pitch <= high:
        logger.warning(
            "pitch value %.2f is outside the Sarvam API accepted range [%s, %s] for %s; "
            "clamping to nearest bound. Please update your code.",
            pitch,
            low,
            high,
            model,
        )
        return max(low, min(high, pitch))
    return pitch


@dataclass
class SarvamTTSOptions:
    """Options for the Sarvam.ai TTS service.

    Args:
        target_language_code: BCP-47 language code for supported Indian languages
        api_key: Sarvam.ai API key
        text: The text to synthesize (will be provided by stream adapter)
        speaker: Voice to use for synthesis
        pitch: Voice pitch adjustment (-0.75 to 0.75; -0.5 to 0.5 for bulbul:v4-flash)
        pace: Speech rate multiplier (0.3 to 3.0; 0.5 to 2.0 for bulbul:v4-flash)
        loudness: Volume multiplier (0.5 to 2.0; 0.1 to 2.5 for bulbul:v4-flash)
        temperature: Sampling temperature (0.01 to 2.0; 0.01 to 1.0 for bulbul:v4-flash),
            used for v3, v3-beta and v4-flash. bulbul:v4-flash accepts the value then
            forces it to 0.6 server-side, so setting it there has no effect.
        output_audio_bitrate: Output audio bitrate
        min_buffer_size: Minimum character length for flushing
        max_chunk_length: Maximum chunk length for sentence splitting
        speech_sample_rate: Audio sample rate (8000, 16000, 22050, 24000, 32000, 44100, or 48000;
            streaming bulbul:v4-flash is limited to 8000, 16000, 22050 or 24000)
        enable_preprocessing: Whether to use text preprocessing (bulbul:v2 and bulbul:v4-flash;
            bulbul:v4-flash forces it on server-side regardless of this value)
        dict_id: Custom pronunciation dictionary ID (bulbul:v3 and bulbul:v4-flash)
        enable_cached_responses: Enable response caching beta feature (bulbul:v1/v2 only)
        model: The Sarvam TTS model to use
        base_url: API endpoint URL
        ws_url: WebSocket endpoint URL
    """

    target_language_code: SarvamTTSLanguages | str  # BCP-47 for supported Indian languages
    api_key: str  # Sarvam.ai API key
    text: str | None = None  # Will be provided by the stream adapter
    speaker: SarvamTTSSpeakers | str | None = None
    pitch: float = 0.0
    pace: float = 1.0
    loudness: float = 1.0
    temperature: float = 0.6
    output_audio_bitrate: SarvamTTSOutputAudioBitrate | str = "128k"
    min_buffer_size: int = 50
    max_chunk_length: int = 150
    speech_sample_rate: int = 22050  # Default 22050 Hz
    enable_preprocessing: bool = False
    dict_id: str | None = None  # Custom pronunciation dictionary (bulbul:v3 only)
    enable_cached_responses: bool | None = None  # Response caching beta (bulbul:v1/v2 only)
    model: SarvamTTSModels | str = "bulbul:v2"  # Default to v2
    base_url: str = SARVAM_TTS_BASE_URL
    ws_url: str = SARVAM_TTS_WS_URL
    send_completion_event: bool = True
    output_audio_codec: str = "mp3"


# Streaming bulbul:v4-flash rejects the higher REST sample rates, and OPUS narrows it further.
_V4_STREAM_SAMPLE_RATES = (8000, 16000, 22050, 24000)
_V4_STREAM_OPUS_SAMPLE_RATES = (8000, 16000, 24000)


class _Config(TypedDict):
    """The synthesis options, shared by the REST body and the websocket's config message."""

    target_language_code: str
    speaker: str | None
    pace: float
    model: str
    speech_sample_rate: int
    output_audio_codec: str
    pitch: NotRequired[float]
    loudness: NotRequired[float]
    enable_preprocessing: NotRequired[bool]
    enable_cached_responses: NotRequired[bool]
    temperature: NotRequired[float]
    dict_id: NotRequired[str]
    output_audio_bitrate: NotRequired[str]
    min_buffer_size: NotRequired[int]
    max_chunk_length: NotRequired[int]


def _config(opts: SarvamTTSOptions) -> _Config:
    config = _Config(
        target_language_code=opts.target_language_code,
        speaker=opts.speaker,
        pace=opts.pace,
        model=opts.model,
        speech_sample_rate=opts.speech_sample_rate,
        output_audio_codec=opts.output_audio_codec,
    )
    if opts.model in ("bulbul:v2", "bulbul:v4-flash"):
        config["pitch"] = opts.pitch
        config["loudness"] = opts.loudness
        config["enable_preprocessing"] = opts.enable_preprocessing
    # v3 and v4 silently ignore caching, so it is only ever sent for v2
    if opts.model == "bulbul:v2" and opts.enable_cached_responses is not None:
        config["enable_cached_responses"] = opts.enable_cached_responses
    if opts.model in _V3_PIPELINE_MODELS:
        config["temperature"] = opts.temperature
    if opts.model in ("bulbul:v3", "bulbul:v4-flash") and opts.dict_id is not None:
        config["dict_id"] = opts.dict_id
    return config


class _Request(_Config):
    text: str


def _websocket_url(opts: SarvamTTSOptions) -> str:
    """Build the TTS websocket URL, validating the stream-only limits of bulbul:v4-flash."""

    url = opts.ws_url
    if opts.model == "bulbul:v4-flash":
        allowed = (
            _V4_STREAM_OPUS_SAMPLE_RATES
            if opts.output_audio_codec == "opus"
            else _V4_STREAM_SAMPLE_RATES
        )
        if opts.speech_sample_rate not in allowed:
            raise APIError(
                "Sarvam refuses this stream's sample rate",
                kind="invalid_request",
                detail=f"speech_sample_rate must be one of {', '.join(str(r) for r in allowed)} "
                f"when streaming bulbul:v4-flash with codec '{opts.output_audio_codec}'",
            )
        if not url.rstrip("/").endswith("/v2"):
            url = f"{url.rstrip('/')}/v2"
    return f"{url}?model={opts.model}&send_completion_event={opts.send_completion_event}"


_MESSAGE_STATUS_RE = re.compile(r"\s*(\d{3})\s*:")
_CODE_STATUS_RE = re.compile(r"\s*\d{3}\s*")


def _error_status_code(error_data: _ErrorData) -> int | None:
    """Extract the HTTP-equivalent status code from a Sarvam websocket error frame.

    Schema rejections carry ``code`` (e.g. 422, or the string ``"422"``). Others omit
    it and prefix the message instead, as in ``"400: Speaker '...' is not compatible
    with model bulbul:v4-flash"``. Returns None when neither form is present.
    """
    code = error_data.get("code")
    if isinstance(code, bool):  # bool is an int subclass; never a status code
        return None
    if isinstance(code, int):
        return code
    if isinstance(code, str) and _CODE_STATUS_RE.fullmatch(code):
        return int(code)

    match = _MESSAGE_STATUS_RE.match(str(error_data.get("message")))
    return int(match.group(1)) if match else None


class TTS(tts.TTS):
    """Sarvam.ai Text-to-Speech implementation.

    This class provides text-to-speech functionality using the Sarvam.ai API.
    Sarvam.ai specializes in high-quality TTS for Indian languages.

    Args:
        target_language_code: BCP-47 language code for supported Indian languages
        model: Sarvam TTS model to use (bulbul:v2)
        speaker: Voice to use for synthesis
        speech_sample_rate: Audio sample rate in Hz
        num_channels: Number of audio channels (Sarvam outputs mono)
        pitch: Voice pitch adjustment (-0.75 to 0.75; -0.5 to 0.5 for bulbul:v4-flash) -
            only supported in v2 and v4-flash
        pace: Speech rate multiplier (0.3 to 3.0; 0.5 to 2.0 for bulbul:v4-flash)
        loudness: Volume multiplier (0.5 to 2.0; 0.1 to 2.5 for bulbul:v4-flash) -
            only supported in v2 and v4-flash
        temperature: Sampling temperature (0.01 to 2.0; 0.01 to 1.0 for bulbul:v4-flash),
            only used in v3, v3-beta and v4-flash. bulbul:v4-flash accepts the value then
            forces it to 0.6 server-side, so setting it there has no effect.
        dict_id: Custom pronunciation dictionary ID (bulbul:v3 and bulbul:v4-flash)
        enable_cached_responses: Enable response caching beta feature (bulbul:v1/v2 only)
        output_audio_bitrate: Output audio bitrate (default 128k)
        min_buffer_size: Minimum character length for flushing (30 to 200)
        max_chunk_length: Maximum chunk length for sentence splitting (50 to 500)
        enable_preprocessing: Whether to use text preprocessing (bulbul:v4-flash forces it
            on server-side regardless of this value)
        api_key: Sarvam.ai API key (required)
        base_url: API endpoint URL
        ws_url: WebSocket endpoint URL
        http_session: Optional HTTP session to use
        output_audio_codec: Optionally choose the output codec format (mp3)
    """

    def __init__(
        self,
        *,
        target_language_code: SarvamTTSLanguages | str = "en-IN",
        model: SarvamTTSModels | str = "bulbul:v3",
        speaker: SarvamTTSSpeakers | str | None = None,
        speech_sample_rate: int = 22050,
        num_channels: int = 1,  # Sarvam output is mono WAV
        pitch: float = 0.0,
        pace: float = 1.0,
        loudness: float = 1.0,
        temperature: float = 0.6,
        output_audio_bitrate: SarvamTTSOutputAudioBitrate | str = "128k",
        min_buffer_size: int = 50,
        max_chunk_length: int = 150,
        enable_preprocessing: bool = False,
        dict_id: str | None = None,
        enable_cached_responses: bool | None = None,
        api_key: str | None = None,
        base_url: str = SARVAM_TTS_BASE_URL,
        ws_url: str = SARVAM_TTS_WS_URL,
        http_session: http.HttpSession | None = None,
        send_completion_event: bool = True,
        output_audio_codec: str = "mp3",
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=speech_sample_rate,
            num_channels=num_channels,
        )

        self._api_key = api_key or os.environ.get("SARVAM_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Sarvam API key is required. Provide it directly or set SARVAM_API_KEY env var."
            )

        # Validate inputs early
        if not target_language_code or not target_language_code.strip():
            raise ValueError("Target language code is required and cannot be empty")
        if not model or not model.strip():
            raise ValueError("Model is required and cannot be empty")
        if speaker is None:
            # speaker = "shubh"
            if model == "bulbul:v4-flash":
                speaker = "shubh_en_narration_gentle"
            elif model == "bulbul:v3-beta" or model == "bulbul:v3":
                speaker = "shubh"
            else:
                speaker = "anushka"

        # Validate parameter ranges
        pitch = _clamp_pitch(model, pitch)
        _validate_param(model, "pace", pace)
        _validate_param(model, "loudness", loudness)
        _validate_param(model, "temperature", temperature)
        if output_audio_bitrate not in ALLOWED_OUTPUT_AUDIO_BITRATES:
            raise ValueError(
                f"output_audio_bitrate must be one of {', '.join(sorted(ALLOWED_OUTPUT_AUDIO_BITRATES))}"
            )
        if not 30 <= min_buffer_size <= 200:
            raise ValueError("min_buffer_size must be between 30 and 200")
        if not 50 <= max_chunk_length <= 500:
            raise ValueError("max_chunk_length must be between 50 and 500")
        if speech_sample_rate not in [8000, 16000, 22050, 24000, 32000, 44100, 48000]:
            raise ValueError(
                "Sample rate must be one of 8000, 16000, 22050, 24000, 32000, 44100, or 48000 Hz"
            )
        if output_audio_codec not in ALLOWED_OUTPUT_AUDIO_CODECS:
            raise ValueError(
                f"output_audio_codec must be one of {','.join(sorted(ALLOWED_OUTPUT_AUDIO_CODECS))}"
            )

        # Validate model-speaker compatibility
        if not validate_model_speaker_compatibility(model, speaker):
            compatible_speakers = MODEL_SPEAKER_COMPATIBILITY.get(model, {}).get("all", [])
            raise ValueError(
                f"Speaker '{speaker}' is not compatible with model '{model}'. "
                f"Please choose a compatible speaker from: {', '.join(compatible_speakers)}"
            )

        self._opts = SarvamTTSOptions(
            target_language_code=LanguageCode(target_language_code),
            model=model,
            speaker=speaker,
            speech_sample_rate=speech_sample_rate,
            pitch=pitch,
            pace=pace,
            loudness=loudness,
            temperature=temperature,
            output_audio_bitrate=output_audio_bitrate,
            min_buffer_size=min_buffer_size,
            max_chunk_length=max_chunk_length,
            enable_preprocessing=enable_preprocessing,
            dict_id=dict_id,
            enable_cached_responses=enable_cached_responses,
            api_key=self._api_key,
            base_url=base_url,
            ws_url=ws_url,
            send_completion_event=send_completion_event,
            output_audio_codec=output_audio_codec,
        )
        self._session = http_session
        self._streams = weakref.WeakSet[SynthesizeStream]()
        self._mux = _Pool(self)

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "Sarvam"

    def _ensure_session(self) -> http.HttpSession:
        if not self._session:
            self._session = http.http_session()
        return self._session

    def update_options(
        self,
        *,
        model: str | None = None,
        target_language_code: SarvamTTSLanguages | str | None = None,
        speaker: str | None = None,
        pitch: float | None = None,
        pace: float | None = None,
        loudness: float | None = None,
        temperature: float | None = None,
        output_audio_bitrate: SarvamTTSOutputAudioBitrate | str | None = None,
        min_buffer_size: int | None = None,
        max_chunk_length: int | None = None,
        enable_preprocessing: bool | None = None,
        dict_id: str | None = None,
        enable_cached_responses: bool | None = None,
        send_completion_event: bool | None = None,
        output_audio_codec: str | None = None,
        sentence_tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
        pacing: NotGivenOr[tts.SentenceStreamPacer | None] = NOT_GIVEN,
    ) -> None:
        """Update TTS options with validation.

        Changes land on a copy that is committed, with the text options, only after every
        model-dependent limit validates against the resulting model. A rejected update
        therefore changes nothing, and switching models cannot carry over a speaker or a
        pitch/pace/loudness/temperature the new model refuses.
        """
        with self._updating_options(sentence_tokenizer=sentence_tokenizer, pacing=pacing):
            opts = replace(self._opts)

            if target_language_code is not None:
                if not target_language_code.strip():
                    raise ValueError("Target language code cannot be empty")
                opts.target_language_code = LanguageCode(target_language_code)

            if model is not None:
                if not model.strip():
                    raise ValueError("Model cannot be empty")
                opts.model = model

            if speaker is not None:
                if not speaker.strip():
                    raise ValueError("Speaker cannot be empty")
                opts.speaker = speaker

            if pitch is not None:
                opts.pitch = pitch

            if pace is not None:
                opts.pace = pace

            if loudness is not None:
                opts.loudness = loudness

            if temperature is not None:
                opts.temperature = temperature

            if output_audio_bitrate is not None:
                if output_audio_bitrate not in ALLOWED_OUTPUT_AUDIO_BITRATES:
                    raise ValueError(
                        "output_audio_bitrate must be one of "
                        f"{', '.join(sorted(ALLOWED_OUTPUT_AUDIO_BITRATES))}"
                    )
                opts.output_audio_bitrate = output_audio_bitrate

            if min_buffer_size is not None:
                if not 30 <= min_buffer_size <= 200:
                    raise ValueError("min_buffer_size must be between 30 and 200")
                opts.min_buffer_size = min_buffer_size

            if max_chunk_length is not None:
                if not 50 <= max_chunk_length <= 500:
                    raise ValueError("max_chunk_length must be between 50 and 500")
                opts.max_chunk_length = max_chunk_length

            if enable_preprocessing is not None:
                opts.enable_preprocessing = enable_preprocessing

            if dict_id is not None:
                opts.dict_id = dict_id

            if enable_cached_responses is not None:
                opts.enable_cached_responses = enable_cached_responses

            if send_completion_event is not None:
                opts.send_completion_event = send_completion_event

            if output_audio_codec is not None:
                if output_audio_codec not in ALLOWED_OUTPUT_AUDIO_CODECS:
                    raise ValueError(
                        "output_audio_codec must be one of "
                        f"{','.join(sorted(ALLOWED_OUTPUT_AUDIO_CODECS))}"
                    )
                opts.output_audio_codec = output_audio_codec

            # Re-check every model-dependent limit against the resulting model, not just
            # the arguments this call passed: the v4-flash speaker catalogue and bounds
            # are disjoint from v3's, so a bare model switch can invalidate stored values.
            if opts.speaker is not None and not validate_model_speaker_compatibility(
                opts.model, opts.speaker
            ):
                compatible_speakers = MODEL_SPEAKER_COMPATIBILITY.get(opts.model, {}).get("all", [])
                raise ValueError(
                    f"Speaker '{opts.speaker}' incompatible with {opts.model}. "
                    f"Compatible speakers: {', '.join(compatible_speakers)}"
                )
            _validate_param(opts.model, "pace", opts.pace)
            _validate_param(opts.model, "loudness", opts.loudness)
            _validate_param(opts.model, "temperature", opts.temperature)
            opts.pitch = _clamp_pitch(opts.model, opts.pitch)

        # a socket is configured once, when it opens: a change reaches the sockets opened next
        changed = opts != self._opts
        self._opts = opts
        if changed:
            self._mux.invalidate()

    def _synthesize(
        self, chunk: tts.SpeechChunk, *, conn_options: APIConnectOptions
    ) -> ChunkedStream:
        return ChunkedStream(
            tts=self, input_text=chunk.text, conn_options=conn_options, opts=replace(self._opts)
        )

    def _stream(self, *, conn_options: APIConnectOptions) -> SynthesizeStream:
        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    def prewarm(self) -> None:
        self._mux.prewarm()

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()
        await self._mux.aclose()


class _Batch(TypedDict):
    request_id: NotRequired[str | None]
    audios: list[str]


_BATCH = TypeAdapter(_Batch)


class ChunkedStream(tts.ChunkedStream):
    def __init__(
        self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions, opts: SarvamTTSOptions
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = opts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request: _Request = {
            **_config(self._opts),
            "text": self._input_text,
            "output_audio_bitrate": self._opts.output_audio_bitrate,
            "min_buffer_size": self._opts.min_buffer_size,
            "max_chunk_length": self._opts.max_chunk_length,
        }
        async with self._tts._ensure_session().post(
            self._opts.base_url,
            json=request,
            headers={
                "api-subscription-key": self._opts.api_key,
                "Content-Type": "application/json",
                "User-Agent": USER_AGENT,
            },
            conn_options=self._conn_options,
        ) as res:
            if res.status != 200:
                error_text = await res.text()
                raise APIStatusError(
                    "Sarvam TTS request failed",
                    status_code=res.status,
                    body=error_text,
                    detail=error_text,
                )
            try:
                batch = _BATCH.validate_python(await res.json())
            except ValidationError as e:
                raise APIError("Sarvam TTS sent an invalid response", kind="protocol") from e
            if not batch["audios"]:
                raise APIError("Sarvam TTS sent no audio", kind="protocol")

            output_emitter.initialize(
                request_id=batch.get("request_id") or utils.shortuuid(),
                sample_rate=self._tts.sample_rate,
                num_channels=self._tts.num_channels,
                mime_type=_codec_to_mime_type(self._opts.output_audio_codec),
            )
            for audio in batch["audios"]:
                output_emitter.push(base64.b64decode(audio))


# Sarvam's text-to-speech socket: configured once, then each segment is its text and a flush,
# answered in order with audio and a `final` event. Each message declares only the fields read.


class _Data(TypedDict, total=False):
    request_id: str | None


class _AudioData(_Data, total=False):
    audio: str


class _EventData(_Data, total=False):
    event_type: str | None


class _ErrorData(_Data, total=False):
    code: str | int | None
    message: JsonValue


class _Audio(TypedDict):
    type: Literal["audio"]
    data: NotRequired[_AudioData | None]
    request_id: NotRequired[str | None]


class _Event(TypedDict):
    type: Literal["event"]
    data: NotRequired[_EventData | None]
    request_id: NotRequired[str | None]


class _Events(TypedDict):
    type: Literal["events"]
    data: NotRequired[_EventData | None]
    request_id: NotRequired[str | None]


class _Error(TypedDict):
    type: Literal["error"]
    data: NotRequired[_ErrorData | None]
    request_id: NotRequired[str | None]


class _Errors(TypedDict):
    type: Literal["errors"]
    data: NotRequired[_ErrorData | None]
    request_id: NotRequired[str | None]


_Message = _Audio | _Event | _Events | _Error | _Errors


class _ConfigMessage(TypedDict):
    type: Literal["config"]
    data: _Config


class _TextData(TypedDict):
    text: str


class _Text(TypedDict):
    type: Literal["text"]
    data: _TextData


class _Control(TypedDict):
    type: Literal["flush", "ping"]


_Command = _ConfigMessage | _Text | _Control


class _Pool(tts.WebSocketMultiplexer[_Message, _Command]):
    """Sarvam sockets, one stream at a time each, configured with the TTS's options when they
    open and reused by the next stream."""

    routed = False
    keepalive = tts.Keepalive(every=_KEEPALIVE_INTERVAL, frame=_Control(type="ping"))
    max_session_duration = 3600

    def __init__(self, tts: TTS) -> None:
        self._tts = tts

    async def connect(self, conn_options: APIConnectOptions) -> tts.WebSocketTarget:
        opts = self._tts._opts
        return tts.WebSocketTarget(
            _websocket_url(opts),
            headers={"api-subscription-key": opts.api_key, "User-Agent": USER_AGENT},
        )

    def setup(self) -> _ConfigMessage:
        opts = self._tts._opts
        config = _config(opts)
        if opts.model in _V3_PIPELINE_MODELS:
            config["output_audio_bitrate"] = opts.output_audio_bitrate
            config["min_buffer_size"] = opts.min_buffer_size
            config["max_chunk_length"] = opts.max_chunk_length
        return _ConfigMessage(type="config", data=config)


class SynthesizeStream(tts.WebSocketStream[_Message, _Command]):
    """A stream on a pooled Sarvam socket; its segments are answered in the order they went."""

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self.multiplexer = tts._mux
        self.mime_type = _codec_to_mime_type(tts._opts.output_audio_codec)
        # the segments awaiting their audio, in the order they went
        self._open = deque[str]()

    def encode(self, item: tts.SegmentInput | tts.EndOfText) -> _Command | None:
        match item:
            case tts.SegmentStart():
                self._open.append(item.segment_id)
                return None
            case tts.SegmentText():
                return _Text(type="text", data=_TextData(text=item.chunk.text + " "))
            case tts.SegmentEnd():
                return _Control(type="flush")
            case tts.EndOfText():
                return None
            case _:
                assert_never(item)

    def on_message(self, message: _Message) -> None:
        data = message.get("data") or {}
        request_id = message.get("request_id") or data.get("request_id")
        if request_id and request_id != self.request_id:
            self.report(tts.ProviderRequestId(request_id))
        segment = self._open[0] if self._open else ""
        if message["type"] == "audio":
            if audio := (message.get("data") or {}).get("audio"):
                try:
                    decoded = base64.b64decode(audio, validate=True)
                except binascii.Error as e:
                    raise APIError("Sarvam TTS sent malformed audio", kind="protocol") from e
                self.report(tts.SegmentAudio(segment, decoded))
        elif message["type"] == "event" or message["type"] == "events":
            if (message.get("data") or {}).get("event_type") == "final" and self._open:
                self.report(tts.SegmentDone(self._open.popleft()))
        elif message["type"] == "error" or message["type"] == "errors":
            raise _error(message.get("data") or {})
        else:
            self._skip_unknown_message(message)


def _error(error_data: _ErrorData) -> APIError:
    message = error_data.get("message")
    detail = None if message is None else str(message)
    if (status := _error_status_code(error_data)) is None:
        # an error frame without a status refused the request
        return APIError("TTS API error from Sarvam", kind="invalid_request", detail=detail)
    return APIStatusError(
        f"TTS API error from Sarvam (status {status})",
        status_code=status,
        detail=detail,
        request_id=error_data.get("request_id"),
    )
