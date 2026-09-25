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

import asyncio
import base64
import contextlib
import enum
import json
import os
import platform
import re
import weakref
from dataclasses import dataclass, replace
from typing import Literal

import aiohttp
import numpy as np

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    LanguageCode,
    __version__ as livekit_version,
    tokenize,
    tts,
    utils,
)

from .log import logger

USER_AGENT = f"Livekit/{livekit_version} Python/{platform.python_version()}"

SARVAM_TTS_BASE_URL = "https://api.sarvam.ai/text-to-speech"
SARVAM_TTS_WS_URL = "wss://api.sarvam.ai/text-to-speech/ws"

# Sarvam TTS WebSocket connections idle out after 60 seconds. We defend against
# this on two layers:
#
# 1. Protocol-level: aiohttp ``heartbeat`` sends WebSocket PING frames every
#    ``_WS_HEARTBEAT_INTERVAL`` seconds and auto-handles PONGs even while no
#    application-level ``receive()`` is in flight. This is what actually keeps
#    the TCP connection alive while it sits idle in the pool.
# 2. Application-level: ``_keepalive_loop`` sends a Sarvam-defined
#    ``{"type": "ping"}`` JSON message every ``_KEEPALIVE_INTERVAL`` seconds.
#    This guards against a server-side application idle timer that may not be
#    reset by protocol-level pings, and serves as a fast detector for stale
#    pooled connections.
_WS_HEARTBEAT_INTERVAL: float = 20.0
_KEEPALIVE_INTERVAL: float = 30.0

# Sarvam TTS specific models and speakers
SarvamTTSModels = Literal["bulbul:v2", "bulbul:v3-beta", "bulbul:v3", "bulbul:v4-flash"]
SarvamTTSOutputAudioBitrate = Literal["32k", "64k", "96k", "128k", "192k"]

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
    # G.711 telephony codecs are decoded to linear 16-bit PCM in this plugin
    # before being pushed to the AudioEmitter, so they advertise audio/pcm.
    "mulaw": "audio/pcm",
    "alaw": "audio/pcm",
}

_TELEPHONY_CODECS: frozenset[str] = frozenset({"mulaw", "alaw"})


def _codec_to_mime_type(codec: str) -> str:
    """Map a Sarvam output_audio_codec value to the MIME type the framework decoder expects."""
    mime = _CODEC_TO_MIME.get(codec)
    if mime is None:
        raise ValueError(f"Unsupported output_audio_codec: {codec}")
    return mime


def _build_mulaw_table() -> np.ndarray:
    """Build an ITU-T G.711 mu-law to linear 16-bit PCM lookup table."""
    table = np.zeros(256, dtype=np.int16)
    for i in range(256):
        u = (~i) & 0xFF
        sign = -1 if (u & 0x80) else 1
        exponent = (u >> 4) & 0x07
        mantissa = u & 0x0F
        sample = ((mantissa << 3) + 0x84) << exponent
        table[i] = sign * (sample - 0x84)
    return table


def _build_alaw_table() -> np.ndarray:
    """Build an ITU-T G.711 A-law to linear 16-bit PCM lookup table.

    Note: A-law sign convention is inverted relative to mu-law -- when the
    sign bit of the XOR'd byte is set, the sample is positive.
    """
    table = np.zeros(256, dtype=np.int16)
    for i in range(256):
        a = i ^ 0x55
        sign = 1 if (a & 0x80) else -1
        exponent = (a >> 4) & 0x07
        mantissa = a & 0x0F
        if exponent == 0:
            sample = (mantissa << 4) + 8
        else:
            sample = ((mantissa << 4) + 0x108) << (exponent - 1)
        table[i] = sign * sample
    return table


_MULAW_TABLE: np.ndarray = _build_mulaw_table()
_ALAW_TABLE: np.ndarray = _build_alaw_table()


def _decode_telephony(codec: str, data: bytes) -> bytes:
    """Decode 8-bit G.711 mu-law/A-law samples to little-endian linear 16-bit PCM."""
    if codec == "mulaw":
        table = _MULAW_TABLE
    elif codec == "alaw":
        table = _ALAW_TABLE
    else:
        raise ValueError(f"_decode_telephony does not support codec: {codec}")
    pcm = table[np.frombuffer(data, dtype=np.uint8)]
    return pcm.astype("<i2").tobytes()


# Supported languages in BCP-47 format. The IN22 languages are bulbul:v4-flash only and
# additionally require the `enable_in22_languages` flag on the subscription.
SarvamTTSLanguages = Literal[
    "as-IN",  # Assamese (IN22)
    "bn-IN",  # Bengali
    "brx-IN",  # Bodo (IN22)
    "doi-IN",  # Dogri (IN22)
    "en-IN",  # English (India)
    "gu-IN",  # Gujarati
    "hi-IN",  # Hindi
    "kn-IN",  # Kannada
    "kok-IN",  # Konkani (IN22)
    "ks-IN",  # Kashmiri (IN22)
    "mai-IN",  # Maithili (IN22)
    "ml-IN",  # Malayalam
    "mni-IN",  # Manipuri (IN22)
    "mr-IN",  # Marathi
    "ne-IN",  # Nepali (IN22)
    "od-IN",  # Odia
    "pa-IN",  # Punjabi
    "sa-IN",  # Sanskrit (IN22)
    "sat-IN",  # Santali (IN22)
    "sd-IN",  # Sindhi (IN22)
    "ta-IN",  # Tamil
    "te-IN",  # Telugu
    "ur-IN",  # Urdu (IN22)
]

SarvamTTSSpeakers = Literal[
    # bulbul:v2 Female (lowercase)
    "anushka",
    "manisha",
    "vidya",
    "arya",
    # bulbul:v2 Male (lowercase)
    "abhilash",
    "karun",
    "hitesh",
    # bulbul:v3-beta Customer Care
    "shubh",
    "ritu",
    "rahul",
    "pooja",
    "simran",
    "kavya",
    "amit",
    "ratan",
    "rohan",
    "dev",
    "ishita",
    "shreya",
    "manan",
    "sumit",
    "priya",
    # bulbul:v3-beta Content Creation
    "aditya",
    "kabir",
    "neha",
    "varun",
    "roopa",
    "aayan",
    "ashutosh",
    "advait",
    # bulbul:v3-beta International
    "amelia",
    "sophia",
    # bulbul:v3
    "suhani",
    "rupali",
    "tanya",
    "shruti",
    "kavitha",
]

# bulbul:v4-flash wire names. v4-flash shares no public speaker name with v3, and the
# target language is encoded in the name itself (``_as_``, ``_en_``, ``_enhi_``, ...).
BULBUL_V4_FLASH_SPEAKERS = [
    "kangkana_as_conversational",
    "mouchumi_as_conversational",
    "bappa_bn_conversation",
    "roopa_bn_conversational",
    "bimal_bn_suspense",
    "aditi_en_stories",
    "aparna_en_companion",
    "aparna_en_edtech",
    "ashwin_en_sports",
    "ashwin_en_sports_energetic",
    "chandrika_en_stories",
    "dev_en_recovery",
    "dev_en_conversational",
    "deven_en_conversation",
    "ishita_en_customer",
    "ishita_en_medical",
    "ishita_en_numbers",
    "ishita_en_social",
    "ishita_en_stories",
    "kalpit_en_edtech",
    "nachiket_en_ads",
    "neha_en_customer",
    "neha_en_latenight",
    "nupur_en_kids",
    "ojas_en_social",
    "ritu_en_edtech",
    "ritu_en_latenight",
    "ritu_en_medical",
    "ritu_en_reels",
    "rohan_en_recovery",
    "roopa_en_conversational",
    "rustom_en_suspense",
    "sanchita_en_companion",
    "sanchita_en_insurance",
    "sanchita_en_recovery",
    "sanchita_en_market",
    "sanchita_en_social",
    "shabana_en_edtech",
    "shalini_en_companion",
    "shalini_en_customer",
    "shubh_en_narration",
    "shubh_en_numbers",
    "shubh_en_ads",
    "shubh_en_recovery",
    "shubh_en_audiobook",
    "shubh_en_narration_gentle",
    "shubh_en_sports",
    "simran_en_narration",
    "simran_en_automobile",
    "simran_en_conversation",
    "simran_en_customer",
    "simran_en_edtech",
    "simran_en_edtech_bot",
    "simran_en_sales",
    "simran_en_recovery",
    "simran_en_ads",
    "simran_en_therapist",
    "sunny_en_social",
    "varun_en_ads",
    "varun_en_suspense",
    "zarina_en_conversation",
    "amelia_en_conversational",
    "sophia_en_conversational",
    "girish_en_documentary",
    "girish_en_devotional",
    "payal_en_edtech",
    "sarang_en_narration",
    "ishita_enhi_companion",
    "ishita_enhi_customer",
    "ishita_enhi_customer_expressive",
    "sanchita_enhi_companion",
    "shalini_enhi_companion",
    "shalini_enhi_customer",
    "shubh_enhi_companion",
    "shubh_enhi_ads",
    "shubh_enhi_banking",
    "simran_enhi_companion",
    "simran_enhi_customer",
    "simran_enhi_banking_expressive",
    "sunny_enhi_customer",
    "bhavik_gu_conversation",
    "pooja_gu_conversational",
    "pooja_gu_customer",
    "aayan_hi_conversational",
    "amit_hi_conversational",
    "ashutosh_hi_conversational",
    "kabir_hi_conversational",
    "kavya_hi_conversational",
    "manan_hi_conversational",
    "rahul_hi_conversational",
    "sumit_hi_conversational",
    "aditya_hi_conversational",
    "aditya_hi_sales",
    "anand_hi_documentary",
    "anand_hi_news",
    "aparna_hi_customer",
    "aparna_hi_kyc",
    "ashok_hi_character",
    "ashok_hi_news",
    "chhavi_hi_kids",
    "ishita_hi_ads",
    "ishita_hi_edtech",
    "ishita_hi_banking",
    "ishita_hi_ads_informal",
    "ishita_hi_devotional",
    "ishita_hi_numbers",
    "ishita_hi_social",
    "kunal_hi_kids",
    "mahesh_hi_documentary",
    "mani_hi_devotional",
    "mani_hi_conversational",
    "mohit_hi_conversational",
    "nachiket_hi_devotional",
    "priya_hi_recovery",
    "ratan_hi_latenight",
    "ratan_hi_customer_expressive",
    "ratan_hi_documentary",
    "ratan_hi_devotional",
    "ratan_hi_recovery",
    "ratan_hi_social",
    "ratan_hi_sports",
    "ratan_hi_latenight_warm",
    "rehan_hi_social",
    "ritu_hi_customer_utility",
    "ritu_hi_kids",
    "ritu_hi_conversation",
    "ritu_hi_customer",
    "ritu_hi_edtech",
    "ritu_hi_ads_formal",
    "ritu_hi_banking",
    "ritu_hi_ads_informal",
    "ritu_hi_insurance",
    "ritu_hi_edtech_bot",
    "ritu_hi_medical",
    "ritu_hi_sales",
    "ritu_hi_reels",
    "ritu_hi_social",
    "ritu_hi_customer_warm",
    "ritu_hi_social_lively",
    "roopa_hi_companion",
    "roopa_hi_narration",
    "roopa_hi_recovery",
    "roopa_hi_market",
    "roopa_hi_conversational",
    "sanchita_hi_assistant",
    "sanchita_hi_edtech",
    "sanchita_hi_banking",
    "sanchita_hi_feedback",
    "sanchita_hi_ads_formal",
    "sanchita_hi_ads_informal",
    "sanchita_hi_interview",
    "sanchita_hi_romantic",
    "sanchita_hi_market",
    "sanchita_hi_social",
    "sanchita_hi_kyc",
    "sarika_hi_conversation",
    "shalini_hi_companion",
    "shalini_hi_social",
    "shreya_hi_conversational",
    "shreya_hi_news",
    "shruti_hi_edtech",
    "shubh_hi_customer",
    "shubh_hi_ecomm",
    "shubh_hi_stories_mixed",
    "shubh_hi_devotional",
    "shubh_hi_ads",
    "shubh_hi_recovery",
    "shubh_hi_stories_dramatic",
    "simran_hi_assistant",
    "simran_hi_narration",
    "simran_hi_automobile",
    "simran_hi_conversation",
    "simran_hi_news_breaking",
    "simran_hi_social_energetic",
    "simran_hi_social_excited",
    "simran_hi_latenight",
    "simran_hi_news",
    "simran_hi_recovery",
    "simran_hi_sales",
    "suchitra_hi_ecomm",
    "suhani_hi_social",
    "sunny_hi_ads",
    "sunny_hi_reels",
    "tarun_hi_conversational",
    "tarun_hi_sales",
    "chaitra_hi_customer",
    "shilpa_hi_narration",
    "tanya_hi_narration",
    "aarti_hi_customer",
    "advait_hi_character",
    "aryaman_hi_ads",
    "chirag_hi_social",
    "girish_hi_devotional",
    "mukul_hi_ads",
    "mukul_hi_suspense",
    "suman_hi_companion",
    "vaibhav_hi_social",
    "vandana_hi_ecomm",
    "vipul_hi_social",
    "chaitra_kn_conversation",
    "chaitra_kn_narration",
    "chetan_kn_conversation",
    "suchitra_kn_narration",
    "ishita_mr_conversational",
    "mrunal_mr_narration",
    "neha_mr_narration",
    "nilesh_mr_conversation",
    "ritu_mr_insurance",
    "ritu_mr_narration",
    "rupali_mr_stories",
    "soham_mr_narration",
    "mukul_mr_stories",
    "anand_pa_conversation",
    "anand_pa_customer",
    "harpreet_pa_narration",
    "jaspal_pa_banking",
    "gokul_ta_narration",
    "vetri_ta_ads",
    "vetri_ta_suspense",
    "vijay_ta_narration",
    "kavitha_te_conversation",
    "kavitha_te_narration",
    "pooja_te_conversation",
    "tarun_te_narration",
]

# Model-Speaker compatibility mapping
MODEL_SPEAKER_COMPATIBILITY = {
    "bulbul:v2": {
        "female": ["anushka", "manisha", "vidya", "arya"],
        "male": ["abhilash", "karun", "hitesh"],
        "all": ["anushka", "manisha", "vidya", "arya", "abhilash", "karun", "hitesh"],
    },
    "bulbul:v3-beta": {
        "female": [
            "ritu",
            "pooja",
            "simran",
            "kavya",
            "ishita",
            "shreya",
            "priya",
            "neha",
            "roopa",
            "amelia",
            "sophia",
        ],
        "male": [
            "shubh",
            "rahul",
            "amit",
            "ratan",
            "rohan",
            "dev",
            "manan",
            "sumit",
            "aditya",
            "kabir",
            "varun",
            "aayan",
            "ashutosh",
            "advait",
        ],
        "all": [
            "shubh",
            "ritu",
            "rahul",
            "pooja",
            "simran",
            "kavya",
            "amit",
            "ratan",
            "rohan",
            "dev",
            "ishita",
            "shreya",
            "manan",
            "sumit",
            "priya",
            "aditya",
            "kabir",
            "neha",
            "varun",
            "roopa",
            "aayan",
            "ashutosh",
            "advait",
            "amelia",
            "sophia",
        ],
    },
    "bulbul:v3": {
        "female": [
            "ritu",
            "pooja",
            "simran",
            "kavya",
            "ishita",
            "shreya",
            "priya",
            "neha",
            "roopa",
            "amelia",
            "sophia",
            "suhani",
            "rupali",
            "tanya",
            "shruti",
            "kavitha",
        ],
        "male": [
            "shubh",
            "rahul",
            "amit",
            "ratan",
            "rohan",
            "dev",
            "manan",
            "sumit",
            "aditya",
            "kabir",
            "varun",
            "aayan",
            "ashutosh",
            "advait",
        ],
        "all": [
            "shubh",
            "ritu",
            "rahul",
            "pooja",
            "simran",
            "kavya",
            "amit",
            "ratan",
            "rohan",
            "dev",
            "ishita",
            "shreya",
            "manan",
            "sumit",
            "priya",
            "aditya",
            "kabir",
            "neha",
            "varun",
            "roopa",
            "aayan",
            "ashutosh",
            "advait",
            "amelia",
            "sophia",
            "suhani",
            "rupali",
            "tanya",
            "shruti",
            "kavitha",
        ],
    },
    "bulbul:v4-flash": {"all": BULBUL_V4_FLASH_SPEAKERS},
}


class ConnectionState(enum.Enum):
    """WebSocket connection states for TTS."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    FAILED = "failed"


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


# Accepted [min, max] per synthesis parameter. bulbul:v4-flash is stricter than v2/v3.
_PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "pitch": (-0.75, 0.75),
    "pace": (0.3, 3.0),
    "loudness": (0.5, 2.0),
    "temperature": (0.01, 2.0),
}
_V4_PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "pitch": (-0.5, 0.5),
    "pace": (0.5, 2.0),
    "loudness": (0.1, 2.5),
    "temperature": (0.01, 1.0),
}


def _param_bounds(model: str, param: str) -> tuple[float, float]:
    """Accepted range for a synthesis parameter on the given model."""
    return (_V4_PARAM_BOUNDS if model == "bulbul:v4-flash" else _PARAM_BOUNDS)[param]


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
        word_tokenizer: Tokenizer for processing text
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
    word_tokenizer: tokenize.tokenizer.SentenceTokenizer | None = None
    send_completion_event: bool = True
    output_audio_codec: str = "mp3"


# Streaming bulbul:v4-flash rejects the higher REST sample rates, and OPUS narrows it further.
_V4_STREAM_SAMPLE_RATES = (8000, 16000, 22050, 24000)
_V4_STREAM_OPUS_SAMPLE_RATES = (8000, 16000, 24000)


def _model_extra_fields(opts: SarvamTTSOptions) -> dict[str, object]:
    """Model-specific fields shared by the REST body and the websocket config."""
    extra: dict[str, object] = {}
    if opts.model in ("bulbul:v2", "bulbul:v4-flash"):
        extra["pitch"] = opts.pitch
        extra["loudness"] = opts.loudness
        extra["enable_preprocessing"] = opts.enable_preprocessing
    # v3 and v4 silently ignore caching, so it is only ever sent for v2
    if opts.model == "bulbul:v2" and opts.enable_cached_responses is not None:
        extra["enable_cached_responses"] = opts.enable_cached_responses
    if opts.model in _V3_PIPELINE_MODELS:
        extra["temperature"] = opts.temperature
    if opts.model in ("bulbul:v3", "bulbul:v4-flash") and opts.dict_id is not None:
        extra["dict_id"] = opts.dict_id
    return extra


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
            raise ValueError(
                f"speech_sample_rate must be one of {', '.join(str(r) for r in allowed)} "
                f"when streaming bulbul:v4-flash with codec '{opts.output_audio_codec}'"
            )
        if not url.rstrip("/").endswith("/v2"):
            url = f"{url.rstrip('/')}/v2"
    return f"{url}?model={opts.model}&send_completion_event={opts.send_completion_event}"


_MESSAGE_STATUS_RE = re.compile(r"\s*(\d{3})\s*:")


def _error_status_code(error_data: object) -> int:
    """Extract the HTTP-equivalent status code from a Sarvam websocket error frame.

    Schema rejections carry an integer ``code`` (e.g. 422). Others omit it and prefix
    the message instead, as in ``"400: Speaker '...' is not compatible with model
    bulbul:v4-flash"``. Returns -1 when neither form is present.
    """
    if not isinstance(error_data, dict):
        return -1

    code = error_data.get("code")
    if isinstance(code, bool):  # bool is an int subclass; never a status code
        return -1
    if isinstance(code, int):
        return code

    match = _MESSAGE_STATUS_RE.match(str(error_data.get("message", "")))
    return int(match.group(1)) if match else -1


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
        http_session: Optional aiohttp session to use
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
        http_session: aiohttp.ClientSession | None = None,
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

        # Initialize word tokenizer for streaming
        word_tokenizer = tokenize.basic.SentenceTokenizer()

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
            word_tokenizer=word_tokenizer,
            send_completion_event=send_completion_event,
            output_audio_codec=output_audio_codec,
        )
        self._session = http_session
        self._streams = weakref.WeakSet[SynthesizeStream]()
        # Maps id(ws) -> background keepalive task that pings the server while
        # the connection sits idle in the pool. Sarvam closes idle connections
        # after 60 s; pinging every 30 s keeps them alive for reuse.
        self._ws_keepalive_tasks: dict[int, asyncio.Task[None]] = {}
        # Maps id(ws) -> the options that socket was handshaken with. The pool
        # builds sockets from whatever options this TTS holds at connect time,
        # while each stream carries its own snapshot, so a stream reads these
        # to keep its config frame from contradicting the socket it was handed.
        self._ws_handshake_opts: dict[int, SarvamTTSOptions] = {}

        self._pool = utils.ConnectionPool[aiohttp.ClientWebSocketResponse](
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
            max_session_duration=3600,  # 1 hour
            mark_refreshed_on_get=False,
        )

    async def _connect_ws(self, timeout: float) -> aiohttp.ClientWebSocketResponse:
        session = self._ensure_session()
        headers = {
            "api-subscription-key": self._opts.api_key,
            "User-Agent": USER_AGENT,
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate, br",
        }
        # Add model parameter to URL like the client does
        ws_url = _websocket_url(self._opts)

        logger.info("Connecting to Sarvam TTS WebSocket")

        try:
            ws = await asyncio.wait_for(
                session.ws_connect(
                    ws_url,
                    headers=headers,
                    # Send protocol-level WebSocket PING frames every
                    # ``_WS_HEARTBEAT_INTERVAL`` seconds. aiohttp handles the
                    # PONG accounting on its own and will close the connection
                    # locally if the server stops responding -- this is what
                    # actually keeps the TCP connection alive while it sits
                    # idle in the pool (no ``receive()`` call to auto-pong).
                    heartbeat=_WS_HEARTBEAT_INTERVAL,
                ),
                timeout,
            )
        except Exception as e:
            logger.error(
                "Failed to connect to Sarvam TTS WebSocket",
                extra={"error": str(e), "url": ws_url},
                exc_info=True,
            )
            raise APIConnectionError(f"WebSocket connection failed: {e}") from e

        self._start_keepalive(ws)
        self._ws_handshake_opts[id(ws)] = replace(self._opts)
        return ws

    async def _close_ws(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        await self._stop_keepalive(ws)
        self._ws_handshake_opts.pop(id(ws), None)
        await ws.close()

    def _start_keepalive(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        """Spawn a background task that keeps ``ws`` alive with periodic pings.

        Idempotent: if a live keepalive task is already registered for ``ws``
        the call is a no-op. Callers that need a fresh task must invoke
        ``_stop_keepalive`` first.
        """
        if _KEEPALIVE_INTERVAL <= 0:
            return
        existing = self._ws_keepalive_tasks.get(id(ws))
        if existing is not None and not existing.done():
            return
        task = asyncio.create_task(self._keepalive_loop(ws), name="sarvam-tts-ws-keepalive")
        self._ws_keepalive_tasks[id(ws)] = task

    async def _stop_keepalive(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        """Cancel the keepalive task associated with ``ws`` (if any)."""
        task = self._ws_keepalive_tasks.pop(id(ws), None)
        if task is None or task.done():
            return
        task.cancel()
        with contextlib.suppress(BaseException):
            await task

    async def _keepalive_loop(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        """Keep a pooled WebSocket alive while it sits idle.

        This loop does two things in lockstep:

        1. Actively calls ``ws.receive()`` so that aiohttp can process
           server-initiated PONG frames (and any other messages). aiohttp
           only resets its internal "PONG not received" timer when a PONG
           is read via ``receive()`` -- without an active reader, the
           protocol-level heartbeat will tear the connection down even
           though the server is happily replying.

        2. Sends a Sarvam-defined ``{"type": "ping"}`` JSON message
           whenever ``receive()`` times out (i.e. nothing has come in for
           ``_KEEPALIVE_INTERVAL`` seconds). This resets Sarvam's
           server-side application idle timer (documented as 60 s).

        Any CLOSE/CLOSED/CLOSING/ERROR message from the server, or any
        write failure, evicts the connection from the pool so the next
        checkout creates a fresh one instead of handing the dead one out.
        """
        try:
            while not ws.closed:
                try:
                    msg = await asyncio.wait_for(ws.receive(), timeout=_KEEPALIVE_INTERVAL)
                except asyncio.TimeoutError:
                    # No incoming message within the interval -- send our
                    # app-level ping to reset Sarvam's idle timer.
                    if ws.closed:
                        return
                    try:
                        await ws.send_str(json.dumps({"type": "ping"}))
                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        logger.debug(
                            "Sarvam TTS keepalive ping failed; evicting connection from pool",
                            extra={"error": str(e)},
                        )
                        with contextlib.suppress(Exception):
                            self._pool.remove(ws)
                        return
                    continue

                # We received something. CLOSE/CLOSED/CLOSING/ERROR means
                # the server tore the connection down -- evict and exit.
                if msg.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSING,
                    aiohttp.WSMsgType.ERROR,
                ):
                    logger.debug(
                        "Sarvam TTS WebSocket closed while idle in pool; evicting connection",
                        extra={
                            "msg_type": str(msg.type),
                            "close_code": ws.close_code,
                        },
                    )
                    with contextlib.suppress(Exception):
                        self._pool.remove(ws)
                    return
                # Otherwise -- PONG, TEXT, BINARY, etc. -- discard. We are
                # idle in the pool so any unsolicited TTS traffic from a
                # previous request is no longer relevant. The act of
                # receiving has already reset aiohttp's heartbeat.
        except asyncio.CancelledError:
            pass

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "Sarvam"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
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
    ) -> None:
        """Update TTS options with validation.

        Changes land on a copy that is committed only after every model-dependent
        limit validates against the resulting model. A rejected update therefore
        leaves the live options untouched, and switching models cannot carry over
        a speaker or a pitch/pace/loudness/temperature the new model refuses.
        """
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

        # model and send_completion_event are pinned in the handshake URL (and v4-flash
        # is served on a different path), so a pooled socket would keep synthesising
        # with the previous ones. Retire them; checked-out streams finish untouched.
        reconnect = (opts.model, opts.send_completion_event) != (
            self._opts.model,
            self._opts.send_completion_event,
        )
        self._opts = opts
        if reconnect:
            self._pool.invalidate()

    # Implement the abstract synthesize method
    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions | None = None
    ) -> ChunkedStream:
        """Synthesize text to audio using Sarvam.ai TTS API."""
        if conn_options is None:
            conn_options = DEFAULT_API_CONNECT_OPTIONS
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        """Create a streaming TTS session."""
        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    def prewarm(self) -> None:
        """Prewarm WebSocket connections."""
        self._pool.prewarm()

    async def aclose(self) -> None:
        """Close all active streams and connections."""
        for stream in list(self._streams):
            await stream.aclose()

        self._streams.clear()
        await self._pool.aclose()


class ChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        """Run the Sarvam.ai TTS request and emit audio via the output emitter."""
        payload: dict[str, object] = {
            "target_language_code": self._opts.target_language_code,
            "text": self._input_text,
            "speaker": self._opts.speaker,
            "pace": self._opts.pace,
            "speech_sample_rate": self._opts.speech_sample_rate,
            "model": self._opts.model,
            "output_audio_bitrate": self._opts.output_audio_bitrate,
            "min_buffer_size": self._opts.min_buffer_size,
            "max_chunk_length": self._opts.max_chunk_length,
            "output_audio_codec": self._opts.output_audio_codec,
        }
        payload.update(_model_extra_fields(self._opts))
        headers = {
            "api-subscription-key": self._opts.api_key,
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        }
        mime_type = _codec_to_mime_type(self._opts.output_audio_codec)
        try:
            async with self._tts._ensure_session().post(
                url=self._opts.base_url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(
                    total=self._conn_options.timeout,
                    sock_connect=self._conn_options.timeout,
                ),
            ) as res:
                if res.status != 200:
                    error_text = await res.text()
                    logger.error(f"Sarvam TTS API error: {res.status} - {error_text}")
                    raise APIStatusError(
                        message=f"Sarvam TTS API Error ({res.status}): {error_text}",
                        status_code=res.status,
                        body=error_text,
                    )

                response_json = await res.json()
                request_id = response_json.get("request_id", "")
                audios = response_json.get("audios", [])
                if not audios or not isinstance(audios, list):
                    raise APIConnectionError("Sarvam TTS API response invalid: no audio data")

                output_emitter.initialize(
                    request_id=request_id or "unknown",
                    sample_rate=self._tts.sample_rate,
                    num_channels=self._tts.num_channels,
                    mime_type=mime_type,
                )
                # handle multiple audio chunks
                for b64 in audios:
                    wav_bytes = base64.b64decode(b64)
                    if self._opts.output_audio_codec in _TELEPHONY_CODECS:
                        wav_bytes = _decode_telephony(self._opts.output_audio_codec, wav_bytes)
                    output_emitter.push(wav_bytes)
        except asyncio.TimeoutError as e:
            raise APITimeoutError("Sarvam TTS API request timed out") from e
        except aiohttp.ClientError as e:
            raise APIConnectionError(f"Sarvam TTS API connection error: {e}") from e


class SynthesizeStream(tts.SynthesizeStream):
    """WebSocket-based streaming TTS for Sarvam.ai."""

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions):
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)
        self._segments_ch = utils.aio.Chan[tokenize.SentenceStream]()

        # Connection state management
        self._connection_state = ConnectionState.DISCONNECTED
        self._session_id = id(self)
        self._client_request_id: str | None = None
        self._server_request_id: str | None = None

        # Task management for cleanup
        self._send_task: asyncio.Task | None = None
        self._recv_task: asyncio.Task | None = None
        self._ws_conn: aiohttp.ClientWebSocketResponse | None = None

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        self._segments_ch = utils.aio.Chan[tokenize.SentenceStream]()
        request_id = utils.shortuuid()
        self._client_request_id = request_id
        self._server_request_id = None
        mime_type = _codec_to_mime_type(self._opts.output_audio_codec)
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.speech_sample_rate,
            num_channels=1,
            mime_type=mime_type,
            stream=True,
            frame_size_ms=50,
        )

        async def _tokenize_input() -> None:
            """tokenize text from the input_ch to sentences"""
            word_stream = None
            async for input in self._input_ch:
                if isinstance(input, str):
                    if word_stream is None:
                        tokenizer_instance: tokenize.tokenizer.SentenceTokenizer
                        if self._opts.word_tokenizer is None:
                            # Fallback to basic tokenizer if none provided
                            tokenizer_instance = tokenize.basic.SentenceTokenizer()
                        else:
                            tokenizer_instance = self._opts.word_tokenizer
                        word_stream = tokenizer_instance.stream()
                        self._segments_ch.send_nowait(word_stream)
                    word_stream.push_text(input)
                elif isinstance(input, self._FlushSentinel):
                    if word_stream:
                        word_stream.end_input()
                    word_stream = None

            if word_stream is not None:
                word_stream.end_input()

            self._segments_ch.close()

        async def _process_segments() -> None:
            async for word_stream in self._segments_ch:
                await self._run_ws(word_stream, output_emitter)

        tasks = [
            asyncio.create_task(_tokenize_input()),
            asyncio.create_task(_process_segments()),
        ]
        try:
            await asyncio.gather(*tasks)
        except (APIStatusError, APIConnectionError, APITimeoutError):
            raise
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=request_id, body=None
            ) from None
        except Exception as e:
            raise APIConnectionError(f"TTS stream failed: {e}") from e
        finally:
            await utils.aio.gracefully_cancel(*tasks)
            output_emitter.end_input()

    def _adopt_handshake_opts(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        """Align this stream's model with the socket the pool handed it.

        The pool builds sockets from the TTS's options at connect time while each
        stream carries its own snapshot, so a stream constructed before a model
        switch would otherwise send a config frame for the old model over a socket
        handshaken for the new one -- and v4-flash is served on a different path.

        Only ``model`` is carried by both the handshake and the config frame, so it
        is the only field that has to agree. Everything else in the frame is sent
        per segment and legitimately follows this stream's snapshot.
        """
        handshake = self._tts._ws_handshake_opts.get(id(ws))
        if handshake is None or handshake.model == self._opts.model:
            # The socket already speaks this stream's model, so its speaker and
            # tuning are valid and must win. Taking the socket's instead would
            # revert a config-only update_options for as long as it stays pooled.
            return

        # A different model, so this stream's speaker and tuning bounds are not
        # valid for it; take the whole model-coupled set from the socket. Language,
        # sample rate and codec ride in the config frame and the output emitter is
        # already initialized from them, so those stay snapshotted.
        self._opts = replace(
            handshake,
            target_language_code=self._opts.target_language_code,
            speech_sample_rate=self._opts.speech_sample_rate,
            output_audio_codec=self._opts.output_audio_codec,
        )

    async def _run_ws(
        self, word_stream: tokenize.SentenceStream, output_emitter: tts.AudioEmitter
    ) -> None:
        segment_id = utils.shortuuid()
        output_emitter.start_segment(segment_id=segment_id)

        logger.info(
            "Starting TTS WebSocket session",
            extra={**self._build_log_context(), "user-agent": USER_AGENT},
        )

        input_sent_event = asyncio.Event()

        async def send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            try:
                data: dict[str, object] = {
                    "target_language_code": self._opts.target_language_code,
                    "speaker": self._opts.speaker,
                    "pace": self._opts.pace,
                    "model": self._opts.model,
                    "speech_sample_rate": self._opts.speech_sample_rate,
                    "output_audio_codec": self._opts.output_audio_codec,
                }
                data.update(_model_extra_fields(self._opts))
                if self._opts.model in _V3_PIPELINE_MODELS:
                    data["output_audio_bitrate"] = self._opts.output_audio_bitrate
                    data["min_buffer_size"] = self._opts.min_buffer_size
                    data["max_chunk_length"] = self._opts.max_chunk_length
                config_msg = {"type": "config", "data": data}
                logger.debug(
                    "Sending TTS config",
                    # carries speaker and dict_id, so it is tagged for redaction
                    extra={**self._build_log_context(), "lk.pii.config": config_msg},
                )
                await ws.send_str(json.dumps(config_msg))
                input_sent_event.set()

                started = False
                text_chunks_sent = 0
                async for word in word_stream:
                    if not started:
                        self._mark_started()
                        started = True
                    text_msg = {"type": "text", "data": {"text": word.token}}
                    await ws.send_str(json.dumps(text_msg))
                    text_chunks_sent += 1

                flush_msg = {"type": "flush"}
                await ws.send_str(json.dumps(flush_msg))

            except (ConnectionResetError, RuntimeError) as e:
                err_str = str(e).lower()
                if "closing" in err_str or "closed" in err_str:
                    # The transport is dead -- almost always a stale pooled
                    # connection that the server already closed (60s idle
                    # timeout). Raise so the pool evicts it via the
                    # ``async with`` __aexit__; the framework will retry
                    # against a fresh connection. Real user interruptions
                    # propagate as ``CancelledError``, not this branch.
                    logger.debug(
                        "Sarvam TTS WebSocket transport closed before send "
                        "completed; pool will replace this connection",
                        extra=self._build_log_context(),
                    )
                    raise APIConnectionError(
                        "Sarvam TTS WebSocket transport closed before send completed"
                    ) from e
                logger.error(
                    f"Error in send task: {e}", extra=self._build_log_context(), exc_info=True
                )
                raise APIConnectionError(f"Send task failed: {e}") from e
            except Exception as e:
                logger.error(
                    f"Error in send task: {e}", extra=self._build_log_context(), exc_info=True
                )
                raise APIConnectionError(f"Send task failed: {e}") from e
            finally:
                input_sent_event.set()

        async def recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            await input_sent_event.wait()
            try:
                while True:
                    msg = await ws.receive(timeout=self._conn_options.timeout)

                    if msg.type in (
                        aiohttp.WSMsgType.CLOSE,
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.CLOSING,
                    ):
                        close_code = ws.close_code if ws.close_code is not None else msg.data
                        close_reason = msg.extra
                        is_expected_close = close_code in (1000, 1001, None)
                        if not is_expected_close:
                            logger.error(
                                "WebSocket connection closed by server",
                                extra={
                                    **self._build_log_context(),
                                    "close_code": close_code,
                                    "close_reason": close_reason,
                                },
                            )
                            raw_close = {
                                "msg_type": str(msg.type),
                                "close_code": close_code,
                                "close_reason": close_reason,
                            }
                            raise APIStatusError(
                                message=(
                                    "Sarvam TTS WebSocket closed with non-graceful status: "
                                    f"{json.dumps(raw_close, ensure_ascii=False)}"
                                ),
                                status_code=int(close_code) if isinstance(close_code, int) else -1,
                                body=raw_close,
                            )
                        logger.info(
                            "WebSocket connection closed by server",
                            extra={
                                **self._build_log_context(),
                                "close_code": close_code,
                                "close_reason": close_reason,
                            },
                        )
                        break

                    if msg.type == aiohttp.WSMsgType.TEXT:
                        success = await self._handle_websocket_message(msg.data, output_emitter)
                        if not success:
                            break

                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        error_msg = f"WebSocket error: {msg.data}"
                        logger.error(error_msg, extra=self._build_log_context())
                        raise APIConnectionError(error_msg)

            except asyncio.TimeoutError as e:
                logger.error("WebSocket received timeout", extra=self._build_log_context())
                raise APITimeoutError("WebSocket receive timeout") from e
            except (APIStatusError, APIConnectionError, APITimeoutError):
                raise
            except ConnectionResetError as e:
                err_str = str(e).lower()
                if "closing" in err_str or "closed" in err_str:
                    # Same reasoning as ``send_task``: the transport is dead
                    # (typically a stale pooled connection). Raise so the
                    # pool evicts it and the framework retries against a
                    # fresh connection. Real interruptions surface as
                    # ``CancelledError``.
                    logger.debug(
                        "Sarvam TTS WebSocket transport closed during"
                        " receive; pool will replace this connection",
                        extra=self._build_log_context(),
                    )
                    raise APIConnectionError(
                        "Sarvam TTS WebSocket transport closed during receive"
                    ) from e
                logger.error(
                    f"Error in receive task: {e}", extra=self._build_log_context(), exc_info=True
                )
                raise
            except Exception as e:
                logger.error(
                    f"Error in receive task: {e}", extra=self._build_log_context(), exc_info=True
                )
                raise

        try:
            async with self._tts._pool.connection(timeout=self._conn_options.timeout) as ws:
                # Pause the keepalive task while we own the connection so its
                # ping frames don't interleave with config / text / flush
                # traffic. The task was started either by ``_connect_ws`` (for
                # a fresh connection) or by the previous ``_run_ws`` invocation
                # that returned this connection to the pool.
                await self._tts._stop_keepalive(ws)
                keepalive_should_resume = False
                self._adopt_handshake_opts(ws)

                try:
                    self._acquire_time = self._tts._pool.last_acquire_time
                    self._connection_reused = self._tts._pool.last_connection_reused
                    self._ws_conn = ws
                    self._connection_state = ConnectionState.CONNECTED

                    logger.info("WebSocket connected successfully", extra=self._build_log_context())

                    self._send_task = asyncio.create_task(send_task(ws))
                    self._recv_task = asyncio.create_task(recv_task(ws))

                    tasks = [self._send_task, self._recv_task]

                    try:
                        await asyncio.gather(*tasks)
                        logger.info(
                            "WebSocket session completed successfully",
                            extra=self._build_log_context(),
                        )
                        keepalive_should_resume = True
                    finally:
                        input_sent_event.set()
                        await utils.aio.gracefully_cancel(*tasks)
                        self._send_task = None
                        self._recv_task = None
                finally:
                    # Resume the keepalive only when the session completed
                    # cleanly. On exception the pool will discard the
                    # connection via ``remove(conn)`` and ``_close_ws`` will
                    # run, so restarting here would be wasted work (and the
                    # task would be cancelled immediately anyway).
                    if keepalive_should_resume:
                        self._tts._start_keepalive(ws)

        except (aiohttp.ClientConnectorError, asyncio.TimeoutError) as e:
            self._connection_state = ConnectionState.FAILED
            logger.error(f"Connection failed: {e}", extra=self._build_log_context())
            raise APIConnectionError(f"Failed to connect to TTS WebSocket: {e}") from e
        except (APIStatusError, APIConnectionError, APITimeoutError):
            self._connection_state = ConnectionState.FAILED
            raise
        except Exception as e:
            self._connection_state = ConnectionState.FAILED
            logger.error(
                f"Unexpected error in WebSocket session: {e}",
                extra=self._build_log_context(),
                exc_info=True,
            )
            raise APIStatusError(f"TTS WebSocket session failed: {e}") from e
        finally:
            self._connection_state = ConnectionState.DISCONNECTED
            self._ws_conn = None

    async def _handle_websocket_message(
        self, msg_data: str, output_emitter: tts.AudioEmitter
    ) -> bool:
        """Handle WebSocket message with proper error handling.

        Returns:
            True if processing should continue, False if stream should end
        """
        try:
            resp = json.loads(msg_data)
            msg_type = resp.get("type")
            self._maybe_set_server_request_id(resp)
            if self._server_request_id:
                # Expose the server-assigned request id on the tts_request_run span so
                # users can correlate traces with Sarvam's logs for debugging. Deduped
                # internally, so calling on every message is fine.
                output_emitter._note_provider_request_id(self._server_request_id)

            if not msg_type:
                logger.warning(
                    "Received message without type field",
                    extra={**self._build_log_context(), "lk.pii.data": resp},
                )
                return True

            # logger.debug(f"Processing message type: {msg_type}", extra=self._build_log_context())

            if msg_type == "audio":
                return await self._handle_audio_message(resp, output_emitter)
            elif msg_type == "error":
                await self._handle_error_message(resp)
                return False  # Stop processing on error
            elif msg_type == "event":
                return await self._handle_event_message(resp, output_emitter)
            else:
                logger.debug(f"Unknown message type: {msg_type}", extra=self._build_log_context())
                return True

        except json.JSONDecodeError as e:
            logger.warning(
                f"Invalid JSON in WebSocket message: {e}",
                extra={**self._build_log_context(), "lk.pii.raw_data": msg_data[:200]},
            )
            return True  # Continue processing
        except (APIStatusError, APIConnectionError):
            # Preserve provider-originated status/body/retry metadata.
            raise
        except Exception as e:
            logger.error(
                f"Error processing WebSocket message: {e}",
                extra=self._build_log_context(),
                exc_info=True,
            )
            raise APIStatusError(
                message=f"Message processing error: {e}. Raw server message: {msg_data}",
                body={"raw_message": msg_data},
            ) from e

    async def _handle_audio_message(self, resp: dict, output_emitter: tts.AudioEmitter) -> bool:
        """Handle audio message with proper error handling."""
        try:
            audio_data = resp.get("data", {}).get("audio", "")
            if not audio_data:
                logger.debug("Received empty audio data", extra=self._build_log_context())
                return True

            audio_bytes = base64.b64decode(audio_data)
            if self._opts.output_audio_codec in _TELEPHONY_CODECS:
                audio_bytes = _decode_telephony(self._opts.output_audio_codec, audio_bytes)
            output_emitter.push(audio_bytes)

            return True

        except Exception as e:  # base64 decode error
            logger.error(f"Invalid base64 audio data: {e}", extra=self._build_log_context())
            # Don't stop processing for audio decode errors
            return True

    async def _handle_error_message(self, resp: dict) -> None:
        """Handle error messages from the API."""
        error_data = resp.get("data", {})
        error_msg = error_data.get("message", "Unknown error")
        error_code = _error_status_code(error_data)

        # The provider decides what goes in these fields, so they are tagged for
        # redaction and kept out of the log body, which collectors cannot redact.
        # This is the one place the frame is recorded in full.
        logger.error(
            "TTS API error",
            extra={
                **self._build_log_context(),
                "error_code": error_code,
                "lk.pii.error_message": error_msg,
                "lk.pii.raw_message": resp,
            },
        )

        # APIStatusError derives retryability from the status code (4xx permanent
        # except 408/429/499, 5xx transient), so forwarding Sarvam's own code is
        # all that is needed -- an unrecognized frame stays retryable via -1.
        #
        # Nothing provider-written goes on the exception: its __str__ renders both
        # message and body, and the framework logs that with %s when it retries,
        # where no collector can redact it. status_code and request_id carry the
        # non-PII identifiers a caller needs to correlate against Sarvam's logs.
        raise APIStatusError(
            message=f"TTS API error from Sarvam (status {error_code})",
            status_code=error_code,
            request_id=error_data.get("request_id") if isinstance(error_data, dict) else None,
        )

    async def _handle_event_message(self, resp: dict, output_emitter: tts.AudioEmitter) -> bool:
        """Handle event messages from the API."""
        event_data = resp.get("data", {})
        event_type = event_data.get("event_type")
        self._maybe_set_server_request_id(event_data)

        if event_type == "final":
            logger.debug("Generation complete event received", extra=self._build_log_context())
            output_emitter.end_input()
            return False  # Stop processing
        else:
            logger.debug(f"Unknown event type: {event_type}", extra=self._build_log_context())
            return True

    def _build_log_context(self) -> dict:
        """Build consistent logging context."""
        return {
            "session_id": self._session_id,
            "connection_state": self._connection_state.value,
            "model": self._opts.model,
            "speaker": self._opts.speaker,
            "client_request_id": self._client_request_id,
            "server_request_id": self._server_request_id,
        }

    def _maybe_set_server_request_id(self, data: dict) -> None:
        """Capture server-assigned request_id once it is available."""
        if self._server_request_id is not None:
            return

        request_id = None
        if isinstance(data, dict):
            request_id = data.get("request_id")
            if request_id is None:
                nested = data.get("data")
                if isinstance(nested, dict):
                    request_id = nested.get("request_id")
                metadata = data.get("metadata")
                if request_id is None and isinstance(metadata, dict):
                    request_id = metadata.get("request_id")

        if request_id:
            self._server_request_id = str(request_id)

    async def aclose(self) -> None:
        """Close the stream and cleanup resources.

        The base class cancels ``_run`` (and therefore all child tasks
        spawned inside ``_run_ws``).  The ``ConnectionPool`` context manager
        handles WebSocket cleanup on cancellation.  We only need to close
        the segments channel so that ``_process_segments`` unblocks.
        """
        self._segments_ch.close()
        await super().aclose()
