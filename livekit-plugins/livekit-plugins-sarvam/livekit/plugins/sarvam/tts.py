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

PATCH NOTE:
    Added a keepalive ping task in _run_ws that sends {"type": "ping"} every
    KEEPALIVE_INTERVAL seconds while the WebSocket is idle between agent turns.
    This prevents Sarvam's server from closing the connection due to inactivity,
    eliminating the "Cannot write to closing transport" error and the resulting
    audible silence during reconnect.
"""

from __future__ import annotations

import asyncio
import base64
import enum
import json
import os
import weakref
from dataclasses import dataclass, replace
from typing import Literal

import aiohttp

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tokenize,
    tts,
    utils,
)

from .log import logger

SARVAM_TTS_BASE_URL = "https://api.sarvam.ai/text-to-speech"
SARVAM_TTS_WS_URL = "wss://api.sarvam.ai/text-to-speech/ws"

# How often (in seconds) to send a ping to keep the WS alive between turns
KEEPALIVE_INTERVAL = 10  # seconds — safely below any reasonable idle timeout

# Sarvam TTS specific models and speakers
SarvamTTSModels = Literal["bulbul:v2", "bulbul:v3-beta", "bulbul:v3"]
SarvamTTSOutputAudioBitrate = Literal["32k", "64k", "96k", "128k", "192k"]

ALLOWED_OUTPUT_AUDIO_BITRATES: set[str] = {"32k", "64k", "96k", "128k", "192k"}
ALLOWED_OUTPUT_AUDIO_CODECS: set[str] = {"linear16", "mulaw", "alaw", "opus", "flac", "aac", "wav", "mp3"}

SarvamTTSOutputAudioCodec = Literal["linear16", "mulaw", "alaw", "opus", "flac", "aac", "wav", "mp3"]

# Supported languages in BCP-47 format
SarvamTTSLanguages = Literal[
    "bn-IN",  # Bengali
    "en-IN",  # English (India)
    "gu-IN",  # Gujarati
    "hi-IN",  # Hindi
    "kn-IN",  # Kannada
    "ml-IN",  # Malayalam
    "mr-IN",  # Marathi
    "od-IN",  # Odia
    "pa-IN",  # Punjabi
    "ta-IN",  # Tamil
    "te-IN",  # Telugu
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


@dataclass
class SarvamTTSOptions:
    """Options for the Sarvam.ai TTS service."""

    target_language_code: SarvamTTSLanguages | str
    api_key: str
    text: str | None = None
    speaker: SarvamTTSSpeakers | str | None = None
    pitch: float = 0.0
    pace: float = 1.0
    loudness: float = 1.0
    temperature: float = 0.6
    output_audio_bitrate: SarvamTTSOutputAudioBitrate | str = "128k"
    output_audio_codec: SarvamTTSOutputAudioCodec | str = "mp3"
    min_buffer_size: int = 50
    max_chunk_length: int = 150
    speech_sample_rate: int = 22050
    enable_preprocessing: bool = False
    model: SarvamTTSModels | str = "bulbul:v2"
    base_url: str = SARVAM_TTS_BASE_URL
    ws_url: str = SARVAM_TTS_WS_URL
    word_tokenizer: tokenize.tokenizer.SentenceTokenizer | None = None
    send_completion_event: bool = True


class TTS(tts.TTS):
    """Sarvam.ai Text-to-Speech implementation."""

    def __init__(
        self,
        *,
        target_language_code: SarvamTTSLanguages | str,
        model: SarvamTTSModels | str = "bulbul:v2",
        speaker: SarvamTTSSpeakers | str | None = None,
        speech_sample_rate: int = 22050,
        num_channels: int = 1,
        pitch: float = 0.0,
        pace: float = 1.0,
        loudness: float = 1.0,
        temperature: float = 0.6,
        output_audio_bitrate: SarvamTTSOutputAudioBitrate | str = "128k",
        output_audio_codec: SarvamTTSOutputAudioCodec | str = "mp3",
        min_buffer_size: int = 50,
        max_chunk_length: int = 150,
        enable_preprocessing: bool = False,
        api_key: str | None = None,
        base_url: str = SARVAM_TTS_BASE_URL,
        ws_url: str = SARVAM_TTS_WS_URL,
        http_session: aiohttp.ClientSession | None = None,
        send_completion_event: bool = True,
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

        if not target_language_code or not target_language_code.strip():
            raise ValueError("Target language code is required and cannot be empty")
        if not model or not model.strip():
            raise ValueError("Model is required and cannot be empty")
        if speaker is None:
            if model == "bulbul:v3-beta" or model == "bulbul:v3":
                speaker = "shubh"
            else:
                speaker = "anushka"

        # Fix: default sample rate for v3-beta should be 24000, not 22050
        if speech_sample_rate == 22050 and (model == "bulbul:v3-beta" or model == "bulbul:v3"):
            speech_sample_rate = 24000

        # Validate parameter ranges
        # Fix: pitch suitable range is -0.75 to 0.75 per Sarvam docs
        if not -0.75 <= pitch <= 0.75:
            raise ValueError("Pitch must be between -0.75 and 0.75")
        # Fix: pace range differs per model — v2 supports 0.3-3.0, v3/v3-beta supports 0.5-2.0
        if model == "bulbul:v2":
            if not 0.3 <= pace <= 3.0:
                raise ValueError("Pace must be between 0.3 and 3.0 for bulbul:v2")
        else:
            if not 0.5 <= pace <= 2.0:
                raise ValueError("Pace must be between 0.5 and 2.0 for bulbul:v3/v3-beta")
        if not 0.5 <= loudness <= 2.0:
            raise ValueError("Loudness must be between 0.5 and 2.0")
        if not 0.01 <= temperature <= 1.0:
            raise ValueError("Temperature must be between 0.01 and 1.0")
        if output_audio_bitrate not in ALLOWED_OUTPUT_AUDIO_BITRATES:
            raise ValueError(
                f"output_audio_bitrate must be one of {', '.join(sorted(ALLOWED_OUTPUT_AUDIO_BITRATES))}"
            )
        if output_audio_codec not in ALLOWED_OUTPUT_AUDIO_CODECS:
            raise ValueError(
                f"output_audio_codec must be one of {', '.join(sorted(ALLOWED_OUTPUT_AUDIO_CODECS))}"
            )
        if not 30 <= min_buffer_size <= 200:
            raise ValueError("min_buffer_size must be between 30 and 200")
        if not 50 <= max_chunk_length <= 500:
            raise ValueError("max_chunk_length must be between 50 and 500")
        if speech_sample_rate not in [8000, 16000, 22050, 24000]:
            raise ValueError("Sample rate must be 8000, 16000, 22050, or 24000 Hz")

        if not validate_model_speaker_compatibility(model, speaker):
            compatible_speakers = MODEL_SPEAKER_COMPATIBILITY.get(model, {}).get("all", [])
            raise ValueError(
                f"Speaker '{speaker}' is not compatible with model '{model}'. "
                f"Please choose a compatible speaker from: {', '.join(compatible_speakers)}"
            )

        word_tokenizer = tokenize.basic.SentenceTokenizer()

        self._opts = SarvamTTSOptions(
            target_language_code=target_language_code,
            model=model,
            speaker=speaker,
            speech_sample_rate=speech_sample_rate,
            pitch=pitch,
            pace=pace,
            loudness=loudness,
            temperature=temperature,
            output_audio_bitrate=output_audio_bitrate,
            output_audio_codec=output_audio_codec,
            min_buffer_size=min_buffer_size,
            max_chunk_length=max_chunk_length,
            enable_preprocessing=enable_preprocessing,
            api_key=self._api_key,
            base_url=base_url,
            ws_url=ws_url,
            word_tokenizer=word_tokenizer,
            send_completion_event=send_completion_event,
        )
        self._session = http_session
        self._streams = weakref.WeakSet[SynthesizeStream]()

        self._pool = utils.ConnectionPool[aiohttp.ClientWebSocketResponse](
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
            max_session_duration=3600,
            mark_refreshed_on_get=False,
        )

    async def _connect_ws(self, timeout: float) -> aiohttp.ClientWebSocketResponse:
        session = self._ensure_session()
        headers = {
            "api-subscription-key": self._opts.api_key,
            "User-Agent": "LiveKit-Sarvam-TTS/1.0",
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate, br",
        }
        ws_url = f"{self._opts.ws_url}?model={self._opts.model}&send_completion_event={self._opts.send_completion_event}"

        logger.info("Connecting to Sarvam TTS WebSocket")

        try:
            return await asyncio.wait_for(
                session.ws_connect(
                    ws_url,
                    headers=headers,
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

    async def _close_ws(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        await ws.close()

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
        speaker: str | None = None,
        pitch: float | None = None,
        pace: float | None = None,
        loudness: float | None = None,
        temperature: float | None = None,
        output_audio_bitrate: SarvamTTSOutputAudioBitrate | str | None = None,
        min_buffer_size: int | None = None,
        max_chunk_length: int | None = None,
        enable_preprocessing: bool | None = None,
        send_completion_event: bool | None = None,
        output_audio_codec: SarvamTTSOutputAudioCodec | str | None = None,
    ) -> None:
        """Update TTS options with validation."""
        if model is not None:
            if not model.strip():
                raise ValueError("Model cannot be empty")
            self._opts.model = model
            if speaker is None and self._opts.speaker is not None:
                if not validate_model_speaker_compatibility(self._opts.model, self._opts.speaker):
                    compatible_speakers = MODEL_SPEAKER_COMPATIBILITY.get(self._opts.model, {}).get(
                        "all", []
                    )
                    raise ValueError(
                        f"Speaker '{self._opts.speaker}' incompatible with {self._opts.model}. "
                        f"Compatible speakers: {', '.join(compatible_speakers)}"
                    )
        if speaker is not None:
            if not speaker.strip():
                raise ValueError("Speaker cannot be empty")
            if not validate_model_speaker_compatibility(self._opts.model, speaker):
                compatible_speakers = MODEL_SPEAKER_COMPATIBILITY.get(self._opts.model, {}).get(
                    "all", []
                )
                raise ValueError(
                    f"Speaker '{speaker}' incompatible with {self._opts.model}. "
                    f"Compatible speakers: {', '.join(compatible_speakers)}"
                )
            self._opts.speaker = speaker

        if pitch is not None:
            if not -20.0 <= pitch <= 20.0:
                raise ValueError("Pitch must be between -20.0 and 20.0")
            self._opts.pitch = pitch

        if pace is not None:
            if not 0.5 <= pace <= 2.0:
                raise ValueError("Pace must be between 0.5 and 2.0")
            self._opts.pace = pace

        if loudness is not None:
            if not 0.5 <= loudness <= 2.0:
                raise ValueError("Loudness must be between 0.5 and 2.0")
            self._opts.loudness = loudness

        if temperature is not None:
            if not 0.01 <= temperature <= 1.0:
                raise ValueError("Temperature must be between 0.01 and 1.0")
            self._opts.temperature = temperature

        if output_audio_bitrate is not None:
            if output_audio_bitrate not in ALLOWED_OUTPUT_AUDIO_BITRATES:
                raise ValueError(
                    "output_audio_bitrate must be one of "
                    f"{', '.join(sorted(ALLOWED_OUTPUT_AUDIO_BITRATES))}"
                )
            self._opts.output_audio_bitrate = output_audio_bitrate

        if min_buffer_size is not None:
            if not 30 <= min_buffer_size <= 200:
                raise ValueError("min_buffer_size must be between 30 and 200")
            self._opts.min_buffer_size = min_buffer_size

        if max_chunk_length is not None:
            if not 50 <= max_chunk_length <= 500:
                raise ValueError("max_chunk_length must be between 50 and 500")
            self._opts.max_chunk_length = max_chunk_length

        if enable_preprocessing is not None:
            self._opts.enable_preprocessing = enable_preprocessing

        if send_completion_event is not None:
            self._opts.send_completion_event = send_completion_event

        if output_audio_codec is not None:
            if output_audio_codec not in ALLOWED_OUTPUT_AUDIO_CODECS:
                raise ValueError(
                    f"output_audio_codec must be one of {', '.join(sorted(ALLOWED_OUTPUT_AUDIO_CODECS))}"
                )
            self._opts.output_audio_codec = output_audio_codec

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
        payload = {
            "target_language_code": self._opts.target_language_code,
            "text": self._input_text,
            "speaker": self._opts.speaker,
            "pace": self._opts.pace,
            "speech_sample_rate": self._opts.speech_sample_rate,
            "model": self._opts.model,
            "output_audio_bitrate": self._opts.output_audio_bitrate,
            "min_buffer_size": self._opts.min_buffer_size,
            "max_chunk_length": self._opts.max_chunk_length,
        }
        if self._opts.model == "bulbul:v2":
            payload["pitch"] = self._opts.pitch
            payload["loudness"] = self._opts.loudness
            payload["enable_preprocessing"] = self._opts.enable_preprocessing
        if self._opts.model in ("bulbul:v3", "bulbul:v3-beta"):
            payload["temperature"] = self._opts.temperature
        headers = {
            "api-subscription-key": self._opts.api_key,
            "Content-Type": "application/json",
        }
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
                        message="Sarvam TTS API Error",
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
                    mime_type="audio/mp3",
                )
                for b64 in audios:
                    wav_bytes = base64.b64decode(b64)
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
        self._keepalive_task: asyncio.Task | None = None  # PATCH: keepalive task handle
        self._ws_conn: aiohttp.ClientWebSocketResponse | None = None

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        self._client_request_id = request_id
        self._server_request_id = None
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.speech_sample_rate,
            num_channels=1,
            mime_type="audio/mp3",
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
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=request_id, body=None
            ) from None
        except Exception as e:
            raise APIConnectionError() from e
        finally:
            await utils.aio.gracefully_cancel(*tasks)
            output_emitter.end_input()

    async def _run_ws(
        self, word_stream: tokenize.SentenceStream, output_emitter: tts.AudioEmitter
    ) -> None:
        segment_id = utils.shortuuid()
        output_emitter.start_segment(segment_id=segment_id)

        logger.info("Starting TTS WebSocket session", extra=self._build_log_context())

        # ------------------------------------------------------------------
        # PATCH: keepalive_task — sends {"type": "ping"} every KEEPALIVE_INTERVAL
        # seconds to prevent Sarvam's server from closing an idle connection
        # between agent turns. Cancelled as soon as send_task starts writing
        # real data, and also on session end.
        # ------------------------------------------------------------------
        async def keepalive_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            ping_msg = json.dumps({"type": "ping"})
            try:
                while True:
                    await asyncio.sleep(KEEPALIVE_INTERVAL)
                    if ws.closed:
                        logger.debug("Keepalive: WS already closed, stopping", extra=self._build_log_context())
                        break
                    await ws.send_str(ping_msg)
                    logger.debug("Keepalive ping sent", extra=self._build_log_context())
            except asyncio.CancelledError:
                pass  # Normal — cancelled when send_task takes over or session ends
            except Exception as e:
                logger.warning(f"Keepalive ping failed: {e}", extra=self._build_log_context())
        # ------------------------------------------------------------------

        async def send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            try:
                # PATCH: cancel keepalive before writing real data so they
                # don't interleave on the socket.
                if self._keepalive_task and not self._keepalive_task.done():
                    self._keepalive_task.cancel()
                    try:
                        await self._keepalive_task
                    except asyncio.CancelledError:
                        pass
                    self._keepalive_task = None

                # PATCH: check connection is still alive before writing.
                # The pool may return a stale connection that Sarvam has already
                # closed server-side. Detecting it here avoids writing to a
                # closing transport and causing audible silence.
                if ws.closed or ws.close_code is not None:
                    raise APIConnectionError(
                        f"Stale WebSocket from pool (closed={ws.closed}, "
                        f"close_code={ws.close_code}), forcing fresh connection"
                    )

                # Send initial config
                data: dict[str, object] = {
                    "target_language_code": self._opts.target_language_code,
                    "speaker": self._opts.speaker,
                    "pace": self._opts.pace,
                    "model": self._opts.model,
                    "output_audio_bitrate": self._opts.output_audio_bitrate,
                    "output_audio_codec": self._opts.output_audio_codec,
                    # Fix: docs define speech_sample_rate as string enum
                    "speech_sample_rate": str(self._opts.speech_sample_rate),
                    "min_buffer_size": self._opts.min_buffer_size,
                    "max_chunk_length": self._opts.max_chunk_length,
                }
                if self._opts.model == "bulbul:v2":
                    data["pitch"] = self._opts.pitch
                    data["loudness"] = self._opts.loudness
                    data["enable_preprocessing"] = self._opts.enable_preprocessing
                if self._opts.model in ("bulbul:v3", "bulbul:v3-beta"):
                    data["temperature"] = self._opts.temperature
                config_msg = {"type": "config", "data": data}
                logger.debug(
                    "Sending TTS config", extra={**self._build_log_context(), "config": config_msg}
                )
                await ws.send_str(json.dumps(config_msg))

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

            except Exception as e:
                logger.error(
                    f"Error in send task: {e}", extra=self._build_log_context(), exc_info=True
                )
                raise APIConnectionError(f"Send task failed: {e}") from e

        async def recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            try:
                while True:
                    msg = await ws.receive(timeout=self._conn_options.timeout)

                    if msg.type in (
                        aiohttp.WSMsgType.CLOSE,
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.CLOSING,
                    ):
                        logger.info(
                            "WebSocket connection closed by server", extra=self._build_log_context()
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
            except Exception as e:
                logger.error(
                    f"Error in receive task: {e}", extra=self._build_log_context(), exc_info=True
                )
                raise

        try:
            async with self._tts._pool.connection(timeout=self._conn_options.timeout) as ws:
                self._ws_conn = ws
                self._connection_state = ConnectionState.CONNECTED

                logger.info("WebSocket connected successfully", extra=self._build_log_context())

                # PATCH: start keepalive immediately after getting the connection
                # from the pool — this keeps the connection alive if it was idle
                # in the pool between agent turns.
                self._keepalive_task = asyncio.create_task(keepalive_task(ws))

                self._send_task = asyncio.create_task(send_task(ws))
                self._recv_task = asyncio.create_task(recv_task(ws))

                tasks = [self._send_task, self._recv_task]

                try:
                    await asyncio.gather(*tasks)
                    logger.info(
                        "WebSocket session completed successfully", extra=self._build_log_context()
                    )
                except Exception as e:
                    logger.error(
                        f"WebSocket session failed: {e}",
                        extra=self._build_log_context(),
                        exc_info=True,
                    )
                    raise
                finally:
                    # PATCH: always cancel keepalive on session end
                    if self._keepalive_task and not self._keepalive_task.done():
                        self._keepalive_task.cancel()
                        try:
                            await self._keepalive_task
                        except asyncio.CancelledError:
                            pass
                    self._keepalive_task = None

                    await utils.aio.gracefully_cancel(*tasks)
                    self._send_task = None
                    self._recv_task = None

        except (aiohttp.ClientConnectorError, asyncio.TimeoutError) as e:
            self._connection_state = ConnectionState.FAILED
            logger.error(f"Connection failed: {e}", extra=self._build_log_context())
            raise APIConnectionError(f"Failed to connect to TTS WebSocket: {e}") from e
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
        try:
            resp = json.loads(msg_data)
            msg_type = resp.get("type")
            self._maybe_set_server_request_id(resp)

            if not msg_type:
                logger.warning(
                    "Received message without type field",
                    extra={**self._build_log_context(), "data": resp},
                )
                return True

            if msg_type == "audio":
                return await self._handle_audio_message(resp, output_emitter)
            elif msg_type == "error":
                await self._handle_error_message(resp)
                return False
            elif msg_type == "event":
                return await self._handle_event_message(resp, output_emitter)
            else:
                logger.debug(f"Unknown message type: {msg_type}", extra=self._build_log_context())
                return True

        except json.JSONDecodeError as e:
            logger.warning(
                f"Invalid JSON in WebSocket message: {e}",
                extra={**self._build_log_context(), "raw_data": msg_data[:200]},
            )
            return True
        except Exception as e:
            logger.error(
                f"Error processing WebSocket message: {e}",
                extra=self._build_log_context(),
                exc_info=True,
            )
            raise APIStatusError(f"Message processing error: {e}") from e

    async def _handle_audio_message(self, resp: dict, output_emitter: tts.AudioEmitter) -> bool:
        try:
            audio_data = resp.get("data", {}).get("audio", "")
            if not audio_data:
                logger.debug("Received empty audio data", extra=self._build_log_context())
                return True

            audio_bytes = base64.b64decode(audio_data)
            output_emitter.push(audio_bytes)
            return True

        except Exception as e:
            logger.error(f"Invalid base64 audio data: {e}", extra=self._build_log_context())
            return True

    async def _handle_error_message(self, resp: dict) -> None:
        error_data = resp.get("data", {})
        error_msg = error_data.get("message", "Unknown error")
        error_code = error_data.get("code", "unknown")

        logger.error(
            f"TTS API error: {error_msg}",
            extra={
                **self._build_log_context(),
                "error_code": error_code,
                "error_message": error_msg,
            },
        )

        recoverable_errors = ["rate_limit", "temporary_unavailable", "timeout"]
        is_recoverable = any(err in str(error_msg).lower() for err in recoverable_errors)

        if is_recoverable:
            raise APIConnectionError(f"Recoverable TTS API error: {error_msg}")
        else:
            raise APIStatusError(message=f"TTS API error: {error_msg}", status_code=500)

    async def _handle_event_message(self, resp: dict, output_emitter: tts.AudioEmitter) -> bool:
        event_data = resp.get("data", {})
        event_type = event_data.get("event_type")
        self._maybe_set_server_request_id(event_data)

        if event_type == "final":
            logger.debug("Generation complete event received", extra=self._build_log_context())
            output_emitter.end_input()
            return False
        else:
            logger.debug(f"Unknown event type: {event_type}", extra=self._build_log_context())
            return True

    def _build_log_context(self) -> dict:
        return {
            "session_id": self._session_id,
            "connection_state": self._connection_state.value,
            "model": self._opts.model,
            "speaker": self._opts.speaker,
            "client_request_id": self._client_request_id,
            "server_request_id": self._server_request_id,
        }

    def _maybe_set_server_request_id(self, data: dict) -> None:
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
        logger.debug("Starting TTS stream cleanup", extra=self._build_log_context())

        self._connection_state = ConnectionState.DISCONNECTED

        # PATCH: cancel keepalive task on stream close
        if self._keepalive_task and not self._keepalive_task.done():
            self._keepalive_task.cancel()
            try:
                await self._keepalive_task
            except asyncio.CancelledError:
                pass
            self._keepalive_task = None

        tasks_to_cancel = []
        for task_attr in ["_send_task", "_recv_task"]:
            task = getattr(self, task_attr, None)
            if task and not task.done():
                tasks_to_cancel.append(task)

        if tasks_to_cancel:
            for task in tasks_to_cancel:
                task.cancel()
            try:
                await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
            except Exception as e:
                logger.warning(f"Error cancelling tasks: {e}", extra=self._build_log_context())

        if self._ws_conn and not self._ws_conn.closed:
            try:
                await self._ws_conn.close()
                logger.debug("WebSocket connection closed", extra=self._build_log_context())
            except Exception as e:
                logger.warning(f"Error closing WebSocket: {e}", extra=self._build_log_context())

        for channel_name, channel in [("segments", self._segments_ch), ("input", self._input_ch)]:
            try:
                if hasattr(channel, "closed") and not channel.closed:
                    if hasattr(channel, "close"):
                        channel.close()
                    logger.debug(f"{channel_name} channel closed", extra=self._build_log_context())
            except Exception as e:
                logger.warning(
                    f"Error closing {channel_name} channel: {e}", extra=self._build_log_context()
                )

        try:
            await super().aclose()
        except Exception as e:
            logger.warning(f"Error in parent cleanup: {e}", extra=self._build_log_context())
        finally:
            self._client_request_id = None
            self._server_request_id = None