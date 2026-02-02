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

# Sarvam TTS specific models and speakers
SarvamTTSModels = Literal["bulbul:v2", "bulbul:v3-beta"]

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
    """Options for the Sarvam.ai TTS service.

    Args:
        target_language_code: BCP-47 language code for supported Indian languages
        api_key: Sarvam.ai API key
        text: The text to synthesize (will be provided by stream adapter)
        speaker: Voice to use for synthesis
        pitch: Voice pitch adjustment (-20.0 to 20.0)
        pace: Speech rate multiplier (0.5 to 2.0)
        loudness: Volume multiplier (0.5 to 2.0)
        speech_sample_rate: Audio sample rate (8000, 16000, 22050, or 24000)
        enable_preprocessing: Whether to use text preprocessing
        model: The Sarvam TTS model to use
        base_url: API endpoint URL
        ws_url: WebSocket endpoint URL
        word_tokenizer: Tokenizer for processing text
    """

    target_language_code: SarvamTTSLanguages | str  # BCP-47 for supported Indian languages
    api_key: str  # Sarvam.ai API key
    text: str | None = None  # Will be provided by the stream adapter
    speaker: SarvamTTSSpeakers | str = "anushka"  # Default speaker compatible with v2
    pitch: float = 0.0
    pace: float = 1.0
    loudness: float = 1.0
    speech_sample_rate: int = 22050  # Default 22050 Hz
    enable_preprocessing: bool = False
    model: SarvamTTSModels | str = "bulbul:v2"  # Default to v2 as it has more recent speakers
    base_url: str = SARVAM_TTS_BASE_URL
    ws_url: str = SARVAM_TTS_WS_URL
    word_tokenizer: tokenize.tokenizer.SentenceTokenizer | None = None
    send_completion_event: bool = True


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
        pitch: Voice pitch adjustment (-20.0 to 20.0) - only supported in v2 for now
        pace: Speech rate multiplier (0.5 to 2.0)
        loudness: Volume multiplier (0.5 to 2.0) - only supported in v2 for now
        enable_preprocessing: Whether to use text preprocessing
        api_key: Sarvam.ai API key (required)
        base_url: API endpoint URL
        ws_url: WebSocket endpoint URL
        http_session: Optional aiohttp session to use
    """

    def __init__(
        self,
        *,
        target_language_code: SarvamTTSLanguages | str,
        model: SarvamTTSModels | str = "bulbul:v2",
        speaker: SarvamTTSSpeakers | str = "anushka",
        speech_sample_rate: int = 22050,
        num_channels: int = 1,  # Sarvam output is mono WAV
        pitch: float = 0.0,
        pace: float = 1.0,
        loudness: float = 1.0,
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

        # Validate inputs early
        if not target_language_code or not target_language_code.strip():
            raise ValueError("Target language code is required and cannot be empty")
        if not model or not model.strip():
            raise ValueError("Model is required and cannot be empty")
        if not speaker or not speaker.strip():
            raise ValueError("Speaker is required and cannot be empty")

        # Validate parameter ranges
        if not -20.0 <= pitch <= 20.0:
            raise ValueError("Pitch must be between -20.0 and 20.0")
        if not 0.5 <= pace <= 2.0:
            raise ValueError("Pace must be between 0.5 and 2.0")
        if not 0.5 <= loudness <= 2.0:
            raise ValueError("Loudness must be between 0.5 and 2.0")
        if speech_sample_rate not in [8000, 16000, 22050, 24000]:
            raise ValueError("Sample rate must be 8000, 16000, 22050, or 24000 Hz")

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
            target_language_code=target_language_code,
            model=model,
            speaker=speaker,
            speech_sample_rate=speech_sample_rate,
            pitch=pitch,
            pace=pace,
            loudness=loudness,
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
            max_session_duration=3600,  # 1 hour
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
        # Add model parameter to URL like the client does
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
        enable_preprocessing: bool | None = False,
        send_completion_event: bool | None = True,
    ) -> None:
        """Update TTS options with validation."""
        if model is not None:
            if not model.strip():
                raise ValueError("Model cannot be empty")
            self._opts.model = model

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

        if enable_preprocessing is not None:
            self._opts.enable_preprocessing = enable_preprocessing

        if send_completion_event is not None:
            self._opts.send_completion_event = send_completion_event

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
        payload = {
            "target_language_code": self._opts.target_language_code,
            "text": self._input_text,
            "speaker": self._opts.speaker,
            "pace": self._opts.pace,
            "speech_sample_rate": self._opts.speech_sample_rate,
            "enable_preprocessing": self._opts.enable_preprocessing,
            "model": self._opts.model,
        }
        # Only include pitch and loudness for v2 model (not supported in v3-beta)
        if self._opts.model == "bulbul:v2":
            payload["pitch"] = self._opts.pitch
            payload["loudness"] = self._opts.loudness
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
                        message=f"Sarvam TTS API Error: {error_text}",
                        status_code=res.status,
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
                # handle multiple audio chunks
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

        async def send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            try:
                # Send initial config
                config_msg = {
                    "type": "config",
                    "data": {
                        "target_language_code": self._opts.target_language_code,
                        "speaker": self._opts.speaker,
                        "pitch": self._opts.pitch,
                        "pace": self._opts.pace,
                        "loudness": self._opts.loudness,
                        "enable_preprocessing": self._opts.enable_preprocessing,
                        "model": self._opts.model,
                    },
                }
                logger.debug(
                    "Sending TTS config", extra={**self._build_log_context(), "config": config_msg}
                )
                await ws.send_str(json.dumps(config_msg))

                # Count text chunks sent
                started = False
                text_chunks_sent = 0
                # Send text chunks
                async for word in word_stream:
                    if not started:
                        self._mark_started()
                        started = True
                    text_msg = {"type": "text", "data": {"text": word.token}}
                    await ws.send_str(json.dumps(text_msg))
                    text_chunks_sent += 1

                # Send flush signal
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
                            break  # Stop processing on error or completion

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

        # Use connection pool for WebSocket management
        try:
            async with self._tts._pool.connection(timeout=self._conn_options.timeout) as ws:
                self._ws_conn = ws
                self._connection_state = ConnectionState.CONNECTED

                logger.info("WebSocket connected successfully", extra=self._build_log_context())

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
                    # Gracefully cancel tasks
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
        """Handle WebSocket message with proper error handling.

        Returns:
            True if processing should continue, False if stream should end
        """
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

            logger.debug(f"Processing message type: {msg_type}", extra=self._build_log_context())

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
                extra={**self._build_log_context(), "raw_data": msg_data[:200]},
            )
            return True  # Continue processing
        except Exception as e:
            logger.error(
                f"Error processing WebSocket message: {e}",
                extra=self._build_log_context(),
                exc_info=True,
            )
            raise APIStatusError(f"Message processing error: {e}") from e

    async def _handle_audio_message(self, resp: dict, output_emitter: tts.AudioEmitter) -> bool:
        """Handle audio message with proper error handling."""
        try:
            audio_data = resp.get("data", {}).get("audio", "")
            if not audio_data:
                logger.debug("Received empty audio data", extra=self._build_log_context())
                return True

            audio_bytes = base64.b64decode(audio_data)
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
        error_code = error_data.get("code", "unknown")

        logger.error(
            f"TTS API error: {error_msg}",
            extra={
                **self._build_log_context(),
                "error_code": error_code,
                "error_message": error_msg,
            },
        )

        # Determine if error is recoverable based on error code/type
        recoverable_errors = ["rate_limit", "temporary_unavailable", "timeout"]
        is_recoverable = any(err in str(error_msg).lower() for err in recoverable_errors)

        if is_recoverable:
            raise APIConnectionError(f"Recoverable TTS API error: {error_msg}")
        else:
            raise APIStatusError(message=f"TTS API error: {error_msg}", status_code=500)

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
        """Close the stream and cleanup resources."""
        logger.debug("Starting TTS stream cleanup", extra=self._build_log_context())

        self._connection_state = ConnectionState.DISCONNECTED

        # Cancel running tasks first
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

        # Close WebSocket connection
        if self._ws_conn and not self._ws_conn.closed:
            try:
                await self._ws_conn.close()
                logger.debug("WebSocket connection closed", extra=self._build_log_context())
            except Exception as e:
                logger.warning(f"Error closing WebSocket: {e}", extra=self._build_log_context())

        # Close channels
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

        # Call parent cleanup
        try:
            await super().aclose()
        except Exception as e:
            logger.warning(f"Error in parent cleanup: {e}", extra=self._build_log_context())
        finally:
            self._client_request_id = None
            self._server_request_id = None
