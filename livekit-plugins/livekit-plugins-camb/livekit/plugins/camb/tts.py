# Copyright 2023 LiveKit, Inc.
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
import os
from dataclasses import replace
from typing import Any

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from .log import logger
from .models import (
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_FORMAT,
    DEFAULT_VOICE_ID,
    NUM_CHANNELS,
    SAMPLE_RATE,
    OutputFormat,
    SpeechModel,
    _TTSOptions,
)


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        # Authentication
        api_key: str | None = None,
        base_url: str = "https://client.camb.ai/apis",
        credentials_info: NotGivenOr[dict] = NOT_GIVEN,  # Future Vertex AI
        credentials_file: NotGivenOr[str] = NOT_GIVEN,  # Future Vertex AI
        # Voice selection
        voice_id: int = DEFAULT_VOICE_ID,
        language: str = DEFAULT_LANGUAGE,
        # Model selection
        model: SpeechModel = DEFAULT_MODEL,
        # Voice parameters
        speed: float = 1.0,
        user_instructions: str | None = None,
        # Audio configuration
        output_format: OutputFormat = DEFAULT_OUTPUT_FORMAT,
        enhance_named_entities: bool = False,
        sample_rate: int = SAMPLE_RATE,
        # HTTP client
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """
        Create a new instance of Camb.ai TTS.

        ``api_key`` must be set to your Camb.ai API key, either using the argument or by setting the
        ``CAMB_API_KEY`` environmental variable.

        Args:
            api_key: Camb.ai API key. If not provided, reads from CAMB_API_KEY env var.
            base_url: Camb.ai API base URL.
            credentials_info: GCP credentials dict for Vertex AI (future support).
            credentials_file: GCP credentials file path for Vertex AI (future support).
            voice_id: Voice ID to use. Use list_voices() to discover available voices.
            language: BCP-47 locale (e.g., 'en-us', 'fr-fr').
            model: MARS model to use ('mars-8', 'mars-8-flash', 'mars-8-instruct', etc.).
            speed: Speech rate (default: 1.0).
            user_instructions: Style/tone guidance (3-1000 chars, requires mars-8-instruct model).
            output_format: Audio output format (default: 'pcm_s16le').
            enhance_named_entities: Enhanced pronunciation for named entities.
            sample_rate: Audio sample rate in Hz (default: 24000).
            http_session: Optional aiohttp session to reuse.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )

        # Authentication
        self._api_key = api_key or os.environ.get("CAMB_API_KEY")
        if not self._api_key and not is_given(credentials_info) and not is_given(credentials_file):
            raise ValueError(
                "Camb.ai API key must be provided via api_key parameter or "
                "CAMB_API_KEY environment variable"
            )

        # TODO: Vertex AI client initialization when details available
        if is_given(credentials_info) or is_given(credentials_file):
            logger.warning("Vertex AI support not yet implemented")

        self._credentials_info = credentials_info
        self._credentials_file = credentials_file
        self._base_url = base_url

        # HTTP session management
        self._session = http_session
        self._close_session_on_cleanup = http_session is None

        # Configuration
        self._opts = _TTSOptions(
            voice_id=voice_id,
            language=language,
            speech_model=model,
            speed=speed,
            output_format=output_format,
            user_instructions=user_instructions,
            enhance_named_entities=enhance_named_entities,
        )

    @property
    def model(self) -> str:
        return self._opts.speech_model

    @property
    def provider(self) -> str:
        return "Camb.ai"

    def update_options(
        self,
        *,
        voice_id: int | None = None,
        language: str | None = None,
        model: SpeechModel | None = None,
        speed: float | None = None,
        user_instructions: str | None = None,
    ) -> None:
        """Update TTS options dynamically."""
        if voice_id is not None:
            self._opts.voice_id = voice_id
        if language is not None:
            self._opts.language = language
        if model is not None:
            self._opts.speech_model = model
        if speed is not None:
            self._opts.speed = speed
        if user_instructions is not None:
            self._opts.user_instructions = user_instructions

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    async def aclose(self) -> None:
        if self._close_session_on_cleanup and self._session:
            await self._session.close()


class ChunkedStream(tts.ChunkedStream):
    def __init__(
        self,
        *,
        tts: TTS,
        input_text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        """
        Main synthesis logic using POST /tts-stream endpoint.

        Flow:
        1. Prepare request payload
        2. Make HTTP POST to /tts-stream with streaming response
        3. Initialize AudioEmitter with audio format details
        4. Stream audio chunks to output_emitter
        5. Handle errors and map to LiveKit exceptions
        """

        # Prepare request payload
        payload: dict[str, Any] = {
            "text": self._input_text,
            "voice_id": self._opts.voice_id,
            "language": self._opts.language,
            "speech_model": self._opts.speech_model,
            "output_configuration": {
                "format": self._opts.output_format,
            },
            "voice_settings": {
                "speed": self._opts.speed,
            },
            "enhance_named_entities_pronunciation": self._opts.enhance_named_entities,
        }

        # Add user instructions if provided (requires mars-8-instruct model)
        if self._opts.user_instructions:
            payload["user_instructions"] = self._opts.user_instructions

        # Ensure we have an API key
        if not self._tts._api_key:
            raise ValueError("API key is required but not set")

        headers = {
            "x-api-key": self._tts._api_key,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        # Log request for debugging
        logger.debug(
            f"Camb.ai TTS request: url={self._tts._base_url}/tts-stream, "
            f"voice_id={self._opts.voice_id}, model={self._opts.speech_model}, "
            f"text_length={len(self._input_text)}"
        )

        # Session management
        session = self._tts._session
        close_session = False
        if session is None:
            session = aiohttp.ClientSession()
            close_session = True

        try:
            # Make streaming request with longer timeout for TTS synthesis
            # Use max of conn_options.timeout or 60 seconds for TTS
            timeout_seconds = max(self._conn_options.timeout, 60.0)
            async with session.post(
                f"{self._tts._base_url}/tts-stream",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout_seconds),
            ) as response:
                # Check for errors
                if response.status != 200:
                    error_body = await response.text()
                    raise APIStatusError(
                        f"Camb.ai TTS request failed: {error_body}",
                        status_code=response.status,
                        request_id=response.headers.get("x-request-id"),
                        body=error_body,
                    )

                # Initialize audio emitter
                request_id = response.headers.get("x-request-id", utils.shortuuid())

                # Determine MIME type based on output format
                # pcm_s16le/pcm_s32le return raw PCM, not WAV
                if self._opts.output_format in ("pcm_s16le", "pcm_s32le"):
                    mime_type = "audio/pcm"
                elif self._opts.output_format == "wav":
                    mime_type = "audio/wav"
                elif self._opts.output_format == "flac":
                    mime_type = "audio/flac"
                else:  # adts or other
                    mime_type = "audio/aac"

                output_emitter.initialize(
                    request_id=request_id,
                    sample_rate=self._tts._sample_rate,
                    num_channels=NUM_CHANNELS,
                    mime_type=mime_type,
                )

                # Stream audio chunks
                async for chunk in response.content.iter_chunked(8192):
                    if chunk:
                        output_emitter.push(chunk)

                output_emitter.flush()

        except aiohttp.ClientError as e:
            raise APIConnectionError(f"Camb.ai connection failed: {str(e)}") from e
        except asyncio.TimeoutError:
            raise APITimeoutError("Camb.ai TTS request timed out") from None
        except Exception as e:
            # Re-raise LiveKit exceptions
            if isinstance(e, (APIStatusError, APIConnectionError, APITimeoutError)):
                raise
            raise APIConnectionError(f"Unexpected error: {str(e)}") from e
        finally:
            if close_session:
                await session.close()
