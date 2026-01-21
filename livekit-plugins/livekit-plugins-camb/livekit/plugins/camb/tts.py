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

import os
from dataclasses import replace

import httpx

from camb.client import AsyncCambAI
from camb.core.api_error import ApiError
from camb.types.stream_tts_output_configuration import StreamTtsOutputConfiguration
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
    MODEL_SAMPLE_RATES,
    NUM_CHANNELS,
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
        user_instructions: str | None = None,
        # Audio configuration
        output_format: OutputFormat = DEFAULT_OUTPUT_FORMAT,
        enhance_named_entities: bool = False,
        sample_rate: int | None = None,
        # HTTP client
        http_session: httpx.AsyncClient | None = None,
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
            model: MARS model to use ('mars-flash', 'mars-pro', 'mars-instruct').
            user_instructions: Style/tone guidance (3-1000 chars, requires mars-instruct model).
            output_format: Audio output format (default: 'pcm_s16le').
            enhance_named_entities: Enhanced pronunciation for named entities.
            sample_rate: Audio sample rate in Hz. If None, auto-detected from model.
            http_session: Optional httpx.AsyncClient session to reuse.
        """
        # Get sample rate from model if not specified
        resolved_sample_rate = sample_rate or MODEL_SAMPLE_RATES.get(model, 22050)

        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=resolved_sample_rate,
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

        # SDK client management
        self._http_session = http_session
        self._client: AsyncCambAI | None = None
        self._close_client_on_cleanup = http_session is None

        # Configuration
        self._opts = _TTSOptions(
            voice_id=voice_id,
            language=language,
            speech_model=model,
            output_format=output_format,
            user_instructions=user_instructions,
            enhance_named_entities=enhance_named_entities,
        )

    def _ensure_client(self) -> AsyncCambAI:
        """Lazily create the SDK client."""
        if self._client is None:
            if not self._api_key:
                raise ValueError("API key is required but not set")
            self._client = AsyncCambAI(
                api_key=self._api_key,
                base_url=self._base_url,
                httpx_client=self._http_session,
            )
        return self._client

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
        user_instructions: str | None = None,
    ) -> None:
        """Update TTS options dynamically."""
        if voice_id is not None:
            self._opts.voice_id = voice_id
        if language is not None:
            self._opts.language = language
        if model is not None:
            self._opts.speech_model = model
            # Update sample rate to match the new model
            self._sample_rate = MODEL_SAMPLE_RATES.get(model, 22050)
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
        if self._close_client_on_cleanup and self._client is not None:
            await self._client.aclose()
            self._client = None


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
        Main synthesis logic using the Camb.ai SDK.

        Flow:
        1. Get SDK client from TTS instance
        2. Call SDK's streaming TTS method
        3. Initialize AudioEmitter with audio format details
        4. Stream audio chunks to output_emitter
        5. Handle errors and map to LiveKit exceptions
        """
        client = self._tts._ensure_client()

        # Log request for debugging
        logger.debug(
            f"Camb.ai TTS request: voice_id={self._opts.voice_id}, "
            f"model={self._opts.speech_model}, text_length={len(self._input_text)}"
        )

        # Prepare output configuration
        output_config = StreamTtsOutputConfiguration(
            format=self._opts.output_format,
            sample_rate=self._tts._sample_rate,
        )

        # Determine MIME type based on output format
        if self._opts.output_format in ("pcm_s16le", "pcm_s32le"):
            mime_type = "audio/pcm"
        elif self._opts.output_format == "wav":
            mime_type = "audio/wav"
        elif self._opts.output_format == "flac":
            mime_type = "audio/flac"
        else:  # adts or other
            mime_type = "audio/aac"

        try:
            # Call SDK's streaming TTS method
            stream = client.text_to_speech.tts(
                text=self._input_text,
                voice_id=self._opts.voice_id,
                language=self._opts.language,
                speech_model=self._opts.speech_model,
                user_instructions=self._opts.user_instructions,
                enhance_named_entities_pronunciation=self._opts.enhance_named_entities,
                output_configuration=output_config,
            )

            # Initialize audio emitter
            request_id = utils.shortuuid()
            output_emitter.initialize(
                request_id=request_id,
                sample_rate=self._tts._sample_rate,
                num_channels=NUM_CHANNELS,
                mime_type=mime_type,
            )

            # Stream audio chunks from SDK
            async for chunk in stream:
                output_emitter.push(chunk)

            output_emitter.flush()

        except httpx.ConnectError as e:
            raise APIConnectionError(f"Camb.ai connection failed: {str(e)}") from e
        except httpx.TimeoutException:
            raise APITimeoutError("Camb.ai TTS request timed out") from None
        except ApiError as e:
            raise APIStatusError(
                f"Camb.ai TTS request failed: {e.body}",
                status_code=e.status_code or 500,
                request_id=e.headers.get("x-request-id") if e.headers else None,
                body=e.body,
            ) from e
        except Exception as e:
            # Re-raise LiveKit exceptions
            if isinstance(e, (APIStatusError, APIConnectionError, APITimeoutError)):
                raise
            raise APIConnectionError(f"Unexpected error: {str(e)}") from e
