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
    MODEL_SAMPLE_RATES,
    NUM_CHANNELS,
    OutputFormat,
    SpeechModel,
    _TTSOptions,
)

API_BASE_URL = "https://client.camb.ai/apis"
API_KEY_HEADER = "x-api-key"


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = API_BASE_URL,
        credentials_info: NotGivenOr[dict] = NOT_GIVEN,  # Future Vertex AI
        credentials_file: NotGivenOr[str] = NOT_GIVEN,  # Future Vertex AI
        voice_id: int = DEFAULT_VOICE_ID,
        language: str = DEFAULT_LANGUAGE,
        model: SpeechModel = DEFAULT_MODEL,
        user_instructions: str | None = None,
        output_format: OutputFormat = DEFAULT_OUTPUT_FORMAT,
        enhance_named_entities: bool = False,
        sample_rate: int | None = None,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """
        Create a new instance of Camb.ai TTS.

        ``api_key`` must be set to your Camb.ai API key, either using the argument or by
        setting the ``CAMB_API_KEY`` environmental variable.

        Args:
            api_key: Camb.ai API key. If not provided, reads from CAMB_API_KEY env var.
            base_url: Camb.ai API base URL.
            credentials_info: GCP credentials dict for Vertex AI (future support).
            credentials_file: GCP credentials file path for Vertex AI (future support).
            voice_id: Voice ID to use. Use list_voices() to discover available voices.
            language: BCP-47 locale (e.g., 'en-us', 'fr-fr').
            model: MARS model to use ('mars-flash', 'mars-pro', 'mars-instruct').
            user_instructions: Style/tone guidance (3-1000 chars, requires mars-instruct).
            output_format: Audio output format (default: 'pcm_s16le').
            enhance_named_entities: Enhanced pronunciation for named entities.
            sample_rate: Audio sample rate in Hz. If None, auto-detected from model.
            http_session: Optional aiohttp.ClientSession to reuse.
        """
        resolved_sample_rate = sample_rate or MODEL_SAMPLE_RATES.get(model, 22050)

        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=resolved_sample_rate,
            num_channels=NUM_CHANNELS,
        )

        self._api_key = api_key or os.environ.get("CAMB_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Camb.ai API key must be provided via api_key parameter or "
                "CAMB_API_KEY environment variable"
            )

        if is_given(credentials_info) or is_given(credentials_file):
            logger.warning("Vertex AI credentials provided but not yet implemented - using API key")

        self._credentials_info = credentials_info
        self._credentials_file = credentials_file
        self._base_url = base_url
        self._session = http_session

        self._opts = _TTSOptions(
            voice_id=voice_id,
            language=language,
            speech_model=model,
            output_format=output_format,
            user_instructions=user_instructions,
            enhance_named_entities=enhance_named_entities,
        )

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

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
        pass


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
        logger.debug(
            f"Camb.ai TTS request: voice_id={self._opts.voice_id}, "
            f"model={self._opts.speech_model}, text_length={len(self._input_text)}"
        )

        # Determine MIME type based on output format
        if self._opts.output_format in ("pcm_s16le", "pcm_s32le"):
            mime_type = "audio/pcm"
        elif self._opts.output_format == "wav":
            mime_type = "audio/wav"
        elif self._opts.output_format == "flac":
            mime_type = "audio/flac"
        else:  # adts
            mime_type = "audio/aac"

        # Build request payload
        payload: dict = {
            "text": self._input_text,
            "voice_id": self._opts.voice_id,
            "language": self._opts.language,
            "speech_model": self._opts.speech_model,
            "enhance_named_entities_pronunciation": self._opts.enhance_named_entities,
            "output_configuration": {
                "format": self._opts.output_format,
            },
        }
        if self._opts.user_instructions:
            payload["user_instructions"] = self._opts.user_instructions

        try:
            headers: dict[str, str] = {"Content-Type": "application/json"}
            if self._tts._api_key:
                headers[API_KEY_HEADER] = self._tts._api_key

            async with self._tts._ensure_session().post(
                f"{self._tts._base_url}/tts-stream",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(
                    total=60,
                    sock_connect=self._conn_options.timeout,
                ),
            ) as resp:
                if resp.status != 200:
                    content = await resp.text()
                    raise APIStatusError(
                        "Camb.ai TTS failed",
                        status_code=resp.status,
                        request_id=resp.headers.get("x-request-id"),
                        body=content,
                    )

                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=self._tts._sample_rate,
                    num_channels=NUM_CHANNELS,
                    mime_type=mime_type,
                )

                async for data, _ in resp.content.iter_chunks():
                    output_emitter.push(data)

                output_emitter.flush()

        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message,
                status_code=e.status,
                request_id=None,
                body=None,
            ) from e
        except Exception as e:
            if isinstance(e, (APIStatusError, APIConnectionError, APITimeoutError)):
                raise
            raise APIConnectionError() from e
