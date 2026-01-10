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
from dataclasses import dataclass, replace

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
from .models import DEFAULT_VOICE_ID, OutputOptions, PromptOptions, TTSLanguages, TTSModels, Voice

API_BASE_URL = "https://api.typecast.ai/v1/text-to-speech"
AUTHORIZATION_HEADER = "X-API-KEY"
NUM_CHANNELS = 1


@dataclass
class _TTSOptions:
    model: str
    voice: str
    api_key: str
    base_url: str
    sample_rate: int
    language: str | None
    seed: int | None
    prompt_options: PromptOptions
    output_options: OutputOptions


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        model: TTSModels | str = "ssfm-v21",
        voice: str = DEFAULT_VOICE_ID,
        api_key: str | None = None,
        language: NotGivenOr[TTSLanguages | str] = NOT_GIVEN,
        seed: NotGivenOr[int] = NOT_GIVEN,
        prompt_options: NotGivenOr[PromptOptions] = NOT_GIVEN,
        output_options: NotGivenOr[OutputOptions] = NOT_GIVEN,
        sample_rate: int = 44100,
        http_session: aiohttp.ClientSession | None = None,
        base_url: str = API_BASE_URL,
    ) -> None:
        """
        Create a new instance of Typecast TTS.

        Args:
            model: TTS model to use (default: "ssfm-v21")
            voice: Voice ID for synthesis (default: DEFAULT_VOICE_ID)
            api_key: Typecast API key. If not provided, will use TYPECAST_API_KEY environment variable
            language: Language code in ISO 639-3 format (e.g., "eng", "kor", "jpn")
            seed: Random seed for reproducible synthesis
            prompt_options: Options for emotional expression (emotion_preset, emotion_intensity)
            output_options: Options for audio output (volume, audio_pitch, audio_tempo, audio_format)
            sample_rate: Audio sample rate in Hz (default: 44100)
            http_session: Optional aiohttp ClientSession
            base_url: API endpoint URL
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )

        api_key = api_key or os.environ.get("TYPECAST_API_KEY")
        if not api_key:
            raise ValueError(
                "Typecast API key is required. "
                "Set TYPECAST_API_KEY environment variable or provide api_key parameter."
            )

        self._opts = _TTSOptions(
            model=model,
            voice=voice,
            api_key=api_key,
            base_url=base_url,
            sample_rate=sample_rate,
            language=language if is_given(language) else None,
            seed=seed if is_given(seed) else None,
            prompt_options=prompt_options if is_given(prompt_options) else PromptOptions(),
            output_options=output_options if is_given(output_options) else OutputOptions(),
        )
        self._session = http_session

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "Typecast"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    async def list_voices(self, *, model: TTSModels | str | None = None) -> list[Voice]:
        """
        List all available Typecast voices.

        Args:
            model: Optional model filter (e.g., "ssfm-v21"). If not provided, returns voices for all models.

        Returns:
            List of Voice objects containing id, name, model, and supported emotions.

        Example:
            ```python
            tts = typecast.TTS()
            voices = await tts.list_voices()
            for voice in voices:
                print(f"{voice.name} ({voice.id})")
                print(f"  Emotions: {', '.join(voice.emotions)}")
            ```
        """
        # Extract base URL from text-to-speech endpoint
        base_url = self._opts.base_url.replace("/text-to-speech", "")
        url = f"{base_url}/voices"

        params = {}
        if model:
            params["model"] = model

        try:
            async with self._ensure_session().get(
                url,
                headers={AUTHORIZATION_HEADER: self._opts.api_key},
                params=params,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    error_body = None
                    try:
                        error_body = await resp.json()
                    except Exception:
                        error_body = await resp.text()

                    raise APIStatusError(
                        message=resp.reason or "Failed to list voices",
                        status_code=resp.status,
                        request_id=None,
                        body=error_body,
                    )

                data = await resp.json()

                return [
                    Voice(
                        id=v["voice_id"],
                        name=v["voice_name"],
                        model=v["model"],
                        emotions=v["emotions"],
                    )
                    for v in data
                ]

        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message,
                status_code=e.status,
                request_id=None,
                body=None,
            ) from None
        except Exception as e:
            if isinstance(e, (APIStatusError, APITimeoutError)):
                raise
            raise APIConnectionError() from e

    def update_options(
        self,
        *,
        model: NotGivenOr[str] = NOT_GIVEN,
        voice: NotGivenOr[str] = NOT_GIVEN,
        language: NotGivenOr[TTSLanguages | str] = NOT_GIVEN,
        seed: NotGivenOr[int] = NOT_GIVEN,
        prompt_options: NotGivenOr[PromptOptions] = NOT_GIVEN,
        output_options: NotGivenOr[OutputOptions] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
    ) -> None:
        """
        Update TTS options.

        Args:
            model: TTS model
            voice: Voice ID
            language: Language code (ISO 639-3)
            seed: Random seed
            prompt_options: Emotion settings
            output_options: Audio output settings
            sample_rate: Sample rate in Hz
        """
        if is_given(model):
            self._opts.model = model
        if is_given(voice):
            self._opts.voice = voice
        if is_given(language):
            self._opts.language = language
        if is_given(seed):
            self._opts.seed = seed
        if is_given(prompt_options):
            self._opts.prompt_options = prompt_options
        if is_given(output_options):
            self._opts.output_options = output_options
        if is_given(sample_rate):
            self._opts.sample_rate = sample_rate

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> ChunkedStream:
        """
        Synthesize text to speech.

        Args:
            text: Text to synthesize
            conn_options: API connection options

        Returns:
            ChunkedStream: Audio stream
        """
        return ChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
        )


class ChunkedStream(tts.ChunkedStream):
    """
    Chunked stream for Typecast TTS synthesis.

    Note: Typecast API returns the complete audio file in one response,
    but we chunk it for compatibility with the LiveKit Agents framework.
    """

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
        Internal method to run TTS synthesis.
        Fetches audio from Typecast API and emits it in chunks.
        """
        request_id = utils.shortuuid()

        # Build API payload
        payload: dict = {
            "text": self._input_text,
            "model": self._opts.model,
            "voice_id": self._opts.voice,
            "prompt": self._opts.prompt_options.to_dict(),
            "output": self._opts.output_options.to_dict(),
        }

        # Add optional parameters
        if self._opts.language is not None:
            payload["language"] = self._opts.language
        if self._opts.seed is not None:
            payload["seed"] = self._opts.seed

        headers = {
            "Content-Type": "application/json",
            AUTHORIZATION_HEADER: self._opts.api_key,
        }

        logger.debug(
            "Typecast TTS request",
            extra={
                "text_length": len(self._input_text),
                "model": self._opts.model,
                "voice": self._opts.voice,
                "language": self._opts.language,
            },
        )

        try:
            async with self._tts._ensure_session().post(
                self._opts.base_url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(
                    total=30,
                    sock_connect=self._conn_options.timeout,
                ),
            ) as resp:
                if resp.status != 200:
                    error_body = None
                    try:
                        error_body = await resp.json()
                    except Exception:
                        error_body = await resp.text()

                    raise APIStatusError(
                        message=resp.reason or "Unknown error occurred",
                        status_code=resp.status,
                        request_id=request_id,
                        body=error_body,
                    )

                # Initialize output emitter with audio/wav mime type
                # AudioEmitter will handle WAV header parsing automatically
                output_emitter.initialize(
                    request_id=request_id,
                    sample_rate=self._opts.sample_rate,
                    num_channels=NUM_CHANNELS,
                    mime_type="audio/wav",
                )

                # Read and emit audio in chunks
                async for chunk in resp.content.iter_chunked(1024):
                    if chunk:
                        output_emitter.push(chunk)

                output_emitter.flush()

        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message,
                status_code=e.status,
                request_id=request_id,
                body=None,
            ) from None
        except Exception as e:
            if isinstance(e, (APIStatusError, APITimeoutError)):
                raise
            raise APIConnectionError() from e
