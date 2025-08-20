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

"""Speech-to-Text implementation for Sarvam.ai

This module provides an STT implementation that uses the Sarvam.ai API.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Literal

import aiohttp

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    stt,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer

from .log import logger

# Sarvam API details
SARVAM_STT_BASE_URL = "https://api.sarvam.ai/speech-to-text"

# Supported Sarvam models
SarvamSTTModels = Literal["saarika:v1", "saarika:v2", "saarika:flash", "saarika:v2.5"]


@dataclass
class SarvamSTTOptions:
    """Options for the Sarvam.ai STT service.

    Args:
        language: BCP-47 language code, e.g., "hi-IN", "en-IN"
        model: The Sarvam STT model to use
        api_key: Sarvam.ai API key
        base_url: API endpoint URL
    """

    language: str  # BCP-47 language code, e.g., "hi-IN", "en-IN"
    model: SarvamSTTModels | str = "saarika:v2.5"
    api_key: str | None = None
    base_url: str = SARVAM_STT_BASE_URL


class STT(stt.STT):
    """Sarvam.ai Speech-to-Text implementation.

    This class provides speech-to-text functionality using the Sarvam.ai API.
    Sarvam.ai specializes in high-quality STT for Indian languages.

    Args:
        language: BCP-47 language code, e.g., "hi-IN", "en-IN"
        model: The Sarvam STT model to use
        api_key: Sarvam.ai API key (falls back to SARVAM_API_KEY env var)
        base_url: API endpoint URL
        http_session: Optional aiohttp session to use
    """

    def __init__(
        self,
        *,
        language: str,
        model: SarvamSTTModels | str = "saarika:v2.5",
        api_key: str | None = None,
        base_url: str = SARVAM_STT_BASE_URL,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        super().__init__(capabilities=stt.STTCapabilities(streaming=False, interim_results=False))

        self._api_key = api_key or os.environ.get("SARVAM_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Sarvam API key is required. "
                "Provide it directly or set SARVAM_API_KEY environment variable."
            )

        self._opts = SarvamSTTOptions(
            language=language,
            model=model,
            api_key=self._api_key,
            base_url=base_url,
        )
        self._session = http_session
        self._logger = logger.getChild(self.__class__.__name__)

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        model: NotGivenOr[SarvamSTTModels | str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        """Recognize speech using Sarvam.ai API.

        Args:
            buffer: Audio buffer containing speech data
            language: BCP-47 language code (overrides the one set in constructor)
            model: Sarvam model to use (overrides the one set in constructor)
            conn_options: Connection options for API requests

        Returns:
            A SpeechEvent containing the transcription result

        Raises:
            APIConnectionError: On network connection errors
            APIStatusError: On API errors (non-200 status)
            APITimeoutError: On API timeout
        """
        opts_language = self._opts.language if isinstance(language, type(NOT_GIVEN)) else language
        opts_model = self._opts.model if isinstance(model, type(NOT_GIVEN)) else model

        wav_bytes = rtc.combine_audio_frames(buffer).to_wav_bytes()

        form_data = aiohttp.FormData()
        form_data.add_field("file", wav_bytes, filename="audio.wav", content_type="audio/wav")

        # Add model and language_code to the form data if specified
        # Sarvam API docs state language_code is optional for saarika:v2x but mandatory for v1
        # Model is also optional, defaults to saarika:v2.5
        if opts_language:
            form_data.add_field("language_code", opts_language)
        if opts_model:
            form_data.add_field("model", str(opts_model))

        headers = {"api-subscription-key": self._opts.api_key}

        try:
            async with self._ensure_session().post(
                url=self._opts.base_url,
                data=form_data,
                headers=headers,
                timeout=aiohttp.ClientTimeout(
                    total=conn_options.timeout,
                    sock_connect=conn_options.timeout,
                ),
            ) as res:
                if res.status != 200:
                    error_text = await res.text()
                    self._logger.error(f"Sarvam API error: {res.status} - {error_text}")
                    raise APIStatusError(
                        message=f"Sarvam API Error: {error_text}",
                        status_code=res.status,
                    )

                response_json = await res.json()
                self._logger.debug(f"Sarvam API response: {response_json}")

                transcript_text = response_json.get("transcript", "")
                request_id = response_json.get("request_id", "")
                detected_language = response_json.get("language_code")
                if not isinstance(detected_language, str):
                    detected_language = opts_language or ""

                start_time = 0.0
                end_time = 0.0

                # Try to get timestamps if available
                timestamps_data = response_json.get("timestamps")
                if timestamps_data and isinstance(timestamps_data, dict):
                    words_ts_start = timestamps_data.get("start_time_seconds")
                    words_ts_end = timestamps_data.get("end_time_seconds")
                    if isinstance(words_ts_start, list) and len(words_ts_start) > 0:
                        start_time = words_ts_start[0]
                    if isinstance(words_ts_end, list) and len(words_ts_end) > 0:
                        end_time = words_ts_end[-1]

                # If start/end times are still 0, use buffer duration as an estimate for end_time
                if start_time == 0.0 and end_time == 0.0:
                    # Calculate duration from buffer - AudioBuffer can be list[AudioFrame]
                    # or AudioFrame
                    try:
                        if isinstance(buffer, list):
                            # Calculate total duration from all frames
                            total_samples = sum(frame.samples_per_channel for frame in buffer)
                            if buffer and total_samples > 0:
                                sample_rate = buffer[0].sample_rate
                                end_time = total_samples / sample_rate
                        elif hasattr(buffer, "duration"):
                            end_time = buffer.duration / 1000.0  # buffer.duration is in ms
                        elif hasattr(buffer, "samples_per_channel") and hasattr(
                            buffer, "sample_rate"
                        ):
                            # Single AudioFrame
                            end_time = buffer.samples_per_channel / buffer.sample_rate
                    except Exception as duration_error:
                        self._logger.warning(
                            f"Could not calculate audio duration: {duration_error}"
                        )
                        end_time = 0.0

                alternatives = [
                    stt.SpeechData(
                        language=detected_language,
                        text=transcript_text,
                        start_time=start_time,
                        end_time=end_time,
                        confidence=1.0,  # Sarvam doesn't provide confidence score in this response
                    )
                ]

                return stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    request_id=request_id,
                    alternatives=alternatives,
                )

        except asyncio.TimeoutError as e:
            self._logger.error(f"Sarvam API timeout: {e}")
            raise APITimeoutError("Sarvam API request timed out") from e
        except aiohttp.ClientError as e:
            self._logger.error(f"Sarvam API client error: {e}")
            raise APIConnectionError(f"Sarvam API connection error: {e}") from e
        except Exception as e:
            self._logger.error(f"Error during Sarvam STT processing: {e}")
            raise APIConnectionError(f"Unexpected error in Sarvam STT: {e}") from e
