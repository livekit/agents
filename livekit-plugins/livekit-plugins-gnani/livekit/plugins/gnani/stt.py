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

from __future__ import annotations

import io
import os
from dataclasses import dataclass

import aiohttp

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    NotGivenOr,
    stt,
    utils,
)
from livekit.agents.stt import SpeechEventType, STTCapabilities
from livekit.agents.utils import AudioBuffer, is_given

from .log import logger
from .models import GnaniLanguages


@dataclass
class _STTOptions:
    language: str


class STT(stt.STT):
    def __init__(
        self,
        *,
        language: GnaniLanguages | str = "en-IN",
        api_key: str | None = None,
        organization_id: str | None = None,
        user_id: str | None = None,
        base_url: str = "https://api.vachana.ai/stt/v3",
        http_session: aiohttp.ClientSession | None = None,
    ):
        """Create a new instance of Gnani STT.
        Args:
            language: Target transcription language code (e.g., "en-IN", "hi-IN").
                     Defaults to "en-IN".
            api_key: Your Gnani API key. If not provided, will look for GNANI_API_KEY
                    environment variable.
            organization_id: Organization ID for the API. If not provided, will look for
                           GNANI_ORG_ID environment variable.
            user_id: User ID for the API. If not provided, will look for GNANI_USER_ID
                    environment variable.
            base_url: The base URL for Gnani API. Defaults to "https://api.vachana.ai/stt/v3".
            http_session: Optional aiohttp ClientSession to use for requests.
        Raises:
            ValueError: If required API credentials are not provided or found in environment variables.
        """
        super().__init__(capabilities=STTCapabilities(streaming=False, interim_results=False))

        # Get API credentials from parameters or environment variables
        api_key_value = api_key or os.environ.get("GNANI_API_KEY")
        organization_id_value = organization_id or os.environ.get("GNANI_ORG_ID")
        user_id_value = user_id or os.environ.get("GNANI_USER_ID")

        if not api_key_value:
            raise ValueError(
                "Gnani API key is required. Set it via api_key parameter or GNANI_API_KEY environment variable."
            )
        if not organization_id_value:
            raise ValueError(
                "Gnani Organization ID is required. Set it via organization_id parameter or GNANI_ORG_ID environment variable."
            )
        if not user_id_value:
            raise ValueError(
                "Gnani User ID is required. Set it via user_id parameter or GNANI_USER_ID environment variable."
            )

        # After validation, we know these are strings
        self._api_key: str = api_key_value
        self._organization_id: str = organization_id_value
        self._user_id: str = user_id_value

        self._base_url = base_url
        self._opts = _STTOptions(language=language)
        self._session = http_session

    @property
    def model(self) -> str:
        return "gnani-stt-v3"

    @property
    def provider(self) -> str:
        return "Gnani"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def update_options(self, *, language: GnaniLanguages | str | None = None) -> None:
        """Update STT options.
        Args:
            language: Target transcription language code.
        """
        if language is not None:
            self._opts.language = language

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        """Recognize speech from audio buffer using Gnani API.
        Args:
            buffer: Audio buffer containing the audio frames to transcribe.
            language: Optional language override for this request.
            conn_options: API connection options including timeout.
        Returns:
            SpeechEvent containing the transcription result.
        Raises:
            APIConnectionError: If there's a connection error.
            APIStatusError: If the API returns an error status.
            APITimeoutError: If the request times out.
        """
        try:
            # Use provided language or fall back to configured language
            target_language = language if is_given(language) else self._opts.language

            # Combine audio frames into a single frame
            combined_frame = rtc.combine_audio_frames(buffer)

            # Convert to WAV format for upload
            wav_bytes = combined_frame.to_wav_bytes()

            # Generate a unique request ID
            import uuid

            request_id = f"req_{uuid.uuid4().hex[:12]}"

            # Prepare the multipart form data
            form_data = aiohttp.FormData()
            form_data.add_field("language_code", target_language)
            form_data.add_field(
                "audio_file",
                io.BytesIO(wav_bytes),
                filename="audio.wav",
                content_type="audio/wav",
            )

            # Prepare headers
            headers = {
                "X-API-Key-ID": self._api_key,
                "X-Organization-ID": self._organization_id,
                "X-API-User-ID": self._user_id,
                "X-API-Request-ID": request_id,
            }

            # Make the API request
            session = self._ensure_session()
            async with session.post(
                self._base_url,
                data=form_data,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=conn_options.timeout),
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Gnani API error ({response.status}): {error_text}")
                    raise APIStatusError(
                        message=f"Gnani API returned status {response.status}",
                        status_code=response.status,
                        request_id=request_id,
                        body=error_text,
                    )

                result = await response.json()

                # Check if the response indicates success
                if not result.get("success", False):
                    error_msg = result.get("error", "Unknown error")
                    logger.error(f"Gnani API returned failure: {error_msg}")
                    raise APIConnectionError(f"Gnani API error: {error_msg}")

                # Extract the transcript
                transcript = result.get("transcript", "").strip()

                # Create and return the speech event
                return self._create_speech_event(
                    text=transcript, language=target_language, request_id=request_id
                )

        except aiohttp.ClientResponseError as e:
            logger.error(f"Gnani API response error: {e.status} {e.message}")
            raise APIStatusError(
                message=e.message,
                status_code=e.status,
                request_id=None,
                body=None,
            ) from e
        except aiohttp.ClientError as e:
            logger.error(f"Gnani connection error: {e}")
            raise APIConnectionError(f"Gnani connection error: {e}") from e
        except Exception as e:
            logger.exception(f"Unexpected error during Gnani recognition: {e}")
            raise APIConnectionError(f"An unexpected error occurred: {e}") from e

    def _create_speech_event(self, text: str, language: str, request_id: str) -> stt.SpeechEvent:
        """Create a SpeechEvent from transcription result.
        Args:
            text: The transcribed text.
            language: The language of the transcription.
            request_id: The request ID for this transcription.
        Returns:
            SpeechEvent containing the transcription.
        """
        return stt.SpeechEvent(
            type=SpeechEventType.FINAL_TRANSCRIPT,
            request_id=request_id,
            alternatives=[
                stt.SpeechData(
                    text=text,
                    language=language,
                    confidence=1.0,  # Gnani API doesn't return confidence scores
                )
            ],
        )
