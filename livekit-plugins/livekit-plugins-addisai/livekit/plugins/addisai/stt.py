# Copyright 2026 LiveKit, Inc.
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
import dataclasses
import json
import os
from dataclasses import dataclass
from typing import Any

import aiohttp

from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APITimeoutError,
    LanguageCode,
    stt,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, is_given

from ._utils import (
    API_BASE_URL,
    parse_json_response,
    response_request_id,
    unwrap_data,
    validate_language,
)
from .models import Language
from .version import __version__


@dataclass
class _STTOptions:
    language: Language
    base_url: str


class STT(stt.STT):
    """Batch speech recognition using AddisAI's addis-whisper model."""

    def __init__(
        self,
        *,
        language: Language | str = "am",
        api_key: str | None = None,
        base_url: str = API_BASE_URL,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """Create an AddisAI speech-to-text client.

        Args:
            language: Transcription language: ``"am"`` for Amharic or ``"om"``
                for Afaan Oromo.
            api_key: AddisAI API key. Falls back to ``ADDIS_API_KEY``.
            base_url: AddisAI API base URL.
            http_session: Optional shared :class:`aiohttp.ClientSession`.

        Raises:
            ValueError: If the API key is missing or the language is unsupported.
        """
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False,
                interim_results=False,
                aligned_transcript=False,
            )
        )

        resolved_api_key = api_key or os.environ.get("ADDIS_API_KEY")
        if not resolved_api_key:
            raise ValueError("AddisAI API key is required, either as api_key or ADDIS_API_KEY")

        self._api_key = resolved_api_key
        self._opts = _STTOptions(
            language=validate_language(language),
            base_url=base_url.rstrip("/"),
        )
        self._session = http_session

    @property
    def model(self) -> str:
        return "addis-whisper"

    @property
    def provider(self) -> str:
        return "AddisAI"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = utils.http_context.http_session()
        return self._session

    def update_options(self, *, language: NotGivenOr[Language | str] = NOT_GIVEN) -> None:
        """Update the transcription language for subsequent requests."""
        if is_given(language):
            self._opts.language = validate_language(language)

    def _sanitize_options(self, *, language: NotGivenOr[str] = NOT_GIVEN) -> _STTOptions:
        options = dataclasses.replace(self._opts)
        if is_given(language):
            options.language = validate_language(language)
        return options

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        options = self._sanitize_options(language=language)
        form = aiohttp.FormData()
        form.add_field(
            "audio",
            rtc.combine_audio_frames(buffer).to_wav_bytes(),
            filename="audio.wav",
            content_type="audio/wav",
        )
        form.add_field(
            "request_data",
            json.dumps({"language_code": options.language}),
        )

        try:
            async with self._ensure_session().post(
                f"{options.base_url}/api/v2/stt",
                data=form,
                headers={
                    "x-api-key": self._api_key,
                    "Accept": "application/json",
                    "X-Addis-Client": f"livekit-plugins-addisai/{__version__}",
                },
                timeout=aiohttp.ClientTimeout(
                    total=30,
                    sock_connect=conn_options.timeout,
                ),
            ) as response:
                payload = await parse_json_response(response)
                data = unwrap_data(payload)
                usage = data.get("usage_metadata")
                request_id = _request_id_from_usage(usage) or response_request_id(response) or ""
                confidence = _confidence_from_payload(payload, data)

                return stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    request_id=request_id,
                    alternatives=[
                        stt.SpeechData(
                            text=str(data.get("transcription") or ""),
                            language=LanguageCode(options.language),
                            confidence=confidence,
                            metadata={"usage": usage} if isinstance(usage, dict) else None,
                        )
                    ],
                )
        except asyncio.TimeoutError as exc:
            raise APITimeoutError() from exc
        except APIError:
            raise
        except aiohttp.ClientError as exc:
            raise APIConnectionError("failed to connect to AddisAI STT") from exc
        except Exception as exc:
            raise APIConnectionError("failed to recognize speech with AddisAI") from exc


def _request_id_from_usage(usage: object) -> str | None:
    if not isinstance(usage, dict):
        return None
    value = usage.get("requestId") or usage.get("request_id")
    return str(value) if value else None


def _confidence_from_payload(
    payload: dict[str, Any],
    data: dict[str, Any],
) -> float:
    value = payload.get("confidence", data.get("confidence", 0.0))
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
