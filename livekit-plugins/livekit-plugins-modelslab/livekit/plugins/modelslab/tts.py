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
from typing import Final

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tts,
    utils,
)
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import is_given

from .models import ModelsLabAudioFormat

MODELSLAB_BASE_URL: Final[str] = "https://modelslab.com/api/v6"
MODELSLAB_TTS_ENDPOINT: Final[str] = f"{MODELSLAB_BASE_URL}/voice/text_to_audio"
MODELSLAB_FETCH_ENDPOINT: Final[str] = f"{MODELSLAB_BASE_URL}/voice/fetch"

NUM_CHANNELS: Final[int] = 1
DEFAULT_SAMPLE_RATE: Final[int] = 24000
POLL_INTERVAL_SECONDS: Final[float] = 2.0
POLL_TIMEOUT_SECONDS: Final[float] = 300.0

MIME_TYPE: dict[str, str] = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
}


@dataclass
class _TTSOptions:
    voice_id: str
    output_format: ModelsLabAudioFormat
    sample_rate: int
    api_key: str


class TTS(tts.TTS):
    """
    Text-to-Speech (TTS) plugin for ModelsLab.

    Uses the ModelsLab Community TTS API to synthesize speech from text.
    Supports async generation with automatic polling for longer texts.

    See https://docs.modelslab.com for more information.

    .. note::
        ModelsLab's API key is passed in the JSON body rather than via an HTTP header.
        The key will appear in request payloads; use appropriate log-scrubbing in
        production environments.
    """

    def __init__(
        self,
        *,
        voice_id: str = "default",
        output_format: ModelsLabAudioFormat = "mp3",
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        api_key: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """
        Create a new instance of ModelsLab TTS.

        See: https://docs.modelslab.com/text-to-speech/community-tts/generate

        Args:
            voice_id: The voice ID to use for synthesis. Defaults to ``"default"``.
                Browse available voices at https://modelslab.com/voice-cloning.
            output_format: Audio output format. Options: ``"mp3"``, ``"wav"``.
                Default is ``"mp3"``.
            sample_rate: Output sample rate in Hz. Default is 24000.
            api_key: ModelsLab API key. Defaults to the ``MODELSLAB_API_KEY``
                environment variable.
            http_session: Optional aiohttp ClientSession. A new session is created
                if not provided.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )

        api_key = api_key or os.environ.get("MODELSLAB_API_KEY")
        if not api_key:
            raise ValueError(
                "ModelsLab API key is required, either as argument or set "
                "MODELSLAB_API_KEY environment variable"
            )

        self._opts = _TTSOptions(
            voice_id=voice_id,
            output_format=output_format,
            sample_rate=sample_rate,
            api_key=api_key,
        )
        self._session = http_session

    @property
    def provider(self) -> str:
        return "ModelsLab"

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> ChunkedStream:
        return ChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
        )

    def update_options(
        self,
        *,
        voice_id: NotGivenOr[str] = NOT_GIVEN,
        output_format: NotGivenOr[ModelsLabAudioFormat] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
    ) -> None:
        """
        Update the TTS options.

        Args:
            voice_id: The voice ID to update.
            output_format: Audio output format (``"mp3"`` or ``"wav"``).
            sample_rate: Output sample rate in Hz.
        """
        if is_given(voice_id):
            self._opts.voice_id = voice_id
        if is_given(output_format):
            self._opts.output_format = output_format  # type: ignore[assignment]
        if is_given(sample_rate):
            self._opts.sample_rate = sample_rate

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session


class ChunkedStream(tts.ChunkedStream):
    """Synthesize text to speech using ModelsLab, with polling support."""

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
        payload = {
            "key": self._opts.api_key,
            "prompt": self._input_text,
            "voice_id": self._opts.voice_id,
            "output_format": self._opts.output_format,
        }

        try:
            # Step 1: Submit generation request
            async with self._tts._ensure_session().post(
                MODELSLAB_TTS_ENDPOINT,
                json=payload,
                timeout=aiohttp.ClientTimeout(
                    sock_connect=self._conn_options.timeout,
                    total=30,
                ),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()

            # Step 2: Poll if processing
            if data.get("status") == "processing":
                generation_id = data.get("id")
                if not generation_id:
                    raise APIConnectionError("ModelsLab returned processing status without an id")
                data = await self._poll(generation_id)

            # Step 3: Validate success
            if data.get("status") == "error":
                message = data.get("message") or data.get("messege") or "Unknown error"
                raise APIStatusError(
                    f"ModelsLab TTS error: {message}",
                    status_code=400,
                    body=str(data),
                )

            if data.get("status") != "success":
                raise APIConnectionError(
                    f"Unexpected ModelsLab status: {data.get('status')}"
                )

            output_urls = data.get("output") or []
            if not output_urls:
                raise APIConnectionError("ModelsLab returned no audio output")

            # Step 4: Download audio and emit
            audio_url = output_urls[0]
            async with self._tts._ensure_session().get(
                audio_url,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as audio_resp:
                audio_resp.raise_for_status()
                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=self._opts.sample_rate,
                    num_channels=NUM_CHANNELS,
                    mime_type=MIME_TYPE[self._opts.output_format],
                )
                async for chunk in audio_resp.content.iter_chunked(4096):
                    output_emitter.push(chunk)

            output_emitter.flush()

        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                e.message,
                status_code=e.status,
                body="",
            ) from e
        except (APIConnectionError, APITimeoutError, APIStatusError):
            raise
        except Exception as e:
            raise APIConnectionError() from e

    async def _poll(self, generation_id: str | int) -> dict:
        """Poll the fetch endpoint until generation completes."""
        fetch_url = f"{MODELSLAB_FETCH_ENDPOINT}/{generation_id}"
        fetch_payload = {"key": self._opts.api_key}
        elapsed = 0.0

        while elapsed < POLL_TIMEOUT_SECONDS:
            await asyncio.sleep(POLL_INTERVAL_SECONDS)
            elapsed += POLL_INTERVAL_SECONDS

            async with self._tts._ensure_session().post(
                fetch_url,
                json=fetch_payload,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()

            if data.get("status") != "processing":
                return data

        raise APITimeoutError(
            f"ModelsLab TTS timed out after {POLL_TIMEOUT_SECONDS}s (id={generation_id})"
        )
