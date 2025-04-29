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
import base64
import json
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

from .models import TTSLangCodes, TTSModels

API_BASE_URL = "api.neuphonic.com"
AUTHORIZATION_HEADER = "X-API-KEY"
NUM_CHANNELS = 1


@dataclass
class _TTSOptions:
    base_url: str
    model: TTSModels | str
    lang_code: TTSLangCodes | str
    sample_rate: int
    speed: float
    voice_id: str | None


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        model: TTSModels | str = "neu_hq",
        api_key: str | None = None,
        voice_id: str | None = None,
        lang_code: TTSLangCodes | str = "en",
        speed: float = 1.0,
        sample_rate: int = 22050,
        http_session: aiohttp.ClientSession | None = None,
        base_url: str = API_BASE_URL,
    ) -> None:
        """
        Create a new instance of the Neuphonic TTS.

        See https://docs.neuphonic.com for more documentation on all of these options, or go to https://app.neuphonic.com/ to test out different options.

        Args:
            model (TTSModels | str, optional): The Neuphonic model to use. See Defaults to "neu_hq".
            voice_id (str, optional): The voice ID for the desired voice. Defaults to None.
            lang_code (TTSLanguages | str, optional): The language code for synthesis. Defaults to "en".
            encoding (TTSEncodings | str, optional): The audio encoding format. Defaults to "pcm_mulaw".
            speed (float, optional): The audio playback speed. Defaults to 1.0.
            sample_rate (int, optional): The audio sample rate in Hz. Defaults to 22050.
            api_key (str | None, optional): The Neuphonic API key. If not provided, it will be read from the NEUPHONIC_API_TOKEN environment variable.
            http_session (aiohttp.ClientSession | None, optional): An existing aiohttp ClientSession to use. If not provided, a new session will be created.
            base_url (str, optional): The base URL for the Neuphonic API. Defaults to "api.neuphonic.com".
        """  # noqa: E501
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )

        self._api_key = api_key or os.environ.get("NEUPHONIC_API_TOKEN")
        if not self._api_key:
            raise ValueError("API key must be provided or set in NEUPHONIC_API_TOKEN")

        self._opts = _TTSOptions(
            model=model,
            voice_id=voice_id,
            lang_code=lang_code,
            speed=speed,
            sample_rate=sample_rate,
            base_url=base_url,
        )

        self._session = http_session

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

        return self._session

    def update_options(
        self,
        *,
        model: NotGivenOr[TTSModels] = NOT_GIVEN,
        voice_id: NotGivenOr[str] = NOT_GIVEN,
        lang_code: NotGivenOr[TTSLangCodes] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
    ) -> None:
        """
        Update the Text-to-Speech (TTS) configuration options.

        This method allows updating the TTS settings, including model type, voice_id, lang_code,
        encoding, speed and sample_rate. If any parameter is not provided, the existing value will be
        retained.

        Args:
            model (TTSModels | str, optional): The Neuphonic model to use.
            voice_id (str, optional): The voice ID for the desired voice.
            lang_code (TTSLanguages | str, optional): The language code for synthesis..
            encoding (TTSEncodings | str, optional): The audio encoding format.
            speed (float, optional): The audio playback speed.
            sample_rate (int, optional): The audio sample rate in Hz.
        """  # noqa: E501
        if is_given(model):
            self._opts.model = model
        if is_given(voice_id):
            self._opts.voice_id = voice_id
        if is_given(lang_code):
            self._opts.lang_code = lang_code
        if is_given(speed):
            self._opts.speed = speed
        if is_given(sample_rate):
            self._opts.sample_rate = sample_rate

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)


class ChunkedStream(tts.ChunkedStream):
    """Synthesize chunked text using the SSE endpoint"""

    def __init__(
        self,
        *,
        tts: TTS,
        input_text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.SynthesizedAudioEmitter):
        try:
            async with self._tts._ensure_session().post(
                f"https://{self._opts.base_url}/sse/speak/{self._opts.lang_code}",
                headers={
                    AUTHORIZATION_HEADER: self._tts._api_key,
                },
                json={
                    "text": self._input_text,
                    "voice_id": self._opts.voice_id,
                    "model": self._opts.model,
                    "lang_code": self._opts.lang_code,
                    "encoding": "pcm_linear",
                    "sampling_rate": self._opts.sample_rate,
                    "speed": self._opts.speed,
                },
                timeout=aiohttp.ClientTimeout(
                    total=30,
                    sock_connect=self._conn_options.timeout,
                ),
                # large read_bufsize to avoid `ValueError: Chunk too big`
                read_bufsize=10 * 1024 * 1024,
            ) as resp:
                resp.raise_for_status()

                output_emitter.start(
                    request_id=utils.shortuuid(),
                    sample_rate=self._opts.sample_rate,
                    num_channels=1,
                    format="audio/pcm",
                )

                async for line in resp.content:
                    message = line.decode("utf-8")
                    if not message:
                        continue

                    parsed_message = _parse_sse_message(message)

                    if (
                        parsed_message is not None
                        and parsed_message.get("data", {}).get("audio") is not None
                    ):
                        audio_bytes = base64.b64decode(parsed_message["data"]["audio"])
                        output_emitter.push(audio_bytes)

                output_emitter.flush()
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=None, body=None
            ) from None
        except Exception as e:
            raise APIConnectionError() from e


def _parse_sse_message(message: str) -> dict | None:
    """
    Parse each response from the SSE endpoint.

    The message will either be a string reading:
    - `event: error`
    - `event: message`
    - `data: { "status_code": 200, "data": {"audio": ... } }`
    """
    message = message.strip()

    if not message or "data" not in message:
        return None

    _, value = message.split(": ", 1)
    message_dict = json.loads(value)

    if message_dict.get("errors") is not None:
        raise Exception(
            f"received error status {message_dict['status_code']}: {message_dict['errors']}"
        )

    return message_dict
