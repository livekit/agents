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

import asyncio
import base64
import json
import os
from dataclasses import dataclass, replace
from typing import Any, Literal, Union
from urllib.parse import urljoin

import aiohttp

from livekit.agents import tts, utils
from livekit.agents._exceptions import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
)
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.voice.io import TimedString

DEFAULT_BIT_RATE = 64000
DEFAULT_ENCODING = "OGG_OPUS"
DEFAULT_MODEL = "inworld-tts-1"
DEFAULT_SAMPLE_RATE = 48000
DEFAULT_URL = "https://api.inworld.ai/"
DEFAULT_VOICE = "Ashley"
DEFAULT_TEMPERATURE = 1.1
DEFAULT_SPEAKING_RATE = 1.0
NUM_CHANNELS = 1

Encoding = Union[Literal["LINEAR16", "MP3", "OGG_OPUS", "ALAW", "MULAW", "FLAC"], str]
TimestampType = Literal["TIMESTAMP_TYPE_UNSPECIFIED", "WORD", "CHARACTER"]
TextNormalization = Literal["APPLY_TEXT_NORMALIZATION_UNSPECIFIED", "ON", "OFF"]


@dataclass
class _TTSOptions:
    model: str
    encoding: Encoding
    voice: str
    sample_rate: int
    bit_rate: NotGivenOr[int] = NOT_GIVEN
    speaking_rate: NotGivenOr[float] = NOT_GIVEN
    temperature: NotGivenOr[float] = NOT_GIVEN
    timestamp_type: NotGivenOr[TimestampType] = NOT_GIVEN
    text_normalization: NotGivenOr[TextNormalization] = NOT_GIVEN

    @property
    def mime_type(self) -> str:
        if self.encoding == "MP3":
            return "audio/mpeg"
        elif self.encoding == "OGG_OPUS":
            return "audio/ogg"
        elif self.encoding == "FLAC":
            return "audio/flac"
        elif self.encoding in ("ALAW", "MULAW"):
            return "audio/basic"
        else:
            return "audio/wav"


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        voice: NotGivenOr[str] = NOT_GIVEN,
        model: NotGivenOr[str] = NOT_GIVEN,
        encoding: NotGivenOr[Encoding] = NOT_GIVEN,
        bit_rate: NotGivenOr[int] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        speaking_rate: NotGivenOr[float] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        timestamp_type: NotGivenOr[TimestampType] = NOT_GIVEN,
        text_normalization: NotGivenOr[TextNormalization] = NOT_GIVEN,
        base_url: str = DEFAULT_URL,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """
        Create a new instance of Inworld TTS.

        Args:
            api_key (str, optional): The Inworld API key.
                If not provided, it will be read from the INWORLD_API_KEY environment variable.
            voice (str, optional): The voice to use. Defaults to "Ashley".
            model (str, optional): The Inworld model to use. Defaults to "inworld-tts-1".
            encoding (str, optional): The encoding to use. Defaults to "OGG_OPUS".
            bit_rate (int, optional): Bits per second of the audio. Defaults to 64000.
            sample_rate (int, optional): The audio sample rate in Hz. Defaults to 48000.
            speaking_rate (float, optional): The speed of the voice, in the range [0.5, 1.5].
                Defaults to 1.0.
            temperature (float, optional): Determines the degree of randomness when sampling audio
                tokens to generate the response. Range [0, 2]. Defaults to 1.1.
            timestamp_type (str, optional): Controls timestamp metadata returned with the audio.
                Use "WORD" for word-level timestamps or "CHARACTER" for character-level.
                Useful for karaoke-style captions, word highlighting, and lipsync.
            text_normalization (str, optional): Controls text normalization. When "ON", numbers,
                dates, and abbreviations are expanded (e.g., "Dr." -> "Doctor"). When "OFF",
                text is read exactly as written. Defaults to automatic.
            base_url (str, optional): The base URL for the Inworld TTS API.
                Defaults to "https://api.inworld.ai/".
            http_session (aiohttp.ClientSession, optional): The HTTP session to use.
        """
        if not utils.is_given(sample_rate):
            sample_rate = DEFAULT_SAMPLE_RATE
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )

        api_key = api_key or os.getenv("INWORLD_API_KEY", "")
        if not api_key:
            raise ValueError("Inworld API key required. Set INWORLD_API_KEY or provide api_key.")

        self._authorization = f"Basic {api_key}"
        self._base_url = base_url
        self._session = http_session

        self._opts = _TTSOptions(
            voice=voice if utils.is_given(voice) else DEFAULT_VOICE,
            model=model if utils.is_given(model) else DEFAULT_MODEL,
            encoding=encoding if utils.is_given(encoding) else DEFAULT_ENCODING,
            bit_rate=bit_rate if utils.is_given(bit_rate) else DEFAULT_BIT_RATE,
            sample_rate=sample_rate if utils.is_given(sample_rate) else DEFAULT_SAMPLE_RATE,
            speaking_rate=speaking_rate if utils.is_given(speaking_rate) else DEFAULT_SPEAKING_RATE,
            temperature=temperature if utils.is_given(temperature) else DEFAULT_TEMPERATURE,
            timestamp_type=timestamp_type,
            text_normalization=text_normalization,
        )

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "Inworld"

    def update_options(
        self,
        *,
        voice: NotGivenOr[str] = NOT_GIVEN,
        model: NotGivenOr[str] = NOT_GIVEN,
        encoding: NotGivenOr[Encoding] = NOT_GIVEN,
        bit_rate: NotGivenOr[int] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        speaking_rate: NotGivenOr[float] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        timestamp_type: NotGivenOr[TimestampType] = NOT_GIVEN,
        text_normalization: NotGivenOr[TextNormalization] = NOT_GIVEN,
    ) -> None:
        """
        Update the TTS configuration options.

        Args:
            voice (str, optional): The voice to use.
            model (str, optional): The Inworld model to use.
            encoding (str, optional): The encoding to use.
            bit_rate (int, optional): Bits per second of the audio.
            sample_rate (int, optional): The audio sample rate in Hz.
            speaking_rate (float, optional): The speed of the voice.
            temperature (float, optional): Determines the degree of randomness when sampling audio
                tokens to generate the response.
            timestamp_type (str, optional): Controls timestamp metadata ("WORD" or "CHARACTER").
            text_normalization (str, optional): Controls text normalization ("ON" or "OFF").
        """
        if utils.is_given(voice):
            self._opts.voice = voice
        if utils.is_given(model):
            self._opts.model = model
        if utils.is_given(encoding):
            self._opts.encoding = encoding
        if utils.is_given(bit_rate):
            self._opts.bit_rate = bit_rate
        if utils.is_given(sample_rate):
            self._opts.sample_rate = sample_rate
        if utils.is_given(speaking_rate):
            self._opts.speaking_rate = speaking_rate
        if utils.is_given(temperature):
            self._opts.temperature = temperature
        if utils.is_given(timestamp_type):
            self._opts.timestamp_type = timestamp_type
        if utils.is_given(text_normalization):
            self._opts.text_normalization = text_normalization

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

        return self._session

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)


class ChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        try:
            audio_config: dict[str, Any] = {
                "audioEncoding": self._opts.encoding,
            }
            if utils.is_given(self._opts.bit_rate):
                audio_config["bitrate"] = self._opts.bit_rate
            if utils.is_given(self._opts.sample_rate):
                audio_config["sampleRateHertz"] = self._opts.sample_rate
            if utils.is_given(self._opts.temperature):
                audio_config["temperature"] = self._opts.temperature
            if utils.is_given(self._opts.speaking_rate):
                audio_config["speakingRate"] = self._opts.speaking_rate

            body_params: dict[str, Any] = {
                "text": self._input_text,
                "voiceId": self._opts.voice,
                "modelId": self._opts.model,
                "audioConfig": audio_config,
            }
            if utils.is_given(self._opts.temperature):
                body_params["temperature"] = self._opts.temperature
            if utils.is_given(self._opts.timestamp_type):
                body_params["timestampType"] = self._opts.timestamp_type
            if utils.is_given(self._opts.text_normalization):
                body_params["applyTextNormalization"] = self._opts.text_normalization

            async with self._tts._ensure_session().post(
                urljoin(self._tts._base_url, "/tts/v1/voice:stream"),
                headers={
                    "Authorization": self._tts._authorization,
                },
                json=body_params,
                timeout=aiohttp.ClientTimeout(sock_connect=self._conn_options.timeout),
                # large read_bufsize to avoid `ValueError: Chunk too big`
                read_bufsize=10 * 1024 * 1024,
            ) as resp:
                resp.raise_for_status()

                request_id = utils.shortuuid()
                output_emitter.initialize(
                    request_id=request_id,
                    sample_rate=self._opts.sample_rate,
                    num_channels=NUM_CHANNELS,
                    mime_type=self._opts.mime_type,
                )

                async for line in resp.content:
                    if not line:
                        break
                    data = json.loads(line)
                    if result := data.get("result"):
                        # Handle timestamp info if present
                        if timestamp_info := result.get("timestampInfo"):
                            timed_strings = _parse_timestamp_info(timestamp_info)
                            if timed_strings:
                                output_emitter.push_timed_transcript(timed_strings)

                        if audio_content := result.get("audioContent"):
                            output_emitter.push(base64.b64decode(audio_content))
                            output_emitter.flush()
                    elif error := data.get("error"):
                        raise APIStatusError(
                            message=error.get("message"),
                            status_code=error.get("code"),
                            request_id=request_id,
                            body=None,
                        )
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=None, body=None
            ) from None
        except Exception as e:
            raise APIConnectionError() from e


def _parse_timestamp_info(timestamp_info: dict[str, Any]) -> list[TimedString]:
    """Parse timestamp info from API response into TimedString objects."""
    timed_strings: list[TimedString] = []

    # Handle word-level alignment
    if word_align := timestamp_info.get("wordAlignment"):
        words = word_align.get("words", [])
        starts = word_align.get("wordStartTimeSeconds", [])
        ends = word_align.get("wordEndTimeSeconds", [])

        for word, start, end in zip(words, starts, ends):
            timed_strings.append(TimedString(word, start_time=start, end_time=end))

    # Handle character-level alignment
    if char_align := timestamp_info.get("characterAlignment"):
        chars = char_align.get("characters", [])
        starts = char_align.get("characterStartTimeSeconds", [])
        ends = char_align.get("characterEndTimeSeconds", [])

        for char, start, end in zip(chars, starts, ends):
            timed_strings.append(TimedString(char, start_time=start, end_time=end))

    return timed_strings
