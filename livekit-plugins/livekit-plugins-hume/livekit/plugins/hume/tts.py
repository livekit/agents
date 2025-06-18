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
from enum import Enum
from typing import Any, TypedDict

import aiohttp

from livekit.agents import APIConnectionError, APIConnectOptions, APITimeoutError, tts, utils
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

API_AUTH_HEADER = "X-Hume-Api-Key"
STREAM_PATH = "/v0/tts/stream/json"
DEFAULT_BASE_URL = "https://api.hume.ai"
SUPPORTED_SAMPLE_RATE = 48000


class VoiceId(TypedDict, total=False):
    id: str
    provider: VoiceProvider | None


class VoiceName(TypedDict, total=False):
    name: str
    provider: VoiceProvider | None


class UtteranceOptions(TypedDict, total=False):
    description: str | None
    speed: float | None
    voice: VoiceId | VoiceName | None
    trailing_silence: float | None


class Utterance(UtteranceOptions, total=False):
    text: str


class ContextGenerationId(TypedDict, total=False):
    generation_id: str


class ContextUtterances(TypedDict, total=False):
    utterances: list[Utterance]


class VoiceProvider(str, Enum):
    hume = "HUME_AI"
    custom = "CUSTOM_VOICE"


class AudioFormat(str, Enum):
    mp3 = "mp3"
    wav = "wav"
    pcm = "pcm"


@dataclass
class _TTSOptions:
    api_key: str
    base_url: str
    utterance_options: Utterance
    context: ContextGenerationId | ContextUtterances | None
    instant_mode: bool | None
    audio_format: AudioFormat

    def http_url(self, path: str) -> str:
        return f"{self.base_url}{path}"


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        utterance_options: NotGivenOr[UtteranceOptions] = NOT_GIVEN,
        context: ContextGenerationId | ContextUtterances | None = None,
        instant_mode: bool = True,
        audio_format: AudioFormat = AudioFormat.mp3,
        base_url: str = DEFAULT_BASE_URL,
        http_session: aiohttp.ClientSession | None = None,
    ):
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=SUPPORTED_SAMPLE_RATE,
            num_channels=1,
        )
        key = api_key or os.environ.get("HUME_API_KEY")
        if not key:
            raise ValueError("Hume API key is required via api_key or HUME_API_KEY env var")

        self._opts = _TTSOptions(
            api_key=key,
            utterance_options=utterance_options if is_given(utterance_options) else {},
            context=context,
            instant_mode=instant_mode,
            audio_format=audio_format,
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
        utterance_options: NotGivenOr[UtteranceOptions] = NOT_GIVEN,
        context: NotGivenOr[ContextGenerationId | ContextUtterances] = NOT_GIVEN,
        instant_mode: NotGivenOr[bool] = NOT_GIVEN,
        audio_format: NotGivenOr[AudioFormat] = NOT_GIVEN,
    ) -> None:
        if is_given(utterance_options):
            self._opts.utterance_options = utterance_options
        if is_given(context):
            self._opts.context = context
        if is_given(instant_mode):
            self._opts.instant_mode = instant_mode
        if is_given(audio_format):
            self._opts.audio_format = audio_format

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> tts.ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)


class ChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        utterance: UtteranceOptions = dict(self._opts.utterance_options)  # Make a copy
        utterance["text"] = self._input_text

        payload: dict[str, Any] = {
            "utterances": [utterance],
            "strip_headers": True,
            "instant_mode": self._opts.instant_mode,
            "format": {"type": self._opts.audio_format.value},
        }
        if self._opts.context:
            payload["context"] = self._opts.context

        try:
            async with self._tts._ensure_session().post(
                self._opts.http_url(STREAM_PATH),
                headers={API_AUTH_HEADER: self._opts.api_key, "Content-Type": "application/json"},
                json=payload,
                timeout=aiohttp.ClientTimeout(total=None, sock_connect=self._conn_options.timeout),
                # large read_bufsize to avoid `ValueError: Chunk too big`
                read_bufsize=10 * 1024 * 1024,
            ) as resp:
                resp.raise_for_status()

                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=SUPPORTED_SAMPLE_RATE,
                    num_channels=self._tts.num_channels,
                    mime_type=f"audio/{self._opts.audio_format.value}",
                )
                
                print(f"Hume API response: {resp}")

                async for raw_line in resp.content:
                    print(f"Hume API response: {raw_line}")
                    line = raw_line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line.decode())
                    except json.JSONDecodeError as e:
                        continue

                    audio_b64 = data.get("audio")
                    if audio_b64:
                        output_emitter.push(base64.b64decode(audio_b64))

                output_emitter.flush()

        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except Exception as e:
            raise APIConnectionError() from e
