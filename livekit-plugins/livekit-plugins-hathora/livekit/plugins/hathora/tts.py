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
import wave
import io
from collections import deque
from dataclasses import dataclass
from typing import Optional, Literal, ByteString, Tuple

import aiohttp

from livekit.agents import (
    APIConnectOptions,
    APIStatusError,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

from .constants import (
    API_AUTH_HEADER,
    USER_AGENT,
)
from .log import logger
from .models import (
    TTSModels,
    KokoroVoices
)


def _decode_audio_payload(
    audio_bytes: bytes,
    *,
    fallback_sample_rate: int,
    fallback_channels: int = 1,
) -> Tuple[bytes, int, int]:
    """Convert a WAV/PCM payload into raw PCM samples for TTSAudioRawFrame."""

    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wav_reader:
            channels = wav_reader.getnchannels()
            sample_rate = wav_reader.getframerate()
            frames = wav_reader.readframes(wav_reader.getnframes())
            return frames, sample_rate, channels
    except (wave.Error, EOFError):
        # If the payload is already raw PCM, just pass it through.
        return audio_bytes, fallback_sample_rate, fallback_channels

@dataclass
class BaseTTSOptions:
    base_url: str
    model: TTSModels
    api_key: Optional[str] = None

@dataclass
class KokoroTTSOptions(BaseTTSOptions):
    """Kokoro is an open-weight TTS model with 82 million parameters.

    This service uses the Hathora-hosted Kokoro model via the HTTP API.

    [Documentation](https://models.hathora.dev/model/hexgrad-kokoro-82m)

    @param base_url: Base URL for the Hathora Kokoro TTS API.
    @param api_key: API key for authentication with the Hathora service;
        provisiion one [here](https://models.hathora.dev/tokens).
    @param voice: Voice to use for synthesis. See the
        [Hathora docs](https://models.hathora.dev/model/hexgrad-kokoro-82m)
        for the default value; [list of voices](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md).
    @param speed: Speech speed multiplier (0.5 = half speed, 2.0 = double speed, default: 1).
    """

    model: Literal['hexgrad_kokoro'] = 'hexgrad_kokoro'
    voice: Optional[KokoroVoices | str] = None
    speed: Optional[float] = None

@dataclass
class ChatterboxTTSOptions(BaseTTSOptions):
    """Chatterbox is a public text-to-speech model optimized for natural and expressive voice synthesis.

    This service uses the Hathora-hosted Chatterbox model via the HTTP API.

    [Documentation](https://models.hathora.dev/model/resemble-ai-chatterbox)

    @param base_url: Base URL for the Hathora Kokoro TTS API.
    @param api_key: API key for authentication with the Hathora service;
        provisiion one [here](https://models.hathora.dev/tokens).
    @param exaggeration: Controls emotional intensity (default: 0.5).
    @param audio_prompt: Reference audio file for voice cloning.
    @param cfg_weight: Controls adherence to reference voice (default: 0.5).
    """

    model: Literal['resembleai_chatterbox'] = 'resembleai_chatterbox'
    audio_prompt: Optional[ByteString] = None
    exaggeration: Optional[float] = None
    cfg_weight: Optional[float] = None

class TTS(tts.TTS):
    def __init__(
        self,
        *,
        opts: KokoroTTSOptions | ChatterboxTTSOptions,
    ):
        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=False,
            ),
            sample_rate=24000,
            num_channels=1
        )

        self._opts = opts
        self._opts.api_key = self._opts.api_key or os.environ.get("HATHORA_API_KEY")

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "Hathora"

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> tts.ChunkedStream:
        if isinstance(self._opts, KokoroTTSOptions):
            return KokoroChunkedStream(
                tts=self,
                input_text=text,
                conn_options=conn_options
            )
        elif isinstance(self._opts, ChatterboxTTSOptions):
            return ChatterboxChunkedStream(
                tts=self,
                input_text=text,
                conn_options=conn_options
            )

class KokoroChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts = tts
        self._opts = tts._opts
        self._text = input_text
        self._conn_options = conn_options

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        should_flush = False
        try:
            url = f"{self._opts.base_url}"

            payload = {
                "text": self._text
            }

            if self._opts.voice is not None:
                payload["voice"] = self._opts.voice
            if self._opts.speed is not None:
                payload["speed"] = self._opts.speed

            output_emitter.initialize(
                request_id=utils.shortuuid(),
                sample_rate=24000,
                num_channels=1,
                mime_type="audio/pcm",
            )
            should_flush = True

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    headers={
                        API_AUTH_HEADER: f"Bearer {self._opts.api_key}", "Accept": "application/octet-stream",
                        "User-Agent": USER_AGENT,
                    },
                    json=payload,
                ) as resp:
                    audio_data = await resp.read()

            pcm_audio, sample_rate, num_channels = _decode_audio_payload(
                audio_data,
                fallback_sample_rate=self._tts.sample_rate or 24000,
            )

            output_emitter.push(pcm_audio)

        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=f"Hathora error: {e}",
                status_code=e.status,
                request_id=None,
                body=None
            )
        finally:
            if should_flush:
                output_emitter.flush()

class ChatterboxChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts = tts
        self._opts = tts._opts
        self._text = input_text
        self._conn_options = conn_options

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        should_flush = False
        try:
            url = f"{self._opts.base_url}"

            url_query_params = []
            if self._opts.exaggeration is not None:
                url_query_params.append(f"exaggeration={self._opts.exaggeration}")
            if self._opts.cfg_weight is not None:
                url_query_params.append(f"cfg_weight={self._opts.cfg_weight}")

            if len(url_query_params) > 0:
                url += "?" + "&".join(url_query_params)

            api_key = self._opts.api_key or os.getenv("HATHORA_API_KEY")

            form_data = aiohttp.FormData()
            form_data.add_field("text", self._text)

            if self._opts.audio_prompt is not None:
                form_data.add_field("audio_prompt", self._opts.audio_prompt, filename="audio.wav", content_type="application/octet-stream")

            output_emitter.initialize(
                request_id=utils.shortuuid(),
                sample_rate=24000,
                num_channels=1,
                mime_type="audio/pcm",
            )
            should_flush = True

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    headers={"Authorization": f"Bearer {api_key}"},
                    data=form_data,
                ) as resp:
                    audio_data = await resp.read()

            pcm_audio, sample_rate, num_channels = _decode_audio_payload(
                audio_data,
                fallback_sample_rate=self._tts.sample_rate or 24000,
            )

            output_emitter.push(pcm_audio)

        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=f"Hathora error: {e}",
                status_code=e.status,
                request_id=None,
                body=None
            )
        finally:
            if should_flush:
                output_emitter.flush()
