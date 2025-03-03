# Copyright 202 LiveKit, Inc.
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
from dataclasses import dataclass
from typing import Optional

import aiohttp
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tts,
    utils,
)

from .log import logger
from .models import TTSModels


@dataclass
class _TTSOptions:
    model: TTSModels | str
    speaker: str
    sample_rate: int
    speed_alpha: float
    reduce_latency: bool
    pause_between_brackets: bool
    phonemize_between_brackets: bool


DEFAULT_API_URL = "https://users.rime.ai/v1/rime-tts"


NUM_CHANNELS = 1


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        model: TTSModels | str = "mist",
        speaker: str = "lagoon",
        sample_rate: int = 22050,
        speed_alpha: float = 1.0,
        reduce_latency: bool = False,
        pause_between_brackets: bool = False,
        phonemize_between_brackets: bool = False,
        api_key: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=False,
            ),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )
        self._api_key = api_key or os.environ.get("RIME_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Rime API key is required, either as argument or set RIME_API_KEY environmental variable"
            )

        self._opts = _TTSOptions(
            model=model,
            speaker=speaker,
            sample_rate=sample_rate,
            speed_alpha=speed_alpha,
            reduce_latency=reduce_latency,
            pause_between_brackets=pause_between_brackets,
            phonemize_between_brackets=phonemize_between_brackets,
        )
        self._session = http_session

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

        return self._session

    def synthesize(
        self,
        text: str,
        *,
        conn_options: Optional[APIConnectOptions] = None,
        segment_id: str | None = None,
    ) -> "ChunkedStream":
        return ChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
            opts=self._opts,
            session=self._ensure_session(),
            segment_id=segment_id,
            api_key=self._api_key,
        )

    def update_options(
        self,
        *,
        model: TTSModels | None,
        speaker: str | None,
    ) -> None:
        self._opts.model = model or self._opts.model
        self._opts.speaker = speaker or self._opts.speaker


class ChunkedStream(tts.ChunkedStream):
    """Synthesize using the chunked api endpoint"""

    def __init__(
        self,
        tts: TTS,
        input_text: str,
        opts: _TTSOptions,
        session: aiohttp.ClientSession,
        conn_options: Optional[APIConnectOptions] = None,
        segment_id: str | None = None,
        api_key: str | None = None,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._opts = opts
        self._session = session
        self._segment_id = segment_id or utils.shortuuid()
        self._api_key = api_key

    async def _run(self) -> None:
        request_id = utils.shortuuid()
        headers = {
            "accept": "audio/mp3",
            "Authorization": f"Bearer {self._api_key}",
            "content-type": "application/json",
        }
        payload = {
            "speaker": self._opts.speaker,
            "text": self._input_text,
            "modelId": self._opts.model,
            "samplingRate": self._opts.sample_rate,
            "speedAlpha": self._opts.speed_alpha,
            "reduceLatency": self._opts.reduce_latency,
            "pauseBetweenBrackets": self._opts.pause_between_brackets,
            "phonemizeBetweenBrackets": self._opts.phonemize_between_brackets,
        }

        decoder = utils.codecs.AudioStreamDecoder(
            sample_rate=self._opts.sample_rate,
            num_channels=NUM_CHANNELS,
        )

        decode_task: Optional[asyncio.Task] = None
        try:
            async with self._session.post(
                DEFAULT_API_URL, headers=headers, json=payload
            ) as response:
                if not response.content_type.startswith("audio"):
                    content = await response.text()
                    logger.error("Rime returned non-audio data: %s", content)
                    return

                async def _decode_loop():
                    try:
                        async for bytes_data, _ in response.content.iter_chunks():
                            decoder.push(bytes_data)
                    finally:
                        decoder.end_input()

                decode_task = asyncio.create_task(_decode_loop())
                emitter = tts.SynthesizedAudioEmitter(
                    event_ch=self._event_ch,
                    request_id=request_id,
                    segment_id=self._segment_id,
                )
                async for frame in decoder:
                    emitter.push(frame)
                emitter.flush()

        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message,
                status_code=e.status,
                request_id=request_id,
                body=None,
            ) from e
        except Exception as e:
            raise APIConnectionError() from e
        finally:
            if decode_task:
                await utils.aio.gracefully_cancel(decode_task)
            await decoder.aclose()
