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

import aiohttp
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tts,
    utils,
)

from .log import logger
from .models import TTSEncoding, TTSModels

ACCEPT_HEADER = {
    "pcm": "audio/pcm",
    "mp3": "audio/mp3",
}


@dataclass
class _TTSOptions:
    model: TTSModels | str
    speaker: str
    audio_format: TTSEncoding
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
        audio_format: TTSEncoding = "pcm",
        sample_rate: int = 16000,
        speed_alpha: float = 1.0,
        reduce_latency: bool = False,
        pause_between_brackets: bool = False,
        phonemize_between_brackets: bool = False,
        api_key: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """
        Create a new instance of Rime TTS.

        ``api_key`` must be set to your Rime API key, either using the argument or by setting the
        ``RIME_API_KEY`` environmental variable.

        Args:
            model: The TTS model to use. defaults to "mist"
            speaker: The speaker to use. defaults to "lagoon"
            audio_format: The audio format to use. defaults to "pcm"
            sample_rate: The sample rate to use. defaults to 16000
            speed_alpha: The speed alpha to use. defaults to 1.0
            reduce_latency: Whether to reduce latency. defaults to False
            pause_between_brackets: Whether to pause between brackets. defaults to False
            phonemize_between_brackets: Whether to phonemize between brackets. defaults to False
            api_key: The Rime API key to use.
            http_session: The HTTP session to use. defaults to a new session
        """
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
            audio_format=audio_format,
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
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
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
        conn_options: APIConnectOptions,
        session: aiohttp.ClientSession,
        segment_id: str | None = None,
        api_key: str | None = None,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._opts = opts
        self._session = session
        self._segment_id = segment_id or utils.shortuuid()
        self._api_key = api_key

    async def _run(self) -> None:
        stream = utils.audio.AudioByteStream(
            sample_rate=self._opts.sample_rate, num_channels=NUM_CHANNELS
        )
        self._mp3_decoder = utils.codecs.Mp3StreamDecoder()
        request_id = utils.shortuuid()
        headers = {
            "accept": ACCEPT_HEADER[self._opts.audio_format],
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
        try:
            async with self._session.post(
                DEFAULT_API_URL, headers=headers, json=payload
            ) as response:
                if not response.content_type.startswith("audio"):
                    content = await response.text()
                    logger.error("Rime returned non-audio data: %s", content)
                    return

                if self._opts.audio_format == "mp3":
                    async for chunk in response.content.iter_chunked(1024):
                        for frame in self._mp3_decoder.decode_chunk(chunk):
                            self._event_ch.send_nowait(
                                tts.SynthesizedAudio(
                                    request_id=request_id,
                                    frame=frame,
                                    segment_id=self._segment_id,
                                )
                            )
                else:
                    async for chunk in response.content.iter_chunked(1024):
                        for frame in stream.write(chunk):
                            self._event_ch.send_nowait(
                                tts.SynthesizedAudio(
                                    request_id=request_id,
                                    frame=frame,
                                    segment_id=self._segment_id,
                                )
                            )

                    for frame in stream.flush():
                        self._event_ch.send_nowait(
                            tts.SynthesizedAudio(
                                request_id=request_id,
                                frame=frame,
                                segment_id=self._segment_id,
                            )
                        )

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
