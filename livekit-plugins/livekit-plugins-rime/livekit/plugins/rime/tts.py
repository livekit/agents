from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import List, Literal

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

_Encoding = Literal["pcm", "mp3", "x-mulaw"]


def _sampling_rate_from_format(output_format: TTSEncoding) -> int:
    split = output_format.split("_")
    return int(split[1])


ACCEPT_HEADER = {
    "pcm": "audio/pcm",
    "mp3": "audio/mp3",
    "mulaw": "audio/x-mulaw",
}


def _sampling_rate_from_format(output_format: TTSEncoding) -> int | None:
    if output_format.startswith("pcm"):
        return 16000
    elif output_format.startswith("mp3"):
        return 22050
    elif output_format.startswith("mulaw"):
        return None

    raise ValueError(f"Unknown format: {output_format}")


@dataclass
class _TTSOptions:
    modelId: TTSModels | str
    speaker: str
    samplingRate: int
    speedAlpha: float
    reduceLatency: bool
    pauseBetweenBrackets: bool
    phonemizeBetweenBrackets: bool
    api_key: str


@dataclass
class Voice:
    name: str


DEFAULT_API_URL = "https://users.rime.ai/v1/rime-tts"
VOICES_URL = "https://users.rime.ai/data/voices/all.json"

DEFAULT_MODEL_ID = "mist"
DEFAULT_SPEAKER = "lagoon"
DEFAULT_SPEED_ALPHA = 1.0
DEFAULT_PAUSE_BETWEEN_BRACKETS = False
DEFAULT_PHONEMIZE_BETWEEN_BRACKETS = False
AUTHORIZATION_HEADER = "rime-api-key"
RIME_TTS_CHANNELS = 1


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        modelId: TTSModels | str = DEFAULT_MODEL_ID,
        speaker: str = DEFAULT_SPEAKER,
        audioFormat: TTSEncoding | _Encoding = "wav",
        samplingRate: int | None = None,
        speedAlpha: float = DEFAULT_SPEED_ALPHA,
        reduceLatency: bool = False,
        pauseBetweenBrackets: bool = DEFAULT_PAUSE_BETWEEN_BRACKETS,
        phonemizeBetweenBrackets: bool = DEFAULT_PHONEMIZE_BETWEEN_BRACKETS,
        api_url: str = DEFAULT_API_URL,
        api_key: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=False,
            ),
            sampling_rate=_sampling_rate_from_format(audioFormat),
            num_channels=RIME_TTS_CHANNELS,
        )
        api_key = api_key or os.environ.get("RIME_API_KEY")
        if not api_key:
            raise ValueError("RIME_API_KEY must be set")

        self._opts = _TTSOptions(
            modelId=modelId,
            speaker=speaker,
            samplingRate=samplingRate,
            speedAlpha=speedAlpha,
            reduceLatency=reduceLatency,
            pauseBetweenBrackets=pauseBetweenBrackets,
            phonemizeBetweenBrackets=phonemizeBetweenBrackets,
            api_url=api_url,
            api_key=api_key,
        )
        self._session = http_session

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

        return self._session

    async def list_voices(self) -> List[Voice]:
        async with self._ensure_session().get(VOICES_URL) as resp:
            return await resp.json()

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "ChunkedStream":
        return ChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
            opts=self._opts,
            session=self._ensure_session(),
        )


class ChunkedStream(tts.ChunkedStream):
    """Synthesize using the chunked api endpoint"""

    def __init__(
        self,
        tts: TTS,
        input_text: str,
        opts: _TTSOptions,
        conn_options: APIConnectOptions,
        session: aiohttp.ClientSession,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._opts, self._session = opts, session

    async def _run(self) -> None:
        stream = utils.audio.AudioByteStream(
            sampling_rate=self._opts.samplingRate, num_channels=RIME_TTS_CHANNELS
        )
        self._mp3_decoder = utils.codecs.Mp3StreamDecoder()
        request_id = utils.shortuuid()
        url = self._opts.api_url
        headers = {
            "accept": ACCEPT_HEADER[self._opts.audioFormat],
            AUTHORIZATION_HEADER: f"Bearer {self._opts.api_key}",
            "content-type": "application/json",
        }
        payload = {
            "speaker": self._opts.speaker,
            "text": self._input_text,
            "modelId": self._opts.modelId,
            "samplingRate": self._opts.samplingRate,
            "speedAlpha": self._opts.speedAlpha,
            "reduceLatency": self._opts.reduceLatency,
            "pauseBetweenBrackets": self._opts.pauseBetweenBrackets,
            "phonemizeBetweenBrackets": self._opts.phonemizeBetweenBrackets,
        }
        try:
            async with self._session.post(
                url=url, headers=headers, json=payload, stream=True
            ) as response:
                if not response.content_type.startswith("audio"):
                    content = await response.text()
                    logger.error("Rime returned non-audio data: %s", content)
                    return

                if self._opts.audioFormat == "mp3":
                    async for chunk in response.iter_content(chunk_size=1024):
                        for frame in self._mp3_decoder.decode_chunk(chunk):
                            self._event_ch.send_nowait(
                                tts.SynthesizedAudio(
                                    request_id=request_id,
                                    frame=frame,
                                )
                            )
                else:
                    async for chunk in response.iter_content(chunk_size=1024):
                        for frame in stream.write(chunk):
                            self._event_ch.send_nowait(
                                tts.SynthesizedAudio(
                                    request_id=request_id,
                                    frame=frame,
                                )
                            )

                    for frame in stream.flush():
                        self._event_ch.send_nowait(
                            tts.SynthesizedAudio(request_id=request_id, frame=frame)
                        )

        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message,
                status_code=e.status,
                request_id=None,
                body=None,
            ) from e
        except Exception as e:
            raise APIConnectionError() from e
