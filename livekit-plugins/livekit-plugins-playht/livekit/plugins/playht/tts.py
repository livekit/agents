from __future__ import annotations

import asyncio
import base64
import dataclasses
import json
import os
import io
from dataclasses import dataclass
from typing import Any, List, Literal
from pyht import Client, TTSOptions, Format

import aiohttp
from livekit.agents import tts, utils, tokenize
from livekit import rtc

from .log import logger
from .models import TTSEncoding, TTSEngines

_Encoding = Literal["mp3", "pcm"]


def _sample_rate_from_format(output_format: TTSEncoding) -> int:
    split = output_format.split("_") 
    return int(split[1])


def _encoding_from_format(output_format: TTSEncoding) -> _Encoding:
    if output_format.startswith("mp3"):
        return "mp3"
    elif output_format.startswith("pcm"):
        return "pcm"

    raise ValueError(f"Unknown format: {output_format}")


@dataclass
class Voice:
    id: str
    name: str
    voice_engine: TTSEngines


DEFAULT_VOICE = Voice(
    id="s3://peregrine-voices/mel22/manifest.json",
    name="Will",
    voice_engine="PlayHT2.0"
)

API_BASE_URL_V1 = "https://api.play.ht/api/v2"
AUTHORIZATION_HEADER = "AUTHORIZATION"
USERID_HEADER = "X-USER-ID"
PLAYHT_TTS_SAMPLE_RATE = 24000
PLAYHT_TTS_CHANNELS = 1


@dataclass
class _TTSOptions:
    api_key: str
    user_id: str
    voice: Voice
    base_url: str
    sample_rate: int


class TTS(tts.TTS):
    def __init__(
            self,
            *,
            voice: Voice = DEFAULT_VOICE,
            api_key: str | None = None,
            user_id: str | None = None,
            base_url: str | None = None,
            http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=False,
            ),
            sample_rate=PLAYHT_TTS_SAMPLE_RATE,
            num_channels=PLAYHT_TTS_CHANNELS,
        )
        api_key = api_key or os.environ.get("PLAYHT_API_KEY")
        if not api_key:
            raise ValueError("PLAYHT_API_KEY must be set")

        user_id = user_id or os.environ.get("PLAYHT_USER_ID")
        if not user_id:
            raise ValueError("PLAYHT_USER_ID mus be set")

        self._opts = _TTSOptions(
            voice=voice,
            user_id=user_id,
            api_key=api_key,
            base_url=base_url or API_BASE_URL_V1,
            sample_rate=self.sample_rate,
        )
        self._session = http_session

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

        return self._session

    async def list_voices(self) -> List[Voice]:
        async with self._ensure_session().get(
                f"{self._opts.base_url}/voices",
                headers={
                    "accept": "application/json",
                    AUTHORIZATION_HEADER: self._opts.api_key,
                    USERID_HEADER: self._opts.user_id
                },
        ) as resp:
            return _dict_to_voices_list(await resp.json())

    def synthesize(self, text: str) -> "ChunkedStream":
        return ChunkedStream(text, self._opts, self._ensure_session())


class ChunkedStream(tts.ChunkedStream):
    """Synthesize using the chunked api endpoint"""

    def __init__(
            self, text: str, opts: _TTSOptions, session: aiohttp.ClientSession
    ) -> None:
        super().__init__()
        self._text, self._opts, self._session = text, opts, session

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        stream = utils.audio.AudioByteStream(
            sample_rate=self._opts.sample_rate, num_channels=1
        )
        request_id = utils.shortuuid()
        segment_id = utils.shortuuid()
        parent_path = os.path.dirname(os.path.abspath(__file__))
        client = Client(self._opts.user_id, self._opts.api_key)
        options = TTSOptions(
            voice=self._opts.voice.id,
            sample_rate=PLAYHT_TTS_SAMPLE_RATE,
            format=Format.FORMAT_WAV,
            speed=1
        )

        try:
            response = client.tts(text=self._text, voice_engine=self._opts.voice.voice_engine, options=options)
            audio_buffer = io.BytesIO()

            for bytes_data in response:
                for frame in stream.write(bytes_data):
                    self._event_ch.send_nowait(
                        tts.SynthesizedAudio(
                            request_id=request_id,
                            segment_id=segment_id,
                            frame=frame,
                        )
                    )

                audio_buffer.write(bytes_data)

            for frame in stream.flush():
                self._event_ch.send_nowait(
                    tts.SynthesizedAudio(
                        request_id=request_id, segment_id=segment_id, frame=frame
                    )
                )
        except Exception as e:
            print(e)


def _dict_to_voices_list(data: dict[str, Any]):
    voices: List[Voice] = []
    for voice in data["text"]:
        voices.append(
            Voice(
                id=voice["id"],
                name=voice["name"],
                voice_engine=voice["voice_engine"]
            )
        )
    return voices

