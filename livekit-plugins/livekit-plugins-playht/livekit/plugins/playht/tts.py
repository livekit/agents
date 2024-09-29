from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, List, Literal

import aiohttp

from livekit.agents import tts, utils

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
    voice_engine="PlayHT2.0",
)

ACCEPT_HEADER = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "ogg": "audio/ogg",
    "flac": "audio/flac",
    "mulaw": "audio/basic",  # commonly used for mulaw
}

API_BASE_URL_V2 = "https://api.play.ht/api/v2"
AUTHORIZATION_HEADER = "AUTHORIZATION"
USERID_HEADER = "X-USER-ID"
PLAYHT_TTS_SAMPLE_RATE = 48000
PLAYHT_TTS_CHANNELS = 1

_TTSEncoding = Literal["mp3", "wav", "ogg", "flac", "mulaw"]


@dataclass
class _TTSOptions:
    api_key: str
    user_id: str
    voice: Voice
    base_url: str
    sample_rate: int
    encoding: _TTSEncoding


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        voice: Voice = DEFAULT_VOICE,
        api_key: str | None = None,
        user_id: str | None = None,
        base_url: str | None = None,
        encoding: _TTSEncoding | None = "wav",
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
            base_url=base_url or API_BASE_URL_V2,
            sample_rate=PLAYHT_TTS_SAMPLE_RATE,
            encoding=encoding,
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
                USERID_HEADER: self._opts.user_id,
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
        self._mp3_decoder = utils.codecs.Mp3StreamDecoder()
        request_id = utils.shortuuid()
        segment_id = utils.shortuuid()
        url = f"{API_BASE_URL_V2}/tts/stream"
        headers = {
            "accept": ACCEPT_HEADER[self._opts.encoding],
            "content-type": "application/json",
            AUTHORIZATION_HEADER: self._opts.api_key,
            USERID_HEADER: self._opts.user_id,
        }
        json_data = {
            "text": self._text,
            "output_format": self._opts.encoding,
            "voice": self._opts.voice.id,
        }
        async with self._session.post(url=url, headers=headers, json=json_data) as resp:
            if not resp.content_type.startswith("audio/"):
                content = await resp.text()
                logger.error("playHT returned non-audio data: %s", content)
                return

            encoding = _encoding_from_format(self._opts.encoding)
            if encoding == "mp3":
                async for bytes_data, _ in resp.content.iter_chunks():
                    for frame in self._mp3_decoder.decode_chunk(bytes_data):
                        self._event_ch.send_nowait(
                            tts.SynthesizedAudio(
                                request_id=request_id,
                                segment_id=segment_id,
                                frame=frame,
                            )
                        )
            else:
                async for bytes_data, _ in resp.content.iter_chunks():
                    for frame in stream.write(bytes_data):
                        self._event_ch.send_nowait(
                            tts.SynthesizedAudio(
                                request_id=request_id,
                                segment_id=segment_id,
                                frame=frame,
                            )
                        )

                for frame in stream.flush():
                    self._event_ch.send_nowait(
                        tts.SynthesizedAudio(
                            request_id=request_id, segment_id=segment_id, frame=frame
                        )
                    )


def _dict_to_voices_list(data: dict[str, Any]):
    voices: List[Voice] = []
    for voice in data["text"]:
        voices.append(
            Voice(
                id=voice["id"], name=voice["name"], voice_engine=voice["voice_engine"]
            )
        )
    return voices
