from __future__ import annotations

import os
from dataclasses import dataclass

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    tts,
    utils,
)
from pyht import AsyncClient as PlayHTAsyncClient
from pyht.client import Format, Language, TTSOptions

from .log import logger


@dataclass
class _TTSOptions:
    voice: str
    format: Format
    sample_rate: int
    speed: float
    language: Language


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        voice: str = "s3://voice-cloning-zero-shot/775ae416-49bb-4fb6-bd45-740f205d20a1/jennifersaad/manifest.json",
        format: str = "mp3",
        sample_rate: int = 24000,
        speed: float = 1.0,
        language: str = "english",
        api_key: str | None = None,
        user_id: str | None = None,
    ) -> None:
        """
        Initialize the PlayHT TTS engine.

        Args:
            voice (str): The voice to use.
            format (str): The audio format. Options are 'mp3', 'wav', 'mulaw', 'flac', 'ogg', 'raw'.
            sample_rate (int): The sample rate in Hz. Options are 8000, 16000, 24000, 44100, 48000.
            speed (float): The speed of the audio. Default is 1.0.
            language (str): The language of the text. Default is 'english'.
            api_key (str): The PlayHT API key. Can be set via environment variable PLAY_HT_API_KEY.
            user_id (str): The PlayHT user ID. Can be set via environment variable PLAY_HT_USER_ID.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=True,
            ),
            sample_rate=sample_rate,
            num_channels=1,
        )

        api_key = api_key or os.environ.get("PLAY_HT_API_KEY")
        user_id = user_id or os.environ.get("PLAY_HT_USER_ID")

        if not api_key or not user_id:
            raise ValueError(
                "PlayHT API key and user ID are required, either as arguments or set PLAY_HT_API_KEY and PLAY_HT_USER_ID environment variables"
            )

        self._client = PlayHTAsyncClient(
            user_id=user_id,
            api_key=api_key,
        )
        self._opts = _TTSOptions(
            voice=voice,
            format=Format.FORMAT_MP3,  # default for now
            sample_rate=sample_rate,
            speed=speed,
            language=Language(language),
        )

    def update_options(
        self, *, voice: str | None = None, language: str | None = None
    ) -> None:
        """
        Update the TTS options.

        Args:
            voice (str, optional): The voice to use.
            language (str, optional): The language of the text.
        """
        if voice:
            self._opts.voice = voice
        if language:
            self._opts.language = Language(language)

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
        )

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> "SynthesizeStream":
        return SynthesizeStream(
            tts=self,
            conn_options=conn_options,
            opts=self._opts,
        )


class ChunkedStream(tts.ChunkedStream):
    def __init__(
        self,
        *,
        tts: TTS,
        input_text: str,
        conn_options: APIConnectOptions,
        opts: _TTSOptions,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._client = tts._client
        self._opts = opts
        self._mp3_decoder = utils.codecs.Mp3StreamDecoder()

    async def _run(self) -> None:
        request_id = utils.shortuuid()
        bstream = utils.audio.AudioByteStream(
            sample_rate=self._opts.sample_rate, num_channels=1
        )
        tts_options = TTSOptions(
            voice=self._opts.voice,
            format=self._opts.format,
            sample_rate=self._opts.sample_rate,
            speed=self._opts.speed,
            language=self._opts.language,
        )

        try:
            async for chunk in self._client.tts(self._input_text, tts_options):
                for frame in self._mp3_decoder.decode_chunk(chunk):
                    for frame in bstream.write(frame.data.tobytes()):
                        self._event_ch.send_nowait(
                            tts.SynthesizedAudio(
                                request_id=request_id,
                                frame=frame,
                            )
                        )
            for frame in bstream.flush():
                self._event_ch.send_nowait(
                    tts.SynthesizedAudio(request_id=request_id, frame=frame)
                )
        except Exception as e:
            logger.error(f"Error in PlayHT ChunkedStream: {e}")
            raise APIConnectionError() from e


class SynthesizeStream(tts.SynthesizeStream):
    def __init__(
        self,
        *,
        tts: TTS,
        conn_options: APIConnectOptions,
        opts: _TTSOptions,
    ):
        super().__init__(tts=tts, conn_options=conn_options)
        self._client = tts._client
        self._opts = opts
        self._mp3_decoder = utils.codecs.Mp3StreamDecoder()

    async def _run(self) -> None:
        request_id = utils.shortuuid()
        bstream = utils.audio.AudioByteStream(
            sample_rate=self._opts.sample_rate,
            num_channels=1,
        )
        tts_options = TTSOptions(
            voice=self._opts.voice,
            format=self._opts.format,
            sample_rate=self._opts.sample_rate,
            speed=self._opts.speed,
            language=self._opts.language,
        )

        async def text_stream():
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    continue
                if data is None:
                    break
                yield data

        try:
            async for chunk in self._client.stream_tts_input(
                text_stream(), tts_options
            ):
                for frame in self._mp3_decoder.decode_chunk(chunk):
                    for frame in bstream.write(frame.data.tobytes()):
                        self._event_ch.send_nowait(
                            tts.SynthesizedAudio(
                                request_id=request_id,
                                frame=frame,
                            )
                        )
            for frame in bstream.flush():
                self._event_ch.send_nowait(
                    tts.SynthesizedAudio(request_id=request_id, frame=frame)
                )
        except Exception as e:
            logger.error(f"Error in PlayHT SynthesizeStream: {e}")
            raise APIConnectionError() from e
