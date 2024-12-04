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
    temperature: float
    top_p: float
    text_guidance: float
    voice_guidance: float
    style_guidance: float
    repetition_penalty: float
    disable_stabilization: bool


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        voice: str = "s3://voice-cloning-zero-shot/775ae416-49bb-4fb6-bd45-740f205d20a1/jennifersaad/manifest.json",
        sample_rate: int = 24000,
        speed: float = 1.0,
        language: str = "english",
        api_key: str | None = None,
        user_id: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        text_guidance: float | None = None,
        voice_guidance: float | None = None,
        style_guidance: float | None = None,
        repetition_penalty: float | None = None,
    ) -> None:
        """
        Initialize the PlayHT TTS engine.

        Args:
            voice (str): A URL pointing to a Play voice manifest file. (e.g. "s3://voice-cloning-zero-shot/775ae416-49bb-4fb6-bd45-740f205d20a1/jennifersaad/manifest.json").
            sample_rate (int): The sample rate in Hz. Options are 8000, 16000, 24000, 44100, 48000.
            speed (float): The speed of the audio. Default is 1.0.
            language (str): The language of the text. Default is 'english'.
            api_key (str): The PlayHT API key. Can be set via environment variable PLAY_HT_API_KEY.
            user_id (str): The PlayHT user ID. Can be set via environment variable PLAY_HT_USER_ID.
            > The following options are inference-time hyperparameters of the text-to-speech model; if unset, the model will use default values chosen by PlayHT.
            temperature (float): The temperature of the model.
            top_p (float): The top-p value of the model.
            text_guidance (float): The text guidance of the model.
            voice_guidance (float): The voice guidance of the model.
            style_guidance (float): (Play3.0-mini-http and Play3.0-mini-ws only) The style guidance of the model.
            repetition_penalty (float): The repetition penalty of the model.
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
            temperature=temperature,
            top_p=top_p,
            text_guidance=text_guidance,
            voice_guidance=voice_guidance,
            style_guidance=style_guidance,
            repetition_penalty=repetition_penalty,
        )

    def update_options(
        self,
        *,
        voice: str | None = None,
        language: str | None = None,
        sample_rate: int | None = None,
        speed: float | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        text_guidance: float | None = None,
        voice_guidance: float | None = None,
        style_guidance: float | None = None,
        repetition_penalty: float | None = None,
    ) -> None:
        """
        Update the TTS options.

        Args:
            voice (str, optional): The voice to use.
            language (str, optional): The language of the text.
            sample_rate (int, optional): The sample rate of the audio.
            speed (float, optional): The speed of the audio.
            temperature (float, optional): The temperature of the model.
            top_p (float, optional): The top-p value of the model.
            text_guidance (float, optional): The text guidance of the model.
            voice_guidance (float, optional): The voice guidance of the model.
            style_guidance (float, optional): The style guidance of the model.
            repetition_penalty (float, optional): The repetition penalty of the model.
        """
        updates = {
            "voice": voice,
            "language": Language(language) if language else None,
            "sample_rate": sample_rate,
            "speed": speed,
            "temperature": temperature,
            "top_p": top_p,
            "text_guidance": text_guidance,
            "voice_guidance": voice_guidance,
            "style_guidance": style_guidance,
            "repetition_penalty": repetition_penalty,
        }
        for k, v in updates.items():
            if v is not None:
                setattr(self._opts, k, v)

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
