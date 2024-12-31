from __future__ import annotations

import asyncio
import os
import weakref
from dataclasses import dataclass, fields

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    tokenize,
    tts,
    utils,
)
from pyht import AsyncClient as PlayHTAsyncClient  # type: ignore
from pyht.client import Format, Language, TTSOptions  # type: ignore

from .log import logger
from .models import TTSModel

NUM_CHANNELS = 1


@dataclass
class _Options:
    model: TTSModel | str
    tts_options: TTSOptions
    word_tokenizer: tokenize.WordTokenizer


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        user_id: str | None = None,
        voice: str = "s3://voice-cloning-zero-shot/d9ff78ba-d016-47f6-b0ef-dd630f59414e/female-cs/manifest.json",
        language: str = "english",
        sample_rate: int = 24000,
        model: TTSModel | str = "Play3.0-mini-ws",
        word_tokenizer: tokenize.WordTokenizer = tokenize.basic.WordTokenizer(
            ignore_punctuation=False
        ),
        **kwargs,
    ) -> None:
        """
        Initialize the PlayAI TTS engine.

        Args:
            api_key (str): PlayAI API key.
            user_id (str): PlayAI user ID.
            voice (str): Voice manifest URL.
            model (TTSModel): TTS model, defaults to "Play3.0-mini-ws".
            language (str): language, defaults to "english".
            sample_rate (int): sample rate (Hz), A number greater than or equal to 8000, and must be less than or equal to 48000
            word_tokenizer (tokenize.WordTokenizer): Tokenizer for processing text. Defaults to basic WordTokenizer.
            **kwargs: Additional options.
        """

        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=False,
            ),
            sample_rate=sample_rate,
            num_channels=1,
        )

        api_key = api_key or os.environ.get("PLAYHT_API_KEY")
        user_id = user_id or os.environ.get("PLAYHT_USER_ID")

        if not api_key or not user_id:
            raise ValueError(
                "PlayHT API key and user ID are required. Set environment variables PLAYHT_API_KEY and PLAYHT_USER_ID or pass them explicitly."
            )
        _validate_kwargs(kwargs)
        self._config = TTSOptions(
            voice=voice,
            format=Format.FORMAT_MP3,  # Default format for now
            sample_rate=sample_rate,
            language=Language(language),
            **kwargs,
        )

        self._opts = _Options(
            model=model,
            tts_options=self._config,
            word_tokenizer=word_tokenizer,
        )

        # Initialize client
        self._client = PlayHTAsyncClient(
            user_id=user_id,
            api_key=api_key,
        )
        self._streams = weakref.WeakSet[SynthesizeStream]()

    def update_options(
        self,
        *,
        voice: str | None = None,
        model: TTSModel | str | None = None,
        language: str | None = None,
        **kwargs,
    ) -> None:
        """
        Update the TTS options.
        """
        updates = {}
        if voice is not None:
            updates["voice"] = voice
        if language is not None:
            updates["language"] = Language(language)
        tts_kwargs = {k: v for k, v in kwargs.items()}

        self._config = _update_options(self._config, **updates, **tts_kwargs)

        if model is not None:
            self._opts.model = model

        for stream in self._streams:
            stream._config = _update_options(stream._config, **updates, **tts_kwargs)
            if model is not None:
                stream._opts.model = model

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
        stream = SynthesizeStream(
            tts=self,
            conn_options=conn_options,
            opts=self._opts,
        )
        self._streams.add(stream)
        return stream


class ChunkedStream(tts.ChunkedStream):
    def __init__(
        self,
        *,
        tts: TTS,
        input_text: str,
        conn_options: APIConnectOptions,
        opts: _Options,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._client = tts._client
        self._opts = opts
        self._config = self._opts.tts_options
        self._mp3_decoder = utils.codecs.Mp3StreamDecoder()

    async def _run(self) -> None:
        request_id = utils.shortuuid()
        bstream = utils.audio.AudioByteStream(
            sample_rate=self._config.sample_rate, num_channels=NUM_CHANNELS
        )

        try:
            async for chunk in self._client.tts(
                text=self._input_text,
                options=self._config,
                voice_engine=self._opts.model,
                streaming=True,
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
            raise APIConnectionError() from e


class SynthesizeStream(tts.SynthesizeStream):
    def __init__(
        self,
        *,
        tts: TTS,
        conn_options: APIConnectOptions,
        opts: _Options,
    ):
        super().__init__(tts=tts, conn_options=conn_options)
        self._client = tts._client
        self._opts = opts
        self._config = self._opts.tts_options
        self._segments_ch = utils.aio.Chan[tokenize.WordStream]()
        self._mp3_decoder = utils.codecs.Mp3StreamDecoder()

    async def _run(self) -> None:
        request_id = utils.shortuuid()
        segment_id = utils.shortuuid()
        bstream = utils.audio.AudioByteStream(
            sample_rate=self._config.sample_rate,
            num_channels=NUM_CHANNELS,
        )
        last_frame: rtc.AudioFrame | None = None

        def _send_last_frame(*, segment_id: str, is_final: bool) -> None:
            nonlocal last_frame
            if last_frame is not None:
                self._event_ch.send_nowait(
                    tts.SynthesizedAudio(
                        request_id=request_id,
                        segment_id=segment_id,
                        frame=last_frame,
                        is_final=is_final,
                    )
                )
                last_frame = None

        input_task = asyncio.create_task(self._tokenize_input())
        try:
            text_stream = await self._create_text_stream()
            async for chunk in self._client.stream_tts_input(
                text_stream=text_stream,
                options=self._config,
                voice_engine=self._opts.model,
            ):
                for frame in self._mp3_decoder.decode_chunk(chunk):
                    for frame in bstream.write(frame.data.tobytes()):
                        _send_last_frame(segment_id=segment_id, is_final=False)
                        last_frame = frame

            for frame in bstream.flush():
                _send_last_frame(segment_id=segment_id, is_final=False)
                last_frame = frame
            _send_last_frame(segment_id=segment_id, is_final=True)
        except Exception as e:
            raise APIConnectionError() from e
        finally:
            await utils.aio.gracefully_cancel(input_task)
            self._client.close()

    @utils.log_exceptions(logger=logger)
    async def _tokenize_input(self):
        # Converts incoming text into WordStreams and sends them into _segments_ch
        word_stream = None
        async for input in self._input_ch:
            if isinstance(input, str):
                if word_stream is None:
                    word_stream = self._opts.word_tokenizer.stream()
                    self._segments_ch.send_nowait(word_stream)
                word_stream.push_text(input)
            elif isinstance(input, self._FlushSentinel):
                if word_stream:
                    word_stream.end_input()
                word_stream = None
        self._segments_ch.close()

    @utils.log_exceptions(logger=logger)
    async def _create_text_stream(self):
        async def text_stream():
            async for word_stream in self._segments_ch:
                async for word in word_stream:
                    yield word.token

        return text_stream()


def _update_options(config: TTSOptions, **kwargs) -> TTSOptions:
    _validate_kwargs(kwargs)
    for k, v in kwargs.items():
        if v is not None:
            setattr(config, k, v)
    return config


def _validate_kwargs(kwargs: dict) -> None:
    valid_keys = {field.name for field in fields(TTSOptions)}
    invalid_keys = set(kwargs.keys()) - valid_keys
    if invalid_keys:
        raise ValueError(
            f"Invalid parameters: {invalid_keys}. Allowed parameters: {valid_keys}"
        )
