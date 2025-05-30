from __future__ import annotations

import asyncio
import os
import weakref
from dataclasses import dataclass, fields, replace

from pyht import AsyncClient as PlayAsyncClient
from pyht.client import Format, Language, TTSOptions as PlayOptions

from livekit.agents import APIConnectionError, APIConnectOptions, tokenize, tts, utils
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import is_given

from .log import logger
from .models import TTSModel

NUM_CHANNELS = 1


@dataclass
class _TTSOptions:
    model: TTSModel | str
    play_options: PlayOptions
    word_tokenizer: tokenize.WordTokenizer


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        user_id: NotGivenOr[str] = NOT_GIVEN,
        voice: str = "s3://voice-cloning-zero-shot/d9ff78ba-d016-47f6-b0ef-dd630f59414e/female-cs/manifest.json",
        language: str = "english",
        sample_rate: int = 24000,
        model: TTSModel | str = "Play3.0-mini",
        word_tokenizer: tokenize.WordTokenizer | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize the PlayAI TTS engine.

        Args:
            api_key (str): PlayAI API key.
            user_id (str): PlayAI user ID.
            voice (str): Voice manifest URL.
            model (TTSModel): TTS model, defaults to "Play3.0-mini".
            language (str): language, defaults to "english".
            sample_rate (int): sample rate (Hz), A number greater than or equal to 8000, and must be less than or equal to 48000
            word_tokenizer (tokenize.WordTokenizer): Tokenizer for processing text. Defaults to basic WordTokenizer.
            **kwargs: Additional options.
        """  # noqa: E501

        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=True,
            ),
            sample_rate=sample_rate,
            num_channels=1,
        )

        if not word_tokenizer:
            word_tokenizer = tokenize.basic.WordTokenizer(ignore_punctuation=False)

        pyht_api_key = api_key if is_given(api_key) else os.environ.get("PLAYHT_API_KEY")
        pyht_user_id = user_id if is_given(user_id) else os.environ.get("PLAYHT_USER_ID")

        if not pyht_api_key or not pyht_user_id:
            raise ValueError(
                "PlayHT API key and user ID are required. Set environment variables PLAYHT_API_KEY and PLAYHT_USER_ID or pass them explicitly."  # noqa: E501
            )

        _validate_kwargs(kwargs)

        self._opts = _TTSOptions(
            model=model,
            play_options=PlayOptions(
                voice=voice,
                format=Format.FORMAT_OGG,
                sample_rate=sample_rate,
                language=Language(language),
                **kwargs,
            ),
            word_tokenizer=word_tokenizer,
        )

        self._client = PlayAsyncClient(
            user_id=pyht_user_id,
            api_key=pyht_api_key,
        )
        self._streams = weakref.WeakSet[SynthesizeStream]()

    def update_options(
        self,
        *,
        voice: NotGivenOr[str] = NOT_GIVEN,
        model: NotGivenOr[TTSModel | str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        **kwargs,
    ) -> None:
        """
        Update the TTS options.
        """
        updates = {}
        if is_given(voice):
            updates["voice"] = voice
        if is_given(language):
            updates["language"] = Language(language)
        updates.update(kwargs)

        _validate_kwargs(updates)

        for key, value in updates.items():
            if value is not None:
                setattr(self._opts.play_options, key, value)

        if is_given(model):
            self._opts.model = model

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        await self._client.close()


class ChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts = tts
        self._opts = replace(self._tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=self._opts.play_options.sample_rate,  # type: ignore
            num_channels=1,
            mime_type="audio/ogg",
        )

        try:
            async for chunk in self._tts._client.tts(
                text=self._input_text,
                options=self._opts.play_options,
                voice_engine=self._opts.model,
                protocol="http",
                streaming=True,
            ):
                output_emitter.push(chunk)

        except Exception as e:
            raise APIConnectionError() from e


class SynthesizeStream(tts.SynthesizeStream):
    def __init__(
        self,
        *,
        tts: TTS,
        conn_options: APIConnectOptions,
    ):
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts = tts
        self._opts = replace(self._tts._opts)
        self._segments_ch = utils.aio.Chan[tokenize.WordStream]()

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        if self._opts.model == "PlayDialog-turbo":
            protocol = "http"
        else:
            protocol = "ws"

        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=self._opts.play_options.sample_rate,  # type: ignore
            num_channels=1,
            mime_type="audio/ogg",
            stream=True,
        )
        output_emitter.start_segment(segment_id=utils.shortuuid())

        input_task = asyncio.create_task(self._tokenize_input())
        try:
            text_stream = await self._create_text_stream()

            async for chunk in self._tts._client.stream_tts_input(
                text_stream=text_stream,
                options=self._opts.play_options,
                voice_engine=self._opts.model,
                protocol=protocol,
            ):
                output_emitter.push(chunk)

        except Exception as e:
            raise APIConnectionError() from e
        finally:
            await utils.aio.gracefully_cancel(input_task)

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
                    self._mark_started()
                    yield word.token

        return text_stream()


def _validate_kwargs(kwargs: dict) -> None:
    valid_keys = {field.name for field in fields(PlayOptions)}
    invalid_keys = set(kwargs.keys()) - valid_keys
    if invalid_keys:
        raise ValueError(f"Invalid parameters: {invalid_keys}. Allowed parameters: {valid_keys}")
