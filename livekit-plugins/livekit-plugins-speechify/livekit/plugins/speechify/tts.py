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

import asyncio
import base64
import os
from dataclasses import dataclass, replace

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
    tokenize,
    tts,
    utils,
)
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import is_given
from livekit.agents.voice.io import TimedString
from speechify.client import AsyncSpeechify
from speechify.core.api_error import ApiError
from speechify.types.get_voice import GetVoice

from .models import TTSModels

DEFAULT_VOICE_ID = "dominic_32"
DEFAULT_MODEL: TTSModels = "simba-3.2"
SAMPLE_RATE = 24000
NUM_CHANNELS = 1
AUDIO_FORMAT = "pcm"
MIME_TYPE = "audio/pcm"


@dataclass
class _TTSOptions:
    voice_id: str
    model: NotGivenOr[TTSModels]
    language: NotGivenOr[str]
    loudness_normalization: NotGivenOr[bool]
    text_normalization: NotGivenOr[bool]


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        voice_id: str = DEFAULT_VOICE_ID,
        model: TTSModels = DEFAULT_MODEL,
        language: NotGivenOr[str] = NOT_GIVEN,
        loudness_normalization: NotGivenOr[bool] = NOT_GIVEN,
        text_normalization: NotGivenOr[bool] = NOT_GIVEN,
        token: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
        client: AsyncSpeechify | None = None,
    ) -> None:
        """Create a new instance of Speechify TTS.

        Synthesis uses the Speechify ``/audio/speech`` endpoint, which returns
        raw PCM (24 kHz mono) together with word-level speech marks. ``stream()``
        splits input into sentences and issues one request per sentence, emitting
        audio and aligned word timestamps as each sentence completes for
        near-streaming time-to-first-audio.

        Args:
            voice_id: Id of the voice to synthesize with. The voice must support
                the chosen ``model`` (see the ``/v1/voices`` endpoint). Defaults
                to ``dominic_32``.
            model: Synthesis model. One of ``simba-english``,
                ``simba-multilingual``, ``simba-3.0`` or ``simba-3.2``. Defaults
                to ``simba-3.2``.
            language: BCP-47 language code of the input (e.g. ``en-US``).
            loudness_normalization: Normalize output loudness to a standard
                level. Increases latency slightly when enabled.
            text_normalization: Expand numbers, dates, etc. into words before
                synthesis. Increases latency slightly when enabled.
            token: Speechify API key. Falls back to the ``SPEECHIFY_API_KEY``
                environment variable.
            base_url: Override the Speechify API base URL.
            tokenizer: Sentence tokenizer used to chunk input in ``stream()``.
            client: A preconfigured ``AsyncSpeechify`` client. When provided,
                ``token`` and ``base_url`` are ignored.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True, aligned_transcript=True),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )

        if client is not None:
            self._client = client
        else:
            resolved_token = token if is_given(token) else os.environ.get("SPEECHIFY_API_KEY")
            if not resolved_token:
                raise ValueError(
                    "Speechify API key is required, either as the token argument "
                    "or via the SPEECHIFY_API_KEY environment variable"
                )
            self._client = AsyncSpeechify(
                token=resolved_token,
                base_url=base_url if is_given(base_url) else None,
            )

        self._tokenizer = tokenizer if is_given(tokenizer) else tokenize.basic.SentenceTokenizer()
        self._opts = _TTSOptions(
            voice_id=voice_id,
            model=model,
            language=language,
            loudness_normalization=loudness_normalization,
            text_normalization=text_normalization,
        )

    @property
    def model(self) -> str:
        return self._opts.model if is_given(self._opts.model) else "unknown"

    @property
    def provider(self) -> str:
        return "Speechify"

    async def list_voices(self) -> list[GetVoice]:
        """List the voices available for the configured Speechify account."""
        voices: list[GetVoice] = await self._client.voices.list()
        return voices

    def update_options(
        self,
        *,
        voice_id: NotGivenOr[str] = NOT_GIVEN,
        model: NotGivenOr[TTSModels] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        loudness_normalization: NotGivenOr[bool] = NOT_GIVEN,
        text_normalization: NotGivenOr[bool] = NOT_GIVEN,
    ) -> None:
        if is_given(voice_id):
            self._opts.voice_id = voice_id
        if is_given(model):
            self._opts.model = model
        if is_given(language):
            self._opts.language = language
        if is_given(loudness_normalization):
            self._opts.loudness_normalization = loudness_normalization
        if is_given(text_normalization):
            self._opts.text_normalization = text_normalization

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(
        self,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SynthesizeStream:
        return SynthesizeStream(tts=self, conn_options=conn_options)


def _request_kwargs(text: str, opts: _TTSOptions) -> dict[str, object]:
    options: dict[str, bool] = {}
    if is_given(opts.loudness_normalization):
        options["loudness_normalization"] = opts.loudness_normalization
    if is_given(opts.text_normalization):
        options["text_normalization"] = opts.text_normalization

    kwargs: dict[str, object] = {
        "audio_format": AUDIO_FORMAT,
        "input": text,
        "voice_id": opts.voice_id,
    }
    if is_given(opts.model):
        kwargs["model"] = opts.model
    if is_given(opts.language):
        kwargs["language"] = opts.language
    if options:
        kwargs["options"] = options
    return kwargs


def _timed_transcript(speech_marks: object, offset: float) -> list[TimedString]:
    chunks = getattr(speech_marks, "chunks", None)
    if not chunks:
        return []
    out: list[TimedString] = []
    for chunk in chunks:
        value = getattr(chunk, "value", None)
        start = getattr(chunk, "start_time", None)
        if value is None or start is None:
            continue
        end = getattr(chunk, "end_time", None)
        out.append(
            TimedString(
                text=value,
                start_time=start / 1000 + offset,
                end_time=(end / 1000 + offset) if end is not None else NOT_GIVEN,
            )
        )
    return out


def _raise_from(e: Exception) -> None:
    if isinstance(e, APIError):
        raise e
    if isinstance(e, ApiError):
        raise APIStatusError(
            message=str(e.body) if e.body is not None else "Speechify API error",
            status_code=e.status_code or -1,
            request_id=None,
            body=None,
        ) from None
    if isinstance(e, asyncio.TimeoutError):
        raise APITimeoutError() from None
    raise APIConnectionError() from e


class ChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        try:
            response = await self._tts._client.audio.speech(
                **_request_kwargs(self._input_text, self._opts),
                request_options={"timeout_in_seconds": int(self._conn_options.timeout)},
            )
            output_emitter.initialize(
                request_id=utils.shortuuid(),
                sample_rate=SAMPLE_RATE,
                num_channels=NUM_CHANNELS,
                mime_type=MIME_TYPE,
            )
            timed = _timed_transcript(response.speech_marks, 0.0)
            if timed:
                output_emitter.push_timed_transcript(timed)
            output_emitter.push(base64.b64decode(response.audio_data))
            output_emitter.flush()
        except Exception as e:
            _raise_from(e)


class SynthesizeStream(tts.SynthesizeStream):
    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
            mime_type=MIME_TYPE,
            stream=True,
        )
        output_emitter.start_segment(segment_id=request_id)

        sent_stream = self._tts._tokenizer.stream()

        async def _forward_input() -> None:
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    sent_stream.flush()
                    continue
                sent_stream.push_text(data)
            sent_stream.end_input()

        async def _synthesize() -> None:
            offset = 0.0
            async for ev in sent_stream:
                if not (text := ev.token.strip()):
                    continue
                self._mark_started()
                response = await self._tts._client.audio.speech(
                    **_request_kwargs(text, self._opts),
                    request_options={"timeout_in_seconds": int(self._conn_options.timeout)},
                )
                audio = base64.b64decode(response.audio_data)
                timed = _timed_transcript(response.speech_marks, offset)
                if timed:
                    output_emitter.push_timed_transcript(timed)
                output_emitter.push(audio)
                offset += len(audio) / (2 * SAMPLE_RATE * NUM_CHANNELS)

            output_emitter.end_segment()

        tasks = [
            asyncio.create_task(_forward_input()),
            asyncio.create_task(_synthesize()),
        ]
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            _raise_from(e)
        finally:
            await sent_stream.aclose()
            await utils.aio.cancel_and_wait(*tasks)
