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
import os
from dataclasses import dataclass

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tts,
    utils,
)
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import is_given

from .log import logger
from .models import Gender, TTSEncoding, TTSModels, VoiceType

_DefaultEncoding: TTSEncoding = "ogg_24000"


def _sample_rate_from_encoding(output_encoding: TTSEncoding) -> int:
    split = output_encoding.split("_")
    return int(split[1])


def _audio_format_from_encoding(encoding: TTSEncoding) -> str:
    split = encoding.split("_")
    return split[0]


DEFAULT_VOICE_ID = "jack"
API_BASE_URL_V1 = "https://api.sws.speechify.com/v1"
AUTHORIZATION_HEADER = "Authorization"
CALLER_HEADER = "x-caller"


@dataclass
class Voice:
    id: str
    type: VoiceType
    display_name: str
    gender: Gender
    avatar_image: str | None
    models: list[TTSModels]
    locale: str


@dataclass
class _TTSOptions:
    base_url: NotGivenOr[str]
    token: NotGivenOr[str]
    voice_id: str
    encoding: TTSEncoding
    language: NotGivenOr[str]
    model: NotGivenOr[TTSModels]
    loudness_normalization: NotGivenOr[bool]
    text_normalization: NotGivenOr[bool]
    follow_redirects: bool
    sample_rate: int


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        voice_id: NotGivenOr[str] = DEFAULT_VOICE_ID,
        encoding: NotGivenOr[TTSEncoding] = NOT_GIVEN,
        model: NotGivenOr[TTSModels] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        loudness_normalization: NotGivenOr[bool] = NOT_GIVEN,
        text_normalization: NotGivenOr[bool] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
        follow_redirects: bool = True,
    ) -> None:
        """
        Create a new instance of Speechify TTS.

        Args:
            voice_id (NotGivenOr[str]): Voice ID. Defaults to `cliff`.
            encoding (NotGivenOr[TTSEncoding]): Audio encoding to use. Optional. Defaults to `wav_48000`.
            model (NotGivenOr[TTSModels]): TTS model to use. Optional.
            base_url (NotGivenOr[str]): Custom base URL for the API. Optional.
            api_key (NotGivenOr[str]): Speechify API key. Can be set via argument or `SPEECHIFY_API_KEY` environment variable
            language (NotGivenOr[str]): Language code for the TTS model. Optional.
            loudness_normalization (NotGivenOr[bool]): Whether to normalize the loudness of the audio. Optional.
            text_normalization (NotGivenOr[bool]): Whether to normalize the text. Optional.
            http_session (aiohttp.ClientSession | None): Custom HTTP session for API requests. Optional.
            follow_redirects (bool): Whether to follow redirects in HTTP requests. Defaults to True.
        """  # noqa: E501

        if not is_given(encoding):
            encoding = _DefaultEncoding

        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=False,
            ),
            sample_rate=_sample_rate_from_encoding(encoding),
            num_channels=1,
        )

        speechify_token = api_key if is_given(api_key) else os.environ.get("SPEECHIFY_API_KEY")
        if not (speechify_token):
            raise ValueError(
                "Speechify API key is required, either as argument or set SPEECHIFY_API_KEY environment variable"  # noqa: E501
            )

        self._opts = _TTSOptions(
            model=model,
            voice_id=voice_id,
            language=language,
            base_url=base_url if is_given(base_url) else API_BASE_URL_V1,
            token=speechify_token,
            follow_redirects=follow_redirects,
            encoding=encoding,
            sample_rate=_sample_rate_from_encoding(encoding),
            loudness_normalization=loudness_normalization,
            text_normalization=text_normalization,
        )
        self._session = http_session

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

        return self._session

    async def list_voices(self) -> list[Voice]:
        async with self._ensure_session().get(
            f"{self._opts.base_url}/voices",
            headers=_get_headers(self._opts.token),
        ) as resp:
            return await resp.json()

    def update_options(
        self,
        *,
        voice_id: NotGivenOr[str] = NOT_GIVEN,
        model: NotGivenOr[TTSModels] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        loudness_normalization: NotGivenOr[bool] = NOT_GIVEN,
        text_normalization: NotGivenOr[bool] = NOT_GIVEN,
    ) -> None:
        """
        Args:
            voice_id (NotGivenOr[str]): Voice ID.
            model (NotGivenOr[TTSModels | str]): TTS model to use.
            language (NotGivenOr[str]): Language code for the TTS model.
        """
        if is_given(model):
            self._opts.model = model
        if is_given(voice_id):
            self._opts.voice_id = voice_id
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
        *,
        tts: TTS,
        input_text: str,
        opts: _TTSOptions,
        conn_options: APIConnectOptions,
        session: aiohttp.ClientSession,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._opts, self._session = opts, session

    async def _run(self) -> None:
        request_id = utils.shortuuid()
        data = {
            "input": self._input_text,
            "voice_id": self._opts.voice_id,
            "language": self._opts.language if is_given(self._opts.language) else None,
            "model": self._opts.model if is_given(self._opts.model) else None,
            "audio_format": _audio_format_from_encoding(self._opts.encoding),
            "options": {
                "loudness_normalization": self._opts.loudness_normalization
                if is_given(self._opts.loudness_normalization)
                else None,
                "text_normalization": self._opts.text_normalization
                if is_given(self._opts.text_normalization)
                else None,
            },
        }

        decoder = utils.codecs.AudioStreamDecoder(
            sample_rate=self._opts.sample_rate,
            num_channels=1,
        )

        decode_task: asyncio.Task | None = None
        try:
            async with self._session.post(
                _synthesize_url(self._opts),
                headers=_get_headers(self._opts.token, encoding=self._opts.encoding),
                json=data,
                timeout=aiohttp.ClientTimeout(connect=self._conn_options.timeout, total=30),
            ) as resp:
                if not resp.content_type.startswith("audio/"):
                    content = await resp.text()
                    logger.error("speechify returned non-audio data: %s", content)
                    return

                async def _decode_loop():
                    try:
                        async for bytes_data, _ in resp.content.iter_chunks():
                            decoder.push(bytes_data)
                    finally:
                        decoder.end_input()

                decode_task = asyncio.create_task(_decode_loop())
                emitter = tts.SynthesizedAudioEmitter(
                    event_ch=self._event_ch,
                    request_id=request_id,
                )
                async for frame in decoder:
                    emitter.push(frame)
                emitter.flush()
                await decode_task
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
        finally:
            if decode_task:
                await utils.aio.gracefully_cancel(decode_task)
            await decoder.aclose()


def _synthesize_url(opts: _TTSOptions) -> str:
    """Construct the Speechify stream URL."""
    return f"{opts.base_url}/audio/stream"


def _get_headers(token: str, *, encoding: TTSEncoding | None = None) -> dict[str, str]:
    """Construct the headers for the Speechify API."""
    headers = {
        AUTHORIZATION_HEADER: f"Bearer {token}" if not token.startswith("Bearer ") else token
    }

    if encoding:
        accept = ""
        format = _audio_format_from_encoding(encoding)
        if format == "ogg":
            accept = "audio/ogg"
        elif format == "mp3":
            accept = "audio/mpeg"
        elif format == "aac":
            accept = "audio/aac"

        # docs does not specify mime type for wav
        # https://docs.sws.speechify.com/v1/api-reference/api-reference/tts/audio/stream

        if accept:
            headers["Accept"] = accept
    headers[CALLER_HEADER] = "livekit"
    return headers
