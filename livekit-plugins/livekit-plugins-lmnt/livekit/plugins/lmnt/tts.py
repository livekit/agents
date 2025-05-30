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
import json
import os
import weakref
from dataclasses import dataclass, replace
from typing import Final

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
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

from .log import logger
from .models import LMNTAudioFormats, LMNTLanguages, LMNTModels, LMNTSampleRate

LMNT_BASE_URL: Final[str] = "https://api.lmnt.com/v1/ai/speech/bytes"
LMNT_STREAM_URL: Final[str] = "https://api.lmnt.com/v1/ai/speech/stream"
NUM_CHANNELS: Final[int] = 1
MIME_TYPE: dict[str, str] = {
    "aac": "audio/aac",
    "mp3": "audio/mpeg",
    "mulaw": "audio/basic",
    "raw": "application/octet-stream",
    "wav": "audio/wav",
}


@dataclass
class _TTSOptions:
    sample_rate: LMNTSampleRate
    model: LMNTModels
    format: LMNTAudioFormats
    language: LMNTLanguages
    num_channels: int
    voice: str
    api_key: str
    temperature: float
    top_p: float


class TTS(tts.TTS):
    """
    Text-to-Speech (TTS) plugin for LMNT.
    """

    def __init__(
        self,
        *,
        model: LMNTModels = "blizzard",
        voice: str = "leah",
        language: LMNTLanguages | None = None,
        format: LMNTAudioFormats = "mp3",
        sample_rate: LMNTSampleRate = 24000,
        api_key: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
        temperature: float = 1.0,
        top_p: float = 0.8,
        use_websockets: bool = True,
    ) -> None:
        """
        Create a new instance of LMNT TTS.

        See: https://docs.lmnt.com/api-reference/speech/synthesize-speech-bytes

        Args:
            model: The model to use for synthesis. Default is "blizzard".
                Learn more at: https://docs.lmnt.com/guides/models
            voice: The voice ID to use. Default is "leah". Find more amazing voices at https://app.lmnt.com/
            language: Two-letter ISO 639-1 language code. Defaults to None.
                See: https://docs.lmnt.com/api-reference/speech/synthesize-speech-bytes#body-language
            format: Output file format. Options: aac, mp3, mulaw, raw, wav. Default is "mp3".
            sample_rate: Output sample rate in Hz. Default is 24000.
                See: https://docs.lmnt.com/api-reference/speech/synthesize-speech-bytes#body-sample-rate
            api_key: API key for authentication. Defaults to the LMNT_API_KEY environment variable.
            http_session: Optional aiohttp ClientSession. A new session is created if not provided.
            temperature: Influences how expressive and emotionally varied the speech becomes.
                Lower values (like 0.3) create more neutral, consistent speaking styles.
                Higher values (like 1.0) allow for more dynamic emotional range and speaking styles.
                Default is 1.0.
            top_p: Controls the stability of the generated speech.
                A lower value (like 0.3) produces more consistent, reliable speech.
                A higher value (like 0.9) gives more flexibility in how words are spoken,
                but might occasionally produce unusual intonations or speech patterns.
                Default is 0.8.
            use_websockets: If True, uses websockets for real time streaming synthesis.
                Else, synthesis is done chunk by chunk over HTTP. Default is True.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True if use_websockets else False),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )
        api_key = api_key or os.environ.get("LMNT_API_KEY")
        if not api_key:
            raise ValueError(
                "LMNT API key is required. "
                "Set it via environment variable or pass it as an argument."
            )

        if not language:
            language = "auto" if model == "blizzard" else "en"

        self._opts = _TTSOptions(
            model=model,
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
            language=language,
            voice=voice,
            format=format,
            api_key=api_key,
            temperature=temperature,
            top_p=top_p,
        )

        self._session = http_session
        self._pool = utils.ConnectionPool[aiohttp.ClientWebSocketResponse](
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
            max_session_duration=90,
            mark_refreshed_on_get=True,
        )
        self._streams = weakref.WeakSet[SynthesizeStream]()

    async def _connect_ws(self, timeout: float) -> aiohttp.ClientWebSocketResponse:
        session = self._ensure_session()
        return await asyncio.wait_for(session.ws_connect(LMNT_STREAM_URL), timeout)

    async def _close_ws(self, ws: aiohttp.ClientWebSocketResponse):
        await ws.close()

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
        )

    def update_options(
        self,
        *,
        model: NotGivenOr[LMNTModels] = NOT_GIVEN,
        voice: NotGivenOr[str] = NOT_GIVEN,
        language: NotGivenOr[LMNTLanguages] = NOT_GIVEN,
        format: NotGivenOr[LMNTAudioFormats] = NOT_GIVEN,
        sample_rate: NotGivenOr[LMNTSampleRate] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        """
        Update the TTS options.

        Args:
            model: The model to use for synthesis. Learn more at: https://docs.lmnt.com/guides/models
            voice: The voice ID to update.
            language: Two-letter ISO 639-1 code.
                See: https://docs.lmnt.com/api-reference/speech/synthesize-speech-bytes#body-language
            format: Audio output format. Options: aac, mp3, mulaw, raw, wav.
            sample_rate: Output sample rate in Hz.
            temperature: Controls the expressiveness of the speech. A number between 0.0 and 1.0.
            top_p: Controls the stability of the generated speech. A number between 0.0 and 1.0.
        """
        if is_given(model):
            self._opts.model = model
        if is_given(voice):
            self._opts.voice = voice
        if is_given(language):
            self._opts.language = language
        if is_given(format):
            self._opts.format = format
        if is_given(sample_rate):
            self._opts.sample_rate = sample_rate
        if is_given(temperature):
            self._opts.temperature = temperature
        if is_given(top_p):
            self._opts.top_p = top_p

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

        return self._session

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()
        await self._pool.aclose()


class ChunkedStream(tts.ChunkedStream):
    """Synthesize text to speech in chunks."""

    def __init__(
        self,
        *,
        tts: TTS,
        input_text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        data = {
            "text": self._input_text,
            "voice": self._opts.voice,
            "language": self._opts.language,
            "sample_rate": self._opts.sample_rate,
            "model": self._opts.model,
            "format": self._opts.format,
            "temperature": self._opts.temperature,
            "top_p": self._opts.top_p,
        }

        try:
            async with self._tts._ensure_session().post(
                LMNT_BASE_URL,
                headers={
                    "Content-Type": "application/json",
                    "X-API-Key": self._opts.api_key,
                },
                json=data,
                timeout=aiohttp.ClientTimeout(
                    total=30,
                    sock_connect=self._conn_options.timeout,
                ),
            ) as resp:
                resp.raise_for_status()
                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=self._opts.sample_rate,
                    num_channels=NUM_CHANNELS,
                    mime_type=MIME_TYPE[self._opts.format],
                )
                async for data, _ in resp.content.iter_chunks():
                    output_emitter.push(data)

                output_emitter.flush()
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message,
                status_code=e.status,
                request_id=None,
                body=None,
            ) from None
        except Exception as e:
            raise APIConnectionError() from e


class SynthesizeStream(tts.SynthesizeStream):
    """Synthesize text to speech in a stream using websockets"""

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._sent_tokenizer_stream = tokenize.basic.SentenceTokenizer(min_sentence_len=10).stream()
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()

        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=NUM_CHANNELS,
            mime_type=MIME_TYPE[self._opts.format],
            stream=True,
        )

        async def _sentence_stream_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            """
            This task sends sentences to the LMNT service.
            It uses a SentenceTokenizer to split the input text into sentences.
            """
            async for sentence in self._sent_tokenizer_stream:
                self._mark_started()
                await ws.send_str(json.dumps({"text": sentence.token + " "}))
            await ws.send_str('{"eof": true}')

        async def _input_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            async for text in self._input_ch:
                if isinstance(text, self._FlushSentinel):
                    self._sent_tokenizer_stream.flush()
                    continue

                self._sent_tokenizer_stream.push_text(text)
            self._sent_tokenizer_stream.end_input()

        async def _recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            output_emitter.start_segment(segment_id=request_id)
            while True:
                msg = await ws.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    output_emitter.end_input()
                    break

                try:
                    output_emitter.push(msg.data)
                except Exception as e:
                    logger.warning("Failed to parse LMNT message: %s", e)

        try:
            async with self._tts._pool.connection(timeout=self._conn_options.timeout) as ws:
                init_msg = {
                    "X-API-Key": self._opts.api_key,
                    "voice": self._opts.voice,
                    "format": self._opts.format,
                    "language": "en" if self._opts.language == "auto" else self._opts.language,
                    "sample_rate": self._opts.sample_rate,
                    "model": self._opts.model,
                    "temperature": self._opts.temperature,
                    "top_p": self._opts.top_p,
                }

                await ws.send_str(json.dumps(init_msg))

                tasks = [
                    asyncio.create_task(_input_task(ws)),
                    asyncio.create_task(_sentence_stream_task(ws)),
                    asyncio.create_task(_recv_task(ws)),
                ]
                try:
                    await asyncio.gather(*tasks)
                finally:
                    await utils.aio.gracefully_cancel(*tasks)
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
