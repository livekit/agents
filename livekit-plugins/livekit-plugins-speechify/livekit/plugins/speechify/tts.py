# Copyright 2024 LiveKit, Inc.
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
from dataclasses import asdict, dataclass, field
from typing import Literal
from urllib.parse import urljoin

import aiohttp
from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectOptions,
    tokenize,
    tts,
    utils,
)

from .log import logger
from .models import (
    TTSDefaultVoiceId,
    TTSEncoding,
    TTSModel,
    TTSVoice,
)

API_AUTH_HEADER = "Authorization"

NUM_CHANNELS = 1
BUFFERED_WORDS_COUNT = 8


@dataclass
class _TTSOptions:
    model: TTSModel | str
    encoding: TTSEncoding
    sample_rate: int
    voice: TTSVoice
    endpoint: str
    api_key: str
    language: str


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        model: TTSModel | str = "simba-english",
        language: str = "en",
        encoding: TTSEncoding = "raw",
        voice: TTSVoice = TTSDefaultVoiceId,
        sample_rate: int = 24000,
        endpoint: str | None = None,
        api_key: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """
        Create a new instance of Speechify TTS.

        Args:
            model (TTSModels, optional): The Speechify TTS model to use. Defaults to "simba-english".
            language (str, optional): The language code for synthesis. Defaults to "en".
            encoding (TTSEncoding, optional): The audio encoding format. Defaults to "opus".
            voice (str, optional): The voice ID.
            sample_rate (int, optional): The audio sample rate in Hz. Defaults to 24000.
            endpoint (str, optional): The Speechify API endpoint. If not provided, it will be read from the SPEECHIFY_ENDPOINT environment variable.
            api_key (str, optional): The Speechify API key. If not provided, it will be read from the SPEECHIFY_API_KEY environment variable.
            http_session (aiohttp.ClientSession | None, optional): An existing aiohttp ClientSession to use. If not provided, a new session will be created.
        """

        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )

        api_key = api_key or os.environ.get("SPEECHIFY_API_KEY")
        if not api_key:
            raise ValueError("SPEECHIFY_API_KEY must be set")
        endpoint = endpoint or os.environ.get("SPEECHIFY_ENDPOINT")
        if not endpoint:
            raise ValueError("SPEECHIFY_ENDPOINT must be set")

        self._opts = _TTSOptions(
            model=model,
            language=language,
            encoding=encoding,
            sample_rate=sample_rate,
            voice=voice,
            endpoint=endpoint,
            api_key=api_key,
        )
        self._session = http_session

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

        return self._session

    def update_options(
        self,
        *,
        model: TTSModel | None = None,
        language: str | None = None,
        voice: TTSVoice | None = None,
    ) -> None:
        """
        Update the Text-to-Speech (TTS) configuration options.

        This method allows updating the TTS settings, including model type, language, voice.
        If any parameter is not provided, the existing value will be retained.

        Args:
            model (TTSModel, optional): The Speechify TTS model to use. Defaults to "simba-english".
            language (str, optional): The language code for synthesis. Defaults to "en".
            voice (TTSVoice, optional): The voice ID.
        """
        self._opts.model = model or self._opts.model
        self._opts.language = language or self._opts.language
        self._opts.voice = voice or self._opts.voice

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.ChunkedStream:
        return ChunkedStreamWrapper(
            tts=self,
            input_text=text,
            conn_options=conn_options,
        )

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> "SynthesizeStream":
        return SynthesizeStream(
            tts=self,
            conn_options=conn_options,
            opts=self._opts,
            session=self._ensure_session(),
        )


class SynthesizeStream(tts.SynthesizeStream):
    def __init__(
        self,
        *,
        tts: TTS,
        conn_options: APIConnectOptions,
        opts: _TTSOptions,
        session: aiohttp.ClientSession,
    ):
        super().__init__(tts=tts, conn_options=conn_options)
        self._opts, self._session = opts, session
        self._sent_tokenizer_stream = tokenize.basic.SentenceTokenizer(
            min_sentence_len=BUFFERED_WORDS_COUNT
        ).stream()

    async def _run(self) -> None:
        request_id = utils.shortuuid()
        decoder: utils.codecs.AudioStreamDecoder | None = None
        if self._opts.encoding == "opus":
            decoder = utils.codecs.AudioStreamDecoder()

        is_finished = False

        async def _sentence_stream_task(ws: aiohttp.ClientWebSocketResponse):
            nonlocal is_finished
            async for ev in self._sent_tokenizer_stream:
                payload = _Payload(
                    format=self._opts.encoding,
                    items=[
                        _TextItem(
                            text=ev.token,
                            speaker=self._opts.voice,
                        )
                    ],
                )
                await ws.send_str(json.dumps(asdict(payload)))
            is_finished = True

        async def _input_task():
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    self._sent_tokenizer_stream.flush()
                    continue
                self._sent_tokenizer_stream.push_text(data)
            self._sent_tokenizer_stream.end_input()

        def _send_frame(frame: rtc.AudioFrame | None, is_final: bool) -> None:
            if frame is not None:
                self._event_ch.send_nowait(
                    tts.SynthesizedAudio(
                        request_id=request_id,
                        frame=frame,
                        is_final=is_final,
                    )
                )

        async def _recv_task(ws: aiohttp.ClientWebSocketResponse):
            audio_bstream = utils.audio.AudioByteStream(
                sample_rate=self._opts.sample_rate,
                num_channels=NUM_CHANNELS,
            )

            try:
                while True:
                    msg = await ws.receive()
                    if msg.type in (
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.CLOSE,
                        aiohttp.WSMsgType.CLOSING,
                    ):
                        if is_finished:
                            break
                        raise Exception("Speechify connection closed unexpectedly")

                    if msg.type != aiohttp.WSMsgType.BINARY:
                        logger.warning(
                            "unexpected Speechify message type %s, %s",
                            msg.type,
                            msg.data,
                        )
                        continue

                    if decoder:
                        decoder.push(msg.data)
                    else:
                        for frame in audio_bstream.write(msg.data):
                            _send_frame(frame, is_final=False)
            finally:
                if decoder:
                    decoder.end_input()

        async def _consume_decoder_task(decoder: utils.codecs.AudioStreamDecoder):
            async for frame in decoder:
                # logger.info(f"received decoder frame: {frame}")
                _send_frame(frame, is_final=False)
            await decoder.aclose()

        url = urljoin(self._opts.endpoint, "/ws/generate-audio-stream")
        headers = {
            API_AUTH_HEADER: self._opts.api_key,
        }

        ws: aiohttp.ClientWebSocketResponse | None = None

        try:
            ws = await asyncio.wait_for(
                self._session.ws_connect(url, headers=headers),
                self._conn_options.timeout,
            )

            tasks = [
                asyncio.create_task(_input_task()),
                asyncio.create_task(_sentence_stream_task(ws)),
                asyncio.create_task(_recv_task(ws)),
            ]
            if decoder:
                tasks.append(asyncio.create_task(_consume_decoder_task(decoder)))

            try:
                await asyncio.gather(*tasks)
            finally:
                await utils.aio.gracefully_cancel(*tasks)
        finally:
            if ws is not None:
                await ws.close()


class ChunkedStreamWrapper(tts.ChunkedStream):
    def __init__(
        self,
        *,
        tts: TTS,
        input_text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(
            tts=tts,
            input_text=input_text,
            conn_options=conn_options,
        )
        self._stream: SynthesizeStream | None = None

    async def _run(self) -> None:
        # Create the underlying stream
        self._stream = self._tts.stream(conn_options=self._conn_options)

        try:
            # Push all text at once and end input
            self._stream.push_text(self._input_text)
            self._stream.end_input()

            # Forward all events from the stream
            async for event in self._stream:
                self._event_ch.send_nowait(event)
        finally:
            if self._stream is not None:
                await self._stream.aclose()


@dataclass
class _TextItem:
    text: str
    type: Literal["text"] = "text"
    normalize_text: bool = False
    """Expands dates, acronyms, abbreviations, units of measurements (e.g. 100km), and currencies. Costs 22.5ms of latency."""
    speaker: TTSVoice = TTSDefaultVoiceId
    # emotion


@dataclass
class _Payload:
    format: TTSEncoding
    items: list[_TextItem] = field(default_factory=list)
