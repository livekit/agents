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
import json
import os
from dataclasses import dataclass, replace
from typing import Any

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
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from .log import logger

SUPPORTED_SAMPLE_RATE = 48000


@dataclass
class _TTSOptions:
    voice: str | None
    voice_id: str | None
    word_tokenizer: tokenize.WordTokenizer
    json_config: dict[str, Any] | None = None


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model_endpoint: str | None = "wss://eu.api.gradium.ai/api/speech/tts",
        model_name: str = "default",
        voice: str | None = None,
        voice_id: str | None = "YTpq7expH9539ERJ",
        json_config: dict[str, Any] | None = None,
        http_session: aiohttp.ClientSession | None = None,
        word_tokenizer: tokenize.WordTokenizer | None = None,
    ) -> None:
        """
        Initialize the Gradium TTS.

        Args:
            api_key (str): Gradium API key, or `GRADIUM_API_KEY` env var.
            model_endpoint (str): Gradium model endpoint, or `GRADIUM_MODEL_ENDPOINT` env var.
            model_name (str): Model name.
            voice (str): Speaker voice.
            voice_id (str): Speaker voice ID.
            word_tokenizer (tokenize.WordTokenizer): Tokenizer for processing text. Defaults to basic WordTokenizer.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=SUPPORTED_SAMPLE_RATE,
            num_channels=1,
        )

        api_key = api_key or os.environ.get("GRADIUM_API_KEY")

        if not api_key:
            raise ValueError(
                "Gradium API key is required. "
                "Pass one in via the `api_key` parameter, "
                "or set it as the `GRADIUM_API_KEY` environment variable"
            )

        model_endpoint = model_endpoint or os.environ.get("GRADIUM_MODEL_ENDPOINT")

        if not model_endpoint:
            raise ValueError(
                "The model endpoint is required, you can find it in the Gradium dashboard"
            )

        self._api_key = api_key
        self._model_endpoint = model_endpoint
        self._model_name = model_name

        if not word_tokenizer:
            word_tokenizer = tokenize.basic.WordTokenizer(ignore_punctuation=False)
        self._opts = _TTSOptions(
            voice=voice,
            voice_id=voice_id,
            word_tokenizer=word_tokenizer,
            json_config=json_config,
        )
        self._session = http_session

    @property
    def model(self) -> str:
        return "unknown"

    @property
    def provider(self) -> str:
        return "Gradium"

    async def _connect_ws(self, timeout: float) -> aiohttp.ClientWebSocketResponse:
        return await asyncio.wait_for(
            self._ensure_session().ws_connect(
                self._model_endpoint,
                headers={"x-api-key": self._api_key, "x-api-source": "livekit"},
            ),
            timeout,
        )

    async def _close_ws(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        await ws.close()

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

        return self._session

    def update_options(
        self,
        *,
        voice: NotGivenOr[str] = NOT_GIVEN,
        json_config: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> None:
        if is_given(voice):
            self._opts.voice = voice
        if is_given(json_config):
            self._opts.json_config = json_config

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        return SynthesizeStream(tts=self, conn_options=conn_options)

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        return ChunkedStream(
            tts=self,
            api_key=self._api_key,
            input_text=text,
            model_endpoint=self._model_endpoint,
            model_name=self._model_name,
            conn_options=conn_options,
        )


class ChunkedStream(tts.ChunkedStream):
    def __init__(
        self,
        *,
        tts: TTS,
        api_key: str,
        model_endpoint: str,
        model_name: str,
        input_text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(
            tts=tts,
            input_text=input_text,
            conn_options=conn_options,
        )

        self._tts: TTS = tts
        self._api_key = api_key
        self._model_endpoint = model_endpoint
        self._model_name = model_name
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        # TODO(laurent): once we support the POST requests, we should use it here rather than the websocket API.
        try:
            async with self._tts._ensure_session().ws_connect(
                self._tts._model_endpoint,
                headers={"x-api-key": self._tts._api_key},
                timeout=aiohttp.ClientWSTimeout(ws_receive=self._conn_options.timeout, ws_close=10),
            ) as ws:
                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=SUPPORTED_SAMPLE_RATE,
                    num_channels=1,
                    mime_type="audio/pcm",
                )

                setup_msg: dict[str, Any] = {
                    "type": "setup",
                    "model_name": self._model_name,
                    "output_format": "pcm",
                }
                if self._opts.voice is not None:
                    setup_msg["voice"] = self._opts.voice
                if self._opts.voice_id is not None:
                    setup_msg["voice_id"] = self._opts.voice_id
                if self._opts.json_config is not None:
                    setup_msg["json_config"] = json.dumps(self._opts.json_config)
                await ws.send_str(json.dumps(setup_msg))
                text_msg = {"type": "text", "text": self._input_text}
                await ws.send_str(json.dumps(text_msg))
                flush_msg = {"type": "end_of_stream"}
                await ws.send_str(json.dumps(flush_msg))

                while True:
                    msg = await ws.receive()
                    if msg.type in (
                        aiohttp.WSMsgType.CLOSE,
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.CLOSING,
                    ):
                        # TODO(laurent): once we support returning eos in the api, we should enable this back.
                        # raise APIStatusError("Gradium websocket connection closed unexpectedly")
                        break

                    if msg.type == aiohttp.WSMsgType.TEXT:
                        msg_data = json.loads(msg.data)
                        type_ = msg_data.get("type")
                        if type_ == "text":
                            # TODO(laurent): handle text timestamps?
                            pass
                        elif type_ == "ready":
                            pass
                        elif type_ == "audio":
                            audio = base64.b64decode(msg_data["audio"])
                            output_emitter.push(audio)
                        elif type_ == "end_of_stream":
                            break
                        else:
                            logger.warning(f"unknown message type: {type_}")
                output_emitter.flush()

        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=None, body=None
            ) from None
        except Exception as e:
            raise APIConnectionError() from e


class SynthesizeStream(tts.SynthesizeStream):
    """Stream-based text-to-speech synthesis using Gradium WebSocket API."""

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions):
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        segments_ch = utils.aio.Chan[tokenize.WordStream]()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=SUPPORTED_SAMPLE_RATE,
            num_channels=1,
            mime_type="audio/pcm",
            stream=True,
        )

        async def _tokenize_input() -> None:
            # Converts incoming text into WordStreams and sends them into _segments_ch
            word_stream = None
            async for input in self._input_ch:
                if isinstance(input, str):
                    if word_stream is None:
                        word_stream = self._opts.word_tokenizer.stream()
                        segments_ch.send_nowait(word_stream)
                    word_stream.push_text(input)
                elif isinstance(input, self._FlushSentinel):
                    if word_stream:
                        word_stream.end_input()
                    word_stream = None

            segments_ch.close()

        async def _run_segments() -> None:
            async for word_stream in segments_ch:
                await self._run_ws(word_stream, output_emitter)

        tasks = [
            asyncio.create_task(_tokenize_input()),
            asyncio.create_task(_run_segments()),
        ]
        try:
            await asyncio.gather(*tasks)
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=request_id, body=None
            ) from None
        except Exception as e:
            raise APIConnectionError() from e
        finally:
            await utils.aio.gracefully_cancel(*tasks)

    async def _run_ws(
        self, word_stream: tokenize.WordStream, output_emitter: tts.AudioEmitter
    ) -> None:
        segment_id = utils.shortuuid()
        output_emitter.start_segment(segment_id=segment_id)

        async def send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            setup_msg: dict[str, Any] = {
                "type": "setup",
                "model_name": self._tts._model_name,
                "output_format": "pcm",
            }
            if self._opts.voice is not None:
                setup_msg["voice"] = self._opts.voice
            if self._opts.voice_id is not None:
                setup_msg["voice_id"] = self._opts.voice_id
            if self._opts.json_config is not None:
                setup_msg["json_config"] = json.dumps(self._opts.json_config)

            await ws.send_str(json.dumps(setup_msg))

            async for word in word_stream:
                text_msg = {"type": "text", "text": f"{word.token} "}
                self._mark_started()
                await ws.send_str(json.dumps(text_msg))

            flush_msg = {"type": "end_of_stream"}
            await ws.send_str(json.dumps(flush_msg))

        async def recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            while True:
                msg = await ws.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    # TODO(laurent): once we support returning eos in the api, we should enable this back.
                    # raise APIStatusError("Gradium websocket connection closed unexpectedly")
                    output_emitter.end_segment()
                    break

                if msg.type == aiohttp.WSMsgType.TEXT:
                    msg_data = json.loads(msg.data)
                    type_ = msg_data.get("type")
                    if type_ == "text":
                        # TODO(laurent): handle text timestamps?
                        pass
                    elif type_ == "ready":
                        pass
                    elif type_ == "audio":
                        audio = base64.b64decode(msg_data["audio"])
                        output_emitter.push(audio)
                    elif type_ == "end_of_stream":
                        output_emitter.end_segment()
                        break
                    else:
                        logger.warning(f"unknown message type: {type_}")

        async with self._tts._ensure_session().ws_connect(
            self._tts._model_endpoint,
            headers={"x-api-key": self._tts._api_key},
            timeout=aiohttp.ClientWSTimeout(ws_receive=self._conn_options.timeout, ws_close=10),
        ) as ws:
            tasks = [
                asyncio.create_task(send_task(ws)),
                asyncio.create_task(recv_task(ws)),
            ]

            try:
                await asyncio.gather(*tasks)
            finally:
                await utils.aio.gracefully_cancel(*tasks)
