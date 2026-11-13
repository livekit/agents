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
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Any

from pydantic import JsonValue

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    http,
    tokenize,
    tts,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from .log import logger

SUPPORTED_SAMPLE_RATE = 48000


def _decode(raw: str | bytes) -> dict[str, Any]:
    """A provider message, which must be a JSON object."""
    try:
        data = json.loads(raw)
    except ValueError:
        data = None
    if not isinstance(data, dict):
        raise APIError("Gradium TTS sent a malformed message", kind="protocol", detail=str(raw))
    return data


@dataclass
class _TTSOptions:
    voice: str | None
    voice_id: str | None
    pronunciation_id: str | None
    json_config: dict[str, JsonValue] | None = None


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model_endpoint: str | None = None,
        model_name: str = "default",
        voice: str | None = None,
        voice_id: str | None = "4SZHfMpw-p46Ywgs",
        pronunciation_id: str | None = None,
        json_config: dict[str, JsonValue] | None = None,
        http_session: http.HttpSession | None = None,
    ) -> None:
        """
        Initialize the Gradium TTS.

        Args:
            api_key (str): Gradium API key, or `GRADIUM_API_KEY` env var.
            model_endpoint (str): Gradium model endpoint, or `GRADIUM_MODEL_ENDPOINT` env var.
            model_name (str): Model name.
            voice (str): Speaker voice.
            voice_id (str): Speaker voice ID.
            pronunciation_id (str): Optional pronunciation ID for controlling TTS pronunciation.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True, text_unit="words"),
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

        model_endpoint = (
            model_endpoint
            or os.environ.get("GRADIUM_MODEL_ENDPOINT")
            or "wss://api.gradium.ai/api/speech/tts"
        )

        self._api_key = api_key
        self._model_endpoint = model_endpoint
        self._model_name = model_name

        self._opts = _TTSOptions(
            voice=voice,
            voice_id=voice_id,
            pronunciation_id=pronunciation_id,
            json_config=json_config,
        )
        self._session = http_session

    @property
    def model(self) -> str:
        return "unknown"

    @property
    def provider(self) -> str:
        return "Gradium"

    async def _connect_ws(self, conn_options: APIConnectOptions) -> http.WebSocket:
        return await self._ensure_session().ws_connect(
            self._model_endpoint,
            headers={"x-api-key": self._api_key, "x-api-source": "livekit"},
            conn_options=conn_options,
        )

    async def _close_ws(self, ws: http.WebSocket) -> None:
        await ws.close()

    def _ensure_session(self) -> http.HttpSession:
        if not self._session:
            self._session = http.http_session()

        return self._session

    def update_options(
        self,
        *,
        voice: NotGivenOr[str] = NOT_GIVEN,
        voice_id: NotGivenOr[str] = NOT_GIVEN,
        json_config: NotGivenOr[dict[str, JsonValue]] = NOT_GIVEN,
        sentence_tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
        pacing: NotGivenOr[tts.SentenceStreamPacer | None] = NOT_GIVEN,
    ) -> None:
        super().update_options(sentence_tokenizer=sentence_tokenizer, pacing=pacing)
        if is_given(voice):
            self._opts.voice = voice
        if is_given(voice_id):
            self._opts.voice_id = voice_id
        if is_given(json_config):
            self._opts.json_config = json_config

    def _stream(self, *, conn_options: APIConnectOptions) -> SynthesizeStream:
        return SynthesizeStream(
            connect_ws=self._connect_ws,
            model_name=self._model_name,
            opts=replace(self._opts),
            tts=self,
            conn_options=conn_options,
        )

    def _synthesize(
        self, chunk: tts.SpeechChunk, *, conn_options: APIConnectOptions
    ) -> ChunkedStream:
        return ChunkedStream(
            connect_ws=self._connect_ws,
            opts=replace(self._opts),
            tts=self,
            api_key=self._api_key,
            input_text=chunk.text,
            model_endpoint=self._model_endpoint,
            model_name=self._model_name,
            conn_options=conn_options,
        )


class ChunkedStream(tts.ChunkedStream):
    def __init__(
        self,
        *,
        connect_ws: Callable[[APIConnectOptions], Awaitable[http.WebSocket]],
        opts: _TTSOptions,
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
        self._opts = opts
        self._connect_ws = connect_ws

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        # TODO(laurent): once we support the POST requests, we should use it here rather than the websocket API.
        try:
            ws = await self._connect_ws(self._conn_options)
            try:
                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=SUPPORTED_SAMPLE_RATE,
                    num_channels=1,
                    mime_type="audio/pcm",
                )

                setup_msg: dict[str, JsonValue] = {
                    "type": "setup",
                    "model_name": self._model_name,
                    "output_format": "pcm",
                }
                if self._opts.voice is not None:
                    setup_msg["voice"] = self._opts.voice
                if self._opts.voice_id is not None:
                    setup_msg["voice_id"] = self._opts.voice_id
                if self._opts.pronunciation_id is not None:
                    setup_msg["pronunciation_id"] = self._opts.pronunciation_id
                if self._opts.json_config is not None:
                    setup_msg["json_config"] = json.dumps(self._opts.json_config)
                await ws.send_str(json.dumps(setup_msg))
                text_msg = {"type": "text", "text": self._input_text}
                await ws.send_str(json.dumps(text_msg))
                flush_msg = {"type": "end_of_stream"}
                await ws.send_str(json.dumps(flush_msg))

                while True:
                    msg = await ws.receive(timeout=self._conn_options.read_timeout)
                    if msg.type is http.WebSocketMessageType.CLOSE:
                        # TODO(laurent): once we support returning eos in the api, we should enable this back.
                        # raise APIStatusError("Gradium websocket connection closed unexpectedly")
                        break

                    if msg.type is http.WebSocketMessageType.TEXT:
                        msg_data = _decode(msg.data)
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
            finally:
                await ws.close()

        except APIError:
            raise
        except Exception as e:
            raise APIConnectionError() from e


class SynthesizeStream(tts.SynthesizeStream):
    """Stream-based text-to-speech synthesis using Gradium WebSocket API."""

    def __init__(
        self,
        *,
        connect_ws: Callable[[APIConnectOptions], Awaitable[http.WebSocket]],
        model_name: str,
        opts: _TTSOptions,
        tts: TTS,
        conn_options: APIConnectOptions,
    ):
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = opts
        self._connect_ws = connect_ws
        self._model_name = model_name

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        segments_ch = utils.aio.Chan[utils.aio.Chan[str]]()
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=SUPPORTED_SAMPLE_RATE,
            num_channels=1,
            mime_type="audio/pcm",
            stream=True,
        )

        async def _segment_input() -> None:
            segment: utils.aio.Chan[str] | None = None
            async for input in self.input:
                if isinstance(input, tts.SpeechChunk):
                    if segment is None:
                        segment = utils.aio.Chan[str]()
                        segments_ch.send_nowait(segment)
                    segment.send_nowait(input.text)
                elif segment is not None:
                    segment.close()
                    segment = None
            if segment is not None:
                segment.close()
            segments_ch.close()

        async def _run_segments() -> None:
            async for segment in segments_ch:
                await self._run_ws(segment, output_emitter)

        try:
            async with utils.aio.TaskSet() as tasks:
                await asyncio.gather(tasks.spawn(_segment_input()), tasks.spawn(_run_segments()))
        except APIError:
            raise
        except Exception as e:
            raise APIConnectionError() from e

    async def _run_ws(self, segment: utils.aio.Chan[str], output_emitter: tts.AudioEmitter) -> None:
        segment_id = utils.shortuuid()
        output_emitter.start_segment(segment_id=segment_id)

        read_deadline = self._core_attempts

        async def send_task(ws: http.WebSocket) -> None:
            setup_msg: dict[str, JsonValue] = {
                "type": "setup",
                "model_name": self._model_name,
                "output_format": "pcm",
            }
            if self._opts.voice is not None:
                setup_msg["voice"] = self._opts.voice
            if self._opts.voice_id is not None:
                setup_msg["voice_id"] = self._opts.voice_id
            if self._opts.pronunciation_id is not None:
                setup_msg["pronunciation_id"] = self._opts.pronunciation_id
            if self._opts.json_config is not None:
                setup_msg["json_config"] = json.dumps(self._opts.json_config)

            await ws.send_str(json.dumps(setup_msg))

            async for chunk in segment:
                text_msg = {"type": "text", "text": f"{chunk} "}
                self.mark_started()
                await ws.send_str(json.dumps(text_msg))
                read_deadline.expect_response()

            flush_msg = {"type": "end_of_stream"}
            await ws.send_str(json.dumps(flush_msg))
            read_deadline.expect_response(final=True)

        async def recv_task(ws: http.WebSocket) -> None:
            while True:
                msg = await ws.receive()
                if msg.type is http.WebSocketMessageType.CLOSE:
                    # TODO(laurent): once we support returning eos in the api, we should enable this back.
                    # raise APIStatusError("Gradium websocket connection closed unexpectedly")
                    output_emitter.end_segment()
                    break

                if msg.type is http.WebSocketMessageType.TEXT:
                    msg_data = _decode(msg.data)
                    type_ = msg_data.get("type")
                    if type_ == "text":
                        # TODO(laurent): handle text timestamps?
                        pass
                    elif type_ == "ready":
                        pass
                    elif type_ == "audio":
                        audio = base64.b64decode(msg_data["audio"])
                        read_deadline.audio_received()
                        output_emitter.push(audio)
                    elif type_ == "end_of_stream":
                        output_emitter.end_segment()
                        break
                    else:
                        logger.warning(f"unknown message type: {type_}")

        ws = await self._connect_ws(self._conn_options)
        try:
            async with utils.aio.TaskSet() as tasks:
                await asyncio.gather(tasks.spawn(send_task(ws)), tasks.spawn(recv_task(ws)))
        finally:
            await ws.close()
