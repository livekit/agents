# Copyright 2025 LiveKit, Inc.
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
import weakref
from collections.abc import Callable
from dataclasses import dataclass, replace

from pydantic import BaseModel, ConfigDict, Field

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

from .log import logger
from .models import TTSModels

RESEMBLE_WEBSOCKET_URL = "wss://websocket.cluster.resemble.ai/stream"
RESEMBLE_REST_API_URL = "https://f.cluster.resemble.ai/synthesize"
DEFAULT_VOICE_UUID = "55592656"


class _SynthesisResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    success: bool = False
    issues: list[str] = Field(default_factory=lambda: ["Unknown error"])
    audio_content: str | None = None


class _WebSocketMessage(BaseModel):
    type: str
    audio_content: str | None = None
    request_id: int | str | None = None


@dataclass
class _TTSOptions:
    voice_uuid: str
    sample_rate: int
    model: str | None


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        voice_uuid: str | None = None,
        model: TTSModels | str | None = None,
        sample_rate: int = 44100,
        http_session: http.HttpSession | None = None,
        use_streaming: bool = True,
    ) -> None:
        """
        Create a new instance of the Resemble TTS.

        See https://docs.app.resemble.ai/docs/text_to_speech/ for more documentation on all of these options.

        Args:
            voice_uuid (str, optional): The voice UUID for the desired voice. Defaults to None.
            model (TTSModels | str | None, optional): The model to use for synthesis. Can be "chatterbox" or "chatterbox-turbo". Defaults to None.
            sample_rate (int, optional): The audio sample rate in Hz. Defaults to 44100.
            api_key (str | None, optional): The Resemble API key. If not provided, it will be read from the RESEMBLE_API_KEY environment variable.
            http_session (http.HttpSession | None, optional): An existing HttpSession to use. If not provided, a new session will be created.
            use_streaming (bool, optional): Whether to use streaming or not. Defaults to True.
        """  # noqa: E501
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=use_streaming),
            sample_rate=sample_rate,
            num_channels=1,
        )

        api_key = api_key or os.environ.get("RESEMBLE_API_KEY")
        if not api_key:
            raise ValueError(
                "Resemble API key is required, either as argument or set RESEMBLE_API_KEY"
                " environment variable"
            )
        self._api_key = api_key

        if voice_uuid is None:
            voice_uuid = DEFAULT_VOICE_UUID

        self._opts = _TTSOptions(
            voice_uuid=voice_uuid,
            sample_rate=sample_rate,
            model=model,
        )

        self._session = http_session
        self._streams = weakref.WeakSet[SynthesizeStream]()
        self._pool = utils.ConnectionPool[http.WebSocket](
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
        )

    @property
    def model(self) -> str:
        return self._opts.model or "unknown"

    @property
    def provider(self) -> str:
        return "Resemble"

    async def _connect_ws(self, conn_options: APIConnectOptions) -> http.WebSocket:
        return await self._ensure_session().ws_connect(
            RESEMBLE_WEBSOCKET_URL,
            headers={"Authorization": f"Bearer {self._api_key}"},
            conn_options=conn_options,
        )

    async def _close_ws(self, ws: http.WebSocket) -> None:
        await ws.close()

    def _ensure_session(self) -> http.HttpSession:
        if not self._session:
            self._session = http.http_session()

        return self._session

    def prewarm(self) -> None:
        self._pool.prewarm()

    def update_options(
        self,
        *,
        voice_uuid: str | None = None,
        model: TTSModels | str | None = None,
        sentence_tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
        pacing: NotGivenOr[tts.SentenceStreamPacer | None] = NOT_GIVEN,
    ) -> None:
        """
        Update the Text-to-Speech (TTS) configuration options.

        Args:
            voice_uuid (str, optional): The voice UUID for the desired voice.
            model (TTSModels | str | None, optional): The model to use for synthesis.
            sentence_tokenizer (SentenceTokenizer, optional): What splits the text into the chunks
                the TTS is sent.
            pacing (SentenceStreamPacer | None, optional): A ``SentenceStreamPacer``, or ``None`` to
                send each chunk as soon as it is split.
        """  # noqa: E501
        super().update_options(sentence_tokenizer=sentence_tokenizer, pacing=pacing)
        self._opts.voice_uuid = voice_uuid or self._opts.voice_uuid
        self._opts.model = model if model is not None else self._opts.model

    def _synthesize(
        self, chunk: tts.SpeechChunk, *, conn_options: APIConnectOptions
    ) -> ChunkedStream:
        return ChunkedStream(
            tts=self,
            opts=replace(self._opts),
            http_session=self._ensure_session,
            api_key=self._api_key,
            input_text=chunk.text,
            conn_options=conn_options,
        )

    def _stream(self, *, conn_options: APIConnectOptions) -> SynthesizeStream:
        stream = SynthesizeStream(
            tts=self, opts=replace(self._opts), pool=self._pool, conn_options=conn_options
        )
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()

        self._streams.clear()
        await self._pool.aclose()


class ChunkedStream(tts.ChunkedStream):
    """Synthesize text into speech in one go using Resemble AI's REST API."""

    def __init__(
        self,
        *,
        tts: TTS,
        opts: _TTSOptions,
        http_session: Callable[[], http.HttpSession],
        api_key: str,
        input_text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._opts = opts
        self._http_session = http_session
        self._api_key = api_key

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        try:
            payload: dict[str, object] = {
                "voice_uuid": self._opts.voice_uuid,
                "data": self._input_text,
                "sample_rate": self._opts.sample_rate,
                "precision": "PCM_16",
            }
            if self._opts.model is not None:
                payload["model"] = self._opts.model

            async with self._http_session().post(
                RESEMBLE_REST_API_URL,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                json=payload,
                conn_options=self._conn_options,
            ) as resp:
                await resp.raise_for_status()
                response = _SynthesisResponse.model_validate(await resp.json())

                if not response.success:
                    raise APIError(
                        "Resemble API returned failure",
                        body=response.model_dump_json(exclude_unset=True),
                        detail="; ".join(response.issues),
                    )

                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=self._opts.sample_rate,
                    num_channels=1,
                    mime_type="audio/wav",
                )

                if response.audio_content is None:
                    raise APIConnectionError("Resemble response omitted audio content")
                audio_bytes = base64.b64decode(response.audio_content)

                output_emitter.push(audio_bytes)
                output_emitter.flush()

        except APIError:
            raise
        except Exception as e:
            raise APIConnectionError() from e


class SynthesizeStream(tts.SynthesizeStream):
    """Stream-based text-to-speech synthesis using Resemble AI WebSocket API.


    This implementation connects to Resemble's WebSocket API for real-time streaming
    synthesis. Note that this requires a Business plan subscription with Resemble AI.
    """

    def __init__(
        self,
        *,
        tts: TTS,
        opts: _TTSOptions,
        pool: utils.ConnectionPool[http.WebSocket],
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._opts = opts
        self._pool = pool

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        segments_ch = utils.aio.Chan[utils.aio.Chan[str]]()
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            stream=True,
            mime_type="audio/mp3",
        )

        async def _split_segments() -> None:
            segment: utils.aio.Chan[str] | None = None
            async for item in self.input:
                if isinstance(item, tts.SpeechChunk):
                    if segment is None:
                        segment = utils.aio.Chan[str]()
                        segments_ch.send_nowait(segment)
                    segment.send_nowait(item.text)
                elif segment is not None:
                    segment.close()
                    segment = None

            if segment is not None:
                segment.close()

            segments_ch.close()

        async def _process_segments() -> None:
            async for segment in segments_ch:
                await self._run_ws(segment, output_emitter)

        try:
            async with utils.aio.TaskSet() as tasks:
                await asyncio.gather(
                    tasks.spawn(_split_segments()), tasks.spawn(_process_segments())
                )
        except APIError:
            raise
        except Exception as e:
            raise APIConnectionError() from e

    async def _run_ws(self, segment: utils.aio.Chan[str], output_emitter: tts.AudioEmitter) -> None:
        segment_id = utils.shortuuid()
        output_emitter.start_segment(segment_id=segment_id)

        last_index = 0
        input_ended = False

        read_deadline = self._core_attempts

        async def _send_task(ws: http.WebSocket) -> None:
            nonlocal input_ended, last_index
            async for text in segment:
                last_index += 1
                payload: dict[str, object] = {
                    "voice_uuid": self._opts.voice_uuid,
                    "data": text,
                    "request_id": last_index,
                    "sample_rate": self._opts.sample_rate,
                    "precision": "PCM_16",
                    "output_format": "mp3",
                }
                if self._opts.model is not None:
                    payload["model"] = self._opts.model
                self.mark_started()
                await ws.send_str(json.dumps(payload))
                read_deadline.expect_response()

            input_ended = True

        async def _recv_task(ws: http.WebSocket) -> None:
            while True:
                msg = await ws.receive()
                if msg.type == http.WebSocketMessageType.CLOSE:
                    raise ws.close_error("Resemble connection closed unexpectedly")

                if msg.type != http.WebSocketMessageType.TEXT:
                    logger.warning("Unexpected Resemble message type %s", msg.type)
                    continue

                data = _WebSocketMessage.model_validate_json(msg.data)
                if data.type == "audio":
                    if data.audio_content:
                        b64data = base64.b64decode(data.audio_content)
                        read_deadline.progress()
                        output_emitter.push(b64data)

                elif data.type == "audio_end":
                    index = data.request_id
                    if index is None:
                        raise APIConnectionError("Resemble completion omitted request identity")
                    if index == last_index:
                        read_deadline.complete_response()
                    if index == last_index and input_ended:
                        output_emitter.end_segment()
                        break
                else:
                    logger.error(
                        "Unexpected Resemble message",
                        extra={"lk.pii.data": data.model_dump(mode="json")},
                    )

        async with (
            self._pool.connection(conn_options=self._conn_options) as ws,
            utils.aio.TaskSet() as tasks,
        ):
            await asyncio.gather(tasks.spawn(_send_task(ws)), tasks.spawn(_recv_task(ws)))
