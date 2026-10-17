# Copyright 2026 LiveKit, Inc.
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
import binascii
import json
import os
import time
import weakref
from collections.abc import AsyncIterable
from typing import Literal
from urllib.parse import urlsplit

import aiohttp

from livekit import rtc
from livekit.agents import APIConnectionError, APIError, APIStatusError, llm, utils
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)

REALTIME_URL = "wss://relay.addisassistant.com/ws"
INPUT_SAMPLE_RATE = 16_000
OUTPUT_SAMPLE_RATE = 24_000
NUM_CHANNELS = 1


class RealtimeModel(llm.DuplexModel):
    """AddisAI's conversational, interruption-capable realtime audio model."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        url: str = REALTIME_URL,
        http_session: aiohttp.ClientSession | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        resolved_api_key = api_key or os.environ.get("ADDIS_API_KEY")
        if not resolved_api_key:
            raise ValueError("AddisAI API key is required, either as api_key or ADDIS_API_KEY")
        _validate_realtime_url(url)
        super().__init__(
            capabilities=llm.DuplexCapabilities(
                user_transcription=False,
                auto_tool_reply_generation=False,
                mutable_chat_context=False,
                mutable_instructions=False,
                mutable_tools=False,
            )
        )
        self._api_key = resolved_api_key
        self._url = url
        self._http_session = http_session
        self._conn_options = conn_options
        self._sessions = weakref.WeakSet[RealtimeSession]()

    @property
    def model(self) -> str:
        return "አሌፍ-1.2-realtime-audio"

    @property
    def provider(self) -> str:
        return "AddisAI"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self._http_session is None:
            self._http_session = utils.http_context.http_session()
        return self._http_session

    def session(self) -> RealtimeSession:
        session = RealtimeSession(self)
        self._sessions.add(session)
        return session

    async def aclose(self) -> None:
        await asyncio.gather(*(session.aclose() for session in list(self._sessions)))


class RealtimeSession(llm.DuplexSession[Literal["addisai_server_event"]]):
    def __init__(self, model: RealtimeModel) -> None:
        super().__init__(model)
        self._model = model
        self._tools = llm.ToolContext.empty()
        self._input_ch = utils.aio.Chan[rtc.AudioFrame]()
        self._audio_ch = utils.aio.Chan[llm.DuplexAudioFrame]()
        self._input_resampler: rtc.AudioResampler | None = None
        self._output_stream = utils.audio.AudioByteStream(
            OUTPUT_SAMPLE_RATE,
            NUM_CHANNELS,
            samples_per_channel=OUTPUT_SAMPLE_RATE // 10,
        )
        self._main_task = asyncio.create_task(self._run(), name="AddisAIRealtimeSession")

    @property
    def audio_stream(self) -> AsyncIterable[llm.DuplexAudioFrame]:
        return self._audio_ch

    @property
    def tools(self) -> llm.ToolContext:
        return self._tools.copy()

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        if self._closing:
            return
        if self._input_resampler and frame.sample_rate != self._input_resampler._input_rate:
            self._input_resampler = None
        if self._input_resampler is None and (
            frame.sample_rate != INPUT_SAMPLE_RATE or frame.num_channels != NUM_CHANNELS
        ):
            self._input_resampler = rtc.AudioResampler(
                input_rate=frame.sample_rate,
                output_rate=INPUT_SAMPLE_RATE,
                num_channels=NUM_CHANNELS,
            )
        frames = self._input_resampler.push(frame) if self._input_resampler else [frame]
        for output in frames:
            self._input_ch.send_nowait(output)

    async def _run(self) -> None:
        try:
            await self._configured.wait()
            if self._closing:
                return
            started = time.perf_counter()
            ws = await asyncio.wait_for(
                self._model._ensure_session().ws_connect(
                    self._model._url,
                    params={"apiKey": self._model._api_key},
                    heartbeat=30,
                ),
                timeout=self._model._conn_options.timeout,
            )
            self._report_connection_acquired(time.perf_counter() - started)
            ready = asyncio.Event()

            async def send_audio() -> None:
                await ready.wait()
                async for frame in self._input_ch:
                    await ws.send_json(
                        {
                            "data": base64.b64encode(frame.data).decode("ascii"),
                            "mimeType": "audio/pcm;rate=16000",
                        }
                    )

            async def receive_audio() -> None:
                async for message in ws:
                    if message.type != aiohttp.WSMsgType.TEXT:
                        continue
                    try:
                        event = json.loads(message.data)
                    except json.JSONDecodeError as exc:
                        raise APIConnectionError("AddisAI realtime returned invalid JSON") from exc
                    if not isinstance(event, dict):
                        continue
                    self.emit("addisai_server_event", event)
                    if event.get("setupComplete") or (
                        event.get("type") == "status"
                        and "ready" in str(event.get("message") or "").lower()
                    ):
                        ready.set()
                        continue
                    if isinstance(event.get("error"), dict):
                        error = event["error"]
                        raise APIStatusError(
                            str(error.get("message") or "AddisAI realtime returned an error"),
                            status_code=int(error.get("status") or 500),
                            body=event,
                        )
                    content = event.get("serverContent")
                    if not isinstance(content, dict):
                        continue
                    turn = content.get("modelTurn")
                    if isinstance(turn, dict) and isinstance(turn.get("parts"), list):
                        for part in turn["parts"]:
                            inline = part.get("inlineData") if isinstance(part, dict) else None
                            encoded = inline.get("data") if isinstance(inline, dict) else None
                            if not isinstance(encoded, str) or not encoded:
                                continue
                            try:
                                audio = base64.b64decode(encoded, validate=True)
                            except (ValueError, binascii.Error) as exc:
                                raise APIConnectionError(
                                    "AddisAI realtime returned invalid base64 audio"
                                ) from exc
                            for frame in self._output_stream.write(audio):
                                self._audio_ch.send_nowait(llm.DuplexAudioFrame(frame=frame))
                    if content.get("turnComplete"):
                        for frame in self._output_stream.flush():
                            self._audio_ch.send_nowait(llm.DuplexAudioFrame(frame=frame))

            tasks = [asyncio.create_task(send_audio()), asyncio.create_task(receive_audio())]
            try:
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    task.result()
                await utils.aio.gracefully_cancel(*pending)
            finally:
                await ws.close()
        except asyncio.CancelledError:
            raise
        except APIError as exc:
            self._emit_error(exc)
        except (aiohttp.ClientError, asyncio.TimeoutError):
            self._emit_error(APIConnectionError("AddisAI realtime connection failed"))
        finally:
            self._audio_ch.close()

    def _emit_error(self, error: Exception) -> None:
        self.emit(
            "error",
            llm.RealtimeModelError(
                timestamp=time.time(),
                label=self._model.label,
                error=error,
                recoverable=False,
            ),
        )

    async def aclose(self) -> None:
        await super().aclose()
        self._input_ch.close()
        await utils.aio.cancel_and_wait(self._main_task)

    async def _update_instructions(self, instructions: str) -> None:
        # The public relay protocol currently has no instruction update event.
        return

    async def _append_items(self, items: list[llm.ChatItem]) -> None:
        # The public relay protocol currently accepts audio only.
        return

    async def _update_tools(self, tools: list[llm.Tool]) -> None:
        self._tools = llm.ToolContext(tools)
        if tools:
            raise llm.RealtimeError("AddisAI realtime does not expose tool calls")

    def _update_options(
        self, *, tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN
    ) -> None:
        if tool_choice is not NOT_GIVEN and tool_choice is not None:
            raise llm.RealtimeError("AddisAI realtime does not expose tool choice")


def _validate_realtime_url(url: str) -> None:
    parsed = urlsplit(url)
    local = parsed.hostname in ("localhost", "127.0.0.1", "::1")
    if (
        not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or (parsed.scheme != "wss" and not (local and parsed.scheme == "ws"))
    ):
        raise ValueError("url must be a secure WebSocket URL without credentials or query parameters")
