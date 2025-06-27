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
import dataclasses
import json
import os
import weakref
from dataclasses import dataclass
from typing import Callable
from urllib.parse import urlencode

import aiohttp

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectOptions,
    APIStatusError,
    stt,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, is_given

from .log import logger


class _PeriodicCollector:
    def __init__(self, duration: float, callback: Callable[[float], None]):
        self._duration = duration
        self._callback = callback
        self._collected_value = 0.0
        self._task: asyncio.Task | None = None
        self._lock = asyncio.Lock()

    async def push(self, value: float) -> None:
        async with self._lock:
            self._collected_value += value
            if not self._task:
                self._task = asyncio.create_task(self._run())

    async def flush(self) -> None:
        async with self._lock:
            if self._task:
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
                self._task = None

            if self._collected_value > 0:
                self._callback(self._collected_value)
                self._collected_value = 0.0

    async def _run(self) -> None:
        await asyncio.sleep(self._duration)
        async with self._lock:
            self._callback(self._collected_value)
            self._collected_value = 0.0
            self._task = None


@dataclass
class STTOptions:
    model: str
    sample_rate: int
    language: NotGivenOr[str] = NOT_GIVEN
    prompt: NotGivenOr[str] = NOT_GIVEN
    temperature: NotGivenOr[float] = NOT_GIVEN
    response_format: str = "verbose_json"
    timestamp_granularities: NotGivenOr[list[str]] = NOT_GIVEN


class STT(stt.STT):
    def __init__(
        self,
        *,
        model: str = "fireworks-asr-large",
        api_key: NotGivenOr[str] = NOT_GIVEN,
        sample_rate: int = 16000,
        language: NotGivenOr[str] = NOT_GIVEN,
        prompt: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        timestamp_granularities: NotGivenOr[list[str]] = NOT_GIVEN,
        response_format: str = "verbose_json",
        http_session: aiohttp.ClientSession | None = None,
    ):
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=True, interim_results=True),
        )
        fireworks_api_key = api_key if is_given(api_key) else os.environ.get("FIREWORKS_API_KEY")
        if fireworks_api_key is None:
            raise ValueError(
                "Fireworks API key is required. "
                "Pass one in via the `api_key` parameter, "
                "or set it as the `FIREWORKS_API_KEY` environment variable"
            )
        self._api_key = fireworks_api_key
        self._opts = STTOptions(
            model=model,
            sample_rate=sample_rate,
            language=language,
            prompt=prompt,
            temperature=temperature,
            response_format=response_format,
            timestamp_granularities=timestamp_granularities,
        )
        self._session = http_session
        self._streams = weakref.WeakSet[SpeechStream]()

    @property
    def session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        raise NotImplementedError(
            "FireworksAI STT does not support batch recognition, use stream() instead"
        )

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        config = dataclasses.replace(self._opts)
        stream = SpeechStream(
            stt=self,
            opts=config,
            conn_options=conn_options,
            api_key=self._api_key,
            http_session=self.session,
        )
        self._streams.add(stream)
        return stream

    def update_options(
        self,
        *,
        model: NotGivenOr[str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        prompt: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        timestamp_granularities: NotGivenOr[list[str]] = NOT_GIVEN,
    ) -> None:
        if is_given(model):
            self._opts.model = model
        if is_given(language):
            self._opts.language = language
        if is_given(prompt):
            self._opts.prompt = prompt
        if is_given(temperature):
            self._opts.temperature = temperature
        if is_given(timestamp_granularities):
            self._opts.timestamp_granularities = timestamp_granularities

        for stream in self._streams:
            stream.update_options(
                model=model,
                language=language,
                prompt=prompt,
                temperature=temperature,
                timestamp_granularities=timestamp_granularities,
            )


class SpeechStream(stt.SpeechStream):
    _CLOSE_MSG: str = json.dumps({"checkpoint_id": "final"})

    def __init__(
        self,
        *,
        stt: STT,
        opts: STTOptions,
        conn_options: APIConnectOptions,
        api_key: str,
        http_session: aiohttp.ClientSession,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)

        self._opts = opts
        self._api_key = api_key
        self._session = http_session
        self._transcript_state: dict[str, str] = {}
        self._reconnect_event = asyncio.Event()
        self._last_full_transcript: str = ""
        self._speaking = False
        self._audio_duration_collector = _PeriodicCollector(
            callback=self._on_audio_duration_report,
            duration=10.0,
        )

    def update_options(
        self,
        *,
        model: NotGivenOr[str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        prompt: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        timestamp_granularities: NotGivenOr[list[str]] = NOT_GIVEN,
    ) -> None:
        if is_given(model):
            self._opts.model = model
        if is_given(language):
            self._opts.language = language
        if is_given(prompt):
            self._opts.prompt = prompt
        if is_given(temperature):
            self._opts.temperature = temperature
        if is_given(timestamp_granularities):
            self._opts.timestamp_granularities = timestamp_granularities

        self._reconnect_event.set()

    async def _run(self) -> None:
        """
        Run a single websocket connection to Fireworks and make sure to reconnect
        when something went wrong.
        """

        closing_ws = False

        async def send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal closing_ws

            samples_per_buffer = self._opts.sample_rate // 20  # 50ms chunk
            audio_bstream = utils.audio.AudioByteStream(
                sample_rate=self._opts.sample_rate,
                num_channels=1,
                samples_per_channel=samples_per_buffer,
            )

            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    frames = audio_bstream.flush()
                else:
                    frames = audio_bstream.write(data.data.tobytes())

                for frame in frames:
                    await self._audio_duration_collector.push(frame.duration)
                    await ws.send_bytes(frame.data.tobytes())

            closing_ws = True
            await ws.send_str(self._CLOSE_MSG)

        async def recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal closing_ws
            while True:
                try:
                    msg = await asyncio.wait_for(ws.receive(), timeout=5)
                except asyncio.TimeoutError:
                    if closing_ws:
                        break
                    continue

                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if closing_ws:
                        return

                    raise APIStatusError(
                        "Fireworks connection closed unexpectedly",
                    )

                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.error("unexpected FireworksAI message type %s", msg.type)
                    continue

                try:
                    self._process_stream_event(json.loads(msg.data))
                except Exception:
                    logger.exception("failed to process FireworksAI message")

        ws: aiohttp.ClientWebSocketResponse | None = None

        while True:
            try:
                ws = await self._connect_ws()
                tasks = [
                    asyncio.create_task(send_task(ws)),
                    asyncio.create_task(recv_task(ws)),
                ]
                wait_reconnect_task = asyncio.create_task(self._reconnect_event.wait())

                try:
                    done, _ = await asyncio.wait(
                        (asyncio.gather(*tasks), wait_reconnect_task),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in done:
                        if task != wait_reconnect_task:
                            task.result()

                    if wait_reconnect_task not in done:
                        break

                    self._reconnect_event.clear()
                finally:
                    await utils.aio.gracefully_cancel(*tasks, wait_reconnect_task)
            finally:
                if self._speaking:
                    self._speaking = False
                    end_event = stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH)
                    self._event_ch.send_nowait(end_event)

                if ws is not None:
                    await ws.close()

                await self._audio_duration_collector.flush()

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        live_config = {
            "model": self._opts.model,
            "language": self._opts.language if is_given(self._opts.language) else None,
            "prompt": self._opts.prompt if is_given(self._opts.prompt) else None,
            "temperature": self._opts.temperature if is_given(self._opts.temperature) else None,
            "response_format": self._opts.response_format,
            "timestamp_granularities": (
                self._opts.timestamp_granularities
                if is_given(self._opts.timestamp_granularities)
                else None
            ),
        }

        headers = {
            "User-Agent": "Python/3.12",
            "Authorization": self._api_key,
        }

        ws_url = "wss://audio-streaming.us-virginia-1.direct.fireworks.ai/v1/audio/transcriptions/streaming"
        filtered_config = {k: v for k, v in live_config.items() if v is not None}
        url = f"{ws_url}?{urlencode(filtered_config, doseq=True)}"
        ws = await self._session.ws_connect(url, headers=headers)
        return ws

    def _process_stream_event(self, data: dict) -> None:
        if "segments" in data:
            for segment in data["segments"]:
                self._transcript_state[segment["id"]] = segment["text"]

            # The state dictionary may not be sorted, so we must sort it by the segment ID
            # before joining the text.
            sorted_segments = sorted(self._transcript_state.items(), key=lambda item: int(item[0]))
            full_transcript = " ".join([text for _, text in sorted_segments])

            if not full_transcript or full_transcript == self._last_full_transcript:
                return

            if not self._speaking:
                self._speaking = True
                start_event = stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)
                self._event_ch.send_nowait(start_event)

            self._last_full_transcript = full_transcript
            # This is always an interim transcript because the server maintains the state
            # and sends updates.
            logger.debug('Interim Transcript: "%s"', full_transcript)
            final_event = stt.SpeechEvent(
                type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                alternatives=[
                    stt.SpeechData(language=self._opts.language or "en", text=full_transcript)
                ],
            )
            self._event_ch.send_nowait(final_event)

    def _on_audio_duration_report(self, duration: float) -> None:
        logger.info(
            "Fireworks Audio Usage Report: %.2fs (Note: a slight, expected system "
            "delay may cause this to differ from the report interval)",
            duration,
        )
        usage_event = stt.SpeechEvent(
            type=stt.SpeechEventType.RECOGNITION_USAGE,
            recognition_usage=stt.RecognitionUsage(audio_duration=duration),
        )
        self._event_ch.send_nowait(usage_event)
