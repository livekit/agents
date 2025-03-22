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

import aiohttp

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectOptions,
    APIStatusError,
    stt,
    utils,
)
from livekit.agents.utils import AudioBuffer

from .log import logger
from .types import (
    AudioSettings,
    ClientMessageType,
    ConnectionSettings,
    ServerMessageType,
    TranscriptionConfig,
)
from .utils import get_access_token, sanitize_url


class STT(stt.STT):
    def __init__(
        self,
        *,
        transcription_config: TranscriptionConfig = TranscriptionConfig(
            language="en",
            operating_point="enhanced",
            enable_partials=True,
            max_delay=0.7,
        ),
        connection_settings: ConnectionSettings = ConnectionSettings(
            url="wss://eu2.rt.speechmatics.com/v2",
        ),
        audio_settings: AudioSettings = AudioSettings(),
        http_session: aiohttp.ClientSession | None = None,
        extra_headers: dict | None = None,
    ):
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
            ),
        )
        self._transcription_config = transcription_config
        self._audio_settings = audio_settings
        self._connection_settings = connection_settings
        self._extra_headers = extra_headers or {}
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
        language: str | None,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        raise NotImplementedError("Not implemented")

    def stream(
        self,
        *,
        language: str | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        config = dataclasses.replace(self._audio_settings)
        stream = SpeechStream(
            stt=self,
            transcription_config=self._transcription_config,
            audio_settings=config,
            connection_settings=self._connection_settings,
            conn_options=conn_options,
            http_session=self.session,
            extra_headers=self._extra_headers,
        )
        self._streams.add(stream)
        return stream


class SpeechStream(stt.SpeechStream):
    def __init__(
        self,
        *,
        stt: STT,
        transcription_config: TranscriptionConfig,
        audio_settings: AudioSettings,
        connection_settings: ConnectionSettings,
        conn_options: APIConnectOptions,
        http_session: aiohttp.ClientSession,
        extra_headers: dict | None = None,
    ) -> None:
        super().__init__(
            stt=stt, conn_options=conn_options, sample_rate=audio_settings.sample_rate
        )
        self._transcription_config = transcription_config
        self._audio_settings = audio_settings
        self._connection_settings = connection_settings
        self._session = http_session
        self._extra_headers = extra_headers or {}
        self._speech_duration: float = 0

        self._reconnect_event = asyncio.Event()
        self._recognition_started = asyncio.Event()
        self._seq_no = 0

    async def _run(self):
        closing_ws = False

        async def send_task(ws: aiohttp.ClientWebSocketResponse):
            nonlocal closing_ws

            start_recognition_msg = {
                "message": ClientMessageType.StartRecognition,
                "audio_format": self._audio_settings.asdict(),
                "transcription_config": self._transcription_config.asdict(),
            }
            await ws.send_str(json.dumps(start_recognition_msg))

            await self._recognition_started.wait()

            audio_bstream = utils.audio.AudioByteStream(
                sample_rate=self._audio_settings.sample_rate,
                num_channels=1,
            )

            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    frames = audio_bstream.flush()
                else:
                    frames = audio_bstream.write(data.data.tobytes())

                for frame in frames:
                    self._seq_no += 1
                    self._speech_duration += frame.duration
                    await ws.send_bytes(frame.data.tobytes())

            closing_ws = True
            await ws.send_str(
                json.dumps(
                    {
                        "message": ClientMessageType.EndOfStream,
                        "last_seq_no": self._seq_no,
                    }
                )
            )

        async def recv_task(ws: aiohttp.ClientWebSocketResponse):
            nonlocal closing_ws
            while True:
                msg = await ws.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if closing_ws:  # close is expected, see SpeechStream.aclose
                        return

                    # this will trigger a reconnection, see the _run loop
                    raise APIStatusError(
                        message="Speechmatics connection closed unexpectedly"
                    )

                try:
                    data = json.loads(msg.data)
                    self._process_stream_event(data, closing_ws)
                except Exception:
                    logger.exception("failed to process Speechmatics message")

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
                        [asyncio.gather(*tasks), wait_reconnect_task],
                        return_when=asyncio.FIRST_COMPLETED,
                    )  # type: ignore
                    for task in done:
                        if task != wait_reconnect_task:
                            task.result()

                    if wait_reconnect_task not in done:
                        break

                    self._reconnect_event.clear()
                finally:
                    await utils.aio.gracefully_cancel(*tasks, wait_reconnect_task)
            finally:
                if ws is not None:
                    await ws.close()

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        api_key = self._connection_settings.api_key or os.environ.get(
            "SPEECHMATICS_API_KEY"
        )
        if api_key is None:
            raise ValueError(
                "Speechmatics API key is required. "
                "Pass one in via ConnectionSettings.api_key parameter, "
                "or set `SPEECHMATICS_API_KEY` environment variable"
            )
        if self._connection_settings.get_access_token:
            api_key = await get_access_token(api_key)
        headers = {
            "Authorization": f"Bearer {api_key}",
            **self._extra_headers,
        }
        url = sanitize_url(
            self._connection_settings.url, self._transcription_config.language
        )
        return await self._session.ws_connect(
            url,
            ssl=self._connection_settings.ssl_context,
            headers=headers,
        )

    def _process_stream_event(self, data: dict, closing_ws: bool) -> None:
        message_type = data["message"]

        if message_type == ServerMessageType.RecognitionStarted:
            self._recognition_started.set()

        elif message_type == ServerMessageType.AddPartialTranscript:
            alts = live_transcription_to_speech_data(data)
            if len(alts) > 0 and alts[0].text:
                interim_event = stt.SpeechEvent(
                    type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                    alternatives=alts,
                )
                self._event_ch.send_nowait(interim_event)

        elif message_type == ServerMessageType.AddTranscript:
            alts = live_transcription_to_speech_data(data)
            if len(alts) > 0 and alts[0].text:
                final_event = stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=alts,
                )
                self._event_ch.send_nowait(final_event)

            if self._speech_duration > 0:
                usage_event = stt.SpeechEvent(
                    type=stt.SpeechEventType.RECOGNITION_USAGE,
                    alternatives=[],
                    recognition_usage=stt.RecognitionUsage(
                        audio_duration=self._speech_duration
                    ),
                )
                self._event_ch.send_nowait(usage_event)
                self._speech_duration = 0

        elif message_type == ServerMessageType.EndOfTranscript:
            if closing_ws:
                pass
            else:
                raise Exception("Speechmatics connection closed unexpectedly")


def live_transcription_to_speech_data(data: dict) -> list[stt.SpeechData]:
    speech_data: list[stt.SpeechData] = []

    for result in data.get("results", []):
        start_time, end_time, is_eos = (
            result.get("start_time", 0),
            result.get("end_time", 0),
            result.get("is_eos", False),
        )

        for alt in result.get("alternatives", []):
            content, confidence, language = (
                alt.get("content", "").strip(),
                alt.get("confidence", 1.0),
                alt.get("language", "en"),
            )

            if not content:
                continue

            # append punctuation to the previous result
            if is_eos and speech_data:
                speech_data[-1].text += content
            elif speech_data and start_time == speech_data[-1].end_time:
                speech_data[-1].text += " " + content
            else:
                speech_data.append(
                    stt.SpeechData(language, content, start_time, end_time, confidence)
                )

    return speech_data
