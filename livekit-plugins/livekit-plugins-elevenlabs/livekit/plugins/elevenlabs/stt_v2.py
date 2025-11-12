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
import typing
import weakref
from dataclasses import dataclass

import aiohttp

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    stt,
    utils,
)
from livekit.agents.stt import SpeechEventType, STTCapabilities
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from .models import STTAudioFormat, STTCommitStrategy, STTModels

API_BASE_URL_V1 = "https://api.elevenlabs.io/v1"
AUTHORIZATION_HEADER = "xi-api-key"


@dataclass
class STTOptions:
    api_key: str
    base_url: str
    language_code: str | None = None
    model_id: STTModels = "scribe_v2_realtime"
    audio_format: STTAudioFormat = "pcm_16000"
    sample_rate: int = 16000
    commit_strategy: STTCommitStrategy = "vad"
    vad_silence_threshold_secs: float | None = None
    vad_threshold: float | None = None
    min_speech_duration_ms: int | None = None
    min_silence_duration_ms: int | None = None


class STTv2(stt.STT):
    def __init__(
        self,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
        language_code: NotGivenOr[str] = NOT_GIVEN,
        model_id: STTModels = "scribe_v2_realtime",
        sample_rate: int = 16000,
        commit_strategy: STTCommitStrategy = "vad",
        vad_silence_threshold_secs: NotGivenOr[float] = NOT_GIVEN,
        vad_threshold: NotGivenOr[float] = NOT_GIVEN,
        min_speech_duration_ms: NotGivenOr[int] = NOT_GIVEN,
        min_silence_duration_ms: NotGivenOr[int] = NOT_GIVEN,
    ) -> None:
        """
        Create a new instance of ElevenLabs STT v2 with streaming support.

        Args:
            api_key (NotGivenOr[str]): ElevenLabs API key. Can be set via argument or `ELEVEN_API_KEY` environment variable.
            base_url (NotGivenOr[str]): Custom base URL for the API. Optional.
            http_session (aiohttp.ClientSession | None): Custom HTTP session for API requests. Optional.
            language_code (NotGivenOr[str]): Language code for the STT model. Optional.
            model_id (STTModels): Model ID for Scribe. Default is "scribe_v2_realtime".
            sample_rate (int): Audio sample rate in Hz. Default is 16000.
            commit_strategy (STTCommitStrategy): Strategy for committing transcriptions. "vad" commits automatically when speech ends, "manual" requires explicit commits. Default is "vad".
            vad_silence_threshold_secs (NotGivenOr[float]): Silence threshold in seconds for VAD (must be between 0.3 and 3.0). Optional.
            vad_threshold (NotGivenOr[float]): Threshold for voice activity detection (must be between 0.1 and 0.9). Optional.
            min_speech_duration_ms (NotGivenOr[int]): Minimum speech duration in milliseconds (must be between 50 and 2000). Optional.
            min_silence_duration_ms (NotGivenOr[int]): Minimum silence duration in milliseconds (must be between 50 and 2000). Optional.
        """  # noqa: E501
        super().__init__(capabilities=STTCapabilities(streaming=True, interim_results=True))

        elevenlabs_api_key = api_key if is_given(api_key) else os.environ.get("ELEVEN_API_KEY")
        if not elevenlabs_api_key:
            raise ValueError(
                "ElevenLabs API key is required, either as argument or "
                "set ELEVEN_API_KEY environmental variable"
            )

        # Determine audio format based on sample rate
        audio_format = typing.cast(STTAudioFormat, f"pcm_{sample_rate}")

        self._opts = STTOptions(
            api_key=elevenlabs_api_key,
            base_url=base_url if is_given(base_url) else API_BASE_URL_V1,
            model_id=model_id,
            audio_format=audio_format,
            sample_rate=sample_rate,
            commit_strategy=commit_strategy,
            vad_silence_threshold_secs=vad_silence_threshold_secs
            if is_given(vad_silence_threshold_secs)
            else None,
            vad_threshold=vad_threshold if is_given(vad_threshold) else None,
            min_speech_duration_ms=min_speech_duration_ms
            if is_given(min_speech_duration_ms)
            else None,
            min_silence_duration_ms=min_silence_duration_ms
            if is_given(min_silence_duration_ms)
            else None,
        )
        if is_given(language_code):
            self._opts.language_code = language_code
        self._session = http_session
        self._streams = weakref.WeakSet[SpeechStreamv2]()

    @property
    def model(self) -> str:
        return self._opts.model_id

    @property
    def provider(self) -> str:
        return "ElevenLabs"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

        return self._session

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStreamv2:
        return SpeechStreamv2(
            stt=self,
            opts=self._opts,
            conn_options=conn_options,
            language=language if is_given(language) else self._opts.language_code,
            http_session=self._ensure_session(),
        )


class SpeechStreamv2(stt.SpeechStream):
    """Streaming speech recognition using ElevenLabs Scribe v2 realtime API"""

    def __init__(
        self,
        *,
        stt: STTv2,
        opts: STTOptions,
        conn_options: APIConnectOptions,
        language: str | None,
        http_session: aiohttp.ClientSession,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)
        self._opts = opts
        self._language = language
        self._session = http_session

    async def _run(self) -> None:
        """Run the streaming transcription session"""
        closing_ws = False

        async def send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal closing_ws

            # Buffer audio into chunks (50ms chunks)
            samples_per_chunk = self._opts.sample_rate // 20
            audio_bstream = utils.audio.AudioByteStream(
                sample_rate=self._opts.sample_rate,
                num_channels=1,
                samples_per_channel=samples_per_chunk,
            )

            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    frames = audio_bstream.flush()
                    # Send any remaining frames
                    for frame in frames:
                        audio_b64 = base64.b64encode(frame.data.tobytes()).decode("utf-8")
                        await ws.send_str(
                            json.dumps(
                                {
                                    "message_type": "input_audio_chunk",
                                    "audio_base_64": audio_b64,
                                    "commit": False,
                                    "sample_rate": self._opts.sample_rate,
                                }
                            )
                        )
                    # Send commit message if using manual commit strategy
                    if self._opts.commit_strategy == "manual":
                        await ws.send_str(
                            json.dumps(
                                {
                                    "message_type": "input_audio_chunk",
                                    "audio_base_64": "",
                                    "commit": True,
                                    "sample_rate": self._opts.sample_rate,
                                }
                            )
                        )
                else:
                    frames = audio_bstream.write(data.data.tobytes())
                    for frame in frames:
                        audio_b64 = base64.b64encode(frame.data.tobytes()).decode("utf-8")
                        await ws.send_str(
                            json.dumps(
                                {
                                    "message_type": "input_audio_chunk",
                                    "audio_base_64": audio_b64,
                                    "commit": False,
                                    "sample_rate": self._opts.sample_rate,
                                }
                            )
                        )

            closing_ws = True

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
                        message="ElevenLabs connection closed unexpectedly",
                    )

                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue

                try:
                    data = json.loads(msg.data)
                    self._process_stream_event(data)
                except Exception:
                    pass

        ws: aiohttp.ClientWebSocketResponse | None = None

        try:
            ws = await self._connect_ws()
            tasks = [
                asyncio.create_task(send_task(ws)),
                asyncio.create_task(recv_task(ws)),
            ]

            try:
                await asyncio.gather(*tasks)
            finally:
                await utils.aio.gracefully_cancel(*tasks)
        finally:
            if ws is not None:
                await ws.close()

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        """Establish WebSocket connection to ElevenLabs Scribe v2 API"""
        # Build query parameters
        params = [
            f"model_id={self._opts.model_id}",
            f"encoding={self._opts.audio_format}",
            f"sample_rate={self._opts.sample_rate}",
            f"commit_strategy={self._opts.commit_strategy}",
        ]

        if self._opts.vad_silence_threshold_secs is not None:
            params.append(f"vad_silence_threshold_secs={self._opts.vad_silence_threshold_secs}")
        if self._opts.vad_threshold is not None:
            params.append(f"vad_threshold={self._opts.vad_threshold}")
        if self._opts.min_speech_duration_ms is not None:
            params.append(f"min_speech_duration_ms={self._opts.min_speech_duration_ms}")
        if self._opts.min_silence_duration_ms is not None:
            params.append(f"min_silence_duration_ms={self._opts.min_silence_duration_ms}")
        if self._language:
            params.append(f"language_code={self._language}")

        query_string = "&".join(params)

        # Convert HTTPS URL to WSS
        base_url = self._opts.base_url.replace("https://", "wss://").replace("http://", "ws://")
        ws_url = f"{base_url}/speech-to-text/realtime?{query_string}"

        try:
            ws = await asyncio.wait_for(
                self._session.ws_connect(
                    ws_url,
                    headers={AUTHORIZATION_HEADER: self._opts.api_key},
                ),
                self._conn_options.timeout,
            )
        except (aiohttp.ClientConnectorError, asyncio.TimeoutError) as e:
            raise APIConnectionError("Failed to connect to ElevenLabs") from e

        return ws

    def _process_stream_event(self, data: dict) -> None:
        """Process incoming WebSocket messages from ElevenLabs"""
        message_type = data.get("message_type")

        if message_type == "partial_transcript":
            # Interim/partial transcripts
            text = data.get("transcript", "")
            if text:
                interim_event = stt.SpeechEvent(
                    type=SpeechEventType.INTERIM_TRANSCRIPT,
                    alternatives=[
                        stt.SpeechData(
                            language=self._language or "en",
                            text=text,
                        )
                    ],
                )
                self._event_ch.send_nowait(interim_event)

        elif message_type == "committed_transcript":
            # Final committed transcripts
            text = data.get("transcript", "")
            if text:
                final_event = stt.SpeechEvent(
                    type=SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=[
                        stt.SpeechData(
                            language=self._language or "en",
                            text=text,
                        )
                    ],
                )
                self._event_ch.send_nowait(final_event)

                # Send end of speech event
                self._event_ch.send_nowait(stt.SpeechEvent(type=SpeechEventType.END_OF_SPEECH))

        elif message_type == "committed_transcript_with_timestamps":
            # Committed transcript with word-level timestamps
            text = data.get("transcript", "")
            if text:
                final_event = stt.SpeechEvent(
                    type=SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=[
                        stt.SpeechData(
                            language=self._language or "en",
                            text=text,
                        )
                    ],
                )
                self._event_ch.send_nowait(final_event)

                # Send end of speech event
                self._event_ch.send_nowait(stt.SpeechEvent(type=SpeechEventType.END_OF_SPEECH))
