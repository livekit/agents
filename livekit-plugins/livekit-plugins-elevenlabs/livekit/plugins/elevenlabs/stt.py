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
import weakref
from dataclasses import dataclass
from typing import TypedDict

import aiohttp

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    stt,
    utils,
)
from livekit.agents.stt import SpeechEventType, STTCapabilities
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, http_context, is_given

from .log import logger
from .models import STTRealtimeSampleRates

API_BASE_URL_V1 = "https://api.elevenlabs.io/v1"
AUTHORIZATION_HEADER = "xi-api-key"


class VADOptions(TypedDict, total=False):
    vad_silence_threshold_secs: float | None
    """Silence threshold in seconds for VAD. Default to 1.5"""
    vad_threshold: float | None
    """Threshold for voice activity detection. Default to 0.4"""
    min_speech_duration_ms: int | None
    """Minimum speech duration in milliseconds. Default to 250"""
    min_silence_duration_ms: int | None
    """Minimum silence duration in milliseconds. Default to 2500"""


@dataclass
class STTOptions:
    api_key: str
    base_url: str
    language_code: str | None
    tag_audio_events: bool
    sample_rate: STTRealtimeSampleRates
    server_vad: NotGivenOr[VADOptions | None]


class STT(stt.STT):
    def __init__(
        self,
        *,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        language_code: NotGivenOr[str] = NOT_GIVEN,
        tag_audio_events: bool = True,
        use_realtime: bool = False,
        sample_rate: STTRealtimeSampleRates = 16000,
        server_vad: NotGivenOr[VADOptions] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """
        Create a new instance of ElevenLabs STT.

        Args:
            api_key (NotGivenOr[str]): ElevenLabs API key. Can be set via argument or `ELEVEN_API_KEY` environment variable.
            base_url (NotGivenOr[str]): Custom base URL for the API. Optional.
            language_code (NotGivenOr[str]): Language code for the STT model. Optional.
            tag_audio_events (bool): Whether to tag audio events like (laughter), (footsteps), etc. in the transcription.
                Only supported for Scribe v1 model. Default is True.
            use_realtime (bool): Whether to use "scribe_v2_realtime" model for streaming mode. Default is False.
            sample_rate (STTRealtimeSampleRates): Audio sample rate in Hz. Default is 16000.
            server_vad (NotGivenOr[VADOptions]): Server-side VAD options, only supported for Scribe v2 realtime model.
            http_session (aiohttp.ClientSession | None): Custom HTTP session for API requests. Optional.
        """  # noqa: E501

        super().__init__(capabilities=STTCapabilities(streaming=use_realtime, interim_results=True))

        if not use_realtime and is_given(server_vad):
            logger.warning("Server-side VAD is only supported for Scribe v2 realtime model")

        elevenlabs_api_key = api_key if is_given(api_key) else os.environ.get("ELEVEN_API_KEY")
        if not elevenlabs_api_key:
            raise ValueError(
                "ElevenLabs API key is required, either as argument or "
                "set ELEVEN_API_KEY environmental variable"
            )
        self._opts = STTOptions(
            api_key=elevenlabs_api_key,
            base_url=base_url if is_given(base_url) else API_BASE_URL_V1,
            language_code=language_code or None,
            tag_audio_events=tag_audio_events,
            sample_rate=sample_rate,
            server_vad=server_vad,
        )
        self._session = http_session
        self._streams = weakref.WeakSet[SpeechStream]()

    @property
    def model(self) -> str:
        return "Scribe"

    @property
    def provider(self) -> str:
        return "ElevenLabs"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = http_context.http_session()

        return self._session

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        if is_given(language):
            self._opts.language_code = language

        wav_bytes = rtc.combine_audio_frames(buffer).to_wav_bytes()
        form = aiohttp.FormData()
        form.add_field("file", wav_bytes, filename="audio.wav", content_type="audio/x-wav")
        form.add_field("model_id", "scribe_v1")
        form.add_field("tag_audio_events", str(self._opts.tag_audio_events).lower())
        if self._opts.language_code:
            form.add_field("language_code", self._opts.language_code)

        try:
            async with self._ensure_session().post(
                f"{self._opts.base_url}/speech-to-text",
                data=form,
                headers={AUTHORIZATION_HEADER: self._opts.api_key},
            ) as response:
                response_json = await response.json()
                if response.status != 200:
                    raise APIStatusError(
                        message=response_json.get("detail", "Unknown ElevenLabs error"),
                        status_code=response.status,
                        request_id=None,
                        body=response_json,
                    )
                extracted_text = response_json.get("text")
                language_code = response_json.get("language_code")
                speaker_id = None
                start_time, end_time = 0, 0
                words = response_json.get("words")
                if words:
                    speaker_id = words[0].get("speaker_id", None)
                    start_time = min(w.get("start", 0) for w in words)
                    end_time = max(w.get("end", 0) for w in words)

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

        return self._transcription_to_speech_event(
            language_code=language_code,
            text=extracted_text,
            start_time=start_time,
            end_time=end_time,
            speaker_id=speaker_id,
        )

    def _transcription_to_speech_event(
        self,
        language_code: str,
        text: str,
        start_time: float,
        end_time: float,
        speaker_id: str | None,
    ) -> stt.SpeechEvent:
        return stt.SpeechEvent(
            type=SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[
                stt.SpeechData(
                    text=text,
                    language=language_code,
                    speaker_id=speaker_id,
                    start_time=start_time,
                    end_time=end_time,
                )
            ],
        )

    def update_options(
        self,
        *,
        tag_audio_events: NotGivenOr[bool] = NOT_GIVEN,
        server_vad: NotGivenOr[VADOptions] = NOT_GIVEN,
    ) -> None:
        if is_given(tag_audio_events):
            self._opts.tag_audio_events = tag_audio_events

        if is_given(server_vad):
            self._opts.server_vad = server_vad

        for stream in self._streams:
            stream.update_options(server_vad=server_vad)

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        stream = SpeechStream(
            stt=self,
            opts=self._opts,
            conn_options=conn_options,
            language=language if is_given(language) else self._opts.language_code,
            http_session=self._ensure_session(),
        )
        self._streams.add(stream)
        return stream


class SpeechStream(stt.SpeechStream):
    """Streaming speech recognition using ElevenLabs Scribe v2 realtime API"""

    def __init__(
        self,
        *,
        stt: STT,
        opts: STTOptions,
        conn_options: APIConnectOptions,
        language: str | None,
        http_session: aiohttp.ClientSession,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)
        self._opts = opts
        self._language = language
        self._session = http_session
        self._reconnect_event = asyncio.Event()
        self._speaking = False  # Track if we're currently in a speech segment

    def update_options(
        self,
        *,
        server_vad: NotGivenOr[VADOptions] = NOT_GIVEN,
    ) -> None:
        if is_given(server_vad):
            self._opts.server_vad = server_vad
            self._reconnect_event.set()

    async def _run(self) -> None:
        """Run the streaming transcription session"""
        closing_ws = False

        async def keepalive_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            try:
                while True:
                    await ws.ping()
                    await asyncio.sleep(30)
            except Exception:
                return

        @utils.log_exceptions(logger=logger)
        async def send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal closing_ws

            # Buffer audio into chunks (50ms chunks)
            samples_50ms = self._opts.sample_rate // 20
            audio_bstream = utils.audio.AudioByteStream(
                sample_rate=self._opts.sample_rate,
                num_channels=1,
                samples_per_channel=samples_50ms,
            )

            async for data in self._input_ch:
                # Write audio bytes to buffer and get 50ms frames
                frames: list[rtc.AudioFrame] = []
                if isinstance(data, rtc.AudioFrame):
                    frames.extend(audio_bstream.write(data.data.tobytes()))
                elif isinstance(data, self._FlushSentinel):
                    frames.extend(audio_bstream.flush())

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

        @utils.log_exceptions(logger=logger)
        async def recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal closing_ws

            while True:
                msg = await ws.receive()

                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if closing_ws or self._session.closed:
                        return
                    raise APIStatusError(message="ElevenLabs STT connection closed unexpectedly")

                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.warning("unexpected ElevenLabs STT message type %s", msg.type)
                    continue

                try:
                    parsed = json.loads(msg.data)
                    self._process_stream_event(parsed)
                except Exception:
                    logger.exception("failed to process ElevenLabs STT message")

        ws: aiohttp.ClientWebSocketResponse | None = None

        while True:
            try:
                ws = await self._connect_ws()
                tasks = [
                    asyncio.create_task(send_task(ws)),
                    asyncio.create_task(recv_task(ws)),
                    asyncio.create_task(keepalive_task(ws)),
                ]
                tasks_group = asyncio.gather(*tasks)
                wait_reconnect_task = asyncio.create_task(self._reconnect_event.wait())

                try:
                    done, _ = await asyncio.wait(
                        (tasks_group, wait_reconnect_task),
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
                    tasks_group.cancel()
                    tasks_group.exception()  # Retrieve exception to prevent it from being logged
            finally:
                if ws is not None:
                    await ws.close()

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        """Establish WebSocket connection to ElevenLabs Scribe v2 API"""
        commit_strategy = "manual" if self._opts.server_vad is None else "vad"
        params = [
            "model_id=scribe_v2_realtime",
            f"encoding=pcm_{self._opts.sample_rate}",
            f"commit_strategy={commit_strategy}",
        ]

        if server_vad := self._opts.server_vad:
            if (
                vad_silence_threshold_secs := server_vad.get("vad_silence_threshold_secs")
            ) is not None:
                params.append(f"vad_silence_threshold_secs={vad_silence_threshold_secs}")
            if (vad_threshold := server_vad.get("vad_threshold")) is not None:
                params.append(f"vad_threshold={vad_threshold}")
            if (min_speech_duration_ms := server_vad.get("min_speech_duration_ms")) is not None:
                params.append(f"min_speech_duration_ms={min_speech_duration_ms}")
            if (min_silence_duration_ms := server_vad.get("min_silence_duration_ms")) is not None:
                params.append(f"min_silence_duration_ms={min_silence_duration_ms}")

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
        text = data.get("text", "")

        speech_data = stt.SpeechData(
            language=self._language or "en",
            text=text,
        )

        if message_type == "partial_transcript":
            logger.debug("Received message type partial_transcript: %s", data)

            if text:
                # Send START_OF_SPEECH if we're not already speaking
                if not self._speaking:
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(type=SpeechEventType.START_OF_SPEECH)
                    )
                    self._speaking = True

                # Send INTERIM_TRANSCRIPT
                interim_event = stt.SpeechEvent(
                    type=SpeechEventType.INTERIM_TRANSCRIPT,
                    alternatives=[speech_data],
                )
                self._event_ch.send_nowait(interim_event)

        elif message_type == "committed_transcript":
            logger.debug("Received message type committed_transcript: %s", data)

            # Final committed transcripts - these are sent to the LLM/TTS layer in LiveKit agents
            # and trigger agent responses (unlike partial transcripts which are UI-only)

            if text:
                # Send START_OF_SPEECH if we're not already speaking
                if not self._speaking:
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(type=SpeechEventType.START_OF_SPEECH)
                    )
                    self._speaking = True

                # Send FINAL_TRANSCRIPT but keep speaking=True
                # Multiple commits can occur within the same speech segment
                final_event = stt.SpeechEvent(
                    type=SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=[speech_data],
                )
                self._event_ch.send_nowait(final_event)
            else:
                # Empty commit signals end of speech segment (similar to Cartesia's is_final flag)
                # This groups multiple committed transcripts into one speech segment
                if self._speaking:
                    self._event_ch.send_nowait(stt.SpeechEvent(type=SpeechEventType.END_OF_SPEECH))
                    self._speaking = False

        elif message_type == "session_started":
            # Session initialization message - informational only
            session_id = data.get("session_id", "unknown")
            logger.debug("Session started with ID: %s", session_id)

        elif message_type == "committed_transcript_with_timestamps":
            logger.debug("Received message type committed_transcript_with_timestamps: %s", data)

        # Error handling for known ElevenLabs error types
        elif message_type in (
            "auth_error",
            "quota_exceeded",
            "transcriber_error",
            "input_error",
            "error",
        ):
            error_msg = data.get("message", "Unknown error")
            error_details = data.get("details", "")
            details_suffix = " - " + error_details if error_details else ""
            logger.error(
                "ElevenLabs STT error [%s]: %s%s",
                message_type,
                error_msg,
                details_suffix,
            )
            raise APIConnectionError(f"{message_type}: {error_msg}{details_suffix}")
        else:
            logger.warning("ElevenLabs STT unknown message type: %s, data: %s", message_type, data)
