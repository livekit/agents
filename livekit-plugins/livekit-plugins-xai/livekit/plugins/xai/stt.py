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
import uuid
import weakref
from dataclasses import dataclass

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
from livekit.agents.language import LanguageCode
from livekit.agents.types import (
    NOT_GIVEN,
    NotGivenOr,
    TimedString,
)
from livekit.agents.utils import AudioBuffer, is_given

from ._utils import PeriodicCollector
from .log import logger
from .types import STTLanguages

SAMPLE_RATE = 16000
XAI_WEBSOCKET_URL = "wss://api.x.ai/v1/stt"
XAI_REST_URL = "https://api.x.ai/v1/stt"


@dataclass
class STTOptions:
    enable_interim_results: bool
    sample_rate: int
    enable_diarization: bool
    language: STTLanguages | str
    endpointing: int


class STT(stt.STT):
    def __init__(
        self,
        *,
        enable_interim_results: bool = True,
        sample_rate: int = SAMPLE_RATE,
        enable_diarization: bool = False,
        language: STTLanguages | str = "en",
        endpointing: int = 100,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """Create a new instance of xAI STT.

        Args:
            enable_interim_results (bool, optional): Whether to return interim (non-final) transcription results. Defaults to True.
            sample_rate: The sample rate of the audio in Hz. Defaults to 16000.
            enable_diarization: Whether to enable speaker diarization. Words will include a speaker field. Defaults to False.
            language: BCP-47 language code for transcription (e.g. "en", "fr", "de"). Defaults to "en".
            endpointing: Silence duration in milliseconds before an utterance-final event is fired. xAI's default is 10ms, but we default to 100ms for better compatibility with LK EOT models.
            api_key: Your xAI API key. If not provided, will look for XAI_API_KEY environment variable.
            http_session: Optional aiohttp ClientSession to use for requests.

        Raises:
            ValueError: If no API key is provided or found in environment variables.

        Note:
            The api_key must be set either through the constructor argument or by setting
            the XAI_API_KEY environmental variable.
        """  # noqa: E501

        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=enable_interim_results,
                diarization=enable_diarization,
                aligned_transcript="word",
            )
        )

        xai_api_key = api_key if is_given(api_key) else os.environ.get("XAI_API_KEY")
        if not xai_api_key:
            raise ValueError("xAI API key is required")
        self._api_key = xai_api_key

        self._opts = STTOptions(
            enable_interim_results=enable_interim_results,
            sample_rate=sample_rate,
            enable_diarization=enable_diarization,
            language=language,
            endpointing=endpointing,
        )
        self._session = http_session
        self._streams = weakref.WeakSet[SpeechStream]()

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

        return self._session

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        lang = language if is_given(language) else self._opts.language
        form = aiohttp.FormData()
        form.add_field(
            "file",
            rtc.combine_audio_frames(buffer).to_wav_bytes(),
            filename="audio.wav",
            content_type="audio/wav",
        )
        form.add_field("language", lang)
        form.add_field("format", "true")

        try:
            async with self._ensure_session().post(
                url=XAI_REST_URL,
                data=form,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Accept": "application/json",
                },
                timeout=aiohttp.ClientTimeout(
                    total=30,
                    sock_connect=conn_options.timeout,
                ),
            ) as res:
                return _prerecorded_transcription_to_speech_event(
                    await res.json(), enable_diarization=self._opts.enable_diarization
                )
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

    def _sanitize_options(self, *, language: NotGivenOr[str] = NOT_GIVEN) -> STTOptions:
        config = dataclasses.replace(self._opts)
        if is_given(language):
            config.language = language
        return config

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        config = self._sanitize_options(language=language)
        stream = SpeechStream(
            stt=self,
            conn_options=conn_options,
            opts=config,
            api_key=self._api_key,
            http_session=self._ensure_session(),
        )
        self._streams.add(stream)
        return stream

    def update_options(
        self,
        *,
        interim_results: NotGivenOr[bool] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        enable_diarization: NotGivenOr[bool] = NOT_GIVEN,
        language: NotGivenOr[STTLanguages | str] = NOT_GIVEN,
        endpointing: NotGivenOr[int] = NOT_GIVEN,
    ) -> None:
        if is_given(interim_results):
            self._opts.enable_interim_results = interim_results

        if is_given(sample_rate):
            self._opts.sample_rate = sample_rate

        if is_given(enable_diarization):
            self._opts.enable_diarization = enable_diarization

        if is_given(language):
            self._opts.language = language

        if is_given(endpointing):
            self._opts.endpointing = endpointing

        for stream in self._streams:
            stream.update_options(
                enable_interim_results=interim_results,
                sample_rate=sample_rate,
                enable_diarization=enable_diarization,
                language=language,
                endpointing=endpointing,
            )


class SpeechStream(stt.RecognizeStream):
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
        self._speaking = False
        self._emitted_chunk_final = False
        self._audio_duration_collector = PeriodicCollector(
            callback=self._on_audio_duration_report,
            duration=5.0,
        )

        self._request_id = str(uuid.uuid4())
        self._reconnect_event = asyncio.Event()
        self._server_ready = asyncio.Event()

    def update_options(
        self,
        *,
        enable_interim_results: NotGivenOr[bool] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        enable_diarization: NotGivenOr[bool] = NOT_GIVEN,
        language: NotGivenOr[STTLanguages | str] = NOT_GIVEN,
        endpointing: NotGivenOr[int] = NOT_GIVEN,
    ) -> None:
        if is_given(enable_interim_results):
            self._opts.enable_interim_results = enable_interim_results

        if is_given(sample_rate):
            self._opts.sample_rate = sample_rate

        if is_given(enable_diarization):
            self._opts.enable_diarization = enable_diarization

        if is_given(language):
            self._opts.language = language

        if is_given(endpointing):
            self._opts.endpointing = endpointing

        self._reconnect_event.set()

    async def _run(self) -> None:
        closing_ws = False

        @utils.log_exceptions(logger=logger)
        async def send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal closing_ws

            await self._server_ready.wait()

            samples_50ms = self._opts.sample_rate // 20
            audio_bstream = utils.audio.AudioByteStream(
                sample_rate=self._opts.sample_rate,
                num_channels=1,
                samples_per_channel=samples_50ms,
            )

            async for data in self._input_ch:
                frames: list[rtc.AudioFrame] = []
                if isinstance(data, rtc.AudioFrame):
                    frames.extend(audio_bstream.write(data.data.tobytes()))
                elif isinstance(data, self._FlushSentinel):
                    frames.extend(audio_bstream.flush())

                for frame in frames:
                    self._audio_duration_collector.push(frame.duration)
                    await ws.send_bytes(frame.data.tobytes())

            self._audio_duration_collector.flush()
            await ws.send_str(json.dumps({"type": "audio.done"}))
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

                    raise APIStatusError(message="xAI connection closed unexpectedly")

                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.warning("unexpected xAI message type %s", msg.type)
                    continue

                try:
                    self._process_stream_event(json.loads(msg.data))
                except Exception:
                    logger.exception("failed to process xAI message")

        ws: aiohttp.ClientWebSocketResponse | None = None

        while True:
            try:
                closing_ws = False
                ws = await self._connect_ws()
                self._server_ready.clear()
                tasks = [
                    asyncio.create_task(send_task(ws)),
                    asyncio.create_task(recv_task(ws)),
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
                    tasks_group.exception()
            finally:
                if ws is not None:
                    await ws.close()

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        params = {
            "encoding": "pcm",
            "sample_rate": str(self._opts.sample_rate),
            "interim_results": str(self._opts.enable_interim_results).lower(),
            "diarize": str(self._opts.enable_diarization).lower(),
            "language": str(self._opts.language),
            "endpointing": str(self._opts.endpointing),
        }
        try:
            ws = await asyncio.wait_for(
                self._session.ws_connect(
                    XAI_WEBSOCKET_URL,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    params=params,
                ),
                self._conn_options.timeout,
            )
        except (aiohttp.ClientConnectorError, asyncio.TimeoutError) as e:
            raise APIConnectionError("failed to connect to xAI") from e
        return ws

    def _on_audio_duration_report(self, duration: float) -> None:
        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=stt.SpeechEventType.RECOGNITION_USAGE,
                request_id=self._request_id,
                recognition_usage=stt.RecognitionUsage(audio_duration=duration),
            )
        )

    def _process_stream_event(self, data: dict) -> None:
        msg_type = data.get("type", "")

        if msg_type == "transcript.created":
            self._server_ready.set()

        elif msg_type == "transcript.partial":
            text = data.get("text", "")
            is_final = data.get("is_final", False)
            speech_final = data.get("speech_final", False)
            words = data.get("words", [])
            language = data.get("language", "")

            if not text:
                return

            if not self._speaking:
                self._speaking = True
                self._event_ch.send_nowait(
                    stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)
                )

            if is_final:
                if not speech_final:
                    self._emitted_chunk_final = True
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(
                            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                            request_id=self._request_id,
                            alternatives=[
                                _words_to_speech_data(
                                    words,
                                    text,
                                    language=language,
                                    enable_diarization=self._opts.enable_diarization,
                                )
                            ],
                        )
                    )
                else:
                    if not self._emitted_chunk_final:
                        self._event_ch.send_nowait(
                            stt.SpeechEvent(
                                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                                request_id=self._request_id,
                                alternatives=[
                                    _words_to_speech_data(
                                        words,
                                        text,
                                        language=language,
                                        enable_diarization=self._opts.enable_diarization,
                                    )
                                ],
                            )
                        )
                    self._emitted_chunk_final = False
                    self._speaking = False
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH)
                    )
            else:
                if self._opts.enable_interim_results:
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(
                            type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                            request_id=self._request_id,
                            alternatives=[
                                stt.SpeechData(language=LanguageCode(language), text=text)
                            ],
                        )
                    )

        elif msg_type == "transcript.done":
            text = data.get("text", "")
            words = data.get("words", [])
            language = data.get("language", "")
            if text:
                self._event_ch.send_nowait(
                    stt.SpeechEvent(
                        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                        request_id=self._request_id,
                        alternatives=[
                            _words_to_speech_data(
                                words,
                                text,
                                language=language,
                                enable_diarization=self._opts.enable_diarization,
                            )
                        ],
                    )
                )
            if self._speaking:
                self._speaking = False
                self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH))

        elif msg_type == "error":
            logger.error("xAI STT error: %s", data.get("message", "unknown error"))

        else:
            logger.warning("received unexpected message from xAI: %s", msg_type)


def _words_to_speech_data(
    words: list[dict], text: str, *, language: str, enable_diarization: bool
) -> stt.SpeechData:
    speaker_id = (
        f"S{words[0]['speaker']}"
        if enable_diarization and words and "speaker" in words[0]
        else None
    )
    return stt.SpeechData(
        language=LanguageCode(language),
        text=text,
        start_time=words[0].get("start", 0.0) if words else 0.0,
        end_time=words[-1].get("end", 0.0) if words else 0.0,
        speaker_id=speaker_id,
        words=[
            TimedString(
                w.get("text", ""), start_time=w.get("start", 0.0), end_time=w.get("end", 0.0)
            )
            for w in words
        ]
        or None,
    )


def _prerecorded_transcription_to_speech_event(
    data: dict, *, enable_diarization: bool
) -> stt.SpeechEvent:
    text = data.get("text", "")
    words = data.get("words", [])
    language = data.get("language", "")
    return stt.SpeechEvent(
        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
        alternatives=[
            _words_to_speech_data(
                words, text, language=language, enable_diarization=enable_diarization
            )
        ],
    )
