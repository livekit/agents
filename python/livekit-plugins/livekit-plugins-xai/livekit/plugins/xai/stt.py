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
from urllib.parse import urlencode

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APITimeoutError,
    LanguageCode,
    http,
    stt,
    utils,
)
from livekit.agents.types import (
    NOT_GIVEN,
    NotGivenOr,
    TranscriptChunk,
)
from livekit.agents.utils import is_given

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
    vad_threshold: NotGivenOr[float] = NOT_GIVEN
    smart_turn: NotGivenOr[float] = NOT_GIVEN
    smart_turn_timeout: NotGivenOr[int] = NOT_GIVEN
    keyterm: NotGivenOr[list[str]] = NOT_GIVEN


class _Word(BaseModel):
    model_config = ConfigDict(strict=True)
    text: str = ""
    start: float = 0.0
    end: float = 0.0
    speaker: int | str | None = None


class _Transcript(BaseModel):
    model_config = ConfigDict(strict=True)
    type: str = ""
    text: str = ""
    language: str = ""
    is_final: bool = False
    speech_final: bool = False
    words: list[_Word] = Field(default_factory=list[_Word])
    message: str = "unknown error"


def _decode_transcript(data: object) -> _Transcript:
    try:
        return _Transcript.model_validate(data)
    except ValidationError as error:
        raise APIError("invalid xAI transcript", kind="protocol", detail=str(data)) from error


class STT(stt.STT):
    def __init__(
        self,
        *,
        enable_interim_results: bool = True,
        sample_rate: int = SAMPLE_RATE,
        enable_diarization: bool = False,
        language: STTLanguages | str = "en",
        endpointing: int = 100,
        vad_threshold: NotGivenOr[float] = NOT_GIVEN,
        smart_turn: NotGivenOr[float] = NOT_GIVEN,
        smart_turn_timeout: NotGivenOr[int] = NOT_GIVEN,
        keyterm: NotGivenOr[list[str]] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        http_session: http.HttpSession | None = None,
    ) -> None:
        """Create a new instance of xAI STT.

        Args:
            enable_interim_results (bool, optional): Whether to return interim (non-final) transcription results. Defaults to True.
            sample_rate: The sample rate of the audio in Hz. Defaults to 16000.
            enable_diarization: Whether to enable speaker diarization. Words will include a speaker field. Defaults to False.
            language: BCP-47 language code for transcription (e.g. "en", "fr", "de"). Defaults to "en".
            endpointing: Silence duration in milliseconds before an utterance-final event is fired. xAI's default is 10ms, but we default to 100ms for better compatibility with LK EOT models.
            vad_threshold: Voice-activity detection threshold, range 0.0-1.0. Lower values capture quieter/noisier speech. When unset, xAI's websocket default (0.08) applies.
            smart_turn: Confidence threshold (0.0-1.0) for xAI's ML end-of-turn ("Smart Turn") prediction at silence boundaries. When unset, Smart Turn is left at xAI's default behavior.
            smart_turn_timeout: Maximum silence in milliseconds (1-5000) before an utterance-final event is forced even when Smart Turn predicts the speaker is continuing.
            keyterm: Key terms/phrases (max 100, 50 chars each) to bias transcription accuracy toward.
            api_key: Your xAI API key. If not provided, will look for XAI_API_KEY environment variable.
            http_session: Optional HttpSession to use for requests.

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
            vad_threshold=vad_threshold,
            smart_turn=smart_turn,
            smart_turn_timeout=smart_turn_timeout,
            keyterm=keyterm,
        )
        self._session = http_session
        self._streams = weakref.WeakSet[SpeechStream]()

    def _ensure_session(self) -> http.HttpSession:
        if not self._session:
            self._session = http.http_session()

        return self._session

    async def _recognize_impl(
        self,
        buffer: list[rtc.AudioFrame] | rtc.AudioFrame,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        lang = language if is_given(language) else self._opts.language
        form = http.FormData()
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
                XAI_REST_URL,
                data=form,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Accept": "application/json",
                },
                conn_options=conn_options,
            ) as res:
                await res.raise_for_status()
                return _prerecorded_transcription_to_speech_event(
                    await res.json(), enable_diarization=self._opts.enable_diarization
                )
        except TimeoutError as e:
            raise APITimeoutError() from e
        except APIError:
            raise
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
        vad_threshold: NotGivenOr[float] = NOT_GIVEN,
        smart_turn: NotGivenOr[float] = NOT_GIVEN,
        smart_turn_timeout: NotGivenOr[int] = NOT_GIVEN,
        keyterm: NotGivenOr[list[str]] = NOT_GIVEN,
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

        if is_given(vad_threshold):
            self._opts.vad_threshold = vad_threshold

        if is_given(smart_turn):
            self._opts.smart_turn = smart_turn

        if is_given(smart_turn_timeout):
            self._opts.smart_turn_timeout = smart_turn_timeout

        if is_given(keyterm):
            self._opts.keyterm = keyterm

        for stream in self._streams:
            stream.update_options(
                enable_interim_results=interim_results,
                sample_rate=sample_rate,
                enable_diarization=enable_diarization,
                language=language,
                endpointing=endpointing,
                vad_threshold=vad_threshold,
                smart_turn=smart_turn,
                smart_turn_timeout=smart_turn_timeout,
                keyterm=keyterm,
            )


class SpeechStream(stt.RecognizeStream):
    def __init__(
        self,
        *,
        stt: STT,
        opts: STTOptions,
        conn_options: APIConnectOptions,
        api_key: str,
        http_session: http.HttpSession,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)
        self._opts = opts
        self._api_key = api_key
        self._session = http_session
        self._speaking = False
        self._emitted_chunk_final = False

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
        vad_threshold: NotGivenOr[float] = NOT_GIVEN,
        smart_turn: NotGivenOr[float] = NOT_GIVEN,
        smart_turn_timeout: NotGivenOr[int] = NOT_GIVEN,
        keyterm: NotGivenOr[list[str]] = NOT_GIVEN,
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

        if is_given(vad_threshold):
            self._opts.vad_threshold = vad_threshold

        if is_given(smart_turn):
            self._opts.smart_turn = smart_turn

        if is_given(smart_turn_timeout):
            self._opts.smart_turn_timeout = smart_turn_timeout

        if is_given(keyterm):
            self._opts.keyterm = keyterm

        self._reconnect_event.set()

    async def _run(self) -> None:
        closing_ws = False

        async def send_task(ws: http.WebSocket) -> None:
            nonlocal closing_ws

            await self._server_ready.wait()

            samples_50ms = self._opts.sample_rate // 20
            audio_bstream = utils.media.AudioByteStream(
                sample_rate=self._opts.sample_rate,
                num_channels=1,
                samples_per_channel=samples_50ms,
            )

            async for data in self.input:
                frames: list[rtc.AudioFrame] = []
                if isinstance(data, rtc.AudioFrame):
                    frames.extend(audio_bstream.push(data.data.tobytes()))
                else:
                    frames.extend(audio_bstream.flush())

                for frame in frames:
                    await ws.send_bytes(frame.data.tobytes())

            await ws.send_str(json.dumps({"type": "audio.done"}))
            closing_ws = True

        async def recv_task(ws: http.WebSocket) -> None:
            nonlocal closing_ws
            while True:
                # a heartbeat timeout raises APIConnectionError here, so _main_task reconnects
                try:
                    msg = await ws.receive()
                except APIConnectionError:
                    if closing_ws or self._session.closed:
                        return
                    raise
                if msg.type == http.WebSocketMessageType.CLOSE:
                    if closing_ws or self._session.closed:
                        return

                    raise ws.close_error("xAI connection closed unexpectedly")

                if msg.type != http.WebSocketMessageType.TEXT:
                    logger.warning("unexpected xAI message type %s", msg.type)
                    continue

                try:
                    raw = json.loads(msg.data)
                except json.JSONDecodeError as e:
                    raise APIError(
                        "xAI sent a malformed message", kind="protocol", detail=str(msg.data)
                    ) from e
                self._process_stream_event(raw)

        ws: http.WebSocket | None = None

        while True:
            try:
                closing_ws = False
                self._speaking = False
                self._emitted_chunk_final = False
                ws = await self._connect_ws()
                self._server_ready.clear()
                async with utils.aio.TaskSet() as tasks:
                    io = asyncio.gather(tasks.spawn(send_task(ws)), tasks.spawn(recv_task(ws)))
                    reconnect = tasks.spawn(self._reconnect_event.wait())
                    if await tasks.first_completed(io, reconnect) is not reconnect:
                        break
                    self._reconnect_event.clear()

            finally:
                if ws is not None:
                    await ws.close()

    async def _connect_ws(self) -> http.WebSocket:
        # keyterm may repeat, so params is a list of pairs rather than a dict
        params = [
            ("encoding", "pcm"),
            ("sample_rate", str(self._opts.sample_rate)),
            ("interim_results", str(self._opts.enable_interim_results).lower()),
            ("diarize", str(self._opts.enable_diarization).lower()),
            ("language", str(self._opts.language)),
            ("endpointing", str(self._opts.endpointing)),
        ]
        if is_given(self._opts.vad_threshold):
            params.append(("vad_threshold", str(self._opts.vad_threshold)))
        if is_given(self._opts.smart_turn):
            params.append(("smart_turn", str(self._opts.smart_turn)))
        if is_given(self._opts.smart_turn_timeout):
            params.append(("smart_turn_timeout", str(self._opts.smart_turn_timeout)))
        if is_given(self._opts.keyterm):
            params.extend(("keyterm", term) for term in self._opts.keyterm)
        return await self._session.ws_connect(
            f"{XAI_WEBSOCKET_URL}?{urlencode(params)}",
            headers={"Authorization": f"Bearer {self._api_key}"},
            # Without this a silently dropped socket (half-open TCP, no FIN/RST)
            # is never noticed: recv_task parks on ws.receive() forever, and the
            # retry in _main_task only runs when something raises. Matches the
            # Deepgram and Muse STT plugins.
            heartbeat=30.0,
            conn_options=self._conn_options,
        )

    def _process_stream_event(self, raw_data: object) -> None:
        data = _decode_transcript(raw_data)
        msg_type = data.type

        if msg_type == "transcript.created":
            self._server_ready.set()

        elif msg_type == "transcript.partial":
            text = data.text
            is_final = data.is_final
            speech_final = data.speech_final
            words = data.words
            language = data.language

            if not text:
                return

            if not self._speaking:
                self._speaking = True
                self.send(stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH))

            if is_final:
                if not speech_final:
                    self._emitted_chunk_final = True
                    self.send(
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
                        self.send(
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
                    self.send(stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH))
            else:
                if self._opts.enable_interim_results:
                    self.send(
                        stt.SpeechEvent(
                            type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                            request_id=self._request_id,
                            alternatives=[
                                stt.SpeechData(language=LanguageCode(language), text=text)
                            ],
                        )
                    )

        elif msg_type == "transcript.done":
            text = data.text
            words = data.words
            language = data.language
            if text:
                self.send(
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
                self.send(stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH))

        elif msg_type == "error":
            raise APIConnectionError("xAI STT error", detail=data.message)

        else:
            logger.warning("received unexpected message from xAI: %s", msg_type)


def _words_to_speech_data(
    words: list[_Word], text: str, *, language: str, enable_diarization: bool
) -> stt.SpeechData:
    speaker_id = (
        f"S{words[0].speaker}"
        if enable_diarization and words and words[0].speaker is not None
        else None
    )
    return stt.SpeechData(
        language=LanguageCode(language),
        text=text,
        start_time=words[0].start if words else 0.0,
        end_time=words[-1].end if words else 0.0,
        speaker_id=speaker_id,
        words=[TranscriptChunk(w.text, start_time=w.start, end_time=w.end) for w in words] or None,
    )


def _prerecorded_transcription_to_speech_event(
    raw_data: object, *, enable_diarization: bool
) -> stt.SpeechEvent:
    data = _decode_transcript(raw_data)
    text = data.text
    words = data.words
    language = data.language
    return stt.SpeechEvent(
        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
        alternatives=[
            _words_to_speech_data(
                words, text, language=language, enable_diarization=enable_diarization
            )
        ],
    )
