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

import dataclasses
import os
import uuid
import weakref
from dataclasses import dataclass
from typing import Annotated, Literal, NotRequired, TypedDict
from urllib.parse import urlencode

from pydantic import Field, TypeAdapter, ValidationError

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
)
from livekit.agents.types import (
    NOT_GIVEN,
    NotGivenOr,
    TranscriptChunk,
)
from livekit.agents.utils import is_given

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


class _Word(TypedDict, total=False):
    text: str
    start: float
    end: float
    speaker: int | str


class _Result(TypedDict, total=False):
    """A batch transcript, and the fields of a streamed one."""

    text: str
    language: str
    words: list[_Word]


_RESULT = TypeAdapter(_Result)


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


class _Created(TypedDict):
    type: Literal["transcript.created"]


class _Partial(_Result):
    type: Literal["transcript.partial", "transcript.done"]
    is_final: NotRequired[bool]
    speech_final: NotRequired[bool]


class _Error(TypedDict):
    type: Literal["error"]
    message: NotRequired[str]


_Message = Annotated[_Created | _Partial | _Error, Field(discriminator="type")]


class _AudioDone(TypedDict):
    type: Literal["audio.done"]


class SpeechStream(stt.SocketStream[_Message, _AudioDone]):
    goodbye = _AudioDone(type="audio.done")
    frame_ms = 50
    # audio waits for transcript.created
    awaits_ready = True
    # speech_final is xAI's own end of the utterance
    provider_vad = True

    def __init__(
        self, *, stt: STT, opts: STTOptions, conn_options: APIConnectOptions, api_key: str
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)
        self._opts = opts
        self._api_key = api_key
        self._request_id = str(uuid.uuid4())
        # the utterance's text already went out as chunk finals: its speech_final repeats it
        self._emitted_chunk_final = False

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

        self.reconnect("options changed")

    async def connect(self, conn_options: APIConnectOptions) -> http.SocketTarget:
        self._emitted_chunk_final = False
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
        url = XAI_WEBSOCKET_URL.replace("http", "ws", 1)
        return http.SocketTarget(
            f"{url}?{urlencode(params)}",
            headers={"Authorization": f"Bearer {self._api_key}"},
        )

    def encode(
        self, item: rtc.AudioFrame | stt.RecognizeStream.Flush | http.EndOfInput
    ) -> http.Encoded[_AudioDone]:
        return bytes(item.data) if isinstance(item, rtc.AudioFrame) else None

    def on_message(
        self, message: _Created | _Partial | _Error
    ) -> stt.Report | list[stt.Report] | http.Ready | http.Done | None:
        if message["type"] == "transcript.created":
            return http.READY
        if message["type"] == "error":
            raise APIError("xAI STT error", kind="invalid_request", detail=message.get("message"))

        text = message.get("text", "")
        if message["type"] == "transcript.done":
            # the whole stream is transcribed
            if text:
                self.report(self._transcript(message, final=True))
            self.report(stt.SpeechEnded())
            return http.DONE

        if not text:
            return None
        if not message.get("is_final", False):
            if not self._opts.enable_interim_results:
                return None
            return stt.Transcript(text, self._language(message), request_id=self._request_id)
        if not message.get("speech_final", False):
            self._emitted_chunk_final = True
            return self._transcript(message, final=True)
        reports: list[stt.Report] = []
        if not self._emitted_chunk_final:
            reports.append(self._transcript(message, final=True))
        self._emitted_chunk_final = False
        return [*reports, stt.SpeechEnded()]

    def _language(self, message: _Partial) -> LanguageCode:
        return LanguageCode(message.get("language") or self._opts.language)

    def _transcript(self, message: _Partial, *, final: bool) -> stt.Transcript:
        data = _words_to_speech_data(
            message.get("words", []),
            message.get("text", ""),
            language=self._language(message),
            enable_diarization=self._opts.enable_diarization,
            offset=self.start_time_offset,
        )
        return stt.Transcript(
            data.text,
            data.language,
            final=final,
            request_id=self._request_id,
            start_time=data.start_time,
            end_time=data.end_time,
            speaker_id=data.speaker_id,
            words=data.words,
        )


def _words_to_speech_data(
    words: list[_Word],
    text: str,
    *,
    language: LanguageCode,
    enable_diarization: bool,
    offset: float = 0.0,
) -> stt.SpeechData:
    speaker = words[0].get("speaker") if enable_diarization and words else None
    return stt.SpeechData(
        language=language,
        text=text,
        start_time=words[0].get("start", 0.0) + offset if words else 0.0,
        end_time=words[-1].get("end", 0.0) + offset if words else 0.0,
        speaker_id=f"S{speaker}" if speaker is not None else None,
        words=[
            TranscriptChunk(
                w.get("text", ""),
                start_time=w.get("start", 0.0) + offset,
                end_time=w.get("end", 0.0) + offset,
                start_time_offset=offset,
            )
            for w in words
        ]
        or None,
    )


def _prerecorded_transcription_to_speech_event(
    raw_data: object, *, enable_diarization: bool
) -> stt.SpeechEvent:
    try:
        data = _RESULT.validate_python(raw_data)
    except ValidationError as error:
        raise APIError("invalid xAI transcript", kind="protocol", detail=str(raw_data)) from error
    return stt.SpeechEvent(
        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
        alternatives=[
            _words_to_speech_data(
                data.get("words", []),
                data.get("text", ""),
                language=LanguageCode(data.get("language", "")),
                enable_diarization=enable_diarization,
            )
        ],
    )
