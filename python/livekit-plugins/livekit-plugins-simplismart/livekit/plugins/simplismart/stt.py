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

"""Speech-to-Text implementation for SimpliSmart

This module provides an STT implementation that uses the SimpliSmart API.
"""

import asyncio
import base64
import os
from collections.abc import Callable
from typing import Any, Literal, cast
from urllib.parse import urlparse

from pydantic import BaseModel

from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    LanguageCode,
    http,
    stt,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from .log import logger
from .models import STTModels

SIMPLISMART_BASE_URL = "https://api.simplismart.live/predict"


class SimplismartSTTOptions(BaseModel):
    language: LanguageCode | None = None
    task: Literal["transcribe", "translate"] = "transcribe"
    without_timestamps: bool = True
    vad_model: Literal["silero", "frame"] = "frame"
    vad_filter: bool = True
    vad_onset: float | None = 0.5
    vad_offset: float | None = None
    min_speech_duration_ms: int = 0
    max_speech_duration_s: float = 30
    min_silence_duration_ms: int = 2000
    speech_pad_ms: int = 400
    initial_prompt: str | None = None
    hotwords: str | None = None
    num_speakers: int = 0
    compression_ratio_threshold: float | None = 2.4
    beam_size: int = 4
    temperature: float = 0.0
    multilingual: bool = False
    max_tokens: float | None = 400
    log_prob_threshold: float | None = -1.0
    length_penalty: int = 1
    repetition_penalty: float = 1.01
    strict_hallucination_reduction: bool = False


class STT(stt.STT):
    def __init__(
        self,
        *,
        base_url: str = SIMPLISMART_BASE_URL,
        api_key: str | None = None,
        streaming: bool = False,
        model: STTModels | str = "openai/whisper-large-v3-turbo",
        language: str = "en",
        task: Literal["transcribe", "translate"] = "transcribe",
        without_timestamps: bool = True,
        vad_model: Literal["silero", "frame"] = "frame",
        vad_filter: bool = True,
        vad_onset: float | None = 0.5,
        vad_offset: float | None = None,
        min_speech_duration_ms: int = 0,
        max_speech_duration_s: float = 30,
        min_silence_duration_ms: int = 2000,
        speech_pad_ms: int = 400,
        initial_prompt: str | None = None,
        hotwords: str | None = None,
        num_speakers: int = 0,
        compression_ratio_threshold: float | None = 2.4,
        beam_size: int = 4,
        temperature: float = 0.0,
        multilingual: bool = False,
        max_tokens: float | None = 400,
        log_prob_threshold: float | None = -1.0,
        length_penalty: int = 1,
        repetition_penalty: float = 1.01,
        strict_hallucination_reduction: bool = False,
        http_session: http.HttpSession | None = None,
    ):
        """
        Configuration options for the SimpliSmart STT (Speech-to-Text) engine.

        Note:
            Streaming transcription is not publicly available at this time.

        Args:
            language (str): Language code for transcription (default: "en").
            task (Literal["transcribe", "translate"]): Operation to perform, either "transcribe" or "translate".
            model (STTModels | str): Model identifier for the backend STT model.
            without_timestamps (bool): If True, disables timestamp generation in transcripts.
            vad_model (Literal["silero", "frame"]): Voice Activity Detection model to use ("silero" or "frame").
            vad_filter (bool): Whether to apply VAD to filter input audio.
            vad_onset (float | None): Time (in seconds) for VAD onset boundary.
            vad_offset (float | None): Time (in seconds) for VAD offset boundary.
            min_speech_duration_ms (int): Minimum duration (ms) for a valid speech segment.
            max_speech_duration_s (float): Maximum speech segment duration (seconds).
            min_silence_duration_ms (int): Minimum silence duration (ms) to split speech.
            speech_pad_ms (int): Padding (ms) added to boundaries of detected speech.
            initial_prompt (str | None): Optional initial prompt for contextual biasing.
            hotwords (str | None): Comma-separated list of hotwords to bias recognition.
            num_speakers (int): Number of speakers for diarization.
            compression_ratio_threshold (float | None): Threshold for output compression ratio.
            beam_size (int): Beam size for the decoder.
            temperature (float): Decoding temperature (affects randomness).
            multilingual (bool): Whether to permit multilingual recognition.
            max_tokens (float | None): Maximum number of output tokens for the model.
            log_prob_threshold (float | None): Log probability threshold for word filtering.
            length_penalty (int): Penalty for longer transcriptions.
            repetition_penalty (float): Penalty for repeated words during decoding.
            strict_hallucination_reduction (bool): Whether to apply hallucination reduction.
        """
        if streaming:
            base_url = f"wss://{urlparse(base_url).netloc}/ws/audio"

        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=streaming,
                interim_results=False,
                aligned_transcript="word",
            )
        )

        self._api_key = api_key or os.environ.get("SIMPLISMART_API_KEY")
        if not self._api_key:
            raise ValueError("SIMPLISMART_API_KEY is not set")

        self._model = model
        self._opts = SimplismartSTTOptions(
            language=LanguageCode(language),
            task=task,
            without_timestamps=without_timestamps,
            vad_model=vad_model,
            vad_filter=vad_filter,
            vad_onset=vad_onset,
            vad_offset=vad_offset,
            min_speech_duration_ms=min_speech_duration_ms,
            max_speech_duration_s=max_speech_duration_s,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
            initial_prompt=initial_prompt,
            hotwords=hotwords,
            num_speakers=num_speakers,
            compression_ratio_threshold=compression_ratio_threshold,
            beam_size=beam_size,
            temperature=temperature,
            multilingual=multilingual,
            max_tokens=max_tokens,
            log_prob_threshold=log_prob_threshold,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            strict_hallucination_reduction=strict_hallucination_reduction,
        )
        self._base_url = base_url
        self._session = http_session
        # Strong ownership: a stream that finishes and gets garbage collected would
        # otherwise take its still-open per-stream HttpSession with it.
        # Streams discard themselves in SpeechStream.aclose() once closed.
        self._streams: set[SpeechStream] = set()

    @property
    def provider(self) -> str:
        return "Simplismart"

    @property
    def model(self) -> str:
        return self._model

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
        resolved_language: str | None = language if is_given(language) else self._opts.language
        wav_bytes = rtc.combine_audio_frames(buffer).to_wav_bytes()

        audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")
        payload = self._opts.model_dump()

        payload["audio_data"] = audio_b64
        payload["language"] = resolved_language
        payload["model"] = self._model

        try:
            async with self._ensure_session().post(
                self._base_url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                conn_options=conn_options,
            ) as res:
                await res.raise_for_status()

                response_json = cast(dict[str, Any], await res.json())
                timestamps = response_json.get("timestamps", [])
                transcription = response_json.get("transcription", [])

                info = response_json.get("info", {})
                detected_language = LanguageCode(info.get("language", resolved_language or "en"))

                start_time = timestamps[0][0] if timestamps else 0.0
                end_time = timestamps[-1][1] if timestamps else 0.0
                request_id = response_json.get("request_id", "")
                text = "".join(transcription)

                alternatives = [
                    stt.SpeechData(
                        language=detected_language,
                        text=text,
                        start_time=start_time,
                        end_time=end_time,
                    ),
                ]

                return stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    request_id=request_id,
                    alternatives=alternatives,
                )
        except APIError:
            raise
        except Exception as e:
            raise APIConnectionError("Unexpected error in Simplismart STT", detail=str(e)) from e

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        **kwargs: object,
    ) -> "SpeechStream":
        """Create a streaming transcription session."""
        opts_language = LanguageCode(language) if is_given(language) else self._opts.language

        # Create options for the stream
        stream_opts = SimplismartSTTOptions(language=opts_language)

        # Create a fresh session for this stream to avoid conflicts
        stream_session = http.HttpSession()

        if self._api_key is None:
            raise ValueError("API key cannot be None")
        stream = SpeechStream(
            stt=self,
            opts=stream_opts,
            conn_options=conn_options,
            api_key=self._api_key,
            http_session=stream_session,
            base_url=self._base_url,
            on_closed=self._streams.discard,
        )
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        """Close every stream this instance created.

        ``stream()`` gives each ``SpeechStream`` its own ``HttpSession``,
        and only ``SpeechStream.aclose()`` closes it. Streams are owned strongly
        (see ``self._streams``) so a finished, garbage-collected stream cannot take
        its open session with it; closed streams discard themselves, so the set
        never grows without bound.
        """
        closed = list(self._streams)
        for stream in closed:
            await stream.aclose()
        # Remove only what this call closed: a stream created concurrently (or one
        # whose aclose was cancelled mid-cleanup) stays tracked for a later close.
        self._streams.difference_update(closed)


class SpeechStream(stt.RecognizeStream):
    """Simplismart streaming speech-to-text implementation."""

    _CHUNK_DURATION_MS = 50
    _SAMPLE_RATE = 16000

    def __init__(
        self,
        *,
        stt: STT,
        opts: SimplismartSTTOptions,
        conn_options: APIConnectOptions,
        api_key: str,
        http_session: http.HttpSession,
        base_url: str,
        on_closed: Callable[["SpeechStream"], None],
    ) -> None:
        self._opts = opts
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=self._SAMPLE_RATE)
        self._api_key = api_key
        self._session = http_session
        self._request_id = str(id(self))
        self.ws_url = base_url
        self._on_closed = on_closed

    async def _run(self) -> None:
        async def send_task(ws: http.WebSocket) -> None:
            # forward audio to simplismart in chunks of 50ms
            samples_50ms = self._SAMPLE_RATE // 20
            audio_bstream = utils.media.AudioByteStream(
                sample_rate=self._SAMPLE_RATE,
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

        async def recv_task(ws: http.WebSocket) -> None:
            while True:
                try:
                    msg = await ws.receive()
                except APIConnectionError:
                    if self._session.closed:
                        return
                    raise
                if msg.type == http.WebSocketMessageType.CLOSE:
                    # the agent session ending closes the http session
                    if self._session.closed:
                        return
                    raise ws.close_error("Simplismart STT connection closed")
                if msg.type != http.WebSocketMessageType.BINARY:
                    logger.warning("unexpected simplismart message type %s", msg.type)
                    continue
                try:
                    transcript = msg.binary.decode("utf-8")
                except UnicodeDecodeError as e:
                    raise APIError(
                        "Simplismart STT sent a malformed message",
                        kind="protocol",
                        detail=repr(msg.binary),
                    ) from e
                self._handle_transcript_data(transcript)

        ws = await self._connect_ws()
        try:
            await ws.send_json({"language": self._opts.language})
            async with utils.aio.TaskSet() as tasks:
                await asyncio.gather(tasks.spawn(send_task(ws)), tasks.spawn(recv_task(ws)))
        finally:
            await ws.close()

    async def _connect_ws(self) -> http.WebSocket:
        return await self._session.ws_connect(
            self.ws_url,
            headers={"Authorization": f"Bearer {self._api_key}"},
            # notices a silently dropped socket (half-open TCP, no FIN/RST)
            heartbeat=30.0,
            conn_options=self._conn_options,
        )

    async def aclose(self) -> None:
        await super().aclose()
        await self._session.aclose()
        # Release the STT's strong ownership only after session cleanup succeeds.
        self._on_closed(self)

    def _handle_transcript_data(self, transcript: str) -> None:
        self.send(
            stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                request_id=self._request_id,
                alternatives=[
                    stt.SpeechData(
                        language=LanguageCode(self._opts.language or "en"), text=transcript
                    )
                ],
            )
        )
