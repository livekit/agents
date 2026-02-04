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
import contextlib
import json
import os
import traceback
import weakref
from typing import Any, Literal
from urllib.parse import urlparse

import aiohttp
from pydantic import BaseModel

from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    stt,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer
from livekit.agents.utils.misc import is_given

from .log import logger
from .models import STTModels

SIMPLISMART_BASE_URL = "https://api.simplismart.live/predict"


class SimplismartSTTOptions(BaseModel):
    language: str | None = None
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
        http_session: aiohttp.ClientSession | None = None,
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
            language=language,
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
        self._streams = weakref.WeakSet[SpeechStream]()

    @property
    def provider(self) -> str:
        return "Simplismart"

    @property
    def model(self) -> str:
        return self._model

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
                timeout=aiohttp.ClientTimeout(
                    total=conn_options.timeout,
                ),
            ) as res:
                if res.status != 200:
                    error_text = await res.text()
                    logger.error(f"Simplismart API error: {res.status} - {error_text}")
                    raise APIStatusError(
                        message=f"Simplismart API Error: {error_text}",
                        status_code=res.status,
                        request_id=None,
                        body=error_text,
                    )

                response_json = await res.json()
                timestamps = response_json.get("timestamps", [])
                transcription = response_json.get("transcription", [])

                info = response_json.get("info", {})
                detected_language = info.get("language", resolved_language or "en")

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
        except asyncio.TimeoutError as e:
            logger.error(f"Simplismart API timeout: {e}")
            raise APITimeoutError("Simplismart API request timed out") from e
        except aiohttp.ClientError as e:
            logger.error(f"Simplismart API client error: {e}")
            raise APIConnectionError(f"Simplismart API connection error: {e}") from e
        except APIStatusError:
            raise
        except Exception as e:
            logger.error(f"Error during Simplismart STT processing: {traceback.format_exc()}")
            raise APIConnectionError(f"Unexpected error in Simplismart STT: {e}") from e

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        **kwargs: Any,
    ) -> "SpeechStream":
        """Create a streaming transcription session."""
        opts_language = language if is_given(language) else self._opts.language

        # Create options for the stream
        stream_opts = SimplismartSTTOptions(language=opts_language)

        # Create a fresh session for this stream to avoid conflicts
        stream_session = aiohttp.ClientSession()

        if self._api_key is None:
            raise ValueError("API key cannot be None")
        stream = SpeechStream(
            stt=self,
            opts=stream_opts,
            conn_options=conn_options,
            api_key=self._api_key,
            http_session=stream_session,
        )
        self._streams.add(stream)
        return stream


class SpeechStream(stt.SpeechStream):
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
        http_session: aiohttp.ClientSession,
    ) -> None:
        self._opts = opts
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=self._SAMPLE_RATE)
        self._api_key = api_key
        self._session = http_session
        self._reconnect_event = asyncio.Event()
        self._request_id = str(id(self))
        self.ws_url = stt._base_url

    async def _run(self) -> None:
        @utils.log_exceptions(logger=logger)
        async def send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            # forward audio to simplismart in chunks of 50ms
            samples_50ms = self._SAMPLE_RATE // 20
            audio_bstream = utils.audio.AudioByteStream(
                sample_rate=self._SAMPLE_RATE,
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
                    await ws.send_bytes(frame.data.tobytes())

        @utils.log_exceptions(logger=logger)
        async def recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            while True:
                msg = await ws.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    # close is expected, see SpeechStream.aclose
                    # or when the agent session ends, the http session is closed
                    if self._session.closed:
                        return

                    # this will trigger a reconnection, see the _run loop
                    self._reconnect_event.set()
                    return

                if msg.type != aiohttp.WSMsgType.BINARY:
                    logger.warning("unexpected simplismart message type %s", msg.type)
                    continue

                try:
                    self._handle_transcript_data(msg.data.decode("utf-8"))
                except Exception:
                    logger.exception("failed to process simplismart message")

        ws: aiohttp.ClientWebSocketResponse | None = None

        while True:
            try:
                ws = await self._connect_ws()
                await self._send_initial_config(ws)
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

                    # propagate exceptions from completed tasks
                    for task in done:
                        if task != wait_reconnect_task:
                            task.result()

                    if wait_reconnect_task not in done:
                        break

                    self._reconnect_event.clear()
                finally:
                    await utils.aio.gracefully_cancel(*tasks, wait_reconnect_task)
                    tasks_group.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        tasks_group.exception()  # retrieve the exception
            finally:
                if ws is not None:
                    await ws.close()

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        try:
            ws = await asyncio.wait_for(
                self._session.ws_connect(
                    self.ws_url,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                ),
                self._conn_options.timeout,
            )
        except (aiohttp.ClientConnectorError, asyncio.TimeoutError) as e:
            raise APIConnectionError("failed to connect to simplismart") from e
        return ws

    async def aclose(self) -> None:
        await super().aclose()
        if self._session and not self._session.closed:
            await self._session.close()

    async def _send_initial_config(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        """Send initial configuration message with language for Simplismart models."""
        try:
            config_message = {"language": self._opts.language}
            await ws.send_json(config_message)
            logger.info(
                "Sent initial config for Simplismart model",
                extra={"request_id": self._request_id, "language": self._opts.language},
            )
        except Exception as e:
            logger.error(
                f"Failed to send initial configuration: {e}",
                extra={"request_id": self._request_id},
                exc_info=True,
            )
            raise APIConnectionError(f"Failed to send initial config: {e}") from e

    def _handle_transcript_data(self, data: str) -> None:
        """Handle transcription result messages."""
        transcript_text = data
        request_id = self._request_id

        try:
            # Create usage event with proper metrics extraction
            metrics: dict[str, float] = {}
            request_data = {
                "original_id": request_id,
                "processing_latency": metrics.get("processing_latency", 0.0),
            }
            usage_event = stt.SpeechEvent(
                type=stt.SpeechEventType.RECOGNITION_USAGE,
                request_id=json.dumps(request_data),
                recognition_usage=stt.RecognitionUsage(
                    audio_duration=metrics.get("audio_duration", 0.0),
                ),
            )
            self._event_ch.send_nowait(usage_event)

            # Create speech data
            speech_data = stt.SpeechData(
                language=self._opts.language or "en",
                text=transcript_text,
            )

            # Create final transcript event with request_id
            speech_event = stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                request_id=request_id,
                alternatives=[speech_data],
            )
            self._event_ch.send_nowait(speech_event)

            logger.debug(
                "Transcript processed successfully",
                extra={
                    "request_id": self._request_id,
                    "text_length": len(transcript_text),
                    "language": self._opts.language,
                    "confidence": speech_data.confidence,
                },
            )

        except Exception as e:
            logger.error(
                f"Error processing transcript data: {e}",
                extra={
                    "request_id": self._request_id,
                    "transcript_text": transcript_text,
                },
                exc_info=True,
            )
            raise
