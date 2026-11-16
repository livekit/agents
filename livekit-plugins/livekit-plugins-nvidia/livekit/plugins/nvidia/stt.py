# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

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

import asyncio
import logging
import os
import queue
import threading
from collections import Counter
from collections.abc import Generator
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Literal

import riva.client
from riva.client.proto.riva_asr_pb2 import (
    RivaSpeechRecognitionConfigRequest,
    SpeakerDiarizationConfig,
)

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APITimeoutError,
    LanguageCode,
    stt,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, is_given
from livekit.agents.voice.io import TimedString

from . import auth
from .utils import to_api_error

logger = logging.getLogger(__name__)


EndpointingMode = Literal["low_latency"]
InferenceMode = Literal["auto", "streaming", "offline"]


@dataclass(frozen=True)
class EndpointingConfig:
    """Configure NVIDIA Speech endpoint detection for streaming recognition.

    Attributes:
        mode: Endpointing preset. ``"low_latency"`` applies the plugin's low-latency
            defaults before any field overrides.
        start_history: VAD start-history override in audio frames.
        start_threshold: Dimensionless VAD speech-start threshold override.
        stop_history: VAD stop-history override in audio frames.
        stop_threshold: Dimensionless VAD speech-stop threshold override.
        stop_history_eou: End-of-utterance stop-history override in audio frames.
        stop_threshold_eou: Dimensionless end-of-utterance stop-threshold override.

    ``None`` leaves a field at the value selected by ``mode``. Values of ``-1`` or
    ``-1.0`` ask NVIDIA Speech to use its model default for that field.
    """

    mode: EndpointingMode = "low_latency"
    start_history: int | None = None
    start_threshold: float | None = None
    stop_history: int | None = None
    stop_threshold: float | None = None
    stop_history_eou: int | None = None
    stop_threshold_eou: float | None = None


@dataclass
class STTOptions:
    model: str
    function_id: str
    punctuate: bool
    language_code: LanguageCode
    sample_rate: int
    use_ssl: bool
    server: str
    enable_diarization: bool
    max_speaker_count: int
    interim_results: bool
    profanity_filter: bool
    verbatim_transcripts: bool
    boosted_lm_words: list[str] | None
    boosted_lm_score: float
    endpointing: EndpointingConfig | None
    options: dict[str, Any]
    enable_word_time_offsets: bool
    inference_mode: InferenceMode


class STT(stt.STT):
    def __init__(
        self,
        *,
        model: str = "parakeet-1.1b-en-US-asr-streaming-silero-vad-sortformer",
        function_id: str = "1598d209-5e27-4d3c-8079-4751568b1081",
        punctuate: bool = True,
        language_code: str = "en-US",
        sample_rate: int = 16000,
        server: str = "grpc.nvcf.nvidia.com:443",
        use_ssl: bool = True,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        enable_diarization: bool = False,
        max_speaker_count: int = 0,
        interim_results: bool = True,
        profanity_filter: bool = False,
        verbatim_transcripts: bool = False,
        boosted_lm_words: list[str] | None = None,
        boosted_lm_score: float = 4.0,
        endpointing: EndpointingConfig | None = None,
        options: dict[str, Any] | None = None,
        enable_word_time_offsets: bool = True,
        inference_mode: InferenceMode = "auto",
    ):
        """Create an NVIDIA Speech recognition client.

        Args:
            model: NVIDIA Speech model name sent with recognition requests.
            function_id: NVIDIA-hosted NVCF function ID. Local deployments may use
                the value expected by their gateway.
            punctuate: Whether to enable automatic punctuation.
            language_code: Recognition language code, such as ``"en-US"``.
            sample_rate: Streaming input sample rate in Hz.
            server: NVIDIA Speech gRPC endpoint.
            use_ssl: Whether to use TLS for the gRPC connection.
            api_key: NVIDIA API key. When omitted, reads ``NVIDIA_API_KEY``.
            enable_diarization: Whether to request speaker diarization.
            max_speaker_count: Maximum expected speakers when diarization is enabled.
            interim_results: Whether streaming recognition emits interim transcripts.
            profanity_filter: Whether to mask profane words in transcripts.
            verbatim_transcripts: Whether to request verbatim transcript formatting.
            boosted_lm_words: Words or phrases to boost during recognition.
            boosted_lm_score: Global boost score applied to ``boosted_lm_words``.
            endpointing: Streaming endpoint-detection settings. ``None`` leaves the
                deployed model's endpointing configuration unchanged.
            options: Provider-specific escape hatch. ``recognition_config`` values are
                passed to ``riva.client.RecognitionConfig``; ``custom_configuration``
                is applied to the streaming configuration when supported by the client.
            enable_word_time_offsets: Whether to request word timestamps.
            inference_mode: ``"streaming"`` or ``"offline"`` advertises only that
                capability; ``"auto"`` preserves support for both APIs.

        Raises:
            ValueError: If the inference mode is invalid or hosted authentication is
                enabled without an API key.
        """
        if inference_mode not in ("auto", "streaming", "offline"):
            raise ValueError("inference_mode must be 'auto', 'streaming', or 'offline'")

        if enable_diarization and "sortformer" not in model.lower():
            logger.warning(
                "Speaker diarization is enabled but model '%s' may not support it. "
                "Diarization requires a Sortformer-based model.",
                model,
            )

        supports_streaming = inference_mode in ("auto", "streaming")
        supports_offline = inference_mode in ("auto", "offline")
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=supports_streaming,
                interim_results=interim_results if supports_streaming else False,
                diarization=enable_diarization,
                aligned_transcript="word" if enable_word_time_offsets else False,
                offline_recognize=supports_offline,
            ),
        )

        if is_given(api_key):
            self.nvidia_api_key = api_key
        else:
            self.nvidia_api_key = os.getenv("NVIDIA_API_KEY", "")
            if use_ssl and not self.nvidia_api_key:
                raise ValueError(
                    "NVIDIA_API_KEY is not set while using SSL. Either pass api_key parameter, "
                    "set NVIDIA_API_KEY environment variable or disable SSL and use a locally "
                    "hosted NVIDIA Speech service."
                )

        self._opts = STTOptions(
            model=model,
            function_id=function_id,
            punctuate=punctuate,
            language_code=LanguageCode(language_code),
            sample_rate=sample_rate,
            server=server,
            use_ssl=use_ssl,
            enable_diarization=enable_diarization,
            max_speaker_count=max_speaker_count,
            interim_results=interim_results,
            profanity_filter=profanity_filter,
            verbatim_transcripts=verbatim_transcripts,
            boosted_lm_words=boosted_lm_words,
            boosted_lm_score=boosted_lm_score,
            endpointing=endpointing,
            options=dict(options or {}),
            enable_word_time_offsets=enable_word_time_offsets,
            inference_mode=inference_mode,
        )
        self._asr_service: riva.client.ASRService | None = None

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "NVIDIA Speech"

    def _ensure_asr_service(self) -> riva.client.ASRService:
        if self._asr_service is None:
            riva_auth = auth.create_riva_auth(
                api_key=self.nvidia_api_key,
                function_id=self._opts.function_id,
                server=self._opts.server,
                use_ssl=self._opts.use_ssl,
            )
            self._asr_service = riva.client.ASRService(riva_auth)
        return self._asr_service

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        if self._opts.inference_mode == "streaming":
            raise ValueError(
                "recognize() requires inference_mode='offline'; use stream() for streaming models"
            )

        effective_language = (
            LanguageCode(language) if is_given(language) else self._opts.language_code
        )
        frame = rtc.combine_audio_frames(buffer)
        config = self._create_recognition_config(
            language=effective_language,
            sample_rate=frame.sample_rate,
            audio_channel_count=frame.num_channels,
        )
        service = self._ensure_asr_service()

        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(service.offline_recognize, frame.data.tobytes(), config),
                timeout=conn_options.timeout,
            )
        except asyncio.TimeoutError:
            raise APITimeoutError("NVIDIA Speech STT request timed out") from None
        except Exception as e:
            raise to_api_error(e, operation="NVIDIA Speech offline STT request") from e

        return _response_to_speech_event(
            response,
            language=effective_language,
            request_id=f"nvidia-{id(response)}",
            event_type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            enable_diarization=self._opts.enable_diarization,
            is_final=True,
        )

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.RecognizeStream:
        if self._opts.inference_mode == "offline":
            raise ValueError(
                "stream() requires inference_mode='streaming'; use recognize() for offline models"
            )

        effective_language = (
            LanguageCode(language) if is_given(language) else self._opts.language_code
        )
        return SpeechStream(stt=self, conn_options=conn_options, language=effective_language)

    def _create_recognition_config(
        self,
        *,
        language: LanguageCode,
        sample_rate: int,
        audio_channel_count: int,
    ) -> riva.client.RecognitionConfig:
        recognition_kwargs = {
            "encoding": riva.client.AudioEncoding.LINEAR_PCM,
            "language_code": language,
            "model": self._opts.model,
            "max_alternatives": 1,
            "profanity_filter": self._opts.profanity_filter,
            "enable_automatic_punctuation": self._opts.punctuate,
            "verbatim_transcripts": self._opts.verbatim_transcripts,
            "sample_rate_hertz": sample_rate,
            "audio_channel_count": audio_channel_count,
            "enable_word_time_offsets": self._opts.enable_word_time_offsets,
        }
        recognition_kwargs.update(self._opts.options.get("recognition_config", {}))
        recognition_config = riva.client.RecognitionConfig(**recognition_kwargs)

        if self._opts.enable_diarization:
            diarization_config = SpeakerDiarizationConfig(
                enable_speaker_diarization=True,
                max_speaker_count=self._opts.max_speaker_count,
            )
            recognition_config.diarization_config.CopyFrom(diarization_config)

        if self._opts.boosted_lm_words:
            riva.client.add_word_boosting_to_config(
                recognition_config,
                self._opts.boosted_lm_words,
                self._opts.boosted_lm_score,
            )

        return recognition_config

    def _apply_streaming_config_extensions(
        self,
        streaming_config: riva.client.StreamingRecognitionConfig,
    ) -> None:
        endpointing_values = _endpointing_values(self._opts.endpointing)
        if endpointing_values is not None:
            riva.client.add_endpoint_parameters_to_config(
                streaming_config,
                *endpointing_values,
            )

        custom_configuration = self._opts.options.get("custom_configuration", "")
        if custom_configuration and hasattr(riva.client, "add_custom_configuration_to_config"):
            riva.client.add_custom_configuration_to_config(
                streaming_config,
                custom_configuration,
            )
        elif custom_configuration:
            raise ValueError(
                "custom_configuration is not supported by the installed nvidia-riva-client "
                "version. Remove options['custom_configuration'] or install a version "
                "that exposes add_custom_configuration_to_config."
            )

    def log_asr_models(self, asr_service: riva.client.ASRService) -> dict:
        config_response = asr_service.stub.GetRivaSpeechRecognitionConfig(
            RivaSpeechRecognitionConfigRequest()
        )

        asr_models = {}
        for model_config in config_response.model_config:
            model_type = model_config.parameters.get("type")
            if self._opts.inference_mode == "auto" or model_type == (
                "online" if self._opts.inference_mode == "streaming" else "offline"
            ):
                language_code = model_config.parameters["language_code"]
                model = {"model": [model_config.model_name]}
                if language_code in asr_models:
                    asr_models[language_code].append(model)
                else:
                    asr_models[language_code] = [model]

        asr_models = dict(sorted(asr_models.items()))
        return asr_models


@dataclass
class _RecognitionAttempt:
    """One streaming RPC. NVIDIA counts its timestamps from the first audio it receives."""

    # Transport to the SDK's blocking request generator, not retained after it is read.
    audio_queue: queue.Queue[bytes | None]
    # Completes after the worker scheduled its last response on the event loop.
    done_fut: asyncio.Future[None]
    thread: threading.Thread | None = None
    input_closed: bool = False
    received_final_transcript: bool = False
    # Set when the stream stops waiting for the worker; its late responses are dropped.
    abandoned: bool = False


def _close_attempt_input(attempt: _RecognitionAttempt) -> None:
    if not attempt.input_closed:
        attempt.input_closed = True
        attempt.audio_queue.put(None)


async def _join_attempt(attempt: _RecognitionAttempt) -> None:
    if attempt.thread is None or not attempt.thread.is_alive():
        return
    await asyncio.to_thread(attempt.thread.join, 1.0)
    if attempt.thread.is_alive():
        logger.warning("NVIDIA Speech recognition worker did not stop within one second")


class SpeechStream(stt.SpeechStream):
    """Recognize each flushed segment with one streaming RPC.

    An RPC starts at the first frame read after the previous RPC ended, so its
    timestamps count from the audio that restarts the stream's connection clock.
    Audio is not replayed: when an RPC ends, audio it received without returning a
    final transcript is lost, and unread input stays queued for the next RPC.
    """

    def __init__(self, *, stt: STT, conn_options: APIConnectOptions, language: str):
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=stt._opts.sample_rate)
        self._stt = stt
        self._language = language

        self._speaking = False
        self._request_id = ""

        self._asr_service = self._stt._ensure_asr_service()

    async def _run(self) -> None:
        config = self._create_streaming_config()
        event_loop = asyncio.get_running_loop()
        attempt: _RecognitionAttempt | None = None
        input_aiter = self._input_ch.__aiter__()
        input_task = asyncio.create_task(anext(input_aiter))

        try:
            while True:
                waiters: set[asyncio.Future[Any]] = {input_task}
                if attempt is not None:
                    waiters.add(attempt.done_fut)
                await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)

                # Handle input read before the RPC ended first: it was read on that
                # RPC's clock. The read in progress is left for the next RPC.
                if input_task.done():
                    try:
                        data = input_task.result()
                    except StopAsyncIteration:
                        break
                    if isinstance(data, rtc.AudioFrame):
                        audio_bytes = data.data.tobytes()
                        if audio_bytes:
                            if attempt is None:
                                attempt = self._start_recognition_attempt(config, event_loop)
                            if not attempt.done_fut.done():
                                attempt.audio_queue.put(audio_bytes)
                    elif isinstance(data, self._FlushSentinel) and attempt is not None:
                        flushed, attempt = attempt, None
                        await self._finish_recognition_attempt(flushed, flushed=True)
                    input_task = asyncio.create_task(anext(input_aiter))

                if attempt is not None and attempt.done_fut.done():
                    # Leave unread audio queued while the old RPC is joined.
                    input_task.cancel()
                    with suppress(asyncio.CancelledError, StopAsyncIteration):
                        await input_task
                    ended, attempt = attempt, None
                    await self._finish_recognition_attempt(ended, flushed=False)
                    input_task = asyncio.create_task(anext(input_aiter))
        except APIError as e:
            if self._input_ch.closed and self._input_ch.empty():
                # No input is left for a retry: it would report an empty flush as success.
                e.retryable = False
            raise
        finally:
            # A cancelled read leaves its item in the channel.
            input_task.cancel()
            with suppress(asyncio.CancelledError, StopAsyncIteration):
                await input_task
            if attempt is not None:
                # No clock restart happens before this: late responses still belong to it.
                _close_attempt_input(attempt)
                with suppress(asyncio.CancelledError, asyncio.TimeoutError, Exception):
                    await asyncio.wait_for(asyncio.shield(attempt.done_fut), timeout=1.0)
                await _join_attempt(attempt)
                attempt.abandoned = True

    def _start_recognition_attempt(
        self,
        config: riva.client.StreamingRecognitionConfig,
        event_loop: asyncio.AbstractEventLoop,
    ) -> _RecognitionAttempt:
        attempt = _RecognitionAttempt(
            audio_queue=queue.Queue(),
            done_fut=event_loop.create_future(),
        )
        recognition_thread = threading.Thread(
            target=self._recognition_worker,
            args=(config, attempt, event_loop),
            name="nvidia-asr-recognition",
            daemon=True,
        )
        attempt.thread = recognition_thread
        recognition_thread.start()
        return attempt

    async def _finish_recognition_attempt(
        self, attempt: _RecognitionAttempt, *, flushed: bool
    ) -> None:
        """End an RPC and restart the connection clock for the next one.

        ``flushed`` ends the RPC at a flush. Otherwise the RPC ended on its own, which
        is an error unless it returned a final transcript first.

        Raises:
            APIError: The RPC failed, or ended on its own without a final transcript.
        """
        _close_attempt_input(attempt)
        try:
            # The worker scheduled every response before completing this future, so
            # all of them are on the event channel once it is done.
            await attempt.done_fut
            if not flushed and not attempt.received_final_transcript:
                raise APIConnectionError("NVIDIA Speech streaming STT ended unexpectedly")
        finally:
            await _join_attempt(attempt)
            # Reached without the worker's completion only when cancelled.
            attempt.abandoned = True
            self._finish_speech_segment()
            # The next frame read starts the next RPC and its clock.
            self._restart_connection_clock()

    def _finish_speech_segment(self) -> None:
        if self._speaking:
            self._speaking = False
            self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH))
        self._request_id = ""

    def _create_streaming_config(self) -> riva.client.StreamingRecognitionConfig:
        recognition_config = self._stt._create_recognition_config(
            language=LanguageCode(self._language),
            sample_rate=self._stt._opts.sample_rate,
            audio_channel_count=1,
        )

        streaming_config = riva.client.StreamingRecognitionConfig(
            config=recognition_config,
            interim_results=self._stt._opts.interim_results,
        )
        self._stt._apply_streaming_config_extensions(streaming_config)
        return streaming_config

    def _recognition_worker(
        self,
        config: riva.client.StreamingRecognitionConfig,
        attempt: _RecognitionAttempt,
        event_loop: asyncio.AbstractEventLoop,
    ) -> None:
        """Run the blocking RPC, handing each response to the event loop in arrival order."""
        error: Exception | None = None
        try:
            audio_generator = self._audio_chunk_generator(attempt.audio_queue)

            response_generator = self._asr_service.streaming_response_generator(
                audio_generator, config
            )

            for response in response_generator:
                event_loop.call_soon_threadsafe(self._handle_response, attempt, response)

        except Exception as e:
            error = e
        finally:
            event_loop.call_soon_threadsafe(self._complete_done_future, attempt.done_fut, error)

    def _complete_done_future(
        self, done_fut: asyncio.Future[None], error: Exception | None
    ) -> None:
        if done_fut.done():
            return
        if error is not None:
            done_fut.set_exception(
                to_api_error(error, operation="NVIDIA Speech streaming STT request")
            )
        else:
            done_fut.set_result(None)

    def _audio_chunk_generator(
        self, audio_queue: queue.Queue[bytes | None]
    ) -> Generator[bytes, None, None]:
        """
        The NVIDIA Speech SDK requires a generator for realtime STT, so the LiveKit
        async input channel is bridged through a thread-safe queue.
        """
        while True:
            audio_chunk = audio_queue.get()

            if not audio_chunk:
                break

            yield audio_chunk

    def _handle_response(self, attempt: _RecognitionAttempt, response: Any) -> None:
        """Emit a response's events. Runs on the stream's event loop."""
        if attempt.abandoned:
            # A later RPC may own the connection clock by now.
            return
        if not hasattr(response, "results") or not response.results:
            return

        self._request_id = f"nvidia-{id(response)}"

        for result in response.results:
            if not hasattr(result, "alternatives") or not result.alternatives:
                continue

            self._mark_provider_progress()
            alternative = result.alternatives[0]
            transcript = getattr(alternative, "transcript", "")
            is_final = getattr(result, "is_final", False)

            if not transcript.strip():
                continue

            if not self._speaking:
                self._speaking = True
                self._event_ch.send_nowait(
                    stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)
                )

            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT
                    if is_final
                    else stt.SpeechEventType.INTERIM_TRANSCRIPT,
                    request_id=self._request_id,
                    alternatives=[
                        _convert_to_speech_data(
                            alternative,
                            language=LanguageCode(self._language),
                            enable_diarization=self._stt._opts.enable_diarization,
                            is_final=is_final,
                        )
                    ],
                )
            )

            if is_final:
                attempt.received_final_transcript = True
                self._speaking = False
                self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH))


def _endpointing_values(
    endpointing: EndpointingConfig | None,
) -> tuple[int, float, int, int, float, float] | None:
    if endpointing is None:
        return None

    if endpointing.mode == "low_latency":
        # Match the endpointing defaults used by NVIDIA's Pipecat service.
        values = {
            "start_history": -1,
            "start_threshold": -1.0,
            "stop_history": 500,
            "stop_history_eou": 240,
            "stop_threshold": -1.0,
            "stop_threshold_eou": -1.0,
        }
    else:
        raise ValueError(f"Unsupported NVIDIA endpointing mode: {endpointing.mode}")

    for key in values:
        override = getattr(endpointing, key)
        if override is not None:
            values[key] = override

    return (
        int(values["start_history"]),
        float(values["start_threshold"]),
        int(values["stop_history"]),
        int(values["stop_history_eou"]),
        float(values["stop_threshold"]),
        float(values["stop_threshold_eou"]),
    )


def _response_to_speech_event(
    response: Any,
    *,
    language: LanguageCode,
    request_id: str,
    event_type: stt.SpeechEventType,
    enable_diarization: bool,
    is_final: bool,
) -> stt.SpeechEvent:
    results = [
        result for result in getattr(response, "results", []) if getattr(result, "alternatives", [])
    ]
    alternatives = [
        _combine_result_alternatives(
            [
                result.alternatives[alternative_index]
                for result in results
                if len(result.alternatives) > alternative_index
            ],
            language=language,
            enable_diarization=enable_diarization,
            is_final=is_final,
        )
        for alternative_index in range(
            max((len(result.alternatives) for result in results), default=0)
        )
    ]

    if not alternatives:
        alternatives.append(stt.SpeechData(language=language, text=""))

    return stt.SpeechEvent(
        type=event_type,
        request_id=request_id,
        alternatives=alternatives,
    )


def _combine_result_alternatives(
    alternatives: list[Any],
    *,
    language: LanguageCode,
    enable_diarization: bool,
    is_final: bool,
) -> stt.SpeechData:
    transcripts = [
        transcript.strip()
        for alternative in alternatives
        if (transcript := getattr(alternative, "transcript", ""))
    ]
    confidences = [
        float(getattr(alternative, "confidence", 0.0) or 0.0)
        for alternative in alternatives
        if hasattr(alternative, "confidence")
    ]
    words = [
        word for alternative in alternatives for word in (getattr(alternative, "words", []) or [])
    ]

    start_time = 0.0
    end_time = 0.0
    speaker_id: str | None = None
    timed_words: list[TimedString] | None = None

    if words:
        start_time = _time_offset_seconds(getattr(words[0], "start_time", 0))
        end_time = _time_offset_seconds(getattr(words[-1], "end_time", 0))
        timed_words = [
            TimedString(
                text=getattr(word, "word", ""),
                start_time=_time_offset_seconds(getattr(word, "start_time", 0)),
                end_time=_time_offset_seconds(getattr(word, "end_time", 0)),
            )
            for word in words
        ]

        if enable_diarization and is_final:
            speaker_tags = [getattr(word, "speaker_tag", 0) for word in words]
            if speaker_tags:
                speaker = Counter(speaker_tags).most_common(1)[0][0]
                speaker_id = f"S{speaker}"

    return stt.SpeechData(
        language=language,
        start_time=start_time,
        end_time=end_time,
        confidence=(sum(confidences) / len(confidences)) if confidences else 0.0,
        text=" ".join(transcripts),
        speaker_id=speaker_id,
        words=timed_words,
    )


def _convert_to_speech_data(
    alternative: Any,
    *,
    language: LanguageCode,
    enable_diarization: bool,
    is_final: bool,
) -> stt.SpeechData:
    return _combine_result_alternatives(
        [alternative],
        language=language,
        enable_diarization=enable_diarization,
        is_final=is_final,
    )


def _time_offset_seconds(value: Any) -> float:
    if value is None:
        return 0.0

    seconds = getattr(value, "seconds", None)
    nanos = getattr(value, "nanos", None)
    if seconds is not None or nanos is not None:
        return float(seconds or 0) + float(nanos or 0) / 1_000_000_000

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0

    if isinstance(value, int):
        return numeric / 1000.0

    return numeric
