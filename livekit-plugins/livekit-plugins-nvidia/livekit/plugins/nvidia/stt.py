# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

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

import grpc
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
    APIStatusError,
    APITimeoutError,
    LanguageCode,
    stt,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, is_given
from livekit.agents.voice.io import TimedString

from . import auth

logger = logging.getLogger(__name__)


EndpointingMode = Literal["low_latency"]
InferenceMode = Literal["auto", "streaming", "offline"]


@dataclass(frozen=True)
class EndpointingConfig:
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
        if inference_mode not in ("auto", "streaming", "offline"):
            raise ValueError("inference_mode must be 'auto', 'streaming', or 'offline'")

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
            raise _to_stt_api_error(e, operation="NVIDIA Speech offline STT request") from e

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
    audio_queue: queue.Queue[bytes | None]
    done_fut: asyncio.Future[None]
    thread: threading.Thread | None = None
    input_closed: bool = False
    final_transcript_emitted: bool = False


class SpeechStream(stt.SpeechStream):
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
        active_audio: list[bytes] = []
        active_attempt: _RecognitionAttempt | None = None

        try:
            async for data in self._input_ch:
                if isinstance(data, rtc.AudioFrame):
                    audio_bytes = data.data.tobytes()
                    if not audio_bytes:
                        continue

                    if active_attempt is None:
                        active_attempt = self._start_recognition_attempt(config, event_loop)
                    active_audio.append(audio_bytes)
                    active_attempt.audio_queue.put(audio_bytes)
                elif isinstance(data, self._FlushSentinel) and active_attempt is not None:
                    await self._finish_recognition_segment(
                        config=config,
                        audio=active_audio,
                        attempt=active_attempt,
                        event_loop=event_loop,
                    )
                    active_audio = []
                    active_attempt = None
        finally:
            if active_attempt is not None:
                self._close_attempt_input(active_attempt)
                if not active_attempt.done_fut.done():
                    with suppress(asyncio.CancelledError, asyncio.TimeoutError, Exception):
                        await asyncio.wait_for(asyncio.shield(active_attempt.done_fut), timeout=1.0)
                await self._join_attempt(active_attempt)

    def _start_recognition_attempt(
        self,
        config: riva.client.StreamingRecognitionConfig,
        event_loop: asyncio.AbstractEventLoop,
    ) -> _RecognitionAttempt:
        audio_queue: queue.Queue[bytes | None] = queue.Queue()
        done_fut: asyncio.Future[None] = event_loop.create_future()
        attempt = _RecognitionAttempt(audio_queue=audio_queue, done_fut=done_fut)
        recognition_thread = threading.Thread(
            target=self._recognition_worker,
            args=(config, attempt, event_loop),
            name="nvidia-asr-recognition",
            daemon=True,
        )
        attempt.thread = recognition_thread
        recognition_thread.start()
        return attempt

    async def _finish_recognition_segment(
        self,
        *,
        config: riva.client.StreamingRecognitionConfig,
        audio: list[bytes],
        attempt: _RecognitionAttempt,
        event_loop: asyncio.AbstractEventLoop,
    ) -> None:
        current_attempt = attempt
        try:
            for retry_count in range(self._conn_options.max_retry + 1):
                attempt_to_join = current_attempt
                self._close_attempt_input(current_attempt)
                try:
                    await current_attempt.done_fut
                    break
                except APIError as e:
                    if current_attempt.final_transcript_emitted:
                        logger.warning(
                            "NVIDIA Speech stream failed after a final transcript; "
                            "not replaying audio"
                        )
                        break
                    if not e.retryable:
                        raise
                    if retry_count == self._conn_options.max_retry:
                        raise APIConnectionError(
                            f"NVIDIA Speech streaming STT failed after {retry_count + 1} attempts",
                            retryable=False,
                        ) from e

                    await asyncio.sleep(self._conn_options._interval_for_retry(retry_count))
                    current_attempt = self._start_recognition_attempt(config, event_loop)
                    for audio_chunk in audio:
                        current_attempt.audio_queue.put(audio_chunk)
                finally:
                    await self._join_attempt(attempt_to_join)
        finally:
            self._finish_speech_segment()

    @staticmethod
    def _close_attempt_input(attempt: _RecognitionAttempt) -> None:
        if not attempt.input_closed:
            attempt.input_closed = True
            attempt.audio_queue.put(None)

    @staticmethod
    async def _join_attempt(attempt: _RecognitionAttempt) -> None:
        if attempt.thread is None or not attempt.thread.is_alive():
            return
        await asyncio.to_thread(attempt.thread.join, 1.0)
        if attempt.thread.is_alive():
            logger.warning("NVIDIA Speech recognition worker did not stop within one second")

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
        error: Exception | None = None
        try:
            audio_generator = self._audio_chunk_generator(attempt.audio_queue)

            response_generator = self._asr_service.streaming_response_generator(
                audio_generator, config
            )

            for response in response_generator:
                if self._handle_response(response, event_loop=event_loop):
                    attempt.final_transcript_emitted = True

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
                _to_stt_api_error(error, operation="NVIDIA Speech streaming STT request")
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

    def _handle_response(self, response: Any, *, event_loop: asyncio.AbstractEventLoop) -> bool:
        final_transcript_emitted = False
        if not hasattr(response, "results") or not response.results:
            return False

        self._request_id = f"nvidia-{id(response)}"

        for result in response.results:
            if not hasattr(result, "alternatives") or not result.alternatives:
                continue

            alternative = result.alternatives[0]
            transcript = getattr(alternative, "transcript", "")
            is_final = getattr(result, "is_final", False)

            if not transcript.strip():
                continue

            if not self._speaking:
                self._speaking = True
                event_loop.call_soon_threadsafe(
                    self._event_ch.send_nowait,
                    stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH),
                )

            speech_data = _convert_to_speech_data(
                alternative,
                language=LanguageCode(self._language),
                start_time_offset=self.start_time_offset,
                enable_diarization=self._stt._opts.enable_diarization,
                is_final=is_final,
            )

            event_type = (
                stt.SpeechEventType.FINAL_TRANSCRIPT
                if is_final
                else stt.SpeechEventType.INTERIM_TRANSCRIPT
            )
            event_loop.call_soon_threadsafe(
                self._event_ch.send_nowait,
                stt.SpeechEvent(
                    type=event_type,
                    request_id=self._request_id,
                    alternatives=[speech_data],
                ),
            )

            if is_final and self._speaking:
                final_transcript_emitted = True
                self._speaking = False
                event_loop.call_soon_threadsafe(
                    self._event_ch.send_nowait,
                    stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH),
                )

        return final_transcript_emitted


def _to_stt_api_error(error: Exception, *, operation: str) -> APIError:
    if isinstance(error, APIError):
        return error

    if isinstance(error, grpc.RpcError):
        code = error.code()
        details = error.details() or str(error)
        if code == grpc.StatusCode.DEADLINE_EXCEEDED:
            return APITimeoutError(f"{operation} timed out: {details}")
        if code == grpc.StatusCode.CANCELLED:
            return APIConnectionError(f"{operation} was cancelled by NVIDIA Speech")

        status_codes = {
            grpc.StatusCode.INVALID_ARGUMENT: 400,
            grpc.StatusCode.NOT_FOUND: 404,
            grpc.StatusCode.ALREADY_EXISTS: 409,
            grpc.StatusCode.PERMISSION_DENIED: 403,
            grpc.StatusCode.RESOURCE_EXHAUSTED: 429,
            grpc.StatusCode.FAILED_PRECONDITION: 400,
            grpc.StatusCode.ABORTED: 409,
            grpc.StatusCode.OUT_OF_RANGE: 400,
            grpc.StatusCode.UNIMPLEMENTED: 501,
            grpc.StatusCode.INTERNAL: 500,
            grpc.StatusCode.UNAVAILABLE: 503,
            grpc.StatusCode.DATA_LOSS: 500,
            grpc.StatusCode.UNAUTHENTICATED: 401,
        }
        return APIStatusError(
            f"{operation} failed: {details}",
            status_code=status_codes.get(code, -1),
            retryable=code
            in {
                grpc.StatusCode.UNKNOWN,
                grpc.StatusCode.RESOURCE_EXHAUSTED,
                grpc.StatusCode.ABORTED,
                grpc.StatusCode.INTERNAL,
                grpc.StatusCode.UNAVAILABLE,
            },
        )

    if isinstance(error, (TypeError, ValueError)):
        return APIStatusError(f"{operation} failed: {error}", status_code=400, retryable=False)

    return APIConnectionError(f"{operation} failed: {error}")


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
            start_time_offset=0.0,
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
    start_time_offset: float,
    enable_diarization: bool,
    is_final: bool,
) -> stt.SpeechData:
    transcripts = [
        transcript.strip()
        for alternative in alternatives
        if (transcript := getattr(alternative, "transcript", ""))
    ]
    confidences = [
        float(confidence)
        for alternative in alternatives
        if (confidence := getattr(alternative, "confidence", 0.0))
    ]
    words = [
        word for alternative in alternatives for word in (getattr(alternative, "words", []) or [])
    ]

    start_time = 0.0
    end_time = 0.0
    speaker_id: str | None = None
    timed_words: list[TimedString] | None = None

    if words:
        start_time = _time_offset_seconds(getattr(words[0], "start_time", 0)) + start_time_offset
        end_time = _time_offset_seconds(getattr(words[-1], "end_time", 0)) + start_time_offset
        timed_words = [
            TimedString(
                text=getattr(word, "word", ""),
                start_time=_time_offset_seconds(getattr(word, "start_time", 0)) + start_time_offset,
                end_time=_time_offset_seconds(getattr(word, "end_time", 0)) + start_time_offset,
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
    start_time_offset: float,
    enable_diarization: bool,
    is_final: bool,
) -> stt.SpeechData:
    return _combine_result_alternatives(
        [alternative],
        language=language,
        start_time_offset=start_time_offset,
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
