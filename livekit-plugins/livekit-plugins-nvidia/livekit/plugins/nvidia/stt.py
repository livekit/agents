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
from collections import Counter, deque
from collections.abc import Callable, Generator
from contextlib import suppress
from dataclasses import dataclass, field
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

_MAX_REPLAY_AUDIO_SECONDS = 30
_PCM_BYTES_PER_SAMPLE = 2


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
    audio_queue: queue.Queue[bytes | None]
    done_fut: asyncio.Future[None]
    # None until the first audio sample anchors the RPC clock.
    start_time_offset: float | None
    bytes_per_second: int
    audio_bytes: int = 0
    thread: threading.Thread | None = None
    input_closed: bool = False
    # Set when the connection attempt that owns this RPC ends; its results are then ignored.
    abandoned: bool = False
    error: APIError | None = None
    final_transcript_emitted: bool = False
    replay_audio: deque[bytes] = field(default_factory=deque)
    replay_audio_bytes: int = 0
    final_transcript_received: bool = False
    replay_lock: threading.Lock = field(default_factory=threading.Lock)

    def buffer_for_replay(self, audio: bytes, *, max_bytes: int) -> None:
        with self.replay_lock:
            self.audio_bytes += len(audio)
            self.final_transcript_emitted = False
            self.replay_audio.append(audio)
            self.replay_audio_bytes += len(audio)
            while self.replay_audio_bytes > max_bytes:
                oldest = self.replay_audio.popleft()
                excess = self.replay_audio_bytes - max_bytes
                if len(oldest) > excess:
                    self.replay_audio.appendleft(oldest[excess:])
                self.replay_audio_bytes -= min(len(oldest), excess)

    def mark_final_transcript(self) -> None:
        with self.replay_lock:
            self.final_transcript_emitted = True
            self.final_transcript_received = True
            self.replay_audio.clear()
            self.replay_audio_bytes = 0

    def replay_snapshot(self) -> tuple[list[bytes], float, bool]:
        with self.replay_lock:
            # A replayed RPC's clock starts at the first retained PCM sample.
            offset = (self.start_time_offset or 0.0) + (
                self.audio_bytes - self.replay_audio_bytes
            ) / self.bytes_per_second
            return list(self.replay_audio), offset, self.final_transcript_emitted

    def received_final_transcript(self) -> bool:
        with self.replay_lock:
            return self.final_transcript_received


def _close_attempt_input(attempt: _RecognitionAttempt) -> None:
    if not attempt.input_closed:
        attempt.input_closed = True
        attempt.audio_queue.put(None)


class SpeechStream(stt.SpeechStream):
    def __init__(self, *, stt: STT, conn_options: APIConnectOptions, language: str):
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=stt._opts.sample_rate)
        self._stt = stt
        self._language = language

        self._speaking = False
        self._request_id = ""
        self._max_replay_bytes = (
            stt._opts.sample_rate * _PCM_BYTES_PER_SAMPLE * _MAX_REPLAY_AUDIO_SECONDS
        )
        self._asr_service: riva.client.ASRService | None = None
        self._attempt: _RecognitionAttempt | None = None
        self._failed: asyncio.Future[None] | None = None
        # Uncommitted audio of a failed connection, replayed by the next one.
        self._replay: tuple[list[bytes], float] | None = None

    async def _run(self) -> None:
        # Each connection owns its channel. Closing it also cancels a stalled RPC.
        riva_auth = auth.create_riva_auth(
            api_key=self._stt.nvidia_api_key,
            function_id=self._stt._opts.function_id,
            server=self._stt._opts.server,
            use_ssl=self._stt._opts.use_ssl,
        )
        self._asr_service = riva.client.ASRService(riva_auth)
        config = self._create_streaming_config()
        event_loop = asyncio.get_running_loop()
        self._failed = event_loop.create_future()
        drain_started = asyncio.Event()
        attempts: list[_RecognitionAttempt] = []

        def start_attempt(*, start_time_offset: float | None = None) -> _RecognitionAttempt:
            attempt = self._start_recognition_attempt(
                config, event_loop, start_time_offset=start_time_offset
            )
            attempts.append(attempt)
            self._attempt = attempt
            return attempt

        def start_replay(chunks: list[bytes], offset: float) -> _RecognitionAttempt:
            attempt = start_attempt(start_time_offset=offset)
            for chunk in chunks:
                attempt.buffer_for_replay(chunk, max_bytes=self._max_replay_bytes)
                attempt.audio_queue.put(chunk)
            return attempt

        async def send() -> None:
            replay, self._replay = self._replay, None
            # Open the RPC before audio arrives so a rejected session fails early.
            attempt: _RecognitionAttempt | None = (
                start_replay(*replay) if replay is not None else start_attempt()
            )
            async for data in self._input_ch:
                if isinstance(data, rtc.AudioFrame):
                    audio_bytes = data.data.tobytes()
                    if not audio_bytes:
                        continue
                    if attempt is None:
                        attempt = start_attempt()
                    elif attempt.done_fut.done() and attempt.error is None:
                        # NVIDIA Speech ended the RPC cleanly after a final transcript.
                        # Continue on a new RPC with the audio it has not transcribed.
                        chunks, offset, _ = attempt.replay_snapshot()
                        attempt = start_replay(chunks, offset)
                    if attempt.start_time_offset is None:
                        attempt.start_time_offset = self.start_time_offset
                    attempt.buffer_for_replay(audio_bytes, max_bytes=self._max_replay_bytes)
                    attempt.audio_queue.put(audio_bytes)
                elif isinstance(data, self._FlushSentinel) and attempt is not None:
                    if self._input_ch.closed and self._input_ch.empty():
                        drain_started.set()
                    await self._finish_recognition_segment(attempt, start_replay)
                    attempt = None
                    # the send loop reads no audio while a segment drains, so the next frame it
                    # reads is the first one the next RPC receives
                    self._input_ch.restart_provider_clock()

            drain_started.set()
            if attempt is not None:
                await self._finish_recognition_segment(attempt, start_replay)

        try:
            await self._run_connection_tasks(
                asyncio.create_task(send()),
                self._failed,
                sender_waits_for_completion=True,
                drain_started=drain_started,
            )
        finally:
            self._failed.cancel()
            failed_attempt, self._attempt = self._attempt, None
            if failed_attempt is not None:
                chunks, offset, _ = failed_attempt.replay_snapshot()
                if chunks:
                    self._replay = (chunks, offset)
            for attempt in attempts:
                attempt.abandoned = True
                _close_attempt_input(attempt)
            riva_auth.channel.close()

    def _start_recognition_attempt(
        self,
        config: riva.client.StreamingRecognitionConfig,
        event_loop: asyncio.AbstractEventLoop,
        *,
        start_time_offset: float | None = None,
    ) -> _RecognitionAttempt:
        assert self._asr_service is not None
        attempt = _RecognitionAttempt(
            audio_queue=queue.Queue(),
            done_fut=event_loop.create_future(),
            start_time_offset=start_time_offset,
            bytes_per_second=self._stt._opts.sample_rate * _PCM_BYTES_PER_SAMPLE,
        )
        attempt.thread = threading.Thread(
            target=self._recognition_worker,
            args=(config, attempt, self._asr_service, event_loop),
            name="nvidia-asr-recognition",
            daemon=True,
        )
        attempt.thread.start()
        return attempt

    async def _finish_recognition_segment(
        self,
        attempt: _RecognitionAttempt,
        start_replay: Callable[[list[bytes], float], _RecognitionAttempt],
    ) -> None:
        retry_count = 0
        while True:
            _close_attempt_input(attempt)
            try:
                # One RPC drains, not the whole STT stream.
                await asyncio.wait_for(
                    asyncio.shield(attempt.done_fut), self._conn_options.final_response_timeout
                )
            except asyncio.TimeoutError:
                raise APITimeoutError("NVIDIA Speech STT final response timed out") from None
            error = attempt.error
            if error is None:
                break
            replay_audio, replay_offset, final_transcript_emitted = attempt.replay_snapshot()
            if final_transcript_emitted and not replay_audio:
                logger.warning(
                    "NVIDIA Speech stream failed after a final transcript; not replaying audio"
                )
                break
            # Once the input is drained, the stream retry has no audio to resend, so the
            # last segment replays its untranscribed audio here.
            if (
                not (self._input_ch.closed and self._input_ch.empty())
                or not replay_audio
                or not error.retryable
                or retry_count >= self._conn_options.max_retry
            ):
                raise error
            self._emit_error(error, recoverable=True)
            await asyncio.sleep(self._conn_options._interval_for_retry(retry_count))
            retry_count += 1
            attempt = start_replay(replay_audio, replay_offset)
        self._attempt = None
        self._finish_speech_segment()

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
        service: riva.client.ASRService,
        event_loop: asyncio.AbstractEventLoop,
    ) -> None:
        error: Exception | None = None
        try:
            responses = service.streaming_response_generator(
                self._audio_chunk_generator(attempt.audio_queue), config
            )
            for response in responses:
                if attempt.abandoned:
                    break
                event_loop.call_soon_threadsafe(self._handle_response, response, attempt)
        except Exception as e:
            error = e
        finally:
            # Cancellation may finish the agent loop before the SDK thread exits.
            with suppress(RuntimeError):
                event_loop.call_soon_threadsafe(self._complete_attempt, attempt, error)

    def _complete_attempt(self, attempt: _RecognitionAttempt, error: Exception | None) -> None:
        if attempt.done_fut.done():
            return
        if error is not None:
            attempt.error = to_api_error(error, operation="NVIDIA Speech streaming STT request")
        attempt.done_fut.set_result(None)

        # The sender handles the end of an RPC whose input it closed.
        if attempt.abandoned or attempt.input_closed:
            return
        if self._failed is None or self._failed.done():
            return
        if attempt.error is not None:
            self._failed.set_exception(attempt.error)
        elif not attempt.received_final_transcript():
            self._failed.set_exception(
                APIConnectionError("NVIDIA Speech STT stream ended before input ended")
            )

    def _audio_chunk_generator(
        self, audio_queue: queue.Queue[bytes | None]
    ) -> Generator[bytes, None, None]:
        """
        The NVIDIA Speech SDK requires a generator for realtime STT, so the LiveKit
        async input channel is bridged through a thread-safe queue.
        """
        yield from iter(audio_queue.get, None)

    def _handle_response(self, response: Any, attempt: _RecognitionAttempt) -> None:
        if attempt.abandoned or not getattr(response, "results", None):
            return

        self._request_id = f"nvidia-{id(response)}"

        for result in response.results:
            if not getattr(result, "alternatives", None):
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

            speech_data = _convert_to_speech_data(
                alternative,
                language=LanguageCode(self._language),
                start_time_offset=(
                    self.start_time_offset
                    if attempt.start_time_offset is None
                    else attempt.start_time_offset
                ),
                enable_diarization=self._stt._opts.enable_diarization,
                is_final=is_final,
            )

            event_type = (
                stt.SpeechEventType.FINAL_TRANSCRIPT
                if is_final
                else stt.SpeechEventType.INTERIM_TRANSCRIPT
            )
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=event_type,
                    request_id=self._request_id,
                    alternatives=[speech_data],
                )
            )

            if is_final:
                attempt.mark_final_transcript()
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
