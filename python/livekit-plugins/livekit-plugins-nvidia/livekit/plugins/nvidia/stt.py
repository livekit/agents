import asyncio
import logging
import os
import queue
import threading
from collections import Counter
from collections.abc import Generator
from dataclasses import dataclass

import grpc
import riva.client
from riva.client.proto import riva_asr_pb2 as asr_proto

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    LanguageCode,
    stt,
    utils,
)
from livekit.agents.types import NOT_GIVEN, ErrorKind, NotGivenOr
from livekit.agents.utils import is_given
from livekit.agents.voice.io import TranscriptChunk

from . import auth

logger = logging.getLogger(__name__)

# Riva's refusals by what failed; any other status is the connection or the server failing
_GRPC_ERROR_KINDS: dict[grpc.StatusCode, ErrorKind] = {
    grpc.StatusCode.UNAUTHENTICATED: "auth",
    grpc.StatusCode.PERMISSION_DENIED: "auth",
    grpc.StatusCode.INVALID_ARGUMENT: "invalid_request",
    grpc.StatusCode.NOT_FOUND: "invalid_request",
}


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
    ):
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
                diarization=enable_diarization,
                aligned_transcript="word",
                offline_recognize=False,
            ),
        )

        if enable_diarization and "sortformer" not in model:
            logger.warning(
                "Speaker diarization is enabled but model '%s' may not support it. "
                "Diarization requires a Sortformer-based model.",
                model,
            )

        if is_given(api_key):
            self.nvidia_api_key = api_key
        else:
            self.nvidia_api_key = os.getenv("NVIDIA_API_KEY", "")
            if use_ssl and not self.nvidia_api_key:
                raise ValueError(
                    "NVIDIA_API_KEY is not set while using SSL. Either pass api_key parameter, set NVIDIA_API_KEY environment variable "
                    + "or disable SSL and use a locally hosted Riva NIM service."
                )

        logger.info(f"Initializing NVIDIA STT with model: {model}, server: {server}")
        logger.debug(
            f"Function ID: {function_id}, LanguageCode: {language_code}, Sample rate: {sample_rate}"
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
        )

    async def _recognize_impl(
        self,
        buffer: list[rtc.AudioFrame] | rtc.AudioFrame,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        raise NotImplementedError("Not implemented")

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.RecognizeStream:
        effective_language = (
            LanguageCode(language) if is_given(language) else self._opts.language_code
        )
        return SpeechStream(
            stt=self, opts=self._opts, conn_options=conn_options, language=effective_language
        )

    def log_asr_models(
        self, asr_service: riva.client.ASRService
    ) -> dict[str, list[dict[str, list[str]]]]:
        config_response = asr_service.stub.GetRivaSpeechRecognitionConfig(
            asr_proto.RivaSpeechRecognitionConfigRequest()
        )

        asr_models: dict[str, list[dict[str, list[str]]]] = {}
        for model_config in config_response.model_config:
            if model_config.parameters.get("type") == "online":
                language_code = model_config.parameters["language_code"]
                model = {"model": [model_config.model_name]}
                if language_code in asr_models:
                    asr_models[language_code].append(model)
                else:
                    asr_models[language_code] = [model]

        asr_models = dict(sorted(asr_models.items()))
        return asr_models


class SpeechStream(stt.RecognizeStream):
    def __init__(
        self, *, stt: STT, opts: STTOptions, conn_options: APIConnectOptions, language: str
    ):
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)
        self._opts = opts
        self._language = language

        self._speaking = False
        self._request_id = ""

        self._auth = auth.create_riva_auth(
            api_key=stt.nvidia_api_key,
            function_id=self._opts.function_id,
            server=opts.server,
            use_ssl=opts.use_ssl,
        )
        self._asr_service = riva.client.ASRService(self._auth)

        self._event_loop = asyncio.get_running_loop()

    async def _run(self) -> None:
        # per attempt: a retry starts a fresh recognition thread
        self._speaking = False
        audio_queue: queue.Queue[bytes | None] = queue.Queue()
        done_fut: asyncio.Future[None] = self._event_loop.create_future()
        threading.Thread(
            target=self._recognition_worker,
            args=(self._create_streaming_config(), audio_queue, done_fut),
            name="nvidia-asr-recognition",
            daemon=True,
        ).start()

        async def push_audio() -> None:
            async for data in self.input:
                if isinstance(data, rtc.AudioFrame) and (audio_bytes := data.data.tobytes()):
                    audio_queue.put(audio_bytes)
            audio_queue.put(None)

        try:
            # a recognition failure ends the attempt without waiting for the input to end
            async with utils.aio.TaskSet() as tasks:
                await tasks.first_completed(tasks.spawn(push_audio()), asyncio.shield(done_fut))
        finally:
            audio_queue.put(None)
            await done_fut

    def _create_streaming_config(self) -> riva.client.StreamingRecognitionConfig:
        recognition_config = riva.client.RecognitionConfig(
            encoding=riva.client.AudioEncoding.LINEAR_PCM,
            language_code=self._language,
            model=self._opts.model,
            max_alternatives=1,
            enable_automatic_punctuation=self._opts.punctuate,
            sample_rate_hertz=self._opts.sample_rate,
            audio_channel_count=1,
            enable_word_time_offsets=True,
        )

        if self._opts.enable_diarization:
            diarization_config = asr_proto.SpeakerDiarizationConfig(
                enable_speaker_diarization=True,
                max_speaker_count=self._opts.max_speaker_count,
            )
            recognition_config.diarization_config.CopyFrom(diarization_config)

        return riva.client.StreamingRecognitionConfig(
            config=recognition_config,
            interim_results=True,
        )

    def _recognition_worker(
        self,
        config: riva.client.StreamingRecognitionConfig,
        audio_queue: queue.Queue[bytes | None],
        done_fut: asyncio.Future[None],
    ) -> None:
        def audio_chunks() -> Generator[bytes, None, None]:
            while audio_chunk := audio_queue.get():
                yield audio_chunk

        def settle(error: BaseException | None) -> None:
            if done_fut.done():
                return
            if error is None:
                done_fut.set_result(None)
            else:
                done_fut.set_exception(error)

        error: BaseException | None = None
        try:
            for response in self._asr_service.streaming_response_generator(audio_chunks(), config):
                self._handle_response(response)
        except grpc.RpcError as e:
            error = APIConnectionError(
                f"NVIDIA Riva recognition failed: {e.code()}",
                kind=_GRPC_ERROR_KINDS.get(e.code(), "connection"),
            )
        except Exception as e:
            error = APIConnectionError(f"NVIDIA Riva recognition failed: {e}")
        self._event_loop.call_soon_threadsafe(settle, error)

    def _handle_response(self, response: asr_proto.StreamingRecognizeResponse) -> None:
        if not response.results:
            return

        for result in response.results:
            if not result.alternatives:
                continue

            alternative = result.alternatives[0]
            transcript = alternative.transcript
            is_final = result.is_final

            if not transcript.strip():
                continue

            self._request_id = f"nvidia-{id(response)}"

            if not self._speaking and transcript.strip():
                self._speaking = True
                self._event_loop.call_soon_threadsafe(
                    self.send,
                    stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH),
                )

            speech_data = self._convert_to_speech_data(alternative, is_final=is_final)

            if is_final:
                self._event_loop.call_soon_threadsafe(
                    self.send,
                    stt.SpeechEvent(
                        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                        request_id=self._request_id,
                        alternatives=[speech_data],
                    ),
                )

                if self._speaking:
                    self._speaking = False
                    self._event_loop.call_soon_threadsafe(
                        self.send,
                        stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH),
                    )
            else:
                self._event_loop.call_soon_threadsafe(
                    self.send,
                    stt.SpeechEvent(
                        type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                        request_id=self._request_id,
                        alternatives=[speech_data],
                    ),
                )

    def _convert_to_speech_data(
        self, alternative: asr_proto.SpeechRecognitionAlternative, *, is_final: bool
    ) -> stt.SpeechData:
        transcript = alternative.transcript
        confidence = alternative.confidence
        words = alternative.words

        start_time = 0.0
        end_time = 0.0
        speaker_id: str | None = None
        if words:
            start_time = words[0].start_time / 1000.0 + self.start_time_offset
            end_time = words[-1].end_time / 1000.0 + self.start_time_offset

            if self._opts.enable_diarization and is_final:
                speaker_tags = [word.speaker_tag for word in words]
                if speaker_tags:
                    speaker = Counter(speaker_tags).most_common(1)[0][0]
                    speaker_id = f"S{speaker}"

        return stt.SpeechData(
            language=LanguageCode(self._language),
            start_time=start_time,
            end_time=end_time,
            confidence=confidence,
            text=transcript,
            speaker_id=speaker_id,
            words=[
                TranscriptChunk(
                    text=word.word,
                    start_time=word.start_time / 1000.0 + self.start_time_offset,
                    end_time=word.end_time / 1000.0 + self.start_time_offset,
                )
                for word in words
            ]
            if words
            else None,
        )
