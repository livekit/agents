import asyncio
import logging
import os
import queue
import threading
from collections.abc import Generator
from dataclasses import dataclass

import riva.client

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectOptions,
    Language,
    stt,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, is_given
from livekit.agents.voice.io import TimedString

from . import auth

logger = logging.getLogger(__name__)


@dataclass
class STTOptions:
    model: str
    function_id: str
    punctuate: bool
    language_code: Language
    sample_rate: int
    use_ssl: bool
    server: str


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
    ):
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
                aligned_transcript="word",
                offline_recognize=False,
            ),
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
            f"Function ID: {function_id}, Language: {language_code}, Sample rate: {sample_rate}"
        )

        self._opts = STTOptions(
            model=model,
            function_id=function_id,
            punctuate=punctuate,
            language_code=Language(language_code),
            sample_rate=sample_rate,
            server=server,
            use_ssl=use_ssl,
        )

    def _recognize_impl(
        self,
        buffer: AudioBuffer,
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
        effective_language = Language(language) if is_given(language) else self._opts.language_code
        return SpeechStream(stt=self, conn_options=conn_options, language=effective_language)

    def log_asr_models(self, asr_service: riva.client.ASRService) -> dict:
        config_response = asr_service.stub.GetRivaSpeechRecognitionConfig(
            riva.client.RivaSpeechRecognitionConfigRequest()
        )

        asr_models = {}
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


class SpeechStream(stt.SpeechStream):
    def __init__(self, *, stt: STT, conn_options: APIConnectOptions, language: str):
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=stt._opts.sample_rate)
        self._stt = stt
        self._language = language

        self._audio_queue = queue.Queue()
        self._shutdown_event = threading.Event()
        self._recognition_thread = None

        self._speaking = False
        self._request_id = ""

        self._auth = auth.create_riva_auth(
            api_key=self._stt.nvidia_api_key,
            function_id=self._stt._opts.function_id,
            server=stt._opts.server,
            use_ssl=stt._opts.use_ssl,
        )
        self._asr_service = riva.client.ASRService(self._auth)

        self._event_loop = asyncio.get_running_loop()
        self._done_fut = asyncio.Future()

    async def _run(self) -> None:
        config = self._create_streaming_config()

        self._recognition_thread = threading.Thread(
            target=self._recognition_worker,
            args=(config,),
            name="nvidia-asr-recognition",
            daemon=True,
        )
        self._recognition_thread.start()

        try:
            await self._collect_audio()

        finally:
            self._audio_queue.put(None)
            await self._done_fut

    def _create_streaming_config(self) -> riva.client.StreamingRecognitionConfig:
        return riva.client.StreamingRecognitionConfig(
            config=riva.client.RecognitionConfig(
                encoding=riva.client.AudioEncoding.LINEAR_PCM,
                language_code=self._language,
                model=self._stt._opts.model,
                max_alternatives=1,
                enable_automatic_punctuation=self._stt._opts.punctuate,
                sample_rate_hertz=self._stt._opts.sample_rate,
                audio_channel_count=1,
                enable_word_time_offsets=True,
            ),
            interim_results=True,
        )

    async def _collect_audio(self) -> None:
        async for data in self._input_ch:
            if isinstance(data, rtc.AudioFrame):
                audio_bytes = data.data.tobytes()
                if audio_bytes:
                    self._audio_queue.put(audio_bytes)
            elif isinstance(data, self._FlushSentinel):
                break

    def _recognition_worker(self, config: riva.client.StreamingRecognitionConfig) -> None:
        max_retries = 3
        retry_count = 0

        try:
            while retry_count < max_retries:
                try:
                    audio_generator = self._audio_chunk_generator()

                    response_generator = self._asr_service.streaming_response_generator(
                        audio_generator, config
                    )

                    for response in response_generator:
                        self._handle_response(response)
                        retry_count = 0  # Reset on successful response

                    # Normal completion, exit the retry loop
                    break

                except Exception as e:
                    error_msg = str(e).lower()
                    if "start flag" in error_msg or "sequence" in error_msg:
                        retry_count += 1
                        if retry_count < max_retries:
                            logger.warning(
                                f"Riva sequence timeout detected, recreating ASR service "
                                f"(attempt {retry_count}/{max_retries}): {e}"
                            )
                            self._asr_service = riva.client.ASRService(self._auth)
                            continue
                        else:
                            logger.error(
                                f"Max retries ({max_retries}) exceeded for Riva sequence timeout"
                            )
                            raise
                    else:
                        raise

        except Exception:
            logger.exception("Error in NVIDIA recognition thread")
        finally:
            self._event_loop.call_soon_threadsafe(self._done_fut.set_result, None)

    def _audio_chunk_generator(self) -> Generator[bytes, None, None]:
        """
        The nvidia riva SDK requires a generator for realtime STT - so we have to
        wrap the
        """
        while True:
            audio_chunk = self._audio_queue.get()

            if not audio_chunk:
                break

            yield audio_chunk

    def _handle_response(self, response) -> None:
        try:
            if not hasattr(response, "results") or not response.results:
                return

            for result in response.results:
                if not hasattr(result, "alternatives") or not result.alternatives:
                    continue

                alternative = result.alternatives[0]
                transcript = getattr(alternative, "transcript", "")
                is_final = getattr(result, "is_final", False)

                if not transcript.strip():
                    continue

                self._request_id = f"nvidia-{id(response)}"

                if not self._speaking and transcript.strip():
                    self._speaking = True
                    self._event_loop.call_soon_threadsafe(
                        self._event_ch.send_nowait,
                        stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH),
                    )

                speech_data = self._convert_to_speech_data(alternative)

                if is_final:
                    self._event_loop.call_soon_threadsafe(
                        self._event_ch.send_nowait,
                        stt.SpeechEvent(
                            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                            request_id=self._request_id,
                            alternatives=[speech_data],
                        ),
                    )

                    if self._speaking:
                        self._event_loop.call_soon_threadsafe(
                            self._event_ch.send_nowait,
                            stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH),
                        )
                else:
                    self._event_loop.call_soon_threadsafe(
                        self._event_ch.send_nowait,
                        stt.SpeechEvent(
                            type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                            request_id=self._request_id,
                            alternatives=[speech_data],
                        ),
                    )

        except Exception:
            logger.exception("Error handling response")

    def _convert_to_speech_data(self, alternative) -> stt.SpeechData:
        transcript = getattr(alternative, "transcript", "")
        confidence = getattr(alternative, "confidence", 0.0)
        words = getattr(alternative, "words", [])

        start_time = 0.0
        end_time = 0.0
        if words:
            start_time = getattr(words[0], "start_time", 0) / 1000.0 + self.start_time_offset
            end_time = getattr(words[-1], "end_time", 0) / 1000.0 + self.start_time_offset

        return stt.SpeechData(
            language=Language(self._language),
            start_time=start_time,
            end_time=end_time,
            confidence=confidence,
            text=transcript,
            words=[
                TimedString(
                    text=getattr(word, "word", ""),
                    start_time=getattr(word, "start_time", 0) + self.start_time_offset,
                    end_time=getattr(word, "end_time", 0) + self.start_time_offset,
                )
                for word in words
            ]
            if words
            else None,
        )
