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
    APIConnectionError,
    APIConnectOptions,
    stt,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, is_given

from . import auth

logger = logging.getLogger(__name__)


@dataclass
class STTOptions:
    model: str
    function_id: str
    punctuate: bool
    language_code: str
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
            ),
        )

        # Get API key from parameter or environment
        if is_given(api_key):
            self.nvidia_api_key = api_key
        else:
            self.nvidia_api_key = os.getenv("NVIDIA_API_KEY")
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
            language_code=language_code,
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
        effective_language = language if is_given(language) else self._opts.language_code
        return SpeechStream(stt=self, conn_options=conn_options, language=effective_language)


class SpeechStream(stt.SpeechStream):
    def __init__(self, *, stt: STT, conn_options: APIConnectOptions, language: str):
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=stt._opts.sample_rate)
        self._stt = stt
        self._language = language

        self._audio_queue = queue.Queue()
        self._shutdown_event = threading.Event()
        self._recognition_thread = None
        self._thread_exception = None

        self._speaking = False
        self._request_id = ""

        self._auth = auth.create_riva_auth(
            api_key=self._stt.nvidia_api_key,
            function_id=self._stt._opts.function_id,
            server=stt._opts.server,
            use_ssl=stt._opts.use_ssl,
        )
        self._asr_service = riva.client.ASRService(self._auth)

    async def _run(self) -> None:
        try:
            config = self._create_streaming_config()

            self._recognition_thread = threading.Thread(
                target=self._recognition_thread_worker,
                args=(config,),
                name="nvidia-asr-recognition",
                daemon=True,
            )
            self._recognition_thread.start()

            await self._collect_audio()

            if self._recognition_thread:
                await asyncio.to_thread(self._recognition_thread.join)

            if self._thread_exception:
                raise self._thread_exception

        except Exception as e:
            logger.exception("Error in NVIDIA streaming")
            if isinstance(e, APIConnectionError):
                raise e
            raise APIConnectionError(f"NVIDIA ASR streaming failed: {str(e)}") from e
        finally:
            self._shutdown()

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
            ),
            interim_results=True,
        )

    async def _collect_audio(self) -> None:
        try:
            async for data in self._input_ch:
                if self._shutdown_event.is_set():
                    break

                if isinstance(data, rtc.AudioFrame):
                    audio_bytes = data.data.tobytes()
                    if audio_bytes:
                        self._audio_queue.put(audio_bytes)
                elif isinstance(data, self._FlushSentinel):
                    break

        except Exception as e:
            logger.exception(f"Error collecting audio: {e}")
        finally:
            self._shutdown_event.set()

    def _recognition_thread_worker(self, config: riva.client.StreamingRecognitionConfig) -> None:
        try:
            audio_generator = self._audio_chunk_generator()

            response_generator = self._asr_service.streaming_response_generator(
                audio_generator, config
            )

            for response in response_generator:
                if self._shutdown_event.is_set():
                    break
                self._handle_response(response)

        except Exception as e:
            logger.exception(f"Error in NVIDIA recognition thread: {e}")
            self._thread_exception = e

    def _audio_chunk_generator(self) -> Generator[bytes, None, None]:
        while not self._shutdown_event.is_set():
            try:
                audio_chunk = self._audio_queue.get(timeout=0.1)
                yield audio_chunk
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in audio generator: {e}")
                break

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
                    start_event = stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)
                    self._event_ch.send_nowait(start_event)

                speech_data = self._convert_to_speech_data(alternative, is_final)

                if is_final:
                    final_event = stt.SpeechEvent(
                        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                        request_id=self._request_id,
                        alternatives=[speech_data],
                    )
                    self._event_ch.send_nowait(final_event)

                    if self._speaking:
                        self._speaking = False
                        end_event = stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH)
                        self._event_ch.send_nowait(end_event)
                else:
                    interim_event = stt.SpeechEvent(
                        type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                        request_id=self._request_id,
                        alternatives=[speech_data],
                    )
                    self._event_ch.send_nowait(interim_event)

        except Exception as e:
            logger.error(f"Error handling response: {e}")

    def _convert_to_speech_data(self, alternative, is_final: bool) -> stt.SpeechData:
        transcript = getattr(alternative, "transcript", "")
        confidence = getattr(alternative, "confidence", 0.0)
        words = getattr(alternative, "words", [])

        start_time = 0.0
        end_time = 0.0
        if words:
            start_time = getattr(words[0], "start_time", 0) / 1000.0
            end_time = getattr(words[-1], "end_time", 0) / 1000.0

        return stt.SpeechData(
            language=self._language,
            start_time=start_time,
            end_time=end_time,
            confidence=confidence,
            text=transcript,
        )

    def _shutdown(self) -> None:
        logger.debug("Shutting down NVIDIA STT stream")
        self._shutdown_event.set()

        if self._recognition_thread and self._recognition_thread.is_alive():
            try:
                self._recognition_thread.join(timeout=2.0)
                if self._recognition_thread.is_alive():
                    logger.warning("Recognition thread did not shut down cleanly")
            except Exception as e:
                logger.warning(f"Error joining recognition thread: {e}")

    def log_asr_models(self, asr_service: riva.client.ASRService) -> None:
        try:
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

            logger.debug("Available ASR models")
            asr_models = dict(sorted(asr_models.items()))
            logger.debug(asr_models)
        except Exception as e:
            logger.warning(f"Could not retrieve ASR models: {e}")
