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

logger = logging.getLogger(__name__)


@dataclass
class STTOptions:
    model: str
    function_id: str
    punctuate: bool
    language_code: str
    sample_rate: int
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
            if not self.nvidia_api_key:
                raise ValueError(
                    "NVIDIA_API_KEY is not set. Either pass api_key parameter or set NVIDIA_API_KEY environment variable"
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
        # Use provided language or fall back to configured language
        effective_language = language if is_given(language) else self._opts.language_code
        return SpeechStream(stt=self, conn_options=conn_options, language=effective_language)


class SpeechStream(stt.SpeechStream):
    def __init__(self, *, stt: STT, conn_options: APIConnectOptions, language: str):
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=stt._opts.sample_rate)
        self._stt = stt
        self._language = language

        # Threading primitives for sync/async bridge
        self._audio_queue = queue.Queue()
        self._shutdown = threading.Event()

        # Initialize NVIDIA Riva client
        self._auth = riva.client.Auth(
            uri=stt._opts.server,
            use_ssl=True,
            metadata_args=[
                ["authorization", f"Bearer {stt.nvidia_api_key}"],
                ["function-id", stt._opts.function_id],
            ],
        )
        self._asr_service = riva.client.ASRService(self._auth)

    async def _run(self) -> None:
        """Main streaming loop that processes audio and sends it to NVIDIA ASR."""
        logger.debug("Starting NVIDIA ASR streaming session")

        try:
            config = self._create_streaming_config()

            # Run audio collection and recognition concurrently
            await asyncio.gather(
                self._collect_audio(),
                self._process_streaming_recognition(config),
                return_exceptions=True,
            )

        except Exception as e:
            logger.exception("Error in NVIDIA streaming")
            if isinstance(e, APIConnectionError):
                raise e
            raise APIConnectionError(f"NVIDIA ASR streaming failed: {str(e)}") from e
        finally:
            self._shutdown.set()

    def _create_streaming_config(self) -> riva.client.StreamingRecognitionConfig:
        """Create the streaming configuration for NVIDIA ASR."""
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
        """Collect audio frames from LiveKit and put them in the queue."""
        try:
            async for data in self._input_ch:
                if isinstance(data, rtc.AudioFrame):
                    audio_bytes = data.data.tobytes()
                    if audio_bytes:
                        self._audio_queue.put(audio_bytes)
                elif isinstance(data, self._FlushSentinel):
                    logger.debug("Received flush sentinel, ending audio stream")
                    break
        except Exception as e:
            logger.exception(f"Error collecting audio: {e}")
        finally:
            self._shutdown.set()
            logger.debug("Audio collection finished")

    async def _process_streaming_recognition(
        self, config: riva.client.StreamingRecognitionConfig
    ) -> None:
        """Process streaming recognition using NVIDIA ASR service."""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._run_nvidia_streaming, config)
        except Exception as e:
            logger.exception(f"Error in streaming recognition: {e}")
            raise

    def _run_nvidia_streaming(self, config: riva.client.StreamingRecognitionConfig) -> None:
        """Run NVIDIA streaming recognition in a thread (synchronous)."""
        try:
            # Create audio generator for NVIDIA
            audio_generator = self._audio_chunk_generator()

            # Get responses from NVIDIA
            response_generator = self._asr_service.streaming_response_generator(
                audio_generator, config
            )

            # Process each response
            for response in response_generator:
                self._handle_response(response)

        except Exception as e:
            logger.exception(f"Error in NVIDIA streaming thread: {e}")
            raise

    def _audio_chunk_generator(self) -> Generator[bytes, None, None]:
        """Generate audio chunks for NVIDIA (synchronous generator)."""
        logger.debug("Starting audio chunk generator")

        while not self._shutdown.is_set():
            try:
                # Get audio with timeout to allow shutdown check
                audio_chunk = self._audio_queue.get(timeout=0.1)
                yield audio_chunk
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in audio generator: {e}")
                break

        logger.debug("Audio chunk generator finished")

    def _handle_response(self, response) -> None:
        """Handle a single response from NVIDIA ASR."""
        try:
            if not hasattr(response, "results") or not response.results:
                return

            for result in response.results:
                if not hasattr(result, "alternatives") or not result.alternatives:
                    continue

                for alternative in result.alternatives:
                    transcript = getattr(alternative, "transcript", "")
                    confidence = getattr(alternative, "confidence", 0.0)
                    is_final = getattr(result, "is_final", False)

                    if transcript.strip():  # Only log non-empty transcripts
                        status = "FINAL" if is_final else "INTERIM"
                        logger.info(
                            f"Transcript ({status}): '{transcript}' (confidence: {confidence:.3f})"
                        )

        except Exception as e:
            logger.error(f"Error handling response: {e}")

    def log_asr_models(self, asr_service: riva.client.ASRService) -> None:
        """Log available ASR models (utility method)."""
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
