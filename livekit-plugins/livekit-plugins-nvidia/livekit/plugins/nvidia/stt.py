import asyncio
import logging
import os
import queue
import threading
from dataclasses import dataclass

import riva.client
import riva.client.proto.riva_asr_pb2 as riva_asr

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
        super().__init__(stt=stt, conn_options=conn_options)
        self._stt = stt
        self._language = language

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
            # Create streaming configuration
            config = riva.client.StreamingRecognitionConfig(
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
            # Start the streaming recognition
            await self._stream_recognize(config)

        except Exception as e:
            raise e
            # if isinstance(e, APIConnectionError):
            #     raise e
            # # Wrap other exceptions in APIConnectionError
            # raise APIConnectionError(f"NVIDIA ASR streaming failed: {str(e)}") from e

    async def _stream_recognize(self, config: riva.client.StreamingRecognitionConfig) -> None:
        """Handle the bidirectional streaming with NVIDIA ASR."""

        auth = riva.client.Auth(
            use_ssl=True,
            uri=self._stt._opts.server,
            metadata_args=[
                ["authorization", f"Bearer {self._stt.nvidia_api_key}"],
                ["function-id", self._stt._opts.function_id],
            ],
        )

        asr_service = riva.client.ASRService(auth)
        config_response = asr_service.stub.GetRivaSpeechRecognitionConfig(
            riva.client.proto.riva_asr_pb2.RivaSpeechRecognitionConfigRequest()
        )
        asr_models = {}
        for model_config in config_response.model_config:
            if model_config.parameters["type"] == "online":
                language_code = model_config.parameters["language_code"]
                model = {"model": [model_config.model_name]}
                if language_code in asr_models:
                    asr_models[language_code].append(model)
                else:
                    asr_models[language_code] = [model]

        logger.debug("Available ASR models")
        asr_models = dict(sorted(asr_models.items()))
        logger.debug(asr_models)
