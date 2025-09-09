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

    def _recognize_stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.RecognizeStream:
        return SpeechStream(stt=self, conn_options=conn_options, language=language)

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

            # Set up audio buffering - buffer audio in chunks suitable for streaming
            audio_buffer = AudioBuffer(
                capacity=self._stt._opts.sample_rate,  # 1 second capacity
            )

            # Start the streaming recognition
            await self._stream_recognize(config, audio_buffer)

        except Exception as e:
            if isinstance(e, APIConnectionError):
                raise
            # Wrap other exceptions in APIConnectionError
            raise APIConnectionError(f"NVIDIA ASR streaming failed: {str(e)}") from e

    async def _stream_recognize(
        self, config: riva.client.StreamingRecognitionConfig, audio_buffer: AudioBuffer
    ) -> None:
        """Handle the bidirectional streaming with NVIDIA ASR."""

        # Create a queue for audio chunks
        audio_queue = queue.Queue()

        def audio_generator():
            """Generator that yields audio chunks for NVIDIA client."""
            while True:
                try:
                    chunk = audio_queue.get(timeout=0.1)
                    if chunk is None:  # Sentinel to stop
                        break
                    yield chunk
                except queue.Empty:
                    continue

        # Create a thread to handle the gRPC streaming (since NVIDIA client is synchronous)
        streaming_thread = None
        responses_queue = queue.Queue()

        def streaming_worker():
            """Worker thread that handles the synchronous NVIDIA streaming."""
            try:
                responses = self._asr_service.streaming_response_generator(
                    audio_chunks=audio_generator(),
                    streaming_config=config,
                )

                for response in responses:
                    responses_queue.put(("response", response))

            except Exception as e:
                responses_queue.put(("error", e))

        streaming_thread = threading.Thread(target=streaming_worker, daemon=True)
        streaming_thread.start()

        # Process input audio frames
        async def process_audio():
            async for data in self._input_ch:
                if isinstance(data, rtc.AudioFrame):
                    # Convert frame to the expected sample rate if needed
                    frame_data = data.data.tobytes()
                    audio_queue.put(frame_data)

                elif isinstance(data, self._FlushSentinel):
                    # Signal end of audio
                    audio_queue.put(None)
                    break

        # Process responses from NVIDIA
        async def process_responses():
            while streaming_thread and streaming_thread.is_alive():
                try:
                    # Check for responses from the streaming thread
                    try:
                        item_type, item = responses_queue.get(timeout=0.1)
                    except queue.Empty:
                        await asyncio.sleep(0.01)
                        continue

                    if item_type == "error":
                        if isinstance(item, Exception):
                            raise APIConnectionError(f"NVIDIA ASR error: {str(item)}") from item
                        else:
                            raise APIConnectionError(f"NVIDIA ASR error: {item}")

                    response = item
                    await self._process_response(response)

                except APIConnectionError:
                    raise
                except Exception as e:
                    # Wrap in APIConnectionError for consistency
                    raise APIConnectionError(
                        f"Error processing NVIDIA ASR response: {str(e)}"
                    ) from e

        # Run both audio processing and response processing concurrently
        try:
            await asyncio.gather(process_audio(), process_responses())
        finally:
            # Clean up
            if streaming_thread and streaming_thread.is_alive():
                audio_queue.put(None)  # Signal thread to stop
                streaming_thread.join(timeout=1.0)

    async def _process_response(self, response: riva_asr.StreamingRecognizeResponse) -> None:
        """Process a response from NVIDIA ASR and emit appropriate events."""
        if not response.results:
            return

        for result in response.results:
            if not result.alternatives:
                continue

            # Get the best alternative
            alternative = result.alternatives[0]
            transcript = alternative.transcript
            confidence = alternative.confidence

            if not transcript.strip():
                continue

            # Create speech data
            speech_data = stt.SpeechData(
                language=self._language,
                text=transcript,
                confidence=confidence,
                start_time=0.0,  # NVIDIA doesn't provide precise timing in streaming mode
                end_time=0.0,
            )

            # Determine event type based on whether this is a final result
            if result.is_final:
                event_type = stt.SpeechEventType.FINAL_TRANSCRIPT
                logger.debug(f"Final transcript: {transcript}")
            else:
                event_type = stt.SpeechEventType.INTERIM_TRANSCRIPT
                logger.debug(f"Interim transcript: {transcript}")

            # Emit the speech event
            speech_event = stt.SpeechEvent(
                type=event_type,
                alternatives=[speech_data],
            )

            self._event_ch.send_nowait(speech_event)
