import asyncio
import json
import logging
import os
import queue
import threading
from dataclasses import dataclass

import riva.client
from riva.client.proto.riva_audio_pb2 import AudioEncoding

from livekit.agents import (
    APIConnectOptions,
    tokenize,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

logger = logging.getLogger(__name__)


@dataclass
class TTSOptions:
    voice: str
    function_id: str
    server: str
    sample_rate: int
    language_code: str
    word_tokenizer: tokenize.WordTokenizer | tokenize.SentenceTokenizer


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        server: str = "grpc.nvcf.nvidia.com:443",
        voice: str = "Magpie-Multilingual.EN-US.Sofia",
        function_id: str = "877104f7-e885-42b9-8de8-f6e4c6303969",
        language_code: str = "en-US",
        api_key: str | None = None,
    ):
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True, aligned_transcript=True),
            sample_rate=16000,
            num_channels=1,
        )

        if api_key:
            self.nvidia_api_key = api_key
        else:
            self.nvidia_api_key = os.getenv("NVIDIA_API_KEY")
            if not self.nvidia_api_key:
                raise ValueError(
                    "NVIDIA_API_KEY is not set. Either pass api_key parameter or set NVIDIA_API_KEY environment variable"
                )

        self._opts = TTSOptions(
            voice=voice,
            function_id=function_id,
            server=server,
            sample_rate=16000,
            language_code=language_code,
            word_tokenizer=tokenize.blingfire.SentenceTokenizer(),
        )
        self._tts_service = None

    def _ensure_session(self) -> riva.client.SpeechSynthesisService:
        if not self._tts_service:
            logger.debug(
                f"Creating TTS service with auth: {self.nvidia_api_key} and function_id: {self._opts.function_id}"
            )
            auth = riva.client.Auth(
                uri=self._opts.server,
                use_ssl=True,
                metadata_args=[
                    ["authorization", f"Bearer {self.nvidia_api_key}"],
                    ["function-id", self._opts.function_id],
                ],
            )

            # WORKAROUND: Manually patch the auth metadata since get_auth_metadata() is broken
            auth.metadata = [
                ("authorization", f"Bearer {self.nvidia_api_key}"),
                ("function-id", self._opts.function_id),
            ]

            logger.debug(f"Manually set auth metadata: {auth.metadata}")
            self._tts_service = riva.client.SpeechSynthesisService(auth)
        return self._tts_service

    def list_voices(self) -> None:
        """List available TTS voices from NVIDIA."""
        try:
            service = self._ensure_session()
            config_response = service.stub.GetRivaSynthesisConfig(
                riva.client.proto.riva_tts_pb2.RivaSynthesisConfigRequest()
            )
            tts_models = {}
            for model_config in config_response.model_config:
                language_code = model_config.parameters.get("language_code", "unknown")
                voice_name = model_config.parameters.get("voice_name", "unknown")
                subvoices_str = model_config.parameters.get("subvoices", "")

                if subvoices_str:
                    subvoices = [voice.split(":")[0] for voice in subvoices_str.split(",")]
                    full_voice_names = [voice_name + "." + subvoice for subvoice in subvoices]
                else:
                    full_voice_names = [voice_name]

                if language_code in tts_models:
                    tts_models[language_code]["voices"].extend(full_voice_names)
                else:
                    tts_models[language_code] = {"voices": full_voice_names}

            tts_models = dict(sorted(tts_models.items()))
            logger.info("Available TTS voices:")
            logger.info(json.dumps(tts_models, indent=4))

        except Exception as e:
            logger.error(f"Error listing TTS voices: {e}")
            # Don't raise to allow debugging
            logger.warning("TTS voice listing failed, skipping...")
            return

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> tts.ChunkedStream:
        raise NotImplementedError("Chunked synthesis is not supported for NVIDIA TTS")

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> tts.SynthesizeStream:
        return SynthesizeStream(tts=self, conn_options=conn_options, opts=self._opts)


SENT_FLUSH_SENTINEL = object()


class SynthesizeStream(tts.SynthesizeStream):
    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions, opts: TTSOptions):
        super().__init__(tts=tts, conn_options=conn_options)
        self._opts = opts
        self._context_id = utils.shortuuid()
        self._sent_tokenizer_stream = self._opts.word_tokenizer.stream()
        self._token_q = queue.Queue()  # Thread-safe queue
        self._shutdown_event = threading.Event()  # Thread-safe event

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        """Test NVIDIA TTS authentication - simplified for debugging."""
        output_emitter.initialize(
            request_id=self._context_id,
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            stream=True,
            mime_type="audio/pcm",
        )
        output_emitter.start_segment(segment_id=self._context_id)

        async def _input_task() -> None:
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    self._sent_tokenizer_stream.flush()
                    continue
                self._sent_tokenizer_stream.push_text(data)
            self._sent_tokenizer_stream.end_input()

        async def _process_segments() -> None:
            async for word_stream in self._sent_tokenizer_stream:
                self._token_q.put(word_stream)  # Thread-safe put
            self._token_q.put(SENT_FLUSH_SENTINEL)  # Thread-safe put

        def _synthesize_worker() -> None:  # Regular function, not async
            """Runs in separate thread - handles all blocking operations"""
            try:
                service = self._tts._ensure_session()
                while not self._shutdown_event.is_set():
                    try:
                        # Use timeout to allow checking shutdown event
                        token = self._token_q.get(timeout=0.1)
                    except queue.Empty:
                        continue

                    if token is SENT_FLUSH_SENTINEL:
                        break

                    # This entire block is blocking and runs in thread
                    try:
                        responses = service.synthesize_online(
                            token.token,
                            self._opts.voice,
                            self._opts.language_code,
                            sample_rate_hz=self._opts.sample_rate,
                            encoding=AudioEncoding.LINEAR_PCM,
                        )
                        # Iterate over the blocking generator
                        for response in responses:
                            if self._shutdown_event.is_set():
                                break
                            output_emitter.push(response.audio)
                    except Exception as e:
                        logger.error(f"Error in synthesis: {e}")
                        continue
                    finally:
                        self._token_q.task_done()

            except Exception as e:
                logger.error(f"Error in synthesis worker: {e}")

        # Start synthesis worker thread
        synthesize_thread = threading.Thread(
            target=_synthesize_worker,
            name="nvidia-tts-synthesize",
            daemon=True,
        )
        synthesize_thread.start()

        # Run async tasks
        tasks = [
            asyncio.create_task(_input_task()),
            asyncio.create_task(_process_segments()),
        ]

        try:
            await asyncio.gather(*tasks)
            await asyncio.to_thread(synthesize_thread.join, timeout=5.0)
        finally:
            # Signal thread to stop and wait for it
            self._shutdown_event.set()
            output_emitter.end_segment()
