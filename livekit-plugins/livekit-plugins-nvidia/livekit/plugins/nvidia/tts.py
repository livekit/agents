import asyncio
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

from . import auth

logger = logging.getLogger(__name__)


@dataclass
class TTSOptions:
    voice: str
    function_id: str
    server: str
    sample_rate: int
    use_ssl: bool
    language_code: str
    word_tokenizer: tokenize.WordTokenizer | tokenize.SentenceTokenizer


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        server: str = "grpc.nvcf.nvidia.com:443",
        voice: str = "Magpie-Multilingual.EN-US.Leo",
        function_id: str = "877104f7-e885-42b9-8de8-f6e4c6303969",
        language_code: str = "en-US",
        use_ssl: bool = True,
        api_key: str | None = None,
    ):
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=16000,
            num_channels=1,
        )

        if api_key:
            self.nvidia_api_key = api_key
        else:
            self.nvidia_api_key = os.getenv("NVIDIA_API_KEY")
            if use_ssl and not self.nvidia_api_key:
                raise ValueError(
                    "NVIDIA_API_KEY is not set while using SSL. Either pass api_key parameter, set NVIDIA_API_KEY environment variable "
                    + "or disable SSL and use a locally hosted Riva NIM service."
                )

        self._opts = TTSOptions(
            voice=voice,
            function_id=function_id,
            server=server,
            sample_rate=16000,
            use_ssl=use_ssl,
            language_code=language_code,
            word_tokenizer=tokenize.blingfire.SentenceTokenizer(),
        )
        self._tts_service = None

    def _ensure_session(self) -> riva.client.SpeechSynthesisService:
        if not self._tts_service:
            riva_auth = auth.create_riva_auth(
                api_key=self.nvidia_api_key,
                function_id=self._opts.function_id,
                server=self._opts.server,
                use_ssl=self._opts.use_ssl,
            )
            self._tts_service = riva.client.SpeechSynthesisService(riva_auth)
        return self._tts_service

    def list_voices(self) -> dict:
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
        return tts_models

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> tts.ChunkedStream:
        return self._synthesize_with_stream(text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> tts.SynthesizeStream:
        return SynthesizeStream(tts=self, conn_options=conn_options, opts=self._opts)


class SynthesizeStream(tts.SynthesizeStream):
    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions, opts: TTSOptions):
        super().__init__(tts=tts, conn_options=conn_options)
        self._opts = opts
        self._context_id = utils.shortuuid()
        self._sent_tokenizer_stream = self._opts.word_tokenizer.stream()
        self._token_q = queue.Queue()
        self._event_loop = asyncio.get_running_loop()

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=self._context_id,
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            stream=True,
            mime_type="audio/pcm",
        )
        output_emitter.start_segment(segment_id=self._context_id)

        done_fut = asyncio.Future()

        async def _input_task() -> None:
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    self._sent_tokenizer_stream.flush()
                    continue
                self._sent_tokenizer_stream.push_text(data)
            self._sent_tokenizer_stream.end_input()

        async def _process_segments() -> None:
            async for word_stream in self._sent_tokenizer_stream:
                self._token_q.put(word_stream)
            self._token_q.put(None)

        def _synthesize_worker() -> None:
            try:
                service = self._tts._ensure_session()
                while True:
                    token = self._token_q.get()

                    if not token:
                        break

                    try:
                        responses = service.synthesize_online(
                            token.token,
                            self._opts.voice,
                            self._opts.language_code,
                            sample_rate_hz=self._opts.sample_rate,
                            encoding=AudioEncoding.LINEAR_PCM,
                        )
                        for response in responses:
                            self._event_loop.call_soon_threadsafe(
                                output_emitter.push, response.audio
                            )

                    except Exception as e:
                        logger.error(f"Error in synthesis: {e}")
                        continue
            finally:
                self._event_loop.call_soon_threadsafe(done_fut.set_result, None)

        synthesize_thread = threading.Thread(
            target=_synthesize_worker,
            name="nvidia-tts-synthesize",
            daemon=True,
        )
        synthesize_thread.start()

        tasks = [
            asyncio.create_task(_input_task()),
            asyncio.create_task(_process_segments()),
        ]

        try:
            await asyncio.gather(*tasks)
        finally:
            self._token_q.put(None)
            await done_fut
            output_emitter.end_segment()
