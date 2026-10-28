import asyncio
import contextlib
import os
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass

import grpc
import riva.client
from riva.client.proto.riva_audio_pb2 import AudioEncoding
from riva.client.proto.riva_tts_pb2 import RivaSynthesisConfigRequest

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APITimeoutError,
    tokenize,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

from . import auth
from .log import logger


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
            self.nvidia_api_key: str | None = api_key
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
        self._tts_service: riva.client.SpeechSynthesisService | None = None

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

    def list_voices(self) -> dict[str, dict[str, list[str]]]:
        service = self._ensure_session()
        config_response = service.stub.GetRivaSynthesisConfig(RivaSynthesisConfigRequest())
        tts_models: dict[str, dict[str, list[str]]] = {}
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
        return SynthesizeStream(
            tts=self, conn_options=conn_options, opts=self._opts, service=self._ensure_session
        )


class SynthesizeStream(tts.SynthesizeStream):
    def __init__(
        self,
        *,
        tts: TTS,
        conn_options: APIConnectOptions,
        opts: TTSOptions,
        service: Callable[[], riva.client.SpeechSynthesisService],
    ):
        super().__init__(tts=tts, conn_options=conn_options)
        self._get_service = service
        self._opts = opts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        context_id = utils.shortuuid()
        sent_tokenizer_stream = self._opts.word_tokenizer.stream()
        token_q: queue.Queue[tokenize.TokenData | None] = queue.Queue()
        event_loop = asyncio.get_running_loop()
        read_deadline = tts.StreamReadDeadline(self._conn_options)
        stopped = threading.Event()
        active_call: grpc.Call | None = None

        output_emitter.initialize(
            request_id=context_id,
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            stream=True,
            mime_type="audio/pcm",
        )
        output_emitter.start_segment(segment_id=context_id)

        done_fut: asyncio.Future[None] = asyncio.Future()

        async def _input_task() -> None:
            async for data in self.input:
                if isinstance(data, self.Flush):
                    sent_tokenizer_stream.flush()
                    continue
                sent_tokenizer_stream.push_text(data)
            sent_tokenizer_stream.end_input()

        async def _process_segments() -> None:
            async for word_stream in sent_tokenizer_stream:
                token_q.put(word_stream)
            token_q.put(None)

        def _finish(error: Exception | None) -> None:
            if done_fut.done():
                return
            if error is None or stopped.is_set():
                done_fut.set_result(None)
            else:
                done_fut.set_exception(error)

        def _started(call: object) -> None:
            nonlocal active_call
            active_call = call if isinstance(call, grpc.Call) else None
            if stopped.is_set():
                if active_call is not None:
                    active_call.cancel()
                return

        def _push(audio: bytes) -> None:
            if not stopped.is_set():
                read_deadline.progress()
                output_emitter.push(audio)

        def _synthesize_worker() -> None:
            error: Exception | None = None
            try:
                service = self._get_service()
                while (token := token_q.get()) is not None:
                    if stopped.is_set():
                        break
                    event_loop.call_soon_threadsafe(read_deadline.expect_response)
                    responses = service.synthesize_online(
                        token.token,
                        self._opts.voice,
                        self._opts.language_code,
                        sample_rate_hz=self._opts.sample_rate,
                        encoding=AudioEncoding.LINEAR_PCM,
                    )
                    event_loop.call_soon_threadsafe(_started, responses)
                    for response in responses:
                        if stopped.is_set():
                            break
                        event_loop.call_soon_threadsafe(_push, response.audio)
                    event_loop.call_soon_threadsafe(read_deadline.complete_response)
            except Exception as exc:
                error = exc
            finally:
                with contextlib.suppress(RuntimeError):  # The caller's loop may have shut down.
                    event_loop.call_soon_threadsafe(_finish, error)

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
            async with read_deadline.watch():
                await asyncio.gather(*tasks, asyncio.shield(done_fut))
            output_emitter.end_segment()
        except APITimeoutError:
            raise
        except Exception as exc:
            raise APIConnectionError("NVIDIA synthesis failed") from exc
        finally:
            stopped.set()
            if active_call is not None:
                active_call.cancel()
            token_q.put(None)
            await utils.aio.gracefully_cancel(*tasks)
            if not done_fut.done():
                logger.warning("NVIDIA synthesis cleanup incomplete; SDK worker retains ownership")
            await sent_tokenizer_stream.aclose()
