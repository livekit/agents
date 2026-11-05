# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import inspect
import os
import queue
import threading
from collections.abc import Mapping
from dataclasses import dataclass, replace
from os import PathLike
from pathlib import Path
from typing import Any, Literal

import grpc
import riva.client
from riva.client.proto.riva_audio_pb2 import AudioEncoding

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
    tokenize,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from . import auth

InferenceMode = Literal["online", "offline"]


@dataclass
class TTSOptions:
    voice: str
    function_id: str
    server: str
    sample_rate: int
    use_ssl: bool
    language_code: str
    word_tokenizer: tokenize.WordTokenizer | tokenize.SentenceTokenizer
    audio_prompt_file: str | PathLike[str] | None
    quality: int | None
    options: dict[str, Any]
    inference_mode: InferenceMode


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        server: str = "grpc.nvcf.nvidia.com:443",
        voice: str = "Magpie-Multilingual.EN-US.Leo",
        function_id: str = "877104f7-e885-42b9-8de8-f6e4c6303969",
        language_code: str = "en-US",
        sample_rate: int = 16000,
        use_ssl: bool = True,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        audio_prompt_file: str | PathLike[str] | None = None,
        quality: int | None = None,
        word_tokenizer: tokenize.WordTokenizer | tokenize.SentenceTokenizer | None = None,
        options: dict[str, Any] | None = None,
        inference_mode: InferenceMode = "online",
    ):
        """Create an NVIDIA Speech synthesis client.

        Args:
            server: NVIDIA Speech gRPC endpoint.
            voice: Voice name exposed by the deployed synthesis model.
            function_id: NVIDIA-hosted NVCF function ID. Local deployments may use
                the value expected by their gateway.
            language_code: Synthesis language code, such as ``"en-US"``.
            sample_rate: Output PCM sample rate in Hz.
            use_ssl: Whether to use TLS for the gRPC connection.
            api_key: NVIDIA API key. When omitted, reads ``NVIDIA_API_KEY``.
            audio_prompt_file: Reference-audio path for models supporting zero-shot
                voice prompting.
            quality: Zero-shot synthesis quality accepted by compatible clients.
            word_tokenizer: Tokenizer used to split text submitted through ``stream()``
                into synthesis requests.
            options: Additional keyword arguments passed to the installed NVIDIA
                synthesis client after compatibility validation.
            inference_mode: ``"online"`` uses streaming synthesis; ``"offline"`` uses
                the batch synthesis RPC.

        Raises:
            ValueError: If the inference mode is invalid or hosted authentication is
                enabled without an API key.
        """
        if inference_mode not in ("online", "offline"):
            raise ValueError("inference_mode must be either 'online' or 'offline'")

        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=sample_rate,
            num_channels=1,
        )

        if is_given(api_key):
            self.nvidia_api_key = api_key
        else:
            self.nvidia_api_key = os.getenv("NVIDIA_API_KEY")
            if use_ssl and not self.nvidia_api_key:
                raise ValueError(
                    "NVIDIA_API_KEY is not set while using SSL. Either pass api_key parameter, "
                    "set NVIDIA_API_KEY environment variable or disable SSL and use a locally "
                    "hosted NVIDIA Speech service."
                )

        self._opts = TTSOptions(
            voice=voice,
            function_id=function_id,
            server=server,
            sample_rate=sample_rate,
            use_ssl=use_ssl,
            language_code=language_code,
            word_tokenizer=word_tokenizer or tokenize.blingfire.SentenceTokenizer(),
            audio_prompt_file=audio_prompt_file,
            quality=quality,
            options=dict(options or {}),
            inference_mode=inference_mode,
        )
        self._tts_service: riva.client.SpeechSynthesisService | None = None

    @property
    def provider(self) -> str:
        return "NVIDIA Speech"

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

    def _synthesize(self, text: str, opts: TTSOptions):
        service = self._ensure_session()
        synthesize_method = (
            service.synthesize_online if opts.inference_mode == "online" else service.synthesize
        )
        parameters = inspect.signature(synthesize_method).parameters
        kwargs = dict(opts.options)
        kwargs.update(
            {
                "sample_rate_hz": opts.sample_rate,
                "encoding": AudioEncoding.LINEAR_PCM,
            }
        )
        if opts.audio_prompt_file is not None:
            _set_compatible_option(
                kwargs,
                parameters=parameters,
                names=("zero_shot_audio_prompt_file", "audio_prompt_file"),
                value=Path(opts.audio_prompt_file),
                option_name="audio_prompt_file",
            )
        if opts.quality is not None:
            _set_compatible_option(
                kwargs,
                parameters=parameters,
                names=("zero_shot_quality", "quality"),
                value=opts.quality,
                option_name="quality",
            )
        if opts.inference_mode == "offline":
            kwargs["future"] = False

        if not any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
            unsupported = kwargs.keys() - parameters.keys()
            if unsupported:
                names = ", ".join(sorted(unsupported))
                raise ValueError(
                    f"The installed nvidia-riva-client does not support TTS option(s): {names}"
                )

        response = synthesize_method(
            text.lstrip("\n").strip(),
            opts.voice,
            opts.language_code,
            **kwargs,
        )
        if opts.inference_mode == "online":
            return response
        return iter((response,))

    def list_voices(self) -> dict:
        service = self._ensure_session()
        config_response = service.stub.GetRivaSynthesisConfig(
            riva.client.proto.riva_tts_pb2.RivaSynthesisConfigRequest()
        )
        tts_models = {}
        for model_config in config_response.model_config:
            language_codes = [
                code.strip()
                for code in model_config.parameters.get("language_code", "").split(",")
                if code.strip()
            ] or ["unknown"]
            voice_name = model_config.parameters.get("voice_name", "unknown")
            subvoices_str = model_config.parameters.get("subvoices", "")

            if subvoices_str:
                subvoices = [voice.split(":")[0] for voice in subvoices_str.split(",")]
                full_voice_names = [voice_name + "." + subvoice for subvoice in subvoices]
            else:
                full_voice_names = [voice_name]

            for language_code in language_codes:
                voices = tts_models.setdefault(language_code, {"voices": []})["voices"]
                for full_voice_name in full_voice_names:
                    if full_voice_name not in voices:
                        voices.append(full_voice_name)

        tts_models = dict(sorted(tts_models.items()))
        return tts_models

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> tts.ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> tts.SynthesizeStream:
        return SynthesizeStream(tts=self, conn_options=conn_options, opts=self._opts)


class ChunkedStream(tts.ChunkedStream):
    """Synthesis stream for a complete NVIDIA Speech text request."""

    def __init__(
        self,
        *,
        tts: TTS,
        input_text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
        )

        done_fut: asyncio.Future[None] = asyncio.Future()
        event_loop = asyncio.get_running_loop()

        def _synthesize_worker() -> None:
            error: Exception | None = None
            try:
                for response in self._tts._synthesize(self._input_text, self._opts):
                    event_loop.call_soon_threadsafe(output_emitter.push, response.audio)
            except Exception as e:
                error = e
            finally:
                event_loop.call_soon_threadsafe(_complete_future, done_fut, error)

        synthesize_thread = threading.Thread(
            target=_synthesize_worker,
            name="nvidia-tts-chunked-synthesize",
            daemon=True,
        )
        synthesize_thread.start()
        await done_fut
        output_emitter.flush()


class SynthesizeStream(tts.SynthesizeStream):
    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions, opts: TTSOptions):
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        context_id = utils.shortuuid()
        sent_tokenizer_stream = self._opts.word_tokenizer.stream()
        token_q: queue.Queue = queue.Queue()
        event_loop = asyncio.get_running_loop()

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
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    sent_tokenizer_stream.flush()
                    break
                sent_tokenizer_stream.push_text(data)
            sent_tokenizer_stream.end_input()

        async def _process_segments() -> None:
            async for word_stream in sent_tokenizer_stream:
                token = word_stream.token
                if not token.strip():
                    continue
                self._mark_started()
                token_q.put(token)
            token_q.put(None)

        def _synthesize_worker() -> None:
            error: Exception | None = None
            try:
                while True:
                    token = token_q.get()

                    if not token:
                        break

                    for response in self._tts._synthesize(token, self._opts):
                        event_loop.call_soon_threadsafe(output_emitter.push, response.audio)

            except Exception as e:
                error = e
            finally:
                event_loop.call_soon_threadsafe(_complete_future, done_fut, error)

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
            token_q.put(None)
            try:
                await done_fut
            finally:
                output_emitter.end_segment()
                await sent_tokenizer_stream.aclose()


def _complete_future(fut: asyncio.Future[None], error: Exception | None) -> None:
    if fut.done():
        return
    if error is not None:
        fut.set_exception(_to_tts_api_error(error, operation="NVIDIA Speech TTS request"))
    else:
        fut.set_result(None)


def _set_compatible_option(
    kwargs: dict[str, Any],
    *,
    parameters: Mapping[str, inspect.Parameter],
    names: tuple[str, ...],
    value: Any,
    option_name: str,
) -> None:
    for name in names:
        if name in parameters:
            kwargs[name] = value
            return

    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        kwargs[names[0]] = value
        return

    raise ValueError(
        f"The installed nvidia-riva-client does not support the TTS {option_name} option"
    )


def _to_tts_api_error(error: Exception, *, operation: str) -> APIError:
    if isinstance(error, APIError):
        return error

    if isinstance(error, grpc.RpcError):
        code = error.code()
        details = error.details() or str(error)
        if code == grpc.StatusCode.DEADLINE_EXCEEDED:
            return APITimeoutError(f"{operation} timed out: {details}")
        if code == grpc.StatusCode.CANCELLED:
            return APIConnectionError(f"{operation} was cancelled by NVIDIA Speech")

        status_codes = {
            grpc.StatusCode.INVALID_ARGUMENT: 400,
            grpc.StatusCode.NOT_FOUND: 404,
            grpc.StatusCode.PERMISSION_DENIED: 403,
            grpc.StatusCode.RESOURCE_EXHAUSTED: 429,
            grpc.StatusCode.FAILED_PRECONDITION: 400,
            grpc.StatusCode.OUT_OF_RANGE: 400,
            grpc.StatusCode.UNIMPLEMENTED: 501,
            grpc.StatusCode.INTERNAL: 500,
            grpc.StatusCode.UNAVAILABLE: 503,
            grpc.StatusCode.UNAUTHENTICATED: 401,
        }
        return APIStatusError(
            f"{operation} failed: {details}",
            status_code=status_codes.get(code, -1),
            retryable=code
            in {
                grpc.StatusCode.UNKNOWN,
                grpc.StatusCode.RESOURCE_EXHAUSTED,
                grpc.StatusCode.INTERNAL,
                grpc.StatusCode.UNAVAILABLE,
            },
        )

    if isinstance(error, (TypeError, ValueError)):
        return APIStatusError(f"{operation} failed: {error}", status_code=400, retryable=False)

    return APIConnectionError(f"{operation} failed: {error}")
