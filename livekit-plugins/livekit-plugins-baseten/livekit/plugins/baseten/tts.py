# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
import os
import weakref
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace

from pydantic import JsonValue

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    http,
    tts,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from ._qwen3_tts import NUM_CHANNELS, SAMPLE_RATE, Qwen3Backend
from .models import TTSModels

_END_SENTINEL = "__END__"


@dataclass
class _TTSOptions:
    language: str
    voice: str
    temperature: float
    max_tokens: int
    buffer_size: int


class TTS(tts.TTS):
    """Text-to-speech for a Baseten-hosted model.

    ``model`` selects the wire protocol, since Baseten's TTS deployments do not
    share one. ``orpheus`` (the default) is the existing behaviour; ``qwen3-tts``
    speaks the session.config protocol and takes a registered voice clone.
    """

    def __init__(
        self,
        *,
        model: TTSModels = "orpheus",
        api_key: str | None = None,
        model_endpoint: str | None = None,
        model_id: str | None = None,
        chain_id: str | None = None,
        voice: NotGivenOr[str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        temperature: float = 0.6,
        max_tokens: int = 2000,
        buffer_size: int = 10,
        task_type: str = "Base",
        instructions: str | None = None,
        max_new_tokens: int | None = None,
        initial_codec_chunk_frames: int | None = None,
        x_vector_only_mode: bool | None = None,
        ref_audio: str | None = None,
        ref_text: str | None = None,
        word_timestamps: bool = False,
        extra_config: Mapping[str, JsonValue] | None = None,
        http_session: http.HttpSession | None = None,
    ) -> None:
        """
        Initialize the Baseten TTS.

        Args:
            model: Which deployment protocol to speak — ``orpheus`` (default) or
                ``qwen3-tts``. They are not interchangeable.
            api_key: Baseten API key, or ``BASETEN_API_KEY`` env var.
            model_id: Baseten truss model ID; builds the endpoint URL for you
                (``qwen3-tts`` only).
            chain_id: Baseten chain ID; builds the endpoint URL for you
                (``qwen3-tts`` only).
            task_type: ``qwen3-tts`` only. ``Base`` for the voice-cloning
                checkpoint; ``CustomVoice``/``VoiceDesign`` deployments expose
                preset speaker names.
            instructions: ``qwen3-tts`` style prompt (CustomVoice/VoiceDesign).
            max_new_tokens: ``qwen3-tts`` cap on audio tokens per sentence.
            initial_codec_chunk_frames: ``qwen3-tts`` first-chunk size; larger
                trades time-to-first-audio for onset quality.
            x_vector_only_mode: ``qwen3-tts``; condition on the speaker embedding
                alone and skip in-context learning from the reference.
            ref_audio: ``qwen3-tts`` reference clip for inline cloning — an
                http(s) URL or a local file path.
            ref_text: Transcript of ``ref_audio``; improves clone fidelity.
            word_timestamps: ``qwen3-tts``; forward word-level alignment to
                LiveKit as a timed transcript.
            extra_config: ``qwen3-tts``; merged into ``session.config`` last.
            model_endpoint: Baseten model endpoint, or ``BASETEN_MODEL_ENDPOINT`` env var.
                Pass a ``wss://`` URL for streaming or an ``https://`` URL for non-streaming.
            voice: Speaker voice. Defaults to ``tara`` for ``orpheus``. Required
                for ``qwen3-tts``, where it names a registered voice clone — that
                checkpoint ships no built-in speakers (see `register_voice`).
            language: Language code. Defaults to ``en`` for ``orpheus`` and
                ``Auto`` for ``qwen3-tts``.
            temperature: Sampling temperature. Defaults to 0.6.
            max_tokens: Maximum tokens for generation. Defaults to 2000.
            buffer_size: Number of words per chunk for streaming. Defaults to 10.
            http_session: Optional HttpSession to reuse.

        Raises:
            ValueError: If the API key or endpoint cannot be resolved, or if
                ``model="qwen3-tts"`` without a ``voice``.
        """
        self._model_name = model
        self._qwen3: Qwen3Backend | None = None

        if model == "qwen3-tts":
            self._qwen3 = Qwen3Backend(
                model_endpoint=model_endpoint,
                model_id=model_id,
                chain_id=chain_id,
                api_key=api_key,
                voice=voice if is_given(voice) else "",
                task_type=task_type,
                language=language if is_given(language) else "Auto",
                instructions=instructions,
                max_new_tokens=max_new_tokens,
                initial_codec_chunk_frames=initial_codec_chunk_frames,
                x_vector_only_mode=x_vector_only_mode,
                ref_audio=ref_audio,
                ref_text=ref_text,
                word_timestamps=word_timestamps,
                extra_config=extra_config,
                http_session=http_session,
            )
            super().__init__(
                capabilities=tts.TTSCapabilities(
                    streaming=True, aligned_transcript=word_timestamps
                ),
                sample_rate=SAMPLE_RATE,
                num_channels=NUM_CHANNELS,
            )
            self._session = http_session
            self._streams = weakref.WeakSet[tts.SynthesizeStream]()
            return

        api_key = api_key or os.environ.get("BASETEN_API_KEY")

        if not api_key:
            raise ValueError(
                "Baseten API key is required. "
                "Pass one in via the `api_key` parameter, "
                "or set it as the `BASETEN_API_KEY` environment variable"
            )

        model_endpoint = model_endpoint or os.environ.get("BASETEN_MODEL_ENDPOINT")

        if not model_endpoint:
            raise ValueError(
                "model_endpoint is required. "
                "Provide it via the constructor or BASETEN_MODEL_ENDPOINT env var."
            )

        is_ws = model_endpoint.startswith(("wss://", "ws://"))

        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=is_ws),
            sample_rate=24000,
            num_channels=1,
        )

        self._api_key = api_key
        self._model_endpoint = model_endpoint

        self._opts = _TTSOptions(
            voice=voice if is_given(voice) else "tara",
            language=language if is_given(language) else "en",
            temperature=temperature,
            max_tokens=max_tokens,
            buffer_size=buffer_size,
        )
        self._session = http_session
        self._streams = weakref.WeakSet[tts.SynthesizeStream]()

    @property
    def model(self) -> str:
        return self._model_name

    @property
    def provider(self) -> str:
        return "Baseten"

    def _ensure_session(self) -> http.HttpSession:
        if not self._session:
            self._session = http.http_session()

        return self._session

    def update_options(
        self,
        *,
        voice: NotGivenOr[str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        max_tokens: NotGivenOr[int] = NOT_GIVEN,
        buffer_size: NotGivenOr[int] = NOT_GIVEN,
    ) -> None:
        if self._qwen3 is not None:
            self._qwen3.update_options(voice=voice, language=language)
            return
        if is_given(voice):
            self._opts.voice = voice
        if is_given(language):
            self._opts.language = language
        if is_given(temperature):
            self._opts.temperature = temperature
        if is_given(max_tokens):
            self._opts.max_tokens = max_tokens
        if is_given(buffer_size):
            self._opts.buffer_size = buffer_size

    def _synthesize(
        self, chunk: tts.SpeechChunk, *, conn_options: APIConnectOptions
    ) -> tts.ChunkedStream:
        if self._qwen3 is not None:
            # Qwen3-TTS is streaming-only; the framework helper disables retries
            # on the inner stream and forwards timed transcripts.
            return self._synthesize_with_stream(chunk, conn_options=conn_options)
        return ChunkedStream(
            tts=self,
            input_text=chunk.text,
            conn_options=conn_options,
            opts=replace(self._opts),
            ensure_session=self._ensure_session,
            model_endpoint=self._model_endpoint,
            api_key=self._api_key,
        )

    def _stream(self, *, conn_options: APIConnectOptions) -> tts.SynthesizeStream:
        stream: tts.SynthesizeStream
        if self._qwen3 is not None:
            stream = self._qwen3.make_stream(self, conn_options=conn_options)
        else:
            stream = SynthesizeStream(
                tts=self,
                conn_options=conn_options,
                opts=replace(self._opts),
                ensure_session=self._ensure_session,
                model_endpoint=self._model_endpoint,
                api_key=self._api_key,
            )
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()
        if self._qwen3 is not None:
            await self._qwen3.aclose()


class ChunkedStream(tts.ChunkedStream):
    def __init__(
        self,
        *,
        tts: TTS,
        input_text: str,
        conn_options: APIConnectOptions,
        opts: _TTSOptions,
        ensure_session: Callable[[], http.HttpSession],
        model_endpoint: str,
        api_key: str,
    ) -> None:
        super().__init__(
            tts=tts,
            input_text=input_text,
            conn_options=conn_options,
        )

        self._opts = opts
        self._ensure_session = ensure_session
        self._model_endpoint = model_endpoint
        self._api_key = api_key

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        try:
            async with self._ensure_session().post(
                self._model_endpoint,
                headers={
                    "Authorization": f"Api-Key {self._api_key}",
                },
                json={
                    "prompt": self._input_text,
                    "voice": self._opts.voice,
                    "temperature": self._opts.temperature,
                    "language": self._opts.language,
                },
                conn_options=self._conn_options,
            ) as resp:
                await resp.raise_for_status()

                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=24000,
                    num_channels=1,
                    mime_type="audio/pcm",
                )

                async for data in resp.iter_bytes():
                    output_emitter.push(data)

                output_emitter.flush()
        except TimeoutError:
            raise APITimeoutError() from None
        except (APIStatusError, APIConnectionError):
            raise
        except Exception as e:
            raise APIConnectionError() from e


class SynthesizeStream(tts.SynthesizeStream):
    def __init__(
        self,
        *,
        tts: TTS,
        conn_options: APIConnectOptions,
        opts: _TTSOptions,
        ensure_session: Callable[[], http.HttpSession],
        model_endpoint: str,
        api_key: str,
    ) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._opts = opts
        self._ensure_session = ensure_session
        self._model_endpoint = model_endpoint
        self._api_key = api_key

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=24000,
            num_channels=1,
            mime_type="audio/pcm",
            stream=True,
        )

        read_deadline = self._core_attempts

        async def _send_task(ws: http.WebSocket) -> None:
            async for data in self.input:
                if isinstance(data, self.Flush):
                    continue
                self.mark_started()
                await ws.send_str(data.text + " ")
                read_deadline.expect_response()
            await ws.send_str(_END_SENTINEL)
            read_deadline.expect_response(final=True)

        async def _recv_task(ws: http.WebSocket) -> None:
            output_emitter.start_segment(segment_id=request_id)
            async for msg in ws:
                if msg.type is http.WebSocketMessageType.BINARY:
                    read_deadline.audio_received()
                    output_emitter.push(msg.binary)
            output_emitter.end_input()

        try:
            ws = await self._ensure_session().ws_connect(
                self._model_endpoint,
                headers={"Authorization": f"Api-Key {self._api_key}"},
                conn_options=self._conn_options,
            )
            try:
                await ws.send_json(
                    {
                        "voice": self._opts.voice,
                        "max_tokens": self._opts.max_tokens,
                        "buffer_size": self._opts.buffer_size,
                    }
                )

                tasks = [
                    asyncio.create_task(_send_task(ws)),
                    asyncio.create_task(_recv_task(ws)),
                ]
                try:
                    await asyncio.gather(*tasks)
                finally:
                    await utils.aio.cancel_and_wait(*tasks)
            finally:
                await ws.close()
        except TimeoutError:
            raise APITimeoutError() from None
        except (APIConnectionError, APIStatusError):
            raise
        except Exception as e:
            raise APIConnectionError() from e
