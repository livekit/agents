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
import ssl
import weakref
from dataclasses import dataclass, replace
from typing import Any

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from . import qwen3_tts
from .models import TTSModels

ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

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
        extra_config: dict[str, Any] | None = None,
        http_session: aiohttp.ClientSession | None = None,
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
            http_session: Optional aiohttp session to reuse.

        Raises:
            ValueError: If the API key or endpoint cannot be resolved, or if
                ``model="qwen3-tts"`` without a ``voice``.
        """
        self._model_name = model
        self._qwen3: qwen3_tts._Qwen3Backend | None = None

        if model == "qwen3-tts":
            self._qwen3 = qwen3_tts._Qwen3Backend(
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
                sample_rate=qwen3_tts.SAMPLE_RATE,
                num_channels=qwen3_tts.NUM_CHANNELS,
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

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

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

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> tts.ChunkedStream:
        if self._qwen3 is not None:
            # Qwen3-TTS is streaming-only; the framework helper disables retries
            # on the inner stream and forwards timed transcripts.
            return self._synthesize_with_stream(text, conn_options=conn_options)
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> tts.SynthesizeStream:
        stream: tts.SynthesizeStream
        if self._qwen3 is not None:
            stream = qwen3_tts.Qwen3SynthesizeStream(
                tts=self, backend=self._qwen3, conn_options=conn_options
            )
        else:
            stream = SynthesizeStream(tts=self, conn_options=conn_options)
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
    ) -> None:
        super().__init__(
            tts=tts,
            input_text=input_text,
            conn_options=conn_options,
        )

        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        try:
            async with self._tts._ensure_session().post(
                self._tts._model_endpoint,
                headers={
                    "Authorization": f"Api-Key {self._tts._api_key}",
                },
                json={
                    "prompt": self._input_text,
                    "voice": self._opts.voice,
                    "temperature": self._opts.temperature,
                    "language": self._opts.language,
                },
                timeout=aiohttp.ClientTimeout(total=30, sock_connect=self._conn_options.timeout),
                ssl=ssl_context,
            ) as resp:
                resp.raise_for_status()

                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=24000,
                    num_channels=1,
                    mime_type="audio/pcm",
                )

                async for data, _ in resp.content.iter_chunks():
                    output_emitter.push(data)

                output_emitter.flush()
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=None, body=None
            ) from None
        except Exception as e:
            raise APIConnectionError() from e


class SynthesizeStream(tts.SynthesizeStream):
    def __init__(
        self,
        *,
        tts: TTS,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=24000,
            num_channels=1,
            mime_type="audio/pcm",
            stream=True,
        )

        async def _send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    continue
                self._mark_started()
                await ws.send_str(data)
            await ws.send_str(_END_SENTINEL)

        async def _recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            output_emitter.start_segment(segment_id=request_id)
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.BINARY:
                    output_emitter.push(msg.data)
                elif msg.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    break
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    raise APIConnectionError()
            output_emitter.end_input()

        try:
            async with self._tts._ensure_session().ws_connect(
                self._tts._model_endpoint,
                headers={"Authorization": f"Api-Key {self._tts._api_key}"},
                ssl=ssl_context,
            ) as ws:
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
                    await utils.aio.gracefully_cancel(*tasks)
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=None, body=None
            ) from None
        except (APIConnectionError, APIStatusError, APITimeoutError):
            raise
        except Exception as e:
            raise APIConnectionError() from e
