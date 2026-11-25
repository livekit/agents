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

import os
import weakref
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TypedDict, assert_never

from pydantic import JsonValue

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APITimeoutError,
    http,
    tokenize,
    tts,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from ._endpoint import resolve_endpoint
from ._qwen3_tts import NUM_CHANNELS, SAMPLE_RATE, Qwen3Options, Qwen3Pool, Qwen3SynthesizeStream
from .models import TTSModels

_END_SENTINEL = "__END__"


@dataclass
class _OrpheusOptions:
    language: str
    voice: str
    temperature: float
    max_tokens: int
    buffer_size: int


class _OrpheusSetup(TypedDict):
    voice: str
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
        api_key = api_key or os.environ.get("BASETEN_API_KEY")
        if not api_key:
            raise ValueError(
                "Baseten API key is required. "
                "Pass one in via the `api_key` parameter, "
                "or set it as the `BASETEN_API_KEY` environment variable"
            )
        self._model_name = model
        self._api_key = api_key
        self._session = http_session
        self._streams = weakref.WeakSet[tts.SynthesizeStream]()
        self._opts: _OrpheusOptions | Qwen3Options
        self._pool: Qwen3Pool | None = None

        if model == "qwen3-tts":
            endpoint = resolve_endpoint(
                model_endpoint, model_id, chain_id, "BASETEN_MODEL_ENDPOINT"
            )
            if not is_given(voice) or not voice:
                raise ValueError("`voice` is required: the Base checkpoint has no presets.")
            self._opts = Qwen3Options(
                voice=voice,
                task_type=task_type,
                language=language if is_given(language) else "Auto",
                instructions=instructions,
                max_new_tokens=max_new_tokens,
                initial_codec_chunk_frames=initial_codec_chunk_frames,
                x_vector_only_mode=x_vector_only_mode,
                ref_audio=ref_audio,
                ref_text=ref_text,
                word_timestamps=word_timestamps,
                extra_config=dict(extra_config or {}),
            )
            self._pool = Qwen3Pool(model_endpoint=endpoint, api_key=api_key, opts=self._opts)
            capabilities = tts.TTSCapabilities(streaming=True, aligned_transcript=word_timestamps)
        else:
            endpoint = model_endpoint or os.environ.get("BASETEN_MODEL_ENDPOINT") or ""
            if not endpoint:
                raise ValueError(
                    "model_endpoint is required. "
                    "Provide it via the constructor or BASETEN_MODEL_ENDPOINT env var."
                )
            self._opts = _OrpheusOptions(
                voice=voice if is_given(voice) else "tara",
                language=language if is_given(language) else "en",
                temperature=temperature,
                max_tokens=max_tokens,
                buffer_size=buffer_size,
            )
            # a wss:// endpoint streams; an https:// one synthesizes each request whole
            capabilities = tts.TTSCapabilities(streaming=endpoint.startswith(("wss://", "ws://")))
        self._model_endpoint = endpoint
        super().__init__(
            capabilities=capabilities, sample_rate=SAMPLE_RATE, num_channels=NUM_CHANNELS
        )

    @property
    def model(self) -> str:
        return self._model_name

    @property
    def provider(self) -> str:
        return "baseten"

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
        sentence_tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
        pacing: NotGivenOr[tts.SentenceStreamPacer | None] = NOT_GIVEN,
    ) -> None:
        """``temperature``, ``max_tokens`` and ``buffer_size`` are orpheus options; a qwen3-tts
        socket takes the new voice and language from the next turn."""
        with self._updating_options(sentence_tokenizer=sentence_tokenizer, pacing=pacing):
            opts = self._opts
            if is_given(voice):
                opts.voice = voice
            if is_given(language):
                opts.language = language
            if isinstance(opts, _OrpheusOptions):
                if is_given(temperature):
                    opts.temperature = temperature
                if is_given(max_tokens):
                    opts.max_tokens = max_tokens
                if is_given(buffer_size):
                    opts.buffer_size = buffer_size
        if (
            not isinstance(opts, _OrpheusOptions)
            and self._pool is not None
            and (is_given(voice) or is_given(language))
        ):
            # the session config is sticky for a socket's life
            self._pool.invalidate()

    def _synthesize(
        self, chunk: tts.SpeechChunk, *, conn_options: APIConnectOptions
    ) -> tts.ChunkedStream:
        match self._opts:
            case _OrpheusOptions():
                return ChunkedStream(
                    tts=self,
                    input_text=chunk.text,
                    conn_options=conn_options,
                    opts=replace(self._opts),
                )
            case Qwen3Options():
                # Qwen3-TTS is streaming-only
                return self._synthesize_with_stream(chunk, conn_options=conn_options)
            case _:
                assert_never(self._opts)

    def _stream(self, *, conn_options: APIConnectOptions) -> tts.SynthesizeStream:
        stream: tts.SynthesizeStream
        match self._opts:
            case _OrpheusOptions():
                stream = SynthesizeStream(
                    tts=self, conn_options=conn_options, opts=replace(self._opts)
                )
            case Qwen3Options():
                assert self._pool is not None
                stream = Qwen3SynthesizeStream(
                    tts=self,
                    pool=self._pool,
                    word_timestamps=self._opts.word_timestamps,
                    conn_options=conn_options,
                )
            case _:
                assert_never(self._opts)
        self._streams.add(stream)
        return stream

    def prewarm(self) -> None:
        if self._pool is not None:
            self._pool.prewarm()

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()
        if self._pool is not None:
            await self._pool.aclose()


class ChunkedStream(tts.ChunkedStream):
    def __init__(
        self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions, opts: _OrpheusOptions
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._baseten = tts
        self._opts = opts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        try:
            async with self._baseten._ensure_session().post(
                self._baseten._model_endpoint,
                headers={"Authorization": f"Api-Key {self._baseten._api_key}"},
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
                    sample_rate=SAMPLE_RATE,
                    num_channels=NUM_CHANNELS,
                    mime_type="audio/pcm",
                )

                async for data in resp.iter_bytes():
                    output_emitter.push(data)

                output_emitter.flush()
        except TimeoutError:
            raise APITimeoutError() from None
        except APIError:
            raise
        except Exception as e:
            raise APIConnectionError() from e


class SynthesizeStream(tts.WebSocketStream[bytes | str, _OrpheusSetup | str]):
    """The Orpheus truss: a session per segment, its text as raw frames, ended by ``__END__``,
    after which the server sends the audio and closes."""

    connection_per_segment = True

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions, opts: _OrpheusOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._baseten = tts
        self._opts = opts
        self._segment_id = ""

    async def connect(self, conn_options: APIConnectOptions) -> tts.WebSocketTarget:
        return tts.WebSocketTarget(
            self._baseten._model_endpoint,
            headers={"Authorization": f"Api-Key {self._baseten._api_key}"},
        )

    def setup(self) -> _OrpheusSetup:
        return _OrpheusSetup(
            voice=self._opts.voice,
            max_tokens=self._opts.max_tokens,
            buffer_size=self._opts.buffer_size,
        )

    def encode(self, item: tts.SegmentInput | tts.EndOfText) -> str | None:
        match item:
            case tts.SegmentStart():
                self._segment_id = item.segment_id
                return None
            case tts.SegmentText():
                return f"{item.chunk.text} "
            case tts.SegmentEnd():
                return _END_SENTINEL
            case tts.EndOfText():
                return None
            case _:
                assert_never(item)

    def on_message(self, message: bytes | str) -> None:
        # text frames carry nothing the stream reads
        if isinstance(message, bytes):
            self.report(tts.SegmentAudio(self._segment_id, message))
