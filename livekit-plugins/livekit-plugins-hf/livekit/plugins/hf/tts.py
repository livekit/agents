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
import threading
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from livekit import rtc
from livekit.agents import APIConnectionError, tts
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from livekit.agents.utils import shortuuid

from . import _hosted
from ._cache import get_or_load, not_cached_error
from .log import logger

DEFAULT_TTS_MODEL = "canopylabs/orpheus-3b-0.1-ft"
DEFAULT_VOICE = "tara"
SNAC_MODEL = "hubertsiuzdak/snac_24khz"

# Orpheus prompt and audio token ids (shared by the Llama-based Orpheus fine-tunes)
_START_OF_HUMAN = 128259
_END_OF_TEXT = 128009
_END_OF_HUMAN = 128260
_START_OF_SPEECH = 128257
_END_OF_SPEECH = 128258
_AUDIO_TOKEN_BASE = 128266
_CODEBOOK_SIZE = 4096
_CODES_PER_FRAME = 7
# one SNAC frame (7 codes) decodes to 2048 samples at 24 kHz, about 85 ms
_SAMPLES_PER_FRAME = 2048
_ORPHEUS_SAMPLE_RATE = 24000

Family = Literal["orpheus", "waveform"]


@dataclass(frozen=True)
class _TTSKey:
    model: str
    revision: str | None
    device: str
    dtype: str
    trust_remote_code: bool


@dataclass
class _LoadedTTS:
    family: Family
    model: Any
    tokenizer: Any
    sample_rate: int
    codec: Any = None
    # generate() is not safe to run concurrently on one model, and jobs in a thread
    # executor share the model across event loops, so this is a thread lock
    lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass
class _TTSOptions:
    model: str
    revision: str | None
    device: str
    dtype: str
    voice: str | None
    temperature: float
    top_p: float
    repetition_penalty: float
    max_new_tokens: int
    chunk_frames: int
    trust_remote_code: bool
    local_files_only: bool

    @property
    def key(self) -> _TTSKey:
        return _TTSKey(
            model=self.model,
            revision=self.revision,
            device=self.device,
            dtype=self.dtype,
            trust_remote_code=self.trust_remote_code,
        )


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        model: str = DEFAULT_TTS_MODEL,
        voice: str | None = None,
        revision: str | None = None,
        device: str = "auto",
        dtype: str = "auto",
        sample_rate: int = _ORPHEUS_SAMPLE_RATE,
        temperature: float = 0.6,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
        max_new_tokens: int = 1200,
        chunk_frames: int = 4,
        trust_remote_code: bool = False,
        local_files_only: bool = True,
        provider: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        """Run a Hugging Face text-to-speech model locally with ``transformers``.

        Two model families are supported, picked from the model's config:

        - **Orpheus** (Llama models that speak in SNAC audio codes), such as
          ``canopylabs/orpheus-3b-0.1-ft`` and its fine-tunes. Audio is streamed while
          the model generates, ``chunk_frames`` SNAC frames (about 85 ms each) at a time.
          Needs the ``snac`` package (``pip install "livekit-plugins-hf[orpheus]"``).
        - **Text-to-waveform** models loaded by ``AutoModelForTextToWaveform``, such as
          the VITS-based ``facebook/mms-tts-*`` voices for about 1,100 languages.

        Each reply is split into sentences by the ``AgentSession`` and synthesized one
        sentence at a time. Weights load in a background thread when the session
        prewarms the TTS, or synchronously with :meth:`load`.

        Args:
            model: Hugging Face Hub repo id or path to a local model directory.
            voice: Orpheus speaker, prepended to the text as ``"{voice}: {text}"``, e.g.
                ``"tara"`` (used when unset with the default model) or
                ``"salt_lug_0001"``. Ignored by local text-to-waveform models. Sent as
                ``voice`` to a provider or endpoint.
            revision: Hub revision (branch, tag or commit) to load.
            device: ``device_map`` passed to ``from_pretrained``.
            dtype: Torch dtype name, e.g. ``"auto"``, ``"float16"`` or ``"bfloat16"``.
            sample_rate: Output sample rate. Model audio is resampled to it.
            temperature: Sampling temperature for Orpheus models.
            top_p: Nucleus sampling probability mass for Orpheus models.
            repetition_penalty: Repetition penalty for Orpheus models.
            max_new_tokens: Maximum audio tokens per sentence for Orpheus models
                (1200 is about 14 s of speech).
            chunk_frames: SNAC frames decoded per audio chunk for Orpheus models.
            trust_remote_code: Allow models that ship their own modeling code.
            local_files_only: Load only from the local Hugging Face cache and never
                download at runtime.
            provider: Hugging Face Inference Provider that runs ``model`` instead of
                this process, e.g. ``"auto"``, ``"fal-ai"`` or ``"deepinfra"``.
            base_url: URL of a dedicated endpoint serving the model, such as a Hugging
                Face Inference Endpoint.
            api_key: Token for the provider or endpoint. Defaults to ``HF_TOKEN``.
            extra_body: Extra request parameters for the provider or endpoint.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=1,
        )
        self._opts = _TTSOptions(
            model=model,
            revision=revision,
            device=device,
            dtype=dtype,
            voice=voice,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            chunk_frames=max(1, chunk_frames),
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        self._hosted = _hosted.HostedOptions(
            provider=provider,
            base_url=base_url,
            api_key=api_key,
            extra_body=dict(extra_body or {}),
        )
        self._client: Any = None
        self._prewarm_task: asyncio.Future[Any] | None = None

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "huggingface"

    def update_options(self, *, voice: str | None = None) -> None:
        if voice is not None:
            self._opts.voice = voice

    def load(self) -> None:
        """Load the model weights into this process, blocking until they are ready.

        Does nothing when the model runs on a provider or endpoint.
        """
        if not self._hosted.enabled:
            self._get_model()

    def prewarm(self) -> None:
        if self._hosted.enabled:
            return
        # prewarm() runs on the event loop and must not block it
        if self._prewarm_task is None:
            self._prewarm_task = asyncio.ensure_future(asyncio.to_thread(self._get_model))
            self._prewarm_task.add_done_callback(_log_prewarm_error)

    def _get_model(self) -> _LoadedTTS:
        opts = self._opts
        return get_or_load(
            ("tts", opts.key), lambda: _load_tts(opts.key, local_files_only=opts.local_files_only)
        )

    def _hosted_client(self) -> Any:
        if self._client is None:
            self._client = _hosted.make_client(self._hosted)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)


class ChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._hf_tts = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        if self._hf_tts._hosted.enabled:
            await self._run_hosted(output_emitter)
            return

        output_emitter.initialize(
            request_id=shortuuid("hf_tts_"),
            sample_rate=self._hf_tts.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
        )
        try:
            loaded = await asyncio.to_thread(self._hf_tts._get_model)
            resampler = (
                rtc.AudioResampler(loaded.sample_rate, self._hf_tts.sample_rate, num_channels=1)
                if loaded.sample_rate != self._hf_tts.sample_rate
                else None
            )

            def _emit(chunk: np.ndarray) -> None:
                pcm = _to_int16(chunk)
                if pcm.size == 0:
                    return
                if resampler is None:
                    output_emitter.push(pcm.tobytes())
                    return
                frame = rtc.AudioFrame(
                    data=pcm.tobytes(),
                    sample_rate=loaded.sample_rate,
                    num_channels=1,
                    samples_per_channel=pcm.size,
                )
                for out in resampler.push(frame):
                    output_emitter.push(out.data.tobytes())

            if loaded.family == "orpheus":
                await self._run_orpheus(loaded, _emit)
            else:
                audio = await asyncio.to_thread(_synthesize_waveform, loaded, self._input_text)
                _emit(audio)

            if resampler is not None:
                for out in resampler.flush():
                    output_emitter.push(out.data.tobytes())
            output_emitter.flush()
        except APIConnectionError:
            raise
        except Exception as e:
            raise APIConnectionError(
                f"local synthesis failed: {type(e).__name__}", retryable=False
            ) from e

    async def _run_hosted(self, output_emitter: tts.AudioEmitter) -> None:
        hosted = self._hf_tts._hosted
        opts = self._hf_tts._opts
        extra_body = dict(hosted.extra_body)
        if opts.voice:
            extra_body.setdefault("voice", opts.voice)
        audio: bytes = await _hosted.call(
            asyncio.wait_for(
                self._hf_tts._hosted_client().text_to_speech(
                    self._input_text,
                    model=_hosted.request_model(hosted, opts.model),
                    extra_body=extra_body or None,
                ),
                _hosted.request_timeout(self._conn_options.timeout),
            ),
            what="hosted synthesis",
        )
        # providers return a whole encoded clip; the emitter decodes and resamples it
        output_emitter.initialize(
            request_id=shortuuid("hf_tts_"),
            sample_rate=self._hf_tts.sample_rate,
            num_channels=1,
            mime_type=_hosted.sniff_audio_mime(audio),
        )
        output_emitter.push(audio)
        output_emitter.flush()

    async def _run_orpheus(self, loaded: _LoadedTTS, emit: Any) -> None:
        opts = self._hf_tts._opts
        loop = asyncio.get_running_loop()
        token_ch: asyncio.Queue[int | None] = asyncio.Queue()
        stop_event = threading.Event()

        def _on_token(token_id: int) -> None:
            loop.call_soon_threadsafe(token_ch.put_nowait, token_id)

        def _generate_blocking() -> None:
            try:
                with loaded.lock:
                    if stop_event.is_set():
                        return
                    _generate_orpheus(loaded, self._input_text, opts, _on_token, stop_event)
            finally:
                loop.call_soon_threadsafe(token_ch.put_nowait, None)

        decoder = _SnacStreamDecoder(chunk_frames=opts.chunk_frames)
        gen_fut = loop.run_in_executor(None, _generate_blocking)
        try:
            while (token_id := await token_ch.get()) is not None:
                if (window := decoder.push(token_id)) is not None:
                    emit(await asyncio.to_thread(_decode_snac, loaded, window))
        finally:
            # stops generate() early when the stream is cancelled mid-sentence
            stop_event.set()
            await asyncio.gather(gen_fut, return_exceptions=True)

        # surface generate() errors that ended the stream early
        gen_fut.result()
        if (window := decoder.flush()) is not None:
            emit(await asyncio.to_thread(_decode_snac, loaded, window))


@dataclass
class _DecodeWindow:
    codes: list[int]
    """the SNAC frames to decode, 7 codes each, including context frames"""
    start: int
    """first sample to keep, after the left context"""
    end: int
    """sample after the last one to keep, before the right context"""


class _SnacStreamDecoder:
    """Turn a stream of Orpheus token ids into SNAC decode windows.

    SNAC is a convolutional decoder, so a frame decoded without its neighbours clicks at
    the edges. Each window adds frames of context on both sides and keeps the middle.
    """

    # measured against one full decode on real Orpheus output: a second frame of right
    # context gains 2-5 dB, more left context or bigger chunks under 0.5 dB
    def __init__(self, *, chunk_frames: int, left_context: int = 4, right_context: int = 2):
        self._chunk = chunk_frames
        self._left = left_context
        self._right = right_context
        self._codes: list[int] = []
        self._in_speech = False
        self._emitted_frames = 0

    def push(self, token_id: int) -> _DecodeWindow | None:
        if token_id == _START_OF_SPEECH:
            # a new speech segment: drop anything before it
            self._in_speech = True
            self._codes = []
            self._emitted_frames = 0
            return None
        if not self._in_speech or not (
            _AUDIO_TOKEN_BASE <= token_id < _AUDIO_TOKEN_BASE + _CODES_PER_FRAME * _CODEBOOK_SIZE
        ):
            return None

        position = len(self._codes) % _CODES_PER_FRAME
        code = token_id - _AUDIO_TOKEN_BASE - position * _CODEBOOK_SIZE
        if not 0 <= code < _CODEBOOK_SIZE:
            # an out-of-place code would decode to noise; the model sometimes slips one in
            return None
        self._codes.append(code)

        frames = len(self._codes) // _CODES_PER_FRAME
        if frames - self._right - self._emitted_frames >= self._chunk:
            return self._window(frames - self._right, right=self._right)
        return None

    def flush(self) -> _DecodeWindow | None:
        frames = len(self._codes) // _CODES_PER_FRAME
        if frames <= self._emitted_frames:
            return None
        return self._window(frames, right=0)

    def _window(self, end_frame: int, *, right: int) -> _DecodeWindow:
        start_frame = self._emitted_frames
        first = max(0, start_frame - self._left)
        window = _DecodeWindow(
            codes=self._codes[first * _CODES_PER_FRAME : (end_frame + right) * _CODES_PER_FRAME],
            start=(start_frame - first) * _SAMPLES_PER_FRAME,
            end=(end_frame - first) * _SAMPLES_PER_FRAME,
        )
        self._emitted_frames = end_frame
        return window


def _log_prewarm_error(fut: asyncio.Future[Any]) -> None:
    if not fut.cancelled() and (exc := fut.exception()) is not None:
        logger.error("failed to load TTS model", exc_info=exc)


def _to_int16(audio: np.ndarray) -> np.ndarray:
    pcm: np.ndarray = (np.clip(audio.reshape(-1), -1.0, 1.0) * 32767.0).astype(np.int16)
    return pcm


def _generate_orpheus(
    loaded: _LoadedTTS,
    text: str,
    opts: _TTSOptions,
    on_token: Any,
    stop_event: threading.Event,
) -> None:
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList
    from transformers.generation.streamers import BaseStreamer

    voice = opts.voice or (DEFAULT_VOICE if opts.model == DEFAULT_TTS_MODEL else None)
    prompt = f"{voice}: {text}" if voice else text
    text_ids = loaded.tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = torch.cat(
        [
            torch.tensor([[_START_OF_HUMAN]], dtype=torch.long),
            text_ids,
            torch.tensor([[_END_OF_TEXT, _END_OF_HUMAN]], dtype=torch.long),
        ],
        dim=1,
    ).to(loaded.model.device)

    class _TokenStreamer(BaseStreamer):
        def __init__(self) -> None:
            self._prompt_skipped = False

        def put(self, value: Any) -> None:
            if not self._prompt_skipped:
                self._prompt_skipped = True
                return
            for token_id in value.reshape(-1).tolist():
                on_token(int(token_id))

        def end(self) -> None:
            pass

    class _StopOnCancel(StoppingCriteria):
        def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
            return stop_event.is_set()

    with torch.inference_mode():
        loaded.model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=opts.max_new_tokens,
            do_sample=opts.temperature > 0,
            temperature=opts.temperature if opts.temperature > 0 else None,
            top_p=opts.top_p if opts.temperature > 0 else None,
            repetition_penalty=opts.repetition_penalty,
            eos_token_id=_END_OF_SPEECH,
            pad_token_id=_END_OF_SPEECH,
            streamer=_TokenStreamer(),
            stopping_criteria=StoppingCriteriaList([_StopOnCancel()]),
        )


def _decode_snac(loaded: _LoadedTTS, window: _DecodeWindow) -> np.ndarray:
    import torch

    codes = window.codes
    frames = len(codes) // _CODES_PER_FRAME
    layer_1, layer_2, layer_3 = [], [], []
    for i in range(frames):
        c = codes[i * _CODES_PER_FRAME : (i + 1) * _CODES_PER_FRAME]
        layer_1.append(c[0])
        layer_2.extend([c[1], c[4]])
        layer_3.extend([c[2], c[3], c[5], c[6]])

    device = next(loaded.codec.parameters()).device
    layers = [
        torch.tensor([layer], dtype=torch.long, device=device)
        for layer in (layer_1, layer_2, layer_3)
    ]
    with torch.inference_mode():
        audio = loaded.codec.decode(layers)
    samples: np.ndarray = audio.reshape(-1).float().cpu().numpy()
    return samples[window.start : window.end]


def _synthesize_waveform(loaded: _LoadedTTS, text: str) -> np.ndarray:
    import torch

    inputs = loaded.tokenizer(text, return_tensors="pt").to(loaded.model.device)
    with loaded.lock, torch.inference_mode():
        output = loaded.model(**inputs)
    waveform = getattr(output, "waveform", None)
    if waveform is None:
        waveform = output[0]
    samples: np.ndarray = waveform.reshape(-1).float().cpu().numpy()
    return samples


def _is_orpheus(config: Any) -> bool:
    """Orpheus models are Llama models with the SNAC audio codes in their vocabulary."""
    vocab_size = int(getattr(config, "vocab_size", 0) or 0)
    return (
        getattr(config, "model_type", "") == "llama"
        and vocab_size >= _AUDIO_TOKEN_BASE + _CODES_PER_FRAME * _CODEBOOK_SIZE
    )


def _load_tts(key: _TTSKey, *, local_files_only: bool) -> _LoadedTTS:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    logger.info("loading TTS model", extra={"model": key.model, "device": key.device})
    kwargs: dict[str, Any] = {
        "revision": key.revision,
        "trust_remote_code": key.trust_remote_code,
        "local_files_only": local_files_only,
    }
    try:
        config: Any = AutoConfig.from_pretrained(key.model, **kwargs)
        tokenizer: Any = AutoTokenizer.from_pretrained(key.model, **kwargs)
        if _is_orpheus(config):
            model: Any = AutoModelForCausalLM.from_pretrained(
                key.model, dtype=key.dtype, device_map=key.device, **kwargs
            )
            loaded = _LoadedTTS(
                family="orpheus",
                model=model,
                tokenizer=tokenizer,
                sample_rate=_ORPHEUS_SAMPLE_RATE,
                codec=_load_snac(model.device, local_files_only=local_files_only),
            )
        else:
            from transformers import AutoModelForTextToWaveform

            model = AutoModelForTextToWaveform.from_pretrained(
                key.model, dtype=key.dtype, device_map=key.device, **kwargs
            )
            loaded = _LoadedTTS(
                family="waveform",
                model=model,
                tokenizer=tokenizer,
                sample_rate=int(getattr(model.config, "sampling_rate", 16000)),
            )
    except OSError as e:
        if not local_files_only:
            raise
        raise not_cached_error(key.model) from e

    loaded.model.eval()
    logger.info("TTS model loaded", extra={"model": key.model, "family": loaded.family})
    return loaded


def _load_snac(device: Any, *, local_files_only: bool) -> Any:
    try:
        from snac import SNAC  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError(
            "Orpheus models need the SNAC codec: pip install 'livekit-plugins-hf[orpheus]'"
        ) from e

    try:
        codec: Any = SNAC.from_pretrained(SNAC_MODEL, local_files_only=local_files_only)
    except OSError as e:
        if not local_files_only:
            raise
        raise not_cached_error(SNAC_MODEL) from e
    # the codec is small; keep it next to the model unless that's a meta/disk offload
    target = device if getattr(device, "type", "cpu") in ("cuda", "mps") else "cpu"
    return codec.eval().to(target)
