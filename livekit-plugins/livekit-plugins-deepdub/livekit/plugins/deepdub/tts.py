from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, replace
from typing import Any, Final

from deepdub import DeepdubClient
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APITimeoutError,
    tokenize,
    tts,
    utils,
)
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import is_given

from .log import logger
from .models import TTSAudioFormat, TTSModels, TTSSampleRate

NUM_CHANNELS: Final[int] = 1
RECV_TIMEOUT: Final[float] = 5.0

MIME_TYPE: dict[str, str] = {
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "mulaw": "audio/basic",
    "headerless-wav": "audio/pcm",
    "s16le": "audio/pcm",
}


@dataclass
class _TTSOptions:
    voice_prompt_id: str
    model: TTSModels
    locale: str
    format: TTSAudioFormat
    sample_rate: TTSSampleRate
    temperature: float | None
    variance: float | None
    tempo: float | None
    seed: int | None
    prompt_boost: bool | None
    accent_base_locale: str | None
    accent_locale: str | None
    accent_ratio: float | None
    api_key: str
    base_url: str | None


class TTS(tts.TTS):
    """Text-to-Speech (TTS) plugin for Deepdub."""

    def __init__(
        self,
        *,
        voice_prompt_id: str ="c9ea4bad-d836-49db-b6ca-b2fa9aba5163_customer-service",
        model: TTSModels = "dd-etts-2.5",
        locale: str = "en-US",
        format: TTSAudioFormat = "mp3",
        sample_rate: TTSSampleRate = 24000,
        temperature: float | None = None,
        variance: float | None = None,
        tempo: float | None = None,
        seed: int | None = None,
        prompt_boost: bool | None = None,
        accent_base_locale: str | None = None,
        accent_locale: str | None = None,
        accent_ratio: float | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        """
        Create a new instance of Deepdub TTS.

        Args:
            voice_prompt_id: The voice prompt ID to use for synthesis,
            model: The model to use. Default is "dd-etts-2.5".
            locale: Language locale string. Default is "en-US".
            format: Output audio format. Default is "mp3".
            sample_rate: Output sample rate in Hz. Default is 24000.
            temperature: Controls expressiveness of speech. Optional.
            variance: Controls variance of speech. Optional.
            tempo: Controls speed of speech. Mutually exclusive with duration. Optional.
            seed: Random seed for reproducibility. Optional.
            prompt_boost: Enable prompt boost. Optional.
            accent_base_locale: Base locale for accent control. Must be set with
                accent_locale and accent_ratio. Optional.
            accent_locale: Target accent locale. Must be set with accent_base_locale
                and accent_ratio. Optional.
            accent_ratio: Accent blending ratio. Must be set with accent_base_locale
                and accent_locale. Optional.
            api_key: API key for authentication. Defaults to DEEPDUB_API_KEY env var.
            base_url: Custom base URL for the API. Optional.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )

        api_key = api_key or os.environ.get("DEEPDUB_API_KEY")
        if not api_key:
            raise ValueError(
                "Deepdub API key is required. Set DEEPDUB_API_KEY env var or pass api_key argument."
            )

        self._opts = _TTSOptions(
            voice_prompt_id=voice_prompt_id,
            model=model,
            locale=locale,
            format=format,
            sample_rate=sample_rate,
            temperature=temperature,
            variance=variance,
            tempo=tempo,
            seed=seed,
            prompt_boost=prompt_boost,
            accent_base_locale=accent_base_locale,
            accent_locale=accent_locale,
            accent_ratio=accent_ratio,
            api_key=api_key,
            base_url=base_url,
        )

        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url is not None:
            client_kwargs["base_url"] = base_url
        self._client = DeepdubClient(**client_kwargs)

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "deepdub"

    def update_options(
        self,
        *,
        voice_prompt_id: NotGivenOr[str] = NOT_GIVEN,
        model: NotGivenOr[TTSModels] = NOT_GIVEN,
        locale: NotGivenOr[str] = NOT_GIVEN,
        format: NotGivenOr[TTSAudioFormat] = NOT_GIVEN,
        sample_rate: NotGivenOr[TTSSampleRate] = NOT_GIVEN,
        temperature: NotGivenOr[float | None] = NOT_GIVEN,
        variance: NotGivenOr[float | None] = NOT_GIVEN,
        tempo: NotGivenOr[float | None] = NOT_GIVEN,
        seed: NotGivenOr[int | None] = NOT_GIVEN,
        prompt_boost: NotGivenOr[bool | None] = NOT_GIVEN,
        accent_base_locale: NotGivenOr[str | None] = NOT_GIVEN,
        accent_locale: NotGivenOr[str | None] = NOT_GIVEN,
        accent_ratio: NotGivenOr[float | None] = NOT_GIVEN,
    ) -> None:
        """Update the TTS options."""
        if is_given(voice_prompt_id):
            self._opts.voice_prompt_id = voice_prompt_id
        if is_given(model):
            self._opts.model = model  # type: ignore[assignment]
        if is_given(locale):
            self._opts.locale = locale
        if is_given(format):
            self._opts.format = format  # type: ignore[assignment]
        if is_given(sample_rate):
            self._opts.sample_rate = sample_rate  # type: ignore[assignment]
        if is_given(temperature):
            self._opts.temperature = temperature
        if is_given(variance):
            self._opts.variance = variance
        if is_given(tempo):
            self._opts.tempo = tempo
        if is_given(seed):
            self._opts.seed = seed
        if is_given(prompt_boost):
            self._opts.prompt_boost = prompt_boost
        if is_given(accent_base_locale):
            self._opts.accent_base_locale = accent_base_locale
        if is_given(accent_locale):
            self._opts.accent_locale = accent_locale
        if is_given(accent_ratio):
            self._opts.accent_ratio = accent_ratio

    async def list_voices(self) -> list[dict[str, Any]]:
        """List available voices from Deepdub."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._client.list_voices)

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> ChunkedStream:
        return ChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
        )

    def stream(
        self,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SynthesizeStream:
        return SynthesizeStream(
            tts=self,
            conn_options=conn_options,
        )


class ChunkedStream(tts.ChunkedStream):
    """Synthesize text to speech in a single request using the Deepdub REST API."""

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
        self._client = tts._client

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        opts = self._opts

        def _synthesize() -> bytes:
            kwargs: dict[str, Any] = {
                "text": self._input_text,
                "voice_prompt_id": opts.voice_prompt_id,
                "model": opts.model,
                "locale": opts.locale,
                "format": opts.format,
                "sample_rate": opts.sample_rate,
            }
            if opts.temperature is not None:
                kwargs["temperature"] = opts.temperature
            if opts.variance is not None:
                kwargs["variance"] = opts.variance
            if opts.tempo is not None:
                kwargs["tempo"] = opts.tempo
            if opts.seed is not None:
                kwargs["seed"] = opts.seed
            if opts.prompt_boost is not None:
                kwargs["prompt_boost"] = opts.prompt_boost
            if opts.accent_base_locale is not None:
                kwargs["accent_base_locale"] = opts.accent_base_locale
                kwargs["accent_locale"] = opts.accent_locale
                kwargs["accent_ratio"] = opts.accent_ratio
            return self._client.tts(**kwargs)

        try:
            loop = asyncio.get_event_loop()
            audio_data = await loop.run_in_executor(None, _synthesize)

            output_emitter.initialize(
                request_id=utils.shortuuid(),
                sample_rate=opts.sample_rate,
                num_channels=NUM_CHANNELS,
                mime_type=MIME_TYPE[opts.format],
            )
            output_emitter.push(audio_data)
            output_emitter.flush()
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except Exception as e:
            raise APIConnectionError() from e


class SynthesizeStream(tts.SynthesizeStream):
    """Stream text to speech using Deepdub WebSocket API."""

    def __init__(
        self,
        *,
        tts: TTS,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)
        # Deepdub websocket streaming is reliable with dd-etts-3.0 + s16le/16000.
        self._opts.model = "dd-etts-3.0"
        self._opts.format = "s16le"
        self._opts.sample_rate = 16000
        self._client = tts._client
        self._segment_id = utils.shortuuid()
        self._sent_tokenizer_stream = tokenize.basic.SentenceTokenizer(min_sentence_len=10).stream()

    async def aclose(self) -> None:
        await self._sent_tokenizer_stream.aclose()
        await super().aclose()

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=self._segment_id,
            sample_rate=self._opts.sample_rate,
            num_channels=NUM_CHANNELS,
            stream=True,
            mime_type=MIME_TYPE[self._opts.format],
        )
        output_emitter.start_segment(segment_id=self._segment_id)

        connect_kwargs: dict[str, Any] = {
            "model": self._opts.model,
            "locale": self._opts.locale,
            "voice_prompt_id": self._opts.voice_prompt_id,
            "format": self._opts.format,
            "sample_rate": self._opts.sample_rate,
        }
        if self._opts.temperature is not None:
            connect_kwargs["temperature"] = self._opts.temperature
        if self._opts.variance is not None:
            connect_kwargs["variance"] = self._opts.variance
        if self._opts.tempo is not None:
            connect_kwargs["tempo"] = self._opts.tempo
        if self._opts.seed is not None:
            connect_kwargs["seed"] = self._opts.seed
        if self._opts.prompt_boost is not None:
            connect_kwargs["prompt_boost"] = self._opts.prompt_boost
        if self._opts.accent_base_locale is not None:
            connect_kwargs["accent_base_locale"] = self._opts.accent_base_locale
            connect_kwargs["accent_locale"] = self._opts.accent_locale
            connect_kwargs["accent_ratio"] = self._opts.accent_ratio

        try:
            async with self._client.async_stream_connect(**connect_kwargs) as conn:
                send_task = asyncio.create_task(self._send_loop(conn))
                recv_task = asyncio.create_task(self._recv_loop(conn, output_emitter))

                try:
                    await asyncio.gather(send_task, recv_task)
                finally:
                    await utils.aio.gracefully_cancel(send_task, recv_task)
        except APITimeoutError:
            raise
        except APIConnectionError:
            raise
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except Exception as e:
            raise APIConnectionError() from e
        finally:
            output_emitter.end_segment()

    async def _send_loop(self, conn: DeepdubClient) -> None:
        """Read from input channel, tokenize into sentences, and send to Deepdub."""

        async def _input_reader() -> None:
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    self._sent_tokenizer_stream.flush()
                    continue
                self._sent_tokenizer_stream.push_text(data)
            self._sent_tokenizer_stream.end_input()

        input_task = asyncio.create_task(_input_reader())

        try:
            async for event in self._sent_tokenizer_stream:
                sentence = event.token
                if sentence.strip():
                    await conn.async_stream_text(text=sentence)
                    self._mark_started()
        finally:
            await utils.aio.gracefully_cancel(input_task)

    async def _recv_loop(self, conn: DeepdubClient, output_emitter: tts.AudioEmitter) -> None:
        """Receive audio chunks from Deepdub WebSocket until timeout."""
        while True:
            try:
                audio_chunk = await asyncio.wait_for(
                    conn.async_stream_recv_audio(),
                    timeout=RECV_TIMEOUT,
                )
                if audio_chunk:
                    output_emitter.push(audio_chunk)
            except asyncio.TimeoutError:
                logger.debug("recv timeout, ending stream")
                break
            except Exception:
                logger.exception("error receiving audio from Deepdub")
                break
