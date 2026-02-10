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
DEFAULT_MODEL: Final[TTSModels] = "dd-etts-2.5"
DEFAULT_FORMAT: Final[TTSAudioFormat] = "mp3"
DEFAULT_SAMPLE_RATE: Final[TTSSampleRate] = 24000

STREAM_FALLBACK_MODEL: Final[TTSModels] = "dd-etts-3.0"
STREAM_FALLBACK_FORMAT: Final[TTSAudioFormat] = "s16le"
STREAM_FALLBACK_SAMPLE_RATE: Final[TTSSampleRate] = 16000

MIME_TYPE: dict[str, str] = {
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "mulaw": "audio/basic",
    "headerless-wav": "audio/pcm",
    "s16le": "audio/pcm",
}


def _is_timeout_error(exc: BaseException) -> bool:
    """Best-effort timeout detection across provider/client exception types."""
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, (asyncio.TimeoutError, TimeoutError)):
            return True
        if "timeout" in type(cur).__name__.lower():
            return True
        cur = cur.__cause__ if cur.__cause__ is not None else cur.__context__
    return False


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
        model: NotGivenOr[TTSModels] = NOT_GIVEN,
        locale: str = "en-US",
        format: NotGivenOr[TTSAudioFormat] = NOT_GIVEN,
        sample_rate: NotGivenOr[TTSSampleRate] = NOT_GIVEN,
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
            model: The model to use. Defaults to "dd-etts-2.5" when not specified.
            locale: Language locale string. Default is "en-US".
            format: Output audio format. Defaults to "mp3" when not specified.
            sample_rate: Output sample rate in Hz. Defaults to 24000 when not specified.
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
        resolved_model: TTSModels = model if is_given(model) else DEFAULT_MODEL
        resolved_format: TTSAudioFormat = format if is_given(format) else DEFAULT_FORMAT
        resolved_sample_rate: TTSSampleRate = (
            sample_rate if is_given(sample_rate) else DEFAULT_SAMPLE_RATE
        )

        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=resolved_sample_rate,
            num_channels=NUM_CHANNELS,
        )

        api_key = api_key or os.environ.get("DEEPDUB_API_KEY")
        if not api_key:
            raise ValueError(
                "Deepdub API key is required. Set DEEPDUB_API_KEY env var or pass api_key argument."
            )

        self._opts = _TTSOptions(
            voice_prompt_id=voice_prompt_id,
            model=resolved_model,
            locale=locale,
            format=resolved_format,
            sample_rate=resolved_sample_rate,
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
        self._stream_model_user_specified = is_given(model)
        self._stream_format_user_specified = is_given(format)
        self._stream_sample_rate_user_specified = is_given(sample_rate)

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
            self._stream_model_user_specified = True
        if is_given(locale):
            self._opts.locale = locale
        if is_given(format):
            self._opts.format = format  # type: ignore[assignment]
            self._stream_format_user_specified = True
        if is_given(sample_rate):
            self._opts.sample_rate = sample_rate  # type: ignore[assignment]
            self._stream_sample_rate_user_specified = True
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
    ) -> tts.ChunkedStream:
        return self._synthesize_with_stream(text=text, conn_options=conn_options)

    def stream(
        self,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SynthesizeStream:
        return SynthesizeStream(
            tts=self,
            conn_options=conn_options,
        )


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
        # Respect user-provided values; only fallback missing stream-critical options.
        fallback_applied: list[str] = []
        if not self._tts._stream_model_user_specified:
            self._opts.model = STREAM_FALLBACK_MODEL
            fallback_applied.append("model")
        if not self._tts._stream_format_user_specified:
            self._opts.format = STREAM_FALLBACK_FORMAT
            fallback_applied.append("format")
        if not self._tts._stream_sample_rate_user_specified:
            self._opts.sample_rate = STREAM_FALLBACK_SAMPLE_RATE
            fallback_applied.append("sample_rate")
        if fallback_applied:
            logger.warning(
                "applying Deepdub streaming fallbacks for unspecified options: %s",
                ", ".join(fallback_applied),
            )
        self._client = tts._client
        self._segment_id = utils.shortuuid()
        self._sent_tokenizer_stream = tokenize.basic.SentenceTokenizer(min_sentence_len=10).stream()

    async def aclose(self) -> None:
        await self._sent_tokenizer_stream.aclose()
        await super().aclose()

    def _initialize_stream_output(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=self._segment_id,
            sample_rate=self._opts.sample_rate,
            num_channels=NUM_CHANNELS,
            stream=True,
            mime_type=MIME_TYPE[self._opts.format],
        )
        output_emitter.start_segment(segment_id=self._segment_id)

    def _build_stream_connect_kwargs(self) -> dict[str, Any]:
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
        return connect_kwargs

    async def _run_websocket_flow(self, conn: Any, output_emitter: tts.AudioEmitter) -> None:
        send_task = asyncio.create_task(self._send_loop(conn))
        recv_task = asyncio.create_task(self._recv_loop(conn, output_emitter))
        try:
            await asyncio.gather(send_task, recv_task)
        finally:
            if not send_task.done():
                send_task.cancel()
            if not recv_task.done():
                recv_task.cancel()
            try:
                await asyncio.wait_for(
                    asyncio.gather(send_task, recv_task, return_exceptions=True),
                    timeout=1.0,
                )
            except asyncio.TimeoutError:
                logger.debug("timed out while cancelling Deepdub stream tasks")

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        self._initialize_stream_output(output_emitter)
        conn_ctx = self._client.async_stream_connect(**self._build_stream_connect_kwargs())
        conn: Any | None = None

        try:
            conn = await asyncio.wait_for(conn_ctx.__aenter__(), timeout=self._conn_options.timeout)
            await self._run_websocket_flow(conn, output_emitter)
        except APITimeoutError:
            raise
        except APIConnectionError:
            raise
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except Exception as e:
            if _is_timeout_error(e):
                raise APITimeoutError() from None
            raise APIConnectionError() from e
        finally:
            if conn is not None:
                try:
                    await asyncio.wait_for(conn_ctx.__aexit__(None, None, None), timeout=1.0)
                except Exception:
                    logger.debug("error while closing Deepdub stream connection", exc_info=True)

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
                    await asyncio.wait_for(
                        conn.async_stream_text(text=sentence),
                        timeout=self._conn_options.timeout,
                    )
                    self._mark_started()
        finally:
            await utils.aio.gracefully_cancel(input_task)

    async def _recv_loop(self, conn: DeepdubClient, output_emitter: tts.AudioEmitter) -> None:
        """Receive audio chunks from Deepdub WebSocket until timeout."""
        received_audio = False
        recv_timeout = self._conn_options.timeout
        while True:
            try:
                audio_chunk = await asyncio.wait_for(
                    conn.async_stream_recv_audio(),
                    timeout=recv_timeout,
                )
                if audio_chunk:
                    output_emitter.push(audio_chunk)
                    received_audio = True
            except asyncio.TimeoutError:
                if received_audio:
                    logger.debug("recv timeout after audio, ending stream")
                    break

                # No text was pushed and input is closed: empty-stream case.
                if not self._pushed_text.strip() and self._input_ch.closed:
                    logger.debug("recv timeout on empty stream, ending stream")
                    break

                raise APITimeoutError() from None
            except asyncio.CancelledError:
                raise
            except Exception as e:
                if _is_timeout_error(e):
                    raise APITimeoutError() from None
                logger.exception("error receiving audio from Deepdub")
                raise APIConnectionError() from e
