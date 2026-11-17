from __future__ import annotations

import asyncio
import os
import time
import uuid
import weakref
from dataclasses import dataclass, replace
from typing import Any

from humain_voice.tts import TTSClient, get_sample_rate

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

from .log import logger
from .models import DEFAULT_API_URL, DEFAULT_MODEL, DEFAULT_VOICE, TTSModels, TTSVoices

# HTTP status used for HUMAIN error codes that fail the same way on every retry.
# Anything not listed is treated as transient unless HUMAIN marks it non-retryable.
_ERROR_STATUS = {
    "TTS_VOICE_NOT_FOUND": 404,
    "TTS_MODEL_NOT_FOUND": 404,
    "TTS_INPUT_NOT_ALLOWED": 400,
    "TTS_INVALID_INPUT": 400,
    "TTS_EMOTION_MARKUP_INVALID": 400,
    "TTS_EMOTION_SEGMENT_COUNT_EXCEEDED": 400,
    "TTS_EMOTION_SEGMENT_TOO_SHORT": 400,
    "VALIDATION_INVALID_UUID": 400,
    "CHARACTER_COUNT_EXCEEDED": 413,
    "VOICE_REFERENCE_COUNT_EXCEEDED": 413,
    "CREDITS_EXHAUSTED": 402,
    "CONCURRENCY_LIMIT_EXCEEDED": 429,
}


@dataclass
class _TTSOptions:
    voice: str
    model: str
    sample_rate: int


class TTS(tts.TTS):
    """HUMAIN Voice Text-to-Speech integration.

    HUMAIN Voice is an Arabic-first speech platform hosted in Saudi Arabia. Its TTS
    speaks Arabic (Saudi, Levantine, Moroccan and Egyptian voices) and English, and
    handles Arabic-English code-switching in one sentence. Requests go over HUMAIN's
    Socket.IO API through the official ``humain-voice`` SDK; one connection per ``TTS``
    serves every request, and each request carries one complete text, so the streaming
    interface sends one sentence per request.

    Args:
        voice: Voice label from HUMAIN's voice list (e.g. ``"mul_nourah"``) or its UUID.
            Defaults to ``"mul_nourah"`` (female, Saudi).
        model: TTS model. Defaults to ``"nebula"`` (24 kHz output).
        api_key: HUMAIN Voice API key. If not provided, reads from the
            ``HUMAIN_API_KEY`` environment variable.
        api_url: API base URL. If not provided, reads from ``HUMAIN_API_URL`` or uses
            ``"https://api.voice.humain.com"``.
        tokenizer: Sentence tokenizer for streaming input. Each sentence becomes one
            request. Defaults to ``tokenize.blingfire.SentenceTokenizer``.
    """

    def __init__(
        self,
        *,
        voice: TTSVoices | str = DEFAULT_VOICE,
        model: TTSModels | str = DEFAULT_MODEL,
        api_key: str | None = None,
        api_url: str | None = None,
        tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
    ) -> None:
        humain_api_key = api_key or os.environ.get("HUMAIN_API_KEY")
        if not humain_api_key:
            raise ValueError(
                "HUMAIN Voice API key is required, either as argument or set"
                " HUMAIN_API_KEY environment variable"
            )
        if not voice:
            raise ValueError("voice must not be empty")
        sample_rate = get_sample_rate(model)

        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True, aligned_transcript=False),
            sample_rate=sample_rate,
            num_channels=1,
        )

        self._api_key = humain_api_key
        self._api_url = api_url or os.environ.get("HUMAIN_API_URL") or DEFAULT_API_URL
        self._opts = _TTSOptions(voice=voice, model=model, sample_rate=sample_rate)
        self._sentence_tokenizer = (
            tokenizer if is_given(tokenizer) else tokenize.blingfire.SentenceTokenizer()
        )
        self._client: TTSClient | None = None
        self._client_lock = asyncio.Lock()
        self._voice_ids: dict[str, str] = {}
        self._streams = weakref.WeakSet[SynthesizeStream]()
        self._prewarm_task: asyncio.Task[None] | None = None

    @property
    def model(self) -> str:
        """Returns the TTS model name, e.g. ``"nebula"``."""
        return self._opts.model

    @property
    def provider(self) -> str:
        """Returns the provider name ``"HUMAIN"``."""
        return "HUMAIN"

    def update_options(self, *, voice: NotGivenOr[TTSVoices | str] = NOT_GIVEN) -> None:
        """Change the voice used by the next requests.

        The voice travels with each request, so the open connection stays valid.

        Args:
            voice: Voice label or UUID.
        """
        if is_given(voice):
            if not voice:
                raise ValueError("voice must not be empty")
            self._opts = replace(self._opts, voice=voice)

    async def _get_client(self, timeout: float) -> tuple[TTSClient, bool]:
        """Return the shared client, connecting it if needed, and whether it was reused."""
        async with self._client_lock:
            if self._client is not None:
                return self._client, True
            client = TTSClient(api_url=self._api_url, api_key=self._api_key)
            try:
                await asyncio.wait_for(client.connect(), timeout)
            except BaseException:
                # Don't leak a half-open connection on timeout or cancellation.
                await asyncio.shield(client.close())
                raise
            self._client = client
            return client, False

    async def _reset_client(self) -> None:
        async with self._client_lock:
            client, self._client = self._client, None
        if client is not None:
            try:
                await client.close()
            except Exception:
                logger.debug("error closing the HUMAIN Voice client")

    async def _resolve_voice(self, client: TTSClient, voice: str, timeout: float) -> str:
        """Map a voice label to its UUID (cached); a UUID passes through."""
        try:
            uuid.UUID(voice)
            return voice
        except ValueError:
            pass
        if voice not in self._voice_ids:
            voices = await client.list_voices(timeout_seconds=timeout)
            self._voice_ids = {v["label"]: v["id"] for v in voices}
            if voice not in self._voice_ids:
                raise APIStatusError("HUMAIN Voice voice not found", status_code=404)
        return self._voice_ids[voice]

    def prewarm(self) -> None:
        """Connect and resolve the voice ahead of the first request."""

        async def _prewarm() -> None:
            timeout = DEFAULT_API_CONNECT_OPTIONS.timeout
            try:
                client, _ = await self._get_client(timeout)
                await self._resolve_voice(client, self._opts.voice, timeout)
            except Exception:
                logger.warning("failed to prewarm the HUMAIN Voice connection")

        if self._prewarm_task is None or self._prewarm_task.done():
            self._prewarm_task = asyncio.create_task(_prewarm())

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> ChunkedStream:
        """Synthesize text to speech in a single request.

        Args:
            text: The text to synthesize.
            conn_options: Connection options for the request.

        Returns:
            A ``ChunkedStream`` that yields ``SynthesizedAudio`` events.
        """
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        """Create a streaming TTS session.

        Args:
            conn_options: Connection options for the stream.

        Returns:
            A ``SynthesizeStream`` for streaming synthesis.
        """
        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        """Clean up all stream and connection resources."""
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()
        if self._prewarm_task is not None:
            await utils.aio.cancel_and_wait(self._prewarm_task)
        await self._reset_client()


async def _synthesize_one(
    engine: TTS,
    opts: _TTSOptions,
    text: str,
    output_emitter: tts.AudioEmitter,
    timeout: float,
) -> tuple[float, bool]:
    """Synthesize one text and push its PCM. Returns (acquire time, connection reused)."""
    started = time.perf_counter()
    client, reused = await engine._get_client(timeout)
    acquire_time = time.perf_counter() - started
    voice_id = await engine._resolve_voice(client, opts.voice, timeout)

    errors: list[Any] = []
    try:
        async for chunk in client.synthesize_stream(
            text,
            voice_id,
            model=opts.model,
            on_error=errors.append,
            request_id=uuid.uuid4(),
            timeout_seconds=timeout,  # bounds every wait for audio
        ):
            if chunk.audio:
                output_emitter.push(chunk.audio)
    except asyncio.TimeoutError:
        raise
    except Exception:
        err = errors[-1] if errors else None
        code = getattr(err, "code", None) or ""
        if code:
            # Don't interpolate HUMAIN's message, which may echo the text.
            message = f"HUMAIN Voice request failed ({code})"
            if code in _ERROR_STATUS:
                raise APIStatusError(message, status_code=_ERROR_STATUS[code]) from None
            retryable = getattr(err, "retryable", None)
            raise APIError(message, retryable=retryable is not False) from None
        # No error event: the connection itself failed. Reconnect on the next attempt.
        await engine._reset_client()
        raise
    return acquire_time, reused


class ChunkedStream(tts.ChunkedStream):
    """Non-streaming TTS synthesis: the full text in one request."""

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
        )
        try:
            self._acquire_time, self._connection_reused = await _synthesize_one(
                self._tts, self._opts, self._input_text, output_emitter, self._conn_options.timeout
            )
            output_emitter.flush()
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except APIError:
            raise
        except Exception as e:
            raise APIConnectionError(type(e).__name__) from None


class SynthesizeStream(tts.SynthesizeStream):
    """Streaming TTS synthesis.

    The input is split into sentences and each sentence is one request on the shared
    connection, so audio starts with the first sentence. A flush ends the current
    segment; the next text opens a new one.
    """

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
            stream=True,
        )
        sent_stream = self._tts._sentence_tokenizer.stream(language="ar")

        async def _input_task() -> None:
            try:
                async for data in self._input_ch:
                    if isinstance(data, self._FlushSentinel):
                        sent_stream.flush()
                        continue
                    sent_stream.push_text(data)
            finally:
                sent_stream.end_input()

        async def _synthesize_task() -> None:
            segment_id: str | None = None
            first = True
            async for ev in sent_stream:
                text = ev.token.strip()
                if not text:
                    continue
                if ev.segment_id != segment_id:
                    if segment_id is not None:
                        output_emitter.end_segment()
                    segment_id = ev.segment_id
                    output_emitter.start_segment(segment_id=segment_id)

                self._mark_started()
                acquire_time, reused = await _synthesize_one(
                    self._tts, self._opts, text, output_emitter, self._conn_options.timeout
                )
                if first:
                    self._acquire_time, self._connection_reused = acquire_time, reused
                    first = False
                output_emitter.flush()

            if segment_id is not None:
                output_emitter.end_segment()

        tasks = [
            asyncio.create_task(_input_task()),
            asyncio.create_task(_synthesize_task()),
        ]
        try:
            await asyncio.gather(*tasks)
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except APIError:
            raise
        except Exception as e:
            raise APIConnectionError(type(e).__name__) from None
        finally:
            await utils.aio.cancel_and_wait(*tasks)
            await sent_stream.aclose()
