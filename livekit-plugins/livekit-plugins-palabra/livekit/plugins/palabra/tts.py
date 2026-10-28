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

"""Palabra realtime text-to-speech plugin for LiveKit Agents.

Transport is the official ``palabra-ai`` SDK (``palabra_ai.TtsSession``).
This module is the LiveKit adapter on top of it.
Each stream owns its ``TtsSession`` so SDK cancellation cannot interrupt another stream.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
import weakref
from dataclasses import dataclass, field, replace

from palabra_ai import (
    AuthError,
    Palabra,
    PalabraError,
    ServerError,
    SessionError,
    TaskError,
    TtsChunk,
    TtsSession,
)

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
from .models import (
    DEFAULT_DEACCENT_STRENGTH,
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_VOICE_ID,
    TTSModels,
)

# The Palabra server caps a single text message at 1024 characters.
_MAX_TEXT_LEN = 1024

# Server `error` codes that are worth retrying.
_RETRYABLE_ERROR_CODES = frozenset({"SERVICE_UNAVAILABLE", "RATE_LIMIT_EXCEEDED"})

# The server accepts at most 20 text messages per second per identity.
# Above that it replies with a RATE_LIMIT_EXCEEDED error frame and does not synthesize the text.
# A base-class retry replays all buffered text at once, so every send is spaced by this interval.
# 0.06 s is ~16 messages per second, one step below the cap.
_MIN_SEND_INTERVAL = 0.06


@dataclass
class _TTSOptions:
    language: str
    voice_id: str
    speed: float | None  # None: the server default is used
    deaccent_strength: float
    model: TTSModels | str
    sample_rate: int


@dataclass
class _SendRate:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_send: float = 0.0


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        language: str = DEFAULT_LANGUAGE,
        voice_id: str = DEFAULT_VOICE_ID,
        speed: NotGivenOr[float] = NOT_GIVEN,
        deaccent_strength: float = DEFAULT_DEACCENT_STRENGTH,
        model: TTSModels | str = DEFAULT_MODEL,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        region: str | None = None,
        tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
    ) -> None:
        """Create a new instance of Palabra realtime TTS.

        Args:
            api_key: Palabra API Key (create one at https://platform.palabra.ai/api-keys).
                Falls back to the ``PALABRA_API_KEY`` environment variable.
            language: BCP-47 language code for synthesis. Defaults to ``"en"``.
            voice_id: Voice identifier; ``"default_low"`` / ``"default_high"`` pick the
                language default voice. Defaults to ``"default_low"``.
            speed: Speech rate in the ``0.0``-``1.0`` range (``0.5`` is natural pace).
                Unset by default, so the server default is used.
            deaccent_strength: Foreign-accent reduction in the ``0.0``-``1.0`` range.
            model: Synthesis model. Defaults to ``"auto"`` (server picks).
            sample_rate: Output sample rate in Hz (``8000``-``48000``). Defaults to ``24000``.
            region: Palabra region (``"eu"`` / ``"us"``). Falls back to the
                ``PALABRA_REGION`` environment variable, then ``"eu"``.
            tokenizer: Sentence tokenizer used to chunk streamed text into utterances.
                Defaults to ``livekit.agents.tokenize.blingfire.SentenceTokenizer``.

        Raises:
            ValueError: no ``api_key`` argument and no ``PALABRA_API_KEY`` in the environment.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=sample_rate,
            num_channels=1,
        )
        api_key = api_key or os.environ.get("PALABRA_API_KEY")
        if not api_key:
            raise ValueError(
                "Palabra api_key is required, either as an argument or set the"
                " PALABRA_API_KEY environment variable"
            )

        self._client = Palabra(api_key, region=region)
        self._opts = _TTSOptions(
            language=language,
            voice_id=voice_id,
            speed=speed if is_given(speed) else None,
            deaccent_strength=deaccent_strength,
            model=model,
            sample_rate=sample_rate,
        )
        self._send_rate = _SendRate()
        self._closed = False
        self._streams = weakref.WeakSet[SynthesizeStream]()
        self._sentence_tokenizer = (
            tokenizer if is_given(tokenizer) else tokenize.blingfire.SentenceTokenizer()
        )

    @property
    def model(self) -> str:
        """Active synthesis model id."""
        return str(self._opts.model)

    @property
    def provider(self) -> str:
        """Provider display name used in metrics."""
        return "Palabra"

    def update_options(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        voice_id: NotGivenOr[str] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
        deaccent_strength: NotGivenOr[float] = NOT_GIVEN,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
    ) -> None:
        """Update synthesis options; unset arguments keep their current value.

        Each new stream receives the new settings. Streams already running keep
        the options they were created with.
        """
        if is_given(language):
            self._opts.language = language
        if is_given(voice_id):
            self._opts.voice_id = voice_id
        if is_given(speed):
            self._opts.speed = speed
        if is_given(deaccent_strength):
            self._opts.deaccent_strength = deaccent_strength
        if is_given(model):
            self._opts.model = model

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> tts.ChunkedStream:
        """Synthesize ``text`` in a single request, over a dedicated realtime session."""
        return self._synthesize_with_stream(text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        """Open a streaming synthesis session that accepts incremental text."""
        if self._closed:
            raise APIConnectionError("palabra tts is closed", retryable=False)
        stream = SynthesizeStream(
            tts=self,
            client=self._client,
            opts=replace(self._opts),
            tokenizer=self._sentence_tokenizer,
            send_rate=self._send_rate,
            conn_options=conn_options,
        )
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        """Close all active streams and their realtime TTS sessions."""
        self._closed = True
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()


class SynthesizeStream(tts.SynthesizeStream):
    """One streaming synthesis request with its own SDK ``TtsSession``.

    Created by ``TTS.stream()`` / ``TTS.synthesize()``; not instantiated directly.
    """

    def __init__(
        self,
        *,
        tts: TTS,
        client: Palabra,
        opts: _TTSOptions,
        tokenizer: tokenize.SentenceTokenizer,
        send_rate: _SendRate,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._client = client
        self._opts = opts
        self._tokenizer = tokenizer
        self._send_rate = send_rate

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
            stream=True,
        )

        sent_tokenizer_stream = self._tokenizer.stream()
        send_complete = asyncio.Event()
        generation_changed = asyncio.Event()
        pending_gens: set[str] = set()

        async def _input_task() -> None:
            async for data in self.input:
                if isinstance(data, self.Flush):
                    sent_tokenizer_stream.flush()
                    continue
                sent_tokenizer_stream.push_text(data)
            sent_tokenizer_stream.end_input()

        async def _send_task(session: TtsSession) -> None:
            loop = asyncio.get_running_loop()
            async for ev in sent_tokenizer_stream:
                text = ev.token
                if not text:
                    continue
                # Each text message is a separate generation on the server.
                # It always ends with its own last_chunk, even when eos=False.
                pieces = [text[i : i + _MAX_TEXT_LEN] for i in range(0, len(text), _MAX_TEXT_LEN)]
                for idx, piece in enumerate(pieces):
                    gen_id = utils.shortuuid("PLBR_")
                    try:
                        async with self._send_rate.lock:
                            slot = max(self._send_rate.last_send + _MIN_SEND_INTERVAL, loop.time())
                            self._send_rate.last_send = slot
                        delay = slot - loop.time()
                        if delay > 0:
                            await asyncio.sleep(delay)
                        # Audio can arrive before send_text returns.
                        pending_gens.add(gen_id)
                        generation_changed.set()
                        self.mark_started()
                        try:
                            async with asyncio.timeout(self._conn_options.write_timeout):
                                await session.send_text(
                                    piece, eos=idx == len(pieces) - 1, generation_id=gen_id
                                )
                        except TimeoutError:
                            raise APITimeoutError(phase="write") from None
                    except BaseException:
                        pending_gens.discard(gen_id)
                        raise
            send_complete.set()
            generation_changed.set()

        async def _recv_task() -> None:
            got_audio = False
            segment_started = False
            while not (send_complete.is_set() and not pending_gens):
                if not pending_gens:
                    await generation_changed.wait()
                    generation_changed.clear()
                    continue
                try:
                    event = await session.receive(timeout=self._conn_options.read_timeout)
                except TimeoutError:
                    # Preserve partial audio when the provider omits its final marker.
                    if got_audio:
                        break
                    raise APITimeoutError(phase="read") from None
                if event is None:
                    raise APIConnectionError("palabra tts session closed unexpectedly")
                if isinstance(event, ServerError):
                    logger.error("palabra tts error: %s", event.code)
                    raise APIError(
                        f"palabra tts error: {event.code}",
                        retryable=event.code in _RETRYABLE_ERROR_CODES,
                    )
                if not isinstance(event, TtsChunk) or event.generation_id not in pending_gens:
                    continue
                if event.audio:
                    got_audio = True
                    if not segment_started:
                        # One segment per stream: the agent opens a fresh stream() per reply.
                        output_emitter.start_segment(segment_id=request_id)
                        segment_started = True
                    output_emitter.push(event.audio)
                if event.last_chunk:
                    pending_gens.discard(event.generation_id)
            output_emitter.end_input()

        session = self._client.tts(
            language=self._opts.language,
            voice_id=self._opts.voice_id,
            speed=self._opts.speed,
            deaccent_strength=self._opts.deaccent_strength,
            model=self._opts.model,
            format="pcm",
            sample_rate=self._opts.sample_rate,
        )
        try:
            try:
                async with asyncio.timeout(self._conn_options.connect_timeout):
                    await session.__aenter__()
            except TimeoutError:
                raise APITimeoutError(phase="connect") from None
            tasks = [
                asyncio.create_task(_input_task()),
                asyncio.create_task(_send_task(session)),
                asyncio.create_task(_recv_task()),
            ]
            try:
                await asyncio.gather(*tasks)
            finally:
                await utils.aio.gracefully_cancel(*tasks)
        except asyncio.CancelledError:
            if pending_gens:
                with contextlib.suppress(Exception):
                    async with asyncio.timeout(self._conn_options.write_timeout):
                        await session.cancel()
            raise
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except AuthError as e:
            raise APIStatusError(str(e), status_code=401, request_id=None, body=None) from None
        except TaskError as e:
            # `code` is a short server enum (e.g. VALIDATION_ERROR); `desc` is free-form
            # server text, so only the code is surfaced.
            raise APIConnectionError(f"palabra tts error: {e.code}") from None
        except (SessionError, PalabraError) as e:
            # The SDK interpolates the underlying websockets error into its message, and the
            # connect URL carries the API key as a ?token= query parameter -- so neither
            # str(e) nor the __cause__ chain is safe to surface.
            raise APIConnectionError(type(e).__name__) from None
        finally:
            primary = sys.exception()
            cleanup_error: BaseException | None = None
            for close in (sent_tokenizer_stream.aclose, session.close):
                try:
                    await close()
                except BaseException as error:
                    if primary is None and cleanup_error is None:
                        cleanup_error = error
                    else:
                        # SDK exceptions can include the URL's auth token; report only the class.
                        logger.error("palabra TTS cleanup failed (%s)", type(error).__name__)
            if cleanup_error is not None:
                raise cleanup_error
