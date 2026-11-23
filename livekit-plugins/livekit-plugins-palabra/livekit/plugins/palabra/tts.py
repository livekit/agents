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
One ``TtsSession`` (one WebSocket) is shared by every stream of a ``TTS`` instance.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import weakref
from dataclasses import dataclass

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


def _session_alive(session: TtsSession) -> bool:
    # The SDK has no public liveness flag; this reads its private `_ws` websocket.
    # close_code is set once the connection is closed by either side.
    ws = session._ws
    return ws is not None and ws.close_code is None


@dataclass
class _TTSOptions:
    language: str
    voice_id: str
    speed: float | None  # None: the server default is used
    deaccent_strength: float
    model: TTSModels | str
    sample_rate: int


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
        # Invariants of the shared session state:
        #   - one TtsSession (one WebSocket) is shared by every stream of this TTS;
        #   - the `init` frame is sent once per connection, so update_options() only
        #     marks _opts_dirty and the next _ensure_session() reopens the session;
        #   - _send_lock orders every socket write (text and cancel) and session open/close;
        #   - _send_lock is held per operation, never for a stream's lifetime.
        self._tts_session: TtsSession | None = None
        self._opts_dirty = False
        self._send_lock = asyncio.Lock()
        self._last_send = 0.0
        self._active_streams_cnt = 0
        # Demultiplexer: one background reader per session.
        # It routes each received event by generation_id into the sending stream's mailbox.
        #   - _routes: generation_id -> owning stream's mailbox;
        #   - unknown generation_id = the generation was already cancelled: dropped;
        #   - ServerError has no generation_id: delivered to every mailbox.
        self._routes: dict[str, asyncio.Queue[TtsChunk | ServerError | None]] = {}
        self._reader_task: asyncio.Task[None] | None = None
        self._prewarm_task: asyncio.Task[None] | None = None
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

    def _new_tts(self) -> TtsSession:
        return self._client.tts(
            language=self._opts.language,
            voice_id=self._opts.voice_id,
            speed=self._opts.speed,
            deaccent_strength=self._opts.deaccent_strength,
            model=self._opts.model,
            format="pcm",
            sample_rate=self._opts.sample_rate,
        )

    async def _ensure_session(self, timeout: float) -> TtsSession:
        # Caller must hold `self._send_lock`.
        if self._closed:
            raise APIConnectionError("palabra tts is closed", retryable=False)
        if self._opts_dirty and self._active_streams_cnt == 0:
            await self._reset_session()
            self._opts_dirty = False
        if self._tts_session is not None and (
            not _session_alive(self._tts_session)
            or self._reader_task is None
            or self._reader_task.done()
        ):
            await self._reset_session()
        if self._tts_session is None:
            session = self._new_tts()
            try:
                await asyncio.wait_for(session.__aenter__(), timeout=timeout)
            except BaseException:
                with contextlib.suppress(Exception):
                    await session.close()
                raise
            self._tts_session = session
            self._reader_task = asyncio.create_task(self._read_loop(session))
        return self._tts_session

    async def _read_loop(self, session: TtsSession) -> None:
        # Sole consumer of the session's events; streams read their own mailboxes.
        stale_chunks = 0
        try:
            async for event in session:
                if isinstance(event, TtsChunk):
                    route = self._routes.get(event.generation_id)
                    if route is None:
                        stale_chunks += 1  # frame of an already-cancelled generation
                        continue
                    route.put_nowait(event)
                elif isinstance(event, ServerError):
                    for route in set(self._routes.values()):
                        route.put_nowait(event)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("palabra tts reader failed")
        finally:
            if stale_chunks:
                logger.debug("dropped %d audio chunks of cancelled generations", stale_chunks)
            for route in set(self._routes.values()):
                route.put_nowait(None)  # the session is gone: wake every stream

    async def _reset_session(self) -> None:
        reader, self._reader_task = self._reader_task, None
        if reader is not None:
            reader.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await reader
        session, self._tts_session = self._tts_session, None
        if session is not None:
            with contextlib.suppress(Exception):
                await session.close()

    async def _drop_session_if_idle(self, *, current_stream_active: bool) -> None:
        # A failed request must not close the socket under other active streams.
        async with self._send_lock:
            dead = self._tts_session is not None and not _session_alive(self._tts_session)
            # Error teardown runs before the failing stream releases its active count.
            other_streams = self._active_streams_cnt - int(current_stream_active)
            if dead or other_streams <= 0:
                await self._reset_session()

    def prewarm(self) -> None:
        """Open the realtime TTS session ahead of the first request.

        Returns immediately; the connection is opened in a background task.
        Failures are ignored — the first stream()/synthesize() call reconnects.
        Does nothing when called outside a running event loop.
        """
        with contextlib.suppress(RuntimeError):
            self._prewarm_task = asyncio.get_running_loop().create_task(self._safe_prewarm())

    async def _safe_prewarm(self) -> None:
        with contextlib.suppress(Exception):
            async with self._send_lock:
                if self._closed:
                    return
                await self._ensure_session(DEFAULT_API_CONNECT_OPTIONS.timeout)

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

        The Palabra ``init`` frame is sent once per connection, so the session is
        reopened on the next stream() call for the new options to take effect.
        Streams already running keep the options they were created with.
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
        self._opts_dirty = True

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> tts.ChunkedStream:
        """Synthesize ``text`` in a single request, over the shared realtime session."""
        return self._synthesize_with_stream(text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        """Open a streaming synthesis session that accepts incremental text."""
        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        """Close all active streams and the realtime TTS session."""
        self._closed = True
        if self._prewarm_task is not None:
            self._prewarm_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._prewarm_task
            self._prewarm_task = None
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()
        async with self._send_lock:
            await self._reset_session()


class SynthesizeStream(tts.SynthesizeStream):
    """One streaming synthesis request, multiplexed onto the shared ``TtsSession``.

    Created by ``TTS.stream()`` / ``TTS.synthesize()``; not instantiated directly.
    """

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._tts._opts.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
            stream=True,
        )

        sent_tokenizer_stream = self._tts._sentence_tokenizer.stream()
        send_complete = asyncio.Event()
        # One generation_id per text message.
        # The TTS reader delivers audio_chunk frames into this mailbox by that id.
        #   - my_gens = every id this stream sent; its routes are removed on exit;
        #   - pending_gens = the ids still waiting for their last_chunk.
        mailbox: asyncio.Queue[TtsChunk | ServerError | None] = asyncio.Queue()
        my_gens: set[str] = set()
        pending_gens: set[str] = set()

        async def _input_task() -> None:
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
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
                    # Register before sending: audio_chunk can arrive before send_text returns.
                    my_gens.add(gen_id)
                    pending_gens.add(gen_id)
                    self._tts._routes[gen_id] = mailbox
                    self._mark_started()
                    try:
                        async with self._tts._send_lock:
                            delay = self._tts._last_send + _MIN_SEND_INTERVAL - loop.time()
                            if delay > 0:
                                await asyncio.sleep(delay)
                            await session.send_text(
                                piece, eos=idx == len(pieces) - 1, generation_id=gen_id
                            )
                            self._tts._last_send = loop.time()
                    except BaseException:
                        pending_gens.discard(gen_id)
                        self._tts._routes.pop(gen_id, None)
                        raise
            send_complete.set()

        async def _recv_task() -> None:
            got_audio = False
            segment_started = False
            while not (send_complete.is_set() and not pending_gens):
                if pending_gens:
                    # Text was sent and its audio has not arrived yet.
                    # Silence longer than the timeout here means the server is stuck.
                    try:
                        event = await asyncio.wait_for(
                            mailbox.get(), timeout=self._conn_options.timeout
                        )
                    except asyncio.TimeoutError:
                        # A sent generation never got its last_chunk.
                        # The server ends errored or stalled generations without one.
                        if got_audio:
                            break  # deliver the partial audio instead of failing the reply
                        raise APITimeoutError() from None
                else:
                    # No sent text is waiting for audio: the pause comes from the LLM input.
                    # Wait in short steps.
                    # After the next send the branch above applies the real timeout.
                    try:
                        event = await asyncio.wait_for(mailbox.get(), timeout=0.25)
                    except asyncio.TimeoutError:
                        continue
                if event is None:
                    raise APIConnectionError("palabra tts session closed unexpectedly")
                if isinstance(event, ServerError):
                    logger.error("palabra tts error: %s", event.code)
                    raise APIError(
                        f"palabra tts error: {event.code}",
                        retryable=event.code in _RETRYABLE_ERROR_CODES,
                    )
                if isinstance(event, TtsChunk):
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

        async def _teardown() -> None:
            for gen_id in my_gens:
                self._tts._routes.pop(gen_id, None)
            await self._tts._drop_session_if_idle(current_stream_active=session_acquired)

        session: TtsSession | None = None
        session_acquired = False
        try:
            async with self._tts._send_lock:
                session = await self._tts._ensure_session(self._conn_options.timeout)
                self._tts._active_streams_cnt += 1
                session_acquired = True
            tasks = [
                asyncio.create_task(_input_task()),
                asyncio.create_task(_send_task(session)),
                asyncio.create_task(_recv_task()),
            ]
            try:
                await asyncio.gather(*tasks)
            finally:
                await sent_tokenizer_stream.aclose()
                await utils.aio.gracefully_cancel(*tasks)
        except asyncio.CancelledError:
            # Interruption: the framework awaits this task before the next reply, so return fast.
            # Cancel only our unfinished generations by id; other streams are not affected.
            if session is not None and pending_gens:
                logger.debug("interrupted: cancelling %d generations", len(pending_gens))
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(
                        self._send_cancel(session, set(pending_gens)), timeout=1.0
                    )
                logger.debug("interrupted: server cancel sent")
            raise
        except APIError:
            await _teardown()
            raise
        except asyncio.TimeoutError:
            await _teardown()
            raise APITimeoutError() from None
        except AuthError as e:
            await _teardown()
            raise APIStatusError(str(e), status_code=401, request_id=None, body=None) from None
        except TaskError as e:
            await _teardown()
            # `code` is a short server enum (e.g. VALIDATION_ERROR); `desc` is free-form
            # server text, so only the code is surfaced.
            raise APIConnectionError(f"palabra tts error: {e.code}") from None
        except (SessionError, PalabraError) as e:
            await _teardown()
            # The SDK interpolates the underlying websockets error into its message, and the
            # connect URL carries the API key as a ?token= query parameter -- so neither
            # str(e) nor the __cause__ chain is safe to surface.
            raise APIConnectionError(type(e).__name__) from None
        except Exception as e:
            await _teardown()
            logger.exception("palabra tts error")
            raise APIConnectionError() from e
        finally:
            for gen_id in my_gens:
                self._tts._routes.pop(gen_id, None)
            if session_acquired:
                async with self._tts._send_lock:
                    self._tts._active_streams_cnt -= 1

    async def _send_cancel(self, session: TtsSession, gen_ids: set[str]) -> None:
        async with self._tts._send_lock:
            for gen_id in gen_ids:
                # The SDK's cancel() takes no generation argument yet.
                # Send the frame directly until palabra-ai exposes cancel(generation_id).
                await session._send({"type": "cancel", "generation_id": gen_id})
