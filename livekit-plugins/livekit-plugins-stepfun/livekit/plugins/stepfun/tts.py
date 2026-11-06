from __future__ import annotations

import asyncio
import json
import weakref
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlencode

import aiohttp

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    tts,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr, TimedString
from livekit.agents.utils import is_given

from . import _utils
from .log import logger
from .models import (
    DEFAULT_TTS_SAMPLE_RATE,
    DEFAULT_TTS_VOICE,
    INSTRUCTION_MAX_CHARS,
    NUM_CHANNELS,
    SYSTEM_VOICES_MODEL,
    SYSTEM_VOICES_PATH,
    TTS_MAX_INPUT_CHARS,
    TTS_MAX_SESSION_DURATION,
    TTS_SAMPLE_RATES,
    TTS_STREAM_PATH,
    TTS_SYNTHESIZE_PATH,
    Cluster,
    Language,
    TTSModels,
    TTSVoices,
)

# Client -> server
_EV_CREATE = "tts.create"
_EV_TEXT_DELTA = "tts.text.delta"
_EV_TEXT_FLUSH = "tts.text.flush"
_EV_TEXT_DONE = "tts.text.done"

# Server -> client
_EV_CONNECTION_DONE = "tts.connection.done"
_EV_RESPONSE_CREATED = "tts.response.created"
_EV_SENTENCE_START = "tts.response.sentence.start"
_EV_SENTENCE_END = "tts.response.sentence.end"
_EV_AUDIO_DELTA = "tts.response.audio.delta"
_EV_AUDIO_DONE = "tts.response.audio.done"
_EV_SUBTITLE = "tts.response.subtitle"
_EV_TEXT_FLUSHED = "tts.text.flushed"
_EV_ERROR = "tts.response.error"

_STREAM_MIME_TYPE = "audio/pcm"

#: SSE returns audio incrementally and requires PCM rather than WAV.
_CHUNKED_STREAM_FORMAT = "sse"
_CHUNKED_RESPONSE_FORMAT = "pcm"
_SSE_AUDIO_DELTA = "speech.audio.delta"
_SSE_AUDIO_DONE = "speech.audio.done"
_SSE_AUDIO_ERROR = "speech.audio.error"

_HANDSHAKE_TIMEOUT = 10.0

#: Length-scaled total deadline to bound responses that continue trickling data.
_HTTP_SYNTH_BASE_TIMEOUT = 30.0
_HTTP_SYNTH_TIMEOUT_PER_CHAR = 0.3
#: Allow synthesis pauses independently of the connection timeout.
_HTTP_SYNTH_IDLE_TIMEOUT = 30.0

if TYPE_CHECKING:
    _WebSocket = aiohttp.ClientWebSocketResponse[bool]
else:
    _WebSocket = aiohttp.ClientWebSocketResponse

#: Characters that open or close an inline directive on the models that parse
#: them. Both the ASCII and the full-width forms appear in provider examples.
_INLINE_DIRECTIVE_CHARS = str.maketrans({"(": None, ")": None, "（": None, "）": None})


@dataclass
class _TTSOptions:
    model: TTSModels | str
    voice: TTSVoices | str
    instruction: NotGivenOr[str]
    language: NotGivenOr[Language | str]
    speed: float
    volume: float
    sample_rate: int
    mode: Literal["default", "sentence"]
    text_normalization: Literal["standard", "enhanced"]
    markdown_filter: bool
    timestamp: bool
    cluster: Cluster
    ws_base_url: str
    http_base_url: str
    # Whether inline directives embedded in downstream text are allowed to
    # reach the server. Defaults to False — see TTS.__init__ for why.
    enable_inline_instruction: bool

    @property
    def supports_instruction(self) -> bool:
        """Whether this model accepts `instruction` and parses inline directives.

        The other models reject `instruction` outright, so it must not be sent
        to them, and they do not interpret parentheses — which also means no
        escaping is needed there.
        """
        return self.model in INSTRUCTION_MAX_CHARS

    def validate(self) -> None:
        if self.sample_rate not in TTS_SAMPLE_RATES:
            raise ValueError(
                f"sample_rate must be one of {sorted(TTS_SAMPLE_RATES)}, got {self.sample_rate}"
            )
        if not 0.5 <= self.speed <= 2.0:
            raise ValueError(f"speed must be between 0.5 and 2.0, got {self.speed}")
        if not 0.1 <= self.volume <= 2.0:
            raise ValueError(f"volume must be between 0.1 and 2.0, got {self.volume}")
        if not is_given(self.instruction):
            return
        if not self.supports_instruction:
            logger.warning("instruction is unsupported by %s; dropping it", self.model)
            self.instruction = NOT_GIVEN
        else:
            cap = INSTRUCTION_MAX_CHARS[self.model]
            if len(self.instruction) > cap:
                raise ValueError(
                    f"instruction for {self.model} is capped at {cap} characters, "
                    f"got {len(self.instruction)}"
                )


@dataclass(eq=False)  # ConnectionPool requires identity hashing.
class _Connection:
    """A streaming TTS socket together with the session the server issued for it."""

    ws: _WebSocket
    session_id: str
    model: str = ""


class TTS(tts.TTS):
    """StepFun speech synthesis."""

    def __init__(
        self,
        *,
        model: TTSModels | str = "stepaudio-3-tts",
        voice: TTSVoices | str = DEFAULT_TTS_VOICE,
        instruction: NotGivenOr[str] = NOT_GIVEN,
        language: NotGivenOr[Language | str] = NOT_GIVEN,
        speed: float = 1.0,
        volume: float = 1.0,
        sample_rate: int = DEFAULT_TTS_SAMPLE_RATE,
        mode: Literal["default", "sentence"] = "default",
        text_normalization: Literal["standard", "enhanced"] = "standard",
        markdown_filter: bool = False,
        timestamp: bool = False,
        api_key: str | None = None,
        cluster: Cluster | None = None,
        base_url: str | None = None,
        http_base_url: str | None = None,
        enable_inline_instruction: bool = False,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """Create a streaming and HTTP speech synthesizer.

        Args:
            model: TTS model ID.
            voice: Voice ID; discover available IDs with list_voices().
            instruction: Performance direction; capped at 200 characters for
                stepaudio-2.5-tts and 500 for stepaudio-3-tts.
            language: Synthesis language; the service defaults to English.
            speed: Playback speed ratio, from 0.5 to 2.0.
            volume: Volume ratio, from 0.1 to 2.0.
            sample_rate: Output rate, defaulting to 24000 Hz.
            mode: Streaming text segmentation, default or sentence.
            text_normalization: HTTP normalization, standard or enhanced.
            markdown_filter: Request Markdown filtering.
            timestamp: Request streaming word-level subtitles.
            api_key: API key, falling back to STEPFUN_API_KEY.
            cluster: "global" or "china" (mainland China); defaults to
                STEPFUN_CLUSTER, then global. Credentials are cluster-specific.
            base_url: Override the WebSocket base URL.
            http_base_url: Override the HTTP base URL.
            enable_inline_instruction: Preserve parenthesized performance directives
                in input text. By default, delimiters are removed and contents spoken.
            http_session: Caller-owned session; otherwise use the LiveKit job session.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=True,
                aligned_transcript=timestamp,
            ),
            # Must match what the server actually produces. A wrong value
            # raises no error; it just plays back at the wrong pitch.
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )

        resolved_cluster = _utils.resolve_cluster(cluster)
        _utils.warn_if_voice_unavailable(voice, resolved_cluster)

        self._api_key = _utils.resolve_api_key(api_key)
        self._opts = _TTSOptions(
            model=model,
            voice=voice,
            instruction=instruction,
            language=language,
            speed=speed,
            volume=volume,
            sample_rate=sample_rate,
            mode=mode,
            text_normalization=text_normalization,
            markdown_filter=markdown_filter,
            timestamp=timestamp,
            cluster=resolved_cluster,
            ws_base_url=_utils.resolve_ws_base_url(base_url, resolved_cluster),
            http_base_url=_utils.resolve_http_base_url(http_base_url, resolved_cluster),
            enable_inline_instruction=enable_inline_instruction,
        )
        self._opts.validate()

        self._session = http_session
        self._streams = weakref.WeakSet[SynthesizeStream | ChunkedStream]()
        self._pool = utils.ConnectionPool[_Connection](
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
            # The server drops an idle streaming connection after roughly 60s,
            # so pooled ones are retired well before that.
            max_session_duration=TTS_MAX_SESSION_DURATION,
        )

    @property
    def model(self) -> str:
        return str(self._opts.model)

    @property
    def provider(self) -> str:
        return "StepFun"

    # -- connections -------------------------------------------------------

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = utils.http_context.http_session()
        return self._session

    async def _connect_ws(self, timeout: float, *, model: str | None = None) -> _Connection:
        """Connect and complete the server's half of the handshake.

        Unlike the ASR endpoint, this one takes the model as a query parameter.
        The server then sends `tts.connection.done` carrying the session id that
        every later frame must quote, so the handshake is finished here rather
        than at first use — that way a prewarmed connection is genuinely ready.
        """
        model = self._opts.model if model is None else model
        url = f"{self._opts.ws_base_url}{TTS_STREAM_PATH}?{urlencode({'model': model})}"

        try:
            ws = await asyncio.wait_for(
                self._ensure_session().ws_connect(url, headers=_utils.auth_headers(self._api_key)),
                timeout,
            )
        except Exception as exc:
            _utils.raise_translated(exc, context="stepfun tts connect")

        try:
            session_id = await asyncio.wait_for(_await_connection_done(ws), _HANDSHAKE_TIMEOUT)
        except BaseException:
            await ws.close()
            raise

        return _Connection(ws=ws, session_id=session_id, model=model)

    async def _close_ws(self, conn: _Connection) -> None:
        await conn.ws.close()

    def prewarm(self) -> None:
        """Open a connection and complete its handshake ahead of time."""
        self._ensure_session()
        self._pool.prewarm()

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()

        await self._pool.aclose()

        # Both caller-supplied and job-scoped HTTP sessions have external owners.
        self._session = None

    # -- configuration -----------------------------------------------------

    def update_options(
        self,
        *,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        voice: NotGivenOr[TTSVoices | str] = NOT_GIVEN,
        instruction: NotGivenOr[str] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
        volume: NotGivenOr[float] = NOT_GIVEN,
        language: NotGivenOr[Language | str] = NOT_GIVEN,
    ) -> None:
        """Apply validated options to future requests; active streams keep their snapshot."""
        opts = replace(self._opts)
        if is_given(model):
            opts.model = model
        if is_given(voice):
            _utils.warn_if_voice_unavailable(voice, self._opts.cluster)
            opts.voice = voice
        if is_given(instruction):
            opts.instruction = instruction
        if is_given(speed):
            opts.speed = speed
        if is_given(volume):
            opts.volume = volume
        if is_given(language):
            opts.language = language

        opts.validate()
        self._opts = opts

        self._pool.invalidate()

    # -- voices ------------------------------------------------------------

    async def list_voices(self) -> dict[str, Any]:
        """Fetch the official voice catalogue for this cluster.

        Each cluster accepts exactly one `model` value on this endpoint and 400s
        on anything else, which is the one place the two genuinely disagree.
        Cloned voices are account-specific and do not appear here.
        """
        url = f"{self._opts.http_base_url}{SYSTEM_VOICES_PATH}"
        params = {"model": SYSTEM_VOICES_MODEL[self._opts.cluster]}

        try:
            async with self._ensure_session().get(
                url, params=params, headers=_utils.auth_headers(self._api_key)
            ) as resp:
                await _utils.raise_for_status(resp)
                data: dict[str, Any] = await resp.json()
                return data
        except Exception as exc:
            _utils.raise_translated(exc, context="stepfun tts list_voices")

    # -- entry points ------------------------------------------------------

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> ChunkedStream:
        """One-shot synthesis over HTTP with incremental SSE audio output."""
        stream = ChunkedStream(tts=self, input_text=text, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    def stream(
        self,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SynthesizeStream:
        """Streaming synthesis over WebSocket, to synthesize as the LLM streams
        its output. Lower latency.
        """
        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream


def _close_detail(msg: Any) -> str:
    """Extract the close code and reason when available."""
    parts: list[str] = []
    code = getattr(msg, "extra", None)
    if isinstance(code, int):
        parts.append(f"close_code={code}")
    reason = getattr(msg, "data", None)
    if isinstance(reason, str) and reason:
        parts.append(f"reason={reason!r}")
    return f" ({', '.join(parts)})" if parts else ""


async def _await_connection_done(ws: _WebSocket) -> str:
    """Read frames until the server issues the session id."""
    while True:
        msg = await ws.receive()

        if msg.type in (
            aiohttp.WSMsgType.CLOSED,
            aiohttp.WSMsgType.CLOSE,
            aiohttp.WSMsgType.CLOSING,
        ):
            raise APIConnectionError(
                f"stepfun tts: websocket closed during handshake{_close_detail(msg)}"
            )

        if msg.type == aiohttp.WSMsgType.ERROR:
            raise APIConnectionError(f"stepfun tts: websocket error: {msg.data}")

        if msg.type != aiohttp.WSMsgType.TEXT:
            continue

        event = json.loads(msg.data)
        etype = event.get("type")

        if etype == _EV_CONNECTION_DONE:
            session_id = (event.get("data") or {}).get("session_id")
            if not session_id:
                raise APIConnectionError("stepfun tts: connection.done carried no session_id")
            return str(session_id)

        if etype == _EV_ERROR:
            raise _utils.translate_tts_ws_error(event, context="stepfun tts handshake")

        logger.debug("stepfun tts: ignoring event during handshake", extra={"type": etype})


class ChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        # Snapshot options in __init__, never lazily in _run() — otherwise this
        # races with the next update_options() call.
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        payload: dict[str, Any] = {
            "model": self._opts.model,
            "voice": self._opts.voice,
            "input": "",
            "response_format": _CHUNKED_RESPONSE_FORMAT,
            "stream_format": _CHUNKED_STREAM_FORMAT,
            "sample_rate": self._opts.sample_rate,
            "speed": self._opts.speed,
            "volume": self._opts.volume,
            "text_normalization": self._opts.text_normalization,
        }
        if self._opts.markdown_filter:
            payload["markdown_filter"] = True
        if is_given(self._opts.language):
            payload["language"] = self._opts.language
        if is_given(self._opts.instruction) and self._opts.supports_instruction:
            payload["instruction"] = self._opts.instruction

        url = f"{self._opts.http_base_url}{TTS_SYNTHESIZE_PATH}"
        headers = {
            **_utils.auth_headers(self._tts._api_key),
            "Accept": "text/event-stream",
        }

        try:
            initialized = False
            for text in _split_for_wire(_escape_text(self.input_text, self._opts)):
                payload["input"] = text
                async with self._tts._ensure_session().post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(
                        total=max(_HTTP_SYNTH_BASE_TIMEOUT, self._conn_options.timeout)
                        + _HTTP_SYNTH_TIMEOUT_PER_CHAR * len(text),
                        connect=self._conn_options.timeout,
                        sock_connect=self._conn_options.timeout,
                        sock_read=max(_HTTP_SYNTH_IDLE_TIMEOUT, self._conn_options.timeout),
                    ),
                ) as resp:
                    await _utils.raise_for_status(resp)

                    if not initialized:
                        # One segment for all requests: each response is raw PCM
                        # without a header, so the chunks concatenate directly.
                        output_emitter.initialize(
                            request_id=resp.headers.get("x-request-id") or utils.shortuuid(),
                            sample_rate=self._opts.sample_rate,
                            num_channels=NUM_CHANNELS,
                            mime_type=_STREAM_MIME_TYPE,
                        )
                        initialized = True

                    completed = False
                    async for event in _utils.iter_sse_events(resp):
                        etype = event.get("type")
                        if etype == _SSE_AUDIO_DELTA:
                            audio = event.get("audio") or ""
                            if audio:
                                output_emitter.push(_utils.b64_to_pcm(audio))
                        elif etype == _SSE_AUDIO_DONE:
                            completed = True
                            break
                        elif etype == _SSE_AUDIO_ERROR:
                            raise _utils.status_error(
                                status_code=-1,
                                message="stepfun tts synthesize: server returned an error",
                                request_id=resp.headers.get("x-request-id"),
                                body=event,
                            )
                    if not completed:
                        # Retryable: the base class restarts synthesis under a
                        # fresh request id, marking partial audio as stale.
                        raise APIConnectionError(
                            "stepfun tts: SSE stream ended before speech.audio.done"
                        )

            output_emitter.flush()
        except Exception as exc:
            _utils.raise_translated(exc, context="stepfun tts synthesize")


class SynthesizeStream(tts.SynthesizeStream):
    """One segment per instance, over one WebSocket session."""

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        try:
            await self._run_impl(output_emitter)
        except Exception as exc:
            _utils.raise_translated(exc, context="stepfun tts stream")

    async def _run_impl(self, output_emitter: tts.AudioEmitter) -> None:

        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=self._opts.sample_rate,
            num_channels=NUM_CHANNELS,
            mime_type=_STREAM_MIME_TYPE,
            stream=True,
        )

        async with self._tts._pool.connection(timeout=self._conn_options.timeout) as conn:
            self._acquire_time = self._tts._pool.last_acquire_time
            self._connection_reused = self._tts._pool.last_connection_reused

            try:
                # A model update may happen after this stream snapshots its options.
                if conn.model and conn.model != self._opts.model:
                    conn = await self._replace_connection(
                        conn, reason="pooled model changed", model=self._opts.model
                    )
                elif conn.ws.closed:
                    # An idle socket may close before the pool expires it;
                    # replace it before sending this segment.
                    conn = await self._replace_connection(
                        conn,
                        model=self._opts.model,
                        reason="pooled websocket was already closed",
                    )
                await self._create_session(conn)

                output_emitter.start_segment(segment_id=conn.session_id)

                tasks = [
                    asyncio.create_task(self._send_task(conn), name="stepfun_tts_send"),
                    asyncio.create_task(
                        self._recv_task(conn, output_emitter), name="stepfun_tts_recv"
                    ),
                ]
                try:
                    await asyncio.gather(*tasks)
                finally:
                    await utils.aio.cancel_and_wait(*tasks)
            finally:
                # remove() alone defers close until another acquisition.
                await conn.ws.close()
                self._tts._pool.remove(conn)
                self._tts._pool.prewarm()

    async def _replace_connection(
        self, conn: _Connection, *, reason: str, model: str | None = None
    ) -> _Connection:
        """Swap a pooled socket that must not be used for this segment.

        `remove()` before the rebind matters: the pool's context manager puts
        the *original* connection back on exit, and a removed one is no longer
        tracked, so a dead socket cannot sneak back into rotation.
        """
        logger.warning(
            "stepfun tts: %s; reconnecting",
            reason,
            extra={"session_id": conn.session_id, "model": model or conn.model},
        )
        await conn.ws.close()
        self._tts._pool.remove(conn)
        fresh = await self._tts._connect_ws(self._conn_options.timeout, model=model)
        self._connection_reused = False
        return fresh

    async def _create_session(self, conn: _Connection) -> None:
        """Send tts.create and wait for the server to accept it.

        Generation may only start after `tts.response.created`; sending text
        before that is out of protocol.
        """
        data: dict[str, Any] = {
            "session_id": conn.session_id,
            "voice_id": self._opts.voice,
            "response_format": "pcm",
            "sample_rate": self._opts.sample_rate,
            # Note the suffixed names: the streaming endpoint takes
            # speed_ratio/volume_ratio where the HTTP one takes speed/volume.
            "speed_ratio": self._opts.speed,
            "volume_ratio": self._opts.volume,
            "mode": self._opts.mode,
        }
        if is_given(self._opts.language):
            data["language"] = self._opts.language
        if is_given(self._opts.instruction) and self._opts.supports_instruction:
            data["instruction"] = self._opts.instruction
        if self._opts.markdown_filter:
            data["markdown_filter"] = True
        if self._opts.timestamp:
            data["timestamp"] = True

        await conn.ws.send_str(json.dumps({"type": _EV_CREATE, "data": data}))

        try:
            await asyncio.wait_for(self._await_created(conn), _HANDSHAKE_TIMEOUT)
        except asyncio.TimeoutError as exc:
            raise APIConnectionError(
                "stepfun tts: timed out waiting for tts.response.created"
            ) from exc

    async def _await_created(self, conn: _Connection) -> None:
        while True:
            msg = await conn.ws.receive()

            if msg.type in (
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSING,
            ):
                raise APIConnectionError(
                    f"stepfun tts: websocket closed before tts.response.created{_close_detail(msg)}"
                )

            if msg.type == aiohttp.WSMsgType.ERROR:
                raise APIConnectionError(f"stepfun tts: websocket error: {msg.data}")

            if msg.type != aiohttp.WSMsgType.TEXT:
                continue

            event = json.loads(msg.data)
            etype = event.get("type")

            if etype == _EV_RESPONSE_CREATED:
                return
            if etype == _EV_ERROR:
                raise _utils.translate_tts_ws_error(event, context="stepfun tts create")

            logger.debug("stepfun tts: ignoring event before created", extra={"type": etype})

    # -- upstream: text ----------------------------------------------------

    async def _send_task(self, conn: _Connection) -> None:
        """Consume self._input_ch, typed as str | _FlushSentinel."""
        async for data in self._input_ch:
            if isinstance(data, self._FlushSentinel):
                await self._send(conn, _EV_TEXT_FLUSH)
                continue

            text = _escape_text(data, self._opts)
            if not text:
                continue

            for chunk in _split_for_wire(text):
                self._mark_started()
                await self._send(conn, _EV_TEXT_DELTA, text=chunk)

        # Input is exhausted, so end generation. The server answers with
        # tts.response.audio.done and then closes the socket.
        await self._send(conn, _EV_TEXT_DONE)

    async def _send(self, conn: _Connection, event_type: str, **fields: Any) -> None:
        await conn.ws.send_str(
            json.dumps(
                {
                    "type": event_type,
                    "data": {"session_id": conn.session_id, **fields},
                }
            )
        )

    # -- downstream: audio -------------------------------------------------

    async def _recv_task(self, conn: _Connection, output_emitter: tts.AudioEmitter) -> None:
        while True:
            msg = await conn.ws.receive()

            if msg.type in (
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSING,
            ):
                # Retryable: the framework re-runs the segment on a fresh
                # connection when no audio had been pushed yet, and the close
                # code is what tells the two cases apart in the logs.
                raise APIConnectionError(
                    f"stepfun tts: websocket closed unexpectedly{_close_detail(msg)}"
                )

            if msg.type == aiohttp.WSMsgType.ERROR:
                raise APIConnectionError(f"stepfun tts: websocket error: {msg.data}")

            if msg.type != aiohttp.WSMsgType.TEXT:
                # Audio arrives as base64 inside JSON text frames; the protocol
                # defines no binary frames on this endpoint.
                logger.debug("stepfun tts: unexpected ws frame type %s", msg.type)
                continue

            try:
                event = json.loads(msg.data)
            except json.JSONDecodeError:
                logger.warning("stepfun tts: failed to decode ws frame")
                continue

            if self._handle_event(event, output_emitter):
                return

    def _handle_event(self, event: dict[str, Any], output_emitter: tts.AudioEmitter) -> bool:
        """Returns True once the segment is complete."""
        etype = event.get("type")
        data = event.get("data") or {}

        if etype == _EV_ERROR:
            raise _utils.translate_tts_ws_error(event, context="stepfun tts")

        if etype == _EV_AUDIO_DELTA:
            audio = data.get("audio")
            if audio:
                output_emitter.push(_utils.b64_to_pcm(audio))
            return False

        if etype == _EV_AUDIO_DONE:
            output_emitter.end_segment()
            return True

        if etype == _EV_SUBTITLE:
            self._push_subtitles(data, output_emitter)
            return False

        if etype in (_EV_SENTENCE_START, _EV_SENTENCE_END):
            logger.debug("stepfun tts: %s", etype, extra={"request_id": data.get("request_id")})
            return False

        if etype in (_EV_TEXT_FLUSHED, _EV_RESPONSE_CREATED, _EV_CONNECTION_DONE):
            logger.debug("stepfun tts: %s", etype)
            return False

        logger.debug("stepfun tts: unhandled event type %s", etype)
        return False

    def _push_subtitles(self, data: dict[str, Any], output_emitter: tts.AudioEmitter) -> None:
        """Translate a subtitle event into timed transcript strings.

        Times are milliseconds and already cumulative across earlier sentences,
        which lines up with one segment per stream. Alignment is asynchronous,
        so an event may arrive after audio for later text; the framework tolerates
        that, and all of them precede audio.done.
        """
        items = data.get("items") or []
        timed = [
            TimedString(
                item.get("text", ""),
                start_time=item.get("start_time", 0) / 1000.0,
                end_time=item.get("end_time", 0) / 1000.0,
            )
            for item in items
            if item.get("text")
        ]
        if timed:
            output_emitter.push_timed_transcript(timed)


def _escape_text(text: str, opts: _TTSOptions) -> str:
    """Neutralize anything downstream text could trip in the directive parser."""
    if opts.enable_inline_instruction or not opts.supports_instruction:
        return text
    return text.translate(_INLINE_DIRECTIVE_CHARS)


def _split_for_wire(text: str) -> list[str]:
    """Split text to respect the per-frame character cap.

    LLM tokens never come close to it, but `session.say()` with a long string
    routed through stream() can, and an over-long frame is rejected outright.
    """
    if len(text) <= TTS_MAX_INPUT_CHARS:
        return [text]
    return [text[i : i + TTS_MAX_INPUT_CHARS] for i in range(0, len(text), TTS_MAX_INPUT_CHARS)]
