from __future__ import annotations

import asyncio
import json
import os
import weakref
from dataclasses import dataclass, replace
from typing import Any

import aiohttp

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
    DEFAULT_BASE_URL,
    DEFAULT_DIALECT,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_SPEAKER,
    TTS_DIALECTS,
    TTS_MODEL,
    TTSDialect,
    TTSSampleRate,
)

# Hamsa rejects a request whose text is longer than this.
MAX_REQUEST_CHARS = 2000

# Close code Hamsa sends when the API key is missing or invalid.
_AUTH_FAILED_CLOSE_CODE = 4001


@dataclass
class _TTSOptions:
    api_key: str
    speaker: str
    dialect: str
    sample_rate: int
    expressiveness: float | None
    base_url: str

    def get_ws_url(self) -> str:
        return f"{self.base_url}/v1/realtime/ws"


def _validate(dialect: str, expressiveness: float | None) -> None:
    if dialect not in TTS_DIALECTS:
        raise ValueError(
            f"unsupported Hamsa dialect {dialect!r}, expected one of {sorted(TTS_DIALECTS)}"
        )
    if expressiveness is not None and not 0 <= expressiveness <= 2:
        raise ValueError("expressiveness must be between 0 and 2")


class TTS(tts.TTS):
    """Hamsa Text-to-Speech integration.

    Hamsa synthesizes Arabic in regional dialects (Egyptian, Levantine, Gulf, Iraqi,
    Modern Standard Arabic) and English over a WebSocket API. Each request carries one
    complete text and the audio streams back as raw 16-bit PCM, so the streaming
    interface sends one sentence per request on a pooled connection.

    Args:
        api_key: Hamsa API key. If not provided, reads from the ``HAMSA_API_KEY``
            environment variable.
        speaker: Pre-built voice name (e.g. ``"Mariam"``) or the UUID of a cloned
            voice. Use a speaker recorded in the chosen ``dialect``. Defaults to
            ``"Mariam"``.
        dialect: Dialect code such as ``"egy"`` (Egyptian), ``"leb"`` (Lebanese),
            ``"syr"`` (Syrian), ``"ksa"`` (Saudi), ``"msa"`` (Modern Standard
            Arabic) or ``"en"`` (English). Defaults to ``"egy"``.
        sample_rate: Output sample rate, ``16000`` or ``8000``. Defaults to ``16000``.
        expressiveness: How expressive the speech is, from ``0`` (flat) to ``2``.
            Sent only when given; Hamsa's default is ``1``.
        base_url: Base URL of the Hamsa API. Defaults to ``"wss://api.tryhamsa.com"``.
        tokenizer: Sentence tokenizer for streaming input. Each sentence becomes one
            request. Defaults to ``tokenize.blingfire.SentenceTokenizer``.
        http_session: Optional shared ``aiohttp.ClientSession``. If not provided,
            the session from the job's HTTP context is used.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        speaker: str = DEFAULT_SPEAKER,
        dialect: TTSDialect | str = DEFAULT_DIALECT,
        sample_rate: TTSSampleRate = DEFAULT_SAMPLE_RATE,
        expressiveness: float | None = None,
        base_url: str = DEFAULT_BASE_URL,
        tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        hamsa_api_key = api_key or os.environ.get("HAMSA_API_KEY")
        if not hamsa_api_key:
            raise ValueError(
                "Hamsa API key is required, either as argument or set"
                " HAMSA_API_KEY environment variable"
            )
        if sample_rate not in (8000, 16000):
            raise ValueError("sample_rate must be 16000 or 8000")
        _validate(dialect, expressiveness)

        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True, aligned_transcript=False),
            sample_rate=sample_rate,
            num_channels=1,
        )

        self._opts = _TTSOptions(
            api_key=hamsa_api_key,
            speaker=speaker,
            dialect=dialect,
            sample_rate=sample_rate,
            expressiveness=expressiveness,
            base_url=base_url,
        )
        self._sentence_tokenizer = (
            tokenizer if is_given(tokenizer) else tokenize.blingfire.SentenceTokenizer()
        )
        self._session = http_session
        self._pool = utils.ConnectionPool[aiohttp.ClientWebSocketResponse](
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
            max_session_duration=300,
            mark_refreshed_on_get=True,
        )
        self._streams = weakref.WeakSet[ChunkedStream | SynthesizeStream]()

    @property
    def model(self) -> str:
        """Returns the model name ``"hamsa-tts"``."""
        return TTS_MODEL

    @property
    def provider(self) -> str:
        """Returns the provider name ``"Hamsa"``."""
        return "Hamsa"

    def update_options(
        self,
        *,
        speaker: NotGivenOr[str] = NOT_GIVEN,
        dialect: NotGivenOr[TTSDialect | str] = NOT_GIVEN,
        expressiveness: NotGivenOr[float | None] = NOT_GIVEN,
    ) -> None:
        """Update the voice settings used by the next requests.

        The settings travel with each request, so pooled connections stay valid.
        Invalid values raise ``ValueError`` and leave the current settings unchanged.

        Args:
            speaker: Voice name or cloned-voice UUID.
            dialect: Dialect code matching the speaker.
            expressiveness: ``0`` to ``2``, or ``None`` for Hamsa's default.
        """
        new = replace(self._opts)
        if is_given(speaker):
            new.speaker = speaker
        if is_given(dialect):
            new.dialect = dialect
        if is_given(expressiveness):
            new.expressiveness = expressiveness
        _validate(new.dialect, new.expressiveness)
        self._opts = new

    async def _connect_ws(self, timeout: float) -> aiohttp.ClientWebSocketResponse:
        session = self._ensure_session()
        # The key goes in a header rather than the URL so it does not end up in
        # exception messages, tracing or intermediary access logs.
        return await asyncio.wait_for(
            session.ws_connect(
                self._opts.get_ws_url(),
                max_msg_size=0,
                headers={"x-api-key": self._opts.api_key},
            ),
            timeout,
        )

    async def _close_ws(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        await ws.close()

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def prewarm(self) -> None:
        """Pre-warm the WebSocket connection pool."""
        self._pool.prewarm()

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> ChunkedStream:
        """Synthesize text to speech in a single request.

        Text longer than Hamsa's 2,000-character request limit is sent as several
        requests on the same connection.

        Args:
            text: The text to synthesize.
            conn_options: Connection options for the request.

        Returns:
            A ``ChunkedStream`` that yields ``SynthesizedAudio`` events.
        """
        stream = ChunkedStream(tts=self, input_text=text, conn_options=conn_options)
        self._streams.add(stream)
        return stream

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
        await self._pool.aclose()


async def _receive_audio(
    ws: aiohttp.ClientWebSocketResponse,
    output_emitter: tts.AudioEmitter,
    *,
    request_id: str,
    timeout: float,
) -> None:
    """Read one request's reply: binary PCM frames until the ``end`` text frame."""
    while True:
        msg = await ws.receive(timeout=timeout)
        if msg.type in (
            aiohttp.WSMsgType.CLOSED,
            aiohttp.WSMsgType.CLOSE,
            aiohttp.WSMsgType.CLOSING,
        ):
            code = ws.close_code or -1
            if code == _AUTH_FAILED_CLOSE_CODE:
                raise APIStatusError(
                    "Hamsa authentication failed", status_code=401, request_id=request_id
                )
            raise APIStatusError(
                "Hamsa connection closed unexpectedly",
                request_id=request_id,
                status_code=code,
                body=None,
            )

        if msg.type == aiohttp.WSMsgType.BINARY:
            output_emitter.push(msg.data)
        elif msg.type == aiohttp.WSMsgType.TEXT:
            try:
                data = json.loads(msg.data)
            except json.JSONDecodeError:
                logger.debug("ignoring non-JSON text frame from Hamsa")
                continue
            kind = data.get("type") if isinstance(data, dict) else None
            if kind == "end":
                return
            if kind in ("ack", "info"):
                continue
            # Anything else is an error report. Don't interpolate the provider's
            # message (it may echo the synthesized text) into the exception.
            logger.debug("Hamsa returned an error frame", extra={"lk.pii.frame": data})
            raise APIError("Hamsa returned an error response", retryable=False)


def _split_text(text: str, limit: int = MAX_REQUEST_CHARS) -> list[str]:
    """Split text into pieces of at most ``limit`` characters, at spaces where possible."""
    text = text.strip()
    if len(text) <= limit:
        return [text] if text else []
    pieces: list[str] = []
    current = ""
    for word in text.split():
        while len(word) > limit:  # a single run longer than the limit: cut it
            if current:
                pieces.append(current)
                current = ""
            pieces.append(word[:limit])
            word = word[limit:]
        if not current:
            current = word
        elif len(current) + 1 + len(word) <= limit:
            current = f"{current} {word}"
        else:
            pieces.append(current)
            current = word
    if current:
        pieces.append(current)
    return pieces


def _build_request(opts: _TTSOptions, text: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "text": text,
        "speaker": opts.speaker,
        "dialect": opts.dialect,
        "languageId": "en" if opts.dialect == "en" else "ar",
        "mulaw": False,
        "sampleRate": "8k" if opts.sample_rate == 8000 else "16k",
    }
    if opts.expressiveness is not None:
        payload["expressiveness"] = opts.expressiveness
    return {"type": "tts", "payload": payload}


class ChunkedStream(tts.ChunkedStream):
    """Non-streaming TTS synthesis: the full text in one request on a pooled WebSocket."""

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
        )

        pieces = _split_text(self._input_text)
        if not pieces:
            output_emitter.flush()  # nothing to say: no request
            return

        try:
            async with self._tts._pool.connection(timeout=self._conn_options.timeout) as ws:
                self._acquire_time = self._tts._pool.last_acquire_time
                self._connection_reused = self._tts._pool.last_connection_reused
                for piece in pieces:
                    await ws.send_str(json.dumps(_build_request(self._opts, piece)))
                    await _receive_audio(
                        ws,
                        output_emitter,
                        request_id=request_id,
                        timeout=self._conn_options.timeout,
                    )
                output_emitter.flush()
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except APIError:
            # Preserve typed errors raised above instead of collapsing them.
            raise
        except aiohttp.WSServerHandshakeError as e:
            raise APIStatusError(
                "Hamsa rejected the connection", status_code=e.status, body=None
            ) from None
        except Exception as e:
            # Keep only the type and drop the cause: a transport error can carry the
            # request headers, which include the API key.
            raise APIConnectionError(type(e).__name__) from None


class SynthesizeStream(tts.SynthesizeStream):
    """Streaming TTS synthesis.

    The input is split into sentences and each sentence is one request on a pooled
    WebSocket, so audio starts with the first sentence. A flush ends the current
    segment; the next text opens a new one.
    """

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
            stream=True,
        )

        language = "en" if self._opts.dialect == "en" else "ar"
        sent_stream = self._tts._sentence_tokenizer.stream(language=language)

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
            # The connection is taken lazily, so a stream with no text makes no request.
            ws: aiohttp.ClientWebSocketResponse | None = None
            segment_id: str | None = None
            try:
                async for ev in sent_stream:
                    text = ev.token.strip()
                    if not text:
                        continue
                    if ws is None:
                        ws = await self._tts._pool.get(timeout=self._conn_options.timeout)
                        self._acquire_time = self._tts._pool.last_acquire_time
                        self._connection_reused = self._tts._pool.last_connection_reused
                    if ev.segment_id != segment_id:
                        if segment_id is not None:
                            output_emitter.end_segment()
                        segment_id = ev.segment_id
                        output_emitter.start_segment(segment_id=segment_id)

                    self._mark_started()
                    # The sentence tokenizer has no length cap; keep each request
                    # within Hamsa's limit.
                    for piece in _split_text(text):
                        await ws.send_str(json.dumps(_build_request(self._opts, piece)))
                        await _receive_audio(
                            ws,
                            output_emitter,
                            request_id=request_id,
                            timeout=self._conn_options.timeout,
                        )
                    output_emitter.flush()

                if segment_id is not None:
                    output_emitter.end_segment()
            except BaseException:
                if ws is not None:
                    self._tts._pool.remove(ws)
                raise
            else:
                if ws is not None:
                    self._tts._pool.put(ws)

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
        except aiohttp.WSServerHandshakeError as e:
            raise APIStatusError(
                "Hamsa rejected the connection", status_code=e.status, body=None
            ) from None
        except Exception as e:
            raise APIConnectionError(type(e).__name__) from None
        finally:
            await utils.aio.cancel_and_wait(*tasks)
            await sent_stream.aclose()
