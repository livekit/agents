from __future__ import annotations

import asyncio
import base64
import contextlib
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
    tts,
    utils,
)
from livekit.agents.language import LanguageCode
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from .log import logger
from .models import (
    DEFAULT_LOCALE,
    DEFAULT_LOCALE_FOR_LANGUAGE,
    DEFAULT_TTS_BASE_URL,
    DEFAULT_TTS_MODEL,
    DEFAULT_TTS_SAMPLE_RATE,
    DEFAULT_VOICE_PROMPT_ID,
    TTS_PATH,
    TTS_SAMPLE_RATES,
    TTSLocales,
    TTSModels,
    TTSSampleRate,
)

_END_STREAM_MESSAGE = json.dumps({"action": "end-stream"})
_CANCEL_MESSAGE = json.dumps({"action": "cancel"})


@dataclass
class _TTSOptions:
    api_key: str
    base_url: str
    model: TTSModels | str
    voice_prompt_id: str
    locale: str
    sample_rate: int
    enable_logging: bool | None

    def get_ws_url(self) -> str:
        return f"{self.base_url.rstrip('/')}{TTS_PATH}"


class TTS(tts.TTS):
    """Deepdub streaming text-to-speech (the LLM Integration / Voice Agents API).

    Text streams in as the LLM produces it and audio streams back while it is
    generated, over one persistent WebSocket session. Deepdub buffers the text and
    picks the segment boundaries itself, so tokens are forwarded as they arrive.
    Each agent reply is one Deepdub turn, closed with ``end-stream`` on flush or at
    the end of the input; an interrupted reply is aborted with ``cancel``.

    Args:
        api_key: Deepdub API key. Defaults to the ``DEEPDUB_API_KEY`` environment
            variable.
        model: Deepdub model. Defaults to ``"dd-etts-4.0"``.
        voice_prompt_id: Voice prompt to speak with. Defaults to the English
            "Call Center Agent (Neutral)" preset. See
            https://docs.deepdub.ai/voice-presets for other presets.
        locale: Locale of the text, for example ``"en-US"`` or ``"he-IL"``. A bare
            language such as ``"en"`` uses that language's default locale.
            Defaults to ``"en-US"``.
        sample_rate: Output sample rate in Hz. Defaults to 24000.
        enable_logging: Whether Deepdub may keep the session's text in its
            server-side logs. Uses the account default when not set.
        base_url: WebSocket host. Defaults to ``"wss://wss.deepdub.ai"``; use
            ``"wss://wss.eu.deepdub.ai"`` for an EU account.
        http_session: Optional shared ``aiohttp.ClientSession``.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: TTSModels | str = DEFAULT_TTS_MODEL,
        voice_prompt_id: str = DEFAULT_VOICE_PROMPT_ID,
        locale: TTSLocales | str = DEFAULT_LOCALE,
        sample_rate: TTSSampleRate | int = DEFAULT_TTS_SAMPLE_RATE,
        enable_logging: bool | None = None,
        base_url: str = DEFAULT_TTS_BASE_URL,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        deepdub_api_key = api_key or os.environ.get("DEEPDUB_API_KEY")
        if not deepdub_api_key:
            raise ValueError(
                "Deepdub API key is required, either as argument or set"
                " DEEPDUB_API_KEY environment variable"
            )
        if sample_rate not in TTS_SAMPLE_RATES:
            raise ValueError(f"sample_rate must be one of {sorted(TTS_SAMPLE_RATES)}")

        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True, aligned_transcript=False),
            sample_rate=sample_rate,
            num_channels=1,
        )

        self._opts = _TTSOptions(
            api_key=deepdub_api_key,
            base_url=base_url,
            model=model,
            voice_prompt_id=voice_prompt_id,
            locale=_resolve_locale(locale),
            sample_rate=sample_rate,
            enable_logging=enable_logging,
        )
        self._session = http_session
        self._pool = utils.ConnectionPool[aiohttp.ClientWebSocketResponse](
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
            max_session_duration=300,
            mark_refreshed_on_get=True,
        )
        self._streams = weakref.WeakSet[SynthesizeStream]()

    @property
    def model(self) -> str:
        """The Deepdub model name."""
        return self._opts.model

    @property
    def provider(self) -> str:
        """The provider name, ``"Deepdub"``."""
        return "Deepdub"

    async def _connect_ws(self, timeout: float) -> aiohttp.ClientWebSocketResponse:
        ws = await asyncio.wait_for(
            self._ensure_session().ws_connect(
                self._opts.get_ws_url(),
                max_msg_size=0,
                headers={"x-api-key": self._opts.api_key},
            ),
            timeout,
        )
        try:
            # The server greets every connection with {"action": "status", "message": "connected"}.
            welcome = await _receive_message(ws, timeout=timeout)
            if welcome.get("action") != "status":
                raise APIConnectionError("unexpected first message from Deepdub")
        except BaseException:
            await ws.close()
            raise
        return ws

    async def _close_ws(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        await ws.close()

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def prewarm(self) -> None:
        """Open a pooled WebSocket connection ahead of the first request."""
        self._pool.prewarm()

    def update_options(
        self,
        *,
        model: NotGivenOr[TTSModels | str] = NOT_GIVEN,
        voice_prompt_id: NotGivenOr[str] = NOT_GIVEN,
        locale: NotGivenOr[TTSLocales | str] = NOT_GIVEN,
    ) -> None:
        """Update synthesis options. They apply to streams created after the call.

        Args:
            model: Deepdub model.
            voice_prompt_id: Voice prompt to speak with.
            locale: Locale of the text.
        """
        if is_given(model):
            self._opts.model = model
        if is_given(voice_prompt_id):
            self._opts.voice_prompt_id = voice_prompt_id
        if is_given(locale):
            self._opts.locale = _resolve_locale(locale)

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> ChunkedStream:
        """Synthesize one complete text as a single Deepdub turn.

        Args:
            text: The text to synthesize.
            conn_options: Connection options for the request.

        Returns:
            A ``ChunkedStream`` that yields audio as it is generated.
        """
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        """Create a streaming session that accepts text as it is produced.

        Args:
            conn_options: Connection options for the session.

        Returns:
            A ``SynthesizeStream``.
        """
        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        """Close all streams and pooled connections."""
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()
        await self._pool.aclose()


class ChunkedStream(tts.ChunkedStream):
    """Synthesis of one complete text: ``stream-text`` followed by ``end-stream``."""

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
        if not self._input_text.strip():
            return

        try:
            async with self._tts._pool.connection(timeout=self._conn_options.timeout) as ws:
                await ws.send_str(json.dumps(_config_message(self._opts)))
                await ws.send_str(_text_message(self._input_text))
                await ws.send_str(_END_STREAM_MESSAGE)
                while True:
                    message = await _receive_message(ws, timeout=self._conn_options.timeout)
                    if _handle_audio(message, output_emitter):
                        break
            output_emitter.flush()
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except APIError:
            raise
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=None, body=None
            ) from None
        except Exception as e:
            # Keep only the exception type: an aiohttp transport error can carry the
            # request headers, which include the API key.
            raise APIConnectionError(type(e).__name__) from None


class SynthesizeStream(tts.SynthesizeStream):
    """Streaming synthesis over one Deepdub session.

    ``push_text`` forwards each token as ``stream-text``. ``flush`` closes the
    current turn with ``end-stream`` and waits for its final chunk before the next
    turn begins; ``end_input`` closes the last turn. Closing the stream while a turn
    is still being spoken sends ``cancel``.
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

        try:
            async with self._tts._pool.connection(timeout=self._conn_options.timeout) as ws:
                self._acquire_time = self._tts._pool.last_acquire_time
                self._connection_reused = self._tts._pool.last_connection_reused
                await ws.send_str(json.dumps(_config_message(self._opts)))
                await self._run_session(ws, output_emitter)
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except APIError:
            raise
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=None, body=None
            ) from None
        except Exception as e:
            # Keep only the exception type: an aiohttp transport error can carry the
            # request headers, which include the API key.
            raise APIConnectionError(type(e).__name__) from None

    async def _run_session(
        self, ws: aiohttp.ClientWebSocketResponse, output_emitter: tts.AudioEmitter
    ) -> None:
        loop = asyncio.get_running_loop()
        timeout = self._conn_options.timeout
        in_turn = False
        turn_done = asyncio.Event()
        last_message_at = loop.time()

        async def _end_turn() -> None:
            nonlocal in_turn
            await ws.send_str(_END_STREAM_MESSAGE)
            ended_at = loop.time()
            # Deepdub is silent while the LLM is still writing, so the deadline starts
            # at end-stream and is extended by every message that follows it.
            while not turn_done.is_set():
                remaining = max(ended_at, last_message_at) + timeout - loop.time()
                if remaining <= 0:
                    raise APITimeoutError(f"Deepdub sent nothing for {timeout}s after end-stream")
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(turn_done.wait(), remaining)
            turn_done.clear()
            in_turn = False

        async def _send_task() -> None:
            nonlocal in_turn
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    if in_turn:
                        await _end_turn()
                    continue
                # A turn starts at the first token with something to say, so every turn
                # sent to Deepdub produces audio and a final chunk.
                if not in_turn:
                    if not data.strip():
                        continue
                    in_turn = True
                    output_emitter.start_segment(segment_id=utils.shortuuid())
                    self._mark_started()
                await ws.send_str(_text_message(data))
            if in_turn:
                await _end_turn()

        async def _recv_task() -> None:
            nonlocal last_message_at
            while True:
                message = await _receive_message(ws, timeout=None)
                last_message_at = loop.time()
                if _handle_audio(message, output_emitter):
                    output_emitter.end_segment()
                    turn_done.set()

        send = asyncio.create_task(_send_task())
        recv = asyncio.create_task(_recv_task())
        try:
            done, _ = await asyncio.wait([send, recv], return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
            if recv in done:
                raise APIConnectionError("Deepdub session ended unexpectedly")
        finally:
            await utils.aio.cancel_and_wait(send, recv)
            if in_turn and not ws.closed:
                # The reply was interrupted: stop Deepdub generating audio nobody hears.
                # The connection is then dropped from the pool rather than reused.
                with contextlib.suppress(Exception):
                    await asyncio.shield(ws.send_str(_CANCEL_MESSAGE))


async def _receive_message(
    ws: aiohttp.ClientWebSocketResponse, *, timeout: float | None
) -> dict[str, Any]:
    """Return the next JSON message, raising on a closed connection or an error."""
    while True:
        msg = await ws.receive(timeout=timeout)
        if msg.type in (
            aiohttp.WSMsgType.CLOSED,
            aiohttp.WSMsgType.CLOSE,
            aiohttp.WSMsgType.CLOSING,
            aiohttp.WSMsgType.ERROR,
        ):
            raise APIStatusError(
                "Deepdub connection closed unexpectedly",
                status_code=ws.close_code or -1,
                request_id=None,
                body=None,
            )
        # Deepdub sends its JSON messages in binary as well as text frames.
        if msg.type not in (aiohttp.WSMsgType.TEXT, aiohttp.WSMsgType.BINARY):
            continue
        try:
            message = json.loads(msg.data)
        except (json.JSONDecodeError, UnicodeDecodeError):
            logger.debug("ignoring a frame from Deepdub that is not JSON")
            continue
        if not isinstance(message, dict):
            continue

        if message.get("action") == "error":
            # Session-level errors are configuration problems; retrying cannot fix them.
            raise APIStatusError(
                f"Deepdub TTS error: {message.get('message')}",
                status_code=400,
                request_id=None,
                body=None,
            )
        if message.get("error"):
            raise APIStatusError(
                f"Deepdub TTS generation failed: {message.get('error')}",
                request_id=message.get("generationId"),
                body=None,
            )
        return message


def _handle_audio(message: dict[str, Any], output_emitter: tts.AudioEmitter) -> bool:
    """Push the audio in a message. Returns ``True`` on the final chunk of the turn."""
    if message.get("action"):
        # `status` and `pong` messages carry no audio.
        return False
    if message.get("isCancelled"):
        return False
    if chunk := message.get("data"):
        output_emitter.push(base64.b64decode(chunk))
    return bool(message.get("isFinal"))


def _config_message(opts: _TTSOptions) -> dict[str, Any]:
    config: dict[str, Any] = {
        "model": opts.model,
        "voicePromptId": opts.voice_prompt_id,
        "locale": opts.locale,
        "format": "s16le",
        "sampleRate": opts.sample_rate,
    }
    if opts.enable_logging is not None:
        config["enableLogging"] = opts.enable_logging
    return {"action": "stream-config", "data": config}


def _text_message(text: str) -> str:
    return json.dumps({"action": "stream-text", "data": {"text": text}})


def _resolve_locale(locale: str) -> str:
    """Normalize a locale, expanding a bare language to its default Deepdub locale."""
    code = LanguageCode(locale)
    if "-" in code:
        return str(code)
    return DEFAULT_LOCALE_FOR_LANGUAGE.get(code.language, str(code))
