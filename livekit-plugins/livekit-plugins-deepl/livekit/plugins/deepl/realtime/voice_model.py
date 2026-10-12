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

"""DeepL Voice API as a duplex model: one speaker's audio in, their speech translated out."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import time
from collections import deque
from collections.abc import AsyncIterable, AsyncIterator, Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

import aiohttp

from livekit import rtc
from livekit.agents import APIConnectionError, APIError, APIStatusError, llm, utils
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import is_given

from ..log import logger
from ..models import Formality, SourceLanguageMode, Voice
from ..version import __version__

DEEPL_API_URL = "https://api.deepl.com"
DEEPL_FREE_API_URL = "https://api-free.deepl.com"
_SESSION_PATH = "/v3/voice/realtime"
_USER_AGENT = f"livekit-plugins-deepl/{__version__}"

INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000
NUM_CHANNELS = 1
_SOURCE_CONTENT_TYPE = f"audio/pcm;encoding=s16le;rate={INPUT_SAMPLE_RATE}"
_TARGET_CONTENT_TYPE = f"audio/pcm;encoding=s16le;rate={OUTPUT_SAMPLE_RATE}"

# input goes out in 100 ms chunks (DeepL recommends 50-250 ms); 3 s are held while disconnected
_INPUT_CHUNK_SAMPLES = INPUT_SAMPLE_RATE // 10
_MAX_INPUT_CHUNKS = 30
_SILENCE_CHUNK = bytes(_INPUT_CHUNK_SAMPLES * 2)
# DeepL closes a session after 30 s without audio, and a muted track sends none
_KEEPALIVE_INTERVAL_S = 10.0

# output is cut into 20 ms frames, handed over slightly ahead of their playback time
_OUTPUT_FRAME_SAMPLES = OUTPUT_SAMPLE_RATE // 50
_PLAYOUT_LEAD_MS = 100.0

# a speaker's turn ends after this much of their audio with no tentative text left, or after
# the longer silence regardless
_TURN_SILENCE_MS = 800.0
_TURN_MAX_SILENCE_MS = 3000.0

# how long DeepL gets to flush its last output after end_of_source_media
_CLOSE_TIMEOUT_S = 2.0
# a connection that lasted this long resets the retry count
_STABLE_CONNECTION_S = 10.0
# statuses a new session would fail with too
_FATAL_STATUS = frozenset({401, 403, 456})

_WS_CLOSED = (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSING)
_IGNORED_MESSAGES = (
    "target_transcript_update",
    "end_of_source_transcript",
    "end_of_target_transcript",
    "end_of_target_media",
)


@dataclass
class _VoiceOptions:
    target_language: str
    source_language: str | None
    source_language_mode: SourceLanguageMode | None
    voice: Voice | None
    formality: Formality | None
    glossary_ids: list[str] | None
    api_key: str
    base_url: str
    conn_options: APIConnectOptions


class _SpeechGate:
    """DeepL sends only speech, never silence, so every frame it sends is worth playing."""

    def update(self, frame: rtc.AudioFrame) -> bool:
        return True

    def deactivate(self) -> None:
        pass


class VoiceModel(llm.DuplexModel):
    """DeepL Voice realtime translation, ready to pass to ``AgentSession(llm=)``.

    Each session interprets one speaker into one language, speaking the translation over them as
    a simultaneous interpreter does, and transcribes both sides. To bridge a conversation, run a
    session per speaker and target language.
    """

    def __init__(
        self,
        *,
        target_language: str,
        source_language: NotGivenOr[str] = NOT_GIVEN,
        source_language_mode: NotGivenOr[SourceLanguageMode] = NOT_GIVEN,
        voice: NotGivenOr[Voice] = NOT_GIVEN,
        formality: NotGivenOr[Formality] = NOT_GIVEN,
        glossary_ids: NotGivenOr[list[str]] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        """
        Args:
            target_language: Language to speak the translation in, as an IETF BCP 47 tag
                (``de``, ``pt-BR``); DeepL speaks one per session.
            source_language: Language the speaker uses. Unset, DeepL detects it.
            source_language_mode: ``fixed`` holds DeepL to ``source_language``, skipping
                detection for lower latency; ``auto`` treats it as a hint. Defaults to
                ``fixed`` when ``source_language`` is given.
            voice: ``male``, ``female``, or ``match`` for a voice close to the speaker's own.
                Defaults to DeepL's voice for the target language. ``match`` falls back to a
                preset voice for languages whose speech DeepL does not synthesize itself.
            formality: Register of the translation; defaults to the target language's own.
            glossary_ids: DeepL glossaries to apply, highest priority first, at most 10.
            api_key: DeepL API key. Falls back to ``DEEPL_API_KEY``, then ``DEEPL_AUTH_KEY``.
            base_url: DeepL API base url. Falls back to ``DEEPL_BASE_URL``, then to the Free
                API for keys ending in ``:fx`` and the Pro API otherwise.
            http_session: Optional shared HTTP session.
            conn_options: Retry/backoff and connection settings.
        """
        super().__init__(
            capabilities=llm.DuplexCapabilities(
                user_transcription=True,
                auto_tool_reply_generation=False,
            )
        )
        if not target_language:
            raise ValueError("target_language is required")

        if is_given(source_language_mode):
            mode: SourceLanguageMode | None = source_language_mode
        else:
            mode = "fixed" if is_given(source_language) else None
        if mode == "fixed" and not source_language:
            raise ValueError("source_language_mode='fixed' needs a source_language")

        key = _resolve_api_key(api_key)
        self._opts = _VoiceOptions(
            target_language=target_language,
            source_language=source_language if is_given(source_language) else None,
            source_language_mode=mode,
            voice=voice if is_given(voice) else None,
            formality=formality if is_given(formality) else None,
            glossary_ids=list(glossary_ids) if is_given(glossary_ids) else None,
            api_key=key,
            base_url=_resolve_base_url(base_url, key),
            conn_options=conn_options,
        )
        self._http_session = http_session
        self._http_session_owned = False

    @property
    def model(self) -> str:
        return "voice"

    @property
    def provider(self) -> str:
        return "DeepL"

    def _ensure_http_session(self) -> aiohttp.ClientSession:
        if not self._http_session:
            try:
                self._http_session = utils.http_context.http_session()
            except RuntimeError:
                self._http_session = aiohttp.ClientSession()
                self._http_session_owned = True
        return self._http_session

    def audio_gate(self) -> llm.AudioGate:
        return _SpeechGate()

    def session(self) -> VoiceSession:
        return VoiceSession(self)

    async def aclose(self) -> None:
        if self._http_session_owned and self._http_session:
            await self._http_session.close()


class VoiceSession(llm.DuplexSession[Literal["deepl_server_event_received"]]):
    """A DeepL Voice session, reached with ``Agent.duplex_session``.

    Exposes one extra event:
    - deepl_server_event_received: every message DeepL sends, as a dict
    """

    def __init__(self, voice_model: VoiceModel) -> None:
        super().__init__(voice_model)
        self._voice_model = voice_model
        self._opts = voice_model._opts
        self._input = _InputBuffer()
        self._output = _OutputPacer()
        self._turns = _SpeakerTurns(self.emit)
        self._audio_ch = utils.aio.Chan[llm.DuplexAudioFrame]()
        self._warned_instructions = False

        # each token opens one connection, so a reconnect exchanges the newest for a new one
        self._token: str | None = None
        self._session_id: str | None = None
        self._connected = False
        self._num_retries = 0

        self._main_atask = asyncio.create_task(self._main_task(), name="DeepLVoiceSession._main")
        self._playout_atask = asyncio.create_task(
            self._playout_task(), name="DeepLVoiceSession._playout"
        )

    # connection

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        has_connected = False
        try:
            while not self._closing:
                connected_at: float | None = None
                try:
                    ws = await self._connect()
                    connected_at = time.monotonic()
                    if has_connected:
                        self._on_reconnected()
                    has_connected = True
                    await self._run_ws(ws)
                    return
                except APIError as e:
                    if self._closing:
                        return
                    await asyncio.sleep(self._retry_delay(e, connected_at))
                except Exception as e:
                    self._emit_error(e, recoverable=False)
                    raise
                finally:
                    self._connected = False
        finally:
            self._output.end()

    def _retry_delay(self, error: APIError, connected_at: float | None) -> float:
        """How long to wait before reconnecting; raises once retrying is pointless."""
        conn_options = self._opts.conn_options
        if connected_at is not None and time.monotonic() - connected_at >= _STABLE_CONNECTION_S:
            self._num_retries = 0

        if conn_options.max_retry == 0 or not error.retryable:
            self._emit_error(error, recoverable=False)
            raise error
        if self._num_retries == conn_options.max_retry:
            self._emit_error(error, recoverable=False)
            raise APIConnectionError(
                f"DeepL Voice connection failed after {self._num_retries} attempts"
            ) from error

        self._emit_error(error, recoverable=True)
        delay = conn_options._interval_for_retry(self._num_retries)
        logger.warning(
            f"DeepL Voice connection failed, retrying in {delay}s",
            exc_info=error,
            extra={"attempt": self._num_retries, "max_retries": conn_options.max_retry},
        )
        self._num_retries += 1
        return delay

    def _on_reconnected(self) -> None:
        # the adapter drops the reply in flight when it hears of a reconnect
        self._output.reanchor()
        self._turns.end()
        self.emit("session_reconnected", llm.RealtimeSessionReconnectedEvent())

    async def _connect(self) -> aiohttp.ClientWebSocketResponse:
        t0 = time.perf_counter()
        url, token = await self._request_stream_url()
        self._token = token
        try:
            ws = await asyncio.wait_for(
                self._voice_model._ensure_http_session().ws_connect(
                    url, params={"token": token}, headers={"User-Agent": _USER_AGENT}
                ),
                self._opts.conn_options.timeout,
            )
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            raise APIConnectionError("DeepL Voice connection error") from e
        self._connected = True
        self._report_connection_acquired(time.perf_counter() - t0)
        return ws

    async def _request_stream_url(self) -> tuple[str, str]:
        """A streaming url and token, resuming the current session if DeepL still has it."""
        if self._token is not None:
            try:
                return await self._call_session_api("GET", token=self._token)
            except APIStatusError as e:
                if e.retryable or e.status_code in _FATAL_STATUS:
                    raise
                logger.warning(
                    "DeepL Voice could not resume the session, starting a new one",
                    extra={"status_code": e.status_code, "trace_id": e.request_id},
                )
        return await self._call_session_api("POST", body=self._session_request_body())

    async def _call_session_api(
        self,
        method: Literal["GET", "POST"],
        *,
        body: dict[str, Any] | None = None,
        token: str | None = None,
    ) -> tuple[str, str]:
        """Create (POST) or resume (GET) a session; returns its streaming url and token."""
        try:
            async with self._voice_model._ensure_http_session().request(
                method,
                f"{self._opts.base_url.rstrip('/')}{_SESSION_PATH}",
                headers={
                    "Authorization": f"DeepL-Auth-Key {self._opts.api_key}",
                    "User-Agent": _USER_AGENT,
                },
                json=body,
                params={"token": token} if token else None,
                timeout=aiohttp.ClientTimeout(total=self._opts.conn_options.timeout),
            ) as resp:
                trace_id = resp.headers.get("X-Trace-ID")
                if resp.status != 200:
                    detail = await resp.text()
                    raise APIStatusError(
                        f"DeepL Voice session request failed ({resp.status}, "
                        f"trace id {trace_id}): {detail}",
                        status_code=resp.status,
                        request_id=trace_id,
                        body=detail,
                    )
                data = await resp.json()
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            raise APIConnectionError("DeepL Voice session request failed") from e

        try:
            streaming_url, new_token = data["streaming_url"], data["token"]
        except (KeyError, TypeError):
            raise APIError(
                "DeepL Voice returned a malformed session response", retryable=False
            ) from None
        self._session_id = data.get("session_id") or self._session_id
        return streaming_url, new_token

    def _session_request_body(self) -> dict[str, Any]:
        opts = self._opts
        body: dict[str, Any] = {
            "source_media_content_type": _SOURCE_CONTENT_TYPE,
            "target_media_languages": [opts.target_language],
            "target_media_content_type": _TARGET_CONTENT_TYPE,
            "message_format": "json",
        }
        if opts.source_language:
            body["source_language"] = opts.source_language
        if opts.source_language_mode:
            body["source_language_mode"] = opts.source_language_mode
        if opts.voice:
            body["target_media_voice"] = opts.voice
        if opts.formality:
            body["formality"] = opts.formality
        if opts.glossary_ids:
            body["glossary_ids"] = opts.glossary_ids
        return body

    # streaming

    async def _run_ws(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        send_task = asyncio.create_task(self._send_loop(ws), name="DeepLVoiceSession._send")
        recv_task = asyncio.create_task(self._recv_loop(ws), name="DeepLVoiceSession._recv")
        try:
            done, _ = await asyncio.wait(
                {send_task, recv_task}, return_when=asyncio.FIRST_COMPLETED
            )
            if send_task in done:
                send_task.result()
                # end_of_source_media is out: let DeepL flush what it holds and end the stream
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(asyncio.shield(recv_task), _CLOSE_TIMEOUT_S)
                if not recv_task.done():
                    return
            recv_task.result()
            if not self._closing:
                raise APIConnectionError("DeepL Voice ended the stream unexpectedly")
        finally:
            await utils.aio.cancel_and_wait(send_task, recv_task)
            await ws.close()

    async def _send_loop(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        sent_at, sent_duration = 0.0, 0.0
        while (chunk := await self._input.get()) is not None:
            # DeepL refuses a chunk that follows the previous one by less than half its duration
            if (wait := sent_at + sent_duration / 2 - time.monotonic()) > 0:
                await asyncio.sleep(wait)
            data = base64.b64encode(chunk).decode()
            await self._ws_send(ws, {"source_media_chunk": {"data": data}})
            sent_at, sent_duration = time.monotonic(), len(chunk) / 2 / INPUT_SAMPLE_RATE
        await self._ws_send(ws, {"end_of_source_media": {}})

    async def _recv_loop(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        while True:
            try:
                msg = await ws.receive()
            except (aiohttp.ClientError, ConnectionError, asyncio.TimeoutError) as e:
                raise APIConnectionError("DeepL Voice receive failed") from e
            if msg.type in _WS_CLOSED:
                if self._closing:
                    return
                raise APIConnectionError("DeepL Voice connection closed unexpectedly")
            if msg.type != aiohttp.WSMsgType.TEXT:
                continue

            try:
                event = json.loads(msg.data)
            except json.JSONDecodeError:
                logger.warning("DeepL Voice sent a message that is not JSON")
                continue
            self.emit("deepl_server_event_received", event)
            try:
                if self._handle_event(event):
                    return
            except APIError:
                raise
            except Exception as e:
                logger.warning(
                    "failed to handle DeepL Voice message",
                    extra={"keys": sorted(event), "error_type": type(e).__name__},
                )

    async def _ws_send(self, ws: aiohttp.ClientWebSocketResponse, msg: dict[str, Any]) -> None:
        try:
            await ws.send_str(json.dumps(msg))
        except (aiohttp.ClientError, ConnectionError, asyncio.TimeoutError) as e:
            raise APIConnectionError("DeepL Voice send failed") from e

    # DeepL messages

    def _handle_event(self, event: dict[str, Any]) -> bool:
        """Handle one DeepL message; True once the stream has ended."""
        if "end_of_stream" in event:
            if not self._closing:
                self._token = None  # a stream DeepL ended itself cannot be resumed
            return True
        if (error := event.get("error")) is not None:
            self._handle_error(error)
        if self._closing:
            return False

        if (update := event.get("source_transcript_update")) is not None:
            self._turns.update(_joined(update.get("concluded")), _joined(update.get("tentative")))
        elif (chunk := event.get("target_media_chunk")) is not None:
            self._handle_target_media(chunk)
        elif not any(key in event for key in _IGNORED_MESSAGES):
            logger.debug("unhandled DeepL Voice message", extra={"keys": sorted(event)})
        return False

    def _handle_error(self, error: dict[str, Any]) -> None:
        # DeepL ends the session after any error, so only a new session can continue
        self._token = None
        code = error.get("error_code")
        raise APIStatusError(
            f"DeepL Voice error: {error.get('error_message') or 'unknown error'}",
            status_code=code if isinstance(code, int) else -1,
            body=error,
        )

    def _handle_target_media(self, chunk: dict[str, Any]) -> None:
        content_type = str(chunk.get("content_type") or "").replace(" ", "")
        if content_type and not (
            content_type.startswith("audio/pcm") and f"rate={OUTPUT_SAMPLE_RATE}" in content_type
        ):
            raise APIError(
                f"DeepL Voice sent {content_type} audio, expected {_TARGET_CONTENT_TYPE}",
                retryable=False,
            )
        # header packets, if any, only initialize a decoder
        packets = (chunk.get("data") or [])[chunk.get("headers") or 0 :]
        self._output.add(b"".join(base64.b64decode(p) for p in packets), chunk.get("text"))

    async def _playout_task(self) -> None:
        try:
            async for start_ms, frame, text in self._output.frames():
                if text is not None:
                    self.emit(
                        "transcript_delta",
                        llm.DuplexOutputTranscriptDelta(text=text, start_ms=start_ms),
                    )
                with contextlib.suppress(utils.aio.channel.ChanClosed):
                    self._audio_ch.send_nowait(llm.DuplexAudioFrame(frame=frame, start_ms=start_ms))
        finally:
            self._audio_ch.close()

    def _emit_error(self, error: Exception, recoverable: bool) -> None:
        self.emit(
            "error",
            llm.RealtimeModelError(
                timestamp=time.time(),
                label=self._voice_model.label,
                error=error,
                recoverable=recoverable,
            ),
        )

    # DuplexSession interface

    @property
    def session_id(self) -> str | None:
        """DeepL's id for the current session."""
        return self._session_id

    @property
    def audio_stream(self) -> AsyncIterable[llm.DuplexAudioFrame]:
        return self._audio_ch

    @property
    def tools(self) -> llm.ToolContext:
        return llm.ToolContext.empty()

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        if self._closing:
            return
        self._turns.on_audio(frame.duration * 1000)
        self._input.push(frame)

    async def aclose(self) -> None:
        await super().aclose()
        self._input.close()
        if self._connected:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(asyncio.shield(self._main_atask), _CLOSE_TIMEOUT_S + 0.5)
        await utils.aio.cancel_and_wait(self._main_atask, self._playout_atask)
        if not self._main_atask.cancelled():
            self._main_atask.exception()  # already reported through the error event
        self._audio_ch.close()

    # framework hooks

    async def _update_instructions(self, instructions: str) -> None:
        if instructions.strip() and not self._warned_instructions:
            self._warned_instructions = True
            logger.warning(
                "DeepL Voice translates what it hears and takes no instructions, ignoring them"
            )

    async def _append_items(self, items: list[llm.ChatItem]) -> None:
        pass

    async def _update_tools(self, tools: list[llm.Tool]) -> None:
        if tools:
            raise llm.RealtimeError(
                "DeepL Voice cannot call tools, leave the agent's tools empty: "
                f"{sorted(tool.id for tool in llm.ToolContext(tools).flatten())}"
            )

    def _update_options(
        self, *, tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN
    ) -> None:
        pass

    def _generate_reply(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        tools: NotGivenOr[list[llm.Tool]] = NOT_GIVEN,
    ) -> None:
        raise llm.RealtimeError(
            "DeepL Voice only speaks translations of what it hears and cannot generate a reply; "
            "remove generate_reply() calls such as an on_enter greeting"
        )


class _InputBuffer:
    """The speaker's audio in DeepL's input format: 16 kHz mono, in 100 ms chunks.

    While nothing can be sent it keeps the most recent few seconds, and an idle stretch yields
    silence so DeepL keeps the session open.
    """

    def __init__(self) -> None:
        self._resampler: rtc.AudioResampler | None = None
        self._resampler_rate = 0
        self._bstream = utils.audio.AudioByteStream(
            INPUT_SAMPLE_RATE, NUM_CHANNELS, samples_per_channel=_INPUT_CHUNK_SAMPLES
        )
        self._chunks: deque[bytes] = deque(maxlen=_MAX_INPUT_CHUNKS)
        self._ready = asyncio.Event()
        self._closed = False
        self._dropped = 0

    def push(self, frame: rtc.AudioFrame) -> None:
        for resampled in self._resample(frame):
            for chunk in self._bstream.write(resampled.data.tobytes()):
                self._append(chunk.data.tobytes())

    async def get(self) -> bytes | None:
        """The next chunk, silence after an idle stretch, or None once closed."""
        while not self._chunks:
            if self._closed:
                return None
            self._ready.clear()
            try:
                await asyncio.wait_for(self._ready.wait(), _KEEPALIVE_INTERVAL_S)
            except asyncio.TimeoutError:
                return _SILENCE_CHUNK
        return self._chunks.popleft()

    def close(self) -> None:
        self._closed = True
        self._chunks.clear()
        self._ready.set()
        # release the resampler's native handle now rather than at process exit
        self._resampler = None

    def _resample(self, frame: rtc.AudioFrame) -> Iterator[rtc.AudioFrame]:
        if frame.sample_rate == INPUT_SAMPLE_RATE and frame.num_channels == NUM_CHANNELS:
            yield frame
            return
        if self._resampler is None or self._resampler_rate != frame.sample_rate:
            self._resampler = rtc.AudioResampler(
                input_rate=frame.sample_rate,
                output_rate=INPUT_SAMPLE_RATE,
                num_channels=NUM_CHANNELS,
            )
            self._resampler_rate = frame.sample_rate
        yield from self._resampler.push(frame)

    def _append(self, chunk: bytes) -> None:
        if len(self._chunks) == self._chunks.maxlen:
            self._dropped += 1
            if self._dropped == 1 or self._dropped % 100 == 0:
                logger.warning(
                    "DeepL Voice is not keeping up with the speaker, dropping the oldest audio",
                    extra={"dropped_chunks": self._dropped},
                )
        self._chunks.append(chunk)
        self._ready.set()


@dataclass
class _Turn:
    item_id: str
    started_at: float = field(default_factory=time.time)
    concluded: str = ""
    tentative: str = ""
    quiet_ms: float = 0.0
    """The speaker's audio since their transcript last changed."""

    @property
    def transcript(self) -> str:
        return (self.concluded + self.tentative).strip()


class _SpeakerTurns:
    """Turns DeepL's transcript of the speaker into turn events for the framework."""

    def __init__(self, emit: Callable[..., None]) -> None:
        self._emit = emit
        self._turn: _Turn | None = None

    def update(self, concluded: str, tentative: str) -> None:
        """Apply a transcript update: concluded text arrives once, tentative text is replaced."""
        turn = self._turn
        if turn is None:
            if not (concluded.strip() or tentative.strip()):
                return
            turn = self._turn = _Turn(item_id=utils.shortuuid("deepl_input_"))
            self._emit("input_speech_started", llm.InputSpeechStartedEvent())

        if not concluded and tentative == turn.tentative:
            return
        turn.concluded += concluded
        turn.tentative = tentative
        turn.quiet_ms = 0.0
        if turn.transcript:
            self._emit_transcript(turn, is_final=False)

    def on_audio(self, duration_ms: float) -> None:
        """Count the speaker's audio; enough of it without new text ends the turn."""
        if (turn := self._turn) is None:
            return
        turn.quiet_ms += duration_ms
        if turn.quiet_ms >= _TURN_MAX_SILENCE_MS or (
            not turn.tentative and turn.quiet_ms >= _TURN_SILENCE_MS
        ):
            turn.tentative = ""  # DeepL concludes it in a later turn
            self.end()

    def end(self) -> None:
        if (turn := self._turn) is None:
            return
        self._turn = None
        if turn.transcript:
            self._emit_transcript(turn, is_final=True)
        self._emit(
            "input_speech_stopped", llm.InputSpeechStoppedEvent(user_transcription_enabled=True)
        )

    def _emit_transcript(self, turn: _Turn, *, is_final: bool) -> None:
        self._emit(
            "input_audio_transcription_completed",
            llm.InputTranscriptionCompleted(
                item_id=turn.item_id,
                transcript=turn.transcript,
                is_final=is_final,
                # the translation plays over the speaker, so the turn is placed where it began
                turn_started_at=turn.started_at,
            ),
        )


@dataclass
class _Frame:
    start_ms: float
    frame: rtc.AudioFrame
    text: str | None
    """The translation this frame begins, if it begins one."""


class _OutputPacer:
    """Releases DeepL's translated speech at playback speed, on a clock that keeps its pauses.

    DeepL sends speech faster than real time and without the silence between segments, while the
    duplex adapter ends a reply when audio stops arriving. So frames are released when they would
    play, stamped in ms since the session began: speech after a pause starts later on that clock.
    """

    def __init__(self) -> None:
        self._t0 = time.monotonic()
        self._queue: deque[_Frame] = deque()
        self._ready = asyncio.Event()
        self._ended = False
        self._scheduled_end_ms = 0.0
        self._released_end_ms = 0.0
        self._pending_text = ""
        self._carry = b""
        self._reanchor = False

    def add(self, pcm: bytes, text: str | None) -> None:
        """Queue translated speech; text without audio waits for the next audio."""
        if text:
            self._pending_text += text
        pcm = self._carry + pcm
        whole = len(pcm) - len(pcm) % 2
        self._carry = pcm[whole:]
        if not whole:
            return

        frame_text: str | None = self._pending_text or None
        self._pending_text = ""
        # audio arriving after the queue ran dry plays from now, leaving a pause on the clock
        start_ms = max(self._scheduled_end_ms, self._now_ms())
        step = _OUTPUT_FRAME_SAMPLES * 2
        for offset in range(0, whole, step):
            data = pcm[offset : min(offset + step, whole)]
            frame = rtc.AudioFrame(
                data=data,
                sample_rate=OUTPUT_SAMPLE_RATE,
                num_channels=NUM_CHANNELS,
                samples_per_channel=len(data) // 2,
            )
            self._queue.append(_Frame(start_ms=start_ms, frame=frame, text=frame_text))
            frame_text = None
            start_ms += frame.duration * 1000
        self._scheduled_end_ms = start_ms
        self._ready.set()

    def reanchor(self) -> None:
        """The reply in flight was dropped: the next frame starts a new one."""
        self._carry = b""
        self._pending_text = ""
        self._reanchor = True

    def end(self) -> None:
        self._ended = True
        self._ready.set()

    async def frames(self) -> AsyncIterator[tuple[int, rtc.AudioFrame, str | None]]:
        """Each frame at its playback time, with its position and the text it begins.

        The adapter anchors a reply's captions to the first text it receives, so a frame that
        resumes after a gap with no text of its own comes with an empty one: the new reply then
        anchors on it rather than on the next segment's words.
        """
        while True:
            if not self._queue:
                if self._ended:
                    return
                self._ready.clear()
                await self._ready.wait()
                continue
            if (delay := self._queue[0].start_ms - _PLAYOUT_LEAD_MS - self._now_ms()) > 0:
                await asyncio.sleep(delay / 1000)
                continue

            item = self._queue.popleft()
            text = item.text
            if text is None and (self._reanchor or item.start_ms > self._released_end_ms + 1):
                text = ""
            self._reanchor = False
            self._released_end_ms = item.start_ms + item.frame.duration * 1000
            yield round(item.start_ms), item.frame, text

    def _now_ms(self) -> float:
        return (time.monotonic() - self._t0) * 1000


def _joined(segments: list[dict[str, Any]] | None) -> str:
    return "".join(segment.get("text") or "" for segment in segments or [])


def _resolve_api_key(api_key: NotGivenOr[str]) -> str:
    key = (
        api_key
        if is_given(api_key)
        else os.environ.get("DEEPL_API_KEY") or os.environ.get("DEEPL_AUTH_KEY")
    )
    if not key:
        raise ValueError(
            "DeepL API key is required, either as argument or set the DEEPL_API_KEY "
            "environment variable"
        )
    return key


def _resolve_base_url(base_url: NotGivenOr[str], api_key: str) -> str:
    if is_given(base_url):
        return base_url
    if env_url := os.environ.get("DEEPL_BASE_URL"):
        return env_url
    # DeepL API Free keys end in ":fx"
    return DEEPL_FREE_API_URL if api_key.endswith(":fx") else DEEPL_API_URL
