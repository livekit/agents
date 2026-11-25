from __future__ import annotations

import asyncio
import os
import time
import weakref
from collections import deque
from dataclasses import dataclass
from typing import Any

import aiohttp
import msgpack

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
    LanguageCode,
    stt,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, is_given
from livekit.agents.voice.events import ConversationItemAddedEvent
from livekit.agents.voice.io import TimedString

from .log import logger
from .models import STTModels, TurnDetectionMode
from .tts import DEFAULT_BASE_URL, USER_AGENT

DEFAULT_STT_MODEL: STTModels = "transcribe-stream-1"

_SAMPLE_RATES = (8000, 12000, 16000, 22050, 24000, 32000, 44100, 48000)
_PROTOCOL_VERSION = 1
# Fish Audio recommends sending 20-40 ms of audio per message.
_CHUNK_MS = 20
# The server closes a session after 30 s without a client message.
_KEEPALIVE_INTERVAL = 10.0
_USAGE_REPORT_INTERVAL = 5.0
_MAX_TERMS = 100
_MAX_TERM_CHARS = 32
_MAX_HISTORY_ITEMS = 32
_MAX_HISTORY_CHARS = 2000
# The server commits the open turn before closing with either, so a new session continues.
_SERVER_RESTARTING_CLOSE_CODE = 1012
# the session reached its maximum duration (an hour)
_MAX_DURATION_CLOSE_CODE = 4413
_RECONNECT_CLOSE_CODES = (_SERVER_RESTARTING_CLOSE_CODE, _MAX_DURATION_CLOSE_CODE)
# the errors announcing those closes
_RECONNECT_ERROR_TYPES = ("server_restarting", "max_duration_exceeded")
_SILENCE_TIMEOUT_RANGE_MS: dict[TurnDetectionMode, tuple[int, int]] = {
    "auto": (3000, 60000),
    "vad": (300, 60000),
    "manual": (300, 60000),
}


@dataclass
class _STTOptions:
    model: STTModels | str
    sample_rate: int
    base_url: str
    turn_detection_mode: TurnDetectionMode
    eager_eot: bool
    silence_timeout_ms: NotGivenOr[int]
    interim_results: bool
    tag_audio_events: bool
    keyterms: list[str]

    def get_ws_url(self, path: str) -> str:
        return f"{self.base_url.replace('http', 'ws', 1)}{path}"


@dataclass
class _Turn:
    index: int
    start_ms: int
    user_turn: bool = False
    draft: str = ""


class STT(stt.STT):
    def __init__(
        self,
        *,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        model: STTModels | str = DEFAULT_STT_MODEL,
        sample_rate: int = 16000,
        turn_detection_mode: TurnDetectionMode = "auto",
        eager_eot: bool = True,
        silence_timeout_ms: NotGivenOr[int] = NOT_GIVEN,
        interim_results: bool = True,
        tag_audio_events: bool = False,
        keyterms: NotGivenOr[list[str]] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """
        Create a new instance of Fish Audio realtime STT.

        See https://docs.fish.audio/api-reference/endpoint/websocket/asr-realtime for more
        details on the Fish Audio realtime ASR WebSocket API.

        It works with the session's own VAD and turn detection, which end Fish Audio's turn
        when they detect that the user is done. To let Fish Audio decide instead, use
        ``AgentSession(turn_handling=TurnHandlingOptions(turn_detection="stt"))``.

        Fish Audio detects the language from the transcript; ``stream(language=...)`` has no
        effect.

        Args:
            api_key (NotGivenOr[str]): Fish Audio API key. Reads ``FISH_API_KEY`` if unset.
            model (STTModels | str): Realtime ASR model. Defaults to ``"transcribe-stream-1"``.
            sample_rate (int): Sample rate of the audio sent to Fish Audio, in Hz: 8000,
                12000, 16000, 22050, 24000, 32000, 44100 or 48000. Defaults to 16000.
            turn_detection_mode (TurnDetectionMode): How Fish Audio ends a turn. ``"auto"``
                judges when the user has finished and ignores turns that aren't the user's
                (noise, other speakers), though a flush from the session commits the open
                turn as the user's. ``"vad"`` ends a turn after ``silence_timeout_ms`` of
                silence. ``"manual"`` ends a turn only when the stream is flushed, which the
                session does when VAD or a turn detector ends the turn, or on
                ``commit_user_turn()``; use it for push-to-talk, and don't combine it with
                ``turn_detection="stt"``. Defaults to ``"auto"``.
            eager_eot (bool): Emit a preflight transcript when the turn has probably ended, so
                preemptive generation can start early. ``"auto"`` mode only. Defaults to True.
            silence_timeout_ms (NotGivenOr[int]): Silence that ends a turn. In ``"auto"``
                mode it is a safety net (3000-60000, server default 5000); in ``"vad"`` mode
                it is the endpointing delay (300-60000, server default 700); in ``"manual"``
                mode it is a safety net (300-60000, server default 60000).
            interim_results (bool): Whether to emit interim transcripts. Defaults to True.
            tag_audio_events (bool): Keep sound tags such as ``[laughter]`` in transcripts.
                Defaults to False.
            keyterms (NotGivenOr[list[str]]): Terms to bias recognition toward. Terms longer
                than 32 characters are dropped, and only the first 100 are kept.
            base_url (NotGivenOr[str]): Custom base URL. Defaults to ``https://api.fish.audio``.
            http_session (aiohttp.ClientSession | None): Optional aiohttp session to use.
        """
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=interim_results,
                aligned_transcript="word",
                offline_recognize=False,
                keyterms=True,
                chat_context=True,
                # the session flushes when its own VAD or turn detector ends the turn
                manual_flush=True,
            )
        )

        fish_api_key = api_key if is_given(api_key) else os.getenv("FISH_API_KEY")
        if not fish_api_key:
            raise ValueError(
                "Fish Audio API key is required, either as argument or set "
                "FISH_API_KEY environment variable"
            )
        self._api_key = fish_api_key

        if sample_rate not in _SAMPLE_RATES:
            raise ValueError(f"sample_rate must be one of {_SAMPLE_RATES}")
        if is_given(silence_timeout_ms):
            _validate_silence_timeout(turn_detection_mode, silence_timeout_ms)

        self._user_keyterms = list(keyterms) if is_given(keyterms) else []
        self._session_keyterms: list[str] = []
        self._opts = _STTOptions(
            model=model,
            sample_rate=sample_rate,
            base_url=base_url if is_given(base_url) else DEFAULT_BASE_URL,
            turn_detection_mode=turn_detection_mode,
            eager_eot=eager_eot,
            silence_timeout_ms=silence_timeout_ms,
            interim_results=interim_results,
            tag_audio_events=tag_audio_events,
            keyterms=_sanitize_terms(self._user_keyterms),
        )
        # the agent's recent replies, which seed the context of each new stream
        self._agent_history: deque[dict[str, str]] = deque(maxlen=_MAX_HISTORY_ITEMS)
        self._session = http_session
        self._streams = weakref.WeakSet[SpeechStream]()

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "FishAudio"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        raise NotImplementedError(
            "Fish Audio realtime STT does not support non-streaming recognize"
        )

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        stream = SpeechStream(
            stt=self,
            opts=self._opts,
            conn_options=conn_options,
            api_key=self._api_key,
            http_session=self._ensure_session(),
        )
        self._streams.add(stream)
        return stream

    def update_options(
        self,
        *,
        eager_eot: NotGivenOr[bool] = NOT_GIVEN,
        silence_timeout_ms: NotGivenOr[int] = NOT_GIVEN,
        tag_audio_events: NotGivenOr[bool] = NOT_GIVEN,
        keyterms: NotGivenOr[list[str]] = NOT_GIVEN,
    ) -> None:
        """Update options mid-session. Active streams apply them without reconnecting."""
        if is_given(silence_timeout_ms):
            _validate_silence_timeout(self._opts.turn_detection_mode, silence_timeout_ms)
            self._opts.silence_timeout_ms = silence_timeout_ms
        if is_given(eager_eot):
            self._opts.eager_eot = eager_eot
        if is_given(tag_audio_events):
            self._opts.tag_audio_events = tag_audio_events
        new_keyterms: NotGivenOr[list[str]] = NOT_GIVEN
        if is_given(keyterms):
            self._user_keyterms = list(keyterms)
            new_keyterms = self._merge_keyterms()

        for stream in self._streams:
            stream._update_options(
                eager_eot=eager_eot,
                silence_timeout_ms=silence_timeout_ms,
                tag_audio_events=tag_audio_events,
                keyterms=new_keyterms,
            )

    def _update_session_keyterms(self, keyterms: list[str]) -> None:
        if keyterms == self._session_keyterms:
            return
        self._session_keyterms = list(keyterms)
        new_keyterms = self._merge_keyterms()
        for stream in self._streams:
            stream._update_options(keyterms=new_keyterms)

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()

    def _merge_keyterms(self) -> list[str]:
        merged = _sanitize_terms(self._user_keyterms + self._session_keyterms)
        self._opts.keyterms = merged
        return merged

    def _push_conversation_item(self, ev: ConversationItemAddedEvent) -> None:
        # Fish Audio adds committed user turns to its context on its own;
        # the agent's replies help it judge whether the next speech is a user turn.
        item = ev.item
        if item.type != "message" or item.role != "assistant":
            return
        text = item.text_content
        if not text:
            return
        context_item = {"role": "assistant", "text": text[-_MAX_HISTORY_CHARS:]}
        self._agent_history.append(context_item)
        for stream in self._streams:
            stream._add_context(context_item)


class SpeechStream(stt.SpeechStream):
    def __init__(
        self,
        *,
        stt: STT,
        opts: _STTOptions,
        conn_options: APIConnectOptions,
        api_key: str,
        http_session: aiohttp.ClientSession,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)
        self._opts = opts
        self._api_key = api_key
        self._session = http_session
        # this conversation so far, sent as context when a session starts
        self._history: deque[dict[str, str]] = deque(stt._agent_history, maxlen=_MAX_HISTORY_ITEMS)
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._send_lock = asyncio.Lock()
        self._control_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._session_id: str | None = None
        self._request_id = ""
        self._language = LanguageCode("en")
        self._speaking = False
        self._turn: _Turn | None = None
        self._made_progress = False
        self._unreported_audio_duration = 0.0

    async def _run(self) -> None:
        while True:
            ws = await self._connect_ws()
            self._input_ch.restart_provider_clock()
            # the config carries every change made so far; later ones are queued
            while not self._control_queue.empty():
                self._control_queue.get_nowait()
            self._ws = ws
            self._made_progress = False
            try:
                try:
                    await self._send(ws, self._config())
                except (aiohttp.ClientError, ConnectionError):
                    raise APIConnectionError("fish audio stt connection lost") from None
                reconnect = await self._run_ws(ws)
            except Exception:
                # the connection failed mid-turn; end the turn with what was heard
                self._close_open_turn()
                raise
            finally:
                self._ws = None
                self._report_usage()
                await ws.close()

            if not reconnect:
                return

            # a no-op when the server committed the turn before closing, as it should
            self._close_open_turn()
            # a session that reached its maximum duration ran its course, even an idle one;
            # don't reopen at once a session restarted before it recognized anything
            if not self._made_progress and ws.close_code != _MAX_DURATION_CLOSE_CODE:
                raise APIConnectionError("fish audio stt session ended by the server")
            logger.debug(
                "fish audio stt session ended by the server, reconnecting",
                extra={"session_id": self._session_id, "close_code": ws.close_code},
            )

    async def _run_ws(self, ws: aiohttp.ClientWebSocketResponse) -> bool:
        """Stream audio over one session. Returns True when the server asks for a new one."""
        closing_ws = False
        self._turn = None

        @utils.log_exceptions(logger=logger)
        async def send_task() -> None:
            nonlocal closing_ws

            samples_per_chunk = self._opts.sample_rate * _CHUNK_MS // 1000
            audio_bstream = utils.audio.AudioByteStream(
                sample_rate=self._opts.sample_rate,
                num_channels=1,
                samples_per_channel=samples_per_chunk,
            )

            try:
                async for data in self._input_ch:
                    frames: list[rtc.AudioFrame] = []
                    if isinstance(data, rtc.AudioFrame):
                        frames.extend(audio_bstream.write(data.data.tobytes()))
                    else:
                        frames.extend(audio_bstream.flush())

                    for frame in frames:
                        await self._send(ws, {"type": "audio", "data": frame.data.tobytes()})
                        self._add_usage(frame.duration)

                    if isinstance(data, self._FlushSentinel):
                        await self._send(ws, {"type": "finalize"})

                # the server commits the open turn, then closes the session
                closing_ws = True
                await self._send(ws, {"type": "audio_done"})
            except (aiohttp.ClientError, ConnectionError):
                # the server closed the session; recv_task reports why
                return

        @utils.log_exceptions(logger=logger)
        async def control_task() -> None:
            try:
                while not closing_ws:
                    try:
                        msg = await asyncio.wait_for(
                            self._control_queue.get(), timeout=_KEEPALIVE_INTERVAL
                        )
                    except asyncio.TimeoutError:
                        msg = {"type": "keepalive"}
                    if closing_ws:
                        return
                    await self._send(ws, msg)
            except (aiohttp.ClientError, ConnectionError):
                return

        async def recv_task() -> bool:
            monitor = utils.aio.WebSocketMonitor(
                ws, liveness_timeout=self._conn_options.liveness_timeout
            )
            while True:
                msg = await monitor.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if closing_ws or self._session.closed:
                        return False

                    close_code = ws.close_code or -1
                    if close_code in _RECONNECT_CLOSE_CODES:
                        return True

                    raise APIStatusError(
                        message="fish audio stt connection closed unexpectedly",
                        status_code=_close_code_to_status(close_code),
                        request_id=self._request_id,
                        body=f"{msg.data=} {msg.extra=}",
                    )

                if msg.type == aiohttp.WSMsgType.ERROR:
                    if closing_ws or self._session.closed:
                        return False
                    raise APIConnectionError("fish audio stt connection lost") from ws.exception()

                if msg.type != aiohttp.WSMsgType.BINARY:
                    logger.warning("unexpected fish audio stt message type %s", msg.type)
                    continue

                try:
                    data = msgpack.unpackb(msg.data)
                    self._process_message(data)
                except APIError:
                    raise
                except Exception:
                    logger.exception("failed to process fish audio stt message")

        send = asyncio.create_task(send_task())
        recv = asyncio.create_task(recv_task())
        control = asyncio.create_task(control_task())
        try:
            done, _ = await asyncio.wait((send, recv, control), return_when=asyncio.FIRST_COMPLETED)
            # surface a failure to send; otherwise wait for the server to close the session
            if send in done:
                send.result()
            if control in done:
                control.result()
            return await recv
        finally:
            await utils.aio.gracefully_cancel(send, recv, control)

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        started_at = time.perf_counter()
        try:
            ws = await asyncio.wait_for(
                self._session.ws_connect(
                    self._opts.get_ws_url("/v1/asr/realtime"),
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "User-Agent": USER_AGENT,
                        "model": self._opts.model,
                    },
                    autoping=False,
                ),
                self._conn_options.timeout,
            )
        except asyncio.TimeoutError:
            raise APITimeoutError("timed out connecting to fish audio stt") from None
        except aiohttp.ClientResponseError as e:
            # 401: bad key, 402: no credit, 429: concurrency limit, 503: restarting.
            # RequestInfo carries the request headers, so don't chain the error (API key)
            request_id = None
            if e.headers is not None:
                request_id = e.headers.get("x-request-id")
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=request_id, body=None
            ) from None
        except Exception as e:
            raise APIConnectionError(
                f"failed to connect to fish audio stt ({type(e).__name__})"
            ) from None

        self._report_connection_acquired(time.perf_counter() - started_at, False)
        return ws

    def _config(self) -> dict[str, Any]:
        turn_detection: dict[str, Any] = {"mode": self._opts.turn_detection_mode}
        if self._opts.turn_detection_mode == "auto":
            turn_detection["eager"] = self._opts.eager_eot
        if is_given(self._opts.silence_timeout_ms):
            turn_detection["silence_timeout_ms"] = self._opts.silence_timeout_ms

        context: dict[str, Any] = {}
        if self._history:
            context["history"] = list(self._history)
        if self._opts.keyterms:
            context["terms"] = self._opts.keyterms

        config: dict[str, Any] = {
            "type": "config",
            "protocol_version": _PROTOCOL_VERSION,
            "audio": {
                "format": "pcm_s16le",
                "sample_rate": self._opts.sample_rate,
                "channels": 1,
            },
            "turn_detection": turn_detection,
            "results": {
                "drafts": self._opts.interim_results,
                "ignore_timestamps": False,
                "diarize": False,
                "tag_audio_events": self._opts.tag_audio_events,
            },
        }
        if context:
            config["context"] = context
        return config

    async def _send(self, ws: aiohttp.ClientWebSocketResponse, msg: dict[str, Any]) -> None:
        payload = msgpack.packb(msg)
        async with self._send_lock:
            await ws.send_bytes(payload)

    def _send_control(self, msg: dict[str, Any]) -> None:
        # while disconnected, the next connection's config carries the change
        if self._ws is None:
            return
        self._control_queue.put_nowait(msg)

    def _add_context(self, item: dict[str, str]) -> None:
        self._history.append(item)
        self._send_control({"type": "context_add", "items": [item]})

    def _update_options(
        self,
        *,
        eager_eot: NotGivenOr[bool] = NOT_GIVEN,
        silence_timeout_ms: NotGivenOr[int] = NOT_GIVEN,
        tag_audio_events: NotGivenOr[bool] = NOT_GIVEN,
        keyterms: NotGivenOr[list[str]] = NOT_GIVEN,
    ) -> None:
        """Send changed options over the live connection; omitted fields keep their values."""
        turn_detection: dict[str, Any] = {}
        if is_given(eager_eot) and self._opts.turn_detection_mode == "auto":
            turn_detection["eager"] = eager_eot
        if is_given(silence_timeout_ms):
            turn_detection["silence_timeout_ms"] = silence_timeout_ms

        results: dict[str, Any] = {}
        if is_given(tag_audio_events):
            results["tag_audio_events"] = tag_audio_events

        msg: dict[str, Any] = {"type": "configure"}
        if turn_detection:
            msg["turn_detection"] = turn_detection
        if results:
            msg["results"] = results
        if is_given(keyterms):
            # replaces the whole list
            msg["context"] = {"terms": keyterms}

        if len(msg) > 1:
            self._send_control(msg)

    def _add_usage(self, duration: float) -> None:
        self._unreported_audio_duration += duration
        if self._unreported_audio_duration >= _USAGE_REPORT_INTERVAL:
            self._report_usage()

    def _report_usage(self) -> None:
        if self._unreported_audio_duration <= 0:
            return
        usage_event = stt.SpeechEvent(
            type=stt.SpeechEventType.RECOGNITION_USAGE,
            request_id=self._request_id,
            alternatives=[],
            recognition_usage=stt.RecognitionUsage(audio_duration=self._unreported_audio_duration),
        )
        self._event_ch.send_nowait(usage_event)
        self._unreported_audio_duration = 0.0

    def _process_message(self, msg: dict[str, Any]) -> None:
        msg_type = msg.get("type")

        if msg_type == "session_started":
            self._session_id = msg.get("session_id")
            self._request_id = msg.get("request_id", "")
            logger.debug(
                "fish audio stt session started",
                extra={"session_id": self._session_id, "request_id": self._request_id},
            )
            return

        if msg_type == "configured":
            if not msg.get("ok"):
                logger.warning("fish audio stt rejected an options update: %s", msg.get("message"))
            return

        if msg_type == "error":
            self._on_error(msg)
            return

        if msg_type == "speech_start":
            self._on_speech_start(msg)
        elif msg_type == "user_turn_taken":
            self._on_user_turn_taken(msg)
        elif msg_type == "draft":
            self._on_draft(msg)
        elif msg_type == "eager_end_of_turn":
            self._on_eager_end_of_turn(msg)
        elif msg_type == "committed":
            self._on_committed(msg)
        else:
            # turn_resumed, progress, session_done, vad, ... need no handling: after a
            # turn_resumed, the next draft stops the preemptive reply
            return

        self._made_progress = True
        self._mark_provider_progress()

    def _current_turn(self, msg: dict[str, Any]) -> _Turn | None:
        turn = self._turn
        if turn is None or turn.index != msg.get("turn_index"):
            return None
        return turn

    def _on_speech_start(self, msg: dict[str, Any]) -> None:
        turn = _Turn(index=msg.get("turn_index", 0), start_ms=msg.get("audio_ms", 0))
        self._turn = turn
        # in "auto" mode, speech may be noise or another speaker until user_turn_taken
        if self._opts.turn_detection_mode != "auto":
            self._start_user_turn(turn)

    def _on_user_turn_taken(self, msg: dict[str, Any]) -> None:
        turn = self._current_turn(msg)
        if turn is None:
            turn = _Turn(index=msg.get("turn_index", 0), start_ms=msg.get("audio_ms", 0))
            self._turn = turn
        self._start_user_turn(turn)

    def _start_user_turn(self, turn: _Turn) -> None:
        turn.user_turn = True
        self._start_speech(turn.start_ms)
        if turn.draft:
            self._send_transcript(stt.SpeechEventType.INTERIM_TRANSCRIPT, turn.draft, None)

    def _start_speech(self, audio_ms: int) -> None:
        if self._speaking:
            return
        self._speaking = True
        start_event = stt.SpeechEvent(
            type=stt.SpeechEventType.START_OF_SPEECH,
            request_id=self._request_id,
            speech_start_time=self._wall_time(audio_ms),
        )
        self._event_ch.send_nowait(start_event)

    def _on_draft(self, msg: dict[str, Any]) -> None:
        turn = self._current_turn(msg)
        if turn is None:
            return
        turn.draft = (msg.get("text") or "").strip()
        if turn.user_turn and turn.draft:
            self._send_transcript(stt.SpeechEventType.INTERIM_TRANSCRIPT, turn.draft, msg)

    def _on_eager_end_of_turn(self, msg: dict[str, Any]) -> None:
        turn = self._current_turn(msg)
        if turn is None or not turn.user_turn:
            return
        text = (msg.get("text") or "").strip()
        if text:
            self._send_transcript(stt.SpeechEventType.PREFLIGHT_TRANSCRIPT, text, msg)

    def _on_committed(self, msg: dict[str, Any]) -> None:
        if self._current_turn(msg) is not None:
            self._turn = None

        # a missing value counts as a user turn
        user_turn = msg.get("user_turn", True)
        if not user_turn:
            logger.debug(
                "ignoring a turn that isn't the user's",
                extra={"turn_kind": msg.get("turn_kind"), "session_id": self._session_id},
            )
            self._end_speech([])
            return

        self._start_speech(msg.get("audio_start_ms", 0))

        # every user turn ends with a final transcript, even an empty one: the session
        # waits for it after flushing the stream
        text = (msg.get("text") or "").strip()
        speech_data = self._speech_data(text, msg)
        final_event = stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            request_id=self._request_id,
            alternatives=[speech_data],
        )
        self._event_ch.send_nowait(final_event)
        if text:
            self._history.append({"role": "user", "text": text[-_MAX_HISTORY_CHARS:]})

        # word timings carry the end of speech; the commit itself lands after the silence
        end_alternatives: list[stt.SpeechData] = []
        if speech_data.words and speech_data.end_time > 0:
            end_alternatives.append(speech_data)
        self._end_speech(end_alternatives)

    def _end_speech(self, alternatives: list[stt.SpeechData]) -> None:
        if not self._speaking:
            return
        self._speaking = False
        end_event = stt.SpeechEvent(
            type=stt.SpeechEventType.END_OF_SPEECH,
            request_id=self._request_id,
            alternatives=alternatives,
        )
        self._event_ch.send_nowait(end_event)

    def _close_open_turn(self) -> None:
        """End a user turn the server never committed, with the text heard so far."""
        turn = self._turn
        self._turn = None
        if not self._speaking:
            return
        if turn is not None and turn.user_turn and turn.draft:
            self._send_transcript(stt.SpeechEventType.FINAL_TRANSCRIPT, turn.draft, None)
        self._end_speech([])

    def _on_error(self, msg: dict[str, Any]) -> None:
        error_type = msg.get("error_type")
        message = msg.get("message") or "unknown error from fish audio stt"
        if msg.get("recoverable") or error_type in _RECONNECT_ERROR_TYPES:
            # the session continues, or closes with a code that starts a new one
            logger.warning(
                "fish audio stt error: %s",
                message,
                extra={"error_type": error_type, "request_id": msg.get("request_id")},
            )
            return

        raise APIStatusError(
            message=message,
            status_code=msg.get("code", -1),
            request_id=msg.get("request_id"),
            body=error_type,
        )

    def _send_transcript(
        self, event_type: stt.SpeechEventType, text: str, msg: dict[str, Any] | None
    ) -> None:
        event = stt.SpeechEvent(
            type=event_type,
            request_id=self._request_id,
            alternatives=[self._speech_data(text, msg)],
        )
        self._event_ch.send_nowait(event)

    def _speech_data(self, text: str, msg: dict[str, Any] | None) -> stt.SpeechData:
        """Build the transcript of a server message, or of locally held text when None."""
        if msg is None:
            msg = {}
        # the language stays as last detected when a message is too short to tell
        language_code = msg.get("language_code")
        if language_code:
            self._language = LanguageCode(language_code)

        offset = self.start_time_offset
        raw_words = msg.get("words") or []
        words: list[TimedString] = []
        for raw_word in raw_words:
            word = TimedString(
                text=raw_word.get("text", ""),
                start_time=raw_word.get("start_ms", 0) / 1000 + offset,
                end_time=raw_word.get("end_ms", 0) / 1000 + offset,
                start_time_offset=offset,
            )
            words.append(word)

        start_time = 0.0
        end_time = 0.0
        if raw_words:
            start_time = raw_words[0].get("start_ms", 0) / 1000 + offset
            end_time = raw_words[-1].get("end_ms", 0) / 1000 + offset
        elif "audio_end_ms" in msg:
            # degraded commit: no word timings
            start_time = msg.get("audio_start_ms", 0) / 1000 + offset
            end_time = msg["audio_end_ms"] / 1000 + offset

        return stt.SpeechData(
            language=self._language,
            text=text,
            start_time=start_time,
            end_time=end_time,
            words=words or None,
        )

    def _wall_time(self, audio_ms: int) -> float | None:
        """Wall-clock time of a position in this connection's audio."""
        input_started_at = self._input_ch.input_started_at
        if input_started_at is None:
            return None
        return input_started_at + self.start_time_offset + audio_ms / 1000


def _validate_silence_timeout(
    turn_detection_mode: TurnDetectionMode, silence_timeout_ms: int
) -> None:
    allowed_range = _SILENCE_TIMEOUT_RANGE_MS[turn_detection_mode]
    low = allowed_range[0]
    high = allowed_range[1]
    if not low <= silence_timeout_ms <= high:
        raise ValueError(
            f"silence_timeout_ms must be between {low} and {high} "
            f"in {turn_detection_mode!r} turn detection mode"
        )


def _sanitize_terms(terms: list[str]) -> list[str]:
    """Drop terms Fish Audio would reject, since an invalid config ends the session."""
    sanitized: list[str] = []
    seen: set[str] = set()
    too_long = 0
    for raw_term in terms:
        term = raw_term.strip()
        if not term or term in seen:
            continue
        if len(term) > _MAX_TERM_CHARS:
            too_long += 1
            continue
        seen.add(term)
        sanitized.append(term)

    if too_long:
        logger.warning("dropped %d keyterm(s) longer than %d characters", too_long, _MAX_TERM_CHARS)
    if len(sanitized) > _MAX_TERMS:
        logger.warning("keeping the first %d of %d keyterms", _MAX_TERMS, len(sanitized))
        sanitized = sanitized[:_MAX_TERMS]
    return sanitized


def _close_code_to_status(close_code: int) -> int:
    # Fish Audio closes with 4000 + an HTTP-style status (4400, 4408, 4500)
    if 4000 <= close_code < 5000:
        return close_code - 4000
    return -1
