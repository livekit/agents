from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import time
from collections.abc import AsyncIterable
from dataclasses import dataclass, field, replace
from typing import Any, Literal, TypedDict
from urllib.parse import urlparse, urlunparse

import aiohttp

from livekit import rtc
from livekit.agents import APIConnectionError, APIError, llm, utils
from livekit.agents.metrics import LLMMetrics, RealtimeModelMetrics
from livekit.agents.metrics.base import Metadata
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import is_given
from openai.types.responses import ResponseTextConfigParam
from openai.types.responses.response_input_item import FunctionCallOutput
from openai.types.shared_params import Reasoning

from ..log import logger
from ..tools import OpenAITool
from . import gpt_live_types as types

SAMPLE_RATE = 24000
NUM_CHANNELS = 1
DEFAULT_MODEL = "gpt-live-1"
DEFAULT_VOICE = "marin"
DEFAULT_BACKEND_MODEL = "gpt-5.6-luna"
OPENAI_BASE_URL = "https://api.openai.com/v1"

# the service also caps startup history at 8192 tokens and an append at 500; there is no tokenizer
# here, so those two are the service's to enforce
_MAX_INPUT_ITEMS = 128

# measured on the alpha: silence is all zeros bar a 0.4 s tick at connect, and speech starts an
# order of magnitude above, so nothing has to be learned from the stream
_SILENCE_RMS = 0.0006
# long enough that a pause between sentences does not split an utterance in two
_MIN_SILENCE_DURATION = 0.8
_MIN_SILENCE_MS = _MIN_SILENCE_DURATION * 1000

# the asks generate_reply sends as commentary. each ends with the same two sentences, which make
# the model speak now rather than wait for the caller; what precedes them says what to speak
# about: an instruction to follow, a typed message to answer, or nothing beyond replying
_SPEAK_NOW = "Do not wait for the caller to speak first. After that, pause and listen."
_ASK_INSTRUCTED = f"Immediately follow the instruction below. {_SPEAK_NOW}"
_ASK_TYPED = f"Reply to the caller now, don't repeat what they said. {_SPEAK_NOW}"
_ASK_BARE = f"Reply to the caller now. {_SPEAK_NOW}"

# session.closed carries the final usage; the service drains first
_SESSION_CLOSE_TIMEOUT = 5.0
_CLOSING_EVENTS = frozenset({"session.usage.updated", "session.closed"})
_FATAL_ERROR_CODES = frozenset(
    {
        "insufficient_quota",
        "invalid_api_key",
        "account_deactivated",
        "billing_hard_limit_reached",
    }
)

Role = Literal["user", "assistant"]
GPTLiveVoices = Literal["aster", "beacon", "cinder", "marin", "stone", "vesper"]

lk_oai_debug = int(os.getenv("LK_OPENAI_DEBUG", 0))


class ResponsesDelegationOptions(TypedDict, total=False):
    """The backend Responses model delegated work runs on, under ``delegation="responses"``.

    A key left unset is not sent, and the service's own default applies.
    """

    model: str
    """Responses model slug; ``gpt-5.6-luna`` when unset."""
    instructions: str
    """Instructions for the backend model, distinct from the voice model's."""
    tool_choice: llm.ToolChoice | None
    parallel_tool_calls: bool
    reasoning: Reasoning
    """Responses reasoning settings, for example ``{"effort": "medium"}``."""
    text: ResponseTextConfigParam
    """Responses text settings, for example ``{"verbosity": "low"}``."""
    service_tier: Literal["auto", "default", "flex", "priority"]
    max_output_tokens: int
    """Upper bound on the tokens one backend response may generate; at least 16."""


@dataclass
class GPTLiveDelegation:
    """Work the model handed to the application, under client delegation.

    It carries no task text: the ask is whatever the conversation says, which the agent's chat
    context holds. Answer it with :meth:`GPTLiveSession.append_commentary`, passing this id.
    """

    id: str
    pending_transcript: str
    """The caller's current turn, not yet in the chat context when the model delegates."""


@dataclass
class _Speech:
    """One speaker's fragments so far: the message they grow into and, for the user, their turn."""

    message_id: str
    text: str = ""
    end_ms: int | None = None
    started_at: float = field(default_factory=time.time)
    quiet_ms: int = 0
    """Input audio pushed since the user's last fragment."""


# Responses delegation hands work to a backend Responses model, whose events arrive wrapped in
# response.event. One backend run is a "response":
#
#   response.created ─► response.output_item.done ×N (function calls) ─► response.completed
#
# A response completes with its calls unanswered. Each result is queued with response.item.create,
# and response.create runs the next response, the continuation, once every call has one; a partial
# batch is rejected. The voice model speaks the continuation's text on its own.
@dataclass
class _DelegatedResponse:
    """The current response of one delegation, and the tool calls it waits on."""

    call_ids: set[str] = field(default_factory=set)
    returned: set[str] = field(default_factory=set)
    completed: bool = False


@dataclass
class _LiveOptions:
    model: str
    voice: str | dict[str, Any]
    delegation: types.DelegationTarget
    responses: ResponsesDelegationOptions
    api_key: str
    base_url: str
    conn_options: APIConnectOptions
    max_session_duration: float | None


class GPTLiveModel(llm.DuplexModel):
    """OpenAI GPT-Live full-duplex voice model, ready to pass to ``AgentSession(llm=)``."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        voice: GPTLiveVoices | str | dict[str, Any] = DEFAULT_VOICE,
        delegation: types.DelegationTarget = "responses",
        responses_options: NotGivenOr[ResponsesDelegationOptions] = NOT_GIVEN,
        api_key: str | None = None,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
        max_session_duration: NotGivenOr[float | None] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        """
        Args:
            model: GPT-Live voice model slug.
            voice: Output voice: a name from :data:`GPTLiveVoices`, another supported name, or
                ``{"id": "voice_..."}`` for an authorized custom voice. Defaults to ``marin``.
                Immutable after the session starts.
            delegation: Where delegated work goes, fixed for the life of the session.
                ``responses`` runs it on a backend model, so ``@function_tool`` works as usual;
                ``client`` hands it to the application as a ``delegation_created`` event, which
                no framework tool can answer.
            responses_options: The backend Responses model, under ``delegation="responses"``.
            api_key: OpenAI API key. Falls back to ``OPENAI_API_KEY``.
            base_url: HTTP base url of the OpenAI API.
            http_session: Optional shared HTTP session.
            max_session_duration: Seconds before the connection is recycled.
            conn_options: Retry/backoff and connection settings.
        """
        super().__init__(
            capabilities=llm.DuplexCapabilities(
                user_transcription=True,
                # the model continues on its own once every tool result reaches the backend
                auto_tool_reply_generation=True,
                mutable_chat_context=False,
                mutable_instructions=False,
                # tools live on the backend model, and a client delegation has none
                mutable_tools=delegation == "responses",
            )
        )
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError(
                "The api_key client option must be set either by passing api_key "
                "to the client or by setting the OPENAI_API_KEY environment variable"
            )

        self._opts = _LiveOptions(
            model=model,
            voice=voice,
            delegation=delegation,
            responses=(
                responses_options if is_given(responses_options) else ResponsesDelegationOptions()
            ),
            api_key=api_key,
            base_url=base_url
            if is_given(base_url)
            else os.getenv("OPENAI_BASE_URL", OPENAI_BASE_URL),
            conn_options=conn_options,
            max_session_duration=max_session_duration if is_given(max_session_duration) else None,
        )
        self._http_session = http_session
        self._http_session_owned = False
        self._provider_label = "OpenAI Live API"

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return urlparse(self._opts.base_url).netloc

    def _ensure_http_session(self) -> aiohttp.ClientSession:
        if not self._http_session:
            try:
                self._http_session = utils.http_context.http_session()
            except RuntimeError:
                self._http_session = aiohttp.ClientSession()
                self._http_session_owned = True
        return self._http_session

    def audio_gate(self) -> llm.AudioGate:
        return llm.FixedGate(_SILENCE_RMS, min_silence_duration=_MIN_SILENCE_DURATION)

    def session(self) -> GPTLiveSession:
        return GPTLiveSession(self)

    async def aclose(self) -> None:
        if self._http_session_owned and self._http_session:
            await self._http_session.close()


class GPTLiveSession(
    llm.DuplexSession[
        Literal["openai_server_event_received", "openai_client_event_queued", "delegation_created"]
    ]
):
    """A session for the OpenAI GPT-Live API (WebSocket), reached with ``Agent.duplex_session``.

    Exposes three extra events:
    - openai_server_event_received: raw server events
    - openai_client_event_queued: raw client events sent to the server
    - delegation_created: a :class:`GPTLiveDelegation`, under client delegation

    The ``append_*`` methods queue context and return without waiting. Their ``*.appended``
    events arrive at the estimated context-injection end; they do not mean speech has finished.
    """

    def __init__(self, duplex_model: GPTLiveModel) -> None:
        super().__init__(duplex_model)
        self._live_model = duplex_model
        self._opts = replace(duplex_model._opts, responses=duplex_model._opts.responses.copy())
        self._tools = llm.ToolContext.empty()
        # the agent's instructions, set by _update_session before session.start and immutable after
        self._instructions: str | None = None
        self._msg_ch = utils.aio.Chan[types.ClientEvent | dict[str, Any]]()
        self._audio_ch = utils.aio.Chan[llm.DuplexAudioFrame]()
        self._input_resampler: rtc.AudioResampler | None = None

        # session.start opens a connection and carries the config that is immutable after it
        self._session_start_sent = False
        self._session_started_fut: asyncio.Future[None] = asyncio.Future()
        self._session_closed_fut: asyncio.Future[None] = asyncio.Future()
        self._session_id: str | None = None
        self._num_retries = 0
        # session usage is reported cumulatively; kept to emit per-event deltas
        self._usage_total = types.Usage()

        # everything the model has been told, both speakers' words included, to reseed a
        # reconnect; the framework's chat context is the adapter's, not this
        self._history = llm.ChatContext.empty()
        self._speech: dict[Role, _Speech] = {}

        # the current response per delegation and the delegation each tool call belongs to,
        # since the framework hands a result back by call id alone
        self._delegated_responses: dict[str | None, _DelegatedResponse] = {}
        self._fnc_call_to_delegation: dict[str, str | None] = {}

        # the newest history item the last ask was about, so an ask never repeats one
        self._asked_item_id: str | None = None

        self._bstream = utils.audio.AudioByteStream(
            SAMPLE_RATE, NUM_CHANNELS, samples_per_channel=SAMPLE_RATE // 10
        )

        self._main_atask = asyncio.create_task(self._main_task(), name="GPTLiveSession._main")

    # outbound

    def send_event(self, event: types.ClientEvent | dict[str, Any]) -> None:
        with contextlib.suppress(utils.aio.channel.ChanClosed):
            self._msg_ch.send_nowait(event)

    def _build_delegation(self) -> types.Delegation:
        if self._opts.delegation == "client":
            return types.Delegation(type="client")
        opts = self._opts.responses
        return types.Delegation(
            type="responses",
            responses=types.ResponsesConfig(
                model=opts.get("model", DEFAULT_BACKEND_MODEL),
                instructions=opts.get("instructions"),
                tools=_build_delegation_tools(self._tools.flatten()) or None,
                tool_choice=_to_tool_choice(opts["tool_choice"]) if "tool_choice" in opts else None,
                parallel_tool_calls=opts.get("parallel_tool_calls"),
                reasoning=opts.get("reasoning"),
                text=opts.get("text"),
                service_tier=opts.get("service_tier"),
                max_output_tokens=opts.get("max_output_tokens"),
            ),
        )

    def _session_start_event(self) -> types.SessionStartEvent:
        """The whole configuration, composed fresh for each connection."""
        # the conversation so far is startup history, newest first until the cap is reached
        items: list[types.InputItem] = []
        dropped = 0
        for item in reversed(self._history.items):
            if (rendered := _render_item(item)) is None:
                continue
            role, text = rendered
            if len(items) >= _MAX_INPUT_ITEMS:
                dropped += 1
                continue
            part = (
                types.OutputTextPart(text=text)
                if role == "assistant"
                else types.InputTextPart(text=text)
            )
            items.append(types.InputItem(role=role, content=[part]))
        if dropped:
            logger.warning(
                "gpt-live startup history exceeds what a session accepts; dropping the oldest",
                extra={"dropped": dropped, "kept": len(items)},
            )
        items.reverse()

        return types.SessionStartEvent(
            event_id=utils.shortuuid("session_start_"),
            session=types.SessionConfig(
                model=self._opts.model,
                instructions=self._instructions,
                input=items or None,
                audio=types.AudioConfig(
                    format=types.AudioFormat(type="audio/pcm", rate=SAMPLE_RATE),
                    output=types.AudioOutput(voice=self._opts.voice),
                ),
                delegation=self._build_delegation(),
            ),
        )

    def _send_delegation_update(self, responses: types.ResponsesConfig) -> None:
        """A sparse session.update: only the backend settings can change once started."""
        if self._opts.delegation != "responses" or not self._session_start_sent:
            return
        self.send_event(
            types.SessionUpdateEvent(
                event_id=utils.shortuuid("delegation_update_"),
                session=types.SessionUpdateConfig(
                    delegation=types.Delegation(type="responses", responses=responses)
                ),
            )
        )

    # connection loop

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        max_retries = self._opts.conn_options.max_retry
        reconnecting = False

        try:
            while not self._msg_ch.closed:
                try:
                    ws_conn = await self._create_ws_conn()
                    if reconnecting:
                        self._reset_for_reconnect()
                        self.emit("session_reconnected", llm.RealtimeSessionReconnectedEvent())
                    try:
                        await self._run_ws(ws_conn)
                    finally:
                        # what arrives now is history for the next connection, not an append
                        self._session_start_sent = False
                except APIError as e:
                    if max_retries == 0 or not e.retryable:
                        self._emit_error(e, recoverable=False)
                        raise
                    elif self._num_retries == max_retries:
                        self._emit_error(e, recoverable=False)
                        raise APIConnectionError(
                            f"{self._live_model._provider_label} connection failed after "
                            f"{self._num_retries} attempts",
                        ) from e
                    else:
                        self._emit_error(e, recoverable=True)
                        interval = self._opts.conn_options._interval_for_retry(self._num_retries)
                        logger.warning(
                            f"{self._live_model._provider_label} connection failed, "
                            f"retrying in {interval}s",
                            exc_info=e,
                        )
                        await asyncio.sleep(interval)
                    self._num_retries += 1
                except Exception as e:
                    logger.error("gpt-live session failed", extra={"error_type": type(e).__name__})
                    error = APIConnectionError("GPT-Live session failed", retryable=False)
                    self._emit_error(error, recoverable=False)
                    raise error from None
                reconnecting = True
        finally:
            self._audio_ch.close()

    def _reset_for_reconnect(self) -> None:
        # a new connection is a new session, reseeded from the history; the rest of what the
        # dropped one was carrying never arrives
        self._bstream.clear()
        self._input_resampler = None
        self._session_started_fut = asyncio.Future()
        self._session_closed_fut = asyncio.Future()
        self._end_speech("user")
        self._speech.clear()
        self._delegated_responses.clear()
        self._fnc_call_to_delegation.clear()
        self._usage_total = types.Usage()
        self._session_id = None

    async def _create_ws_conn(self) -> aiohttp.ClientWebSocketResponse:
        headers = {
            "User-Agent": "LiveKit Agents",
            "Authorization": f"Bearer {self._opts.api_key}",
        }
        parsed = urlparse(self._opts.base_url.replace("http", "ws", 1))
        path = parsed.path.rstrip("/")
        if not path.endswith("/live/sessions"):
            path = f"{path}/live/sessions"
        url = urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))
        if lk_oai_debug:
            logger.debug("connecting to GPT-Live API", extra={"lk.pii.url": url})

        t0 = time.perf_counter()
        try:
            ws = await asyncio.wait_for(
                self._live_model._ensure_http_session().ws_connect(url=url, headers=headers),
                self._opts.conn_options.timeout,
            )
            self._report_connection_acquired(time.perf_counter() - t0)
            return ws
        except (aiohttp.ClientError, asyncio.TimeoutError):
            raise APIConnectionError(
                f"{self._live_model._provider_label} connection error"
            ) from None

    async def _run_ws(self, ws_conn: aiohttp.ClientWebSocketResponse) -> None:
        closing = False

        async def _close_ws() -> None:
            nonlocal closing
            if not closing:
                closing = True
                if self._session_start_sent:
                    with contextlib.suppress(Exception):
                        await self._ws_send(ws_conn, types.SessionCloseEvent())
            if self._session_started_fut.done() and not self._session_started_fut.cancelled():
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(
                        asyncio.shield(self._session_closed_fut), _SESSION_CLOSE_TIMEOUT
                    )
            await ws_conn.close()

        async def _send_task() -> None:
            nonlocal closing
            # instructions, voice and history are immutable once the session starts
            await self._configured.wait()
            if self._closing:
                # closed before the configuration landed; there is no session to start or drain
                closing = True
                await ws_conn.close()
                return
            start = self._session_start_event()
            self._session_start_sent = True
            await self._ws_send(ws_conn, start)

            async for msg in self._msg_ch:
                # the protocol asks for session.started before any audio or command goes out
                if not self._session_started_fut.done():
                    await self._session_started_fut
                await self._ws_send(ws_conn, msg)

            await _close_ws()

        async def _recv_task() -> None:
            while True:
                try:
                    msg = await ws_conn.receive()
                except (aiohttp.ClientError, ConnectionError, asyncio.TimeoutError):
                    raise APIConnectionError("GPT-Live receive failed") from None
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if closing or self._session_closed_fut.done():
                        return
                    raise APIConnectionError(
                        f"{self._live_model._provider_label} connection closed unexpectedly"
                    )
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue

                event = json.loads(msg.data)
                self.emit("openai_server_event_received", event)
                # while closing only the shutdown events matter; the framework has torn down
                if self._closing and event.get("type") not in _CLOSING_EVENTS:
                    continue
                try:
                    self._handle_event(event)
                except Exception as e:
                    if isinstance(e, APIError) and not e.retryable:
                        raise
                    logger.warning(
                        "failed to handle gpt-live event",
                        extra={"type": event.get("type"), "error_type": type(e).__name__},
                    )

        send_task = asyncio.create_task(_send_task(), name="_send_task")
        tasks = [asyncio.create_task(_recv_task(), name="_recv_task"), send_task]
        wait_reconnect_task: asyncio.Task | None = None
        if self._opts.max_session_duration is not None:
            wait_reconnect_task = asyncio.create_task(
                asyncio.sleep(self._opts.max_session_duration), name="_timeout_task"
            )
            tasks.append(wait_reconnect_task)
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                if task != wait_reconnect_task:
                    task.result()
            if wait_reconnect_task in done:
                await utils.aio.cancel_and_wait(send_task)
                await _close_ws()
        finally:
            await utils.aio.cancel_and_wait(*tasks)
            await ws_conn.close()

    async def _ws_send(
        self, ws_conn: aiohttp.ClientWebSocketResponse, event: types.ClientEvent | dict[str, Any]
    ) -> None:
        raw = event if isinstance(event, dict) else event.model_dump(exclude_none=True)
        self.emit("openai_client_event_queued", raw)
        if lk_oai_debug and raw.get("type") != "session.input_audio.append":
            logger.debug("gpt-live client event", extra={"lk.pii.event": raw})
        try:
            await ws_conn.send_str(json.dumps(raw))
        except (aiohttp.ClientError, ConnectionError, asyncio.TimeoutError):
            raise APIConnectionError("GPT-Live send failed") from None

    # inbound events

    def _handle_event(self, event: dict[str, Any]) -> None:
        etype = event.get("type", "")
        if lk_oai_debug and etype != "session.output_audio.delta":
            logger.debug("gpt-live server event", extra={"lk.pii.event": event})

        if etype == "session.started":
            self._handle_session_started(types.SessionStartedEvent.construct(**event))
        elif etype == "session.output_audio.delta":
            self._handle_output_audio_delta(types.OutputAudioDeltaEvent.construct(**event))
        elif etype == "session.output_transcript.delta":
            self._handle_transcript_delta(
                "assistant", types.TranscriptDeltaEvent.construct(**event)
            )
        elif etype == "session.input_transcript.delta":
            self._handle_transcript_delta("user", types.TranscriptDeltaEvent.construct(**event))
        elif etype == "session.delegation.created":
            self._handle_delegation_created(types.SessionDelegationCreatedEvent.construct(**event))
        elif etype == "response.event":
            self._handle_response_event(types.ResponseEventEnvelope.construct(**event))
        elif etype == "session.usage.updated":
            self._handle_session_usage_updated(types.SessionUsageUpdatedEvent.construct(**event))
        elif etype == "session.closed":
            self._handle_session_closed(types.SessionClosedEvent.construct(**event))
        elif etype == "error":
            self._handle_error(types.ErrorEvent.construct(**event).error)
        elif etype in (
            "session.updated",
            "session.input_audio.muted",
            "session.input_audio.unmuted",
            "session.instructions.appended",
            "session.thinking.appended",
            "session.commentary.appended",
        ):
            # Context append receipts arrive at the estimated injection end, not speech end.
            # Nothing waits on these acknowledgments.
            logger.debug(
                "gpt-live acknowledged a command",
                extra={"type": etype, "client_event_id": event.get("client_event_id")},
            )
        elif lk_oai_debug:
            logger.debug("unhandled gpt-live event", extra={"lk.pii.type": etype})

    def _handle_session_started(self, event: types.SessionStartedEvent) -> None:
        self._session_id = event.session.id or self._session_id
        self._num_retries = 0
        if not self._session_started_fut.done():
            self._session_started_fut.set_result(None)

    def _handle_output_audio_delta(self, event: types.OutputAudioDeltaEvent) -> None:
        # every frame is published, silence included: the framework decides what plays
        if not (data := base64.b64decode(event.delta or "")):
            return
        frame = rtc.AudioFrame(
            data=data,
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
            samples_per_channel=len(data) // 2,
        )
        with contextlib.suppress(utils.aio.channel.ChanClosed):
            self._audio_ch.send_nowait(llm.DuplexAudioFrame(frame=frame))

    def _handle_transcript_delta(self, role: Role, event: types.TranscriptDeltaEvent) -> None:
        if not event.delta:
            return
        speech = self._speech.get(role)
        # a pause on the model's clock ends the message even when its fragments arrived together
        if (
            speech is not None
            and speech.end_ms is not None
            and event.start_ms is not None
            and event.start_ms - speech.end_ms > _MIN_SILENCE_MS
        ):
            self._end_speech(role)
            speech = None
        if speech is None:
            speech = self._speech[role] = _Speech(message_id=utils.shortuuid("speech_"))
            self._history.insert(
                llm.ChatMessage(
                    id=speech.message_id,
                    role=role,
                    content=[""],
                    # spoken, not typed: what tells transcribed messages from the app's
                    transcript_confidence=1.0 if role == "user" else None,
                )
            )
            if role == "user":
                self.emit("input_speech_started", llm.InputSpeechStartedEvent())
        speech.text += event.delta
        speech.quiet_ms = 0
        if event.end_ms is not None:
            speech.end_ms = max(speech.end_ms or 0, event.end_ms)
        if isinstance(message := self._history.get_by_id(speech.message_id), llm.ChatMessage):
            message.content[0] = speech.text

        if role == "user":
            self.emit(
                "input_audio_transcription_completed",
                llm.InputTranscriptionCompleted(
                    item_id=speech.message_id,
                    transcript=speech.text,
                    is_final=False,
                    # the model answers over the caller, so the turn is stamped where it began
                    turn_started_at=speech.started_at,
                ),
            )
        else:
            self.emit(
                "transcript_delta",
                llm.DuplexOutputTranscriptDelta(
                    text=event.delta, start_ms=event.start_ms, end_ms=event.end_ms
                ),
            )

    def _end_speech(self, role: Role) -> None:
        """Close a speaker's message; for the user this is the end of their turn."""
        if (speech := self._speech.pop(role, None)) is None or role != "user":
            return
        # the final transcript goes out first, so nothing waits for one after the stop
        self.emit(
            "input_audio_transcription_completed",
            llm.InputTranscriptionCompleted(
                item_id=speech.message_id,
                transcript=speech.text,
                is_final=True,
                turn_started_at=speech.started_at,
            ),
        )
        self.emit(
            "input_speech_stopped", llm.InputSpeechStoppedEvent(user_transcription_enabled=False)
        )

    def _handle_delegation_created(self, event: types.SessionDelegationCreatedEvent) -> None:
        delegation = event.delegation
        if not delegation.id:
            logger.warning("gpt-live delegation has no id; nothing can answer it")
        elif delegation.target == "client":
            speech = self._speech.get("user")
            self.emit(
                "delegation_created",
                GPTLiveDelegation(
                    id=delegation.id, pending_transcript=speech.text if speech else ""
                ),
            )

    def _handle_response_event(self, envelope: types.ResponseEventEnvelope) -> None:
        # the inner event carries no response id, so a delegation's responses are followed in
        # sequence: a continuation is the next response.created under the same delegation, and a
        # response the application started itself has a null delegation
        event = envelope.event
        response = event.response
        d_id = envelope.delegation_id

        if event.type == "response.created":
            self._delegated_responses[d_id] = _DelegatedResponse()

        elif event.type == "response.output_item.done":
            # only the completed item carries the name, call id and arguments together
            item = event.item
            if item is None or item.type != "function_call":
                return
            if not item.call_id or not item.name or item.arguments is None:
                logger.warning(
                    "gpt-live dropping function call with missing fields",
                    extra={"call_id": item.call_id, "name": item.name},
                )
                return
            if (pending := self._delegated_responses.get(d_id)) is None:
                logger.warning(
                    "gpt-live function call outside a known response",
                    extra={"call_id": item.call_id, "delegation_id": d_id},
                )
                pending = self._delegated_responses[d_id] = _DelegatedResponse(completed=True)
            pending.call_ids.add(item.call_id)
            self._fnc_call_to_delegation[item.call_id] = d_id

            fnc_call = llm.FunctionCall(
                id=item.id or utils.shortuuid("fc_"),
                call_id=item.call_id,
                name=item.name,
                arguments=item.arguments,
            )
            self._history.insert(fnc_call)
            self.emit("function_call", fnc_call)

        elif event.type == "response.completed":
            if response is not None and (usage := response.usage) is not None:
                # the voice model is billed by duration; these tokens are the backend's and are
                # reported under its name
                self.emit(
                    "metrics_collected",
                    LLMMetrics(
                        label=self._live_model.label,
                        request_id=response.id or "",
                        timestamp=time.time(),
                        duration=0,
                        ttft=-1,
                        cancelled=False,
                        prompt_tokens=usage.input_tokens,
                        prompt_cached_tokens=usage.input_tokens_details.cached_tokens,
                        cache_creation_tokens=usage.input_tokens_details.cache_write_tokens,
                        completion_tokens=usage.output_tokens,
                        reasoning_tokens=usage.output_tokens_details.reasoning_tokens,
                        total_tokens=usage.total_tokens,
                        tokens_per_second=0,
                        metadata=Metadata(
                            model_name=response.model
                            or self._opts.responses.get("model", DEFAULT_BACKEND_MODEL),
                            model_provider=self._live_model.provider,
                        ),
                    ),
                )
            if (pending := self._delegated_responses.get(d_id)) is not None:
                pending.completed = True
                self._maybe_continue_response(d_id)

        elif event.type in ("response.failed", "response.incomplete"):
            logger.warning(
                "gpt-live backend response did not complete",
                extra={
                    "type": event.type,
                    "delegation_id": d_id,
                    "lk.pii.error": response.error if response else None,
                    "lk.pii.incomplete_details": response.incomplete_details if response else None,
                },
            )
            if (pending := self._delegated_responses.pop(d_id, None)) is not None:
                for call_id in pending.call_ids:
                    self._fnc_call_to_delegation.pop(call_id, None)

    def _maybe_continue_response(self, delegation_id: str | None) -> None:
        # response.create runs the continuation, and only once the response has finished asking
        # and every call it made has its answer; a partial batch is rejected
        pending = self._delegated_responses.get(delegation_id)
        if pending is None or not pending.completed or not pending.call_ids <= pending.returned:
            return
        del self._delegated_responses[delegation_id]
        if not pending.call_ids:
            return
        for call_id in pending.call_ids:
            self._fnc_call_to_delegation.pop(call_id, None)
        self.send_event(types.ResponseCreateEvent(event_id=utils.shortuuid("response_create_")))

    # metrics and errors

    def _handle_session_usage_updated(self, event: types.SessionUsageUpdatedEvent) -> None:
        if event.context_window is not None and event.context_window.usage_ratio is not None:
            logger.debug(
                "gpt-live context window utilization",
                extra={"usage_ratio": event.context_window.usage_ratio},
            )
        self._handle_usage(event.usage)

    def _handle_session_closed(self, event: types.SessionClosedEvent) -> None:
        logger.debug(
            "gpt-live session closed",
            extra={"reason": event.reason, "session_id": self._session_id},
        )
        self._handle_usage(event.usage)
        if not self._session_closed_fut.done():
            self._session_closed_fut.set_result(None)

    def _handle_usage(self, usage: types.Usage) -> None:
        # reported cumulatively for the whole session, so only the delta goes to the collectors
        previous, self._usage_total = self._usage_total, usage
        self.emit(
            "metrics_collected",
            RealtimeModelMetrics(
                timestamp=time.time(),
                request_id=self._session_id or "",
                ttft=-1,
                duration=0,
                session_duration=max(0.0, usage.seconds - previous.seconds),
                cancelled=False,
                label=self._live_model.label,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                tokens_per_second=0,
                input_token_details=RealtimeModelMetrics.InputTokenDetails(),
                output_token_details=RealtimeModelMetrics.OutputTokenDetails(),
                metadata=Metadata(
                    model_name=self._live_model.model, model_provider=self._live_model.provider
                ),
            ),
        )

    def _handle_error(self, error: types.ErrorBody) -> None:
        logger.error(
            "gpt-live returned an error",
            extra={"lk.pii.error": error.model_dump(exclude_none=True)},
        )
        recoverable = (error.code or error.type or "") not in _FATAL_ERROR_CODES
        api_error = APIError(
            message="GPT-Live returned an error",
            retryable=recoverable,
        )
        if not recoverable:
            raise api_error
        self._emit_error(api_error, recoverable=True)

    def _emit_error(self, error: Exception, recoverable: bool) -> None:
        self.emit(
            "error",
            llm.RealtimeModelError(
                timestamp=time.time(),
                label=self._live_model.label,
                error=error,
                recoverable=recoverable,
            ),
        )

    # DuplexSession interface

    @property
    def session_id(self) -> str | None:
        """The service's id for the current connection."""
        return self._session_id

    @property
    def audio_stream(self) -> AsyncIterable[llm.DuplexAudioFrame]:
        return self._audio_ch

    @property
    def tools(self) -> llm.ToolContext:
        return self._tools.copy()

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        # the caller's turn ends on their own audio: this much pushed since their last fragment
        if (speech := self._speech.get("user")) is not None:
            speech.quiet_ms += round(frame.duration * 1000)
            if speech.quiet_ms >= _MIN_SILENCE_MS:
                self._end_speech("user")

        if self._input_resampler and frame.sample_rate != self._input_resampler._input_rate:
            self._input_resampler = None
        if self._input_resampler is None and (
            frame.sample_rate != SAMPLE_RATE or frame.num_channels != NUM_CHANNELS
        ):
            self._input_resampler = rtc.AudioResampler(
                input_rate=frame.sample_rate, output_rate=SAMPLE_RATE, num_channels=NUM_CHANNELS
            )
        frames = self._input_resampler.push(frame) if self._input_resampler else [frame]
        for f in frames:
            for nf in self._bstream.write(f.data.tobytes()):
                self.send_event(
                    types.InputAudioAppendEvent(audio=base64.b64encode(nf.data).decode("utf-8"))
                )

    def append_instructions(self, text: str, *, delegation_id: str | None = None) -> None:
        """Add a standing rule to the model's instructions, capped at 500 tokens."""
        self._append(types.InstructionsAppendEvent, text, delegation_id)

    def append_thinking(self, text: str, *, delegation_id: str | None = None) -> None:
        """Give the model something to know without saying it, capped at 500 tokens."""
        self._append(types.ThinkingAppendEvent, text, delegation_id)

    def append_commentary(self, text: str, *, delegation_id: str | None = None) -> None:
        """Give the model something to say once, in its own words, capped at 500 tokens.

        Under client delegation this answers a :class:`GPTLiveDelegation`; repeated calls with the
        same ``delegation_id`` continue that work.
        """
        self._append(types.CommentaryAppendEvent, text, delegation_id)

    def _append(
        self,
        event_cls: type[types.InstructionsAppendEvent]
        | type[types.ThinkingAppendEvent]
        | type[types.CommentaryAppendEvent],
        text: str,
        delegation_id: str | None,
    ) -> None:
        self.send_event(
            event_cls(
                event_id=utils.shortuuid("append_"), delegation_id=delegation_id, content=text
            )
        )

    def mute_input(self) -> None:
        """Replace microphone input with silence; the model keeps generating and speaking."""
        self.send_event(types.InputAudioMuteEvent(event_id=utils.shortuuid("mute_")))

    def unmute_input(self) -> None:
        self.send_event(types.InputAudioUnmuteEvent(event_id=utils.shortuuid("unmute_")))

    async def aclose(self) -> None:
        await super().aclose()
        if not self._session_started_fut.done():
            self._session_started_fut.cancel()
        self._msg_ch.close()
        with contextlib.suppress(asyncio.CancelledError):
            await self._main_atask

    # framework hooks

    async def _update_instructions(self, instructions: str) -> None:
        if self._session_start_sent and instructions != self._instructions:
            raise llm.RealtimeError(
                "gpt-live voice instructions are immutable after session start; use "
                "append_instructions for a standing rule"
            )
        self._instructions = instructions

    async def _update_tools(self, tools: list[llm.Tool]) -> None:
        self._tools = llm.ToolContext(tools)
        if self._opts.delegation == "client":
            if tools:
                # dropping them silently leaves an agent whose tools simply never run
                raise llm.RealtimeError(
                    "gpt-live client delegation has no tool channel, so the model can never call "
                    f"{sorted(tool.id for tool in self._tools.flatten())}. Leave the agent's tools "
                    "empty and answer delegation_created with append_commentary, or pass "
                    'delegation="responses" to run tools on the backend model.'
                )
            return
        self._send_delegation_update(
            types.ResponsesConfig(tools=_build_delegation_tools(self._tools.flatten()))
        )

    async def _append_items(self, items: list[llm.ChatItem]) -> None:
        self._history.insert(items)
        if not self._session_start_sent:
            return  # startup history, rendered into session.start

        # a system or developer message is a standing rule for the voice model, a tool result
        # answering a call the backend delegated goes back on the backend's channel, and everything
        # else is context for the voice model, as one append
        backend_outputs: list[tuple[llm.FunctionCallOutput, str | None]] = []
        lines: list[str] = []
        for item in items:
            if isinstance(item, llm.ChatMessage) and item.role in ("system", "developer"):
                if text := item.text_content:
                    self.append_instructions(text)
            elif (
                isinstance(item, llm.FunctionCallOutput)
                and item.call_id in self._fnc_call_to_delegation
            ):
                backend_outputs.append((item, self._fnc_call_to_delegation[item.call_id]))
            elif (rendered := _render_item(item)) is not None:
                lines.append("{}: {}".format(*rendered))

        if lines:
            self.append_thinking("\n".join(lines))

        for output, delegation_id in backend_outputs:
            self.send_event(
                types.ResponseItemCreateEvent(
                    event_id=utils.shortuuid("tool_output_"),
                    item=FunctionCallOutput(
                        type="function_call_output", call_id=output.call_id, output=output.output
                    ),
                )
            )
            if (pending := self._delegated_responses.get(delegation_id)) is not None:
                pending.returned.add(output.call_id)
            self._maybe_continue_response(delegation_id)

        # TODO: under client delegation, answer a GPTLiveDelegation handled as a tool call with
        # append_commentary(output, delegation_id=...) here; nothing reaches the model for it yet
        # A manual call to append_commentary() is the only way to answer a GPTLiveDelegation for now

    def _generate_reply(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        tools: NotGivenOr[list[llm.Tool]] = NOT_GIVEN,
    ) -> None:
        if is_given(instructions):
            self.append_commentary(f"{_ASK_INSTRUCTED}\n\n{instructions}")
            return
        # a typed message still the newest thing said rides in the ask, once; after speech has
        # moved the conversation on, the ask points at the context instead
        newest = self._history.items[-1] if self._history.items else None
        typed = (
            newest.text_content
            if isinstance(newest, llm.ChatMessage)
            and newest.role == "user"
            and newest.transcript_confidence is None
            and newest.id != self._asked_item_id
            else None
        )
        self._asked_item_id = newest.id if newest is not None else None
        self.append_commentary(f"{_ASK_TYPED}\n\n{typed}" if typed else _ASK_BARE)

    def _update_options(
        self, *, tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN
    ) -> None:
        if is_given(tool_choice):
            self._opts.responses["tool_choice"] = tool_choice
            self._send_delegation_update(
                types.ResponsesConfig(tool_choice=_to_tool_choice(tool_choice))
            )


def _to_tool_choice(tool_choice: llm.ToolChoice | None) -> str | dict[str, Any]:
    if isinstance(tool_choice, str):
        return tool_choice
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        return {"type": "function", "name": tool_choice["function"]["name"]}
    return "auto"


def _build_delegation_tools(tools: list[llm.Tool]) -> list[dict[str, Any]]:
    oai_tools: list[dict[str, Any]] = []
    for tool in tools:
        if isinstance(tool, llm.FunctionTool):
            oai_tools.append(llm.utils.build_legacy_openai_schema(tool, internally_tagged=True))
        elif isinstance(tool, llm.RawFunctionTool):
            schema = dict(tool.info.raw_schema)
            schema.pop("meta", None)
            schema["type"] = "function"
            oai_tools.append(schema)
        elif isinstance(tool, OpenAITool):
            oai_tools.append(tool.to_dict())
        else:
            logger.debug(
                "gpt-live delegation ignores unsupported tool", extra={"lk.pii.tool": tool}
            )
    return oai_tools


def _render_item(item: llm.ChatItem) -> tuple[types.InputRole, str] | None:
    """A context item as the role and text the Live API carries; tool traffic is narrated."""
    if isinstance(item, llm.ChatMessage):
        if not (text := item.text_content):
            return None
        return ("developer" if item.role == "system" else item.role), text
    if isinstance(item, llm.FunctionCall):
        return "developer", f"Called tool {item.name} with {item.arguments}"
    if isinstance(item, llm.FunctionCallOutput):
        verb = "failed with" if item.is_error else "returned"
        return "developer", f"Tool {item.name or item.call_id} {verb} {item.output}"
    return None
