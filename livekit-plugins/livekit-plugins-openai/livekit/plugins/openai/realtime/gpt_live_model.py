from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import time
from collections.abc import AsyncIterable, Iterator
from dataclasses import dataclass, replace
from typing import Any, Literal
from urllib.parse import urlparse, urlunparse

import aiohttp

from livekit import rtc
from livekit.agents import APIConnectionError, APIError, llm, utils
from livekit.agents.metrics import RealtimeModelMetrics
from livekit.agents.metrics.base import Metadata
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import is_given
from livekit.agents.voice.generation import remove_instructions

from ..log import logger

# GPT-Live is a full-duplex voice model. Unlike the Realtime API it is server-driven: the model
# decides when to speak and self-manages barge-in, so there is no client response.create / cancel /
# truncate / commit. Reasoning and tools are delegated to a backend Responses model; this
# implementation covers the "responses" delegation mode over WebSocket.
#
# output audio streams continuously, inter-turn filler and silence included. every frame reaches
# the DuplexSession audio stream tagged with the assistant turn it belongs to, and the framework's
# DuplexRealtimeAdapter is what gates that stream and segments it into turns.

SAMPLE_RATE = 24000
NUM_CHANNELS = 1
DEFAULT_MODEL = "gpt-live-1-boulder-alpha"
DEFAULT_VOICE = "marin"
DEFAULT_BACKEND_MODEL = "gpt-5.5"
OPENAI_BASE_URL = "https://api.openai.com/v1"
ALPHA_VALUE = "quicksilver=v2"

# insurance only: the framework configures the session before the websocket handshake completes
_CONFIG_READY_TIMEOUT = 2.0

# session.closed carries the final usage; the service drains first, capped at 10s server-side
_SESSION_CLOSE_TIMEOUT = 2.0
_CLOSING_EVENTS = frozenset({"session.usage.updated", "session.closed"})

lk_oai_debug = int(os.getenv("LK_OPENAI_DEBUG", 0))

_FATAL_ERROR_CODES = frozenset(
    {
        "insufficient_quota",
        "invalid_api_key",
        "account_deactivated",
        "billing_hard_limit_reached",
    }
)


def _is_fatal_error(error: object | None) -> bool:
    code = None
    if isinstance(error, dict):
        code = error.get("code") or error.get("type")
    else:
        code = getattr(error, "code", None) or getattr(error, "type", None)
    return isinstance(code, str) and code in _FATAL_ERROR_CODES


def _build_live_url(base_url: str, model: str) -> str:
    """Turn an http(s) base url into the wss GPT-Live endpoint with the model query."""
    if base_url.startswith("http"):
        base_url = base_url.replace("http", "ws", 1)

    parsed = urlparse(base_url)
    path = parsed.path.rstrip("/")
    if not path.endswith("/live"):
        path = f"{path}/live"

    query = f"model={model}"
    return urlunparse((parsed.scheme, parsed.netloc, path, "", query, ""))


def _to_tool_choice(tool_choice: llm.ToolChoice | None) -> Any:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        return tool_choice
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        return {"type": "function", "name": tool_choice["function"]["name"]}
    return "auto"


def _build_delegation_tools(tools: list[llm.Tool]) -> list[dict[str, Any]]:
    """Convert framework tools into Responses-delegation tool entries."""
    oai_tools: list[dict[str, Any]] = []
    for tool in tools:
        if isinstance(tool, llm.FunctionTool):
            oai_tools.append(llm.utils.build_legacy_openai_schema(tool, internally_tagged=True))
        elif isinstance(tool, llm.RawFunctionTool):
            schema = dict(tool.info.raw_schema)
            schema.pop("meta", None)
            schema["type"] = "function"
            oai_tools.append(schema)
        else:
            logger.debug("gpt-live delegation ignores unsupported tool", extra={"tool": tool})
    return oai_tools


def _as_int(value: object) -> int | None:
    return value if isinstance(value, int) else None


def _nested_int(data: dict[str, Any], *path: str) -> int:
    value: Any = data
    for key in path:
        if not isinstance(value, dict):
            return 0
        value = value.get(key)
    return value if isinstance(value, int) else 0


@dataclass
class _LiveOptions:
    model: str
    voice: str
    instructions: str | None
    backend_model: str
    backend_instructions: str | None
    tool_choice: llm.ToolChoice | None
    web_search: bool
    reasoning: dict[str, Any] | None
    service_tier: str | None
    max_output_tokens: int | None
    api_key: str
    base_url: str
    conn_options: APIConnectOptions
    max_session_duration: float | None


class GPTLiveModel(llm.DuplexModel):
    """OpenAI GPT-Live full-duplex voice model (alpha), ready to pass to ``AgentSession(llm=)``."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        voice: str = DEFAULT_VOICE,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        backend_model: str = DEFAULT_BACKEND_MODEL,
        backend_instructions: NotGivenOr[str] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN,
        web_search: bool = False,
        reasoning: NotGivenOr[dict[str, Any] | None] = NOT_GIVEN,
        service_tier: NotGivenOr[str | None] = NOT_GIVEN,
        max_output_tokens: NotGivenOr[int | None] = NOT_GIVEN,
        api_key: str | None = None,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
        max_session_duration: NotGivenOr[float | None] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        """
        Args:
            model: GPT-Live voice model slug.
            voice: Output voice. Immutable after the session starts.
            instructions: Voice-model system instructions. Immutable after the session starts.
            backend_model: Responses model the voice model delegates reasoning and tools to.
            backend_instructions: Instructions for the backend Responses model.
            tool_choice: Tool selection policy for the backend Responses model.
            web_search: Enable the server-side ``web_search`` hosted tool on the backend.
            reasoning: Reasoning config for the backend Responses model, e.g. ``{"effort": "medium"}``.
            service_tier: Backend service tier (``auto``, ``default``, ``flex`` or ``priority``).
            max_output_tokens: Backend max output tokens.
            api_key: OpenAI API key. Falls back to ``OPENAI_API_KEY``.
            base_url: HTTP base url of the OpenAI API.
            http_session: Optional shared HTTP session.
            max_session_duration: Seconds before the connection is recycled.
            conn_options: Retry/backoff and connection settings.
        """  # noqa: E501
        super().__init__(
            capabilities=llm.DuplexCapabilities(
                user_transcription=True,
                # the model continues on its own once a tool result reaches the backend
                auto_tool_reply_generation=True,
                mutable_chat_context=False,
                mutable_instructions=False,
                mutable_tools=True,
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
            instructions=instructions if is_given(instructions) else None,
            backend_model=backend_model,
            backend_instructions=backend_instructions if is_given(backend_instructions) else None,
            tool_choice=tool_choice if is_given(tool_choice) else None,
            web_search=web_search,
            reasoning=reasoning if is_given(reasoning) else None,
            service_tier=service_tier if is_given(service_tier) else None,
            max_output_tokens=max_output_tokens if is_given(max_output_tokens) else None,
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

    def session(self) -> GPTLiveSession:
        return GPTLiveSession(self)

    async def aclose(self) -> None:
        if self._http_session_owned and self._http_session:
            await self._http_session.close()


class GPTLiveSession(
    llm.DuplexSession[Literal["openai_server_event_received", "openai_client_event_queued"]]
):
    """A session for the OpenAI GPT-Live API (WebSocket), reached with ``Agent.duplex_session``.

    Exposes two extra events mirroring the Realtime session:
    - openai_server_event_received: raw server events
    - openai_client_event_queued: raw client events sent to the server
    """

    def __init__(self, duplex_model: GPTLiveModel) -> None:
        super().__init__(duplex_model)
        self._live_model = duplex_model
        self._opts = replace(duplex_model._opts)
        self._tools = llm.ToolContext.empty()
        self._instructions = self._opts.instructions
        self._msg_ch = utils.aio.Chan[dict[str, Any]]()
        self._audio_ch = utils.aio.Chan[llm.DuplexAudioFrame]()
        self._input_resampler: rtc.AudioResampler | None = None

        self._closing = False
        # the first session.update of a connection carries the immutable voice config; it is
        # composed at connect time from the captured instructions/tools, which the framework
        # supplies right after the session is created
        self._initial_config_sent = False
        self._config_ready = asyncio.Event()
        self._session_started_fut: asyncio.Future[None] = asyncio.Future()
        self._session_closed_fut: asyncio.Future[None] = asyncio.Future()
        self._session_id: str | None = None
        # session usage is reported cumulatively; kept to emit per-event deltas
        self._usage_total: dict[str, Any] = {}

        # turn.delta carries only a turn_id (no role); the role comes from turn.created, so route
        # user vs assistant transcript by turn_id instead of by whichever side is currently open
        self._turn_roles: dict[str, str] = {}
        self._user_transcripts: dict[str, str] = {}
        self._assistant_transcripts: dict[str, str] = {}
        # the turn output audio is attributed to, so the framework can tell one apart from the next
        self._assistant_turn_id: str | None = None
        # response.function_call_arguments.done omits the function name; capture it from the
        # earlier response.output_item.added, keyed by item_id
        self._pending_fnc_calls: dict[str, dict[str, Any]] = {}
        # calls this connection delegated; only their results mean anything to the backend
        self._delegated_calls: set[str] = set()

        # local mirror of items already synced to the server; used to diff tool outputs
        self._remote_chat_ctx = llm.ChatContext.empty()

        self._bstream = utils.audio.AudioByteStream(
            SAMPLE_RATE, NUM_CHANNELS, samples_per_channel=SAMPLE_RATE // 10
        )

        self._main_atask = asyncio.create_task(self._main_task(), name="GPTLiveSession._main")

    # -- outbound --------------------------------------------------------------------------------

    def send_event(self, event: dict[str, Any]) -> None:
        with contextlib.suppress(utils.aio.channel.ChanClosed):
            self._msg_ch.send_nowait(event)

    def _create_session_update_event(self) -> dict[str, Any]:
        responses: dict[str, Any] = {"model": self._opts.backend_model}
        if self._opts.backend_instructions is not None:
            responses["instructions"] = self._opts.backend_instructions
        if (tool_choice := _to_tool_choice(self._opts.tool_choice)) is not None:
            responses["tool_choice"] = tool_choice
        if self._opts.reasoning is not None:
            responses["reasoning"] = self._opts.reasoning
        if self._opts.service_tier is not None:
            responses["service_tier"] = self._opts.service_tier
        if self._opts.max_output_tokens is not None:
            responses["max_output_tokens"] = self._opts.max_output_tokens

        tools = _build_delegation_tools(self._tools.flatten())
        if self._opts.web_search:
            tools.insert(0, {"type": "web_search"})
        if tools:
            responses["tools"] = tools

        session: dict[str, Any] = {
            "audio": {"output": {"voice": self._opts.voice}},
            "delegation": {"type": "responses", "responses": responses},
        }
        if self._instructions is not None:
            session["instructions"] = self._instructions

        return {
            "type": "session.update",
            "event_id": utils.shortuuid("session_update_"),
            "session": session,
        }

    def _create_delegation_update_event(self) -> dict[str, Any]:
        """Session.update replacing only the delegation object (tools / tool_choice)."""
        ev = self._create_session_update_event()
        ev["session"].pop("instructions", None)
        ev["session"].pop("audio", None)
        ev["event_id"] = utils.shortuuid("delegation_update_")
        return ev

    # -- connection loop -------------------------------------------------------------------------

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        num_retries = 0
        max_retries = self._opts.conn_options.max_retry
        reconnecting = False

        try:
            while not self._msg_ch.closed:
                try:
                    ws_conn = await self._create_ws_conn()
                    if reconnecting:
                        self._reset_for_reconnect()
                        num_retries = 0
                        self.emit("session_reconnected", llm.RealtimeSessionReconnectedEvent())
                    await self._run_ws(ws_conn)
                except APIError as e:
                    if max_retries == 0 or not e.retryable:
                        self._emit_error(e, recoverable=False)
                        raise
                    elif num_retries == max_retries:
                        self._emit_error(e, recoverable=False)
                        raise APIConnectionError(
                            f"{self._live_model._provider_label} connection failed after "
                            f"{num_retries} attempts",
                        ) from e
                    else:
                        self._emit_error(e, recoverable=True)
                        interval = self._opts.conn_options._interval_for_retry(num_retries)
                        logger.warning(
                            f"{self._live_model._provider_label} connection failed, "
                            f"retrying in {interval}s",
                            exc_info=e,
                        )
                        await asyncio.sleep(interval)
                    num_retries += 1
                except Exception as e:
                    self._emit_error(e, recoverable=False)
                    raise
                reconnecting = True
        finally:
            self._audio_ch.close()

    def _reset_for_reconnect(self) -> None:
        self._initial_config_sent = False
        self._session_started_fut = asyncio.Future()
        self._remote_chat_ctx = llm.ChatContext.empty()
        if self._assistant_turn_id is not None:
            self._end_assistant_turn(self._assistant_turn_id)
        self._turn_roles.clear()
        self._user_transcripts.clear()
        self._assistant_transcripts.clear()
        self._pending_fnc_calls.clear()
        self._delegated_calls.clear()
        # a new connection is a new session, so its usage counters restart from zero
        self._usage_total = {}
        self._session_id = None

    async def _create_ws_conn(self) -> aiohttp.ClientWebSocketResponse:
        headers = {
            "User-Agent": "LiveKit Agents",
            "Authorization": f"Bearer {self._opts.api_key}",
            "OpenAI-Alpha": ALPHA_VALUE,
        }
        url = _build_live_url(self._opts.base_url, self._opts.model)
        if lk_oai_debug:
            logger.debug(f"connecting to GPT-Live API: {url}")

        t0 = time.perf_counter()
        try:
            ws = await asyncio.wait_for(
                self._live_model._ensure_http_session().ws_connect(url=url, headers=headers),
                self._opts.conn_options.timeout,
            )
            self._report_connection_acquired(time.perf_counter() - t0)
            return ws
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            raise APIConnectionError(f"{self._live_model._provider_label} connection error") from e

    async def _run_ws(self, ws_conn: aiohttp.ClientWebSocketResponse) -> None:
        closing = False

        @utils.log_exceptions(logger=logger)
        async def _send_task() -> None:
            nonlocal closing
            # first event of the connection is the composed session.update, which needs the
            # instructions and tools the framework hands over right after session()
            if not self._config_ready.is_set():
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(self._config_ready.wait(), _CONFIG_READY_TIMEOUT)
            self._initial_config_sent = True
            await self._ws_send(ws_conn, self._create_session_update_event())

            async for msg in self._msg_ch:
                # hold input audio until the server acknowledges the session; cancelled on close
                if msg.get("type") == "input_audio.append" and not self._session_started_fut.done():
                    await self._session_started_fut
                await self._ws_send(ws_conn, msg)

            closing = True
            with contextlib.suppress(Exception):
                await self._ws_send(ws_conn, {"type": "session.close"})
            # the service drains, then reports the final usage in session.closed. only worth
            # waiting for when the session was actually running.
            if self._session_started_fut.done() and not self._session_started_fut.cancelled():
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(
                        asyncio.shield(self._session_closed_fut), _SESSION_CLOSE_TIMEOUT
                    )
            await ws_conn.close()

        @utils.log_exceptions(logger=logger)
        async def _recv_task() -> None:
            while True:
                msg = await ws_conn.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if closing:
                        return
                    raise APIConnectionError(
                        f"{self._live_model._provider_label} connection closed unexpectedly"
                    )
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue

                event = json.loads(msg.data)
                self.emit("openai_server_event_received", event)
                # while closing, only the shutdown events matter; anything else would be
                # emitted into a session the framework has already torn down
                if self._closing and event.get("type") not in _CLOSING_EVENTS:
                    continue
                try:
                    self._handle_event(event)
                except Exception as e:
                    if isinstance(e, APIError) and not e.retryable:
                        raise
                    logger.exception(
                        "failed to handle gpt-live event", extra={"type": event.get("type")}
                    )

        tasks = [
            asyncio.create_task(_recv_task(), name="_recv_task"),
            asyncio.create_task(_send_task(), name="_send_task"),
        ]
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
        finally:
            await utils.aio.cancel_and_wait(*tasks)
            await ws_conn.close()

    async def _ws_send(
        self, ws_conn: aiohttp.ClientWebSocketResponse, event: dict[str, Any]
    ) -> None:
        self.emit("openai_client_event_queued", event)
        if lk_oai_debug and event.get("type") != "input_audio.append":
            logger.debug(f">>> {event}")
        await ws_conn.send_str(json.dumps(event))

    # -- inbound event dispatch ------------------------------------------------------------------

    def _handle_event(self, event: dict[str, Any]) -> None:
        etype = event.get("type", "")
        if lk_oai_debug and etype != "output_audio.delta":
            logger.debug(f"<<< {event}")

        if etype in ("session.started", "session.updated"):
            if (session_id := (event.get("session") or {}).get("id")) and isinstance(
                session_id, str
            ):
                self._session_id = session_id
            if not self._session_started_fut.done():
                self._session_started_fut.set_result(None)
        elif etype == "output_audio.delta":
            self._handle_output_audio_delta(event)
        elif etype == "output_transcript.added":
            self._handle_output_transcript_added(event)
        elif etype == "turn.created":
            self._handle_turn_created(event)
        elif etype == "turn.delta":
            self._handle_turn_delta(event)
        elif etype == "turn.done":
            self._handle_turn_done(event)
        elif etype == "response.output_item.added":
            self._handle_response_output_item_added(event)
        elif etype == "response.function_call_arguments.done":
            self._handle_function_call_arguments_done(event)
        elif etype in ("session.usage.updated", "session.closed"):
            self._handle_usage(event)
            if etype == "session.closed" and not self._session_closed_fut.done():
                self._session_closed_fut.set_result(None)
        elif etype == "error":
            self._handle_error(event)
        elif etype == "input_transcript.added":
            # the user-side twin of output_transcript.added, repeating fragments turn.delta already
            # carries; its per-fragment timing has no consumer, since only the agent's own speech is
            # paced against playout
            pass
        elif lk_oai_debug:
            logger.debug(f"unhandled gpt-live event: {etype}")

    # -- model output ----------------------------------------------------------------------------

    def _handle_output_audio_delta(self, event: dict[str, Any]) -> None:
        encoded = event.get("audio")
        if not isinstance(encoded, str):
            return
        data = base64.b64decode(encoded)
        if not data:
            return
        # every frame is published, silence and inter-turn filler included: the model emits audio
        # unconditionally and the framework is what decides which of it is worth playing
        frame = rtc.AudioFrame(
            data=data,
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
            samples_per_channel=len(data) // 2,
        )
        with contextlib.suppress(utils.aio.channel.ChanClosed):
            self._audio_ch.send_nowait(
                llm.DuplexAudioFrame(
                    frame=frame,
                    message_id=self._assistant_turn_id,
                    start_ms=_as_int(event.get("start_ms")),
                )
            )

    def _handle_output_transcript_added(self, event: dict[str, Any]) -> None:
        item = event.get("item") or {}
        text = item.get("text")
        if not isinstance(text, str) or not text:
            return
        # only the open turn may label a fragment: item.id is per fragment, and the framework reads
        # a change of id as a change of turn. before turn.created there is no id, and None says so
        message_id = self._assistant_turn_id
        if message_id is not None:
            self._assistant_transcripts[message_id] = (
                self._assistant_transcripts.get(message_id, "") + text
            )
        self.emit(
            "transcript_delta",
            llm.DuplexTranscriptDelta(
                message_id=message_id,
                text=text,
                start_ms=_as_int(event.get("start_ms")),
                end_ms=_as_int(event.get("end_ms")),
            ),
        )

    def _end_assistant_turn(self, turn_id: str) -> None:
        """End an assistant turn, recording what it said in the local conversation mirror."""
        if self._assistant_turn_id == turn_id:
            self._assistant_turn_id = None
        if transcript := self._assistant_transcripts.pop(turn_id, ""):
            self._remote_chat_ctx.items.append(
                llm.ChatMessage(id=turn_id, role="assistant", content=[transcript])
            )
        self.emit("turn_ended", llm.DuplexTurnEndedEvent(message_id=turn_id))

    # -- user turns ------------------------------------------------------------------------------

    def _emit_user_transcript(self, turn_id: str, *, is_final: bool) -> None:
        self.emit(
            "input_audio_transcription_completed",
            llm.InputTranscriptionCompleted(
                item_id=turn_id,
                transcript=self._user_transcripts.get(turn_id, ""),
                is_final=is_final,
            ),
        )

    def _handle_turn_created(self, event: dict[str, Any]) -> None:
        turn = event.get("turn") or {}
        turn_id = turn.get("id")
        role = turn.get("role")
        if not turn_id or role not in ("user", "assistant"):
            return
        self._turn_roles[turn_id] = role
        if role == "user":
            # a user turn is a projection over transcript fragments, not a state change: the model
            # may well keep speaking over it, so the open assistant turn is left alone and ends on
            # its own turn.done
            self._user_transcripts[turn_id] = turn.get("transcript") or ""
            self.emit("input_speech_started", llm.InputSpeechStartedEvent())
            if self._user_transcripts[turn_id]:
                self._emit_user_transcript(turn_id, is_final=False)
        else:
            self._assistant_turn_id = turn_id
            self.emit("turn_started", llm.DuplexTurnStartedEvent(message_id=turn_id))

    def _handle_turn_delta(self, event: dict[str, Any]) -> None:
        # only user transcript here; assistant text comes from output_transcript.added
        turn_id = event.get("turn_id")
        delta = event.get("delta")
        if not turn_id or not delta:
            return
        if self._turn_roles.get(turn_id) == "user":
            self._user_transcripts[turn_id] = self._user_transcripts.get(turn_id, "") + delta
            self._emit_user_transcript(turn_id, is_final=False)

    def _handle_turn_done(self, event: dict[str, Any]) -> None:
        turn = event.get("turn") or {}
        turn_id = turn.get("id") or event.get("turn_id")
        if not turn_id:
            return
        role = self._turn_roles.pop(turn_id, None) or turn.get("role")
        if role == "user":
            if transcript := turn.get("transcript"):
                self._user_transcripts[turn_id] = transcript
            self._emit_user_transcript(turn_id, is_final=True)
            final_transcript = self._user_transcripts.pop(turn_id, "")
            if final_transcript:
                self._remote_chat_ctx.items.append(
                    llm.ChatMessage(id=turn_id, role="user", content=[final_transcript])
                )
            self.emit(
                "input_speech_stopped",
                llm.InputSpeechStoppedEvent(user_transcription_enabled=True),
            )
        elif role == "assistant":
            self._end_assistant_turn(turn_id)

    # -- function calls --------------------------------------------------------------------------

    def _handle_response_output_item_added(self, event: dict[str, Any]) -> None:
        # capture the function name/call_id early; the later .done event carries only the arguments
        item = event.get("item") or {}
        if item.get("type") != "function_call":
            return
        item_id = item.get("id")
        if item_id:
            self._pending_fnc_calls[item_id] = {
                "call_id": item.get("call_id"),
                "name": item.get("name"),
            }

    def _handle_function_call_arguments_done(self, event: dict[str, Any]) -> None:
        item_id = event.get("item_id")
        pending = self._pending_fnc_calls.pop(item_id, {}) if item_id else {}
        call_id = event.get("call_id") or pending.get("call_id")
        name = event.get("name") or pending.get("name")
        arguments = event.get("arguments")
        if not call_id or not name or arguments is None:
            logger.warning(
                "gpt-live dropping function call with missing fields",
                extra={
                    "call_id": call_id,
                    "name": name,
                    "has_arguments": arguments is not None,
                    "event_keys": list(event.keys()),
                },
            )
            return
        fnc_call = llm.FunctionCall(
            id=item_id or utils.shortuuid("fc_"), call_id=call_id, name=name, arguments=arguments
        )
        # mirror the call so update_chat_ctx can diff the matching output
        self._remote_chat_ctx.items.append(fnc_call)
        self._delegated_calls.add(call_id)
        self.emit("function_call", fnc_call)

    # -- metrics and errors ----------------------------------------------------------------------

    def _handle_usage(self, event: dict[str, Any]) -> None:
        usage = event.get("usage") or {}
        if not usage:
            return
        # session.usage.updated and session.closed both report usage cumulatively for the whole
        # session, so only the delta is reported for the collectors to sum
        previous, self._usage_total = self._usage_total, usage

        def delta(*path: str) -> int:
            return max(0, _nested_int(usage, *path) - _nested_int(previous, *path))

        metrics = RealtimeModelMetrics(
            timestamp=time.time(),
            request_id=self._session_id or "",
            ttft=-1,
            duration=0,
            cancelled=False,
            label=self._live_model.label,
            input_tokens=delta("input_tokens"),
            output_tokens=delta("output_tokens"),
            total_tokens=delta("total_tokens"),
            tokens_per_second=0,
            input_token_details=RealtimeModelMetrics.InputTokenDetails(
                audio_tokens=delta("input_token_details", "audio_tokens"),
                cached_tokens=delta("input_token_details", "cached_tokens"),
                text_tokens=delta("input_token_details", "text_tokens"),
                image_tokens=delta("input_token_details", "image_tokens"),
            ),
            output_token_details=RealtimeModelMetrics.OutputTokenDetails(
                text_tokens=delta("output_token_details", "text_tokens"),
                audio_tokens=delta("output_token_details", "audio_tokens"),
                image_tokens=delta("output_token_details", "image_tokens"),
            ),
            metadata=Metadata(
                model_name=self._live_model.model, model_provider=self._live_model.provider
            ),
        )
        self.emit("metrics_collected", metrics)

    def _handle_error(self, event: dict[str, Any]) -> None:
        error = event.get("error") or {}
        logger.error(
            f"{self._live_model._provider_label} returned an error", extra={"error": error}
        )
        recoverable = not _is_fatal_error(error)
        api_error = APIError(
            message=f"{self._live_model._provider_label} returned an error",
            body=error,
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

    # -- DuplexSession interface -----------------------------------------------------------------

    @property
    def audio_stream(self) -> AsyncIterable[llm.DuplexAudioFrame]:
        return self._audio_ch

    @property
    def chat_ctx(self) -> llm.ChatContext:
        return self._remote_chat_ctx.copy()

    @property
    def tools(self) -> llm.ToolContext:
        return self._tools.copy()

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        for f in self._resample_audio(frame):
            for nf in self._bstream.write(f.data.tobytes()):
                self.send_event(
                    {
                        "type": "input_audio.append",
                        "audio": base64.b64encode(nf.data).decode("utf-8"),
                    }
                )

    def append_context(self, text: str) -> None:
        """Give the model something to know, for it to use from its next turn on.

        The Live API's context is append-only and its entries carry no role, so this cannot seed a
        conversation history. Text added here never reaches ``AgentSession.history``.
        """
        self.send_event(
            {
                "type": "session.context.append",
                "event_id": utils.shortuuid("context_"),
                "content": [{"type": "input_text", "text": text}],
            }
        )

    def send_delegation_output(self, *, call_id: str, output: str) -> None:
        """Return the result of a tool call the backend model delegated to the client."""
        self.send_event(
            {
                "type": "delegation.function_call_output.create",
                "event_id": utils.shortuuid("fnc_output_"),
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output,
                },
            }
        )

    async def aclose(self) -> None:
        self._closing = True
        # unblock the send loop if it is holding audio until session.started
        if not self._session_started_fut.done():
            self._session_started_fut.cancel()
        self._msg_ch.close()
        with contextlib.suppress(asyncio.CancelledError):
            await self._main_atask

    # -- framework hooks -------------------------------------------------------------------------

    async def _update_instructions(self, instructions: str) -> None:
        if self._initial_config_sent:
            if instructions != self._instructions:
                logger.debug(
                    "gpt-live voice instructions are immutable after session start; ignoring update"
                )
            return
        self._instructions = instructions

    async def _update_tools(self, tools: list[llm.Tool]) -> None:
        self._tools = llm.ToolContext(tools)
        if self._initial_config_sent:
            self.send_event(self._create_delegation_update_event())

    async def _update_session(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        chat_ctx: NotGivenOr[llm.ChatContext] = NOT_GIVEN,
        tools: NotGivenOr[list[llm.Tool]] = NOT_GIVEN,
    ) -> None:
        # the voice instructions are immutable after the first session.update, so the handshake
        # waits for the agent's full configuration rather than racing it
        await super()._update_session(instructions=instructions, chat_ctx=chat_ctx, tools=tools)
        self._config_ready.set()

    async def _update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        # the framework hands over the whole context, so what is new is recovered by diffing it
        # against what has already gone out, then routed to the protocol event that carries it
        chat_ctx = chat_ctx.copy(exclude_handoff=True, exclude_config_update=True)
        remove_instructions(chat_ctx)
        diff = llm.utils.compute_chat_ctx_diff(self._remote_chat_ctx, chat_ctx)

        # a revision the model cannot apply is the same defect as a deletion it cannot apply
        if stale := (diff.to_remove + [item_id for _, item_id in diff.to_update]):
            logger.error(
                "gpt-live context is append-only; the model keeps what it has been told",
                extra={"item_ids": stale},
            )

        # one append for the whole turn: the model reads context as a block, so a transcript of who
        # said what stays coherent where message-per-event would arrive as disconnected fragments
        transcript: list[str] = []
        for _, item_id in diff.to_create:
            item = chat_ctx.get_by_id(item_id)
            if isinstance(item, llm.FunctionCall):
                # a call this connection made is already the backend's; anything else is history
                if item.call_id not in self._delegated_calls:
                    transcript.append(f"tool call: {item.name}({item.arguments})")
            elif isinstance(item, llm.FunctionCallOutput):
                if item.call_id in self._delegated_calls:
                    # answers a call still open on the backend, so it goes back on its own channel
                    self.send_delegation_output(call_id=item.call_id, output=item.output)
                else:
                    transcript.append(f"tool result: {item.output}")
            elif isinstance(item, llm.ChatMessage) and (text := item.text_content):
                transcript.append(f"{item.role}: {text}")

        if transcript:
            self.append_context("\n".join(transcript))

        self._remote_chat_ctx = chat_ctx

    def _update_options(
        self, *, tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN
    ) -> None:
        if is_given(tool_choice):
            self._opts.tool_choice = tool_choice
            if self._initial_config_sent:
                self.send_event(self._create_delegation_update_event())

    def _resample_audio(self, frame: rtc.AudioFrame) -> Iterator[rtc.AudioFrame]:
        if self._input_resampler:
            if frame.sample_rate != self._input_resampler._input_rate:
                self._input_resampler = None

        if self._input_resampler is None and (
            frame.sample_rate != SAMPLE_RATE or frame.num_channels != NUM_CHANNELS
        ):
            self._input_resampler = rtc.AudioResampler(
                input_rate=frame.sample_rate,
                output_rate=SAMPLE_RATE,
                num_channels=NUM_CHANNELS,
            )

        if self._input_resampler:
            yield from self._input_resampler.push(frame)
        else:
            yield frame
