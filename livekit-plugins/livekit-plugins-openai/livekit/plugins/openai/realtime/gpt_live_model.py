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
from livekit.agents.metrics import LLMMetrics, RealtimeModelMetrics
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
from ..tools import OpenAITool
from . import gpt_live_types as types

# GPT-Live is a full-duplex voice model. Unlike the Realtime API it is server-driven: the model
# decides when to speak and self-manages barge-in, so there is no client response.create / cancel /
# truncate / commit. Reasoning and tools are delegated, over WebSocket, either to a backend
# Responses model or to the application.
#
# output audio streams continuously, inter-turn filler and silence included. every frame reaches
# the DuplexSession audio stream tagged with the assistant turn it belongs to, and the framework's
# DuplexRealtimeAdapter is what gates that stream and segments it into turns.

SAMPLE_RATE = 24000
NUM_CHANNELS = 1
DEFAULT_MODEL = "gpt-live-1-marble-alpha"
DEFAULT_VOICE = "marin"
DEFAULT_BACKEND_MODEL = "gpt-5.6-sol"
OPENAI_BASE_URL = "https://api.openai.com/v1"
ALPHA_VALUE = "quicksilver=v2"

# TODO: tune the prompts and caps
# service caps, budgeted at the usual ~4 characters a token since there is no tokenizer here
_MAX_INITIAL_ITEMS = 128
_MAX_INITIAL_CHARS = 8192 * 4
_MAX_CONTEXT_CHARS = 500 * 4
_MAX_OPENING_CHARS = 250 * 4
_MAX_FEEDBACK_CHARS = 4096

# no client event asks for a turn, so a reply is requested by putting the ask in the context
_SPEAK_NOW = (
    # "Speak now. Do not wait for the user to say anything first. Afterwards, pause and listen."
    "User said something, reply to it."
)

# session.closed carries the final usage; the service drains first, capped at 10s server-side
_SESSION_CLOSE_TIMEOUT = 5.0
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


# reported before session.opening.completed, and between them they mean the caller heard nothing
_OPENING_FAILURE_CODES = frozenset({"opening_timeout", "opening_no_output_audio"})


def _is_fatal_error(error: types.ErrorBody) -> bool:
    return (error.code or error.type or "") in _FATAL_ERROR_CODES


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


def _to_tool_choice(tool_choice: llm.ToolChoice | None) -> str | dict[str, Any] | None:
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
        elif isinstance(tool, OpenAITool):
            oai_tools.append(tool.to_dict())
        else:
            logger.debug("gpt-live delegation ignores unsupported tool", extra={"tool": tool})
    return oai_tools


def _render_item(item: llm.ChatItem) -> tuple[llm.ChatRole, str] | None:
    """One context item as the role and text the Live API carries.

    A tool call has no role of its own here, so it is narrated to the developer.
    """
    if isinstance(item, llm.ChatMessage):
        return (item.role, text) if (text := item.text_content) else None
    if isinstance(item, llm.FunctionCall):
        return "developer", f"Called tool {item.name} with {item.arguments}"
    if isinstance(item, llm.FunctionCallOutput):
        verb = "failed with" if item.is_error else "returned"
        return "developer", f"Tool {item.name or item.call_id} {verb} {item.output}"
    return None


@dataclass
class GPTLiveDelegation:
    """Work the model handed to the application, under client delegation."""

    id: str
    """Answer it with :meth:`GPTLiveSession.send_delegation_context`."""
    text: str
    """What the model is asking for, in its own words."""


@dataclass
class _LiveOptions:
    model: str
    voice: str
    instructions: str | None
    opening: str | None
    delegation: types.DelegationTarget
    backend_model: str
    backend_instructions: str | None
    tool_choice: llm.ToolChoice | None
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
        opening: NotGivenOr[str] = NOT_GIVEN,
        delegation: types.DelegationTarget = "responses",
        backend_model: str = DEFAULT_BACKEND_MODEL,
        backend_instructions: NotGivenOr[str] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN,
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
            opening: A passage the model speaks first, at most 250 tokens. The server mutes the
                microphone for its duration, so the caller cannot interrupt it. The model may
                reword it, and it cannot be set after the session starts.
            delegation: Where delegated work goes. ``responses`` runs it on a backend model, so
                ``@function_tool`` works as usual; ``client`` hands it to the application as a
                ``delegation_created`` event, which no framework tool can answer.
            backend_model: Responses model the voice model delegates reasoning and tools to.
            backend_instructions: Instructions for the backend Responses model.
            tool_choice: Tool selection policy for the backend Responses model.
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
                # no client event creates a turn, but a speakable context append asks for one
                manual_response_creation=True,
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
            opening=opening if is_given(opening) else None,
            delegation=delegation,
            backend_model=backend_model,
            backend_instructions=backend_instructions if is_given(backend_instructions) else None,
            tool_choice=tool_choice if is_given(tool_choice) else None,
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

    def session(self, *, wait_for_config: bool = False) -> GPTLiveSession:
        return GPTLiveSession(self, wait_for_config=wait_for_config)

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
    """

    def __init__(self, duplex_model: GPTLiveModel, *, wait_for_config: bool = False) -> None:
        super().__init__(duplex_model, wait_for_config=wait_for_config)
        self._live_model = duplex_model
        self._opts = replace(duplex_model._opts)
        self._tools = llm.ToolContext.empty()
        self._instructions = self._opts.instructions
        self._msg_ch = utils.aio.Chan[types.ClientEvent | dict[str, Any]]()
        self._audio_ch = utils.aio.Chan[llm.DuplexAudioFrame]()
        self._input_resampler: rtc.AudioResampler | None = None

        self._closing = False
        # the first session.update of a connection carries the config that is immutable after it
        self._initial_config_sent = False
        # an opening only belongs in a conversation nobody has spoken in yet: a connection that
        # drops before anyone did still carries it, a later one would talk over what is under way
        self._conversation_started = False
        self._opening_failed = False
        self._session_started_fut: asyncio.Future[None] = asyncio.Future()
        self._session_closed_fut: asyncio.Future[None] = asyncio.Future()
        self._session_id: str | None = None
        # session usage is reported cumulatively; kept to emit per-event deltas
        self._usage_total = types.Usage()

        # turn.delta carries only a turn_id (no role); the role comes from turn.created, so route
        # user vs assistant transcript by turn_id instead of by whichever side is currently open
        self._turn_roles: dict[str, types.TurnRole] = {}
        self._user_transcripts: dict[str, str] = {}
        self._assistant_transcripts: dict[str, str] = {}
        # the turn output audio is attributed to, so the framework can tell one apart from the next
        self._assistant_turn_id: str | None = None
        # calls this connection delegated; only their results mean anything to the backend
        self._delegated_calls: set[str] = set()
        self._tools_ignored = False
        # everything the model has been told, and the baseline a context sync diffs against
        self._remote_chat_ctx = llm.ChatContext.empty()

        self._bstream = utils.audio.AudioByteStream(
            SAMPLE_RATE, NUM_CHANNELS, samples_per_channel=SAMPLE_RATE // 10
        )

        self._main_atask = asyncio.create_task(self._main_task(), name="GPTLiveSession._main")

    # -- outbound --------------------------------------------------------------------------------

    def send_event(self, event: types.ClientEvent | dict[str, Any]) -> None:
        with contextlib.suppress(utils.aio.channel.ChanClosed):
            self._msg_ch.send_nowait(event)

    def _build_delegation(self) -> types.Delegation:
        if self._opts.delegation == "client":
            # there is no backend model to give tools to; the model asks the app in plain text
            if (
                tool_ids := [tool.id for tool in self._tools.flatten()]
            ) and not self._tools_ignored:
                self._tools_ignored = True
                logger.warning(
                    "gpt-live client delegation has no tool channel; answer delegation_created "
                    "with send_delegation_context instead",
                    extra={"tools": tool_ids},
                )
            return types.Delegation(type="client")

        tools = _build_delegation_tools(self._tools.flatten())
        return types.Delegation(
            type="responses",
            responses=types.ResponsesConfig(
                model=self._opts.backend_model,
                instructions=self._opts.backend_instructions,
                tools=tools or None,
                tool_choice=_to_tool_choice(self._opts.tool_choice),
                reasoning=self._opts.reasoning,
                service_tier=self._opts.service_tier,
                max_output_tokens=self._opts.max_output_tokens,
            ),
        )

    def _build_initial_items(self) -> list[types.InitialItem]:
        """The conversation so far as startup history, newest first until the caps are reached."""
        items: list[types.InitialItem] = []
        budget = _MAX_INITIAL_CHARS
        dropped = 0
        for item in reversed(self._remote_chat_ctx.items):
            if (rendered := _render_item(item)) is None:
                continue
            role, text = rendered
            budget -= len(text)
            if budget < 0 or len(items) >= _MAX_INITIAL_ITEMS:
                dropped += 1
                continue
            part = (
                types.OutputTextPart(text=text)
                if role == "assistant"
                else types.InputTextPart(text=text)
            )
            items.append(types.InitialItem(role=role, content=[part]))

        if dropped:
            logger.warning(
                "gpt-live startup history exceeds what a session accepts; dropping the oldest",
                extra={"dropped": dropped, "kept": len(items)},
            )
        items.reverse()
        return items

    def _build_opening(self) -> types.Opening | None:
        """The protected passage, carried until someone has taken a turn."""
        if not (text := self._opts.opening) or self._conversation_started:
            return None
        if len(text) > _MAX_OPENING_CHARS:
            logger.warning(
                "gpt-live opening exceeds the 250 token limit; truncating",
                extra={"chars": len(text)},
            )
            text = text[:_MAX_OPENING_CHARS]
        return types.Opening(text=text)

    def _create_session_update_event(self) -> types.SessionUpdateEvent:
        """The whole configuration, composed fresh for each connection."""
        return types.SessionUpdateEvent(
            event_id=utils.shortuuid("session_update_"),
            session=types.SessionConfig(
                instructions=self._instructions,
                opening=self._build_opening(),
                audio=types.AudioConfig(
                    format=types.AudioFormat(type="audio/pcm", rate=SAMPLE_RATE),
                    output=types.AudioOutput(voice=self._opts.voice),
                ),
                delegation=self._build_delegation(),
                initial_items=self._build_initial_items() or None,
            ),
        )

    def _create_delegation_update_event(self) -> types.SessionUpdateEvent:
        """Session.update replacing only the delegation; the rest is immutable after startup."""
        return types.SessionUpdateEvent(
            event_id=utils.shortuuid("delegation_update_"),
            session=types.SessionConfig(delegation=self._build_delegation()),
        )

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
                    try:
                        await self._run_ws(ws_conn)
                    finally:
                        # what arrives now is history for the next connection, not an append
                        self._initial_config_sent = False
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
        # the mirror is left alone: the new connection is seeded with it
        self._session_started_fut = asyncio.Future()
        if self._assistant_turn_id is not None:
            self._end_assistant_turn(self._assistant_turn_id)
        self._turn_roles.clear()
        self._user_transcripts.clear()
        self._assistant_transcripts.clear()
        self._delegated_calls.clear()
        # a new connection is a new session, so its usage counters restart from zero
        self._usage_total = types.Usage()
        self._session_id = None
        self._opening_failed = False

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
            # instructions, voice and history are immutable once the session starts
            await self._await_config()
            config = self._create_session_update_event()
            self._initial_config_sent = True
            await self._ws_send(ws_conn, config)

            async for msg in self._msg_ch:
                # hold input audio until the server acknowledges the session; cancelled on close
                if (
                    isinstance(msg, types.InputAudioAppendEvent)
                    and not self._session_started_fut.done()
                ):
                    await self._session_started_fut
                await self._ws_send(ws_conn, msg)

            closing = True
            with contextlib.suppress(Exception):
                await self._ws_send(ws_conn, types.SessionCloseEvent())
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
        self, ws_conn: aiohttp.ClientWebSocketResponse, event: types.ClientEvent | dict[str, Any]
    ) -> None:
        # apps subscribe to raw json, which is also what the escape hatch sends
        raw = event if isinstance(event, dict) else event.model_dump(exclude_none=True)
        self.emit("openai_client_event_queued", raw)
        if lk_oai_debug and raw.get("type") != "input_audio.append":
            logger.debug(f">>> {raw}")
        await ws_conn.send_str(json.dumps(raw))

    # -- inbound event dispatch ------------------------------------------------------------------

    def _handle_event(self, event: dict[str, Any]) -> None:
        etype = event.get("type", "")
        if lk_oai_debug and etype != "output_audio.delta":
            logger.debug(f"<<< {event}")

        if etype == "session.started":
            self._handle_session_started(types.SessionStartedEvent.construct(**event))
        elif etype == "session.updated":
            self._handle_session_updated(types.SessionUpdatedEvent.construct(**event))
        elif etype == "output_audio.delta":
            self._handle_output_audio_delta(types.OutputAudioDeltaEvent.construct(**event))
        elif etype == "output_transcript.added":
            self._handle_output_transcript_added(
                types.OutputTranscriptAddedEvent.construct(**event)
            )
        elif etype == "turn.created":
            self._handle_turn_created(types.TurnCreatedEvent.construct(**event))
        elif etype == "turn.delta":
            self._handle_turn_delta(types.TurnDeltaEvent.construct(**event))
        elif etype == "turn.done":
            self._handle_turn_done(types.TurnDoneEvent.construct(**event))
        elif etype == "delegation.created":
            self._handle_delegation_created(types.DelegationCreatedEvent.construct(**event))
        elif etype == "response.output_item.done":
            self._handle_response_output_item_done(
                types.ResponseOutputItemDoneEvent.construct(**event)
            )
        elif etype == "session.opening.started":
            self._handle_opening_started(types.SessionOpeningStartedEvent.construct(**event))
        elif etype == "session.opening.completed":
            self._handle_opening_completed(types.SessionOpeningCompletedEvent.construct(**event))
        elif etype == "session.usage.updated":
            self._handle_session_usage_updated(types.SessionUsageUpdatedEvent.construct(**event))
        elif etype == "session.closed":
            self._handle_session_closed(types.SessionClosedEvent.construct(**event))
        elif etype == "error":
            self._handle_error(types.ErrorEvent.construct(**event).error)
        elif etype == "session.context_window.rolled_over":
            self._handle_context_window_rolled_over(
                types.ContextWindowRolledOverEvent.construct(**event)
            )
        elif etype == "input_transcript.added":
            self._handle_input_transcript_added(types.InputTranscriptAddedEvent.construct(**event))
        elif lk_oai_debug:
            logger.debug(f"unhandled gpt-live event: {etype}")

    # -- model output ----------------------------------------------------------------------------

    def _handle_session_started(self, event: types.SessionStartedEvent) -> None:
        if event.session.id:
            self._session_id = event.session.id
        if not self._session_started_fut.done():
            self._session_started_fut.set_result(None)

    def _handle_session_updated(self, event: types.SessionUpdatedEvent) -> None:
        """A receipt for one sparse update, which nothing waits on yet.

        Startup acknowledges with ``session.started``, and only that releases the audio hold. The
        echoed ``event_id`` is where an update that has to be correlated would be answered.
        """

    def _handle_opening_started(self, event: types.SessionOpeningStartedEvent) -> None:
        logger.debug("gpt-live is speaking its opening; the microphone is muted until it ends")

    def _handle_opening_completed(self, event: types.SessionOpeningCompletedEvent) -> None:
        # no failure means the caller heard it, whether or not any of it was transcribed
        self._conversation_started = self._conversation_started or not self._opening_failed
        logger.debug("gpt-live finished its opening")

    def _handle_context_window_rolled_over(self, event: types.ContextWindowRolledOverEvent) -> None:
        """The service summarised the earlier conversation in place, on its own terms."""
        logger.info(
            "gpt-live compacted the session context", extra={"rollover_id": event.rollover_id}
        )

    def _handle_output_audio_delta(self, event: types.OutputAudioDeltaEvent) -> None:
        data = base64.b64decode(event.audio) if event.audio else b""
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
                    turn_id=self._assistant_turn_id,
                    start_ms=event.start_ms,
                )
            )

    def _handle_output_transcript_added(self, event: types.OutputTranscriptAddedEvent) -> None:
        if not (text := event.item.text):
            return
        # only the open turn may label a fragment: item.id is per fragment, and the framework reads
        # a change of id as a change of turn. before turn.created there is no id, and None says so
        turn_id = self._assistant_turn_id
        if turn_id is not None:
            self._assistant_transcripts[turn_id] = (
                self._assistant_transcripts.get(turn_id, "") + text
            )
        self.emit(
            "transcript_delta",
            llm.DuplexTranscriptDelta(
                turn_id=turn_id,
                text=text,
                start_ms=event.start_ms,
                end_ms=event.end_ms,
            ),
        )

    def _handle_input_transcript_added(self, event: types.InputTranscriptAddedEvent) -> None:
        """The user-side twin of output_transcript.added, which nothing here reads.

        It repeats fragments turn.delta already carries, and only the agent's own speech is paced
        against playout, so its per-fragment timing has no consumer.
        """

    def _end_assistant_turn(self, turn_id: str) -> None:
        """End an assistant turn, recording what it said in the local conversation mirror."""
        if self._assistant_turn_id == turn_id:
            self._assistant_turn_id = None
        if transcript := self._assistant_transcripts.pop(turn_id, ""):
            self._remote_chat_ctx.items.append(
                llm.ChatMessage(id=turn_id, role="assistant", content=[transcript])
            )
        self.emit("turn_ended", llm.DuplexTurnEndedEvent(turn_id=turn_id))

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

    def _handle_turn_created(self, event: types.TurnCreatedEvent) -> None:
        turn = event.turn
        if not turn.id or turn.role not in ("user", "assistant"):
            return
        self._conversation_started = True
        self._turn_roles[turn.id] = turn.role
        if turn.role == "user":
            # a user turn is a projection over transcript fragments, not a state change: the model
            # may well keep speaking over it, so the open assistant turn is left alone and ends on
            # its own turn.done
            self._user_transcripts[turn.id] = turn.transcript or ""
            self.emit("input_speech_started", llm.InputSpeechStartedEvent())
            if self._user_transcripts[turn.id]:
                self._emit_user_transcript(turn.id, is_final=False)
        else:
            self._assistant_turn_id = turn.id
            self.emit("turn_started", llm.DuplexTurnStartedEvent(turn_id=turn.id))

    def _handle_turn_delta(self, event: types.TurnDeltaEvent) -> None:
        # only user transcript here; assistant text comes from output_transcript.added
        if not event.turn_id or not event.delta:
            return
        if self._turn_roles.get(event.turn_id) == "user":
            self._user_transcripts[event.turn_id] = (
                self._user_transcripts.get(event.turn_id, "") + event.delta
            )
            self._emit_user_transcript(event.turn_id, is_final=False)

    def _handle_turn_done(self, event: types.TurnDoneEvent) -> None:
        turn = event.turn
        if not (turn_id := turn.id):
            return
        role = self._turn_roles.pop(turn_id, None) or turn.role
        if role == "user":
            if turn.transcript:
                self._user_transcripts[turn_id] = turn.transcript
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

    # -- delegated work --------------------------------------------------------------------------

    def _handle_delegation_created(self, event: types.DelegationCreatedEvent) -> None:
        # a responses delegation is the backend's; only client-targeted work needs the app
        if event.item.target != "client":
            return
        if not event.item.id:
            logger.warning("gpt-live client delegation has no item id; nothing can answer it")
            return
        text = "\n".join(part.text for part in event.item.content if part.text)
        self.emit("delegation_created", GPTLiveDelegation(id=event.item.id, text=text))

    def _handle_response_output_item_done(self, event: types.ResponseOutputItemDoneEvent) -> None:
        # only the completed item carries all three; the arguments event has no name or call id
        item = event.item
        if item.type != "function_call":
            return
        if not item.call_id or not item.name or item.arguments is None:
            logger.warning(
                "gpt-live dropping function call with missing fields",
                extra={
                    "call_id": item.call_id,
                    "name": item.name,
                    "has_arguments": item.arguments is not None,
                },
            )
            return
        fnc_call = llm.FunctionCall(
            id=item.id or utils.shortuuid("fc_"),
            call_id=item.call_id,
            name=item.name,
            arguments=item.arguments,
        )
        # mirror the call so update_chat_ctx can diff the matching output
        self._remote_chat_ctx.items.append(fnc_call)
        self._delegated_calls.add(item.call_id)
        self.emit("function_call", fnc_call)

    # -- metrics and errors ----------------------------------------------------------------------

    def _handle_session_usage_updated(self, event: types.SessionUsageUpdatedEvent) -> None:
        self._handle_usage(event.usage, event.usage_limit)

    def _handle_session_closed(self, event: types.SessionClosedEvent) -> None:
        self._handle_usage(event.usage)
        if not self._session_closed_fut.done():
            self._session_closed_fut.set_result(None)

    def _handle_usage(self, usage: types.Usage, limit: types.UsageLimit | None = None) -> None:
        # cumulative, so the backend entries stay empty until the model delegates something
        logger.debug(
            "gpt-live reported usage", extra={"usage": usage.model_dump(exclude_none=True)}
        )
        if limit is not None and limit.status:
            logger.warning(
                f"{self._live_model._provider_label} reported a usage limit",
                extra={"status": limit.status, "reset_seconds": limit.reset_seconds},
            )
        # session.usage.updated and session.closed both report usage cumulatively for the whole
        # session, so only the delta is reported for the collectors to sum
        previous, self._usage_total = self._usage_total, usage

        def delta(new: int, old: int) -> int:
            return max(0, new - old)

        if usage.backend_model_usage is not None or usage.audio_duration_ms:
            # the frontend is billed by duration alone; every token here was spent by a model
            # each entry names, so it is reported under that name rather than this session's
            before = {entry.model: entry for entry in previous.backend_model_usage or []}
            for entry in usage.backend_model_usage or []:
                was = before.get(entry.model) or types.BackendModelUsage()
                now_in, was_in = entry.input_tokens_details, was.input_tokens_details
                now_out, was_out = entry.output_tokens_details, was.output_tokens_details
                self.emit(
                    "metrics_collected",
                    LLMMetrics(
                        label=self._live_model.label,
                        request_id=self._session_id or "",
                        timestamp=time.time(),
                        duration=0,
                        ttft=-1,
                        cancelled=False,
                        prompt_tokens=delta(entry.input_tokens, was.input_tokens),
                        prompt_cached_tokens=delta(now_in.cached_tokens, was_in.cached_tokens),
                        cache_creation_tokens=delta(
                            now_in.cache_write_tokens, was_in.cache_write_tokens
                        ),
                        completion_tokens=delta(entry.output_tokens, was.output_tokens),
                        reasoning_tokens=delta(now_out.reasoning_tokens, was_out.reasoning_tokens),
                        total_tokens=delta(entry.total_tokens, was.total_tokens),
                        tokens_per_second=0,
                        metadata=Metadata(
                            model_name=entry.model, model_provider=self._live_model.provider
                        ),
                    ),
                )
            input_tokens = output_tokens = total_tokens = 0
            input_details = RealtimeModelMetrics.InputTokenDetails()
            output_details = RealtimeModelMetrics.OutputTokenDetails()
            session_duration = delta(usage.audio_duration_ms, previous.audio_duration_ms) / 1000
        else:
            new_in, old_in = usage.input_token_details, previous.input_token_details
            new_out, old_out = usage.output_token_details, previous.output_token_details
            input_tokens = delta(usage.input_tokens, previous.input_tokens)
            output_tokens = delta(usage.output_tokens, previous.output_tokens)
            total_tokens = delta(usage.total_tokens, previous.total_tokens)
            input_details = RealtimeModelMetrics.InputTokenDetails(
                audio_tokens=delta(new_in.audio_tokens, old_in.audio_tokens),
                cached_tokens=delta(new_in.cached_tokens, old_in.cached_tokens),
                text_tokens=delta(new_in.text_tokens, old_in.text_tokens),
                image_tokens=delta(new_in.image_tokens, old_in.image_tokens),
            )
            output_details = RealtimeModelMetrics.OutputTokenDetails(
                text_tokens=delta(new_out.text_tokens, old_out.text_tokens),
                audio_tokens=delta(new_out.audio_tokens, old_out.audio_tokens),
                image_tokens=delta(new_out.image_tokens, old_out.image_tokens),
            )
            session_duration = 0.0

        metrics = RealtimeModelMetrics(
            timestamp=time.time(),
            request_id=self._session_id or "",
            ttft=-1,
            duration=0,
            session_duration=session_duration,
            cancelled=False,
            label=self._live_model.label,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            tokens_per_second=0,
            input_token_details=input_details,
            output_token_details=output_details,
            metadata=Metadata(
                model_name=self._live_model.model, model_provider=self._live_model.provider
            ),
        )
        self.emit("metrics_collected", metrics)

    def _handle_error(self, error: types.ErrorBody) -> None:
        if (error.code or "") in _OPENING_FAILURE_CODES:
            self._opening_failed = True
        logger.error(
            f"{self._live_model._provider_label} returned an error",
            extra={"error": error.model_dump(exclude_none=True)},
        )
        recoverable = not _is_fatal_error(error)
        api_error = APIError(
            message=error.message or f"{self._live_model._provider_label} returned an error",
            body=error.model_dump(exclude_none=True),
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
                    types.InputAudioAppendEvent(audio=base64.b64encode(nf.data).decode("utf-8"))
                )

    def append_context(self, text: str, *, channel: types.Channel = "commentary") -> None:
        """Give the model something to know, capped at 500 tokens.

        ``commentary`` is silent; ``speakable`` prompts the model to act on the text now, which is
        how an application makes it talk. Nothing added here reaches ``AgentSession.history``.
        """
        if len(text) > _MAX_CONTEXT_CHARS:
            # a second event would be a second ask to speak, so an oversized one is cut instead
            logger.warning(
                "gpt-live context append exceeds the 500 token limit; truncating",
                extra={"channel": channel, "chars": len(text)},
            )
            text = text[:_MAX_CONTEXT_CHARS]
        self.send_event(
            types.SessionContextAppendEvent(
                event_id=utils.shortuuid("context_"),
                channel=channel,
                content=[types.InputTextPart(text=text)],
            )
        )

    def send_delegation_context(
        self, *, delegation_id: str, text: str, channel: types.Channel = "speakable"
    ) -> None:
        """Answer a :class:`GPTLiveDelegation`, capped at 500 tokens.

        Repeated calls continue the same delegation rather than starting a new one;
        ``commentary`` reports progress silently.
        """
        if len(text) > _MAX_CONTEXT_CHARS:
            logger.warning(
                "gpt-live delegation context exceeds the 500 token limit; truncating",
                extra={"delegation_id": delegation_id, "chars": len(text)},
            )
            text = text[:_MAX_CONTEXT_CHARS]
        self.send_event(
            types.DelegationContextAppendEvent(
                event_id=utils.shortuuid("delegation_context_"),
                delegation_item_id=delegation_id,
                channel=channel,
                content=[types.InputTextPart(text=text)],
            )
        )

    def update_delegation(self, target: types.DelegationTarget) -> None:
        """Move delegated work between the backend model and the application, mid-session."""
        self._opts.delegation = target
        if self._initial_config_sent:
            self.send_event(self._create_delegation_update_event())

    def send_delegation_output(self, *, call_id: str, output: str) -> None:
        """Return the result of a tool call the backend model delegated to the client."""
        self.send_event(
            types.DelegationFunctionCallOutputCreateEvent(
                event_id=utils.shortuuid("fnc_output_"),
                item=types.FunctionCallOutputItem(call_id=call_id, output=output),
            )
        )

    def pause_input(self) -> None:
        """Replace microphone input with silence; the model keeps generating and speaking."""
        self.send_event(types.InputAudioPauseEvent())

    def resume_input(self) -> None:
        self.send_event(types.InputAudioResumeEvent())

    def send_feedback(self, text: str) -> None:
        """Record up to 4096 characters against the session, outside the model's context."""
        if len(text) > _MAX_FEEDBACK_CHARS:
            logger.warning(
                "gpt-live feedback exceeds 4096 characters; truncating",
                extra={"chars": len(text)},
            )
        self.send_event(types.SessionFeedbackEvent(text=text[:_MAX_FEEDBACK_CHARS]))

    async def aclose(self) -> None:
        self._closing = True
        # release the send loop, whether it waits on the config or holds audio
        self._config_delivered.set()
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

    async def _update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        # the framework hands over the whole context, so what is new is recovered by diffing it
        # against what has already gone out, then routed to the event that carries it
        chat_ctx = chat_ctx.copy(exclude_handoff=True, exclude_config_update=True)
        remove_instructions(chat_ctx)
        diff = llm.utils.compute_chat_ctx_diff(self._remote_chat_ctx, chat_ctx)

        if not self._initial_config_sent:
            # still startup history, so an edit to it still applies; the connection seeds from it
            self._remote_chat_ctx = chat_ctx
            return

        # a revision the model cannot apply is the same defect as a deletion it cannot apply
        if stale := (diff.to_remove + [item_id for _, item_id in diff.to_update]):
            logger.error(
                "gpt-live context is append-only; the model keeps what it has been told",
                extra={"item_ids": stale},
            )

        lines: list[str] = []
        for _, item_id in diff.to_create:
            item = chat_ctx.get_by_id(item_id)
            if item is None:
                continue
            if isinstance(item, llm.FunctionCallOutput) and item.call_id in self._delegated_calls:
                # answers a call still open on the backend, so it goes back on its own channel
                self.send_delegation_output(call_id=item.call_id, output=item.output)
            elif isinstance(item, llm.FunctionCall) and item.call_id in self._delegated_calls:
                # a call this connection made is already the backend's; anything else is history
                continue
            elif rendered := _render_item(item):
                lines.append("{}: {}".format(*rendered))

        # as few appends as the limit allows: split up, a transcript reads as loose fragments
        block: list[str] = []
        length = 0
        for line in lines:
            if block and length + len(line) > _MAX_CONTEXT_CHARS:
                self.append_context("\n".join(block))
                block, length = [], 0
            block.append(line)
            length += len(line) + 1
        if block:
            self.append_context("\n".join(block))

        self._remote_chat_ctx = chat_ctx

    def _generate_reply(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        tools: NotGivenOr[list[llm.Tool]] = NOT_GIVEN,
    ) -> asyncio.Future[str | None]:
        # there is no response.create: a speakable instruction asks, and the model may decline
        self.append_context(
            instructions if is_given(instructions) else _SPEAK_NOW, channel="speakable"
        )
        # nothing to wait for: acks echo no event id to correlate against and name no turn
        fut: asyncio.Future[str | None] = asyncio.get_running_loop().create_future()
        fut.set_result(None)
        return fut

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
