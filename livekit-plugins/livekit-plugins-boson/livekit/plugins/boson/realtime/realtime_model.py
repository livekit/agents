"""Realtime model for the Boson (Higgs) speech-to-speech API.

Boson speaks a dialect of the OpenAI Realtime protocol, so this builds on the
OpenAI plugin and overrides where the two differ. The differences that reach
callers are:

- ``session.update`` replaces the whole session rather than merging into it, so
  every update resends the full configuration.
- ``instructions`` on ``generate_reply()`` applies to that turn only. Per-response
  ``tools``/``tool_choice`` are accepted by the server and ignored, so they are
  scoped at the session level instead.
- ``conversation.item.create`` is a plain insert; it never triggers a reply.
- The server does not persist sessions. Each connection starts empty, and a
  reconnect replays the chat context from the client.
- Consecutive same-role speech merges into a single conversation item, which is
  re-sent with its accumulated transcript as it grows.
- Output is single-modality: ``["text"]`` or ``["audio"]``, never both.

See the package README for configuration examples.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
from dataclasses import dataclass, replace
from typing import Any, Literal, cast
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import aiohttp
from openai.types.realtime import (
    ConversationItemAdded,
    ConversationItemCreateEvent,
    ConversationItemDeleteEvent,
    ConversationItemInputAudioTranscriptionCompletedEvent,
    InputAudioBufferSpeechStartedEvent,
    InputAudioBufferSpeechStoppedEvent,
    RealtimeErrorEvent,
    ResponseAudioDeltaEvent,
    ResponseAudioDoneEvent,
    ResponseContentPartAddedEvent,
    ResponseCreatedEvent,
    ResponseDoneEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
)
from pydantic import BaseModel, ConfigDict

from livekit import rtc
from livekit.agents import APIConnectionError, APIError, llm, utils
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import is_given
from livekit.plugins.openai.realtime import realtime_model as openai_rt
from livekit.plugins.openai.realtime.utils import (
    calculate_confidence_from_logprobs,
    openai_item_to_livekit_item,
)

from ..log import logger

SAMPLE_RATE = openai_rt.SAMPLE_RATE

# The hosted endpoint, as documented at
# https://docs.boson.ai/models/higgs-realtime/overview
DEFAULT_URL = "wss://api.boson.ai/v1/realtime"

_DEFAULT_TURN_DETECTION = {
    "type": "server_vad",
    "create_response": True,
    "interrupt_response": True,
    "prefix_padding_ms": 300,
    "silence_duration_ms": 500,
    "threshold": 0.55,
}

# Server close codes that a reconnect cannot fix:
# 3000 = invalid API key / ephemeral key.
# 4429 = billing entitlement refused (insufficient_quota / monthly_cap_reached /
# contract_ended / no_billing_account); permanent for the account, not a
# transient rate limit.
_NON_RETRYABLE_CLOSE_CODES = frozenset({3000, 4429})

# The same refusals as an `error` event, which the server sends just before it
# closes with 4429. Reconnecting cannot clear any of them, so they end the
# session rather than being reported as recoverable. Kept separate from the base
# class's own fatal set, which would catch only `insufficient_quota` here: it
# reads `code` in preference to `type`, and three of these four codes are Boson's
# own. `type` is `insufficient_quota` for all of them.
_BOSON_FATAL_ERROR_CODES = frozenset(
    {"insufficient_quota", "monthly_cap_reached", "contract_ended", "no_billing_account"}
)
_BOSON_FATAL_ERROR_TYPES = frozenset({"insufficient_quota"})

# Error codes/types the server returns for expected client/server races that do
# not indicate a real problem (e.g. response.cancel arriving after the response
# already finished). These must not surface as user-facing recoverable errors,
# mirroring the OpenAI base's own "Cancellation failed" message swallow.
_BOSON_NONFATAL_ERROR_CODES = frozenset({"response_not_active", "response_id_mismatch"})
_BOSON_NONFATAL_ERROR_TYPES = frozenset({"voice_output_task_ongoing", "invalid_previous_item_id"})

# Rejections that say the conversation is out of step rather than that one item
# was refused: update_chat_ctx() re-raises these. Every other rejection is left
# to the base, which gathers them and logs a warning, on the grounds that one
# rejected item is not worth failing the turn over.
_CHAT_CTX_ESCALATING_ERROR_TYPES = frozenset({"invalid_previous_item_id"})

# Roles the server's conversation store cannot hold; see
# _livekit_item_to_boson_item and _create_update_chat_ctx_events.
_UNSUPPORTED_ITEM_ROLES = frozenset({"system", "developer"})


@dataclass
class _BosonOptions:
    url: str
    api_key: str | None
    model: str
    voice: str
    instructions: str
    temperature: NotGivenOr[float]
    max_output_tokens: int | Literal["inf"]
    tool_choice: llm.ToolChoice | None
    speed: float
    turn_detection: dict[str, Any] | None
    input_audio_transcription: NotGivenOr[dict[str, Any]]
    noise_reduction: dict[str, Any] | None
    output_modalities: list[Literal["text", "audio"]]
    truncation: Literal["auto", "disabled"]


class RealtimeModel(openai_rt.RealtimeModel):
    """A speech-to-speech model served by the Boson realtime API.

    Pass an instance to ``AgentSession(llm=...)``. Each session it opens is a
    fresh server-side conversation; the model itself holds only configuration.

    Example:
        ```python
        from livekit.agents import Agent, AgentSession
        from livekit.plugins import boson

        # Reads BOSON_API_KEY from the environment and connects to the hosted API.
        session = AgentSession(
            llm=boson.realtime.RealtimeModel(
                instructions="You are a helpful voice assistant.",
                input_audio_transcription_model="higgs-stt-3.1",
            )
        )
        await session.start(agent=Agent(instructions="..."), room=ctx.room)
        ```
    """

    def __init__(
        self,
        *,
        url: str = DEFAULT_URL,
        api_key: NotGivenOr[str | None] = NOT_GIVEN,
        model: str = "higgs-realtime",
        voice: str = "default",
        instructions: str = "You are a helpful AI assistant",
        output_modalities: list[Literal["text", "audio"]] | None = None,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        max_output_tokens: int | Literal["inf"] = "inf",
        tool_choice: llm.ToolChoice | None = "auto",
        speed: float = 1.0,
        turn_detection: NotGivenOr[dict[str, Any] | None | Literal[False]] = NOT_GIVEN,
        input_audio_transcription: NotGivenOr[dict[str, Any] | None] = NOT_GIVEN,
        input_audio_transcription_model: str = "",
        input_audio_transcription_language: str | None = None,
        input_audio_noise_reduction: NotGivenOr[str | dict[str, Any] | None] = NOT_GIVEN,
        truncation: Literal["auto", "disabled"] = "auto",
        query_params: dict[str, str] | None = None,
        http_session: aiohttp.ClientSession | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        """Configure a Boson realtime model.

        Args:
            url: WebSocket endpoint. Defaults to the hosted API. ``http`` and
                ``https`` are accepted and rewritten to ``ws``/``wss``; the path
                is passed through unchanged.
            api_key: Bearer token for the endpoint. Omit it to read
                ``BOSON_API_KEY`` from the environment, which raises
                ``ValueError`` if that is unset too. Pass ``None`` explicitly to
                send no ``Authorization`` header at all, for a local dev server
                that does not authenticate.
            model: Server-side model name.
            voice: Output voice name.
            instructions: System prompt for the session. Unlike OpenAI, the server
                replaces the whole prompt on every session update, so this is
                resent with each one; ``generate_reply(instructions=...)``
                replaces it for a single turn.
            output_modalities: Exactly one of ``["text"]`` or ``["audio"]``.
                ``None`` selects ``["audio"]``. Mixed and empty lists raise
                ``ValueError`` -- the server has no combined mode.
            temperature: Optional sampling temperature. Omit it to use the
                server default.
            max_output_tokens: Cap on tokens per response, or ``"inf"``.
            tool_choice: How the model picks tools (``"auto"``, ``"none"``,
                ``"required"``, or a specific function). Applies to the session:
                the server ignores per-response tool settings.
            speed: Playback rate for synthesized audio; ``1.0`` is unmodified.
            turn_detection: Server VAD settings. Omit for the default
                (``server_vad``, 500 ms silence, threshold 0.55); pass a dict to
                tune it; pass ``None`` or ``False`` to turn server VAD off, which
                leaves the client responsible for committing turns.
            input_audio_transcription: Raw transcription config, for options the
                two arguments below do not cover. A ``prompt`` key is dropped --
                it is not part of the supported wire config.
            input_audio_transcription_model: ASR model for user transcripts, e.g.
                ``"higgs-stt-3.1"``. Transcript events are emitted **only** when
                this is non-empty; leaving it unset still runs ASR server-side for
                the model's own use, but sends nothing back.
            input_audio_transcription_language: ASR language hint, e.g.
                ``"english"``.
            input_audio_noise_reduction: ``"near_field"``, ``"far_field"``, or a
                dict. ``None`` disables it.
            truncation: How the server trims context that no longer fits.
            query_params: Extra query parameters to merge into ``url``.
            http_session: ``aiohttp`` session to connect on. One is created and
                owned by the model if omitted.
            conn_options: Connect timeout and retry policy. Retries stop early on
                close codes a reconnect cannot fix -- a rejected key (3000) or a
                refused billing entitlement (4429).

        Raises:
            ValueError: If ``output_modalities`` is not exactly one of ``["text"]``
                or ``["audio"]``, or if no API key was given and ``BOSON_API_KEY``
                is unset.
        """
        resolved_api_key: str | None
        if is_given(api_key):
            # An explicit None means "this endpoint takes no auth", so it is
            # honored as given rather than falling back to the environment.
            resolved_api_key = api_key
        else:
            # An empty BOSON_API_KEY counts as unset. Carrying it through would
            # connect with no credential and surface as a close code far from
            # the cause.
            resolved_api_key = os.environ.get("BOSON_API_KEY") or None
            if resolved_api_key is None:
                raise ValueError(
                    "The api_key client option must be set either by passing api_key "
                    "to the client or by setting the BOSON_API_KEY environment variable. "
                    "Pass api_key=None explicitly for a local server without auth."
                )

        resolved_output_modalities = _resolve_output_modalities(output_modalities)
        turn_detection_config = (
            dict(_DEFAULT_TURN_DETECTION)
            if not is_given(turn_detection)
            else _copy_dict_or_none(turn_detection)
        )
        input_audio_transcription_config = _build_input_audio_transcription(
            input_audio_transcription=input_audio_transcription,
            model=input_audio_transcription_model,
            language=input_audio_transcription_language,
        )
        noise_reduction_config = _build_noise_reduction(input_audio_noise_reduction)

        # Initialize the LiveKit/OpenAI realtime runtime, then override the pieces
        # that are Boson-specific. The OpenAI base opts are still used by inherited
        # audio, response, metrics, and function-call plumbing.
        super().__init__(
            base_url=url,
            model=model,
            voice=voice,
            modalities=list(resolved_output_modalities),
            tool_choice=tool_choice,
            input_audio_transcription=None,
            turn_detection=None,
            api_key=resolved_api_key or "boson",
            http_session=http_session,
            max_session_duration=None,
            conn_options=conn_options,
            speed=speed,
        )

        self._provider_label = "Boson Realtime API"
        self._boson_opts = _BosonOptions(
            url=_normalize_ws_url(url, query_params or {}),
            api_key=resolved_api_key,
            model=model,
            voice=voice,
            instructions=instructions,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            tool_choice=tool_choice,
            speed=speed,
            turn_detection=turn_detection_config,
            input_audio_transcription=input_audio_transcription_config,
            noise_reduction=noise_reduction_config,
            output_modalities=resolved_output_modalities,
            truncation=truncation,
        )
        self._capabilities.turn_detection = turn_detection_config is not None
        # Whether the server runs VAD is fixed for the model at construction, via
        # the `turn_detection` argument. The framework's per-session override —
        # session(turn_detection_disabled=True), which asks a model configured
        # with server VAD to hand turn-taking back to the client for one session
        # — has no tested path here, so it is declined rather than half-honored.
        self._capabilities.can_disable_turn_detection = False
        self._capabilities.user_transcription = _input_audio_transcription_enabled(
            input_audio_transcription_config
        )
        # The server treats conversation.item.create as a pure insert and never
        # auto-generates a response for it, so the framework must send
        # response.create after posting tool outputs, like OpenAI.
        self._capabilities.auto_tool_reply_generation = False
        self._capabilities.audio_output = "audio" in resolved_output_modalities
        # mutable_chat_context stays True (base default): the server preserves
        # client-supplied item ids, so items are addressable for the base
        # diff/create/delete chat-context synchronization.
        #
        # The server applies a per-response `instructions` to that turn alone,
        # but not per-response `tools`/`tool_choice` — those are accepted and
        # ignored. Override the OpenAI base's per_response_tool_choice=True so
        # the framework scopes them at the session level around
        # generate_reply() instead of putting them in response.create, where
        # they would be silently dropped.
        self._capabilities.per_response_tool_choice = False

    @property
    def model(self) -> str:
        """The configured server-side model name."""
        return self._boson_opts.model

    @property
    def provider(self) -> str:
        """Host this model connects to, used to label metrics and traces."""
        return urlparse(self._boson_opts.url).netloc

    def session(self, *, turn_detection_disabled: bool = False) -> RealtimeSession:
        """Open a new session against this model.

        Args:
            turn_detection_disabled: Ignored. Whether the server runs VAD is fixed
                for the model by ``turn_detection``; this plugin reports
                ``can_disable_turn_detection=False``, so the framework never asks
                for a per-session override.

        Returns:
            A session that connects on first use.
        """
        session = RealtimeSession(self)
        self._sessions.add(session)
        return session

    def update_options(
        self,
        *,
        voice: NotGivenOr[str] = NOT_GIVEN,
        turn_detection: NotGivenOr[Any | None] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
        input_audio_transcription: NotGivenOr[Any | None] = NOT_GIVEN,
        input_audio_noise_reduction: NotGivenOr[Any | None] = NOT_GIVEN,
        max_response_output_tokens: NotGivenOr[int | Literal["inf"] | None] = NOT_GIVEN,
        tracing: NotGivenOr[Any | None] = NOT_GIVEN,
        truncation: NotGivenOr[Any | None] = NOT_GIVEN,
        reasoning: NotGivenOr[Any | None] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        max_output_tokens: NotGivenOr[int | Literal["inf"] | None] = NOT_GIVEN,
    ) -> None:
        """Change the defaults new sessions start from.

        Sessions already open keep their own copy and are unaffected. Omitted
        arguments are left alone.

        Args:
            voice: Output voice name.
            turn_detection: Server VAD settings; ``None`` or ``False`` turns
                server VAD off.
            tool_choice: How the model picks tools.
            speed: Playback rate for synthesized audio.
            input_audio_transcription: Raw transcription config.
            input_audio_noise_reduction: Noise reduction type or dict.
            max_response_output_tokens: Cap on tokens per response. Alias of
                ``max_output_tokens``, kept for the base class signature; if both
                are given, ``max_output_tokens`` wins.
            tracing: Ignored -- not supported by this API.
            truncation: How the server trims context that no longer fits.
            reasoning: Ignored -- not supported by this API.
            temperature: Sampling temperature.
            max_output_tokens: Cap on tokens per response, or ``"inf"``.
        """
        _ = (tracing, reasoning)
        next_max_output_tokens = (
            max_output_tokens if is_given(max_output_tokens) else max_response_output_tokens
        )

        if is_given(voice):
            self._boson_opts.voice = voice
            self._opts.voice = voice
        if is_given(tool_choice):
            self._boson_opts.tool_choice = tool_choice
            self._opts.tool_choice = tool_choice
        if is_given(temperature):
            self._boson_opts.temperature = temperature
        if is_given(next_max_output_tokens) and next_max_output_tokens is not None:
            self._boson_opts.max_output_tokens = next_max_output_tokens
            self._opts.max_response_output_tokens = next_max_output_tokens
        if is_given(speed):
            self._boson_opts.speed = speed
            self._opts.speed = speed
        if is_given(turn_detection):
            self._boson_opts.turn_detection = _copy_dict_or_none(turn_detection)
            self._capabilities.turn_detection = self._boson_opts.turn_detection is not None
        if is_given(input_audio_transcription):
            self._boson_opts.input_audio_transcription = _normalize_input_audio_transcription(
                input_audio_transcription
            )
            self._capabilities.user_transcription = _input_audio_transcription_enabled(
                self._boson_opts.input_audio_transcription
            )
        if is_given(input_audio_noise_reduction):
            self._boson_opts.noise_reduction = _build_noise_reduction(input_audio_noise_reduction)
        # Unlike turn_detection/noise_reduction, `truncation` does not accept
        # `null` on the wire: sending one is rejected hard enough to end the
        # whole session, not just the one request. Guard against `None` the
        # same way max_output_tokens guards its own non-nullable field.
        if is_given(truncation) and truncation is not None:
            self._boson_opts.truncation = truncation

        for session in self._sessions:
            boson_session = cast(RealtimeSession, session)
            boson_session.update_options(
                tool_choice=tool_choice,
                voice=voice,
                temperature=temperature,
                max_response_output_tokens=next_max_output_tokens,
                max_output_tokens=max_output_tokens,
                speed=speed,
                turn_detection=turn_detection,
                input_audio_transcription=input_audio_transcription,
                input_audio_noise_reduction=input_audio_noise_reduction,
                truncation=truncation,
            )


class RealtimeSession(openai_rt.RealtimeSession):
    """One conversation with the Boson realtime API.

    Created by `RealtimeModel.session()` rather than directly. It takes its own
    copy of the model's options, so `update_options` here affects this
    conversation alone.

    The server keeps no session state between connections: a dropped socket is
    reconnected and the chat context is replayed from the client as text. An
    audio turn whose transcript had not arrived by then has nothing to replay
    and is lost -- at most the last untranscribed turn.
    """

    def __init__(self, realtime_model: RealtimeModel) -> None:
        """Prepare a session. Connecting happens on first use.

        Args:
            realtime_model: The model to take configuration from.
        """
        self._boson_model = realtime_model
        self._boson_opts = replace(realtime_model._boson_opts)
        self._closed = False
        self._suppress_next_response_cancel = False
        self._video_unsupported_warned = False
        self._system_item_unsupported_warned = False
        self._unsupported_tool_type_warned = False
        self._per_response_tools_warned = False
        self._current_response_id: str | None = None
        # Set by _handle_error on invalid_previous_item_id, read (and cleared)
        # by update_chat_ctx() right after its base call returns; see there.
        # _chat_ctx_sync_lock keeps that clear/read pair atomic against a
        # concurrent update_chat_ctx() — the base's own lock is taken further
        # in and so does not cover it.
        self._chat_ctx_sync_error: llm.RealtimeError | None = None
        self._chat_ctx_sync_lock = asyncio.Lock()
        # Server-assigned session id (from session.created), kept for logging.
        # The server does not persist sessions: every connection is a fresh
        # session, so reconnection replays the local chat context instead.
        self._session_id: str | None = None
        # Set when the server announces it is ending the session on purpose
        # (session.idle_timeout / session.max_duration_reached); the following
        # close must not trigger a reconnect, which would restart the session
        # the server just ended.
        self._server_terminal_reason: str | None = None
        super().__init__(realtime_model)
        # The base leaves its own copy of the instructions None until
        # update_instructions() is first called, and only prefixes a
        # per-response `instructions` with it when it is truthy. Since the
        # server replaces the whole system prompt with whatever response.create
        # carries, an un-set value there would drop the configured instructions
        # for that turn. Seed it from the session config, and keep the two in
        # sync in update_instructions. Assigned after super().__init__, which
        # sets it to None — and which has already sent the first session.update
        # by then, hence _build_session_update_event reading _boson_opts rather
        # than this.
        self._instructions = self._boson_opts.instructions
        # The base recv loop dispatches OpenAI event types to _handle_* methods
        # and re-emits every raw event on this hook; Boson-specific events are
        # handled off it instead of forking the whole dispatch.
        self.on("openai_server_event_received", self._handle_boson_server_event)
        self.on("session_reconnected", self._on_session_reconnected)

    def send_event(self, event: Any) -> None:
        """Queue a raw client event for the socket.

        Dropped silently once the session is closed, so teardown races do not
        raise.

        Args:
            event: A dict, or a pydantic model to serialize by alias.
        """
        if self._closed or self._msg_ch.closed:
            return
        if isinstance(event, BaseModel):
            event = event.model_dump(by_alias=True, exclude_unset=True, exclude_defaults=False)
        with contextlib.suppress(utils.aio.ChanClosed):
            self._msg_ch.send_nowait(event)

    @property
    def session_id(self) -> str | None:
        """Server-assigned id for the current connection, for correlating logs.

        ``None`` before ``session.created`` arrives. A reconnect yields a new id,
        since the server treats every connection as a new session.
        """
        return self._session_id

    async def _main_task(self) -> None:
        # The base task owns connect/retry/reconnect (including the chat-context
        # replay via _create_update_chat_ctx_events). On terminal failure it
        # leaves pending response futures to their own timeouts; fail them and
        # stop accepting events right away instead.
        try:
            await super()._main_task()
        except Exception as exc:
            error = exc if isinstance(exc, llm.RealtimeError) else llm.RealtimeError(str(exc))
            self._fail_response_created_futures(error)
            self._close_current_generation("Boson realtime session failed")
            self._closed = True
            self._msg_ch.close()
            raise

    async def _create_ws_conn(self) -> aiohttp.ClientWebSocketResponse:
        headers = {"User-Agent": "LiveKit Agents Boson plugin"}
        if self._boson_opts.api_key:
            headers["Authorization"] = f"Bearer {self._boson_opts.api_key}"

        t0 = time.perf_counter()
        try:
            ws = await asyncio.wait_for(
                self._boson_model._ensure_http_session().ws_connect(
                    url=self._boson_opts.url,
                    headers=headers,
                ),
                self._opts.conn_options.timeout,
            )
            self._report_connection_acquired(time.perf_counter() - t0)
            return ws
        except aiohttp.ClientError as exc:
            raise APIConnectionError("Boson realtime client connection error") from exc
        except asyncio.TimeoutError as exc:
            raise APIConnectionError("Boson realtime connection timed out") from exc

    async def _run_ws(self, ws_conn: aiohttp.ClientWebSocketResponse) -> None:
        try:
            await super()._run_ws(ws_conn)
        except APIConnectionError as exc:
            # The base recv loop raises every unexpected close as retryable;
            # reclassify the ones a reconnect cannot fix.
            if self._server_terminal_reason is not None:
                raise APIConnectionError(
                    f"Boson realtime session ended by server: {self._server_terminal_reason}",
                    retryable=False,
                ) from exc
            close_code = ws_conn.close_code
            if close_code is not None:
                raise APIConnectionError(
                    f"Boson realtime WebSocket closed unexpectedly (close_code={close_code}).",
                    retryable=close_code not in _NON_RETRYABLE_CLOSE_CODES,
                ) from exc
            raise

    def _handle_boson_server_event(self, event: dict[str, Any]) -> None:
        event_type = event.get("type")
        if event_type in ("session.created", "session.updated"):
            # Track the server-assigned id for logging/diagnostics.
            session_obj = event.get("session") or {}
            session_id = session_obj.get("id")
            if isinstance(session_id, str) and session_id:
                self._session_id = session_id
        elif event_type == "session.idle_timeout":
            seconds_idle = event.get("seconds_idle")
            self._server_terminal_reason = f"idle timeout ({seconds_idle}s)"
            logger.info(
                "Boson realtime session idle timeout announced by server",
                extra={"session_id": self._session_id, "seconds_idle": seconds_idle},
            )
        elif event_type == "session.max_duration_reached":
            max_duration_sec = event.get("max_duration_sec")
            self._server_terminal_reason = f"max session duration reached ({max_duration_sec}s)"
            logger.info(
                "Boson realtime session max duration announced by server",
                extra={"session_id": self._session_id, "max_duration_sec": max_duration_sec},
            )

    def _on_session_reconnected(self, _event: llm.RealtimeSessionReconnectedEvent) -> None:
        # Per-connection state the base _reconnect doesn't know about. No
        # response from the old connection can be "current" on the new one.
        self._pushed_duration_s = 0.0
        self._current_response_id = None
        self._suppress_next_response_cancel = False
        logger.info("reconnected to Boson realtime session", extra={"session_id": self._session_id})

    def _handle_conversion_item_added(self, event: ConversationItemAdded) -> None:
        item_id = event.item.id
        if item_id is not None and (remote_item := self._remote_chat_ctx.get(item_id)) is not None:
            # The server merges consecutive same-role speech turns into a single
            # item and re-emits conversation.item.added with the same id and
            # cumulative content. Update the mirrored text in place instead of
            # letting the base insert fail with a warning.
            # Audio-input configs re-add with an empty input_audio part and no
            # transcript — keep whatever transcription has already arrived.
            lk_item = openai_item_to_livekit_item(event.item)
            if (
                isinstance(lk_item, llm.ChatMessage)
                and isinstance(remote_item.item, llm.ChatMessage)
                and lk_item.text_content
            ):
                _set_message_text(remote_item.item, lk_item.text_content)
            if fut := self._item_create_future.pop(item_id, None):
                # done(), not cancelled(): _handle_error may already have failed
                # this very future -- it settles by event id, and the
                # invalid_previous_item_id path reads _item_create_future without
                # removing the entry, so a rejected create stays registered here
                # until update_chat_ctx() clears it. A server that rejects an item
                # and then echoes it anyway would land on a future that is done
                # with an exception, where cancelled() is False and set_result
                # raises InvalidStateError.
                if not fut.done():
                    fut.set_result(None)
            return

        super()._handle_conversion_item_added(event)

    def _handle_conversion_item_input_audio_transcription_completed(
        self, event: ConversationItemInputAudioTranscriptionCompletedEvent
    ) -> None:
        self._clear_transcript_accumulator(event.item_id, event.content_index or 0)
        confidence = calculate_confidence_from_logprobs(event.logprobs)

        if remote_item := self._remote_chat_ctx.get(event.item_id):
            assert isinstance(remote_item.item, llm.ChatMessage)
            if event.transcript:
                _set_message_text(remote_item.item, event.transcript)
            remote_item.item.transcript_confidence = confidence

        self.emit(
            "input_audio_transcription_completed",
            llm.InputTranscriptionCompleted(
                item_id=event.item_id,
                transcript=event.transcript,
                is_final=True,
                confidence=confidence,
                # Read rather than popped as the base does: a merged item is
                # re-emitted with its accumulated text, and every one of those
                # revisions has to carry the same turn start -- popping would
                # leave all but the first with None, which is exactly the case
                # this timestamp exists for. The entry goes when the item does;
                # the base clears it in _handle_conversion_item_deleted.
                turn_started_at=self._input_speech_started_at.get(event.item_id),
            ),
        )

    def _create_session_update_event(self) -> dict[str, Any]:
        return self._build_session_update_event("session_update_")

    def _create_tools_update_event(self, tools: list[llm.Tool]) -> dict[str, Any]:
        # The server treats session.update as a full replace (not OpenAI's
        # partial merge), so a tools-only update must carry the whole config.
        return self._build_session_update_event("tools_update_", tools=tools)

    def _warn_unsupported_tool_type_once(self, dropped: int) -> None:
        if self._unsupported_tool_type_warned:
            return
        self._unsupported_tool_type_warned = True
        logger.warning(
            "%d tool(s) were dropped from the Boson realtime session: only function tools "
            "are supported on the wire. The model will not see them.",
            dropped,
        )

    def _warn_unsupported_item_role_once(self) -> None:
        if self._system_item_unsupported_warned:
            return
        self._system_item_unsupported_warned = True
        logger.warning(
            "Boson realtime API only stores assistant/user conversation items; "
            "system/developer chat items are dropped from update_chat_ctx() syncing. "
            "Use session-level instructions for persistent directives."
        )

    def _create_update_chat_ctx_events(
        self, chat_ctx: llm.ChatContext
    ) -> list[ConversationItemCreateEvent | ConversationItemDeleteEvent]:
        # The base diff (used both by update_chat_ctx and by _reconnect's
        # chat-context replay) produces GA-shaped creates; the server stores
        # items in the Boson shape (a single text content part, type "text" for
        # assistant) and has no "root" previous_item_id sentinel. Rebuild each
        # create with the Boson payload. The server preserves client-supplied
        # item ids, so the rest of the base machinery — echo correlation via
        # _item_create_future/_item_delete_future and the _remote_chat_ctx
        # diff — works unchanged.
        #
        # Diff against a text-only mirror of the context: the server stores a
        # single text part per message (audio is represented by its transcript,
        # images are unsupported), and the base GA converter must never see
        # audio frames (rtc.combine_audio_frames raises on empty ones). A
        # message with no text keeps an empty mirror so the base diff retains
        # it when it already exists remotely (e.g. an audio item whose
        # transcription is still pending) instead of deleting it, and filters
        # it out otherwise.
        #
        # Items with a role the server cannot store are dropped here, before
        # the diff runs, not further down while walking its output: an
        # unsupported item never reaches _remote_chat_ctx, so leaving it in
        # the diff input makes every sync see it as "not remote yet". At the
        # head of the context that reads as an insert-before-everything and
        # triggers the full rebuild below — repeatedly, on every single sync,
        # for an item that is never sent either way.
        sanitized: list[llm.ChatItem] = []
        for item in chat_ctx.items:
            if item.type == "message":
                if item.role in _UNSUPPORTED_ITEM_ROLES:
                    self._warn_unsupported_item_role_once()
                    continue
                text = _text_from_content(item.content)
                sanitized.append(
                    llm.ChatMessage(id=item.id, role=item.role, content=[text] if text else [])
                )
            else:
                sanitized.append(item)
        boson_ctx = llm.ChatContext(sanitized)

        base_events = super()._create_update_chat_ctx_events(boson_ctx)
        remote_items = self._remote_chat_ctx.to_chat_ctx().items
        if remote_items and any(
            isinstance(ev, ConversationItemCreateEvent) and ev.previous_item_id == "root"
            for ev in base_events
        ):
            # "root" is the base's marker for the item at the head of the target
            # context, not for one that is new to the server. Two things reach it:
            # a genuine insert ahead of turns the server still has (a caller
            # prepending a summary, say), and a plain text change to the item
            # already at the head, which the base expresses as a delete and a
            # create under the same id.
            #
            # Neither can be sent incrementally. The server has no insert-at-head
            # primitive -- previous_item_id=None always means append-at-tail -- so
            # mapping "root" -> None while the server holds anything would put the
            # item after those turns instead of before them. That is as wrong for
            # the recreated head item, which the delete just took out of its
            # position, as it is for a new one. Rebuild in the target order rather
            # than guess at a partial reorder.
            return self._rebuild_chat_ctx_events(boson_ctx, remote_items)

        events: list[ConversationItemCreateEvent | ConversationItemDeleteEvent] = []
        # Safety net: a create the conversion still cannot express is not sent;
        # remap a previous_item_id pointing at it to its own predecessor.
        dropped: dict[str, str | None] = {}
        deleted_ids: set[str] = set()
        # Ids the server is losing outright because a delete went out and the
        # create meant to follow it was one of the dropped ones.
        unrecreatable_updates: list[str] = []
        for ev in base_events:
            if not isinstance(ev, ConversationItemCreateEvent):
                deleted_ids.add(ev.item_id)
                events.append(ev)
                continue
            assert ev.item.id is not None
            previous_item_id = ev.previous_item_id
            if previous_item_id == "root":
                # The remote context is empty (checked above), so appending at
                # the tail is equivalent to inserting at the head.
                previous_item_id = None
            elif previous_item_id is not None and previous_item_id in dropped:
                previous_item_id = dropped[previous_item_id]
            chat_item = boson_ctx.get_by_id(ev.item.id)
            payload = _livekit_item_to_boson_item(chat_item) if chat_item is not None else None
            if payload is None:
                # The server skips items without text content (or a role it
                # can't store) instead of storing them, so their
                # conversation.item.added echo would never resolve the
                # create future.
                dropped[ev.item.id] = previous_item_id
                if ev.item.id in deleted_ids:
                    unrecreatable_updates.append(ev.item.id)
                continue
            events.append(
                _BosonConversationItemCreateEvent(
                    type="conversation.item.create",
                    event_id=ev.event_id or utils.shortuuid("chat_ctx_create_"),
                    previous_item_id=previous_item_id,
                    item=_BosonConversationItem(**payload),
                )
            )
        if unrecreatable_updates:
            # The base states a content change -- and a reorder -- as a delete
            # followed by a create under one id. When the create is one this
            # client cannot express, because the item's new content has no text
            # left, the delete stands alone: the server drops the turn instead
            # of updating it.
            #
            # The delete is kept rather than suppressed. The caller asked for
            # that text to leave the context, and holding on to the server's
            # copy would keep feeding the model something they took out. But the
            # outcome is bigger than an emptied item, so say so: the turn leaves
            # the model's view entirely, and the remote mirror with it, until a
            # later sync gives it text again.
            logger.warning(
                "%d item(s) the server was holding are being deleted rather than updated; "
                "their new content has no text this client can send. Each returns on a "
                "later sync once it has text again",
                len(unrecreatable_updates),
                extra={"item_ids": unrecreatable_updates},
            )
        return events

    def _rebuild_chat_ctx_events(
        self, boson_ctx: llm.ChatContext, remote_items: list[llm.ChatItem]
    ) -> list[ConversationItemCreateEvent | ConversationItemDeleteEvent]:
        """Delete the server's conversation and send it back in the target order.

        The cost is any item this client cannot put back. A user turn whose
        transcript has not arrived has no text to resend, and no audio is kept
        client-side, so it is deleted with the rest and not recreated: the server
        loses it for as long as it stays untranscribed. It is not lost for good --
        the delete drops it from the remote mirror too, so a later
        update_chat_ctx() recreates it once its text exists, at the tail -- but
        until then the model answers without it. Warned about rather than done
        quietly, since this branch is strictly more destructive than the
        incremental diff it stands in for.
        """
        remote_ids = {remote_item.id for remote_item in remote_items}
        events: list[ConversationItemCreateEvent | ConversationItemDeleteEvent] = [
            ConversationItemDeleteEvent(
                type="conversation.item.delete",
                item_id=remote_item.id,
                event_id=utils.shortuuid("chat_ctx_delete_"),
            )
            for remote_item in remote_items
        ]
        # Wanted by the target context and held by the server, but with nothing
        # this client can send to put it back.
        unrecreatable: list[str] = []
        previous_item_id: str | None = None
        for item in boson_ctx.items:
            payload = _livekit_item_to_boson_item(item)
            if payload is None:
                # Not addressable on the wire (unsupported role, no text,
                # etc.); skip without advancing previous_item_id to it.
                if item.id in remote_ids:
                    unrecreatable.append(item.id)
                continue
            events.append(
                _BosonConversationItemCreateEvent(
                    type="conversation.item.create",
                    event_id=utils.shortuuid("chat_ctx_create_"),
                    previous_item_id=previous_item_id,
                    item=_BosonConversationItem(**payload),
                )
            )
            previous_item_id = item.id
        if unrecreatable:
            logger.warning(
                "reordering the conversation dropped %d item(s) the server was holding "
                "that this client cannot resend; each returns on a later sync once its "
                "transcript arrives",
                len(unrecreatable),
                extra={"item_ids": unrecreatable},
            )
        return events

    def _build_session_update_event(
        self, event_prefix: str, tools: list[llm.Tool] | None = None
    ) -> dict[str, Any]:
        audio_input: dict[str, Any] = {
            "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
            "turn_detection": self._boson_opts.turn_detection,
        }
        if is_given(self._boson_opts.input_audio_transcription):
            audio_input["transcription"] = self._boson_opts.input_audio_transcription
        if self._boson_opts.noise_reduction is not None:
            audio_input["noise_reduction"] = self._boson_opts.noise_reduction

        audio_output: dict[str, Any] = {
            "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
            "voice": self._boson_opts.voice,
            "speed": self._boson_opts.speed,
        }

        payload: dict[str, Any] = {
            "type": "realtime",
            "model": self._boson_opts.model,
            # The session-level value, which update_instructions keeps in sync
            # with the base's own copy. Read from here rather than that copy
            # because the base sends the first session.update from inside its
            # own __init__, before the copy can be seeded.
            "instructions": self._boson_opts.instructions,
            "output_modalities": list(self._boson_opts.output_modalities),
            "audio": {
                "input": audio_input,
                "output": audio_output,
            },
            "tools": self._tools_to_boson_warning_on_drop(
                tools if tools is not None else self._tools.flatten()
            ),
            "tool_choice": _tool_choice_to_boson(self._boson_opts.tool_choice),
            "max_output_tokens": self._boson_opts.max_output_tokens,
            "truncation": self._boson_opts.truncation,
        }
        if is_given(self._boson_opts.temperature):
            payload["temperature"] = self._boson_opts.temperature
        return {
            "type": "session.update",
            "event_id": utils.shortuuid(event_prefix),
            "session": payload,
        }

    def _tools_to_boson_warning_on_drop(self, tools: list[llm.Tool]) -> list[dict[str, Any]]:
        """Convert tools for the wire, warning once if any were dropped.

        _tools_to_boson emits one entry per tool it can express, so a shorter
        result means something was silently left out -- a tool type the wire
        has no shape for. Counting rather than re-testing the types keeps the
        two from drifting apart. Silence here would be the worst outcome: a
        tool the caller registered simply never reaches the model.
        """
        converted = _tools_to_boson(tools)
        if len(converted) < len(tools):
            self._warn_unsupported_tool_type_once(len(tools) - len(converted))
        return converted

    async def update_instructions(self, instructions: str) -> None:
        """Replace the system prompt for the rest of this session.

        Takes effect on the next turn. Because the server replaces the whole
        session on update, this resends the full configuration alongside it.

        Args:
            instructions: The new system prompt.
        """
        # Both copies: _boson_opts feeds the wire, the base's own feeds the
        # per-response prefix in generate_reply. See __init__ on why there are
        # two.
        self._boson_opts.instructions = instructions
        self._instructions = instructions
        self.send_event(self._build_session_update_event("instructions_update_"))

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        """Bring the server's conversation in line with ``chat_ctx``.

        Sends only the difference against what the server is known to hold, as
        item creates and deletes. ``system`` and ``developer`` items are skipped
        -- the server's conversation store has no place for them, and the system
        prompt travels in the session config instead.

        Args:
            chat_ctx: The conversation the server should end up holding.

        Raises:
            llm.RealtimeError: If the server rejects an item, or does not confirm
                the change within five seconds.
        """
        # _chat_ctx_sync_error is a single session-level slot, but the clear
        # and the read below straddle an await. The base serializes its own
        # body on _update_chat_ctx_lock, which it takes *inside* that await —
        # so a second caller can still run the clear while this one is parked
        # in there. Since the base gathers the item futures with
        # return_exceptions=True, that slot is the only channel a per-item
        # failure has: losing it means reporting success for a sync that
        # actually failed. Hold our own lock across the whole pair.
        async with self._chat_ctx_sync_lock:
            self._chat_ctx_sync_error = None
            await super().update_chat_ctx(chat_ctx)
            if (sync_error := self._chat_ctx_sync_error) is not None:
                # _handle_error recorded and raised this early instead of
                # letting the base class's own 5s timeout produce a generic
                # message.
                self._chat_ctx_sync_error = None
                raise sync_error

    def update_options(
        self,
        *,
        tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN,
        voice: NotGivenOr[str] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
        turn_detection: NotGivenOr[Any | None] = NOT_GIVEN,
        input_audio_transcription: NotGivenOr[Any | None] = NOT_GIVEN,
        input_audio_noise_reduction: NotGivenOr[Any | None] = NOT_GIVEN,
        max_response_output_tokens: NotGivenOr[int | Literal["inf"] | None] = NOT_GIVEN,
        tracing: NotGivenOr[Any | None] = NOT_GIVEN,
        truncation: NotGivenOr[Any | None] = NOT_GIVEN,
        reasoning: NotGivenOr[Any | None] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        max_output_tokens: NotGivenOr[int | Literal["inf"] | None] = NOT_GIVEN,
    ) -> None:
        """Change this session's configuration, leaving the model's untouched.

        Omitted arguments are left alone. Because the server replaces the whole
        session on update, one changed field resends everything.

        Args:
            tool_choice: How the model picks tools.
            voice: Output voice name.
            speed: Playback rate for synthesized audio.
            turn_detection: Server VAD settings; ``None`` or ``False`` turns
                server VAD off.
            input_audio_transcription: Raw transcription config.
            input_audio_noise_reduction: Noise reduction type or dict.
            max_response_output_tokens: Cap on tokens per response. Alias of
                ``max_output_tokens``, kept for the base class signature; if both
                are given, ``max_output_tokens`` wins.
            tracing: Ignored -- not supported by this API.
            truncation: How the server trims context that no longer fits.
            reasoning: Ignored -- not supported by this API.
            temperature: Sampling temperature.
            max_output_tokens: Cap on tokens per response, or ``"inf"``.
        """
        _ = (tracing, reasoning)
        next_max_output_tokens = (
            max_output_tokens if is_given(max_output_tokens) else max_response_output_tokens
        )

        if is_given(tool_choice):
            self._boson_opts.tool_choice = tool_choice
            self._opts.tool_choice = tool_choice
        if is_given(voice):
            self._boson_opts.voice = voice
            self._opts.voice = voice
        if is_given(temperature):
            self._boson_opts.temperature = temperature
        if is_given(next_max_output_tokens) and next_max_output_tokens is not None:
            self._boson_opts.max_output_tokens = next_max_output_tokens
            self._opts.max_response_output_tokens = next_max_output_tokens
        if is_given(speed):
            self._boson_opts.speed = speed
            self._opts.speed = speed
        if is_given(turn_detection):
            self._boson_opts.turn_detection = _copy_dict_or_none(turn_detection)
        if is_given(input_audio_transcription):
            self._boson_opts.input_audio_transcription = _normalize_input_audio_transcription(
                input_audio_transcription
            )
        if is_given(input_audio_noise_reduction):
            self._boson_opts.noise_reduction = _build_noise_reduction(input_audio_noise_reduction)
        # Unlike turn_detection/noise_reduction, `truncation` does not accept
        # `null` on the wire: sending one is rejected hard enough to end the
        # whole session, not just the one request. Guard against `None` the
        # same way max_output_tokens guards its own non-nullable field.
        if is_given(truncation) and truncation is not None:
            self._boson_opts.truncation = truncation
        self.send_event(self._build_session_update_event("options_update_"))

    def generate_reply(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        tools: NotGivenOr[list[llm.Tool]] = NOT_GIVEN,
    ) -> asyncio.Future[llm.GenerationCreatedEvent]:
        """Ask the model to reply now, without waiting for the user to speak.

        Args:
            instructions: A system prompt for this turn alone. The server swaps
                it in for the session prompt while the turn still answers from
                the real conversation; the session prompt returns afterwards.
            tool_choice: Not supported per response -- the server accepts and
                ignores it. Passing it logs one warning per session. The
                framework instead scopes tools at the session level around this
                call, so agents do not hit this.
            tools: Not supported per response, as ``tool_choice``.

        Returns:
            A future resolving to the generation event once the server has
            created the response.
        """
        # `instructions` rides in response.create: the server scopes it to that
        # turn, replacing the session prompt for it alone while the turn still
        # answers from the real conversation. The base builds that event,
        # prefixing the session instructions (see __init__ on why they are
        # always set).
        #
        # `tools`/`tool_choice` are not scoped per response by the server — it
        # accepts and ignores them — so they must not be forwarded. The
        # framework already scopes them at the session level around this call
        # (capabilities.per_response_tool_choice is False) and passes them here
        # only when a caller reaches this session directly.
        if is_given(tools) or is_given(tool_choice):
            self._warn_per_response_tools_unsupported_once()
        return super().generate_reply(instructions=instructions)

    def _warn_per_response_tools_unsupported_once(self) -> None:
        if self._per_response_tools_warned:
            return
        self._per_response_tools_warned = True
        logger.warning(
            "Boson realtime API does not apply per-response tools/tool_choice; "
            "they are ignored by generate_reply(). Use update_tools() / "
            "update_options(tool_choice=...) to scope them yourself."
        )

    def push_video(self, frame: rtc.VideoFrame) -> None:
        """Discard a video frame, warning once per session.

        Boson takes no video input. This is on the per-frame path, so an
        unsupported configuration is not worth raising into.

        Args:
            frame: The frame to drop.
        """
        if not self._video_unsupported_warned:
            self._video_unsupported_warned = True
            logger.warning("Boson RealtimeModel does not support video input; frames are ignored.")

    def interrupt(self) -> None:
        """Stop the reply in progress.

        A no-op when nothing is generating. When server VAD already cancelled the
        response itself, the redundant cancel is skipped -- the server answers a
        second one with ``response_not_active``.
        """
        if not self.has_active_generation:
            return
        if self._suppress_next_response_cancel:
            self._suppress_next_response_cancel = False
            logger.debug("Skipping duplicate response.cancel after server-side VAD interruption.")
            return
        event: dict[str, Any] = {
            "type": "response.cancel",
            "event_id": utils.shortuuid("response_cancel_"),
        }
        if self._current_response_id:
            event["response_id"] = self._current_response_id
        self.send_event(event)

    async def aclose(self) -> None:
        """Close the session and its socket.

        Ends any generation in flight and stops further sends. Returns without
        waiting out a retry backoff or a connect already under way.
        """
        self._closing = True
        self._closed = True
        self._close_current_generation("session closed")
        self._msg_ch.close()
        # Cancel instead of the base's await: the main task may be sleeping in a
        # retry backoff or mid-connect, which close should not wait out.
        await utils.aio.cancel_and_wait(self._main_atask)

    def _handle_input_audio_buffer_speech_started(
        self, event: InputAudioBufferSpeechStartedEvent
    ) -> None:
        self._suppress_next_response_cancel = _server_vad_auto_interrupts_response(
            self._boson_opts.turn_detection
        )
        try:
            if event.item_id:
                # setdefault rather than the base's assignment: the server merges
                # consecutive user speech into a single item, so this fires once
                # per fragment under the same id. The turn began at the first of
                # them; assigning would walk the timestamp forward to whichever
                # fragment happened to be last.
                self._input_speech_started_at.setdefault(event.item_id, time.time())
            self.emit("input_speech_started", llm.InputSpeechStartedEvent())
        finally:
            self._suppress_next_response_cancel = False

    def _handle_input_audio_buffer_speech_stopped(
        self, _: InputAudioBufferSpeechStoppedEvent
    ) -> None:
        self._pushed_duration_s = 0.0
        self.emit(
            "input_speech_stopped",
            llm.InputSpeechStoppedEvent(
                user_transcription_enabled=_input_audio_transcription_enabled(
                    self._boson_opts.input_audio_transcription
                )
            ),
        )

    def _is_stale_response_scoped_event(self, response_id: str | None) -> bool:
        """Whether a response-scoped event belongs to a response this session
        has no generation for anymore.

        Two responses can overlap on the wire: a new response's
        response.created may be observed before the previous response's
        terminal event, and trailing events for the older response can keep
        arriving after it. ``_current_response_id`` is updated the moment a
        new response is accepted (see ``_handle_response_created``), so a
        response-scoped event whose id doesn't match it, no matter how late,
        belongs to an already-superseded response and must not touch
        ``_current_generation``.

        What occupies the generation slot decides the rest:

        * a real ``_ResponseGeneration`` — only its own response's events may
          reach the base handlers, per the id check above.
        * the base's discard placeholder (``_DiscardedGeneration``) — let
          everything through. Every base handler no-ops on the placeholder,
          and it is that response's own response.done reaching the base that
          clears it; filtering here would strand the placeholder in the slot.
        * ``None`` — nothing is streaming, so nothing can be attached to
          anything. This must be treated as stale rather than passed through:
          every response-scoped base handler except response.done asserts on
          a missing generation, so a late event let through here raises (the
          base's recv loop catches and logs it) instead of being dropped.
        """
        if isinstance(self._current_generation, openai_rt._DiscardedGeneration):
            return False
        if self._current_generation is None:
            return True
        return response_id is not None and response_id != self._current_response_id

    def _handle_response_created(self, event: ResponseCreatedEvent) -> None:
        client_event_id: str | None = None
        if isinstance(event.response.metadata, dict):
            client_event_id = event.response.metadata.get("client_event_id")
        if client_event_id and client_event_id in self._discarded_event_ids:
            # A response that timed out or was interrupted before the server
            # created it. The base handler cancels it and parks a discard
            # marker in the generation slot so its trailing events are
            # skipped. When a legitimate generation is already streaming,
            # keep it in the slot instead. Its id is never assigned to
            # _current_response_id, so every later event for this discarded
            # response fails _is_stale_response_scoped_event and is dropped —
            # no separate bookkeeping needed.
            active = self._current_generation
            super()._handle_response_created(event)
            if isinstance(active, openai_rt._ResponseGeneration):
                self._current_generation = active
            return

        if self._current_generation is not None:
            self._close_current_generation("new response created before previous response.done")
        self._current_response_id = event.response.id
        super()._handle_response_created(event)

    def _handle_response_output_item_added(self, event: ResponseOutputItemAddedEvent) -> None:
        if self._is_stale_response_scoped_event(event.response_id):
            return
        super()._handle_response_output_item_added(event)

    def _handle_response_content_part_added(self, event: ResponseContentPartAddedEvent) -> None:
        if self._is_stale_response_scoped_event(event.response_id):
            return
        super()._handle_response_content_part_added(event)

    def _handle_response_text_delta(self, event: ResponseTextDeltaEvent) -> None:
        if self._is_stale_response_scoped_event(event.response_id):
            return
        super()._handle_response_text_delta(event)

    def _handle_response_text_done(self, event: ResponseTextDoneEvent) -> None:
        if self._is_stale_response_scoped_event(event.response_id):
            return
        super()._handle_response_text_done(event)

    def _handle_response_audio_transcript_delta(self, event: dict[str, Any]) -> None:
        # Unlike its siblings, the base dispatches this one as a raw dict.
        if self._is_stale_response_scoped_event(event.get("response_id")):
            return
        super()._handle_response_audio_transcript_delta(event)

    def _handle_response_audio_delta(self, event: ResponseAudioDeltaEvent) -> None:
        if self._is_stale_response_scoped_event(event.response_id):
            return
        super()._handle_response_audio_delta(event)

    def _handle_response_audio_done(self, event: ResponseAudioDoneEvent) -> None:
        if self._is_stale_response_scoped_event(event.response_id):
            return
        super()._handle_response_audio_done(event)

    def _handle_response_output_item_done(self, event: ResponseOutputItemDoneEvent) -> None:
        if self._is_stale_response_scoped_event(event.response_id):
            return
        super()._handle_response_output_item_done(event)

    def _handle_response_done(self, event: ResponseDoneEvent) -> None:
        if self._is_stale_response_scoped_event(event.response.id):
            return
        super()._handle_response_done(event)
        self._current_response_id = None

    def _handle_error(self, event: RealtimeErrorEvent) -> None:
        # Unlike the base handler, fail the pending generate_reply future the
        # error refers to (the server reports the offending client event_id and
        # may not follow up with a response.done).
        error: dict[str, Any] = (
            event.error if isinstance(event.error, dict) else event.error.model_dump()
        )
        event_id = error.get("event_id") or event.event_id

        # A rejected conversation.item.create/delete never gets the added/deleted
        # reply its future waits on, so settle it here rather than let
        # update_chat_ctx() sit out the base's 5s timeout and then report a
        # generic one in place of the server's message. The event id is what
        # picks the right future: an updated item sends a delete and a create
        # under the same item id, so only it says which of the two was rejected.
        #
        # Kept ahead of every branch below because it is orthogonal to them --
        # the same error may also name a generate_reply, and the specific
        # invalid_previous_item_id handling further down stays as the fallback
        # for servers that report no event id. Both reach the same future, so
        # settling is shared rather than duplicated; see _settle_chat_ctx_wait.
        if event_id and (chat_ctx_fut := self._chat_ctx_event_futures.pop(event_id, None)):
            self._settle_chat_ctx_wait(chat_ctx_fut, error, event_id)
            # The rejection is the answer to one chat-ctx event, and the future
            # now carries it to update_chat_ctx(), which downgrades a single
            # rejected item to a warning. Emitting a session error as well would
            # report the same event twice and disagree with that downgrade, so
            # stop here -- as the base does after settling a non-fatal one.
            #
            # A fatal error still falls through: it has to end the session
            # whatever client event it happened to name. Escalation is unaffected
            # either way, being decided inside _settle_chat_ctx_wait; and a
            # generate_reply cannot be waiting on this id, since a chat-ctx event
            # id and a response.create event id are never the same id.
            if not _is_boson_fatal_error(error):
                return

        if (
            error.get("code") == "input_audio_buffer_commit_empty"
            and self._boson_opts.turn_detection is not None
        ):
            # commit_user_turn() commits the buffer whether or not server VAD is
            # on, and with it on the server has already committed each segment
            # itself, so ours lands on an emptied one. The base suppresses this
            # too, but keys it on the turn detection in its own opts -- which is
            # always None here, because Boson's lives in _boson_opts and is sent
            # separately. Reading the base's copy would silently never match.
            logger.debug(
                "Ignoring empty commit; server VAD had already committed the turn",
                extra={"lk.pii.error": error, "event_id": event_id},
            )
            return

        if error.get("code") in _BOSON_NONFATAL_ERROR_CODES or (
            error.get("type") in _BOSON_NONFATAL_ERROR_TYPES
        ):
            # Expected client/server races (e.g. response.cancel arriving after
            # the response already finished, or conversation.item.create racing
            # item order during chat-ctx replay) — not a real failure. Mirrors
            # the OpenAI base's own "Cancellation failed" message swallow.
            # Only fail a future this error specifically names; unlike the
            # fallback below, never blanket-fail every pending future for an
            # error that cannot be correlated to one.
            logger.debug(
                "Ignoring non-fatal Boson realtime error",
                extra={"lk.pii.error": error, "event_id": event_id},
            )
            if event_id and event_id in self._response_created_futures:
                self._fail_response_created_futures(
                    llm.RealtimeError(_format_error_message(error, event_id)), event_id=event_id
                )
            if error.get("type") == "invalid_previous_item_id":
                # Fail the one create this rejects, by the item id the server
                # names. Leaving it to time out instead would surface a generic
                # "timed out" 5s later in place of this specific message — and
                # failing "whatever is pending" is not an option either: the
                # error can arrive after its own update_chat_ctx() gave up and
                # returned, while a different one is now in flight.
                #
                # Servers that predate the id report nothing this can be
                # attributed to, so the turn degrades to that timeout.
                rejected_item_id = error.get("item_id")
                if not rejected_item_id:
                    logger.warning(
                        "Could not determine which conversation item an "
                        "invalid_previous_item_id error refers to; "
                        "update_chat_ctx() will report it as a timeout instead.",
                        extra={"lk.pii.error": error},
                    )
                    return
                fut = self._item_create_future.get(rejected_item_id)
                if fut is not None:
                    self._settle_chat_ctx_wait(fut, error, event_id)
            return

        message = _format_error_message(error, event_id)
        if _is_boson_fatal_error(error):
            # Permanent for the account, not for this connection: another attempt
            # buys nothing. Raised rather than emitted, which is how the base
            # signals the same thing -- its recv loop lets a non-retryable
            # APIError through, and _main_task reports it with recoverable=False
            # and stops reconnecting.
            #
            # Today the server follows this with a 4429 close, which
            # _NON_RETRYABLE_CLOSE_CODES would end the session on anyway. Not
            # relied on: this arrives first, and emitting it as recoverable would
            # announce a recovery in the window before the close contradicts it.
            # A refusal sent without a close is handled by the same branch.
            logger.error(
                "%s refused service",
                self._realtime_model._provider_label,
                extra={"lk.pii.error": error},
            )
            raise APIError(message=message, body=error, retryable=False)

        realtime_error = llm.RealtimeError(message)
        self._emit_error(realtime_error, recoverable=True)
        # Only fail the future this error specifically names. An error whose
        # event_id can't be correlated to a pending generate_reply() could
        # belong to a wholly unrelated client operation (e.g. a
        # conversation.item.create failure with its own event_id), so it must
        # not blanket-fail every other in-flight generate_reply() call; the
        # blanket-fail path below is reserved for genuine whole-session
        # failure (see _main_task), where every pending future really is done.
        if event_id and event_id in self._response_created_futures:
            self._fail_response_created_futures(realtime_error, event_id=event_id)

    def _settle_chat_ctx_wait(
        self, fut: asyncio.Future[None], error: dict[str, Any], event_id: str | None
    ) -> None:
        """Fail the update_chat_ctx() waiter that a rejection answers.

        Two lookups arrive here -- the client event id, and the item id the
        server names on invalid_previous_item_id. They find the same future, one
        object the base registered under both keys, and which one gets here first
        depends only on whether the server put an event id on the error.

        So escalation cannot be conditioned on having been the one to settle it.
        ``_chat_ctx_sync_error`` is what makes update_chat_ctx() re-raise rather
        than return while the base logs a warning; deciding it here, from the
        error's own type, keeps it independent of the arrival order.
        """
        sync_error = llm.RealtimeError(_format_error_message(error, event_id))
        if not fut.done():
            fut.set_exception(sync_error)
        if error.get("type") in _CHAT_CTX_ESCALATING_ERROR_TYPES:
            self._chat_ctx_sync_error = sync_error

    def _fail_response_created_futures(
        self, error: Exception, *, event_id: str | None = None
    ) -> None:
        if event_id and event_id in self._response_created_futures:
            futures = [self._response_created_futures.pop(event_id)]
        else:
            futures = list(self._response_created_futures.values())
            self._response_created_futures.clear()
        for fut in futures:
            if not fut.done():
                fut.set_exception(error)


def _normalize_ws_url(url: str, query_params: dict[str, str]) -> str:
    parsed = urlparse(url)
    scheme = parsed.scheme
    if scheme == "http":
        scheme = "ws"
    elif scheme == "https":
        scheme = "wss"
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query.update(query_params)
    return urlunparse(
        (scheme, parsed.netloc, parsed.path, parsed.params, urlencode(query), parsed.fragment)
    )


def _set_message_text(message: llm.ChatMessage, text: str) -> None:
    """Replace the first text part of ``message`` (or append one) with ``text``."""
    text_index = next(
        (idx for idx, content in enumerate(message.content) if isinstance(content, str)),
        None,
    )
    if text_index is None:
        message.content.append(text)
    else:
        message.content[text_index] = text


def _is_boson_fatal_error(error: dict[str, Any]) -> bool:
    """Whether this error is an account-level refusal a reconnect cannot clear."""
    return error.get("code") in _BOSON_FATAL_ERROR_CODES or (
        error.get("type") in _BOSON_FATAL_ERROR_TYPES
    )


def _copy_dict_or_none(value: Any | None) -> dict[str, Any] | None:
    # `False` disables turn detection the same way `None` does (matches the
    # README and Boson's Pipecat client); `dict(False)` would otherwise raise
    # TypeError since bool isn't iterable.
    if value is None or value is False:
        return None
    return dict(value)


def _normalize_input_audio_transcription(
    value: Any | None,
) -> NotGivenOr[dict[str, Any]]:
    if value is None:
        return NOT_GIVEN
    transcription = dict(value)
    transcription.pop("prompt", None)
    return transcription


def _build_input_audio_transcription(
    *,
    input_audio_transcription: NotGivenOr[dict[str, Any] | None],
    model: str,
    language: str | None,
) -> NotGivenOr[dict[str, Any]]:
    has_convenience_options = bool(model) or language is not None
    if not has_convenience_options and (
        not is_given(input_audio_transcription) or input_audio_transcription is None
    ):
        return NOT_GIVEN

    transcription = (
        dict(input_audio_transcription)
        if is_given(input_audio_transcription) and input_audio_transcription is not None
        else {}
    )
    # A transcription `prompt` is not part of the supported wire config; drop
    # it even if the caller passed it through a raw transcription dict.
    transcription.pop("prompt", None)
    if model:
        transcription["model"] = model
    if language is not None:
        transcription["language"] = language
    return transcription


def _input_audio_transcription_enabled(transcription: NotGivenOr[dict[str, Any]]) -> bool:
    """Whether the server will emit user-transcription events for this config.

    The server returns transcript events only when the client sets a non-empty
    ``model``. Omitting the transcription block, sending ``null``, or sending a
    block without a model all run ASR server-side (for the LLM) but emit no
    client-facing ``conversation.item.input_audio_transcription.completed``
    events.
    """
    return is_given(transcription) and bool(transcription.get("model"))


def _resolve_output_modalities(
    output_modalities: list[Literal["text", "audio"]] | None,
) -> list[Literal["text", "audio"]]:
    """Validate ``output_modalities`` to exactly ``["text"]`` or ``["audio"]``.

    The server rejects mixed (``["text", "audio"]``) and empty lists — output is
    single-modality. ``None`` defaults to ``["audio"]``.
    """
    if output_modalities is None:
        return ["audio"]
    if len(output_modalities) != 1 or output_modalities[0] not in ("text", "audio"):
        raise ValueError(
            "output_modalities must be exactly one of ['text'] or ['audio'] "
            f"(got {output_modalities!r}); mixed and empty lists are not supported."
        )
    return list(output_modalities)


def _build_noise_reduction(value: NotGivenOr[Any | None]) -> dict[str, Any] | None:
    """Normalize ``input_audio_noise_reduction`` to the OpenAI object form.

    Accepts a bare type string (``"near_field"`` / ``"far_field"``) or a dict
    (``{"type": ...}``). ``NOT_GIVEN`` and ``None`` both disable it (nothing is
    sent, which the server treats as disabled).
    """
    if not is_given(value) or value is None:
        return None
    if isinstance(value, str):
        return {"type": value}
    return dict(value)


def _server_vad_auto_interrupts_response(turn_detection: dict[str, Any] | None) -> bool:
    """Whether server VAD will auto-cancel the active response on speech start.

    The wire ``interrupt_response`` field does not currently change Higgs
    Realtime's actual behavior: with server VAD enabled the server always
    cancels the active response when it detects speech, regardless of what the
    client sends for this field. So the client's own duplicate-cancel
    suppression must key off whether server VAD is enabled at all, not off the
    (currently inert) ``interrupt_response`` value.
    """
    return turn_detection is not None


def _tools_to_boson(tools: list[llm.Tool]) -> list[dict[str, Any]]:
    boson_tools: list[dict[str, Any]] = []
    for tool in tools:
        if isinstance(tool, llm.FunctionTool):
            boson_tools.append(llm.utils.build_legacy_openai_schema(tool, internally_tagged=True))
        elif isinstance(tool, llm.RawFunctionTool):
            raw_schema = dict(tool.info.raw_schema)
            raw_schema.pop("meta", None)
            raw_schema["type"] = "function"
            boson_tools.append(raw_schema)
    return boson_tools


def _tool_choice_to_boson(tool_choice: llm.ToolChoice | None) -> Any:
    if tool_choice is None:
        return "auto"
    if isinstance(tool_choice, str):
        return tool_choice
    function = tool_choice.get("function", {})
    name = function.get("name")
    if name:
        return {"type": "function", "name": name}
    return "auto"


class _BosonConversationItem(BaseModel):
    """A conversation item in the Boson wire shape.

    Typed only as far as the base ``update_chat_ctx`` machinery needs
    (``item.id`` for echo correlation); the payload rides in extra fields
    because it deviates from the GA models (assistant content uses type
    ``"text"``, which their literals reject).
    """

    model_config = ConfigDict(extra="allow")
    id: str


class _BosonConversationItemCreateEvent(ConversationItemCreateEvent):
    item: _BosonConversationItem  # type: ignore[assignment]


def _livekit_item_to_boson_item(item: llm.ChatItem) -> dict[str, Any] | None:
    if item.type == "message":
        if item.role in _UNSUPPORTED_ITEM_ROLES:
            # Conversation items only accept role "assistant" or "user", so a
            # system/developer item can never be synced as one. Drop it
            # instead of sending a create that would be rejected; use
            # session-level instructions for persistent directives.
            return None
        content_text = _text_from_content(item.content)
        if not content_text:
            return None
        content_type = "text" if item.role == "assistant" else "input_text"
        return {
            "id": item.id,
            "object": "realtime.item",
            "type": "message",
            "role": item.role,
            "content": [{"type": content_type, "text": content_text}],
        }
    if item.type == "function_call":
        return {
            "id": item.id,
            "object": "realtime.item",
            "type": "function_call",
            "call_id": item.call_id,
            "name": item.name,
            "arguments": item.arguments,
            "status": "completed",
        }
    if item.type == "function_call_output":
        return {
            "id": item.id,
            "object": "realtime.item",
            "type": "function_call_output",
            "call_id": item.call_id,
            "output": item.output,
            "status": "completed",
        }
    return None


def _text_from_content(content: list[llm.ChatContent]) -> str:
    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, llm.AudioContent) and part.transcript:
            parts.append(part.transcript)
    return "\n".join(parts)


def _format_error_message(error: dict[str, Any], event_id: str | None) -> str:
    message = error.get("message") or "Boson realtime API error"
    details = {
        "type": error.get("type"),
        "code": error.get("code"),
        "event_id": event_id,
    }
    details = {key: value for key, value in details.items() if value is not None}
    if details:
        return f"{message} ({', '.join(f'{key}={value}' for key, value in details.items())})"
    return message
