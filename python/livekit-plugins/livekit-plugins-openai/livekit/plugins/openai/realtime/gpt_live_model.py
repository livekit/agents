from __future__ import annotations

import base64
import json
import os
import time
from collections.abc import AsyncIterable
from dataclasses import dataclass, field, replace
from typing import Literal, TypedDict, assert_never
from urllib.parse import urlparse, urlunparse

from pydantic import JsonValue, TypeAdapter

from livekit import rtc
from livekit.agents import APIError, TokenUsage, llm, utils
from livekit.agents.llm import duplex as llm_duplex
from livekit.agents.metrics import LLMMetrics, Metadata, RealtimeModelMetrics
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    ErrorKind,
    NotGivenOr,
)
from livekit.agents.utils import is_given
from openai.types.responses import ResponseTextConfigParam
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

# error codes that end the session; any other error event is recoverable
_ERROR_KINDS: dict[str, ErrorKind] = {
    "insufficient_quota": "quota",
    "invalid_api_key": "auth",
    "account_deactivated": "auth",
    "billing_hard_limit_reached": "quota",
}

_TOOL_SCHEMA = TypeAdapter(dict[str, JsonValue])

Role = Literal["user", "assistant"]
GPTLiveVoices = Literal["aster", "beacon", "cinder", "marin", "stone", "vesper"]
_Append = types.InstructionsAppendEvent | types.ThinkingAppendEvent | types.CommentaryAppendEvent


class ResponsesDelegationOptions(TypedDict, total=False):
    """The backend Responses model delegated work runs on, under ``delegation="responses"``.

    A key left unset is not sent, and the service's own default applies.
    """

    model: str
    """Responses model slug; ``gpt-5.6-luna`` when unset. On Azure, the name of a Responses
    deployment in the same resource, which is required there."""
    instructions: str
    """Instructions for the backend model, distinct from the voice model's."""
    tool_choice: llm.ToolChoice | None
    parallel_tool_calls: bool
    reasoning: Reasoning
    """Responses reasoning settings, for example ``{"effort": "medium"}``."""
    text: ResponseTextConfigParam
    """Responses text settings, for example ``{"verbosity": "low"}``."""
    service_tier: types.ServiceTier
    max_output_tokens: int
    """Upper bound on the tokens one backend response may generate; at least 16."""


@dataclass
class GPTLiveDelegation:
    """Work the model handed to the application, under client delegation.

    It carries no task text: the ask is whatever the conversation says, which the agent's chat
    context holds. Answer it with :meth:`GPTVoiceSession.append_commentary`, passing this id.
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
    """Input audio sent since the user's last fragment, gaps in it included."""


# Responses delegation hands work to a backend Responses model, whose events arrive wrapped in
# response.event. One backend run is a "response":
#
#   response.created ─► response.output_item.done ×N (function calls) ─► response.completed
#
# A response completes with its calls unanswered. Each result is queued with response.item.create,
# and one response.create runs the next response, the continuation, once every call in the session
# has one; a partial batch is rejected. The voice model speaks the continuation's text on its own.


@dataclass
class _LiveOptions:
    model: str
    voice: str | types.CustomVoice
    delegation: types.DelegationTarget
    responses: ResponsesDelegationOptions
    service_tier: types.ServiceTier | None
    api_key: str | None
    base_url: str
    conn_options: APIConnectOptions
    max_session_duration: float | None
    is_azure: bool
    entra_token: str | None


class GPTLiveModel(llm.DuplexModel):
    """OpenAI GPT-Live full-duplex voice model, ready to pass to ``VoiceSession(llm=)``."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        voice: GPTLiveVoices | str | types.CustomVoice = DEFAULT_VOICE,
        delegation: types.DelegationTarget = "responses",
        responses_options: NotGivenOr[ResponsesDelegationOptions] = NOT_GIVEN,
        service_tier: NotGivenOr[types.ServiceTier] = NOT_GIVEN,
        api_key: str | None = None,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        max_session_duration: NotGivenOr[float | None] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        azure_deployment: str | None = None,
        entra_token: str | None = None,
    ) -> None:
        """
        Args:
            model: GPT-Live voice model slug. Ignored on Azure, where ``azure_deployment`` names
                the model.
            voice: Output voice: a name from :data:`GPTLiveVoices`, another supported name, or
                ``{"id": "voice_..."}`` for an authorized custom voice. Defaults to ``marin``.
                Immutable after the session starts.
            delegation: Where delegated work goes, fixed for the life of the session.
                ``responses`` runs it on a backend model, so ``@function_tool`` works as usual;
                ``client`` hands it to the application as a ``delegation_created`` event, which
                no framework tool can answer.
            responses_options: The backend Responses model, under ``delegation="responses"``.
            service_tier: Processing tier the session runs under, sent as the
                ``OpenAI-Service-Tier`` header on the connection, for example ``ultrafast``.
                Unset leaves the header off, and the account's default applies.
            api_key: OpenAI API key. Falls back to ``OPENAI_API_KEY``, or to
                ``AZURE_OPENAI_API_KEY`` on Azure unless ``entra_token`` is given.
            base_url: HTTP base url of the OpenAI API. On Azure, the resource endpoint, falling
                back to ``AZURE_OPENAI_ENDPOINT``.
            max_session_duration: Seconds after which the session is ended (its final usage
                kept) and a new one started from the conversation so far.
            conn_options: Retry/backoff and connection settings.
            azure_deployment: Azure OpenAI deployment of the voice model. Giving it or
                ``entra_token`` selects Azure; prefer :meth:`with_azure`.
            entra_token: Microsoft Entra ID token for Azure, instead of ``api_key``.
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
        responses = (
            responses_options if is_given(responses_options) else ResponsesDelegationOptions()
        )
        is_azure = azure_deployment is not None or entra_token is not None
        if not is_azure:
            api_key = api_key or os.environ.get("OPENAI_API_KEY")
            if api_key is None:
                raise ValueError(
                    "The api_key client option must be set either by passing api_key "
                    "to the client or by setting the OPENAI_API_KEY environment variable"
                )
            resolved_base_url = (
                base_url if is_given(base_url) else os.getenv("OPENAI_BASE_URL", OPENAI_BASE_URL)
            )
        else:
            if not azure_deployment:
                raise ValueError("Azure needs azure_deployment, the voice model's deployment name")
            model = azure_deployment

            if api_key and entra_token:
                raise ValueError("api_key and entra_token are mutually exclusive")
            if entra_token is None:
                api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
                if not api_key:
                    raise ValueError(
                        "Missing Azure credentials. Pass api_key or entra_token, "
                        "or set the AZURE_OPENAI_API_KEY environment variable"
                    )

            endpoint = base_url if is_given(base_url) else os.environ.get("AZURE_OPENAI_ENDPOINT")
            if not endpoint:
                raise ValueError(
                    "Missing Azure endpoint. Pass azure_endpoint or base_url, or set the "
                    "AZURE_OPENAI_ENDPOINT environment variable"
                )
            resolved_base_url = endpoint

            # the service resolves the backend model as a deployment of this resource, where the
            # OpenAI default names nothing: every delegated turn would fail while the voice
            # model keeps promising an answer, so ask for the deployment up front
            if delegation == "responses" and not responses.get("model"):
                raise ValueError(
                    "Azure responses delegation needs responses_options['model'], the name of a "
                    "Responses deployment in the same resource; or use delegation='client'"
                )

        if is_azure:
            self._provider = "azure_openai"
        else:
            self._provider = "unknown" if is_given(base_url) else "openai"
        self._opts = _LiveOptions(
            model=model,
            voice=voice,
            delegation=delegation,
            responses=responses,
            service_tier=service_tier if is_given(service_tier) else None,
            api_key=api_key,
            base_url=resolved_base_url,
            conn_options=conn_options,
            max_session_duration=max_session_duration if is_given(max_session_duration) else None,
            is_azure=is_azure,
            entra_token=entra_token,
        )

    @classmethod
    def with_azure(
        cls,
        *,
        azure_deployment: str,
        azure_endpoint: str | None = None,
        api_key: str | None = None,
        entra_token: str | None = None,
        base_url: str | None = None,
        voice: GPTLiveVoices | str | types.CustomVoice = DEFAULT_VOICE,
        delegation: types.DelegationTarget = "responses",
        responses_options: NotGivenOr[ResponsesDelegationOptions] = NOT_GIVEN,
        service_tier: NotGivenOr[types.ServiceTier] = NOT_GIVEN,
        max_session_duration: NotGivenOr[float | None] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> GPTLiveModel:
        """Create a GPTLiveModel served by Azure OpenAI.

        Args:
            azure_deployment: Deployment name of the GPT-Live voice model.
            azure_endpoint: Resource endpoint, such as ``https://<resource>.openai.azure.com``.
                Falls back to ``AZURE_OPENAI_ENDPOINT``.
            api_key: Azure API key. Falls back to ``AZURE_OPENAI_API_KEY`` unless
                ``entra_token`` is given.
            entra_token: Microsoft Entra ID token, instead of ``api_key``.
            base_url: Explicit base URL, for example a gateway in front of the resource.
                Mutually exclusive with ``azure_endpoint``.
            voice: Output voice, as in :class:`GPTLiveModel`.
            delegation: Where delegated work goes, as in :class:`GPTLiveModel`.
            responses_options: The backend Responses model under ``delegation="responses"``.
                Its ``model`` is required, and names a deployment in the same resource.
            service_tier: Processing tier, as in :class:`GPTLiveModel`.
            max_session_duration: Seconds after which the session is renewed, as in
                :class:`GPTLiveModel`.
            conn_options: Retry/backoff and connection settings.

        Returns:
            GPTLiveModel: A model that connects to the Azure resource.

        Raises:
            ValueError: If the deployment, credentials, or endpoint are missing, if both
                ``api_key`` and ``entra_token`` or both ``base_url`` and ``azure_endpoint`` are
                given, or if responses delegation has no backend deployment.

        Example:
            ```python
            from livekit.plugins.openai.realtime import GPTLiveModel

            model = GPTLiveModel.with_azure(
                azure_deployment="gpt-live-1",
                azure_endpoint="https://<resource>.openai.azure.com",
                api_key="<api-key>",
                responses_options={"model": "<responses-deployment>"},
            )
            ```
        """
        if base_url is not None and azure_endpoint is not None:
            raise ValueError("base_url and azure_endpoint are mutually exclusive")
        endpoint = base_url if base_url is not None else azure_endpoint

        return cls(
            voice=voice,
            delegation=delegation,
            responses_options=responses_options,
            service_tier=service_tier,
            api_key=api_key,
            base_url=endpoint if endpoint is not None else NOT_GIVEN,
            max_session_duration=max_session_duration,
            conn_options=conn_options,
            azure_deployment=azure_deployment,
            entra_token=entra_token,
        )

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return self._provider

    def audio_gate(self) -> llm.AudioGate:
        return llm.FixedGate(_SILENCE_RMS, min_silence_duration=_MIN_SILENCE_DURATION)

    def session(self) -> GPTVoiceSession:
        options = replace(self._opts, responses=self._opts.responses.copy())
        return GPTVoiceSession(self, options=options)

    async def aclose(self) -> None:
        pass


class GPTVoiceSession(
    llm.DuplexWebSocketSession[
        types.ServerEvent,
        types.ClientEvent,
        Literal["openai_server_event_received", "openai_client_event_queued", "delegation_created"],
    ]
):
    """A session for the OpenAI GPT-Live API (WebSocket), reached with ``Agent.duplex_session``.

    Each connection is one GPT-Live session, started with the whole configuration and the
    conversation so far; a session ends with ``session.close``, its final usage kept.

    Exposes three extra events:
    - openai_server_event_received: each raw server frame, before it is read
    - openai_client_event_queued: each client event, as it is queued for the server
    - delegation_created: a :class:`GPTLiveDelegation`, under client delegation

    The ``append_*`` methods queue context and return without waiting; what they add is also
    part of the conversation a new session starts with. Their ``*.appended`` events arrive at the
    estimated context-injection end; they do not mean speech has finished.
    """

    frame_ms = 100
    awaits_ready = True

    def __init__(self, duplex_model: GPTLiveModel, *, options: _LiveOptions) -> None:
        super().__init__(
            duplex_model,
            conn_options=options.conn_options,
            input_sample_rate=SAMPLE_RATE,
            input_num_channels=NUM_CHANNELS,
        )
        self.max_session_duration = options.max_session_duration
        self._live_model = duplex_model
        self._opts = options
        self._tools = llm.ToolContext.empty()
        # the agent's instructions, set by _update_session before session.start and immutable after
        self._instructions: str | None = None
        self._audio_ch = utils.aio.Chan[llm.DuplexAudioFrame]()

        # this connection's session.start went: what is added from now on is also sent to it
        self._live = False
        self._connected = False
        # the mute state the input last carried to the service, re-applied to each new session
        self._input_muted = False
        self._session_id: str | None = None
        # session usage is reported cumulatively; kept to emit per-event deltas
        self._usage_seconds = 0.0

        # everything the model has been told, both speakers' words and the appends included, to
        # start each new session with; the framework's chat context is the adapter's, not this
        self._history = llm.ChatContext.empty()
        # the history items the appends added: context, never something said
        self._appended: set[str] = set()
        self._speech: dict[Role, _Speech] = {}

        # delegation_id -> call_ids of its response that has not completed yet; on completion the
        # call_ids move to the open set, and stay there until each output is sent to the backend.
        # response_pending is set once an output is sent and cleared by the response.create
        self._backend_running_responses: dict[str | None, set[str]] = {}
        self._backend_open_calls: set[str] = set()
        self._backend_response_pending = False

        # the newest history item the last ask was about, so an ask never repeats one
        self._asked_item_id: str | None = None

    # the socket

    async def connect(self, conn_options: APIConnectOptions) -> llm_duplex.WebSocketTarget:
        if conn_options.total_timeout is not None:
            # a live session has no operation that a total deadline could bound
            raise APIError("GPT-Live sessions cannot enforce total_timeout", kind="invalid_request")
        if self._connected:
            self._new_session()
        self._connected = True
        self._live = False
        headers = {"User-Agent": "LiveKit Agents"}
        if not self._opts.is_azure:
            headers["Authorization"] = f"Bearer {self._opts.api_key}"
        elif self._opts.entra_token:
            headers["Authorization"] = f"Bearer {self._opts.entra_token}"
        elif self._opts.api_key:
            # Azure answers a bearer api key with a redirect to ?api-key=, which is not
            # followed for a websocket, so the key goes in its own header
            headers["api-key"] = self._opts.api_key
        if self._opts.service_tier:
            headers["OpenAI-Service-Tier"] = self._opts.service_tier
        url = _live_sessions_url(self._opts.base_url, is_azure=self._opts.is_azure)
        if utils.env_int("LK_OPENAI_DEBUG", 0):
            logger.debug("connecting to GPT-Live API", extra={"lk.pii.url": url})
        if self._input_muted:
            # a session starts unmuted, so the mute the app asked for is re-applied to it
            mute = types.InputAudioMuteEvent(type="session.input_audio.mute", event_id=_id())
            self._send(mute)
        return llm_duplex.WebSocketTarget(url, headers=headers)

    def _new_session(self) -> None:
        # a new connection is a new session, started from the history; the rest of what the
        # previous one was carrying never arrives
        self._end_speech("user")
        self._speech.clear()
        self._backend_running_responses.clear()
        self._backend_open_calls.clear()
        self._backend_response_pending = False
        self._usage_seconds = 0.0
        self._session_id = None

    def setup(self) -> types.SessionStartEvent:
        start = self._session_start_event()
        self._live = True
        self._queued(start)
        return start

    def goodbye(self) -> types.SessionCloseEvent:
        # read once per connection, at connect: the socket sends it only when the session ends
        return types.SessionCloseEvent(type="session.close", event_id=_id())

    def encode(
        self, item: rtc.AudioFrame | llm.DuplexInputGap | llm.DuplexInputMute
    ) -> types.ClientEvent | None:
        event: types.ClientEvent
        match item:
            case llm.DuplexInputMute():
                self._input_muted = item.muted
                event = (
                    types.InputAudioMuteEvent(type="session.input_audio.mute", event_id=_id())
                    if item.muted
                    else types.InputAudioUnmuteEvent(
                        type="session.input_audio.unmute", event_id=_id()
                    )
                )
            case llm.DuplexInputGap():
                # dropped audio is silence to the caller's turn
                self._heard_quiet(item.duration)
                return None
            case rtc.AudioFrame():
                self._heard_quiet(item.duration)
                event = types.InputAudioAppendEvent(
                    type="session.input_audio.append", audio=base64.b64encode(item.data).decode()
                )
            case _:
                assert_never(item)
        self._queued(event)
        return event

    def _heard_quiet(self, seconds: float) -> None:
        """The caller's turn ends on their own audio: this much since their last fragment."""
        if (speech := self._speech.get("user")) is not None:
            speech.quiet_ms += round(seconds * 1000)
            if speech.quiet_ms >= _MIN_SILENCE_MS:
                self._end_speech("user")

    def _send(self, event: types.ClientEvent) -> None:
        """A command for the current session; one queued for a session that ended is dropped
        with it, what it carried being in the history the next one starts with."""
        self._queued(event)
        self.send_now(event)

    def _queued(self, event: types.ClientEvent) -> None:
        self.emit("openai_client_event_queued", event)
        if utils.env_int("LK_OPENAI_DEBUG", 0) and event["type"] != "session.input_audio.append":
            logger.debug("gpt-live client event", extra={"lk.pii.event": event})

    def decode(self, frame: bytes | str) -> types.ServerEvent | None:
        # every frame is the application's to see, those the session does not read included
        self.emit("openai_server_event_received", frame)
        return super().decode(frame)

    def on_message(self, message: types.ServerEvent) -> None:
        # while closing only the shutdown events matter; the framework has torn down
        if self._closing and message["type"] not in ("session.usage.updated", "session.closed"):
            return
        if utils.env_int("LK_OPENAI_DEBUG", 0) and message["type"] != "session.output_audio.delta":
            logger.debug("gpt-live server event", extra={"lk.pii.event": message})

        if message["type"] == "session.started":
            if (session := message.get("session")) is not None and (sid := session.get("id")):
                self._session_id = sid
            self.report(llm_duplex.Ready())
        elif message["type"] == "session.output_audio.delta":
            self._handle_output_audio(message.get("delta", ""))
        elif message["type"] == "session.output_transcript.delta":
            self._handle_transcript_delta(
                "assistant",
                message.get("delta", ""),
                message.get("start_ms"),
                message.get("end_ms"),
            )
        elif message["type"] == "session.input_transcript.delta":
            self._handle_transcript_delta(
                "user", message.get("delta", ""), message.get("start_ms"), message.get("end_ms")
            )
        elif message["type"] == "session.delegation.created":
            self._handle_delegation_created(message.get("delegation"))
        elif message["type"] == "response.event":
            self._handle_response_event(message)
        elif message["type"] == "session.usage.updated":
            if (window := message.get("context_window")) is not None:
                logger.debug(
                    "gpt-live context window utilization",
                    extra={"usage_ratio": window.get("usage_ratio")},
                )
            self._handle_usage(message.get("usage"))
        elif message["type"] == "session.closed":
            logger.debug(
                "gpt-live session closed",
                extra={"reason": message.get("reason"), "session_id": self._session_id},
            )
            self._handle_usage(message.get("usage"))
            self.report(llm_duplex.SessionFinished())
        elif message["type"] == "error":
            self._handle_error(message.get("error"))
        else:
            self._skip_unknown_message(message)

    # inbound events

    def _handle_output_audio(self, delta: str) -> None:
        # every frame is published, silence included: the framework decides what plays
        if not (data := base64.b64decode(delta)):
            return
        frame = rtc.AudioFrame(
            data=data,
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
            samples_per_channel=len(data) // 2,
        )
        self._audio_ch.send_nowait(llm.DuplexAudioFrame(frame=frame))

    def _handle_transcript_delta(
        self, role: Role, delta: str, start_ms: int | None, end_ms: int | None
    ) -> None:
        if not delta:
            return
        speech = self._speech.get(role)
        # a pause on the model's clock ends the message even when its fragments arrived together
        if (
            speech is not None
            and speech.end_ms is not None
            and start_ms is not None
            and start_ms - speech.end_ms > _MIN_SILENCE_MS
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
        speech.text += delta
        speech.quiet_ms = 0
        if end_ms is not None:
            speech.end_ms = max(speech.end_ms or 0, end_ms)
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
                llm.DuplexOutputTranscriptDelta(text=delta, start_ms=start_ms, end_ms=end_ms),
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

    def _handle_delegation_created(self, delegation: types.DelegationInfo | None) -> None:
        delegation_id = delegation.get("id") if delegation is not None else None
        if not delegation_id:
            logger.warning("gpt-live delegation has no id; nothing can answer it")
        elif delegation is not None and delegation.get("target") == "client":
            speech = self._speech.get("user")
            self.emit(
                "delegation_created",
                GPTLiveDelegation(
                    id=delegation_id, pending_transcript=speech.text if speech else ""
                ),
            )

    def _handle_response_event(self, envelope: types.ResponseEventEnvelope) -> None:
        # the inner event carries no response id, so a delegation's responses are followed in
        # sequence: a continuation is the next response.created under the same delegation, and a
        # response the application started itself has a null delegation
        event = envelope["event"]
        response = event.get("response")
        d_id = envelope.get("delegation_id")

        if event["type"] == "response.created":
            self._backend_running_responses[d_id] = set()

        elif event["type"] == "response.output_item.done":
            # only the completed item carries the name, call id and arguments together
            item = event.get("item")
            if item is None or item.get("type") != "function_call":
                return
            call_id, name, arguments = item.get("call_id"), item.get("name"), item.get("arguments")
            if item.get("status") != "completed":
                logger.debug(
                    "gpt-live ignoring incomplete function call",
                    extra={"function": name, "status": item.get("status")},
                )
                return
            if not call_id or not name or arguments is None:
                logger.warning(
                    "gpt-live dropping function call with missing fields",
                    extra={"call_id": call_id, "function": name},
                )
                return
            if (calls := self._backend_running_responses.get(d_id)) is None:
                logger.warning(
                    "gpt-live function call outside a known response",
                    extra={"call_id": call_id, "function": name, "delegation_id": d_id},
                )
                calls = self._backend_open_calls
            if call_id in calls:
                return
            calls.add(call_id)

            fnc_call = llm.FunctionCall(
                id=item.get("id") or utils.shortuuid("fc_"),
                call_id=call_id,
                name=name,
                arguments=arguments,
            )
            self._history.insert(fnc_call)
            self.emit("function_call", fnc_call)

        elif event["type"] == "response.completed":
            if response is not None and (usage := response.get("usage")) is not None:
                self._report_backend_usage(response, usage)
            if (calls := self._backend_running_responses.pop(d_id, None)) is not None:
                self._backend_open_calls |= calls
            self._maybe_continue_response()

        elif event["type"] in ("response.failed", "response.incomplete"):
            logger.warning(
                "gpt-live backend response did not complete",
                extra={
                    "type": event["type"],
                    "delegation_id": d_id,
                    "lk.pii.error": response.get("error") if response else None,
                    "lk.pii.incomplete_details": (
                        response.get("incomplete_details") if response else None
                    ),
                },
            )
            # the service discards a failed response's calls: an output for one is refused, so
            # it goes to the voice model as context instead
            self._backend_running_responses.pop(d_id, None)
            self._maybe_continue_response()

    def _report_backend_usage(
        self, response: types.ResponseSnapshot, usage: types.ResponseUsage
    ) -> None:
        # the voice model is billed by duration; these tokens are the backend's and are reported
        # under its name
        input_details = usage.get("input_tokens_details", types.InputTokensDetails())
        output_details = usage.get("output_tokens_details", types.OutputTokensDetails())
        self.emit(
            "metrics_collected",
            LLMMetrics(
                label=self._live_model.label,
                request_id=response.get("id") or "",
                timestamp=time.time(),
                duration=0,
                cancelled=False,
                tokens=TokenUsage(
                    input_tokens=usage.get("input_tokens", 0),
                    output_tokens=usage.get("output_tokens", 0),
                    cache_read_tokens=input_details.get("cached_tokens", 0),
                    cache_write_tokens=input_details.get("cache_write_tokens", 0),
                    output_reasoning_tokens=output_details.get("reasoning_tokens", 0),
                ),
                tokens_per_second=0,
                metadata=Metadata(
                    model_name=response.get("model")
                    or self._opts.responses.get("model", DEFAULT_BACKEND_MODEL),
                    model_provider=self._live_model.provider,
                ),
            ),
        )

    def _maybe_continue_response(self) -> None:
        # one response.create continues the chain, and only once nothing is still asking and every
        # call in the conversation has its answer: the service refuses a partial batch
        if (
            self._backend_running_responses
            or self._backend_open_calls
            or not self._backend_response_pending
        ):
            return
        self._backend_response_pending = False
        self._send(types.ResponseCreateEvent(type="response.create", event_id=_id()))

    # metrics and errors

    def _handle_usage(self, usage: types.Usage | None) -> None:
        # reported cumulatively for the whole session, so only the delta goes to the collectors
        seconds = usage.get("seconds", 0.0) if usage is not None else 0.0
        previous, self._usage_seconds = self._usage_seconds, seconds
        self.emit(
            "metrics_collected",
            RealtimeModelMetrics(
                timestamp=time.time(),
                request_id=self._session_id or "",
                duration=0,
                connection_duration=max(0.0, seconds - previous),
                cancelled=False,
                label=self._live_model.label,
                tokens_per_second=0,
                metadata=Metadata(
                    model_name=self._live_model.model, model_provider=self._live_model.provider
                ),
            ),
        )

    def _handle_error(self, body: types.ErrorBody | None) -> None:
        code = (body.get("code") or body.get("type") or "") if body is not None else ""
        error = APIError(
            "GPT-Live returned an error",
            kind=_ERROR_KINDS.get(code, "server"),
            detail=json.dumps(body),
        )
        if not error.retryable:
            raise error
        self.emit(
            "error",
            llm.RealtimeModelError(
                timestamp=time.time(),
                label=self._live_model.label,
                error=error,
                recoverable=True,
            ),
        )

    # DuplexSession interface

    @property
    def session_id(self) -> str | None:
        """The service's id for the current session."""
        return self._session_id

    @property
    def audio_stream(self) -> AsyncIterable[llm.DuplexAudioFrame]:
        return self._audio_ch

    @property
    def tools(self) -> llm.ToolContext:
        return self._tools.copy()

    def send_event(self, event: types.ClientEvent) -> None:
        """Send a client event to the current session."""
        self._send(event)

    def append_instructions(self, text: str, *, delegation_id: str | None = None) -> None:
        """Add a standing rule to the model's instructions, capped at 500 tokens."""
        self._append(
            types.InstructionsAppendEvent(
                type="session.instructions.append",
                event_id=_id(),
                delegation_id=delegation_id,
                content=text,
            )
        )

    def append_thinking(self, text: str, *, delegation_id: str | None = None) -> None:
        """Give the model something to know without saying it, capped at 500 tokens."""
        self._append(
            types.ThinkingAppendEvent(
                type="session.thinking.append",
                event_id=_id(),
                delegation_id=delegation_id,
                content=text,
            )
        )

    def append_commentary(self, text: str, *, delegation_id: str | None = None) -> None:
        """Give the model something to say once, in its own words, capped at 500 tokens.

        Under client delegation this answers a :class:`GPTLiveDelegation`; repeated calls with the
        same ``delegation_id`` continue that work.
        """
        self._append(
            types.CommentaryAppendEvent(
                type="session.commentary.append",
                event_id=_id(),
                delegation_id=delegation_id,
                content=text,
            )
        )

    def _append(self, event: _Append) -> None:
        # part of the conversation a new session starts with; sent to the current one once it
        # started, before which its session.start carries it
        message = llm.ChatMessage(role="developer", content=[event["content"]])
        self._appended.add(message.id)
        self._history.insert(message)
        if self._live:
            self._send(event)

    def mute_input(self) -> None:
        """Replace microphone input with silence; the model keeps generating and speaking.

        It applies from the audio pushed so far on, and is remembered and re-applied to any session
        a reconnect opens.
        """
        self._input.mute(True)

    def unmute_input(self) -> None:
        self._input.mute(False)

    async def aclose(self) -> None:
        await super().aclose()
        self._audio_ch.close()

    # framework hooks

    async def _update_instructions(self, instructions: str) -> None:
        if self._live and instructions != self._instructions:
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
        if not self._live:
            return  # startup history, rendered into session.start

        # a system or developer message is a standing rule for the voice model, a tool result
        # answering a call the backend delegated goes back on the backend's channel, and everything
        # else is context for the voice model, as one append. The items are in the history
        # already, so these go to the current session only
        backend_outputs: list[llm.FunctionCallOutput] = []
        lines: list[str] = []
        for item in items:
            if isinstance(item, llm.ChatMessage) and item.role in ("system", "developer"):
                if text := item.text_content:
                    self._send(
                        types.InstructionsAppendEvent(
                            type="session.instructions.append",
                            event_id=_id(),
                            delegation_id=None,
                            content=text,
                        )
                    )
            elif isinstance(item, llm.FunctionCallOutput) and (
                item.call_id in self._backend_open_calls
                or any(item.call_id in c for c in self._backend_running_responses.values())
            ):
                backend_outputs.append(item)
            elif (rendered := _render_item(item)) is not None:
                lines.append("{}: {}".format(*rendered))

        if lines:
            self._send(
                types.ThinkingAppendEvent(
                    type="session.thinking.append",
                    event_id=_id(),
                    delegation_id=None,
                    content="\n".join(lines),
                )
            )

        for output in backend_outputs:
            self._send(
                types.ResponseItemCreateEvent(
                    type="response.item.create",
                    event_id=_id(),
                    item=types.FunctionCallOutput(
                        type="function_call_output", call_id=output.call_id, output=output.output
                    ),
                )
            )
            self._backend_open_calls.discard(output.call_id)
            for calls in self._backend_running_responses.values():
                calls.discard(output.call_id)
        if backend_outputs:
            if silenced := [o.name or o.call_id for o in backend_outputs if not o.reply_required]:
                logger.warning(
                    "a tool result wants no reply, but GPT Live will answer it anyway: the "
                    "backend has no way to close a call without a spoken continuation, and an "
                    "unanswered call holds every later tool call. Continuing regardless.",
                    extra={"functions": silenced},
                )
            self._backend_response_pending = True
            self._maybe_continue_response()

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
            self._ask(f"{_ASK_INSTRUCTED}\n\n{instructions}")
            return
        # a typed message still the newest thing said rides in the ask, once; after speech has
        # moved the conversation on, the ask points at the context instead
        said = [item for item in self._history.items if item.id not in self._appended]
        newest = said[-1] if said else None
        typed = (
            newest.text_content
            if isinstance(newest, llm.ChatMessage)
            and newest.role == "user"
            and newest.transcript_confidence is None
            and newest.id != self._asked_item_id
            else None
        )
        self._asked_item_id = newest.id if newest is not None else None
        self._ask(f"{_ASK_TYPED}\n\n{typed}" if typed else _ASK_BARE)

    def _ask(self, text: str) -> None:
        # a request to speak now, for the current session only: a new one never replays it
        self._send(
            types.CommentaryAppendEvent(
                type="session.commentary.append", event_id=_id(), delegation_id=None, content=text
            )
        )

    def _update_options(
        self, *, tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN
    ) -> None:
        if is_given(tool_choice):
            self._opts.responses["tool_choice"] = tool_choice
            self._send_delegation_update(
                types.ResponsesConfig(tool_choice=_to_tool_choice(tool_choice))
            )

    # outbound

    def _build_delegation(self) -> types.Delegation:
        if self._opts.delegation == "client":
            return types.Delegation(type="client")
        opts = self._opts.responses
        config = types.ResponsesConfig(model=opts.get("model", DEFAULT_BACKEND_MODEL))
        if tools := _build_delegation_tools(self._tools.flatten()):
            config["tools"] = tools
        if "instructions" in opts:
            config["instructions"] = opts["instructions"]
        if "tool_choice" in opts:
            config["tool_choice"] = _to_tool_choice(opts["tool_choice"])
        if "parallel_tool_calls" in opts:
            config["parallel_tool_calls"] = opts["parallel_tool_calls"]
        if "reasoning" in opts:
            config["reasoning"] = opts["reasoning"]
        if "text" in opts:
            config["text"] = opts["text"]
        if "service_tier" in opts:
            config["service_tier"] = opts["service_tier"]
        if "max_output_tokens" in opts:
            config["max_output_tokens"] = opts["max_output_tokens"]
        return types.Delegation(type="responses", responses=config)

    def _session_start_event(self) -> types.SessionStartEvent:
        """The whole configuration, composed fresh for each session."""
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
            part = types.TextPart(
                type="output_text" if role == "assistant" else "input_text", text=text
            )
            items.append(types.InputItem(type="message", role=role, content=[part]))
        if dropped:
            logger.warning(
                "gpt-live startup history exceeds what a session accepts; dropping the oldest",
                extra={"dropped": dropped, "kept": len(items)},
            )
        items.reverse()

        session = types.SessionConfig(
            model=self._opts.model,
            audio=types.AudioConfig(
                format=types.AudioFormat(type="audio/pcm", rate=SAMPLE_RATE),
                output=types.AudioOutput(voice=self._opts.voice),
            ),
            delegation=self._build_delegation(),
        )
        if self._instructions is not None:
            session["instructions"] = self._instructions
        if items:
            session["input"] = items
        return types.SessionStartEvent(type="session.start", event_id=_id(), session=session)

    def _send_delegation_update(self, responses: types.ResponsesConfig) -> None:
        """A sparse session.update: only the backend settings can change once started."""
        if self._opts.delegation != "responses" or not self._live:
            return
        self._send(
            types.SessionUpdateEvent(
                type="session.update",
                event_id=_id(),
                session=types.SessionUpdateConfig(
                    delegation=types.Delegation(type="responses", responses=responses)
                ),
            )
        )


def _id() -> str:
    return utils.shortuuid("gpt_live_")


def _live_sessions_url(base_url: str, *, is_azure: bool) -> str:
    """The websocket URL of the sessions endpoint under ``base_url``, without a query string."""
    parsed = urlparse(base_url)
    scheme = {"http": "ws", "https": "wss"}.get(parsed.scheme, parsed.scheme)
    path = parsed.path.rstrip("/")
    if is_azure and path in ("", "/openai"):
        # Azure serves GPT-Live only on its v1 surface: /openai/live/sessions is a 404
        path = "/openai/v1"
    if not path.endswith("/live/sessions"):
        path = f"{path}/live/sessions"
    return urlunparse((scheme, parsed.netloc, path, "", "", ""))


def _to_tool_choice(tool_choice: llm.ToolChoice | None) -> str | types.FunctionToolChoice:
    if tool_choice is None:
        return "auto"
    if isinstance(tool_choice, str):
        return tool_choice
    return types.FunctionToolChoice(type="function", name=tool_choice["function"]["name"])


def _build_delegation_tools(tools: list[llm.Tool]) -> list[dict[str, JsonValue]]:
    oai_tools: list[dict[str, JsonValue]] = []
    for tool in tools:
        if llm.is_function_tool(tool):
            schema = llm.utils.build_legacy_openai_schema(tool, internally_tagged=True)
            oai_tools.append(_TOOL_SCHEMA.validate_python(schema))
        elif llm.is_raw_function_tool(tool):
            raw = dict(tool.info.raw_schema)
            raw.pop("meta", None)
            raw["type"] = "function"
            oai_tools.append(_TOOL_SCHEMA.validate_python(raw))
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
