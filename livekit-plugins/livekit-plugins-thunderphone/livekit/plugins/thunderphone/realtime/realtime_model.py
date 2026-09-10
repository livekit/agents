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

"""ThunderPhone voice agents as a LiveKit realtime model.

ThunderPhone's Realtime WebSocket API speaks the OpenAI Realtime protocol, so
this plugin is a thin layer over ``livekit.plugins.openai.realtime`` that
knows the ThunderPhone-specific parts:

* the endpoint, its query parameters (saved agent, engine, language, caller
  and callee numbers, wire audio) and ``sk_live_`` keys;
* the fact that an inline session *starts* on its first ``session.update``
  and freezes instructions and tools from then on, so the plugin sends one
  fully-configured update instead of the framework's incremental three;
* saved-agent sessions, whose prompt, voice and tools live on ThunderPhone
  and must not be overwritten from the agent;
* ThunderPhone's ``call.*`` events (hang-up, transfer, keypad) and the fact
  that every call ends with the server closing the socket, which must not be
  treated as a connection failure to retry.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import BaseModel

from livekit.agents import llm, utils
from livekit.agents.types import NOT_GIVEN, APIConnectOptions, NotGivenOr
from livekit.agents.utils import is_given
from livekit.plugins.openai.realtime import realtime_model as _openai, utils as _openai_utils

from ..log import logger

DEFAULT_BASE_URL = "wss://api.thunderphone.com/v1/realtime"
"""ThunderPhone's Realtime WebSocket endpoint."""

SERVER_MODEL = "thunderphone-realtime"
"""The model name ThunderPhone echoes; sending anything else back reads as a change."""

SAMPLE_RATE = _openai.SAMPLE_RATE  # 24 kHz PCM, same as OpenAI
API_KEY_ENV = "THUNDERPHONE_API_KEY"
API_KEY_PREFIX = "sk_live_"

# Session keys owned by a saved ThunderPhone agent. The server rejects changes
# to them once the call is live, so they are stripped from every update.
# Session events beyond the framework's own; typed Any because the base class
# fixes its event-name Literal.
CALL_EVENT: Any = "thunderphone_call_event"
CALL_ENDED_EVENT: Any = "thunderphone_call_ended"

_AGENT_OWNED_KEYS = ("instructions", "tools", "tool_choice")
# OpenAI-only knobs ThunderPhone does not use; harmless, but keep the wire clean.
_OPENAI_ONLY_KEYS = ("tracing", "truncation", "reasoning", "max_output_tokens")


class RealtimeModel(_openai.RealtimeModel):
    """Run a ThunderPhone voice agent as the realtime model of an AgentSession.

    Two ways to use it:

    * **Saved agent** — pass ``agent_id``. The agent's prompt, voice, engine,
      languages, tools, greeting and silence handling come from ThunderPhone;
      the LiveKit Agent's instructions and tools are not sent.
    * **Inline** — omit ``agent_id``. Instructions and tools come from the
      LiveKit Agent, ``product`` picks the engine (``spark``, ``bolt`` or
      ``storm``) and ``voice`` the ThunderPhone voice. Function tools run in
      the LiveKit agent.

    Instructions and tools are frozen once the call starts (ThunderPhone
    semantics), so the model declares them immutable and agent handoffs keep
    the first configuration. Every call ends with ThunderPhone closing the
    socket; the session does not reconnect, because a reconnect would start a
    new, separately billed call.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        agent_id: int | str | None = None,
        product: str | None = None,
        voice: NotGivenOr[str] = NOT_GIVEN,
        language: str | None = None,
        from_number: str | None = None,
        to_number: str | None = None,
        live_transcripts: bool = False,
        base_url: str = DEFAULT_BASE_URL,
        http_session: Any = None,
        conn_options: APIConnectOptions | None = None,
        **kwargs: Any,
    ) -> None:
        """Create a ThunderPhone realtime model.

        Args:
            api_key: ThunderPhone secret key (``sk_live_...``). Defaults to the
                ``THUNDERPHONE_API_KEY`` environment variable.
            agent_id: Id of a saved ThunderPhone agent. Mutually exclusive with
                ``product`` and ``voice``.
            product: Engine for inline sessions: ``"spark"``, ``"bolt"`` or ``"storm"``.
            voice: ThunderPhone voice name for inline sessions.
            language: Primary language hint for inline sessions, e.g. ``"es"``.
            from_number: Caller number to record on the call, if the session fronts a phone call.
            to_number: Called number to record on the call.
            live_transcripts: Stream caller transcript fragments while the caller is
                still speaking (billed extra). Off by default: one transcript per turn.
            base_url: WebSocket endpoint. Defaults to the ThunderPhone API.
            http_session: Optional aiohttp session to reuse.
            conn_options: Connection options. Retries default to 0: a ThunderPhone
                session is a call, and reconnecting would start a new one.
            **kwargs: Passed to ``livekit.plugins.openai.realtime.RealtimeModel``
                (``turn_detection``, ``input_audio_transcription``, ``modalities``, ...).
        """
        key = (api_key or os.environ.get(API_KEY_ENV, "")).strip()
        if not key.startswith(API_KEY_PREFIX):
            raise ValueError(
                f"ThunderPhone needs a secret API key starting with '{API_KEY_PREFIX}' "
                f"(pass api_key= or set {API_KEY_ENV})."
            )
        if agent_id is not None and (product or is_given(voice)):
            raise ValueError(
                "agent_id sessions take their engine and voice from the saved agent; "
                "drop product/voice or drop agent_id."
            )

        self._tp_agent_id = str(agent_id) if agent_id is not None else None
        self._tp_live_transcripts = bool(live_transcripts)

        url = _build_url(
            base_url,
            agent_id=self._tp_agent_id,
            product=product,
            language=language,
            from_number=from_number,
            to_number=to_number,
        )
        # The framework recycles an OpenAI socket after max_session_duration
        # (20 min by default) by closing it and reconnecting. On ThunderPhone
        # that would hang up and start a second, separately billed call.
        kwargs["max_session_duration"] = None
        super().__init__(
            model=SERVER_MODEL,
            voice=voice if is_given(voice) else "default",
            base_url=url,
            api_key=key,
            http_session=http_session,
            conn_options=conn_options or APIConnectOptions(max_retry=0),
            **kwargs,
        )
        self._provider_label = "ThunderPhone"
        # ThunderPhone freezes instructions and tools once the call starts, so a
        # later agent handoff keeps the first configuration rather than erroring.
        # Turn taking is ThunderPhone's and always on; the framework must not
        # run its own VAD or turn detector against it.
        self._capabilities = dataclasses.replace(
            self._capabilities,
            mutable_instructions=False,
            mutable_tools=False,
            turn_detection=True,
            can_disable_turn_detection=False,
        )

    @property
    def provider(self) -> str:
        return "ThunderPhone"

    @property
    def agent_id(self) -> str | None:
        """The saved ThunderPhone agent this model runs, if any."""
        return self._tp_agent_id

    def session(self, *, turn_detection_disabled: bool = False) -> RealtimeSession:
        if turn_detection_disabled:
            logger.warning(
                "ThunderPhone turn detection is server-side and always on; "
                "turn_detection_disabled is ignored"
            )
        sess = RealtimeSession(self)
        self._sessions.add(sess)
        return sess


class RealtimeSession(_openai.RealtimeSession):
    """One ThunderPhone call.

    Emits, in addition to the framework events:

    * ``thunderphone_call_event`` with every ``call.*`` event (transfer, keypad,
      speech ignored, ...);
    * ``thunderphone_call_ended`` with the ``call.ended`` event, after which the
      session closes itself.

    ``call_id`` holds the ThunderPhone call id once the session is live, for
    fetching the recording, transcript and grade afterwards.
    """

    def __init__(self, realtime_model: RealtimeModel) -> None:
        self._tp_model = realtime_model
        self._tp_agent_mode = realtime_model.agent_id is not None
        # Inline sessions: hold session.update until the agent has supplied its
        # instructions and tools, then send one update that starts the call.
        self._tp_holding = not self._tp_agent_mode
        self._tp_pending_session: dict[str, Any] = {}
        self._tp_call_ended = False
        # Function call ids longer than OpenAI's 32-character cap come back
        # from the framework hashed; map the hash back to the id the server
        # expects on the function_call_output.
        self._tp_call_ids: dict[str, str] = {}
        self.call_id: int | None = None
        super().__init__(realtime_model)
        self.on("openai_server_event_received", self._tp_on_server_event)

    # ----------------------------------------------------------- outgoing

    def send_event(self, event: Any) -> None:
        payload = (
            event.model_dump(by_alias=True, exclude_unset=True, exclude_defaults=False)
            if isinstance(event, BaseModel)
            else dict(event)
        )
        if payload.get("type") == "session.update":
            payload = self._tp_shape_session_update(payload)
            if self._tp_holding:
                _merge(self._tp_pending_session, payload.get("session") or {})
                return
        elif self._tp_holding and payload.get("type") != "input_audio_buffer.append":
            # Anything but audio is meaningful only once the call exists;
            # flush the configuration first so ordering is preserved.
            self._tp_flush()
        if payload.get("type") == "conversation.item.create":
            item = payload.get("item")
            if isinstance(item, dict) and item.get("type") == "function_call_output":
                call_id = item.get("call_id")
                if isinstance(call_id, str):
                    item["call_id"] = self._tp_call_ids.get(call_id, call_id)
        super().send_event(payload)

    def _tp_shape_session_update(self, payload: dict[str, Any]) -> dict[str, Any]:
        session = dict(payload.get("session") or {})
        session["model"] = SERVER_MODEL
        for key in _OPENAI_ONLY_KEYS:
            session.pop(key, None)
        if self._tp_agent_mode:
            # A saved agent is already running with its own prompt, tools,
            # voice, transcription and turn-taking; the server rejects any
            # change to those. Keep only what a client may still set.
            for key in _AGENT_OWNED_KEYS:
                session.pop(key, None)
            session.pop("output_modalities", None)
            audio = session.get("audio")
            if isinstance(audio, dict):
                for direction in ("input", "output"):
                    leg = audio.get(direction)
                    if isinstance(leg, dict):
                        audio[direction] = {k: v for k, v in leg.items() if k == "format"}
        else:
            config = dict(session.get("config") or {})
            # Opt into ThunderPhone's call.* events (hang-up reason, transfer, keypad).
            config.setdefault("call_events", True)
            session["config"] = config
        if self._tp_model._tp_live_transcripts:
            session["live_transcripts"] = True
        payload["session"] = session
        return payload

    def _tp_flush(self) -> None:
        if not self._tp_holding:
            return
        self._tp_holding = False
        if self._tp_pending_session:
            super().send_event(
                {
                    "event_id": utils.shortuuid("session_update_"),
                    "type": "session.update",
                    "session": self._tp_pending_session,
                }
            )
            self._tp_pending_session = {}

    async def _update_session(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        chat_ctx: NotGivenOr[llm.ChatContext] = NOT_GIVEN,
        tools: NotGivenOr[list[llm.Tool]] = NOT_GIVEN,
    ) -> None:
        # The framework sends instructions, then chat history, then tools. On
        # ThunderPhone the first session.update starts the call and freezes
        # instructions and tools, so send both before any conversation item.
        if is_given(instructions):
            try:
                await self.update_instructions(instructions)
            except llm.RealtimeError:
                logger.exception("failed to update the instructions")
        if is_given(tools):
            try:
                await self.update_tools(tools)
            except llm.RealtimeError:
                logger.exception("failed to update the tools")
        self._tp_flush()
        if is_given(chat_ctx):
            try:
                await self.update_chat_ctx(chat_ctx)
            except llm.RealtimeError:
                logger.exception("failed to update the chat_ctx")

    # ----------------------------------------------------------- incoming

    def _tp_on_server_event(self, event: dict[str, Any]) -> None:
        event_type = event.get("type")
        if event_type == "response.created":
            self._tp_claim_response(event)
            return
        if event_type == "response.output_item.done":
            item = event.get("item")
            if isinstance(item, dict) and item.get("type") == "function_call":
                call_id = item.get("call_id")
                if isinstance(call_id, str):
                    self._tp_call_ids[_openai_utils._shorten_call_id(call_id)] = call_id
            return
        if event_type in ("session.created", "session.updated"):
            session = event.get("session") or {}
            call_id = session.get("call_id") if isinstance(session, dict) else None
            if isinstance(call_id, int):
                self.call_id = call_id
            elif isinstance(call_id, str) and call_id.isdigit():
                self.call_id = int(call_id)
            return
        if not isinstance(event_type, str) or not event_type.startswith("call."):
            return
        self.emit(CALL_EVENT, event)
        if event_type == "call.ended" and not self._tp_call_ended:
            self._tp_call_ended = True
            logger.info("ThunderPhone call ended", extra={"lk.pii.event": event})
            self.emit(CALL_ENDED_EVENT, event)
            # The server closes the socket right after this event. Closing the
            # outgoing channel first makes that close expected, so the session
            # ends cleanly instead of being retried as a connection failure.
            self._msg_ch.close()

    def _tp_claim_response(self, event: dict[str, Any]) -> None:
        # generate_reply() resolves its future by the client_event_id the
        # server echoes in response.metadata. ThunderPhone servers that
        # predate that echo start the response but never label it, so the
        # future times out while the audio plays anyway. Credit an unlabeled
        # response to the oldest pending reply request instead; the hook runs
        # on the raw dict before the framework parses it.
        response = event.get("response")
        if not isinstance(response, dict):
            return
        metadata = response.get("metadata")
        if isinstance(metadata, dict) and metadata.get("client_event_id"):
            return
        pending = next(iter(self._response_created_futures), None)
        if pending is None:
            return
        response["metadata"] = {
            **(metadata if isinstance(metadata, dict) else {}),
            "client_event_id": pending,
        }


def _build_url(base_url: str, **params: str | None) -> str:
    """Compose the endpoint URL; the framework only appends ``?model=``."""
    parts = urlsplit(base_url)
    query = dict(parse_qsl(parts.query, keep_blank_values=False))
    query.update({k: v for k, v in params.items() if v})
    query.update(
        {
            "input_audio_format": "pcm16",
            "input_rate": str(SAMPLE_RATE),
            "output_audio_format": "pcm16",
            "output_rate": str(SAMPLE_RATE),
            # Understood by newer servers for saved-agent sessions; ignored otherwise.
            "call_events": "1",
        }
    )
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), ""))


def _merge(into: dict[str, Any], update: dict[str, Any]) -> None:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(into.get(key), dict):
            _merge(into[key], value)
        else:
            into[key] = value
