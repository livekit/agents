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

"""PolyAI Dialog-RSN-1 as a LiveKit realtime model.

Dialog-RSN-1's realtime API is wire-compatible with the OpenAI Realtime protocol, so this plugin
is a thin layer over ``livekit.plugins.openai.realtime`` that knows the PolyAI-specific parts:

* the endpoint, the ``POLY_API_KEY`` key and the ``dialog-rsn-1`` model;
* PolyAI's session extensions (``poly_knowledge``, ``poly_response_language``,
  ``poly_first_assistant_message``) and tool extensions (``poly_speech``, ``poly_spoken``);
* the ``poly_cited_topics`` and ``poly_out_of_domain`` fields on ``response.done``;
* the options Dialog-RSN-1 rejects (a ``tool_choice`` other than ``"auto"``, per-reply tools,
  ``interrupt_response=False``), refused here before the server drops the whole session update.

Reference: https://docs.poly.ai/dialog-rsn-1/reference
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import aiohttp

from livekit.agents import llm
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import is_given
from livekit.plugins.openai.realtime import realtime_model as _openai

from ..log import logger

DEFAULT_BASE_URL = "wss://api.studio.poly.ai/v1/realtime"
"""The endpoint for keys made at https://studio.poly.ai."""

DEFAULT_MODEL = "dialog-rsn-1"
API_KEY_ENV = "POLY_API_KEY"
BASE_URL_ENV = "DIALOG_REALTIME_URL"

ToolSpeech = Literal["none", "required"]
ResponseDoneCallback = Callable[[dict[str, Any]], Any]

_callback_tasks: set[asyncio.Future[Any]] = set()


@dataclass
class KnowledgeTopic:
    """One topic Dialog-RSN-1 can answer from and cite.

    Args:
        name: The topic's name. ``response.done`` cites topics by this name.
        content: What the model should know about the topic.
        actions: Optional instructions for what to do when the topic comes up.
    """

    name: str
    content: str
    actions: str | None = None

    def to_dict(self) -> dict[str, str]:
        topic = {"name": self.name, "content": self.content}
        if self.actions is not None:
            topic["actions"] = self.actions
        return topic


class RealtimeModel(_openai.RealtimeModel):
    """PolyAI Dialog-RSN-1, ready to pass to ``AgentSession(llm=)``.

    Dialog-RSN-1 is a speech-to-text model: it hears the caller, confirms when their turn is
    over, calls your tools and writes the reply. By default PolyAI's realtime API speaks the
    reply; pass ``modalities=["text"]`` to speak it with a TTS plugin instead.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        voice: str | None = None,
        modalities: NotGivenOr[list[Literal["text", "audio"]]] = NOT_GIVEN,
        api_key: str | None = None,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        turn_detection: NotGivenOr[Any] = NOT_GIVEN,
        knowledge: Sequence[KnowledgeTopic | Mapping[str, str]] | None = None,
        response_language: str | None = None,
        first_message: str | None = None,
        tool_speech: Mapping[str, ToolSpeech] | None = None,
        spoken_arguments: Mapping[str, str] | None = None,
        on_response_done: ResponseDoneCallback | None = None,
        http_session: aiohttp.ClientSession | None = None,
        max_session_duration: NotGivenOr[float | None] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        """Create a Dialog-RSN-1 realtime model.

        Args:
            model: The model name. Defaults to ``"dialog-rsn-1"``.
            voice: A Cartesia voice id for replies the API speaks. Unset, the deployment's
                default voice speaks.
            modalities: ``["audio"]`` (the default) for replies the API speaks, or ``["text"]``
                to speak them with a TTS plugin.
            api_key: A Dialog-RSN-1 API key. Defaults to the ``POLY_API_KEY`` environment
                variable. Create one at https://studio.poly.ai (API Keys tab).
            base_url: The realtime endpoint. Defaults to ``DIALOG_REALTIME_URL``, then
                ``wss://api.studio.poly.ai/v1/realtime``. A key from a regional Studio only
                works on that region's host, for example ``wss://api.us.poly.ai/v1/realtime``.
            turn_detection: Server turn detection, as for the OpenAI plugin. Only
                ``create_response`` is read; timing fields are accepted and ignored.
                ``interrupt_response=False`` is not supported.
            knowledge: Topics the model answers from and cites, sent as ``poly_knowledge``.
            response_language: The language to answer in, as an ISO 639 tag (``"es"``,
                ``"pt-BR"``), sent as ``poly_response_language``.
            first_message: What the assistant says first, word for word, sent as
                ``poly_first_assistant_message``. The server says it as soon as the session is
                configured, so don't also call ``generate_reply()`` to greet.
            tool_speech: Per tool name, ``"none"`` to keep the tool silent or ``"required"`` for a
                spoken line on every call.
            spoken_arguments: Per tool name, one string argument the model says to the caller
                while the tool runs, such as a hold line.
            on_response_done: Called with the raw ``response`` of every ``response.done``, which
                can carry ``poly_cited_topics`` and ``poly_out_of_domain``.
            http_session: Optional shared HTTP session.
            max_session_duration: Seconds before the connection is recycled.
            conn_options: Retry and connection settings.

        Raises:
            ValueError: If no API key is given or set, or an option Dialog-RSN-1 rejects is set.
        """
        api_key = (api_key or os.environ.get(API_KEY_ENV) or "").strip()
        if not api_key:
            raise ValueError(
                f"Dialog-RSN-1 needs an API key: pass api_key or set {API_KEY_ENV}. "
                "Create one at https://studio.poly.ai (API Keys tab)."
            )
        if (
            is_given(turn_detection)
            and getattr(turn_detection, "interrupt_response", None) is False
        ):
            raise ValueError(
                "Dialog-RSN-1 doesn't support interrupt_response=False: the server rejects it "
                "and drops the rest of the session update."
            )
        resolved_base_url = (
            base_url if is_given(base_url) else os.environ.get(BASE_URL_ENV) or DEFAULT_BASE_URL
        )

        init_kwargs: dict[str, Any] = {
            "model": model,
            "modalities": modalities if is_given(modalities) else ["audio"],
            "api_key": api_key,
            "base_url": resolved_base_url,
            "http_session": http_session,
            "conn_options": conn_options,
        }
        if voice is not None:
            init_kwargs["voice"] = voice
        if is_given(turn_detection):
            init_kwargs["turn_detection"] = turn_detection
        if is_given(max_session_duration):
            init_kwargs["max_session_duration"] = max_session_duration
        super().__init__(**init_kwargs)

        # Dialog-RSN-1 takes tools and tool_choice on the session only, not per reply.
        self._capabilities.per_response_tool_choice = False
        self._provider_label = "PolyAI Dialog-RSN-1"

        self._poly_voice = voice
        self._poly_session: dict[str, Any] = {}
        if knowledge is not None:
            self._poly_session["poly_knowledge"] = [
                t.to_dict() if isinstance(t, KnowledgeTopic) else dict(t) for t in knowledge
            ]
        if response_language is not None:
            self._poly_session["poly_response_language"] = response_language
        if first_message is not None:
            self._poly_session["poly_first_assistant_message"] = first_message
        self._tool_speech = dict(tool_speech or {})
        self._spoken_arguments = dict(spoken_arguments or {})
        self._on_response_done = on_response_done

    @property
    def provider(self) -> str:
        return "PolyAI"

    def session(self, *, turn_detection_disabled: bool = False) -> RealtimeSession:
        sess = RealtimeSession(self, turn_detection_disabled=turn_detection_disabled)
        self._sessions.add(sess)
        return sess


class RealtimeSession(_openai.RealtimeSession):
    """A Dialog-RSN-1 session: PolyAI's extensions on the OpenAI Realtime session."""

    def __init__(
        self, realtime_model: RealtimeModel, *, turn_detection_disabled: bool = False
    ) -> None:
        super().__init__(realtime_model, turn_detection_disabled=turn_detection_disabled)
        self.on("openai_server_event_received", self._on_polyai_server_event)

    @property
    def _polyai_model(self) -> RealtimeModel:
        # The base class sends the first session.update from inside its __init__, before this
        # class could set an attribute of its own, so read the model it stored.
        return cast(RealtimeModel, self._realtime_model)

    def _create_session_update_event(self) -> dict[str, Any]:
        event = _as_dict(super()._create_session_update_event())
        session = event.setdefault("session", {})
        if self._polyai_model._poly_voice is None:
            # Leave the voice unset so the deployment's default voice speaks.
            output = session.get("audio", {}).get("output")
            if isinstance(output, dict):
                output.pop("voice", None)
        session.update(self._polyai_model._poly_session)
        return event

    def _create_tools_update_event(self, tools: list[llm.Tool]) -> dict[str, Any]:
        event = super()._create_tools_update_event(tools)
        model = self._polyai_model
        for tool in event.get("session", {}).get("tools") or []:
            name = tool.get("name")
            speech = model._tool_speech.get(name)
            if speech:
                tool["poly_speech"] = speech
            argument = model._spoken_arguments.get(name)
            if argument:
                prop = tool.get("parameters", {}).get("properties", {}).get(argument)
                if prop is None:
                    logger.warning(
                        "spoken argument not found on tool",
                        extra={"tool": name, "argument": argument},
                    )
                else:
                    prop["poly_spoken"] = True
        return event

    def generate_reply(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        tools: NotGivenOr[list[llm.Tool]] = NOT_GIVEN,
    ) -> asyncio.Future[llm.GenerationCreatedEvent]:
        # A reply may override only its instructions; per-reply tools and tool_choice are
        # rejected, so they're dropped here.
        if is_given(tool_choice) or is_given(tools):
            logger.warning(
                "Dialog-RSN-1 doesn't take tools or tool_choice per reply; using the session's"
            )
        return super().generate_reply(instructions=instructions)

    def _on_polyai_server_event(self, event: dict[str, Any]) -> None:
        callback = self._polyai_model._on_response_done
        if callback is None or event.get("type") != "response.done":
            return
        try:
            result = callback(event.get("response") or {})
            if asyncio.iscoroutine(result):
                # Keep a reference until the callback finishes, so it isn't collected mid-run.
                task = asyncio.ensure_future(result)
                _callback_tasks.add(task)
                task.add_done_callback(_callback_tasks.discard)
        except Exception:
            logger.exception("on_response_done callback failed")


def _as_dict(event: Any) -> dict[str, Any]:
    if isinstance(event, dict):
        return event
    dumped: dict[str, Any] = event.model_dump(
        by_alias=True, exclude_unset=True, exclude_defaults=False
    )
    return dumped
