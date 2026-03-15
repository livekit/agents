# Copyright 2025 LiveKit, Inc.
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

from __future__ import annotations

from typing import Any

from google.adk.runners import Runner
from google.genai import types

from livekit.agents import llm, utils
from livekit.agents.llm import ToolChoice
from livekit.agents.llm.chat_context import ChatContext
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)


class LLMAdapter(llm.LLM):
    """Adapts a Google ADK Runner to the LiveKit ``llm.LLM`` interface.

    The ADK Runner manages its own session, tool execution, and multi-step
    reasoning. This adapter bridges the gap by extracting the latest user
    message from a LiveKit ``ChatContext``, forwarding it to the ADK Runner,
    and streaming the agent's text response back as ``ChatChunk`` events.

    Instructions and tools should be configured on the ADK ``LlmAgent`` rather
    than on the LiveKit ``Agent`` (set ``instructions=""`` on the LiveKit side).
    """

    def __init__(
        self,
        runner: Runner,
        *,
        user_id: str = "livekit-user",
    ) -> None:
        """Create a new LLMAdapter.

        Args:
            runner: A configured Google ADK ``Runner`` wrapping an agent.
            user_id: User identifier for the ADK session. Defaults to
                ``"livekit-user"``.
        """
        super().__init__()
        self._runner = runner
        self._user_id = user_id
        self._session_id: str | None = None

    @property
    def model(self) -> str:
        agent = self._runner.agent
        if hasattr(agent, "model") and agent.model:
            return str(agent.model)
        return "unknown"

    @property
    def provider(self) -> str:
        return "google-adk"

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[llm.Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        # these are unused, since tool execution takes place inside ADK
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> ADKStream:
        return ADKStream(
            self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
            runner=self._runner,
            user_id=self._user_id,
        )


class ADKStream(llm.LLMStream):
    """Streams events from a Google ADK Runner execution."""

    def __init__(
        self,
        llm_adapter: LLMAdapter,
        *,
        chat_ctx: ChatContext,
        tools: list[llm.Tool],
        conn_options: APIConnectOptions,
        runner: Runner,
        user_id: str,
    ):
        super().__init__(
            llm_adapter,
            chat_ctx=chat_ctx,
            tools=tools,
            conn_options=conn_options,
        )
        self._runner = runner
        self._user_id = user_id
        self._adapter = llm_adapter

    async def _run(self) -> None:
        # lazily create an ADK session on the first call
        if self._adapter._session_id is None:
            session = await self._runner.session_service.create_session(
                app_name=self._runner.app_name,
                user_id=self._user_id,
            )
            self._adapter._session_id = session.id

        content = _extract_latest_user_content(self._chat_ctx)
        if content is None:
            return

        async for event in self._runner.run_async(
            user_id=self._user_id,
            session_id=self._adapter._session_id,
            new_message=content,
        ):
            if not event.content or not event.content.parts:
                continue

            # only forward the agent's final user-facing response
            if not event.is_final_response():
                continue

            for part in event.content.parts:
                if part.text:
                    self._event_ch.send_nowait(
                        llm.ChatChunk(
                            id=event.id or utils.shortuuid("adk_"),
                            delta=llm.ChoiceDelta(
                                role="assistant",
                                content=part.text,
                            ),
                        )
                    )


def _extract_latest_user_content(chat_ctx: ChatContext) -> types.Content | None:
    """Extract the most recent user message from a ChatContext as ADK Content."""
    for msg in reversed(chat_ctx.messages()):
        if msg.role == "user":
            text = msg.text_content
            if text:
                return types.Content(
                    role="user",
                    parts=[types.Part(text=text)],
                )
    return None
