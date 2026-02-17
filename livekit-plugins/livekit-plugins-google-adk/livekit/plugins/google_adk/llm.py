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

from google.adk.agents import BaseAgent, LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import BaseSessionService, InMemorySessionService, Session
from google.genai import types as genai_types

from livekit.agents import APIConnectionError, APIStatusError, llm, utils
from livekit.agents.llm import ToolChoice
from livekit.agents.llm.chat_context import ChatContext, ChatMessage
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import is_given

from .log import logger


class LLMAdapter(llm.LLM):
    """Wraps a Google ADK agent as a LiveKit LLM plugin.

    ADK handles tool calling and multi-agent orchestration internally.
    LiveKit tools passed via ``chat()`` are not used — define tools on the
    ADK agent instead.
    """

    def __init__(
        self,
        agent: LlmAgent | BaseAgent,
        *,
        runner: Runner | None = None,
        session_service: BaseSessionService | None = None,
        app_name: str = "livekit_adk_app",
        model_name: str | None = None,
    ) -> None:
        super().__init__()
        self._agent = agent
        self._app_name = app_name
        self._model_name = model_name

        if session_service is None:
            session_service = InMemorySessionService()
        self._session_service = session_service

        if runner is not None:
            self._runner = runner
        else:
            self._runner = Runner(
                agent=agent,
                app_name=app_name,
                session_service=session_service,
            )

        # cache of (user_id) -> Session for reuse across turns
        self._sessions: dict[str, Session] = {}

    @property
    def model(self) -> str:
        if self._model_name:
            return self._model_name
        agent_model = getattr(self._agent, "model", None)
        if agent_model:
            return str(agent_model)
        return "adk-agent"

    @property
    def provider(self) -> str:
        return "google-adk"

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[llm.Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        # unused — ADK manages its own tool execution loop
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
            extra_kwargs=extra_kwargs if is_given(extra_kwargs) else {},
        )

    async def _get_or_create_session(
        self,
        user_id: str,
        session_id: str | None = None,
    ) -> Session:
        """Return an existing ADK session or create a new one."""
        cache_key = f"{user_id}:{session_id or ''}"
        if cache_key in self._sessions:
            return self._sessions[cache_key]

        session = await self._session_service.create_session(
            app_name=self._app_name,
            user_id=user_id,
            session_id=session_id,
        )
        self._sessions[cache_key] = session
        logger.debug(
            "created ADK session %s for user %s",
            session.id,
            user_id,
        )
        return session


class ADKStream(llm.LLMStream):
    """LLMStream implementation that delegates to a Google ADK Runner."""

    def __init__(
        self,
        llm_instance: LLMAdapter,
        *,
        chat_ctx: ChatContext,
        tools: list[llm.Tool],
        conn_options: APIConnectOptions,
        runner: Runner,
        extra_kwargs: dict[str, Any],
    ) -> None:
        super().__init__(
            llm_instance,
            chat_ctx=chat_ctx,
            tools=tools,
            conn_options=conn_options,
        )
        self._runner = runner
        self._extra_kwargs = extra_kwargs

    async def _run(self) -> None:
        request_id = utils.shortuuid("adk_")

        # Extract the latest user message from the chat context
        user_text = _extract_latest_user_message(self._chat_ctx)
        if not user_text:
            raise APIConnectionError(
                "google-adk: no user message found in chat context",
                retryable=False,
            )

        user_id: str = self._extra_kwargs.get("user_id", "livekit_user")
        session_id: str | None = self._extra_kwargs.get("session_id", None)

        adapter: LLMAdapter = self._llm  # type: ignore[assignment]
        session = await adapter._get_or_create_session(user_id, session_id)

        content = genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=user_text)],
        )

        retryable = True
        has_emitted_partials = False
        try:
            async for event in self._runner.run_async(
                user_id=user_id,
                session_id=session.id,
                new_message=content,
            ):
                # Once we start getting events, errors should not be retried
                retryable = False

                # Emit streaming partial text tokens
                if event.partial and event.content and event.content.parts:
                    for part in event.content.parts:
                        if part.text:
                            has_emitted_partials = True
                            self._event_ch.send_nowait(
                                llm.ChatChunk(
                                    id=request_id,
                                    delta=llm.ChoiceDelta(
                                        role="assistant",
                                        content=part.text,
                                    ),
                                )
                            )

                # Emit final response text (skip if partials already streamed)
                elif event.is_final_response():
                    if not has_emitted_partials and event.content and event.content.parts:
                        for part in event.content.parts:
                            if part.text:
                                self._event_ch.send_nowait(
                                    llm.ChatChunk(
                                        id=request_id,
                                        delta=llm.ChoiceDelta(
                                            role="assistant",
                                            content=part.text,
                                        ),
                                    )
                                )

                    # Report token usage if available
                    if event.usage_metadata:
                        usage = event.usage_metadata
                        self._event_ch.send_nowait(
                            llm.ChatChunk(
                                id=request_id,
                                usage=llm.CompletionUsage(
                                    completion_tokens=usage.candidates_token_count or 0,
                                    prompt_tokens=usage.prompt_token_count or 0,
                                    prompt_cached_tokens=(usage.cached_content_token_count or 0),
                                    total_tokens=usage.total_token_count or 0,
                                ),
                            )
                        )

        except APIConnectionError:
            raise
        except APIStatusError:
            raise
        except Exception as e:
            raise APIConnectionError(
                f"google-adk: error during agent execution: {e}",
                retryable=retryable,
            ) from e


def _extract_latest_user_message(chat_ctx: ChatContext) -> str | None:
    """Return the text of the most recent user message.

    Falls back to the latest developer/system message when no user message
    exists (e.g. ``generate_reply(instructions=...)`` without user input).
    """
    for item in reversed(chat_ctx.items):
        if isinstance(item, ChatMessage) and item.role == "user":
            text = item.text_content
            if text:
                return text

    # fallback: use the latest developer/system message as the prompt
    for item in reversed(chat_ctx.items):
        if isinstance(item, ChatMessage) and item.role in ("developer", "system"):
            text = item.text_content
            if text:
                return text

    return None
