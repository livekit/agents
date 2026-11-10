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

from typing import Any, Generic

from livekit.agents import llm, utils
from livekit.agents.llm import ToolChoice
from livekit.agents.llm.chat_context import ChatContext
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from pydantic_ai import (
    AgentRunResultEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartStartEvent,
    SystemPromptPart,
    TextPart,
    TextPartDelta,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.agent import AbstractAgent
from pydantic_ai.messages import ModelRequestPart, ModelResponsePart
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.usage import RunUsage

from .log import logger


class LLMAdapter(llm.LLM, Generic[AgentDepsT]):
    """Run a Pydantic AI agent as the LLM of a LiveKit ``AgentSession``.

    Tool calls, retries and structured output are handled by Pydantic AI; the adapter streams the
    agent's text output to LiveKit. The LiveKit chat context is converted to Pydantic AI message
    history on every turn, so edits LiveKit makes to the history (for example after an
    interruption) are reflected in the next run.
    """

    def __init__(
        self,
        agent: AbstractAgent[AgentDepsT, Any],
        *,
        deps: AgentDepsT | None = None,
        model_settings: ModelSettings | None = None,
    ) -> None:
        """Create an adapter around a Pydantic AI agent.

        Args:
            agent: The Pydantic AI agent to run on every LLM turn.
            deps: Dependencies passed to the agent's ``deps`` on every run.
            model_settings: Model settings passed to every run.
        """
        super().__init__()
        self._agent = agent
        self._deps = deps
        self._model_settings = model_settings

    @property
    def model(self) -> str:
        model = self._agent.model
        if isinstance(model, Model):
            return model.model_name
        if isinstance(model, str):
            return model
        return "unknown"

    @property
    def provider(self) -> str:
        return "Pydantic AI"

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[llm.Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        # these are unused, since tool execution takes place in Pydantic AI
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> PydanticAIStream[AgentDepsT]:
        if tools:
            logger.warning(
                "LiveKit tools are ignored by the Pydantic AI adapter; "
                "register tools on the Pydantic AI agent instead",
                extra={"tool_count": len(tools)},
            )
        return PydanticAIStream(
            self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
            agent=self._agent,
            deps=self._deps,
            model_settings=self._model_settings,
        )


class PydanticAIStream(llm.LLMStream, Generic[AgentDepsT]):
    def __init__(
        self,
        llm: LLMAdapter[AgentDepsT],
        *,
        chat_ctx: ChatContext,
        tools: list[llm.Tool],
        conn_options: APIConnectOptions,
        agent: AbstractAgent[AgentDepsT, Any],
        deps: AgentDepsT | None,
        model_settings: ModelSettings | None,
    ) -> None:
        super().__init__(llm, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._agent = agent
        self._deps = deps
        self._model_settings = model_settings

    async def _run(self) -> None:
        history = chat_ctx_to_messages(self._chat_ctx)
        if not history:
            return

        chunk_id = utils.shortuuid("PAI_")
        # The whole LiveKit history is passed as message_history: Pydantic AI resumes from a
        # trailing ModelRequest when no user_prompt is given, so the last user turn (or a
        # trailing system message from generate_reply(instructions=...)) is sent as-is.
        async with self._agent.run_stream_events(
            message_history=history,
            deps=self._deps,  # type: ignore[arg-type]
            model_settings=self._model_settings,
        ) as events:
            async for event in events:
                if isinstance(event, PartStartEvent):
                    if isinstance(event.part, TextPart):
                        self._send_text(chunk_id, event.part.content)
                elif isinstance(event, PartDeltaEvent):
                    if isinstance(event.delta, TextPartDelta):
                        self._send_text(chunk_id, event.delta.content_delta)
                elif isinstance(event, AgentRunResultEvent):
                    usage = _to_completion_usage(event.result.usage())
                    if usage is not None:
                        self._event_ch.send_nowait(llm.ChatChunk(id=chunk_id, usage=usage))

    def _send_text(self, chunk_id: str, content: str) -> None:
        if not content:
            return
        self._event_ch.send_nowait(
            llm.ChatChunk(
                id=chunk_id,
                delta=llm.ChoiceDelta(role="assistant", content=content),
            )
        )


def chat_ctx_to_messages(chat_ctx: ChatContext) -> list[ModelMessage]:
    """Convert a LiveKit ``ChatContext`` into Pydantic AI message history.

    Consecutive request-side items (system, developer and user messages, tool outputs) are
    grouped into one ``ModelRequest``; consecutive response-side items (assistant messages and
    function calls) are grouped into one ``ModelResponse``. Tool outputs become ``ToolReturnPart``s
    whether or not they were errors. Items without text and item types Pydantic AI has no
    equivalent for (agent handoffs, config updates) are skipped.
    """
    messages: list[ModelMessage] = []
    request_parts: list[ModelRequestPart] = []
    response_parts: list[ModelResponsePart] = []

    def flush_request() -> None:
        nonlocal request_parts
        if request_parts:
            messages.append(ModelRequest(parts=request_parts))
            request_parts = []

    def flush_response() -> None:
        nonlocal response_parts
        if response_parts:
            messages.append(ModelResponse(parts=response_parts))
            response_parts = []

    for item in chat_ctx.items:
        if item.type == "message":
            content = item.raw_text_content
            if not content:
                continue
            if item.role == "assistant":
                flush_request()
                response_parts.append(TextPart(content=content))
            elif item.role == "user":
                flush_response()
                request_parts.append(UserPromptPart(content=content))
            elif item.role in ("system", "developer"):
                flush_response()
                request_parts.append(SystemPromptPart(content=content))
        elif item.type == "function_call":
            flush_request()
            response_parts.append(
                ToolCallPart(tool_name=item.name, args=item.arguments, tool_call_id=item.call_id)
            )
        elif item.type == "function_call_output":
            # LiveKit tool errors are already rendered as text for the model, so they are
            # passed as the tool's return content as well.
            flush_response()
            request_parts.append(
                ToolReturnPart(tool_name=item.name, content=item.output, tool_call_id=item.call_id)
            )

    flush_request()
    flush_response()
    return messages


def _to_completion_usage(usage: RunUsage) -> llm.CompletionUsage | None:
    if not usage.requests and not usage.input_tokens and not usage.output_tokens:
        return None
    return llm.CompletionUsage(
        completion_tokens=usage.output_tokens,
        prompt_tokens=usage.input_tokens,
        prompt_cached_tokens=usage.cache_read_tokens,
        cache_creation_tokens=usage.cache_write_tokens,
        cache_read_tokens=usage.cache_read_tokens,
    )
