from __future__ import annotations

import asyncio
import copy
import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from livekit.agents.llm import (
    LLM,
    ChatChunk,
    ChatContext,
    ChoiceDelta,
    FunctionTool,
    FunctionToolCall,
    LLMStream,
    RawFunctionTool,
    ToolChoice,
)
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)


class FakeLLMResponse(BaseModel):
    """Map from input text to output content, tool calls, ttft, and duration"""

    type: Literal["llm"] = "llm"
    input: str
    content: str
    ttft: float
    duration: float
    tool_calls: list[FunctionToolCall] = Field(default_factory=list)

    def speed_up(self, factor: float) -> FakeLLMResponse:
        obj = copy.deepcopy(self)
        obj.ttft /= factor
        obj.duration /= factor
        return obj


class FakeLLM(LLM):
    def __init__(self, *, fake_responses: list[FakeLLMResponse] | None = None) -> None:
        super().__init__()

        self._fake_response_map = (
            {resp.input: resp for resp in fake_responses} if fake_responses else {}
        )

    @property
    def fake_response_map(self) -> dict[str, FakeLLMResponse]:
        return self._fake_response_map

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[FunctionTool | RawFunctionTool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> LLMStream:
        return FakeLLMStream(self, chat_ctx=chat_ctx, tools=tools or [], conn_options=conn_options)


class FakeLLMStream(LLMStream):
    def __init__(
        self,
        llm: FakeLLM,
        *,
        chat_ctx: ChatContext,
        tools: list[FunctionTool | RawFunctionTool],
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(llm, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._llm = llm

    async def _run(self) -> None:
        start_time = time.time()

        index_text = self._get_index_text()
        if index_text not in self._llm.fake_response_map:
            raise ValueError(f"No response found for input: {index_text}")

        resp = self._llm.fake_response_map[index_text]

        await asyncio.sleep(resp.ttft)
        chunk_size = 3
        num_chunks = max(1, len(resp.content) // chunk_size + 1)
        for i in range(num_chunks):
            delta = resp.content[i * chunk_size : (i + 1) * chunk_size]
            self._send_chunk(delta=delta)

        self._send_chunk(tool_calls=resp.tool_calls)

        await asyncio.sleep(resp.duration - (time.time() - start_time))

    def _send_chunk(
        self, *, delta: str | None = None, tool_calls: list[FunctionToolCall] | None = None
    ) -> None:
        self._event_ch.send_nowait(
            ChatChunk(
                id=str(id(self)),
                delta=ChoiceDelta(
                    role="assistant",
                    content=delta,
                    tool_calls=tool_calls or [],
                ),
            )
        )

    def _get_index_text(self) -> str:
        assert self.chat_ctx.items

        # if the last item is a user message
        items = self.chat_ctx.items
        if items[-1].type == "message" and items[-1].role == "user":
            return items[-1].text_content

        # if the last item is a function call output, use the tool output
        if items[-1].type == "function_call_output":
            return items[-1].output

        # find the first system message
        for item in items:
            if item.type == "message" and item.role == "system":
                return item.text_content

        raise ValueError("No input text found")
