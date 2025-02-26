from __future__ import annotations

from typing import Literal, Union

from livekit.agents import (
    APIConnectionError,
    llm,
)
from livekit.agents.llm import ToolChoice
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

from llama_index.core.chat_engine.types import (
    BaseChatEngine,
    StreamingAgentChatResponse,
)
from llama_index.core.llms import ChatMessage, MessageRole

from .log import logger


class LLM(llm.LLM):
    def __init__(
        self,
        *,
        chat_engine: BaseChatEngine,
    ) -> None:
        super().__init__()
        self._chat_engine = chat_engine

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        fnc_ctx: llm.FunctionContext | None = None,
        temperature: float | None = None,
        n: int | None = 1,
        parallel_tool_calls: bool | None = None,
        tool_choice: Union[ToolChoice, Literal["auto", "required", "none"]] | None = None,
    ) -> "LLMStream":
        if fnc_ctx is not None:
            logger.warning("fnc_ctx is currently not supported with llama_index.LLM")

        return LLMStream(
            self,
            chat_engine=self._chat_engine,
            chat_ctx=chat_ctx,
            conn_options=conn_options,
        )


class LLMStream(llm.LLMStream):
    def __init__(
        self,
        llm: LLM,
        *,
        chat_engine: BaseChatEngine,
        chat_ctx: llm.ChatContext,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(llm, chat_ctx=chat_ctx, fnc_ctx=None, conn_options=conn_options)
        self._chat_engine = chat_engine
        self._stream: StreamingAgentChatResponse | None = None

    async def _run(self) -> None:
        chat_ctx = self._chat_ctx.copy()
        user_msg = chat_ctx.messages.pop()

        if user_msg.role != "user":
            raise ValueError("The last message in the chat context must be from the user")

        assert isinstance(user_msg.content, str), "user message content must be a string"

        try:
            if not self._stream:
                self._stream = await self._chat_engine.astream_chat(
                    user_msg.content,
                    chat_history=_to_llama_chat_messages(self._chat_ctx),
                )

            async for delta in self._stream.async_response_gen():
                self._event_ch.send_nowait(
                    llm.ChatChunk(
                        request_id="",
                        choices=[
                            llm.Choice(
                                delta=llm.ChoiceDelta(
                                    role="assistant",
                                    content=delta,
                                )
                            )
                        ],
                    )
                )
        except Exception as e:
            raise APIConnectionError() from e


def _to_llama_chat_messages(chat_ctx: llm.ChatContext) -> list[ChatMessage]:
    return [
        ChatMessage(content=msg.content, role=_to_llama_message_role(msg.role))
        for msg in chat_ctx.messages
    ]


def _to_llama_message_role(chat_role: llm.ChatRole) -> MessageRole:
    if chat_role == "system":
        return MessageRole.SYSTEM
    elif chat_role == "user":
        return MessageRole.USER
    elif chat_role == "assistant":
        return MessageRole.ASSISTANT
    elif chat_role == "tool":
        return MessageRole.TOOL
