from __future__ import annotations

from livekit.agents import llm

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
        self._chat_engine = chat_engine

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        fnc_ctx: llm.FunctionContext | None = None,
        temperature: float | None = None,
        n: int | None = 1,
        parallel_tool_calls: bool | None = None,
    ) -> "LLMStream":
        if fnc_ctx is not None:
            logger.warning("fnc_ctx is currently not supported with llama_index.LLM")

        return LLMStream(
            chat_engine=self._chat_engine,
            chat_ctx=chat_ctx,
        )


class LLMStream(llm.LLMStream):
    def __init__(
        self,
        *,
        chat_engine: BaseChatEngine,
        chat_ctx: llm.ChatContext,
    ) -> None:
        super().__init__(chat_ctx=chat_ctx, fnc_ctx=None)
        self._chat_engine = chat_engine
        self._stream: StreamingAgentChatResponse | None = None

    async def __anext__(self) -> llm.ChatChunk:
        # remove and get the last message from the chat_ctx
        chat_ctx = self._chat_ctx.copy()
        user_msg = chat_ctx.messages.pop()
        if user_msg.role != "user":
            raise ValueError("last message in chat_ctx must be from user")

        assert isinstance(
            user_msg.content, str
        ), "user message content must be a string"

        if not self._stream:
            self._stream = await self._chat_engine.astream_chat(
                user_msg.content, chat_history=_to_llama_chat_messages(self._chat_ctx)
            )

        async for delta in self._stream.async_response_gen():
            return llm.ChatChunk(
                choices=[
                    llm.Choice(
                        delta=llm.ChoiceDelta(
                            role="assistant",
                            content=delta,
                        )
                    )
                ]
            )

        raise StopAsyncIteration


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
