"""LangChain plugin: LLMAdapter over LangGraph graphs."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Annotated

import pytest
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
    GenericFakeChatModel,
)
from langchain_core.messages import AIMessage, AIMessageChunk, AnyMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from livekit.agents import Agent, APIConnectOptions, llm
from livekit.agents.llm import ChatContext
from livekit.plugins.langchain import LLMAdapter


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


@pytest.mark.parametrize("use_generator", [False, True])
async def test_closing_chat_releases_the_graph_iterator(use_generator: bool) -> None:
    """Fix 3ff8eb86a7: graph iterators outlived their chat streams."""
    closed = asyncio.Event()

    class GraphIterator:
        def __aiter__(self) -> GraphIterator:
            return self

        def __init__(self) -> None:
            self._sent = False

        async def __anext__(self) -> str:
            if self._sent:
                await asyncio.Event().wait()
            self._sent = True
            return "hello"

        async def aclose(self) -> None:
            closed.set()

    async def stream_graph() -> AsyncIterator[str]:
        try:
            yield "hello"
            await asyncio.Event().wait()
        finally:
            closed.set()

    class Graph:
        def astream(self, *args: object, **kwargs: object) -> AsyncIterator[str]:
            return stream_graph() if use_generator else GraphIterator()

    # a minimal graph: the adapter only streams it
    model = LLMAdapter(Graph(), stream_mode="custom")  # type: ignore[arg-type]
    async with model.chat(
        chat_ctx=llm.ChatContext(), conn_options=APIConnectOptions(max_retry=0)
    ) as stream:
        first = await anext(stream)
        assert first.delta is not None and first.delta.content == "hello"
    assert closed.is_set(), "the graph iterator outlived its chat stream"


async def test_messages_mode_preserves_complete_nonstreaming_messages() -> None:
    """In messages mode a node's complete, non-streamed message reaches the reply whole."""
    model = FakeMessagesListChatModel(
        responses=[AIMessage(content="Complete response", id="whole")]
    )

    def chat_node(state: MessagesState) -> MessagesState:
        return {"messages": [model.invoke(state["messages"])]}

    graph = StateGraph(MessagesState)
    graph.add_node("chat", chat_node)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)
    adapter = LLMAdapter(graph.compile())
    context = ChatContext()
    context.add_message(role="user", content="Hi")
    async with adapter.chat(chat_ctx=context) as stream:
        chunks = [chunk async for chunk in stream]
    assert [(c.id, c.delta.content if c.delta else None) for c in chunks] == [
        ("whole", "Complete response")
    ]


async def _streamed_text(adapter: LLMAdapter, context: ChatContext) -> list[str]:
    async with adapter.chat(
        chat_ctx=context, conn_options=APIConnectOptions(max_retry=0)
    ) as stream:
        return [
            chunk.delta.content async for chunk in stream if chunk.delta and chunk.delta.content
        ]


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="https://github.com/livekit/agents/issues/4768 reproduces: ToolMessage text is spoken",
)
async def test_tool_node_output_is_not_streamed_as_the_reply() -> None:
    """Only the model's reply is streamed, never a tool's output; https://github.com/livekit/agents/issues/4768 https://github.com/livekit/agents/issues/3064."""

    @tool
    def get_weather(city: str) -> str:
        """Look up the weather."""
        return f"{city}: sunny, 21C"

    def agent(state: MessagesState) -> MessagesState:
        if not any(isinstance(m, ToolMessage) for m in state["messages"]):
            call = {"name": "get_weather", "args": {"city": "Paris"}, "id": "call_1"}
            return {"messages": [AIMessage(content="", tool_calls=[call])]}
        return {"messages": [AIMessage(content="It is sunny in Paris.")]}

    def route(state: MessagesState) -> str:
        last = state["messages"][-1]
        return "tools" if isinstance(last, AIMessage) and last.tool_calls else END

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode([get_weather]))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route)
    graph.add_edge("tools", "agent")
    context = ChatContext()
    context.add_message(role="user", content="What is the weather in Paris?")

    assert await _streamed_text(LLMAdapter(graph.compile()), context) == ["It is sunny in Paris."]


async def test_subgraph_tokens_stream_in_order_across_turns() -> None:
    """Namespaced subgraph events stream as the reply; https://github.com/livekit/agents/issues/3111 https://github.com/livekit/agents/issues/3011."""
    model = GenericFakeChatModel(
        messages=iter([AIMessage("first answer"), AIMessage("second one")])
    )

    async def chat(state: MessagesState) -> MessagesState:
        return {"messages": [await model.ainvoke(state["messages"])]}

    inner = StateGraph(MessagesState)
    inner.add_node("chat", chat)
    inner.add_edge(START, "chat")
    inner.add_edge("chat", END)
    outer = StateGraph(MessagesState)
    outer.add_node("inner", inner.compile())
    outer.add_edge(START, "inner")
    outer.add_edge("inner", END)
    adapter = LLMAdapter(outer.compile(), subgraphs=True)

    context = ChatContext()
    context.add_message(role="user", content="First question")
    first = await _streamed_text(adapter, context)
    context.add_message(role="assistant", content="".join(first))
    context.add_message(role="user", content="Second question")
    second = await _streamed_text(adapter, context)

    assert ("".join(first), "".join(second)) == ("first answer", "second one")


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="https://github.com/livekit/agents/issues/7720 reproduces: prior replies stored twice",
)
async def test_checkpointed_graph_stores_each_prior_reply_once() -> None:
    """A checkpointed graph keeps one copy of each reply; https://github.com/livekit/agents/issues/7720 https://github.com/livekit/agents/pull/2767."""
    model = GenericFakeChatModel(messages=iter([AIMessage("reply 1"), AIMessage("reply 2")]))

    def chat(state: MessagesState) -> MessagesState:
        return {"messages": [model.invoke(state["messages"])]}

    graph = StateGraph(MessagesState)
    graph.add_node("chat", chat)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)
    compiled = graph.compile(checkpointer=InMemorySaver())
    config: RunnableConfig = {"configurable": {"thread_id": "caller"}}
    agent = Agent(instructions="Be brief.", llm=LLMAdapter(compiled, config=config))

    first = await agent.run("Hello")
    second = await agent.run("Again", message_history=first.all_messages())

    assert second.status == "completed"
    assert second.output == "reply 2"
    stored = compiled.get_state(config).values["messages"]
    replies = [m.content for m in stored if isinstance(m, AIMessage)]
    assert replies == ["reply 1", "reply 2"]


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="https://github.com/livekit/agents/issues/6948 reproduces: usage_metadata is dropped",
)
async def test_message_usage_metadata_is_reported_once() -> None:
    """Graph token usage reaches the chat stream; https://github.com/livekit/agents/issues/6948."""
    usage = {"input_tokens": 12, "output_tokens": 4, "total_tokens": 16}

    class Graph:
        async def astream(
            self, *args: object, **kwargs: object
        ) -> AsyncIterator[tuple[AIMessageChunk, dict[str, object]]]:
            yield AIMessageChunk(content="Hello", id="run_1"), {}
            yield AIMessageChunk(content="", id="run_1", usage_metadata=usage), {}

    # a minimal graph: the adapter only streams it
    async with LLMAdapter(Graph()).chat(  # type: ignore[arg-type]
        chat_ctx=ChatContext(), conn_options=APIConnectOptions(max_retry=0)
    ) as stream:
        chunks = [chunk async for chunk in stream]

    reported = [chunk.usage for chunk in chunks if chunk.usage is not None]
    assert [(u.input_tokens, u.output_tokens) for u in reported] == [(12, 4)]
