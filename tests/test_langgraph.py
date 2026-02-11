"""Tests for LangGraph LLMAdapter stream_mode support."""

from __future__ import annotations

import sys
from itertools import cycle
from typing import Annotated

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import StreamWriter
from typing_extensions import TypedDict

from langchain_core.messages import AIMessageChunk

from livekit.agents.llm import ChatContext
from livekit.plugins.langchain import LLMAdapter

# --- State definitions ---


class MessagesState(TypedDict):
    messages: Annotated[list, add_messages]


class CustomState(TypedDict):
    messages: Annotated[list, add_messages]
    custom_output: str


# --- Graph builders ---


def _llm_chain():
    """Build a Runnable chain that LangGraph can intercept for message streaming."""
    fake_llm = GenericFakeChatModel(messages=cycle([AIMessage(content="Hello world from fake")]))
    return (
        RunnableLambda(lambda state: state["messages"])
        | fake_llm
        | RunnableLambda(lambda msg: {"messages": [msg]})
    )


def build_messages_graph():
    """Graph with fake LLM that streams AIMessageChunk tokens."""
    graph = StateGraph(MessagesState)
    graph.add_node("chat", _llm_chain())
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)
    return graph.compile()


def build_custom_graph():
    """Graph with StreamWriter that emits custom payloads."""

    def stream_node(state: CustomState, writer: StreamWriter):
        writer("chunk1")
        writer("chunk2")
        writer({"content": "chunk3"})
        return {"custom_output": "done"}

    graph = StateGraph(CustomState)
    graph.add_node("stream", stream_node)
    graph.add_edge(START, "stream")
    graph.add_edge("stream", END)
    return graph.compile()


def build_combined_graph():
    """Graph with both fake LLM and StreamWriter for multi-mode testing."""
    fake_llm = GenericFakeChatModel(messages=cycle([AIMessage(content="LLM response")]))
    chain = (
        RunnableLambda(lambda state: state["messages"])
        | fake_llm
        | RunnableLambda(lambda msg: {"messages": [msg]})
    )

    def stream_node(state: CustomState, writer: StreamWriter):
        writer("custom chunk")
        return {"custom_output": "done"}

    graph = StateGraph(CustomState)
    graph.add_node("chat", chain)
    graph.add_node("stream", stream_node)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", "stream")
    graph.add_edge("stream", END)
    return graph.compile()


# --- Helper ---


async def collect_chunks(stream) -> list[str]:
    """Collect all content chunks from a stream."""
    chunks = []
    async for chunk in stream:
        if chunk.delta and chunk.delta.content:
            chunks.append(chunk.delta.content)
    return chunks


# --- Tests: messages mode ---


@pytest.mark.asyncio
async def test_messages_mode():
    """Test stream_mode='messages' with fake LLM producing AIMessageChunk tokens."""
    graph = build_messages_graph()
    adapter = LLMAdapter(graph, stream_mode="messages")

    chat_ctx = ChatContext()
    chat_ctx.add_message(role="user", content="Hi")
    stream = adapter.chat(chat_ctx=chat_ctx)
    chunks = await collect_chunks(stream)

    # GenericFakeChatModel splits "Hello world from fake" on whitespace
    assert len(chunks) > 0
    combined = "".join(chunks)
    assert "Hello" in combined
    assert "world" in combined


@pytest.mark.asyncio
async def test_messages_mode_is_default():
    """Test that messages mode is the default behavior."""
    graph = build_messages_graph()
    adapter = LLMAdapter(graph)  # No stream_mode specified

    chat_ctx = ChatContext()
    chat_ctx.add_message(role="user", content="Hi")
    stream = adapter.chat(chat_ctx=chat_ctx)
    chunks = await collect_chunks(stream)

    assert len(chunks) > 0
    combined = "".join(chunks)
    assert "Hello" in combined


# --- Tests: custom mode ---


@pytest.mark.asyncio
@pytest.mark.skipif(sys.version_info < (3, 11), reason="StreamWriter requires Python 3.11+")
async def test_custom_mode_string():
    """Test stream_mode='custom' with string payloads from StreamWriter."""
    graph = build_custom_graph()
    adapter = LLMAdapter(graph, stream_mode="custom")

    chat_ctx = ChatContext()
    chat_ctx.add_message(role="user", content="Hi")
    stream = adapter.chat(chat_ctx=chat_ctx)
    chunks = await collect_chunks(stream)

    assert "chunk1" in chunks
    assert "chunk2" in chunks


@pytest.mark.asyncio
@pytest.mark.skipif(sys.version_info < (3, 11), reason="StreamWriter requires Python 3.11+")
async def test_custom_mode_dict():
    """Test stream_mode='custom' with dict payload containing 'content' key."""
    graph = build_custom_graph()
    adapter = LLMAdapter(graph, stream_mode="custom")

    chat_ctx = ChatContext()
    chat_ctx.add_message(role="user", content="Hi")
    stream = adapter.chat(chat_ctx=chat_ctx)
    chunks = await collect_chunks(stream)

    # {"content": "chunk3"} should be converted to "chunk3"
    assert "chunk3" in chunks


# --- Tests: multi mode ---


@pytest.mark.asyncio
@pytest.mark.skipif(sys.version_info < (3, 11), reason="StreamWriter requires Python 3.11+")
async def test_multi_mode():
    """Test stream_mode=['messages', 'custom'] handles both formats."""
    graph = build_combined_graph()
    adapter = LLMAdapter(graph, stream_mode=["messages", "custom"])

    chat_ctx = ChatContext()
    chat_ctx.add_message(role="user", content="Hi")
    stream = adapter.chat(chat_ctx=chat_ctx)
    chunks = await collect_chunks(stream)

    combined = "".join(chunks)
    # Should have chunks from both messages mode (LLM) and custom mode (StreamWriter)
    assert "LLM" in combined or "response" in combined  # From fake LLM
    assert "custom chunk" in combined  # From StreamWriter


# --- Tests: validation ---


def test_validation_rejects_unsupported_mode():
    """Test that unsupported stream modes are rejected."""
    graph = build_messages_graph()

    with pytest.raises(ValueError, match="Unsupported stream mode"):
        LLMAdapter(graph, stream_mode="values")


def test_validation_rejects_unsupported_in_list():
    """Test that unsupported modes in a list are rejected."""
    graph = build_messages_graph()

    with pytest.raises(ValueError, match="Unsupported stream mode"):
        LLMAdapter(graph, stream_mode=["messages", "updates"])


def test_validation_accepts_supported_modes():
    """Test that supported modes are accepted."""
    graph = build_messages_graph()

    # Should not raise
    LLMAdapter(graph, stream_mode="messages")
    LLMAdapter(graph, stream_mode="custom")
    LLMAdapter(graph, stream_mode=["messages", "custom"])


# --- Tests: mode isolation ---


@pytest.mark.asyncio
@pytest.mark.skipif(sys.version_info < (3, 11), reason="StreamWriter requires Python 3.11+")
async def test_empty_stream_mode_disables_streaming():
    """Test stream_mode=[] produces no output (opt-out of streaming)."""
    graph = build_combined_graph()  # Has both LLM and StreamWriter
    adapter = LLMAdapter(graph, stream_mode=[])

    chat_ctx = ChatContext()
    chat_ctx.add_message(role="user", content="Hi")
    stream = adapter.chat(chat_ctx=chat_ctx)
    chunks = await collect_chunks(stream)

    assert chunks == []


@pytest.mark.asyncio
async def test_custom_mode_no_messages_output():
    """Test stream_mode='custom' produces nothing when graph only has LLM."""
    graph = build_messages_graph()  # LLM only, no StreamWriter
    adapter = LLMAdapter(graph, stream_mode="custom")

    chat_ctx = ChatContext()
    chat_ctx.add_message(role="user", content="Hi")
    stream = adapter.chat(chat_ctx=chat_ctx)
    chunks = await collect_chunks(stream)

    assert chunks == []


@pytest.mark.asyncio
@pytest.mark.skipif(sys.version_info < (3, 11), reason="StreamWriter requires Python 3.11+")
async def test_messages_mode_no_custom_output():
    """Test stream_mode='messages' produces nothing when graph only has StreamWriter."""
    graph = build_custom_graph()  # StreamWriter only, no LLM
    adapter = LLMAdapter(graph, stream_mode="messages")

    chat_ctx = ChatContext()
    chat_ctx.add_message(role="user", content="Hi")
    stream = adapter.chat(chat_ctx=chat_ctx)
    chunks = await collect_chunks(stream)

    assert chunks == []


# --- Tests: subgraph namespace stripping ---
# These tests use a mock graph to verify the namespace stripping logic
# with the exact tuple shapes LangGraph emits when subgraphs=True.
#
# Real LangGraph tuple shapes:
#   subgraphs=True, single mode:
#     ((),            data)               — root graph
#     (('sub:uuid',), data)               — subgraph
#   subgraphs=True, multi mode:
#     ((),            'mode_name', data)   — root graph
#     (('sub:uuid',), 'mode_name', data)  — subgraph


class MockGraph:
    """Mock graph that yields pre-defined items from astream()."""

    def __init__(self, items: list):
        self._items = items

    async def astream(self, state, config=None, **kwargs):
        for item in self._items:
            yield item


@pytest.mark.asyncio
async def test_subgraph_messages_mode():
    """Test namespace stripping for messages mode with subgraphs=True."""
    meta = {"langgraph_step": 1}
    chunk_root = AIMessageChunk(content="root", id="r1")
    chunk_sub = AIMessageChunk(content="sub", id="s1")

    # Exact shapes from LangGraph: (namespace_tuple, (message, metadata))
    mock = MockGraph([
        ((), (chunk_root, meta)),
        (("node_2:abc123",), (chunk_sub, meta)),
    ])
    adapter = LLMAdapter(mock, stream_mode="messages", subgraphs=True)

    chat_ctx = ChatContext()
    chat_ctx.add_message(role="user", content="Hi")
    stream = adapter.chat(chat_ctx=chat_ctx)
    chunks = await collect_chunks(stream)

    assert "root" in chunks
    assert "sub" in chunks


@pytest.mark.asyncio
async def test_subgraph_custom_mode():
    """Test namespace stripping for custom mode with subgraphs=True."""
    # Exact shapes: (namespace_tuple, raw_value)
    mock = MockGraph([
        ((), "root_chunk"),
        (("node_2:abc123",), "sub_chunk"),
        (("node_2:abc123",), {"content": "sub_dict"}),
    ])
    adapter = LLMAdapter(mock, stream_mode="custom", subgraphs=True)

    chat_ctx = ChatContext()
    chat_ctx.add_message(role="user", content="Hi")
    stream = adapter.chat(chat_ctx=chat_ctx)
    chunks = await collect_chunks(stream)

    assert "root_chunk" in chunks
    assert "sub_chunk" in chunks
    assert "sub_dict" in chunks


@pytest.mark.asyncio
async def test_subgraph_multi_mode():
    """Test namespace stripping for multi-mode with subgraphs=True."""
    meta = {"langgraph_step": 1}
    chunk = AIMessageChunk(content="msg", id="m1")

    # Exact shapes: (namespace_tuple, mode_string, data)
    mock = MockGraph([
        ((), "messages", (chunk, meta)),
        ((), "custom", "custom_root"),
        (("node_2:abc123",), "messages", (chunk, meta)),
        (("node_2:abc123",), "custom", "custom_sub"),
    ])
    adapter = LLMAdapter(mock, stream_mode=["messages", "custom"], subgraphs=True)

    chat_ctx = ChatContext()
    chat_ctx.add_message(role="user", content="Hi")
    stream = adapter.chat(chat_ctx=chat_ctx)
    chunks = await collect_chunks(stream)

    assert "msg" in chunks
    assert "custom_root" in chunks
    assert "custom_sub" in chunks
