"""Unit tests for Blaze LLM plugin."""

from __future__ import annotations

import json

import pytest

from livekit.agents import DEFAULT_API_CONNECT_OPTIONS
from livekit.agents.llm import ChatContext, FunctionCallOutput, function_tool
from livekit.plugins.blaze._config import BlazeConfig
from livekit.plugins.blaze.llm import LLM, LLMStream

pytestmark = pytest.mark.unit


def _make_llm(**kwargs: object) -> LLM:
    config = BlazeConfig(api_url="https://api.example.com", api_token="test-token")
    return LLM(bot_id="bot-123", config=config, **kwargs)


def _make_stream(chat_ctx: ChatContext, *, tools: list | None = None) -> LLMStream:
    llm = _make_llm(enable_tools=True)
    stream = object.__new__(LLMStream)
    stream._blaze_llm = llm
    stream._chat_ctx = chat_ctx
    stream._tools = tools or []
    stream._conn_options = DEFAULT_API_CONNECT_OPTIONS
    return stream


def test_llm_provider_and_model() -> None:
    llm = _make_llm()
    assert llm.provider == "Blaze"
    assert llm.model == "bot-123"
    assert llm.bot_id == "bot-123"


def test_llm_update_options() -> None:
    llm = _make_llm()
    llm.update_options(
        deep_search=True,
        agentic_search=True,
        enable_tools=True,
        demographics={"gender": "female", "age": 30},
        auth_token="new-token",
    )

    assert llm._deep_search is True
    assert llm._agentic_search is True
    assert llm._enable_tools is True
    assert llm._demographics == {"gender": "female", "age": 30}
    assert llm._auth_token == "new-token"


def test_convert_messages_skips_system_and_developer() -> None:
    chat_ctx = ChatContext()
    chat_ctx.add_message(role="system", content="You are helpful.")
    chat_ctx.add_message(role="developer", content="Hidden developer prompt.")
    chat_ctx.add_message(role="user", content="Xin chào")

    messages = _make_stream(chat_ctx)._convert_messages()

    assert messages == [{"role": "user", "content": "Xin chào"}]


def test_convert_messages_maps_assistant_and_strips_img_tags() -> None:
    chat_ctx = ChatContext()
    chat_ctx.add_message(role="assistant", content="Hello <img>chart</img> world.")

    messages = _make_stream(chat_ctx)._convert_messages()

    assert messages == [{"role": "assistant", "content": "Hello  world."}]


def test_convert_messages_includes_function_output_as_user() -> None:
    chat_ctx = ChatContext()
    chat_ctx.items.append(
        FunctionCallOutput(
            call_id="call_1",
            name="lookup",
            output="result payload",
            is_error=False,
        )
    )

    messages = _make_stream(chat_ctx)._convert_messages()

    assert messages == [{"role": "user", "content": "result payload"}]


def test_convert_messages_wraps_tool_errors() -> None:
    chat_ctx = ChatContext()
    chat_ctx.items.append(
        FunctionCallOutput(
            call_id="call_1",
            name="lookup",
            output="timeout",
            is_error=True,
        )
    )

    messages = _make_stream(chat_ctx)._convert_messages()

    assert messages == [{"role": "user", "content": "[Tool Error]: timeout"}]


@function_tool
async def sample_tool(query: str) -> str:
    """Look up information."""
    return query


def test_build_tools_param_serializes_function_tools() -> None:
    chat_ctx = ChatContext()
    stream = _make_stream(chat_ctx, tools=[sample_tool])

    tools_param = stream._build_tools_param()

    assert tools_param is not None
    assert len(tools_param) == 1
    assert tools_param[0]["type"] == "function"
    assert tools_param[0]["function"]["name"] == "sample_tool"


def test_build_tools_param_returns_none_without_tools() -> None:
    chat_ctx = ChatContext()
    assert _make_stream(chat_ctx)._build_tools_param() is None


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"content": "hello"}, "hello"),
        ({"text": "world"}, "world"),
        ({"delta": {"text": "delta-text"}}, "delta-text"),
        ({"delta": {"content": "delta-content"}}, "delta-content"),
        ({"choices": []}, None),
    ],
)
def test_extract_content(payload: dict, expected: str | None) -> None:
    stream = _make_stream(ChatContext())
    assert stream._extract_content(payload) == expected


def test_extract_tool_calls_openai_format() -> None:
    stream = _make_stream(ChatContext())
    data = {
        "tool_calls": [
            {
                "id": "call_abc",
                "function": {"name": "sample_tool", "arguments": '{"query":"hi"}'},
            }
        ]
    }

    calls = stream._extract_tool_calls(data)

    assert len(calls) == 1
    assert calls[0].name == "sample_tool"
    assert calls[0].call_id == "call_abc"
    assert json.loads(calls[0].arguments) == {"query": "hi"}


def test_extract_tool_calls_delta_format() -> None:
    stream = _make_stream(ChatContext())
    data = {
        "delta": {
            "tool_calls": [
                {
                    "id": "call_delta",
                    "function": {"name": "sample_tool", "arguments": {}},
                }
            ]
        }
    }

    calls = stream._extract_tool_calls(data)

    assert len(calls) == 1
    assert calls[0].name == "sample_tool"
    assert calls[0].arguments == "{}"
