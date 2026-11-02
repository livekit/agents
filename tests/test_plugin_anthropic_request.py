from __future__ import annotations

import json
from typing import Any

import anthropic
import httpx2 as httpx
import pytest

from livekit.agents.llm import ChatContext, ToolChoice, function_tool
from livekit.plugins.anthropic import LLM

pytestmark = pytest.mark.unit

_STREAM_RESPONSE = (
    b"event: message_start\n"
    b'data: {"type":"message_start","message":{"id":"msg_test","type":"message",'
    b'"role":"assistant","model":"claude","content":[],"stop_reason":null,'
    b'"stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":1}}}\n\n'
    b"event: message_stop\n"
    b'data: {"type":"message_stop"}\n\n'
)


@function_tool
async def get_weather(city: str) -> str:
    """Look up the weather."""
    return city


_NAMED: ToolChoice = {"type": "function", "function": {"name": "get_weather"}}


def _make_llm(model: str, requests: list[dict[str, Any]], **kwargs: Any) -> LLM:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=_STREAM_RESPONSE
        )

    client = anthropic.AsyncClient(
        api_key="sk-ant-test",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    return LLM(model=model, api_key="sk-ant-test", client=client, **kwargs)


async def _tool_choice(model: str, tool_choice: ToolChoice, **kwargs: Any) -> dict[str, Any]:
    requests: list[dict[str, Any]] = []
    instance = _make_llm(model, requests)
    await instance.chat(
        chat_ctx=ChatContext(), tools=[get_weather], tool_choice=tool_choice, **kwargs
    ).collect()
    return requests[0]["tool_choice"]


async def test_forced_tool_choice_sent_as_auto_only_for_rejecting_models() -> None:
    for model in ("claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-1", "claude-mythos-5-1"):
        assert await _tool_choice(model, "required") == {"type": "auto"}
        assert await _tool_choice(model, _NAMED) == {"type": "auto"}

    assert await _tool_choice("claude-sonnet-5-5", "required", parallel_tool_calls=False) == {
        "type": "auto",
        "disable_parallel_tool_use": True,
    }

    # Opus 5, Sonnet 5, Fable 5 and Haiku 5.5 still accept a forced choice.
    for model in ("claude-opus-5", "claude-sonnet-5", "claude-fable-5", "claude-haiku-5-5"):
        assert await _tool_choice(model, "required") == {"type": "any"}
        assert await _tool_choice(model, _NAMED) == {"type": "tool", "name": "get_weather"}


async def test_forced_tool_choice_warning_logged_once(caplog: pytest.LogCaptureFixture) -> None:
    requests: list[dict[str, Any]] = []
    instance = _make_llm("claude-sonnet-5-5", requests, tool_choice="required")
    with caplog.at_level("WARNING"):
        for _ in range(3):
            await instance.chat(chat_ctx=ChatContext(), tools=[get_weather]).collect()

    warnings = [r for r in caplog.records if "forced tool_choice" in r.getMessage()]
    assert len(warnings) == 1
    assert [r["tool_choice"] for r in requests] == [{"type": "auto"}] * 3


async def _request(**kwargs: Any) -> dict[str, Any]:
    requests: list[dict[str, Any]] = []
    llm_kwargs = kwargs.pop("llm_kwargs", {})
    instance = _make_llm("claude-sonnet-5", requests, **llm_kwargs)
    await instance.chat(chat_ctx=ChatContext(), **kwargs).collect()
    return requests[0]


async def test_parallel_tool_calls_sent_without_tool_choice() -> None:
    # Passed to chat()
    request = await _request(tools=[get_weather], parallel_tool_calls=False)
    assert request["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}

    request = await _request(tools=[get_weather], parallel_tool_calls=True)
    assert request["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": False}

    # Set on the LLM
    request = await _request(tools=[get_weather], llm_kwargs={"parallel_tool_calls": False})
    assert request["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}

    # chat() overrides the LLM option
    request = await _request(
        tools=[get_weather],
        parallel_tool_calls=True,
        llm_kwargs={"parallel_tool_calls": False},
    )
    assert request["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": False}


async def test_tool_choice_unchanged_without_parallel_tool_calls() -> None:
    assert "tool_choice" not in await _request(tools=[get_weather])
    # No tools: nothing to send, even with parallel_tool_calls set.
    request = await _request(parallel_tool_calls=False)
    assert "tool_choice" not in request and "tools" not in request
    # An explicit "none" still drops the tools and sends no tool_choice.
    request = await _request(tools=[get_weather], tool_choice="none", parallel_tool_calls=False)
    assert "tool_choice" not in request
    assert request.get("tools", []) == []
    # Explicit choices keep their type and still carry the flag.
    request = await _request(tools=[get_weather], tool_choice="required", parallel_tool_calls=False)
    assert request["tool_choice"] == {"type": "any", "disable_parallel_tool_use": True}
    request = await _request(tools=[get_weather], tool_choice=_NAMED, parallel_tool_calls=False)
    assert request["tool_choice"] == {
        "type": "tool",
        "name": "get_weather",
        "disable_parallel_tool_use": True,
    }


async def test_user_sent_as_metadata_user_id() -> None:
    for extra_kwargs in ({}, {"metadata": anthropic.omit}):
        request = await _request(llm_kwargs={"user": "user-123"}, extra_kwargs=extra_kwargs)
        assert request["metadata"] == {"user_id": "user-123"}

    metadata = {"user_id": "caller", "trace_id": "abc"}
    request = await _request(llm_kwargs={"user": "user-123"}, extra_kwargs={"metadata": metadata})
    assert request["metadata"] == {"user_id": "user-123", "trace_id": "abc"}
    assert metadata == {"user_id": "caller", "trace_id": "abc"}
