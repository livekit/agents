from __future__ import annotations

import json
from typing import Any

import anthropic
import httpx
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


async def _tool_choice(model: str, tool_choice: ToolChoice) -> dict[str, Any]:
    requests: list[dict[str, Any]] = []
    instance = _make_llm(model, requests)
    await instance.chat(
        chat_ctx=ChatContext(), tools=[get_weather], tool_choice=tool_choice
    ).collect()
    return requests[0]["tool_choice"]


async def test_forced_tool_choice_sent_as_auto_for_rejecting_models() -> None:
    for model in ("claude-opus-5-5", "claude-fable-5-1", "claude-mythos-5-1"):
        assert await _tool_choice(model, "required") == {"type": "auto"}
        assert await _tool_choice(model, _NAMED) == {"type": "auto"}


async def test_forced_tool_choice_warning_logged_once(caplog: pytest.LogCaptureFixture) -> None:
    requests: list[dict[str, Any]] = []
    instance = _make_llm("claude-opus-5-5", requests, tool_choice="required")
    with caplog.at_level("WARNING"):
        for _ in range(3):
            await instance.chat(chat_ctx=ChatContext(), tools=[get_weather]).collect()

    warnings = [r for r in caplog.records if "forced tool_choice" in r.getMessage()]
    assert len(warnings) == 1
    assert warnings[0].__dict__.get("lk.pii.model") == "claude-opus-5-5"
    assert [r["tool_choice"] for r in requests] == [{"type": "auto"}] * 3
