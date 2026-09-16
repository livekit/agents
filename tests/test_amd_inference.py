"""AMD structured model calls and output validation."""

from __future__ import annotations

import json
from unittest.mock import Mock

import pytest

from livekit.agents import llm
from livekit.agents.llm.tool_context import get_raw_function_info
from livekit.agents.voice.amd import _inference
from livekit.agents.voice.amd._fsm import AMDClassifyRequest, AMDTurnContext
from livekit.agents.voice.amd.events import AMDCategory

from .fake_llm import FakeLLM, FakeLLMResponse

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]

REQUEST = AMDClassifyRequest(
    stage=AMDCategory.UNCERTAIN,
    allowed_next_categories=sorted(AMDCategory),
    current_turn=AMDTurnContext(
        turn_id=1, transcript="input", transcript_source=None, dtmf_digits=""
    ),
    earlier_turns=[],
    speech_duration=0.5,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    [
        "not JSON",
        "[]",
        '{"category":"unknown"}',
        '```json\n{"category":"human"}\n```',
    ],
)
async def test_classifier_rejects_invalid_tool_arguments(arguments: str) -> None:
    model = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="input",
                content="",
                ttft=0,
                duration=0,
                tool_calls=[
                    llm.FunctionToolCall(
                        name="record_result", arguments=arguments, call_id="result"
                    )
                ],
            )
        ]
    )
    with pytest.raises(ValueError):
        await _inference.classify(model, REQUEST)


@pytest.mark.asyncio
@pytest.mark.parametrize("menu", [False, True])
@pytest.mark.parametrize("fallback", [False, True])
async def test_amd_uses_a_required_structured_tool(
    monkeypatch: pytest.MonkeyPatch, menu: bool, fallback: bool
) -> None:
    payload = (
        {"menu": "Main menu", "options": [{"label": "Talk to a person", "dtmf": "1"}]}
        if menu
        else {"category": "human"}
    )
    model = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input=json.dumps({"transcript": "input"})
                if menu
                else REQUEST.model_dump_json(exclude_none=True),
                content="",
                ttft=0,
                duration=0,
                tool_calls=[
                    llm.FunctionToolCall(
                        name="record_result", arguments=json.dumps(payload), call_id="result"
                    )
                ],
            )
        ]
    )
    chat = Mock(wraps=model.chat)
    monkeypatch.setattr(model, "chat", chat)
    adapter = llm.FallbackAdapter([model]) if fallback else model
    try:
        if menu:
            result = await _inference.extract_ivr_menu(adapter, "input")
            schema = _inference.AMDIVRMenuResponse
        else:
            result = await _inference.classify(adapter, REQUEST)
            schema = _inference.AMDResponse
        assert result == schema.model_validate(payload)
        chat.assert_called_once()
        assert chat.call_args.kwargs["tool_choice"] == "required"
        assert chat.call_args.kwargs["parallel_tool_calls"] is False
        tools = chat.call_args.kwargs["tools"]
        assert len(tools) == 1
        assert (
            get_raw_function_info(tools[0]).raw_schema["parameters"] == schema.model_json_schema()
        )
        tool_ctx = llm.ToolContext(tools)
        assert tool_ctx.parse_function_tools("openai")
        assert tool_ctx.parse_function_tools("google")
        assert tool_ctx.parse_function_tools("anthropic")
    finally:
        await adapter.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("names", [[], ["unexpected"], ["record_result", "record_result"]])
async def test_classifier_requires_exactly_one_result_tool(names: list[str]) -> None:
    model = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="input",
                content='{"category":"human"}',
                ttft=0,
                duration=0,
                tool_calls=[
                    llm.FunctionToolCall(
                        name=name, arguments='{"category":"human"}', call_id=str(i)
                    )
                    for i, name in enumerate(names)
                ],
            )
        ]
    )
    with pytest.raises(ValueError, match="exactly one record_result"):
        await _inference.classify(model, REQUEST)
