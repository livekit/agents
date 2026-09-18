"""AMD structured model calls and output validation."""

from __future__ import annotations

import json
from dataclasses import replace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from livekit.agents import llm
from livekit.agents.llm.tool_context import get_raw_function_info
from livekit.agents.voice.amd import _inference
from livekit.agents.voice.amd._chat_context import AMDRequest
from livekit.agents.voice.amd.events import AMDCategory

from .fake_llm import FakeLLM, FakeLLMResponse

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]

CHAT_CTX = llm.ChatContext()
CHAT_CTX.add_message(role="user", content="input")
REQUEST = AMDRequest(
    stage=AMDCategory.UNCERTAIN,
    allowed_next_categories=sorted(AMDCategory),
    chat_ctx=CHAT_CTX,
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
    with pytest.raises(ValidationError):
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
                input=json.dumps({"transcript": "input"}) if menu else "input",
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
        expected_schema = schema.model_json_schema()
        if not menu:
            expected_schema["$defs"]["AMDCategory"]["enum"] = REQUEST.allowed_next_categories
        assert get_raw_function_info(tools[0]).raw_schema["parameters"] == expected_schema
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stage",
    [
        AMDCategory.UNCERTAIN,
        AMDCategory.MACHINE_SCREENING,
        AMDCategory.MACHINE_VM,
        AMDCategory.MACHINE_IVR,
    ],
)
@pytest.mark.parametrize("category", list(AMDCategory))
async def test_classifier_schema_and_validation_limit_predictions_to_allowed_states(
    stage: AMDCategory,
    category: AMDCategory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from livekit.agents.voice.amd import _fsm

    request = replace(REQUEST, stage=stage, allowed_next_categories=sorted(_fsm.ALLOWED[stage]))
    model = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="input",
                content="",
                ttft=0,
                duration=0,
                tool_calls=[
                    llm.FunctionToolCall(
                        name="record_result",
                        arguments=json.dumps({"category": category}),
                        call_id="result",
                    )
                ],
            )
        ]
    )
    chat = Mock(wraps=model.chat)
    monkeypatch.setattr(model, "chat", chat)
    try:
        if category in request.allowed_next_categories:
            assert (await _inference.classify(model, request)).category == category
        else:
            with pytest.raises(ValueError, match="not allowed"):
                await _inference.classify(model, request)
        tools = chat.call_args.kwargs["tools"]
        schema = get_raw_function_info(tools[0]).raw_schema["parameters"]
        assert schema["$defs"]["AMDCategory"]["enum"] == request.allowed_next_categories
        tool_ctx = llm.ToolContext(tools)
        for provider in ("openai", "google", "anthropic"):
            assert tool_ctx.parse_function_tools(provider)
    finally:
        await model.aclose()
