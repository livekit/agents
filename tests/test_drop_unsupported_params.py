from __future__ import annotations

from typing import Any

import pytest
from openai.types import Reasoning

from livekit.agents.inference.llm import drop_unsupported_params

pytestmark = pytest.mark.unit

_GPT6_SAMPLING = {
    "temperature": 0.2,
    "top_p": 0.9,
    "presence_penalty": 0.5,
    "frequency_penalty": 0.5,
    "logit_bias": {"5": 1},
    "logprobs": True,
    "top_logprobs": 3,
    "stop": ["x"],
}


@pytest.mark.parametrize(
    ("model", "params", "expected"),
    [
        # gpt-5.1+ accept temperature/top_p at effort "none" and reject them while reasoning
        (
            "openai/gpt-5.6-luna",
            {"temperature": 0.2, "top_p": 0.9, "reasoning_effort": "none"},
            {"temperature": 0.2, "top_p": 0.9, "reasoning_effort": "none"},
        ),
        (
            "gpt-5.1",
            {"temperature": 0.2, "reasoning_effort": "none"},
            {"temperature": 0.2, "reasoning_effort": "none"},
        ),
        (
            "openai/gpt-5.4-nano",
            {"temperature": 0.2, "reasoning_effort": "none"},
            {"temperature": 0.2, "reasoning_effort": "none"},
        ),
        (
            "openai/gpt-5.6-luna",
            {"temperature": 0.2, "top_p": 0.9, "reasoning_effort": "low"},
            {"reasoning_effort": "low"},
        ),
        ("openai/gpt-5.6-luna", {"temperature": 0.2, "top_p": 0.9}, {}),
        # the other sampling params are unsupported on gpt-5.x regardless of effort
        (
            "openai/gpt-5.6-luna",
            {
                "temperature": 0.2,
                "frequency_penalty": 0.5,
                "logit_bias": {"5": 1},
                "reasoning_effort": "none",
            },
            {"temperature": 0.2, "reasoning_effort": "none"},
        ),
        # gpt-5/mini/nano's lowest effort is "minimal", so the blanket strip stays
        (
            "openai/gpt-5",
            {"temperature": 0.2, "top_p": 0.9, "reasoning_effort": "minimal"},
            {"reasoning_effort": "minimal"},
        ),
        ("openai/gpt-5-mini", {"temperature": 0.2}, {}),
        # chat-latest variants have no effort floor entry, so they keep the blanket strip
        (
            "openai/gpt-5.1-chat-latest",
            {"temperature": 0.2, "reasoning_effort": "none"},
            {"reasoning_effort": "none"},
        ),
        (
            "openai/o3",
            {"temperature": 0.2, "top_p": 0.9, "reasoning_effort": "none"},
            {"reasoning_effort": "none"},
        ),
        ("openai/gpt-4o", {"temperature": 0.2, "top_p": 0.9}, {"temperature": 0.2, "top_p": 0.9}),
        # xAI reasoning models support temperature/top_p
        (
            "grok-4.20-multi-agent",
            {"temperature": 0.2, "top_p": 0.9, "frequency_penalty": 0.5},
            {"temperature": 0.2, "top_p": 0.9},
        ),
        # the Responses shape carries effort as `reasoning`, an object or a dict
        (
            "gpt-5.6-luna",
            {"temperature": 0.2, "top_p": 0.9, "reasoning": Reasoning(effort="none")},
            {"temperature": 0.2, "top_p": 0.9, "reasoning": Reasoning(effort="none")},
        ),
        (
            "gpt-5.6-luna",
            {"temperature": 0.2, "reasoning": {"effort": "none", "summary": "auto"}},
            {"temperature": 0.2, "reasoning": {"effort": "none", "summary": "auto"}},
        ),
        (
            "gpt-5.6-luna",
            {"temperature": 0.2, "reasoning": Reasoning(effort="low")},
            {"reasoning": Reasoning(effort="low")},
        ),
        (
            "gpt-5.6-luna",
            {"temperature": 0.2, "reasoning": Reasoning()},
            {"reasoning": Reasoning()},
        ),
        (
            "gpt-5",
            {"temperature": 0.2, "reasoning": Reasoning(effort="none")},
            {"reasoning": Reasoning(effort="none")},
        ),
        # gpt-6 rejects every sampling param while reasoning and accepts them all at "none"
        (
            "openai/gpt-6.1-sol",
            {**_GPT6_SAMPLING, "reasoning_effort": "low"},
            {"reasoning_effort": "low"},
        ),
        ("openai/gpt-6.1-sol", {"temperature": 0.2, "top_p": 0.9}, {}),
        (
            "openai/gpt-6.1-sol",
            {"n": 2, "seed": 1, "reasoning_effort": "low"},
            {"n": 2, "seed": 1, "reasoning_effort": "low"},
        ),
        (
            "openai/gpt-6-sol",
            {**_GPT6_SAMPLING, "reasoning_effort": "none"},
            {**_GPT6_SAMPLING, "reasoning_effort": "none"},
        ),
        (
            "openai/gpt-6-luna",
            {"temperature": 0.2, "top_p": 0.9, "reasoning_effort": "low"},
            {"reasoning_effort": "low"},
        ),
        # gpt-6.1-sol has no "none" rung: the gateway coerces it to "low" and the request reasons
        (
            "openai/gpt-6.1-sol",
            {"temperature": 0.2, "reasoning_effort": "none"},
            {"reasoning_effort": "none"},
        ),
        (
            "openai/gpt-6.1-sol",
            {"temperature": 0.2, "reasoning": Reasoning(effort="low")},
            {"reasoning": Reasoning(effort="low")},
        ),
        (
            "openai/gpt-6-sol",
            {"temperature": 0.2, "reasoning": Reasoning(effort="none")},
            {"temperature": 0.2, "reasoning": Reasoning(effort="none")},
        ),
    ],
)
def test_drop_unsupported_params(model: str, params: dict[str, Any], expected: dict) -> None:
    assert drop_unsupported_params(model, params) == expected


@pytest.mark.parametrize(
    ("model", "params", "expected"),
    [
        (
            "openai/gpt-5.6-luna",
            {"temperature": 0.2, "reasoning_effort": "none"},
            {"temperature": 0.2, "reasoning_effort": "none"},
        ),
        # tool-incompatible models lose reasoning_effort first, and then temperature with it
        ("openai/gpt-5.2", {"temperature": 0.2, "reasoning_effort": "low"}, {}),
        ("openai/gpt-5.2", {"temperature": 0.2, "reasoning_effort": "none"}, {}),
    ],
)
def test_drop_unsupported_params_with_tools(
    model: str, params: dict[str, Any], expected: dict
) -> None:
    assert drop_unsupported_params(model, params, tools=[object()]) == expected
