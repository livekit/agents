from __future__ import annotations

from typing import Any

import pytest
from openai.types import Reasoning

from livekit.agents.inference.llm import drop_unsupported_params

pytestmark = pytest.mark.unit


def test_gpt_5_6_keeps_temperature_when_reasoning_effort_is_none() -> None:
    params = drop_unsupported_params(
        "openai/gpt-5.6-luna",
        {"temperature": 0.2, "top_p": 0.9, "reasoning_effort": "none"},
    )
    assert params == {"temperature": 0.2, "top_p": 0.9, "reasoning_effort": "none"}


def _responses_effort(params: dict[str, Any]) -> str | None:
    # the Responses API plugin sends effort as extra["reasoning"], an openai
    # Reasoning object with an .effort attribute
    reasoning = params.get("reasoning")
    return getattr(reasoning, "effort", None)


def test_gpt_5_6_responses_shape_keeps_temperature_at_effort_none() -> None:
    # Responses plugin path: effort travels as Reasoning(effort="none"), not
    # reasoning_effort. The carve-out must recognize both shapes.
    params = drop_unsupported_params(
        "gpt-5.6-luna",
        {"temperature": 0.2, "top_p": 0.9, "reasoning": Reasoning(effort="none")},
    )
    assert params["temperature"] == 0.2
    assert params["top_p"] == 0.9
    assert _responses_effort(params) == "none"
