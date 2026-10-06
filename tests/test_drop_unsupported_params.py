from __future__ import annotations

import pytest

from livekit.agents.inference.llm import drop_unsupported_params

pytestmark = pytest.mark.unit


def test_gpt_5_6_keeps_temperature_when_reasoning_effort_is_none() -> None:
    params = drop_unsupported_params(
        "openai/gpt-5.6-luna",
        {"temperature": 0.2, "top_p": 0.9, "reasoning_effort": "none"},
    )
    assert params == {"temperature": 0.2, "top_p": 0.9, "reasoning_effort": "none"}


def test_gpt_5_1_keeps_temperature_when_reasoning_effort_is_none() -> None:
    params = drop_unsupported_params(
        "gpt-5.1",
        {"temperature": 0.2, "reasoning_effort": "none"},
    )
    assert params == {"temperature": 0.2, "reasoning_effort": "none"}


def test_gpt_5_6_strips_temperature_when_reasoning_effort_is_low() -> None:
    params = drop_unsupported_params(
        "openai/gpt-5.6-luna",
        {"temperature": 0.2, "top_p": 0.9, "reasoning_effort": "low"},
    )
    assert params == {"reasoning_effort": "low"}


def test_gpt_5_6_strips_temperature_when_reasoning_effort_is_omitted() -> None:
    params = drop_unsupported_params(
        "openai/gpt-5.6-luna",
        {"temperature": 0.2, "top_p": 0.9},
    )
    assert params == {}


def test_gpt_5_6_strips_other_reasoning_params_even_at_effort_none() -> None:
    # temperature/top_p are accepted at effort "none"; the other sampling
    # params are not supported on gpt-5.x regardless of effort.
    params = drop_unsupported_params(
        "openai/gpt-5.6-luna",
        {
            "temperature": 0.2,
            "frequency_penalty": 0.5,
            "logit_bias": {"5": 1},
            "reasoning_effort": "none",
        },
    )
    assert params == {"temperature": 0.2, "reasoning_effort": "none"}


def test_gpt_5_original_still_strips_temperature_at_effort_none() -> None:
    # gpt-5/mini/nano's lowest effort is "minimal", not "none" — the blanket
    # strip must stay for them.
    params = drop_unsupported_params(
        "openai/gpt-5",
        {"temperature": 0.2, "top_p": 0.9, "reasoning_effort": "minimal"},
    )
    assert params == {"reasoning_effort": "minimal"}


def test_gpt_5_mini_strips_temperature_even_without_reasoning_effort() -> None:
    params = drop_unsupported_params(
        "openai/gpt-5-mini",
        {"temperature": 0.2},
    )
    assert params == {}


def test_o_series_strips_temperature_unchanged() -> None:
    params = drop_unsupported_params(
        "openai/o3",
        {"temperature": 0.2, "top_p": 0.9, "reasoning_effort": "none"},
    )
    assert params == {"reasoning_effort": "none"}


def test_non_reasoning_model_keeps_temperature() -> None:
    params = drop_unsupported_params(
        "openai/gpt-4o",
        {"temperature": 0.2, "top_p": 0.9},
    )
    assert params == {"temperature": 0.2, "top_p": 0.9}


def test_gpt_5_6_with_tools_keeps_temperature_at_effort_none() -> None:
    params = drop_unsupported_params(
        "openai/gpt-5.6-luna",
        {"temperature": 0.2, "reasoning_effort": "none"},
        tools=[object()],
    )
    assert params == {"temperature": 0.2, "reasoning_effort": "none"}


def test_gpt_5_2_with_tools_strips_reasoning_effort() -> None:
    # regression guard for the tool-incompatible strip, which shares the
    # gpt-5 prefix path
    params = drop_unsupported_params(
        "openai/gpt-5.2",
        {"temperature": 0.2, "reasoning_effort": "low"},
        tools=[object()],
    )
    assert params == {}
