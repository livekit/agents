from __future__ import annotations

from typing import Any

import pytest
from openai.types import Reasoning

from livekit.agents.inference.llm import drop_unsupported_params

pytestmark = pytest.mark.unit


class _ReasoningShim:
    """Mimics openai.types.Reasoning without a hard dependency in assertions."""

    effort: str | None = None


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


def test_gpt_5_original_still_strips_temperature_at_lowest_effort() -> None:
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


def test_gpt_5_2_with_tools_strips_temperature_even_at_effort_none() -> None:
    # the tool strip removes reasoning_effort before the prefix loop runs, so
    # temperature is stripped too — guards the loop-after-tool-strip ordering
    # in drop_unsupported_params. Under the old order the bare temperature
    # would survive and OpenAI would reject the request.
    params = drop_unsupported_params(
        "openai/gpt-5.2",
        {"temperature": 0.2, "reasoning_effort": "none"},
        tools=[object()],
    )
    assert params == {}


def test_gpt_5_4_nano_keeps_temperature_when_reasoning_effort_is_none() -> None:
    # verified against the API 2026-10: gpt-5.4-nano accepts temperature at
    # effort "none" and rejects it at "low", same as 5.4-mini
    params = drop_unsupported_params(
        "openai/gpt-5.4-nano",
        {"temperature": 0.2, "reasoning_effort": "none"},
    )
    assert params == {"temperature": 0.2, "reasoning_effort": "none"}


def test_chat_latest_variants_still_strip_temperature_at_effort_none() -> None:
    # chat-latest variants are not in _MIN_REASONING_EFFORT (exact-key lookup),
    # so they keep the blanket strip even at effort "none".
    params = drop_unsupported_params(
        "openai/gpt-5.1-chat-latest",
        {"temperature": 0.2, "reasoning_effort": "none"},
    )
    assert params == {"reasoning_effort": "none"}


def test_grok_reasoning_model_keeps_sampling_params() -> None:
    # xAI reasoning models support temperature/top_p; the effort-aware
    # carve-out must not touch the xAI branch of _UNSUPPORTED_PARAMS.
    params = drop_unsupported_params(
        "grok-4.20-multi-agent",
        {"temperature": 0.2, "top_p": 0.9, "frequency_penalty": 0.5},
    )
    assert params == {"temperature": 0.2, "top_p": 0.9}


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


def test_gpt_5_6_responses_shape_strips_temperature_at_effort_low() -> None:
    params = drop_unsupported_params(
        "gpt-5.6-luna",
        {"temperature": 0.2, "reasoning": Reasoning(effort="low")},
    )
    assert params == {"reasoning": Reasoning(effort="low")}


def test_gpt_5_6_responses_shape_strips_temperature_when_reasoning_omitted() -> None:
    params = drop_unsupported_params(
        "gpt-5.6-luna",
        {"temperature": 0.2, "reasoning": Reasoning()},
    )
    assert params == {"reasoning": Reasoning()}


def test_gpt_5_responses_shape_strips_temperature_even_at_effort_none() -> None:
    # gpt-5's floor is minimal; the Responses shape must not loosen that
    params = drop_unsupported_params(
        "gpt-5",
        {"temperature": 0.2, "reasoning": Reasoning(effort="none")},
    )
    assert params == {"reasoning": Reasoning(effort="none")}


# gpt-6 rejects every sampling param while it reasons and accepts them all
# again at effort "none" — probed against the API 2026-10-20 on gpt-6.1-sol
# (chat and responses transports), gpt-6-sol and gpt-6-luna.
def test_gpt_6_1_sol_strips_every_sampling_param() -> None:
    params = drop_unsupported_params(
        "openai/gpt-6.1-sol",
        {
            "temperature": 0.2,
            "top_p": 0.9,
            "presence_penalty": 0.5,
            "frequency_penalty": 0.5,
            "logit_bias": {"5": 1},
            "logprobs": True,
            "top_logprobs": 3,
            "stop": ["x"],
            "reasoning_effort": "low",
        },
    )
    assert params == {"reasoning_effort": "low"}


def test_gpt_6_1_sol_strips_temperature_when_reasoning_effort_is_omitted() -> None:
    params = drop_unsupported_params("openai/gpt-6.1-sol", {"temperature": 0.2, "top_p": 0.9})
    assert params == {}


def test_gpt_6_1_sol_keeps_n_and_seed() -> None:
    # both are accepted while reasoning, so neither belongs in the strip set
    params = drop_unsupported_params(
        "openai/gpt-6.1-sol",
        {"n": 2, "seed": 1, "reasoning_effort": "low"},
    )
    assert params == {"n": 2, "seed": 1, "reasoning_effort": "low"}


def test_gpt_6_sol_keeps_every_sampling_param_at_effort_none() -> None:
    params = drop_unsupported_params(
        "openai/gpt-6-sol",
        {
            "temperature": 0.2,
            "presence_penalty": 0.5,
            "logprobs": True,
            "reasoning_effort": "none",
        },
    )
    assert params == {
        "temperature": 0.2,
        "presence_penalty": 0.5,
        "logprobs": True,
        "reasoning_effort": "none",
    }


def test_gpt_6_luna_strips_temperature_while_reasoning() -> None:
    params = drop_unsupported_params(
        "openai/gpt-6-luna",
        {"temperature": 0.2, "top_p": 0.9, "reasoning_effort": "low"},
    )
    assert params == {"reasoning_effort": "low"}


def test_gpt_6_1_sol_strips_even_when_the_client_asks_for_effort_none() -> None:
    # gpt-6.1-sol has no "none" rung, so the gateway coerces the value to the
    # model's floor ("low", its reasoning_effort_fallback) and the request
    # reasons: the strip must not trust the client's spelling of the effort.
    # The coercion is what the gateway's "unexpected reasoning_effort value for
    # model, defaulting to supported value" warn records.
    params = drop_unsupported_params(
        "openai/gpt-6.1-sol",
        {"temperature": 0.2, "reasoning_effort": "none"},
    )
    assert params == {"reasoning_effort": "none"}


def test_gpt_6_1_sol_responses_shape_strips_temperature() -> None:
    params = drop_unsupported_params(
        "openai/gpt-6.1-sol",
        {"temperature": 0.2, "reasoning": Reasoning(effort="low")},
    )
    assert params == {"reasoning": Reasoning(effort="low")}


def test_gpt_6_sol_responses_shape_keeps_temperature_at_effort_none() -> None:
    params = drop_unsupported_params(
        "openai/gpt-6-sol",
        {"temperature": 0.2, "reasoning": Reasoning(effort="none")},
    )
    assert params["temperature"] == 0.2
    assert _responses_effort(params) == "none"
