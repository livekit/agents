from __future__ import annotations

import pytest

from livekit.agents.types import NOT_GIVEN
from livekit.plugins import openai
from livekit.plugins.cerebras import LLM as CerebrasLLM

pytestmark = pytest.mark.unit

# `reasoning_format` is a provider-specific option (xAI/Grok, Cerebras) that is not part of the
# standard OpenAI chat-completions schema, so it must be routed through `extra_body` rather than
# sent as a top-level request field. These tests pin that wiring.


async def test_xai_reasoning_format_threaded_to_extra_body() -> None:
    llm = openai.LLM.with_x_ai(api_key="test", reasoning_format="hidden")
    try:
        assert llm._opts.extra_body == {"reasoning_format": "hidden"}
    finally:
        await llm.aclose()


async def test_xai_reasoning_format_omitted_by_default() -> None:
    llm = openai.LLM.with_x_ai(api_key="test")
    try:
        assert llm._opts.extra_body is NOT_GIVEN
    finally:
        await llm.aclose()


async def test_cerebras_reasoning_format_threaded_to_extra_body() -> None:
    llm = CerebrasLLM(api_key="test", reasoning_format="hidden")
    try:
        assert llm._opts.extra_body == {"reasoning_format": "hidden"}
    finally:
        await llm.aclose()


async def test_cerebras_reasoning_format_omitted_by_default() -> None:
    llm = CerebrasLLM(api_key="test")
    try:
        assert llm._opts.extra_body is NOT_GIVEN
    finally:
        await llm.aclose()
