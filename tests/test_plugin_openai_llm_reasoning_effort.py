from __future__ import annotations

import pytest

from livekit.plugins.openai import LLM
from livekit.plugins.openai.responses import LLM as ResponsesLLM

pytestmark = pytest.mark.plugin("openai")


# Models whose API rejects ``reasoning_effort="minimal"`` and only accepts the
# new ``none/low/medium/high/xhigh`` set, so the plugin must default to
# ``"none"`` when the user didn't pass an explicit value.
_NO_MINIMAL_MODELS = ["gpt-5.1", "gpt-5.2", "gpt-5.4", "gpt-5.4-mini"]


@pytest.mark.parametrize("model", _NO_MINIMAL_MODELS)
def test_chat_completions_llm_defaults_reasoning_effort_to_none(model: str) -> None:
    """Regression for #6147: gpt-5.4-mini and other gpt-5.1+ variants reject
    ``reasoning_effort="minimal"``."""
    llm = LLM(model=model, api_key="sk-test")
    assert llm._opts.reasoning_effort == "none"


@pytest.mark.parametrize("model", _NO_MINIMAL_MODELS)
def test_responses_llm_defaults_reasoning_effort_to_none(model: str) -> None:
    """Same gap on the Responses-API wrapper (mirrors `LLM.__init__`)."""
    llm = ResponsesLLM(model=model, api_key="sk-test")
    assert llm._opts.reasoning is not None
    assert llm._opts.reasoning.effort == "none"


def test_chat_completions_llm_keeps_minimal_default_for_gpt_5() -> None:
    """The pre-existing default for plain gpt-5 (which still accepts
    ``"minimal"``) must keep working."""
    llm = LLM(model="gpt-5", api_key="sk-test")
    assert llm._opts.reasoning_effort == "minimal"
