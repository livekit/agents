"""Hermetic tests for the LangGraph adapter's preemptive-generation opt-out (#5924).

A graph turn can execute tools and mutate state (checkpoints, external
systems) with no rollback when a speculative run is discarded, so the adapter
declares ``chat()`` unsafe to run preemptively by default.
"""

from unittest.mock import MagicMock

import pytest

from livekit.agents import llm
from livekit.plugins.langchain import LLMAdapter

pytestmark = pytest.mark.unit


class TestPreemptiveGenerationSafe:
    def test_base_llm_defaults_to_safe(self) -> None:
        # plain LLM chat() calls are side-effect-free; speculation stays enabled
        class _PlainLLM(llm.LLM):
            def chat(self, **kwargs):  # type: ignore[override]
                raise NotImplementedError

        assert _PlainLLM().preemptive_generation_safe is True

    def test_langgraph_adapter_defaults_to_unsafe(self) -> None:
        adapter = LLMAdapter(MagicMock())
        assert adapter.preemptive_generation_safe is False

    def test_langgraph_adapter_can_opt_in(self) -> None:
        adapter = LLMAdapter(MagicMock(), preemptive_generation_safe=True)
        assert adapter.preemptive_generation_safe is True
