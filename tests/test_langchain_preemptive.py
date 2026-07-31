"""Hermetic tests for the LangGraph adapter's preemptive-generation opt-out (#5924).

A graph turn can execute tools and mutate state (checkpoints, external
systems) with no rollback when a speculative run is discarded, so the adapter
declares ``chat()`` stateful by default.
"""

from unittest.mock import MagicMock

import pytest

from livekit.agents import llm
from livekit.plugins.langchain import LLMAdapter

pytestmark = pytest.mark.unit


class TestStateful:
    def test_base_llm_defaults_to_stateless(self) -> None:
        # plain LLM chat() calls are side-effect-free; speculation stays enabled
        class _PlainLLM(llm.LLM):
            def chat(self, **kwargs):  # type: ignore[override]
                raise NotImplementedError

        assert _PlainLLM().stateful is False

    def test_langgraph_adapter_defaults_to_stateful(self) -> None:
        adapter = LLMAdapter(MagicMock())
        assert adapter.stateful is True

    def test_langgraph_adapter_can_opt_out(self) -> None:
        adapter = LLMAdapter(MagicMock(), stateful=False)
        assert adapter.stateful is False

    def test_fallback_adapter_propagates_declaration(self) -> None:
        # a chat() may run on any wrapped instance, so the wrapper is stateful
        # as soon as one of them is
        from livekit.agents.llm import FallbackAdapter

        class _PlainLLM(llm.LLM):
            def chat(self, **kwargs):  # type: ignore[override]
                raise NotImplementedError

        stateless = _PlainLLM()
        stateful = LLMAdapter(MagicMock())

        assert FallbackAdapter([stateless, stateful]).stateful is True
        assert FallbackAdapter([stateless, _PlainLLM()]).stateful is False
