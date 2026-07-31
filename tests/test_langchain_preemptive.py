"""Hermetic tests for the LangGraph adapter's preemptive-generation opt-out (#5924).

A graph turn can execute tools and mutate state (checkpoints, external
systems) with no rollback when a speculative run is discarded, so the adapter
declares ``chat()`` stateful by default.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from livekit.agents import APIConnectionError, llm
from livekit.agents.llm import FallbackAdapter
from livekit.agents.utils import aio
from livekit.plugins.langchain import LLMAdapter

pytestmark = pytest.mark.unit


class _CountingLLM(llm.LLM):
    """Fails every chat(), counting the calls; optionally declares itself stateful."""

    def __init__(self, *, stateful: bool) -> None:
        super().__init__()
        self._stateful = stateful
        self.calls = 0

    @property
    def stateful(self) -> bool:
        return self._stateful

    def chat(self, **kwargs: Any) -> llm.LLMStream:  # type: ignore[override]
        self.calls += 1
        return _FailingStream(
            self,
            chat_ctx=kwargs["chat_ctx"],
            tools=kwargs.get("tools") or [],
            # honour the adapter's options so the fake fails fast
            conn_options=kwargs["conn_options"],
        )


class _FailingStream(llm.LLMStream):
    async def _run(self) -> None:
        raise APIConnectionError("unavailable")


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
        class _PlainLLM(llm.LLM):
            def chat(self, **kwargs):  # type: ignore[override]
                raise NotImplementedError

        stateless = _PlainLLM()
        stateful = LLMAdapter(MagicMock())

        assert FallbackAdapter([stateless, stateful]).stateful is True
        assert FallbackAdapter([stateless, _PlainLLM()]).stateful is False


class TestFallbackRecoveryProbes:
    """A recovery probe drains and discards a full chat(), so it must not run
    against a stateful instance - that would commit a real turn's side effects
    just to test availability."""

    async def _drain(self, adapter: FallbackAdapter) -> None:
        with pytest.raises(APIConnectionError):
            async with adapter.chat(chat_ctx=llm.ChatContext.empty()) as stream:
                async for _ in stream:
                    pass

    @staticmethod
    async def _settle(adapter: FallbackAdapter) -> None:
        """Await the background recovery probes the adapter may have started."""
        for status in adapter._status:
            if status.recovering_task is not None:
                await aio.cancel_and_wait(status.recovering_task)

    async def test_no_recovery_probe_for_a_stateful_instance(self) -> None:
        stateful = _CountingLLM(stateful=True)
        adapter = FallbackAdapter([stateful])
        try:
            await self._drain(adapter)

            # the probe would be a second, discarded chat() on top of the request
            assert stateful.calls == 1
            assert adapter._status[0].recovering_task is None
        finally:
            await self._settle(adapter)

    async def test_stateless_instance_is_still_probed(self) -> None:
        stateless = _CountingLLM(stateful=False)
        adapter = FallbackAdapter([stateless])
        try:
            await self._drain(adapter)

            assert adapter._status[0].recovering_task is not None
        finally:
            await self._settle(adapter)

    async def test_stateful_instance_is_retried_on_the_next_request(self) -> None:
        # with no probe to bring it back, a failed stateful instance must stay
        # reachable through real traffic instead of being dropped for good
        stateful = _CountingLLM(stateful=True)
        stateless = _CountingLLM(stateful=False)
        adapter = FallbackAdapter([stateful, stateless])
        try:
            await self._drain(adapter)
            assert adapter._status[0].available is False
            first_round = stateful.calls

            await self._drain(adapter)
            assert stateful.calls == first_round + 1
        finally:
            await self._settle(adapter)
