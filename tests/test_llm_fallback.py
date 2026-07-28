from __future__ import annotations

import asyncio

import pytest

from livekit.agents.llm import FallbackAdapter

from .fake_llm import FakeLLM

pytestmark = [pytest.mark.unit]


class PrewarmableLLM(FakeLLM):
    """FakeLLM that opts into prewarming by overriding ``_prewarm_impl``."""

    def __init__(self) -> None:
        super().__init__()
        self.prewarmed = asyncio.Event()

    async def _prewarm_impl(self) -> None:
        self.prewarmed.set()


async def test_prewarm_forwarded_to_primary_llm() -> None:
    primary = PrewarmableLLM()
    fallback = PrewarmableLLM()

    fallback_adapter = FallbackAdapter([primary, fallback])
    try:
        fallback_adapter.prewarm()

        await asyncio.wait_for(primary.prewarmed.wait(), timeout=5)
        assert not fallback.prewarmed.is_set(), (
            "expected only the primary LLM to be prewarmed, the fallbacks should stay cold"
        )
    finally:
        await fallback_adapter.aclose()
        await primary.aclose()
        await fallback.aclose()


async def test_prewarm_forwards_event_loop() -> None:
    primary = PrewarmableLLM()

    fallback_adapter = FallbackAdapter([primary])
    try:
        loop = asyncio.get_running_loop()
        fallback_adapter.prewarm(loop=loop)

        assert primary._prewarm_task is not None, "expected the primary LLM to be prewarmed"
        assert primary._prewarm_task.get_loop() is loop, (
            "expected the prewarm task to be scheduled on the provided event loop"
        )
        await asyncio.wait_for(primary.prewarmed.wait(), timeout=5)
    finally:
        await fallback_adapter.aclose()
        await primary.aclose()
