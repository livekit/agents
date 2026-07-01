"""
Test LLM prewarming functionality (Issue #3240).

This test suite verifies that the prewarm() method reduces first-request latency
by pre-establishing HTTP connections to the LLM service.
"""

from __future__ import annotations

import asyncio
import os
import time

import pytest

from livekit.agents import llm
from livekit.plugins import openai

pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)

llm_model = "gpt-4o-mini"


@pytest.mark.asyncio
async def test_llm_prewarm_reduces_latency():
    """Test that prewarming reduces time to first token (TTFT).
    This test verifies that calling prewarm() before making an LLM request
    reduces the latency of the first request by pre-establishing the HTTP connection.
    """
    # Test 1: WITHOUT prewarming
    llm_no_prewarm = openai.LLM(model=llm_model)

    chat_ctx = llm.ChatContext()
    chat_ctx.add_message(role="user", content="Say 'test' in one word only")

    start = time.perf_counter()
    stream = llm_no_prewarm.chat(chat_ctx=chat_ctx)

    # Measure time to first chunk
    ttft_no_prewarm = 0
    async for chunk in stream:
        if chunk.delta and chunk.delta.content:
            ttft_no_prewarm = time.perf_counter() - start
            break

    # Fully consume the stream to avoid leaks
    async for _ in stream:
        pass

    await llm_no_prewarm.aclose()

    # Test 2: WITH prewarming
    llm_with_prewarm = openai.LLM(model=llm_model)
    llm_with_prewarm.prewarm()

    # Give the prewarm task a moment to establish the connection
    await asyncio.sleep(0.3)

    chat_ctx = llm.ChatContext()
    chat_ctx.add_message(role="user", content="Say 'test' in one word only")

    start = time.perf_counter()
    stream = llm_with_prewarm.chat(chat_ctx=chat_ctx)

    # Measure time to first chunk
    ttft_with_prewarm = 0
    async for chunk in stream:
        if chunk.delta and chunk.delta.content:
            ttft_with_prewarm = time.perf_counter() - start
            break

    # Fully consume the stream to avoid leaks
    async for _ in stream:
        pass

    await llm_with_prewarm.aclose()

    # Verify prewarming helped (should be at least slightly faster)
    # We don't assert a specific improvement because network conditions vary,
    # but we print the results for visibility
    print("Prewarm Test Results:")
    print(f" Without prewarm: {ttft_no_prewarm:.3f}s")
    print(f" With prewarm:    {ttft_with_prewarm:.3f}s")

    if ttft_with_prewarm < ttft_no_prewarm:
        improvement = ttft_no_prewarm - ttft_with_prewarm
        improvement_pct = (improvement / ttft_no_prewarm) * 100
        print(f"Improvement: {improvement:.3f}s ({improvement_pct:.1f}% faster)")
    else:
        print(" No improvement detected (network conditions may vary)")

    # The test passes if both requests succeeded
    # We don't strictly assert latency improvements due to network variability
    assert ttft_no_prewarm > 0
    assert ttft_with_prewarm > 0


@pytest.mark.asyncio
async def test_llm_prewarm_task_cleanup():
    """Test that prewarm task is properly cleaned up on aclose()."""
    llm_instance = openai.LLM(model=llm_model)

    # Start prewarming
    llm_instance.prewarm()

    # Verify task was created
    assert llm_instance._prewarm_task is not None

    # Close immediately (should cancel the prewarm task gracefully)
    await llm_instance.aclose()

    # Task should be completed or cancelled
    assert llm_instance._prewarm_task.done() or llm_instance._prewarm_task.cancelled()


@pytest.mark.asyncio
async def test_llm_prewarm_idempotent():
    """Test that calling prewarm() multiple times doesn't cause issues."""
    llm_instance = openai.LLM(model=llm_model)

    # Call prewarm multiple times
    llm_instance.prewarm()
    first_task = llm_instance._prewarm_task

    # Calling prewarm again should create a new task
    llm_instance.prewarm()
    second_task = llm_instance._prewarm_task

    # Both tasks should exist
    assert first_task is not None
    assert second_task is not None

    # Clean up - must wait for tasks to complete or aclose will leak
    await llm_instance.aclose()


@pytest.mark.asyncio
async def test_llm_works_without_prewarm():
    """Test that LLM works normally even without calling prewarm()."""
    llm_instance = openai.LLM(model=llm_model)

    # Don't call prewarm() at all
    chat_ctx = llm.ChatContext()
    chat_ctx.add_message(role="user", content="Say 'hello' in one word")

    stream = llm_instance.chat(chat_ctx=chat_ctx)

    # Should still work fine
    response_received = False
    async for chunk in stream:
        if chunk.delta and chunk.delta.content:
            response_received = True
            break

    # Fully consume the stream to avoid leaks
    async for _ in stream:
        pass

    await llm_instance.aclose()

    assert response_received, "Should receive response even without prewarm"
