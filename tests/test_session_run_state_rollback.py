from __future__ import annotations

import asyncio

import pytest

from livekit.agents import Agent, AgentSession, RunResult

from .fake_llm import FakeLLM, FakeLLMResponse

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]


class SilentAgent(Agent):
    """Produces no speech at all from on_enter."""

    def __init__(self) -> None:
        super().__init__(instructions="silent agent")


def _llm() -> FakeLLM:
    return FakeLLM(
        fake_responses=[
            FakeLLMResponse(input="hello", content="hi there", ttft=0.1, duration=0.1),
        ]
    )


async def test_run_before_start_then_run_after_start() -> None:
    """A run() that fails before start() must not poison the next run()."""
    async with AgentSession(llm=_llm()) as sess:
        with pytest.raises(RuntimeError, match="isn't running"):
            sess.run(user_input="hello")

        # generate_reply() raised synchronously, so no run state may be left behind.
        assert sess._global_run_state is None

        await asyncio.wait_for(sess.start(SilentAgent()), timeout=5.0)

        result: RunResult[None] | None = None
        try:
            result = sess.run(user_input="hello")
        except RuntimeError as exc:
            pytest.fail(f"run() after start() raised: {exc}")

        assert result is not None
        await asyncio.wait_for(result, timeout=5.0)
        assert result.done()


async def test_start_capture_run_after_failed_pre_start_run() -> None:
    """start(capture_run=True) must not see a dangling run from a failed pre-start run()."""
    async with AgentSession(llm=_llm()) as sess:
        with pytest.raises(RuntimeError, match="isn't running"):
            sess.run(user_input="hello")

        assert sess._global_run_state is None

        result: RunResult[None] | None = None
        try:
            result = await asyncio.wait_for(
                sess.start(SilentAgent(), capture_run=True), timeout=5.0
            )
        except RuntimeError as exc:
            pytest.fail(f"start(capture_run=True) after a failed run() raised: {exc}")

        assert result is not None
        result.expect.next_event().is_agent_handoff(new_agent_type=SilentAgent)
        result.expect.no_more_events()


async def test_run_during_session_close_resets_the_run_slot() -> None:
    """A run() rejected while the session is closing must not leave a stale run state."""
    async with AgentSession(llm=_llm()) as sess:
        await asyncio.wait_for(sess.start(SilentAgent()), timeout=5.0)

        run_errors: list[RuntimeError] = []

        def on_close(_: object) -> None:
            try:
                sess.run(user_input="hello")
            except RuntimeError as exc:
                run_errors.append(exc)

        sess.on("close", on_close)
        await asyncio.wait_for(sess.aclose(), timeout=5.0)

        assert len(run_errors) == 1
        assert "isn't running" in str(run_errors[0]) or "closing" in str(run_errors[0])
        assert sess._global_run_state is None
