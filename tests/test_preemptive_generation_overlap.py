"""
Reproduces the preemptive-generation overlap: when STT preflight emits a
transcript, then the user keeps talking and the final transcript differs, the
preemptive LLM stream is cancelled via `_cancel()` but its abort does not
propagate before a SECOND, fully-fresh LLM stream is kicked off for the real
transcript. The two streams overlap in flight.

Code path:
- AgentActivity.on_preemptive_generation calls `_generate_reply` with
  schedule_speech=False, which runs `_pipeline_reply_task_impl` and starts
  LLM stream A via `perform_llm_inference`.
- AgentActivity.on_end_of_turn schedules `_user_turn_completed_task`, which
  invalidates the preemptive (`preemptive.speech_handle._cancel()` resolves
  interrupt_fut but task A is still mid-run) and then synchronously calls
  `_generate_reply` for the real transcript — starting LLM stream B before
  A's abort propagates through the speech task's interrupted check.

The race window is everything between the `_cancel()` and the next async
yield in A's speech task. With `preemptive_tts: True` the cost doubles
because TTS is also kicked off preemptively.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from livekit.agents import Agent, AgentSession
from livekit.agents.llm import ChatContext, LLMStream, Tool, ToolChoice
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.voice.audio_recognition import (
    _EndOfTurnInfo,
    _PreemptiveGenerationInfo,
)

from .fake_llm import FakeLLM, FakeLLMResponse

# Slow enough that a second chat() invocation will land while the first is
# still mid-stream (waiting on its ttft delay).
STREAM_TTFT_S = 0.08


class OverlapTrackingFakeLLM(FakeLLM):
    def __init__(self, *, fake_responses: list[FakeLLMResponse] | None = None) -> None:
        super().__init__(fake_responses=fake_responses)
        self.call_times: list[float] = []
        self.chat_calls: list[dict[str, Any]] = []

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> LLMStream:
        self.call_times.append(time.monotonic())
        last_item = chat_ctx.items[-1] if chat_ctx.items else None
        if last_item is not None and last_item.type == "message" and last_item.role == "user":
            input_text = last_item.text_content or ""
        else:
            input_text = f"(non-user-input: {last_item.type if last_item else None})"
        self.chat_calls.append({"chat_ctx": chat_ctx, "input_text": input_text})
        return super().chat(
            chat_ctx=chat_ctx,
            tools=tools,
            conn_options=conn_options,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
            extra_kwargs=extra_kwargs,
        )


def _build_llm() -> OverlapTrackingFakeLLM:
    return OverlapTrackingFakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="transcript A",
                content="answer A",
                ttft=STREAM_TTFT_S,
                duration=STREAM_TTFT_S,
            ),
            FakeLLMResponse(
                input="transcript B (different)",
                content="answer B",
                ttft=STREAM_TTFT_S,
                duration=STREAM_TTFT_S,
            ),
        ]
    )


@pytest.mark.asyncio
async def test_preemptive_llm_stream_overlaps_real_reply() -> None:
    """Preemptive LLM stream is still in flight when the real generate_reply
    starts a new one.
    """
    llm_model = _build_llm()
    ready = asyncio.Event()

    class TestAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="test")

        async def on_enter(self) -> None:
            ready.set()

    async with AgentSession(llm=llm_model) as session:
        await session.start(TestAgent())
        await ready.wait()

        # The activity is private — reach in to drive the RecognitionHooks
        # directly. This is the same surface STT preflight + EOU would call.
        activity = session._activity
        assert activity is not None

        # 1) STT preflight: agent kicks off a preemptive LLM call with
        #    transcript A.
        activity.on_preemptive_generation(
            _PreemptiveGenerationInfo(
                new_transcript="transcript A",
                transcript_confidence=0.9,
                started_speaking_at=time.time(),
            )
        )

        # Yield long enough for the speech task to start, create the LLM
        # inference task, and have the inference task invoke chat(). We do
        # NOT wait for the ttft delay — we want stream A to still be
        # mid-flight when stream B starts.
        await asyncio.sleep(STREAM_TTFT_S / 8)

        # Sanity: stream A is in flight now.
        assert len(llm_model.call_times) == 1, (
            f"expected exactly 1 preemptive chat() call, saw {len(llm_model.call_times)}"
        )
        a_started_at = llm_model.call_times[0]

        # 2) User keeps talking; final transcript differs. on_end_of_turn
        #    fires _user_turn_completed_task which: cancels preemptive
        #    (resolves interrupt_fut on handle A), then synchronously calls
        #    _generate_reply(B).
        activity.on_end_of_turn(
            _EndOfTurnInfo(
                skip_reply=False,
                new_transcript="transcript B (different)",
                transcript_confidence=0.95,
                started_speaking_at=time.time() - 1.0,
                stopped_speaking_at=time.time() - 0.05,
                transcription_delay=0.05,
                end_of_turn_delay=0.05,
            )
        )

        # Wait long enough for stream B to be invoked but NOT for stream A's
        # ttft to expire. If chat() for B is invoked here, both streams are
        # overlapping (stream A still in delay).
        await asyncio.sleep(STREAM_TTFT_S / 4)

        # ---- Smoking gun: two LLM streams overlap ------------------------

        # The preemptive LLM call (A) happened.
        assert llm_model.chat_calls[0]["input_text"] == "transcript A"

        # The user-turn-completed LLM call (B) happened — and within the
        # ttft window of A, meaning A had not yet emitted its first token
        # (let alone been aborted).
        assert len(llm_model.call_times) >= 2, (
            f"Expected at least 2 LLM stream invocations (preemptive + final). "
            f"Saw {len(llm_model.call_times)}: "
            f"{[c['input_text'] for c in llm_model.chat_calls]}"
        )

        b_started_at = llm_model.call_times[1]
        overlap_window = b_started_at - a_started_at
        assert overlap_window < STREAM_TTFT_S, (
            f"Expected B to start while A's ttft delay ({STREAM_TTFT_S}s) was "
            f"still in progress. Saw {overlap_window}s between A and B."
        )

        # The B call was for the real (different) transcript.
        assert llm_model.chat_calls[1]["input_text"] == "transcript B (different)"

        # Drain so the session close in __aexit__ doesn't race with in-flight
        # streams.
        await asyncio.sleep(STREAM_TTFT_S * 2)


@pytest.mark.asyncio
async def test_preemptive_invalidation_cancels_speech_tasks() -> None:
    """When the preemptive is invalidated by a divergent transcript,
    `_user_turn_completed_task` cancels the preemptive's speech tasks
    directly (in addition to `_cancel()` on the handle).

    Without the direct task.cancel(), `_cancel()` only resolves the handle's
    interrupt future and the preemptive's pipeline_reply task wouldn't run
    its cleanup until several event-loop ticks later — by which time the
    synchronous `_generate_reply(...)` for the corrected transcript would
    have started a fresh LLM stream.
    """
    llm_model = _build_llm()
    ready = asyncio.Event()

    class TestAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="test")

        async def on_enter(self) -> None:
            ready.set()

    async with AgentSession(llm=llm_model) as session:
        await session.start(TestAgent())
        await ready.wait()

        activity = session._activity
        assert activity is not None

        activity.on_preemptive_generation(
            _PreemptiveGenerationInfo(
                new_transcript="transcript A",
                transcript_confidence=0.9,
                started_speaking_at=time.time(),
            )
        )

        # Let the speech task start, register its inner tasks on the handle,
        # and invoke chat().
        await asyncio.sleep(STREAM_TTFT_S / 8)
        assert len(llm_model.call_times) == 1

        preemptive = activity._preemptive_generation
        assert preemptive is not None
        preemptive_tasks = list(preemptive.speech_handle._tasks)
        assert preemptive_tasks, "expected the preemptive handle to track at least one task"
        assert all(not t.done() for t in preemptive_tasks)

        # Fire EOU with a divergent transcript — triggers the invalidation
        # branch which both _cancel()s the handle AND task.cancel()s each
        # speech task directly.
        activity.on_end_of_turn(
            _EndOfTurnInfo(
                skip_reply=False,
                new_transcript="transcript B (different)",
                transcript_confidence=0.95,
                started_speaking_at=time.time() - 1.0,
                stopped_speaking_at=time.time() - 0.05,
                transcription_delay=0.05,
                end_of_turn_delay=0.05,
            )
        )

        # Yield enough for _user_turn_completed_task to reach the
        # invalidation branch.
        await asyncio.sleep(STREAM_TTFT_S / 4)

        # Every speech task originally linked to the preemptive must be in a
        # cancelled / cancel-pending state.
        for t in preemptive_tasks:
            assert t.cancelling() > 0 or t.done(), (
                f"expected preemptive speech task {t.get_name()} to be cancelled, "
                f"saw cancelling={t.cancelling()} done={t.done()}"
            )

        # Drain.
        await asyncio.sleep(STREAM_TTFT_S * 2)
