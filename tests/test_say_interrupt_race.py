"""
Reproduces a race that drops the LLM's post-tool reply when a tool follows
the pattern:

    ctx.session.say('one moment...')   # schedules a SECOND SpeechHandle (B)
    await some_slow_api()              # tool blocks
    ctx.speech_handle.interrupt()      # interrupts the OWNING handle (A)
    return ...

Why it races: AgentActivity._pipeline_reply_task_impl calls
`speech_handle._mark_generation_done()` BEFORE `await exe_task` (the tool
execution). That hand-off lets the speech queue pump while the tool is still
running, so the parallel say() speech (handle B) is allowed to start. By the
time the tool returns and the pipeline reaches the "schedule tool response"
branch, the tool has already called interrupt() on handle A, and the tool-
response speech is scheduled on that same (now-interrupted) handle. The
pipeline short-circuits on the interrupted check and the response is silently
dropped.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from livekit.agents import Agent, AgentSession, RunContext, function_tool
from livekit.agents.llm import (
    ChatContext,
    FunctionToolCall,
    LLMStream,
    Tool,
    ToolChoice,
)
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.voice.events import SpeechCreatedEvent
from livekit.agents.voice.run_result import RunResult
from livekit.agents.voice.speech_handle import SpeechHandle

from .fake_llm import FakeLLM, FakeLLMResponse

INITIAL_INPUT = "please run the slow tool"
TOOL_REPLY_TEXT = "tool reply: i finished the slow op"


class CountingFakeLLM(FakeLLM):
    def __init__(self, *, fake_responses: list[FakeLLMResponse] | None = None) -> None:
        super().__init__(fake_responses=fake_responses)
        self.calls: list[ChatContext] = []

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
        self.calls.append(chat_ctx)
        return super().chat(
            chat_ctx=chat_ctx,
            tools=tools,
            conn_options=conn_options,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
            extra_kwargs=extra_kwargs,
        )


def _build_llm() -> CountingFakeLLM:
    return CountingFakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input=INITIAL_INPUT,
                content="",
                ttft=0,
                duration=0,
                tool_calls=[
                    FunctionToolCall(name="slow_tool", arguments="{}", call_id="call_1"),
                ],
            ),
            # Second LLM call: the tool returns 'ok', which becomes the
            # function_call_output text.
            FakeLLMResponse(
                input="ok",
                content=TOOL_REPLY_TEXT,
                ttft=0,
                duration=0,
            ),
        ]
    )


@pytest.mark.asyncio
async def test_control_tool_without_say_and_interrupt_gets_tool_reply() -> None:
    """Control: a tool that does NOT call say()/interrupt() gets a tool reply."""
    llm_model = _build_llm()

    @function_tool
    async def slow_tool(ctx: RunContext) -> str:
        """A tool that mimics the production pattern: say() then slow API then interrupt."""
        return "ok"

    agent = Agent(instructions="Test agent that always calls slow_tool.", tools=[slow_tool])

    async with AgentSession(llm=llm_model) as session:
        await session.start(agent)

        result: RunResult[None] = await asyncio.wait_for(
            session.run(user_input=INITIAL_INPUT), timeout=30.0
        )

        # The LLM is called twice: once for the initial reply (which calls the
        # tool), once for the tool reply (after the tool returns).
        assert len(llm_model.calls) == 2

        # Both the function call and the post-tool assistant message land in events.
        assistant_messages = [
            ev.item.text_content or "" for ev in result.events if ev.type == "message"
        ]
        assert TOOL_REPLY_TEXT in assistant_messages


@pytest.mark.asyncio
async def test_say_and_interrupt_inside_tool_still_emits_tool_reply() -> None:
    """say() + interrupt() inside the tool still emits the LLM tool reply.

    Before the fix in agent_activity.py: when a tool called
    `ctx.session.say(...)` then `ctx.speech_handle.interrupt()`, the tool's
    follow-up LLM reply was scheduled on the (now-interrupted) owning handle
    and silently dropped by the interrupted-check at the top of
    `_pipeline_reply_task_impl`.

    Fix: the interrupt flag is scoped to a single playout step. Before
    scheduling the tool-reply step, the framework calls
    `SpeechHandle._reset_interrupt_for_next_step()` to re-arm the interrupt
    future so the same handle can run the new step. Tools that want to
    suppress the LLM reply intentionally should raise `StopResponse`.
    """
    llm_model = _build_llm()
    say_handle_fut: asyncio.Future[SpeechHandle] = asyncio.Future()
    interrupted_at_fut: asyncio.Future[dict[str, Any]] = asyncio.Future()

    @function_tool
    async def slow_tool(ctx: RunContext) -> str:
        """A tool that mimics the production pattern: say() then slow API then interrupt."""
        # (1) Schedule a parallel speech handle. Production tools do this as an
        #     explicit "buy time" prompt before the slow API call.
        say_handle = ctx.session.say("one moment, looking that up")
        say_handle_fut.set_result(say_handle)

        # (2) Simulate a slow API. While we wait, the main speech loop pumps
        #     handle B because the framework already marked handle A's
        #     generation done before awaiting tools.
        await asyncio.sleep(0.05)

        # Capture state right at the interrupt point — handle A is still alive
        # and handle B is mid-flight.
        interrupted_at_fut.set_result(
            {
                "handle_a_id": ctx.speech_handle.id,
                "handle_b_id": say_handle.id,
                "handle_a_interrupted": ctx.speech_handle.interrupted,
                "handle_b_exists": True,
            }
        )

        # (3) Interrupt the owning handle. After this, when the pipeline
        #     re-schedules the tool reply on this SAME handle, the main task
        #     skips it because the handle is interrupted.
        ctx.speech_handle.interrupt()

        return "ok"

    agent = Agent(instructions="Test agent that always calls slow_tool.", tools=[slow_tool])

    speech_observations: list[dict[str, Any]] = []

    async with AgentSession(llm=llm_model) as session:

        def _on_speech_created(ev: SpeechCreatedEvent) -> None:
            handle = ev.speech_handle
            speech_observations.append(
                {
                    "id": handle.id,
                    "source": ev.source,
                    "interrupted": handle.interrupted,
                }
            )

        session.on("speech_created", _on_speech_created)

        await session.start(agent)

        result: RunResult[None] = await asyncio.wait_for(
            session.run(user_input=INITIAL_INPUT), timeout=30.0
        )

        # ---- Race-condition assertions ---------------------------------------

        # 1. Two distinct speech handles existed concurrently: the LLM-reply
        #    handle (source 'generate_reply') and the say() handle (source 'say').
        say_event = next((s for s in speech_observations if s["source"] == "say"), None)
        reply_event = next(
            (s for s in speech_observations if s["source"] == "generate_reply"), None
        )
        assert reply_event is not None, "expected an LLM reply handle"
        assert say_event is not None, "expected a say() handle scheduled by the tool"
        assert say_event["id"] != reply_event["id"]

        # 2. At the moment we interrupted handle A, handle B was also alive.
        snapshot = await interrupted_at_fut
        assert snapshot["handle_a_interrupted"] is False
        assert snapshot["handle_b_exists"] is True
        assert snapshot["handle_a_id"] != snapshot["handle_b_id"]

        # 3. The say()-scheduled handle is handle B (matches the captured id).
        say_handle = await say_handle_fut
        assert say_handle.id == snapshot["handle_b_id"]

        # 4. The LLM is called twice (initial reply + tool reply) and the
        #    tool-reply text reaches the user. Before the fix, the second
        #    pipeline_reply ran on the still-interrupted handle A and was
        #    dropped by the interrupted-check inside
        #    `_pipeline_reply_task_impl`.
        assert len(llm_model.calls) == 2, (
            f"LLM should be called twice (initial reply + tool reply): {len(llm_model.calls)}"
        )

        assistant_messages = [
            ev.item.text_content or "" for ev in result.events if ev.type == "message"
        ]
        assert TOOL_REPLY_TEXT in assistant_messages, (
            f"Expected the post-tool LLM reply to land in the chat context, but it "
            f"was dropped. Got: {assistant_messages}. The race condition may have "
            f"regressed — confirm that the tool-response scheduling site calls "
            f"speech_handle._reset_interrupt_for_next_step()."
        )

        # 5. The fix reuses the SAME owning handle for the tool reply — the
        #    interrupt flag is scoped to one playout step, not the whole turn.
        reply_handles = [s for s in speech_observations if s["source"] == "generate_reply"]
        assert len(reply_handles) == 1, (
            f"Expected one generate_reply handle reused across steps; got "
            f"{[s['id'] for s in reply_handles]}"
        )
