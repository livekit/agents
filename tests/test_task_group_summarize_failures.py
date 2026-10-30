from __future__ import annotations

import asyncio

import pytest

from livekit.agents import Agent, AgentSession, AgentTask, llm
from livekit.agents.beta.workflows.task_group import TaskGroup

from .fake_llm import FakeLLM
from .fake_realtime import FakeRealtimeModel, fake_capabilities

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]


class _DoneTask(AgentTask[str]):
    def __init__(self, value: str) -> None:
        super().__init__(instructions="done")
        self._value = value

    async def on_enter(self) -> None:
        self.complete(self._value)


class _TalkingTask(AgentTask[str]):
    """Completes after putting turns in its context, so there is something to summarize."""

    def __init__(self, value: str) -> None:
        super().__init__(instructions="talking")
        self._value = value

    async def on_enter(self) -> None:
        ctx = self.chat_ctx.copy()
        ctx.add_message(role="user", content="my card number is 4152637489012345")
        ctx.add_message(role="assistant", content="noted")
        await self.update_chat_ctx(ctx)
        self.complete(self._value)


class _ToolResultTask(_DoneTask):
    async def on_enter(self) -> None:
        ctx = self.chat_ctx.copy()
        ctx.insert(
            [
                llm.FunctionCall(call_id=self._value, name="lookup", arguments="{}"),
                llm.FunctionCallOutput(
                    call_id=self._value, name="lookup", output=self._value, is_error=False
                ),
            ]
        )
        await self.update_chat_ctx(ctx, exclude_invalid_function_calls=False)
        self.complete(self._value)


class _SummarizeFailsLLM(FakeLLM):
    """Every summarization call fails the way a rate-limited or dropped provider would."""

    def chat(self, **kwargs):  # type: ignore[override]
        raise RuntimeError("provider rate-limited (simulated)")


class _Group(Agent):
    """Runs a TaskGroup and records how it ended."""

    def __init__(self, task_cls: type[AgentTask[str]], *, summarize: bool) -> None:
        super().__init__(instructions="parent")
        self._task_cls = task_cls
        self._summarize = summarize
        self.outcome: object = None
        self.ctx_items: int = 0
        self.returned_ctx = llm.ChatContext.empty()
        self.summarize_error: Exception | None = None
        self.finished = asyncio.Event()

    async def on_enter(self) -> None:
        tg = TaskGroup(summarize_chat_ctx=self._summarize)
        tg.add(lambda: self._task_cls("alpha"), id="a", description="task a")
        tg.add(lambda: self._task_cls("beta"), id="b", description="task b")
        try:
            res = await tg
            self.outcome = ("ok", res.task_results)
            self.summarize_error = res.summarize_error
        except BaseException as e:  # noqa: BLE001
            self.outcome = ("raised", type(e).__name__, str(e))
        self.returned_ctx = self.chat_ctx.copy()
        self.ctx_items = len(self.returned_ctx.items)
        self.finished.set()


async def _run(model: object, task_cls: type[AgentTask[str]], *, summarize: bool) -> tuple:
    group = _Group(task_cls, summarize=summarize)
    async with AgentSession(llm=model) as session:  # type: ignore[arg-type]
        await session.start(group)
        await asyncio.wait_for(group.finished.wait(), 10)
        assert group.outcome is not None
        return group.outcome, group.ctx_items


@pytest.mark.asyncio
async def test_completed_results_survive_a_failing_summarization() -> None:
    """A summarization failure must not discard results the tasks already produced.

    Both tasks completed before the group tried to summarize, so their results are real
    work. The LLM call that condenses the context into one message is a post-processing
    step: when it fails, the caller still needs to see what the tasks returned.
    """
    outcome, _ = await _run(_SummarizeFailsLLM(), _TalkingTask, summarize=True)

    assert outcome[0] == "ok", f"expected the results to survive, got {outcome}"
    assert outcome[1] == {"a": "alpha", "b": "beta"}


@pytest.mark.asyncio
async def test_completed_results_survive_when_no_llm_can_summarize() -> None:
    """A realtime model cannot summarize, and that must not erase the results either.

    ``TaskGroup`` defaults to ``summarize_chat_ctx=True``, so a realtime session hits
    the summarization branch by default even though a ``RealtimeModel`` is not an
    ``LLM``. The assertion guarding that call fires as a bare ``AssertionError`` and,
    before this change, replaced the whole group outcome.
    """
    model = FakeRealtimeModel(capabilities=fake_capabilities())
    assert not isinstance(model, llm.LLM)

    outcome, _ = await _run(model, _TalkingTask, summarize=True)

    assert outcome[0] == "ok", f"expected the results to survive, got {outcome}"
    assert outcome[1] == {"a": "alpha", "b": "beta"}


@pytest.mark.asyncio
async def test_working_llm_still_summarizes_into_one_turn() -> None:
    """The control: with a healthy LLM the group succeeds and condenses the context."""
    outcome, _ = await _run(FakeLLM(), _TalkingTask, summarize=True)

    assert outcome[0] == "ok"
    assert outcome[1] == {"a": "alpha", "b": "beta"}


@pytest.mark.asyncio
async def test_failed_summary_preserves_tool_results_in_parent_context() -> None:
    parent = _Group(_ToolResultTask, summarize=True)
    async with AgentSession(llm=_SummarizeFailsLLM()) as session:
        await session.start(parent)
        await asyncio.wait_for(parent.finished.wait(), 10)
        assert parent.outcome == ("ok", {"a": "alpha", "b": "beta"})
        assert isinstance(parent.summarize_error, RuntimeError)
        assert [
            item.output
            for item in parent.returned_ctx.items
            if isinstance(item, llm.FunctionCallOutput)
        ] == ["alpha", "beta"]
        assert [
            item.call_id for item in parent.returned_ctx.items if isinstance(item, llm.FunctionCall)
        ] == ["alpha", "beta"]
