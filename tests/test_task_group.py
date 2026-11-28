from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

import pytest

from livekit.agents import Agent, AgentSession, AgentTask, RunContext, function_tool
from livekit.agents.beta.workflows import TaskGroup, TaskGroupProgress, TaskGroupResult
from livekit.agents.llm import (
    ChatContext,
    FunctionToolCall,
    LLMStream,
    Tool,
    ToolContext,
    ToolError,
)
from livekit.agents.types import NOT_GIVEN

from .fake_llm import FakeLLM, FakeLLMResponse

pytestmark = pytest.mark.unit


def _never_started() -> AgentTask:
    raise AssertionError("building the out_of_scope tool doesn't start a task")


def _out_of_scope_targets(group: TaskGroup, *, active_task_id: str) -> list[str] | None:
    tool = group._build_out_of_scope_tool(active_task_id=active_task_id)
    if tool is None:
        return None
    (schema,) = ToolContext([tool]).parse_function_tools("anthropic")
    targets: list[str] = schema["input_schema"]["properties"]["task_ids"]["items"]["enum"]
    return targets


def test_out_of_scope_offered_only_with_another_visited_task() -> None:
    group = (
        TaskGroup()
        .add(_never_started, id="email", description="Collect the email")
        .add(_never_started, id="phone", description="Collect the phone number")
    )

    # the first task of a fresh group has nothing to regress to
    assert _out_of_scope_targets(group, active_task_id="email") is None

    # only the active task has been visited, e.g. a group restored from saved progress:
    # still nothing to regress to, and an empty enum makes Anthropic reject the request
    group._visited_tasks.add("email")
    assert _out_of_scope_targets(group, active_task_id="email") is None

    # later tasks can regress to the other visited tasks
    assert _out_of_scope_targets(group, active_task_id="phone") == ["email"]
    group._visited_tasks.add("phone")
    assert _out_of_scope_targets(group, active_task_id="email") == ["phone"]


# -- resuming from saved progress ---------------------------------------------------------


class _Step(AgentTask[str]):
    """Asks its question on entry, unless built to resume, and completes on its save tool."""

    def __init__(
        self,
        task_id: str,
        *,
        entered: list[_Step],
        chat_ctx: ChatContext | None = None,
        resuming: bool = False,
    ) -> None:
        super().__init__(instructions=f"step {task_id}", chat_ctx=chat_ctx or NOT_GIVEN)
        self.task_id = task_id
        self._entered = entered
        self._resuming = resuming

    async def on_enter(self) -> None:
        self._entered.append(self)
        if not self._resuming:
            self.session.generate_reply(instructions=f"ask {self.task_id}")

    @function_tool
    async def save(self, ctx: RunContext, value: str) -> None:
        """Saves the user's answer."""
        self.complete(value)


class _Root(Agent):
    def __init__(self, group: TaskGroup) -> None:
        super().__init__(instructions="root")
        self.group = group
        self.result: TaskGroupResult | None = None
        self.error: Exception | None = None

    async def on_enter(self) -> None:
        try:
            self.result = await self.group
        except ToolError:
            pass  # the session closed while a task was waiting for the user
        except Exception as e:
            self.error = e


def _group(entered: list[_Step], **kwargs: Any) -> TaskGroup:
    return (
        TaskGroup(summarize_chat_ctx=False, **kwargs)
        .add(lambda: _Step("a", entered=entered), id="a", description="step a")
        .add(lambda: _Step("b", entered=entered), id="b", description="step b")
    )


def _reply(input: str, tool: str, **arguments: Any) -> FakeLLMResponse:
    call = FunctionToolCall(name=tool, arguments=json.dumps(arguments), call_id=f"call_{input}")
    return FakeLLMResponse(input=input, content="", ttft=0, duration=0, tool_calls=[call])


# any other input gets an empty reply, so it only shows what the current task offers
SCRIPT = [
    _reply("a1", "save", value="a1"),
    _reply("b1", "save", value="b1"),
    _reply("redo a", "out_of_scope", task_ids=["a"]),
]


def _offered(tools: list[Tool]) -> tuple[list[str], list[str] | None]:
    """Names of the tools offered, and the out_of_scope targets (None without the tool)."""
    schemas = ToolContext(tools).parse_function_tools("anthropic")
    targets = None
    for schema in schemas:
        if schema["name"] == "out_of_scope":
            targets = sorted(schema["input_schema"]["properties"]["task_ids"]["items"]["enum"])
    return sorted(schema["name"] for schema in schemas), targets


class _RecordingLLM(FakeLLM):
    def __init__(self) -> None:
        super().__init__(fake_responses=SCRIPT)
        self.calls: list[tuple[str | None, list[str], list[str] | None]] = []
        self.contexts: list[ChatContext] = []

    def chat(
        self, *, chat_ctx: ChatContext, tools: list[Tool] | None = None, **kwargs: Any
    ) -> LLMStream:
        last = chat_ctx.items[-1] if chat_ctx.items else None
        text = last.text_content if last is not None and last.type == "message" else None
        self.calls.append((text, *_offered(tools or [])))
        self.contexts.append(chat_ctx.copy())
        return super().chat(chat_ctx=chat_ctx, tools=tools, **kwargs)


async def _until(condition: Callable[[], bool]) -> None:
    for _ in range(200):
        if condition():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not met")


async def _settled(group: TaskGroup, entered: list[_Step]) -> None:
    """Wait until the group's current task has entered, or the group is done."""

    def ready() -> bool:
        task = group.current_task
        return group.done() or (task is not None and not task.done() and task in entered)

    await _until(ready)


AT_B = TaskGroupProgress(pending=["b"], results={"a": "a1"}, visited=["a", "b"])


@pytest.mark.virtual_time
async def test_progress_follows_a_running_group() -> None:
    entered: list[_Step] = []
    group = _group(entered)
    assert group.progress == TaskGroupProgress(pending=["a", "b"], results={}, visited=[])
    assert group.current_task is None

    root = _Root(group)
    async with AgentSession(llm=_RecordingLLM()) as session:
        await session.start(root)
        await _settled(group, entered)
        assert group.current_task is entered[-1]
        assert group.progress == TaskGroupProgress(pending=["a", "b"], results={}, visited=["a"])

        await session.run(user_input="a1")
        await _settled(group, entered)
        assert [s.task_id for s in entered] == ["a", "b"]
        assert group.progress == TaskGroupProgress(
            pending=["b"], results={"a": "a1"}, visited=["a", "b"]
        )

        await session.run(user_input="b1")
        await _until(lambda: root.result is not None)
        assert group.progress == TaskGroupProgress(
            pending=[], results={"a": "a1", "b": "b1"}, visited=["a", "b"]
        )


@pytest.mark.virtual_time
async def test_resume_starts_at_the_first_pending_task_and_keeps_earlier_results() -> None:
    entered: list[_Step] = []
    group = _group(entered, progress=AT_B)
    root = _Root(group)
    async with AgentSession(llm=_RecordingLLM()) as session:
        await session.start(root)
        await _settled(group, entered)
        assert [s.task_id for s in entered] == ["b"]
        assert group.progress == AT_B

        await session.run(user_input="b1")
        await _until(lambda: root.result is not None)
    assert root.result is not None
    assert root.result.task_results == {"a": "a1", "b": "b1"}


@pytest.mark.virtual_time
async def test_resume_runs_the_given_current_task_with_its_own_context() -> None:
    entered: list[_Step] = []
    step_ctx = ChatContext.empty()
    step_ctx.add_message(role="assistant", content="what is b?")
    step_ctx.add_message(role="user", content="earlier answer")
    current = _Step("b", entered=entered, chat_ctx=step_ctx, resuming=True)
    group = _group(entered, progress=AT_B, current_task=current)
    llm = _RecordingLLM()
    async with AgentSession(llm=llm) as session:
        await session.start(_Root(group))
        await _settled(group, entered)
        assert entered == [current]
        assert group.current_task is current

        await session.run(user_input="next answer")
    [ctx] = llm.contexts
    texts = [i.text_content for i in ctx.items if i.type == "message"]
    assert texts[-3:] == ["what is b?", "earlier answer", "next answer"]


@pytest.mark.virtual_time
async def test_resume_can_regress_to_a_restored_visited_task() -> None:
    entered: list[_Step] = []
    group = _group(entered, progress=AT_B)
    llm = _RecordingLLM()
    async with AgentSession(llm=llm) as session:
        await session.start(_Root(group))
        await _settled(group, entered)

        await session.run(user_input="redo a")
        await _settled(group, entered)
        assert [s.task_id for s in entered] == ["b", "a"]
        assert group.progress.pending == ["a", "b"]
    # out_of_scope is not offered to the on_enter replies (ToolFlag.IGNORE_ON_ENTER)
    assert llm.calls == [
        ("ask b", ["save"], None),
        ("redo a", ["out_of_scope", "save"], ["a"]),
        ("ask a", ["save"], None),
    ]


@pytest.mark.virtual_time
async def test_resumed_group_offers_the_same_tools_as_a_long_lived_one() -> None:
    inputs = ["x", "a1", "x", "redo a", "x"]

    # one session that never restarts
    stock_llm = _RecordingLLM()
    entered: list[_Step] = []
    group = _group(entered)
    async with AgentSession(llm=stock_llm) as session:
        await session.start(_Root(group))
        await _settled(group, entered)
        for text in inputs:
            await session.run(user_input=text)
            await _settled(group, entered)

    # a new session per input, each resumed from what the previous one saved: covers a
    # restart inside the first task, after a task completed, and after a regression. The
    # resumed task doesn't ask its question again, so the calls match one for one.
    resumed_llm = _RecordingLLM()
    progress: TaskGroupProgress | None = None
    step_ctx: ChatContext | None = None
    for text in [None, *inputs]:
        entered = []
        current = (
            _Step(progress.pending[0], entered=entered, chat_ctx=step_ctx, resuming=True)
            if progress is not None
            else None
        )
        group = _group(entered, progress=progress, current_task=current)
        async with AgentSession(llm=resumed_llm) as session:
            await session.start(_Root(group))
            await _settled(group, entered)
            if text is not None:
                await session.run(user_input=text)
                await _settled(group, entered)
            progress = group.progress
            assert group.current_task is not None
            step_ctx = group.current_task.chat_ctx.copy()

    assert resumed_llm.calls == stock_llm.calls
    assert [c for c in stock_llm.calls if c[0] == "x"] == [
        ("x", ["save"], None),  # in the first task: nothing to go back to
        ("x", ["out_of_scope", "save"], ["a"]),
        ("x", ["out_of_scope", "save"], ["b"]),  # in a again, after the regression
    ]


@pytest.mark.virtual_time
async def test_progress_naming_an_unknown_task_fails_the_group() -> None:
    entered: list[_Step] = []
    group = _group(entered, progress=TaskGroupProgress(pending=["c"], results={}, visited=[]))
    root = _Root(group)
    async with AgentSession(llm=_RecordingLLM()) as session:
        await session.start(root)
        await _until(lambda: root.error is not None)
    assert isinstance(root.error, ValueError)
    assert "c" in str(root.error)
    assert entered == []


async def test_current_task_requires_progress() -> None:
    with pytest.raises(ValueError):
        TaskGroup(current_task=_Step("a", entered=[]))
