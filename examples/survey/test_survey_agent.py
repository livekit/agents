"""Tests for the survey agent's TaskGroup workflow.

Demonstrates the best practices documented in
https://docs.livekit.io/agents/logic/tasks/#testing-task-groups:

- **Initialize ``userdata``** — the ``session`` fixture passes
  ``userdata=_userdata()`` to every ``AgentSession``. The survey's tasks read
  ``session.userdata.candidate_name`` and write into ``task_results``; in
  Python, accessing an unset ``userdata`` raises ``ValueError``.
- **Sleep before the first ``session.run()``** — the fixture sleeps 0.5s
  after ``sess.start()`` (and the full-flow test sleeps again between
  TaskGroup sub-tasks). ``AgentTask`` transitions briefly leave
  ``session.llm`` unset, and ``session.run()`` does not fall back to it
  during that window.
- **Drive multiple turns** — the LLM often replies conversationally before
  invoking a completion tool. ``_drive_until_called`` sends an initial input,
  then keeps nudging until every expected tool name appears in
  ``sess.history.items``. This is the spirit of the docs' guidance to prefer
  ``contains_function_call()`` over ``next_event()``: don't couple the test
  to a specific event index in a single ``RunResult``.
- **Parse ``item.arguments`` with ``json.loads``** — see
  ``test_commute_task_records_can_commute`` and
  ``test_experience_task_records_years_and_description``. The other tests
  assert on ``userdata`` or membership in the tool-call set instead, where
  argument parsing isn't needed.
- **Don't assert on startup output** — none of the tests inspect the
  ``IntroTask`` greeting produced in ``on_enter``; ``RunResult`` does not
  capture output generated before the first ``session.run()``.
- **Tasks tested in isolation and as a group** — four isolation tests wrap
  a single ``AgentTask`` in ``_SingleTaskAgent``; ``test_full_task_group_flow``
  exercises the full ``TaskGroup`` ordering, ``on_task_completed`` callback,
  and ``task_results`` keying.

Test-only adjustment (not a documented practice, but worth flagging):
``_SurveyAgentForTesting`` sets ``summarize_chat_ctx=False`` so the final
``TaskGroup`` summarization LLM call doesn't run during tests.

Run with::

    uv run pytest examples/survey/test_survey_agent.py
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AsyncExitStack

import pytest

from livekit.agents import Agent, AgentSession, AgentTask, inference, llm
from livekit.agents.beta.workflows import TaskGroup
from livekit.agents.llm import FunctionCall

from .survey_agent import (
    BehavioralTask,
    CommuteTask,
    ExperienceTask,
    IntroTask,
    Userdata,
)

# AgentTask transitions briefly clear `session.llm`; sleep before the first
# `sess.run()` and between TaskGroup sub-tasks so the new sub-task can take
# over. See https://docs.livekit.io/agents/logic/tasks/#testing-task-groups.
_TASK_TRANSITION_DELAY = 0.5


def _llm_model() -> llm.LLM:
    return inference.LLM(
        model="openai/gpt-4.1-mini",
        extra_kwargs={"parallel_tool_calls": False, "temperature": 0.2},
    )


def _userdata() -> Userdata:
    # Initialize userdata: tasks read session.userdata.candidate_name and
    # write task_results. In Python, accessing an unset userdata raises
    # ValueError("AgentSession userdata is not set").
    return Userdata(filename="results-test.csv", candidate_name="", task_results={})


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


SessionStarter = Callable[[Agent], Awaitable[AgentSession]]


@pytest.fixture
async def session() -> AsyncIterator[SessionStarter]:
    """Yield a function that starts an `AgentSession` for a given agent.

    Handles LLM lifecycle, the `_TASK_TRANSITION_DELAY` after `start()`, and
    a non-draining shutdown on teardown so any in-flight LLM activity is
    cancelled instead of allowed to drain (which would let stdout output
    continue past the assertions).
    """
    async with AsyncExitStack() as stack:
        sessions: list[AgentSession] = []

        async def _start(agent: Agent) -> AgentSession:
            model = await stack.enter_async_context(_llm_model())
            sess = AgentSession(llm=model, userdata=_userdata())
            await sess.start(agent)
            sessions.append(sess)
            await asyncio.sleep(_TASK_TRANSITION_DELAY)
            return sess

        try:
            yield _start
        finally:
            for sess in sessions:
                sess.shutdown(drain=False)
                await asyncio.wait_for(sess.aclose(), timeout=10)


async def _run(sess: AgentSession, user_input: str, *, timeout: float = 30):
    return await asyncio.wait_for(sess.run(user_input=user_input), timeout=timeout)


def _called_tools(sess: AgentSession) -> set[str]:
    return {item.name for item in sess.history.items if item.type == "function_call"}


def _last_calls(sess: AgentSession, names: set[str]) -> dict[str, FunctionCall]:
    """Most recent function-call item in `sess.history` for each requested name."""
    found: dict[str, FunctionCall] = {}
    for item in reversed(sess.history.items):
        if item.type == "function_call" and item.name in names and item.name not in found:
            found[item.name] = item
        if found.keys() == names:
            break
    return found


async def _drive_until_called(
    sess: AgentSession,
    *,
    expected: str | set[str],
    initial: str,
    nudge: str = "Yes, that's right. Please go ahead and record it.",
    max_turns: int = 4,
) -> dict[str, FunctionCall]:
    """Drive a task to completion across multiple turns and return the matching calls.

    The LLM may respond conversationally before calling a completion tool.
    We send `initial`, then keep nudging until every tool name in `expected`
    appears in the session's function-call history (or we run out of turns).
    Returns a dict mapping each expected name to the most recent matching call.
    """
    required = {expected} if isinstance(expected, str) else set(expected)
    await _run(sess, initial)
    for _ in range(max_turns - 1):
        if required.issubset(_called_tools(sess)):
            break
        await _run(sess, nudge)
    assert required.issubset(_called_tools(sess)), (
        f"expected tools {required} not called; got {_called_tools(sess)}"
    )
    return _last_calls(sess, required)


class _SingleTaskAgent(Agent):
    """Thin wrapper that runs a single AgentTask and exits.

    Used for isolation tests so each task can be exercised without the full
    TaskGroup orchestration.
    """

    def __init__(self, task: AgentTask) -> None:
        super().__init__(instructions="Run a single survey task.")
        self._task = task

    async def on_enter(self) -> None:
        await self._task


# ---------------------------------------------------------------------------
# Isolated task tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_intro_task_in_isolation(session: SessionStarter) -> None:
    """IntroTask records the candidate's name via record_intro."""
    sess = await session(_SingleTaskAgent(IntroTask()))

    await _drive_until_called(
        sess,
        expected="record_intro",
        initial=(
            "Hi, my name is Alice. I'm a backend engineer with five years of "
            "experience building APIs at Acme."
        ),
    )

    # The IntroTask writes the candidate name into userdata.
    assert sess.userdata.candidate_name.lower() == "alice"


@pytest.mark.asyncio
async def test_commute_task_records_can_commute(session: SessionStarter) -> None:
    """CommuteTask records can_commute=True with the chosen method."""
    sess = await session(_SingleTaskAgent(CommuteTask()))

    calls = await _drive_until_called(
        sess,
        expected="record_commute_flexibility",
        initial="Yes, I can commute three days a week. I usually take the subway.",
    )

    # Function call arguments are stored as raw JSON — parse before asserting.
    args = json.loads(calls["record_commute_flexibility"].arguments)
    assert args["can_commute"] is True
    assert args["commute_method"] == "subway"


@pytest.mark.asyncio
async def test_experience_task_records_years_and_description(
    session: SessionStarter,
) -> None:
    """ExperienceTask captures years_of_experience and a description."""
    sess = await session(_SingleTaskAgent(ExperienceTask()))

    calls = await _drive_until_called(
        sess,
        expected="record_experience",
        initial=(
            "I have five years of experience total. I started as a junior engineer "
            "at Acme working on data pipelines for two years, then moved to Globex "
            "as a senior backend engineer for the past three years."
        ),
    )

    args = json.loads(calls["record_experience"].arguments)
    assert args["years_of_experience"] == 5
    assert "acme" in args["experience_description"].lower()


@pytest.mark.asyncio
async def test_behavioral_task_completes_after_three_records(
    session: SessionStarter,
) -> None:
    """BehavioralTask only completes once strengths, weaknesses, and work
    style have all been recorded — typically requires multiple turns.
    """
    sess = await session(_SingleTaskAgent(BehavioralTask()))

    await _drive_until_called(
        sess,
        expected={"record_strengths", "record_weaknesses", "record_work_style"},
        initial=(
            "My biggest strength is debugging hard distributed systems issues. "
            "My main weakness is that I sometimes over-engineer early prototypes. "
            "I work best as part of a team — that is my work style."
        ),
        max_turns=6,
    )


# ---------------------------------------------------------------------------
# Full TaskGroup flow
# ---------------------------------------------------------------------------


class _SurveyAgentForTesting(Agent):
    """Mirrors the production ``SurveyAgent`` but disables chat-context
    summarization and skips the email step.

    - ``summarize_chat_ctx=False``: summarization issues an additional LLM call
      at the end of the group; disabling it keeps the test focused on task
      orchestration and avoids depending on summary quality.
    - We omit ``GetEmailTask`` and the CSV write to keep the test offline-safe
      and free of filesystem side effects. The orchestration semantics being
      verified — sequential ordering, ``task_results`` keying, and
      ``on_task_completed`` callbacks — are identical to production.

    `done` is set after `task_results` is assigned, so the test can wait on a
    single signal instead of polling.
    """

    def __init__(self, completed_ids: list[str], done: asyncio.Event) -> None:
        super().__init__(instructions="You are a survey agent screening candidates.")
        self._completed_ids = completed_ids
        self._done = done

    async def on_enter(self) -> None:
        async def _on_task_completed(event):  # type: ignore[no-untyped-def]
            self._completed_ids.append(event.task_id)

        group = TaskGroup(
            summarize_chat_ctx=False,
            on_task_completed=_on_task_completed,
        )
        group.add(lambda: IntroTask(), id="intro", description="Collect name and intro.")
        group.add(lambda: CommuteTask(), id="commute", description="Ask about commute.")
        group.add(lambda: ExperienceTask(), id="experience", description="Collect work history.")

        result = await group
        self.session.userdata.task_results = result.task_results
        self._done.set()


@pytest.mark.asyncio
async def test_full_task_group_flow(session: SessionStarter) -> None:
    """The TaskGroup runs IntroTask → CommuteTask → ExperienceTask in order,
    populates ``task_results`` keyed by task id, and fires
    ``on_task_completed`` exactly once per task.
    """
    completed_ids: list[str] = []
    done = asyncio.Event()

    sess = await session(_SurveyAgentForTesting(completed_ids=completed_ids, done=done))

    # Don't assert on startup output — the IntroTask greeting produced in
    # on_enter is not captured in the RunResult below.

    # Drive each task to completion before moving on. Each call loops through
    # additional turns until the expected tool fires — the LLM often replies
    # conversationally before invoking a completion tool. Sleep between
    # sub-tasks for the same reason as the initial transition delay.
    await _drive_until_called(
        sess,
        expected="record_intro",
        initial=(
            "My name is Bob, I'm a software engineer with eight years of experience "
            "focused on APIs."
        ),
    )
    await asyncio.sleep(_TASK_TRANSITION_DELAY)

    await _drive_until_called(
        sess,
        expected="record_commute_flexibility",
        initial="Yes, I can commute three days a week. I'd be driving in.",
    )
    await asyncio.sleep(_TASK_TRANSITION_DELAY)

    await _drive_until_called(
        sess,
        expected="record_experience",
        initial=(
            "I have eight years total — five at Initech on backend systems and the "
            "last three at Hooli leading an API team."
        ),
    )

    # Wait for `_SurveyAgentForTesting.on_enter` to finish: TaskGroup
    # finalization, callback delivery, and `task_results` assignment.
    await asyncio.wait_for(done.wait(), timeout=10)

    results = sess.userdata.task_results
    assert completed_ids == ["intro", "commute", "experience"], (
        f"tasks completed out of order: {completed_ids}"
    )
    assert set(results.keys()) == {"intro", "commute", "experience"}
    assert results["intro"].name.lower() == "bob"
    assert results["commute"].can_commute is True
    assert results["commute"].commute_method == "driving"
    assert results["experience"].years_of_experience == 8
