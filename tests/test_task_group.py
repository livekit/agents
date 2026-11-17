from __future__ import annotations

import pytest

from livekit.agents import AgentTask
from livekit.agents.beta.workflows import TaskGroup
from livekit.agents.llm import ToolContext

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
