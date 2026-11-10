from __future__ import annotations

import asyncio
import contextlib

import pytest

from livekit.agents import Agent, AgentSession, AgentTask, RunContext, function_tool
from livekit.agents.beta.workflows import TaskGroup
from livekit.agents.llm import ChatMessage, FunctionToolCall, ToolError
from livekit.agents.voice.events import CloseReason

from .fake_session import FakeActions, create_session, run_session

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]

SESSION_TIMEOUT = 60.0
# run_session closes the session itself after this delay, the close under test comes first
DRAIN_DELAY = 20.0

QUESTION = "what is your experience"
GOODBYE = "thank you for your time, goodbye"


@function_tool
async def disqualify(context: RunContext) -> None:
    """End the interview and hang up."""
    # the shape of the disqualify tool in examples/survey/agent.py
    context.session.generate_reply(instructions="say_goodbye")
    context.session.shutdown()


class InterviewTask(AgentTask[str]):
    def __init__(self) -> None:
        super().__init__(instructions="interview", tools=[disqualify])

    async def on_enter(self) -> None:
        self.session.generate_reply(instructions="ask_question")

    @function_tool
    async def finish(self, context: RunContext) -> None:
        """Thank the candidate, end the interview and hang up."""
        context.session.generate_reply(instructions="say_goodbye")
        self.complete("finished")
        context.session.shutdown()


class ParentOnEnter(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="parent")

    async def on_enter(self) -> None:
        # run_session pushes its only audio right after start(): let it reach the STT stream
        # before the handoff moves the stream to the task, or the fake STT never starts
        await asyncio.sleep(0.2)
        with contextlib.suppress(ToolError):
            await InterviewTask()


class ParentWithTaskGroup(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="parent")

    async def on_enter(self) -> None:
        await asyncio.sleep(0.2)  # see ParentOnEnter
        with contextlib.suppress(ToolError):
            await TaskGroup(summarize_chat_ctx=False).add(
                InterviewTask, id="interview", description="the interview"
            )


class ParentWithTool(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="parent")

    @function_tool
    async def start_interview(self, context: RunContext) -> str:
        """Start the interview."""
        return await InterviewTask()


class ParentWithToolTaskGroup(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="parent")

    @function_tool
    async def start_interview(self, context: RunContext) -> str:
        """Start the interview."""
        result = await TaskGroup(summarize_chat_ctx=False).add(
            InterviewTask, id="interview", description="the interview"
        )
        return str(result.task_results["interview"])


async def _said_during(session: AgentSession, agent: Agent) -> list[tuple[str | None, bool]]:
    loop = asyncio.get_running_loop()
    messages: list[ChatMessage] = []
    closes: list[tuple[float, CloseReason]] = []
    session.on(
        "conversation_item_added",
        lambda ev: messages.append(ev.item) if ev.item.type == "message" else None,
    )
    session.on("close", lambda ev: closes.append((loop.time(), ev.reason)))
    started_at = loop.time()
    await asyncio.wait_for(
        run_session(session, agent, drain_delay=DRAIN_DELAY), timeout=SESSION_TIMEOUT
    )
    # closed once, by the shutdown() under test and not by run_session
    [(closed_at, reason)] = closes
    assert reason == CloseReason.USER_INITIATED and closed_at - started_at < DRAIN_DELAY
    return [(m.text_content, m.interrupted) for m in messages if m.role == "assistant"]


def _interview_actions(*, from_tool: bool, tool: str) -> FakeActions:
    actions = FakeActions()
    if from_tool:
        actions.add_user_speech(0.5, 1.0, "start")
        actions.add_llm(
            "",
            tool_calls=[FunctionToolCall(name="start_interview", arguments="{}", call_id="s")],
        )
    actions.add_llm(QUESTION, input="ask_question")
    actions.add_tts(1.0)
    actions.add_user_speech(4.0, 5.0, "i will not answer that")
    actions.add_llm("", tool_calls=[FunctionToolCall(name=tool, arguments="{}", call_id="t")])
    actions.add_llm(GOODBYE, input="say_goodbye")
    actions.add_tts(3.0)
    return actions


async def test_shutdown_lets_an_agent_task_finish_its_sentence() -> None:
    """A drained shutdown() while an AgentTask is speaking lets the sentence finish, as it
    does for a plain Agent, instead of cutting it with the forced interrupt of cancel()."""
    actions = FakeActions()
    actions.add_llm(QUESTION, input="ask_question")
    actions.add_tts(5.0)
    session = create_session(actions, with_stt=False)

    # e.g. a max call duration timer, firing while the task is speaking
    asyncio.get_running_loop().call_later(2.0, session.shutdown)
    said = await _said_during(session, ParentOnEnter())
    assert said == [(QUESTION, False)]


@pytest.mark.parametrize(
    "parent", [ParentOnEnter, ParentWithTaskGroup, ParentWithTool, ParentWithToolTaskGroup]
)
async def test_goodbye_queued_before_shutdown_plays_out(parent: type[Agent]) -> None:
    """The goodbye that a tool of the task queues before calling shutdown() plays in full,
    and when the task was awaited from a tool, the parent doesn't answer the cancellation."""
    from_tool = parent in (ParentWithTool, ParentWithToolTaskGroup)
    actions = _interview_actions(from_tool=from_tool, tool="disqualify")
    # what the parent would say to the task the close cancelled
    actions.add_llm("sorry, something went wrong", input="AgentTask interview_task is cancelled")
    actions.add_tts(2.0)
    session = create_session(actions)

    said = await _said_during(session, parent())
    assert said == [(QUESTION, False), (GOODBYE, False)]


async def test_task_completed_before_shutdown_keeps_goodbye_and_parent_reply() -> None:
    """When the task completes before shutdown(), its goodbye plays and the parent still
    answers the result, as a plain Agent's tool reply would during a drained close."""
    actions = _interview_actions(from_tool=True, tool="finish")
    actions.add_llm("the interview is finished", input="finished")
    actions.add_tts(2.0)
    session = create_session(actions)

    said = await _said_during(session, ParentWithTool())
    assert said == [(QUESTION, False), (GOODBYE, False), ("the interview is finished", False)]


async def test_shutdown_without_drain_still_interrupts_an_agent_task() -> None:
    """drain=False, like a participant disconnect or aclose(), still cuts the task off."""
    actions = FakeActions()
    actions.add_llm(QUESTION, input="ask_question")
    actions.add_tts(5.0)
    session = create_session(actions, with_stt=False)

    asyncio.get_running_loop().call_later(2.0, lambda: session.shutdown(drain=False))
    said = await _said_during(session, ParentOnEnter())
    [(text, interrupted)] = said
    assert interrupted and text != QUESTION
