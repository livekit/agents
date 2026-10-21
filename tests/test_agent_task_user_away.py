from __future__ import annotations

import asyncio

import pytest

from livekit.agents import Agent, AgentSession, AgentTask, RunContext, function_tool
from livekit.agents.llm import FunctionToolCall
from livekit.agents.voice.events import UserStateChangedEvent

from .fake_llm import FakeLLM, FakeLLMResponse

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]

USER_AWAY_TIMEOUT = 2.0


class AskEmail(AgentTask[str]):
    def __init__(self) -> None:
        super().__init__(instructions="Ask for the user's e-mail address.")

    async def on_enter(self) -> None:
        self.session.generate_reply(instructions="ask_email")

    @function_tool
    async def save(self, ctx: RunContext, email: str) -> None:
        """Saves the user's e-mail address."""
        self.complete(email)


class CollectingAgent(Agent):
    def __init__(self, *, work_after_task: float = 0.0) -> None:
        super().__init__(instructions="root agent")
        self._work_after_task = work_after_task
        self.task_done = asyncio.Event()

    @function_tool
    async def collect_email(self, ctx: RunContext) -> str:
        """Collects the user's e-mail address."""
        email = await AskEmail()
        self.task_done.set()
        # backend work after the task, such as saving the address
        await asyncio.sleep(self._work_after_task)
        return f"saved {email}"


def _fake_llm() -> FakeLLM:
    return FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="go",
                content="",
                ttft=0,
                duration=0,
                tool_calls=[
                    FunctionToolCall(name="collect_email", arguments="{}", call_id="call_1")
                ],
            ),
            FakeLLMResponse(
                input="ask_email",
                content="What is your e-mail address?",
                ttft=0,
                duration=0,
            ),
            FakeLLMResponse(input="saved me@example.com", content="Thanks!", ttft=0, duration=0),
        ]
    )


async def test_user_goes_away_while_a_task_awaited_from_a_tool_waits_on_them() -> None:
    """The tool awaiting the task must not hold the away timer off: the task is the one
    waiting on the user (#8018)."""
    async with AgentSession(llm=_fake_llm(), user_away_timeout=USER_AWAY_TIMEOUT) as session:
        away: list[UserStateChangedEvent] = []
        session.on("user_state_changed", lambda ev: ev.new_state == "away" and away.append(ev))

        await session.start(CollectingAgent())
        await asyncio.wait_for(session.run(user_input="go"), timeout=5.0)

        # the task has asked its question, and the user stays silent
        await asyncio.sleep(USER_AWAY_TIMEOUT * 3)
        assert isinstance(session.current_agent, AskEmail)
        assert len(away) == 1


async def test_the_tool_holds_the_away_timer_again_once_its_task_is_done() -> None:
    """After the task completes, the tool is back at backend work and holds the timer off
    as any running tool does (#6883), also when the timer was running at that moment."""
    agent = CollectingAgent(work_after_task=USER_AWAY_TIMEOUT * 3)

    async with AgentSession(llm=_fake_llm(), user_away_timeout=USER_AWAY_TIMEOUT) as session:
        away: list[UserStateChangedEvent] = []
        session.on("user_state_changed", lambda ev: ev.new_state == "away" and away.append(ev))

        await session.start(agent)
        await asyncio.wait_for(session.run(user_input="go"), timeout=5.0)

        # the task completes without a user turn, while both sides listen and the timer runs
        task = session.current_agent
        assert isinstance(task, AskEmail)
        await asyncio.sleep(USER_AWAY_TIMEOUT / 2)
        task.complete("me@example.com")
        await asyncio.wait_for(agent.task_done.wait(), timeout=5.0)

        # the tool is saving the address; the user is waiting on it, not away
        await asyncio.sleep(USER_AWAY_TIMEOUT * 2)
        assert away == []
