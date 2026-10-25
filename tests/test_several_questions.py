"""Several questions at once (docs/delegation.md): a task pauses once, on one question that holds
every part its calls asked, and the task's model reads the answer; questions of different tasks
wait in a queue, one in front of the person."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from livekit.agents import Agent, InputRequired, RunContext
from livekit.agents.a2a import A2ASessionContext
from livekit.agents.llm import FunctionToolCall
from livekit.agents.storage import AgentDBStorage
from livekit.agents.voice.agent_run import JOINT_QUESTION
from livekit.agents.voice.tool_executor import UPDATE_TEMPLATE

from .fake_llm import FakeLLMResponse
from .test_a2a_server import (  # noqa: F401  (no_project_env: autouse)
    http_client,
    make_server,
    no_project_env,
    state_of,
    wire,
)
from .test_agent_api import RecordingLLM
from .test_live_inbox import stream, tasks_of, told, until

pytestmark = [pytest.mark.unit]


def _call(name: str, call_id: str, **arguments: str) -> FunctionToolCall:
    return FunctionToolCall(name=name, arguments=json.dumps(arguments), call_id=call_id)


def _reply(text: str, content: str = "", *calls: FunctionToolCall) -> FakeLLMResponse:
    return FakeLLMResponse(input=text, content=content, ttft=0, duration=0, tool_calls=list(calls))


def _asked(frames: list[dict[str, Any]]) -> str:
    return str(frames[-1]["statusUpdate"]["status"]["message"]["parts"][0]["text"])


def _progress(name: str, call_id: str, message: str) -> str:
    return UPDATE_TEMPLATE.format(function_name=name, call_id=call_id, message=message)


async def test_two_calls_of_one_step_that_both_ask_give_one_question_and_one_answer(
    agentdb_storage: AgentDBStorage,
) -> None:
    """One step books two visits and each asks for the email (s18): the task pauses once, on
    the question asked once, and the answer goes to the model, which books both."""
    booked: list[tuple[str, str]] = []
    office = Agent(instructions="back office")

    @office.tool
    async def book_visit(ctx: RunContext, day: str, email: str = "") -> str | InputRequired:
        """Book a visit.

        Args:
            day: the day
            email: where the confirmation goes
        """
        if not email:
            return InputRequired("What email address should the confirmation go to?")
        booked.append((day, email))
        return f"booked {day}"

    office.update_options(
        llm=RecordingLLM(
            fake_responses=[
                _reply(
                    "book monday and tuesday",
                    "",
                    _call("book_visit", "b1", day="monday"),
                    _call("book_visit", "b2", day="tuesday"),
                ),
                _reply(
                    "a@b.co",
                    "",
                    _call("book_visit", "b3", day="monday", email="a@b.co"),
                    _call("book_visit", "b4", day="tuesday", email="a@b.co"),
                ),
                _reply("booked tuesday", "Both are booked."),
            ]
        )
    )

    async def handler(ctx: A2ASessionContext[Any]) -> Any:
        return await office.run(ctx.message, message_history=ctx.message_history)

    server, _ = make_server(agentdb_storage, handler, inbox="live")
    thread_id = (await agentdb_storage.threads.create_thread()).thread_id
    async with http_client(server) as http:
        first = await stream(http, wire("book monday and tuesday", "m1", context_id=thread_id))
        second = await stream(http, wire("a@b.co", "m2", context_id=thread_id))

    assert state_of(first[-1]) == "TASK_STATE_INPUT_REQUIRED"
    assert _asked(first) == "What email address should the confirmation go to?"
    assert state_of(second[-1]) == "TASK_STATE_COMPLETED"
    assert second[0]["task"]["id"] == first[0]["task"]["id"], "the answer continues the task"
    assert booked == [("monday", "a@b.co"), ("tuesday", "a@b.co")]
    tasks = await tasks_of(agentdb_storage, thread_id)
    assert [t.status for t in tasks.values()] == ["completed"], "nothing is asked again"


async def test_a_question_of_work_in_the_background_joins_its_paused_task(
    agentdb_storage: AgentDBStorage,
) -> None:
    """A step asks for the email and starts a check in the background; the check asks too while
    the task waits on the email: the task's one question is asked again with both parts, and
    one answer gives both."""
    release = asyncio.Event()
    started = asyncio.Event()
    got: dict[str, str] = {}
    desk = Agent(instructions="intake")

    @desk.tool
    async def email(ctx: RunContext, address: str = "") -> str | InputRequired:
        """Record the email.

        Args:
            address: the email address
        """
        if not address:
            return InputRequired("Which email?", verbatim=True)
        got["email"] = address
        return f"email {address}"

    @desk.tool
    async def coverage(ctx: RunContext, payer: str = "") -> str | InputRequired:
        """Check the coverage.

        Args:
            payer: the payer
        """
        if payer:
            got["payer"] = payer
            return f"covered by {payer}"
        await ctx.update("checking")
        started.set()
        await release.wait()
        return InputRequired("Which payer?", verbatim=True)

    desk.update_options(
        llm=RecordingLLM(
            fake_responses=[
                _reply("sign me up", "", _call("email", "e1"), _call("coverage", "c1")),
                _reply(
                    "a@b.co, Aetna",
                    "",
                    _call("email", "e2", address="a@b.co"),
                    _call("coverage", "c2", payer="Aetna"),
                ),
                _reply("covered by Aetna", "All set."),
            ]
        )
    )

    async def handler(ctx: A2ASessionContext[Any]) -> Any:
        return await desk.run(ctx.message, message_history=ctx.message_history)

    server, _ = make_server(agentdb_storage, handler, inbox="live")
    thread_id = (await agentdb_storage.threads.create_thread()).thread_id
    async with http_client(server) as http:
        first = await stream(http, wire("sign me up", "m1", context_id=thread_id))
        assert state_of(first[-1]) == "TASK_STATE_INPUT_REQUIRED"
        assert _asked(first) == "Which email?"
        task_id = first[0]["task"]["id"]
        await started.wait()
        release.set()

        async def joined() -> bool:
            row = (await tasks_of(agentdb_storage, thread_id))[task_id]
            text = (row.status_message or {}).get("parts", [{}])[0].get("text")
            return row.status == "input-required" and text == "Which email?\n\nWhich payer?"

        await until(joined)
        answer = await stream(http, wire("a@b.co, Aetna", "m2", context_id=thread_id))

    assert answer[0]["task"]["id"] == task_id
    assert state_of(answer[-1]) == "TASK_STATE_COMPLETED"
    assert got == {"email": "a@b.co", "payer": "Aetna"}


async def test_questions_of_two_tasks_still_queue(agentdb_storage: AgentDBStorage) -> None:
    """The work of two tasks ends together and both ask: one question is asked, the other task
    waits behind it, and is asked once the first is answered; the model reads each answer."""
    release = asyncio.Event()
    started: list[str] = []
    got: dict[str, str] = {}
    desk = Agent(instructions="intake")

    @desk.tool
    async def check(ctx: RunContext, what: str) -> InputRequired:
        """Check something in the background.

        Args:
            what: what to check
        """
        await ctx.update("checking")
        started.append(what)
        await release.wait()
        return InputRequired(
            "Which payer?" if what == "coverage" else "Which email?", verbatim=True
        )

    @desk.tool
    async def record(ctx: RunContext, field: str, value: str) -> str:
        """Record an answer.

        Args:
            field: payer or email
            value: the answer
        """
        got[field] = value
        return f"{field} {value}"

    resumed = [
        told(*ids) + "\n" + JOINT_QUESTION.format(question=q)
        for ids in (("c1", "c2"), ("c2", "c1"))
        for q in ("Which payer?", "Which email?")
    ]
    desk.update_options(
        llm=RecordingLLM(
            fake_responses=[
                _reply("check my coverage", "", _call("check", "c1", what="coverage")),
                _reply(_progress("check", "c1", "checking"), "Checking."),
                _reply("and send a receipt", "", _call("check", "c2", what="receipt")),
                _reply(_progress("check", "c2", "checking"), "That too."),
                *(_reply(text, "") for text in resumed),
                _reply("Aetna", "", _call("record", "r1", field="payer", value="Aetna")),
                _reply("payer Aetna", "Noted."),
                _reply("a@b.co", "", _call("record", "r2", field="email", value="a@b.co")),
                _reply("email a@b.co", "All set."),
            ]
        )
    )

    async def handler(ctx: A2ASessionContext[Any]) -> Any:
        return await desk.run(ctx.message, message_history=ctx.message_history)

    server, _ = make_server(agentdb_storage, handler, inbox="live")
    thread_id = (await agentdb_storage.threads.create_thread()).thread_id
    async with http_client(server) as http:
        first = asyncio.create_task(
            stream(http, wire("check my coverage", "m1", context_id=thread_id))
        )
        await until(lambda: asyncio.sleep(0, result=started == ["coverage"]))
        second = asyncio.create_task(
            stream(http, wire("and send a receipt", "m2", context_id=thread_id))
        )
        await until(lambda: asyncio.sleep(0, result=len(started) == 2))
        release.set()
        frames = await asyncio.wait_for(asyncio.gather(first, second), 30)
        tasks = await tasks_of(agentdb_storage, thread_id)
        ids = [f[0]["task"]["id"] for f in frames]
        assert [tasks[i].status for i in ids] == ["input-required", "input-required"]
        queued = [i for i in ids if tasks[i].data.metadata.get("livekit/queued")]
        assert len(queued) == 1, "one question in front, the other waits"
        front = next(i for i in ids if i not in queued)
        payer_first = tasks[front].status_message["parts"][0]["text"] == "Which payer?"
        answer = await stream(
            http, wire("Aetna" if payer_first else "a@b.co", "m3", context_id=thread_id)
        )
        later = await stream(
            http, wire("a@b.co" if payer_first else "Aetna", "m4", context_id=thread_id)
        )

    assert state_of(answer[-1]) == "TASK_STATE_COMPLETED"
    assert state_of(later[-1]) == "TASK_STATE_COMPLETED"
    assert got == {"payer": "Aetna", "email": "a@b.co"}
