"""Questions in text (design §4 "Flow C", §7): an expert asks by pausing its task, and the person's
answer continues that same task. Seen only on the A2A wire and in what reaches the model."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import timedelta
from typing import Any

import pytest

from livekit.agents import Agent, InputRequired, LiveSession, RunContext, ToolResult, utils
from livekit.agents.a2a import A2ASessionContext
from livekit.agents.a2a.codec import metadata_from_wire
from livekit.agents.a2a.extension import KIND, KIND_DELEGATION, MERGED_INTO, VERBATIM
from livekit.agents.delegation import A2ADelegate
from livekit.agents.llm import (
    ChatContext,
    ChatMessage,
    FunctionCall,
    FunctionCallOutput,
    GenerationCreatedEvent,
    MessageGeneration,
)

from .a2a_driver import (
    Door,
    Reply,
    ScriptedLLM,
    call,
    no_project_env,  # noqa: F401  (autouse: the demo environment must not shape the cards)
    output_of,
    rpc,
    wire,
)
from .fake_realtime import FakeRealtimeModel, fake_capabilities

pytestmark = [pytest.mark.unit]

QUESTION = "What email address?"
EMAIL = "ana@example.com"
DELEGATION = {KIND: KIND_DELEGATION}


def said(text: str) -> Callable[[ChatContext], bool]:
    """A request whose last item is the person's message ``text``."""

    def matches(ctx: ChatContext) -> bool:
        last = ctx.items[-1] if ctx.items else None
        return isinstance(last, ChatMessage) and last.role == "user" and last.text_content == text

    return matches


def user_texts(ctx: ChatContext) -> list[str]:
    return [
        i.text_content or "" for i in ctx.items if isinstance(i, ChatMessage) and i.role == "user"
    ]


def asking(
    *replies: Reply, verbatim: bool = True, booked: Callable[[str], Any] = "booked for {}".format
) -> Agent:
    """An agent on scripted ``replies`` whose ``book`` tool asks for an email until it has one."""
    agent = Agent(instructions="bookings", llm=ScriptedLLM(*replies))

    @agent.tool
    async def book(ctx: RunContext, email: str = "") -> Any:
        """Book the visit.

        Args:
            email: the person's email, when they gave it
        """
        return booked(email) if email else InputRequired(QUESTION, verbatim=verbatim)

    return agent


def model_of(agent: Agent) -> ScriptedLLM:
    assert isinstance(agent.llm, ScriptedLLM)
    return agent.llm


def runs(agent: Agent) -> Callable[[A2ASessionContext[Any]], Any]:
    async def handler(ctx: A2ASessionContext[Any]) -> Any:
        return await agent.run(ctx.message, message_history=ctx.message_history)

    return handler


def front_desk(delegate: A2ADelegate, *replies: Reply) -> Agent:
    return Agent(instructions="front desk", tools=[delegate], llm=ScriptedLLM(*replies))


def delegating(request: str) -> Reply:
    return Reply(tool_calls=[call("ask", "c1", request=request)])


@pytest.mark.rule("R44", "R41", "B8", "B28")
def test_a_question_names_no_tool_and_no_timeout() -> None:
    with pytest.raises(TypeError):
        InputRequired(QUESTION, tool=None)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        InputRequired(QUESTION, timeout=timedelta(seconds=5))  # type: ignore[call-arg]


@pytest.mark.rule("R32", "R34", "R35", "R39", "R44", "B28")
async def test_the_answer_continues_the_waiting_task_as_one_record(
    door: Callable[..., Door],
) -> None:
    agent = asking(
        Reply(tool_calls=[call("book", "b1")]),
        Reply(tool_calls=[call("book", "b2", email=EMAIL)], when=said(EMAIL)),
        Reply("Booked.", when=output_of("b2")),
    )
    desk = door(runs(agent))
    thread = await desk.thread()
    task_id = (await desk.send("book me in", context_id=thread, message_id="m1")).task["id"]

    # the person's message, with no task id, answers the question in front
    done = await desk.send(EMAIL, context_id=thread, message_id="m2")
    assert done.task["id"] == task_id and done.answer == "Booked."
    assert MERGED_INTO not in done.metadata
    assert [m["messageId"] for m in done.task["history"]] == ["m1", "m2"]
    # the asking call keeps its question; the task's model reads the answer as the person's
    request = model_of(agent).requests[1]
    [output] = [i for i in request.items if isinstance(i, FunctionCallOutput)]
    assert QUESTION in str(output.output)
    assert user_texts(request) == ["book me in", EMAIL]

    requests = len(model_of(agent).requests)
    again = await desk.send(EMAIL, context_id=thread, task_id=task_id, message_id="m2")
    assert again.answer == "Booked." and len(model_of(agent).requests) == requests
    later = {**wire("one more thing", "m3", context_id=thread), "taskId": task_id}
    error = (await rpc(desk.http, "message/send", {"message": later})).json()["error"]
    assert "completed" in error["message"]
    assert len(await desk.list_tasks(thread)) == 1


@pytest.mark.rule("R35")
async def test_a_task_id_of_a_running_task_is_a_new_task(door: Callable[..., Door]) -> None:
    entered, release = asyncio.Event(), asyncio.Event()

    async def handler(ctx: A2ASessionContext[Any]) -> str:
        if ctx.message.text_content == "first":
            entered.set()
            await release.wait()
        return "reply to: " + (ctx.message.text_content or "")

    desk = door(handler, inbox="queue")
    thread = await desk.thread()
    first = asyncio.create_task(desk.send("first", context_id=thread))
    await entered.wait()
    [running] = [t["id"] for t in await desk.list_tasks(thread)]
    second = asyncio.create_task(desk.send("second", context_id=thread, task_id=running))
    while len(await desk.list_tasks(thread)) < 2:
        await asyncio.sleep(0.05)
    release.set()
    _, done = await asyncio.gather(first, second)
    assert done.task["id"] != running and done.answer == "reply to: second"


@pytest.mark.rule("R35")
async def test_a_task_id_of_another_lane_is_not_found(
    door: Callable[..., Door], expert: Callable[..., A2ADelegate]
) -> None:
    specialist = expert(runs(asking(Reply(tool_calls=[call("book", "e1")]))))
    desk = door(runs(front_desk(specialist, delegating("book a visit"))))
    thread = await desk.thread()
    asked = await desk.send("book me in", context_id=thread)
    [expert_task] = [t["id"] for t in await desk.list_tasks(thread) if t["id"] != asked.task["id"]]

    message = {**wire(EMAIL, "m2", context_id=thread), "taskId": expert_task}
    error = (await rpc(desk.http, "message/send", {"message": message})).json()["error"]
    assert error["code"] == -32001, "TaskNotFoundError"


@pytest.mark.rule("R52")
async def test_a_message_taken_before_a_late_question_is_not_its_answer(
    door: Callable[..., Door],
) -> None:
    started, go = asyncio.Event(), asyncio.Event()
    agent = Agent(
        instructions="bookings",
        llm=ScriptedLLM(
            Reply(tool_calls=[call("slow_book", "b1")], when=said("book monday")),
            Reply("It is nine.", when=said("what time is it?")),
            Reply("Booked.", when=said(EMAIL)),
        ),
    )

    @agent.tool
    async def slow_book(ctx: RunContext) -> InputRequired:
        """Book the visit."""
        started.set()
        await go.wait()
        return InputRequired(QUESTION, verbatim=True)

    desk = door(runs(agent), inbox="queue")
    thread = await desk.thread()
    first = asyncio.create_task(desk.send("book monday", context_id=thread))
    await started.wait()
    # sent while the question is still on its way: it waits for the lane and is its own task
    second = asyncio.create_task(desk.send("what time is it?", context_id=thread))
    while len(await desk.list_tasks(thread)) < 2:
        await asyncio.sleep(0.05)
    go.set()
    asked, other = await asyncio.gather(first, second)
    assert other.task["id"] != asked.task["id"] and other.answer == "It is nine."

    answer = await desk.send(EMAIL, context_id=thread)
    assert answer.task["id"] == asked.task["id"] and answer.answer == "Booked."


# through a delegate
# ---------------------------------------------------------------------------------------------


@pytest.mark.rule("R40", "B30")
async def test_a_non_answer_goes_to_the_experts_model_in_the_same_task(
    door: Callable[..., Door], expert: Callable[..., A2ADelegate]
) -> None:
    specialist = asking(
        # the question comes after streamed text: the status message is the question
        Reply("Let me check.", tool_calls=[call("book", "e1")]),
        Reply("For the confirmation.", tool_calls=[call("book", "e2")], when=said("why?")),
    )
    desk_agent = front_desk(expert(runs(specialist)), delegating("book a visit"))
    desk = door(runs(desk_agent))
    thread = await desk.thread()
    asked = await desk.send("book me in", context_id=thread)
    assert asked.status_text == QUESTION, "asked as the expert wrote it"
    tasks = {t["id"] for t in await desk.list_tasks(thread)}

    again = await desk.send("why?", context_id=thread)
    assert again.task["id"] == asked.task["id"] and again.state == "input_required"
    assert user_texts(model_of(specialist).requests[1])[-1] == "why?"
    assert {t["id"] for t in await desk.list_tasks(thread)} == tasks
    assert len(model_of(desk_agent).requests) == 1


@pytest.mark.rule("R32", "R40")
async def test_a_nested_experts_question_is_answered_through_each_task(
    door: Callable[..., Door], expert: Callable[..., A2ADelegate]
) -> None:
    second = asking(
        Reply(tool_calls=[call("book", "s1")]),
        Reply(tool_calls=[call("book", "s2", email=EMAIL)], when=said(EMAIL)),
        booked=lambda email: ToolResult(f"Booked, {email}.", verbatim=True, reply_required=False),
    )
    first = front_desk(expert(runs(second), "second"), delegating("book a cleaning"))
    desk_agent = front_desk(expert(runs(first)), delegating("book a visit"))
    desk = door(runs(desk_agent))
    thread = await desk.thread()
    asked = await desk.send("book me in", context_id=thread)
    assert asked.status_text == QUESTION
    waiting = {t["id"] for t in await desk.list_tasks(thread)}
    assert len(waiting) == 3, "one task per lane"

    done = await desk.send(EMAIL, context_id=thread)
    assert done.task["id"] == asked.task["id"] and done.answer == "Booked, ana@example.com."
    # each lane's task went on with the answer as said; no model step forwarded it
    tasks = [await desk.get_task(t["id"], thread) for t in await desk.list_tasks(thread)]
    assert {t["id"] for t in tasks} == waiting
    assert all(t["history"][-1]["parts"][0]["text"] == EMAIL for t in tasks)
    assert len(model_of(desk_agent).requests) == len(model_of(first).requests) == 1


@pytest.mark.rule("R32", "R40")
async def test_on_a_call_the_answer_to_an_experts_question_continues_its_task(
    expert: Callable[..., A2ADelegate],
) -> None:
    specialist = asking(
        Reply(tool_calls=[call("book", "e1")]),
        Reply(tool_calls=[call("book", "e2", email=EMAIL)], when=said(EMAIL)),
        Reply("Booked.", when=output_of("e2")),
        verbatim=False,
    )
    turns: list[tuple[str, str]] = []

    async def handler(ctx: A2ASessionContext[Any]) -> Any:
        turns.append((ctx.task_id, ctx.message.text_content or ""))
        return await specialist.run(ctx.message, message_history=ctx.message_history)

    model = FakeRealtimeModel(capabilities=fake_capabilities(audio_output=False))

    async def reply(*calls: FunctionCall) -> None:
        # the next response the session asked for: these calls and no message
        while not model.active_session._reply_futs:
            await asyncio.sleep(0)
        messages, functions = utils.aio.Chan[MessageGeneration](), utils.aio.Chan[FunctionCall]()
        for c in calls:
            functions.send_nowait(c)
        messages.close()
        functions.close()
        event = GenerationCreatedEvent(
            message_stream=messages, function_stream=functions, user_initiated=True
        )
        model.active_session._reply_futs.pop(0).set_result(event)

    async with LiveSession(llm=model) as session:
        delegate = expert(handler, announce=False)
        await session.start(Agent(instructions="front desk", tools=[delegate]))
        session.generate_reply()
        await reply(FunctionCall(call_id="c1", name="ask", arguments='{"request": "book"}'))
        await reply()  # the question, in the model's words
        session.generate_reply(user_input=EMAIL)
        # the model's wording of the call is replaced by what the person said
        await reply(FunctionCall(call_id="c2", name="ask", arguments='{"request": "an email"}'))
        while len(model_of(specialist).requests) < 3:
            await asyncio.sleep(0.01)

    (asked, _), (answered, text) = turns
    assert answered == asked and text == EMAIL


# one question per task
# ---------------------------------------------------------------------------------------------


@pytest.mark.rule("R45", "B19")
async def test_the_questions_of_one_step_are_one_and_the_model_reads_the_answer(
    door: Callable[..., Door],
) -> None:
    agent = asking(
        Reply(tool_calls=[call("book", "b1"), call("book", "b2"), call("pick_day", "d1")]),
        Reply("Both booked.", when=said(EMAIL)),
    )

    @agent.tool
    async def pick_day(ctx: RunContext) -> InputRequired:
        """Pick the day."""
        return InputRequired("Which day?")

    desk = door(runs(agent))
    thread = await desk.thread()
    asked = await desk.send("book two visits", context_id=thread, **DELEGATION)
    # each question once, in call order, flagged only if all are
    assert asked.status_text == f"{QUESTION}\n\nWhich day?"
    [part] = asked.task["status"]["message"]["parts"]
    assert metadata_from_wire(part.get("metadata")).get(VERBATIM) is None

    done = await desk.send(EMAIL, context_id=thread, task_id=asked.task["id"])
    assert done.answer == "Both booked."
    outputs = [
        o.call_id for o in model_of(agent).requests[-1].items if o.type == "function_call_output"
    ]
    assert outputs == ["b1", "b2", "d1"]


# questions of different tasks on one lane
# ---------------------------------------------------------------------------------------------


@pytest.mark.rule("R53", "B20")
async def test_a_read_back_is_answered_only_by_a_message_of_its_task(
    door: Callable[..., Door],
) -> None:
    ran: list[str] = []
    agent = Agent(
        instructions="front desk",
        llm=ScriptedLLM(
            Reply(tool_calls=[call("cancel", "c1", date="monday")]),
            # another task's "yes": the same call there is not approved by it
            Reply(tool_calls=[call("cancel", "c2", date="monday")]),
            Reply(tool_calls=[call("cancel", "c3", date="monday")]),
            Reply("Canceled.", when=output_of("c3")),
        ),
    )

    @agent.tool(requires_approval=True)
    async def cancel(ctx: RunContext, date: str) -> str:
        """Cancel the visit on a date."""
        ran.append(date)
        return f"canceled {date}"

    desk = door(runs(agent))
    thread = await desk.thread()
    first = await desk.send("cancel monday", context_id=thread)
    other = await desk.send("yes", context_id=thread, **DELEGATION)
    assert other.state == "input_required" and ran == []

    done = await desk.send("yes", context_id=thread, task_id=first.task["id"])
    assert done.answer == "Canceled." and ran == ["monday"]


@pytest.mark.rule("R31", "R34", "R54")
async def test_a_continued_task_does_not_read_what_other_open_tasks_added(
    door: Callable[..., Door],
) -> None:
    agent = asking(
        Reply(tool_calls=[call("book", "b1")], when=said("book me in")),
        Reply(tool_calls=[call("book", "b2")], when=said("book another")),
        Reply("Booked.", when=said(EMAIL)),
        Reply("Nothing else is open.", when=said("anything else?")),
    )
    desk = door(runs(agent))
    thread = await desk.thread()
    booking = await desk.send("book me in", context_id=thread)
    # the waiting task holds no lane, and a delegation with no task id is a task of its own
    other = await desk.send("book another", context_id=thread, **DELEGATION)
    assert other.task["id"] != booking.task["id"] and other.status_text == QUESTION

    await desk.send(EMAIL, context_id=thread, task_id=booking.task["id"])
    assert user_texts(model_of(agent).requests[-1]) == ["book me in", EMAIL]
    # a new task reads everything the lane holds
    await desk.send("anything else?", context_id=thread, **DELEGATION)
    assert "book another" in user_texts(model_of(agent).requests[-1])
