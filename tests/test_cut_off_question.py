"""A question cut off by the person's message (docs/delegation.md): that message is its answer,
and the expert gets it with what the person had of the question. A message the front desk
already handled is not the answer to a question that arrives after it."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from livekit.agents import Agent, AgentServer, InputRequired, LiveSession, RunContext
from livekit.agents.a2a import A2ASessionContext
from livekit.agents.a2a.extension import HEARD, widen
from livekit.agents.delegation import A2ADelegate
from livekit.agents.delegation.a2a import CUT_OFF
from livekit.agents.llm import ChatMessage, FunctionCallOutput, FunctionToolCall
from livekit.agents.llm.tool_context import ASK_IN_OWN_WORDS, HEARD_KEY
from livekit.agents.storage import AgentDBStorage
from livekit.agents.storage.resolve import Thread
from livekit.agents.voice.agent_run import CUT_OFF_QUESTION
from livekit.agents.voice.speech_handle import SpeechHandle

from .fake_llm import FakeLLMResponse
from .test_a2a_server import (  # noqa: F401  (no_project_env: autouse)
    AGENT,
    ENDPOINT,
    http_client,
    make_server,
    no_project_env,
    state_of,
    wire,
)
from .test_agent_api import RecordingLLM
from .test_live_inbox import stream, tasks_of, until

pytestmark = [pytest.mark.unit]

EXPERT = "eligibility"


def _call(name: str, call_id: str, **arguments: str) -> FunctionToolCall:
    return FunctionToolCall(name=name, arguments=json.dumps(arguments), call_id=call_id)


def _reply(text: str, content: str = "", *calls: FunctionToolCall) -> FakeLLMResponse:
    return FakeLLMResponse(input=text, content=content, ttft=0, duration=0, tool_calls=list(calls))


async def _waiting(storage: AgentDBStorage, thread_id: str, count: int) -> None:
    """Until the front desk's lane holds ``count`` tasks: the later messages are in its inbox."""

    async def landed() -> bool:
        return len(await tasks_of(storage, thread_id)) >= count

    await until(landed)


async def test_a_question_cut_off_by_a_waiting_message_takes_it_as_its_answer(
    agentdb_workers: tuple[AgentDBStorage, AgentDBStorage],
) -> None:
    """The back office's question comes back while the person's next message waits: the front
    desk does not drop it. That message continues the one expert task, as the person said it,
    with ``heard`` empty in its metadata and in a line the expert's model reads. On the front
    desk's lane the task that asked ends merged into the message's task, which answers."""
    clinic_storage, expert_storage = agentdb_workers
    go = asyncio.Event()
    cut_off_line = CUT_OFF.format(heard="none")
    expert_llm = RecordingLLM(
        fake_responses=[
            _reply("The person wants the first slot.", "", _call("book", "e1", slot="first")),
            _reply(f"{cut_off_line}\nyeah maybe the second", "The second slot then."),
        ]
    )
    expert = Agent(instructions="back office", llm=expert_llm)

    @expert.tool
    async def book(ctx: RunContext, slot: str) -> InputRequired:
        """Book a slot.

        Args:
            slot: the slot
        """
        await go.wait()
        return InputRequired("What email address?")

    seen: list[Any] = []

    async def expert_door(ctx: A2ASessionContext[Any]) -> Any:
        seen.append(ctx.messages[-1].metadata)
        return await expert.run(ctx.message, message_history=ctx.message_history)

    expert_server = AgentServer(storage=expert_storage, agent_name=EXPERT)
    expert_server.a2a_session(endpoint=ENDPOINT, description="Bookings.")(expert_door)
    async with http_client(expert_server) as expert_http:
        delegate = A2ADelegate(f"http://agents.test/{ENDPOINT}", httpx_client=expert_http)
        front = Agent(
            instructions="front desk",
            tools=[delegate],
            llm=RecordingLLM(
                fake_responses=[
                    _reply(
                        "the first one",
                        "",
                        _call("ask", "c1", request="The person wants the first slot."),
                    ),
                    _reply("The second slot then.", "Sure, the second one."),
                ]
            ),
        )

        async def desk(ctx: A2ASessionContext[Any]) -> Any:
            return await front.run(ctx.message, message_history=ctx.message_history)

        clinic, _ = make_server(clinic_storage, desk, inbox="live")
        thread_id = (await clinic_storage.threads.create_thread()).thread_id
        async with http_client(clinic) as http:
            first = asyncio.create_task(
                stream(http, wire("the first one", "m1", context_id=thread_id))
            )
            await until(lambda: asyncio.sleep(0, result=bool(seen)))
            second = asyncio.create_task(
                stream(http, wire("yeah maybe the second", "m2", context_id=thread_id))
            )
            await _waiting(clinic_storage, thread_id, 2)
            go.set()
            frames = await asyncio.wait_for(asyncio.gather(first, second), 30)
        await delegate.aclose()

    asked_id, answer_id = (f[0]["task"]["id"] for f in frames)
    tasks = await tasks_of(clinic_storage, thread_id)
    assert tasks[asked_id].status == "completed"
    assert tasks[asked_id].data.metadata.get("livekit/mergedInto") == answer_id
    assert tasks[answer_id].status == "completed"
    assert state_of(frames[1][-1]) == "TASK_STATE_COMPLETED"
    # one expert task, continued by the message as the person said it
    expert_tasks = {
        t.id: t for t in await Thread(expert_storage, thread_id, agent_name=EXPERT).tasks()
    }
    assert len(expert_tasks) == 1
    assert [t.status for t in expert_tasks.values()] == ["completed"]
    assert seen[-1].get(widen(HEARD), seen[-1].get(HEARD)) == ""
    said = [i.text_content for i in expert_llm.requests[-1].items if i.type == "message"]
    assert said[-1] == f"{cut_off_line}\nyeah maybe the second"
    # the front desk never showed the question; the call that answered names what was heard
    items = (await Thread(clinic_storage, thread_id, agent_name=AGENT).chat_history()).items
    replies = [i.text_content for i in items if i.type == "message" and i.role == "assistant"]
    assert replies == ["Sure, the second one."]
    answer = [i for i in items if i.type == "function_call" and HEARD_KEY in i.extra]
    assert len(answer) == 1 and answer[0].extra[HEARD_KEY] == ""


async def test_a_cut_off_question_no_delegate_relays_goes_to_the_model(
    agentdb_storage: AgentDBStorage,
) -> None:
    """A question of the agent's own tool, cut off by the person's message: the message goes to
    the task's model, told what the person had of the question, and no tool takes it as said."""
    go = asyncio.Event()
    started = asyncio.Event()
    cards: list[str] = []
    desk = Agent(instructions="front desk")

    @desk.tool
    async def card_number(ctx: RunContext, number: str) -> str:
        """Take the card number.

        Args:
            number: the card number
        """
        cards.append(number)
        return "card saved"

    @desk.tool
    async def pay(ctx: RunContext) -> InputRequired:
        """Pay the bill."""
        started.set()
        await go.wait()
        return InputRequired("Your card number?", verbatim=True)

    llm = RecordingLLM(
        fake_responses=[
            _reply("pay my bill", "", _call("pay", "p1")),
            _reply(CUT_OFF_QUESTION.format(heard="none"), "It is 40 dollars. Your card number?"),
        ]
    )
    desk.update_options(llm=llm)

    async def handler(ctx: A2ASessionContext[Any]) -> Any:
        return await desk.run(ctx.message, message_history=ctx.message_history)

    server, _ = make_server(agentdb_storage, handler, inbox="live")
    thread_id = (await agentdb_storage.threads.create_thread()).thread_id
    async with http_client(server) as http:
        first = asyncio.create_task(stream(http, wire("pay my bill", "m1", context_id=thread_id)))
        await started.wait()
        second = asyncio.create_task(
            stream(http, wire("how much is it?", "m2", context_id=thread_id))
        )
        await _waiting(agentdb_storage, thread_id, 2)
        go.set()
        frames = await asyncio.wait_for(asyncio.gather(first, second), 30)

    assert cards == [], "the message that cut the question off is not handed to a tool"
    request = llm.requests[-1].items
    assert [i.text_content for i in request if i.type == "message"][-2:] == [
        "how much is it?",
        CUT_OFF_QUESTION.format(heard="none"),
    ]
    assert state_of(frames[1][-1]) == "TASK_STATE_COMPLETED"


async def test_a_message_handled_before_a_late_question_is_not_its_answer(
    agentdb_storage: AgentDBStorage,
) -> None:
    """The question is still on its way when the person's next message comes, after the call
    released the floor: the front desk answers that message in its own turn, and the question
    is asked afterwards. The message after the question is its answer."""
    go = asyncio.Event()
    answers: list[str] = []
    desk = Agent(instructions="front desk")

    @desk.tool
    async def ask(ctx: RunContext, request: str) -> str | InputRequired:
        """The back office.

        Args:
            request: the request
        """
        if request.startswith("ana@"):
            answers.append(request)
            return "booked"
        await ctx.update("checking")
        await go.wait()
        return InputRequired("What email address?", verbatim=True)

    ask._relays_answer = True  # type: ignore[attr-defined]  (as a delegate's tool)

    llm = RecordingLLM(
        fake_responses=[
            _reply("book monday", "On it.", _call("ask", "a1", request="book monday")),
            _reply("what time is it?", "It is nine."),
            _reply("booked", "Booked."),
        ]
    )
    desk.update_options(llm=llm)

    async def handler(ctx: A2ASessionContext[Any]) -> Any:
        return await desk.run(ctx.message, message_history=ctx.message_history)

    server, _ = make_server(agentdb_storage, handler, inbox="live")
    thread_id = (await agentdb_storage.threads.create_thread()).thread_id
    async with http_client(server) as http:
        first = asyncio.create_task(stream(http, wire("book monday", "m1", context_id=thread_id)))

        async def backgrounded() -> bool:
            rows = (await tasks_of(agentdb_storage, thread_id)).values()
            return any(t.data.metadata.get("livekit/background") for t in rows)

        await until(backgrounded)
        handled = await stream(http, wire("what time is it?", "m2", context_id=thread_id))
        go.set()
        asked = await asyncio.wait_for(first, 30)
        answer = await stream(http, wire("ana@example.com", "m3", context_id=thread_id))

    assert state_of(handled[-1]) == "TASK_STATE_COMPLETED"
    assert state_of(asked[-1]) == "TASK_STATE_INPUT_REQUIRED"
    assert answers == ["ana@example.com"], "only the message after the question answered it"
    assert state_of(answer[-1]) == "TASK_STATE_COMPLETED"
    tasks = await tasks_of(agentdb_storage, thread_id)
    assert not any(t.data.metadata.get("livekit/mergedInto") for t in tasks.values())


def _cut_off_speech(session: LiveSession, said: str) -> None:
    """The question in front of the person was cut off after ``said``, as a barge-in leaves
    its speech: interrupted, with the part spoken as its message."""
    front = session._activity._front_question()  # type: ignore[union-attr]
    assert front is not None
    handle = SpeechHandle.create()
    handle._chat_items.append(ChatMessage(role="assistant", content=[said], interrupted=True))
    handle.interrupt(force=True)
    front.speech = handle


async def test_on_a_call_a_partly_spoken_question_goes_with_what_was_heard() -> None:
    """On a call the person speaks over the question: their words go to the delegate's tool as
    the answer, with the part of the question they heard."""
    desk = Agent(instructions="front desk")
    calls: list[tuple[str, Any]] = []

    @desk.tool
    async def ask(ctx: RunContext, request: str) -> str | InputRequired:
        """The back office.

        Args:
            request: the request, in full
        """
        calls.append((request, ctx.function_call.extra.get(HEARD_KEY)))
        if request == "the first slot":
            return InputRequired("What email address should the confirmation go to?")
        return "the second slot then"

    ask._relays_answer = True  # type: ignore[attr-defined]  (as a delegate's tool)
    question = "What email address should the confirmation go to?"
    llm = RecordingLLM(
        fake_responses=[
            _reply("the first one", "", _call("ask", "c1", request="the first slot")),
            _reply(ASK_IN_OWN_WORDS.format(question=question), "What email should I use?"),
            _reply("the second slot then", "Sure, the second one."),
        ]
    )
    async with LiveSession(llm=llm) as session:
        await session.start(desk)
        await session.run(user_input="the first one")
        _cut_off_speech(session, "What email")
        await session.run(user_input="yeah maybe the second")

    assert calls == [("the first slot", None), ("yeah maybe the second", "What email")]


async def test_on_a_call_a_cut_off_question_no_delegate_relays_goes_to_the_model() -> None:
    """The person speaks over a question of the agent's own tool: their words are a turn of the
    model, and no tool takes them as said."""
    desk = Agent(instructions="front desk")
    cards: list[str] = []

    @desk.tool
    async def card_number(ctx: RunContext, number: str) -> str:
        """Take the card number.

        Args:
            number: the card number
        """
        cards.append(number)
        return "card saved"

    @desk.tool
    async def pay(ctx: RunContext) -> InputRequired:
        """Pay the bill."""
        return InputRequired("Your card number?", verbatim=True)

    llm = RecordingLLM(
        fake_responses=[
            _reply("pay my bill", "", _call("pay", "p1")),
            _reply("how much is it?", "It is 40 dollars."),
        ]
    )
    async with LiveSession(llm=llm) as session:
        await session.start(desk)
        await session.run(user_input="pay my bill")
        _cut_off_speech(session, "Your")
        await session.run(user_input="how much is it?")

    assert cards == [], "the words that cut the question off are not the card number"
    said = [i.text_content for i in session.history.items if i.type == "message"]
    assert said[-2:] == ["how much is it?", "It is 40 dollars."]


async def test_on_a_call_a_question_back_after_the_person_spoke_over_the_turn_is_kept() -> None:
    """The call returned its question as the person spoke over the turn that made it: the
    question was never said, and the person's message is its answer, with nothing heard."""
    desk = Agent(instructions="front desk")
    calls: list[tuple[str, Any]] = []

    @desk.tool
    async def ask(ctx: RunContext, request: str) -> str:
        """The back office.

        Args:
            request: the request, in full
        """
        calls.append((request, ctx.function_call.extra.get(HEARD_KEY)))
        return "the second slot then"

    ask._relays_answer = True  # type: ignore[attr-defined]  (as a delegate's tool)
    llm = RecordingLLM(fake_responses=[_reply("the second slot then", "Sure.")])
    async with LiveSession(llm=llm) as session:
        await session.start(desk)
        output = FunctionCallOutput(
            call_id="c1",
            name="ask",
            output="waiting for the person's answer to: Which email?",
            is_error=False,
        )
        session._activity._cut_off([(InputRequired("Which email?"), output)])  # type: ignore[union-attr]
        await session.run(user_input="yeah maybe the second")

    assert calls == [("yeah maybe the second", "")]
