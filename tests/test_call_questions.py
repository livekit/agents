"""Questions on a call (design §4, §7 "Questions"): the person's answer goes to the tool that
asked, a follow-up of the answered task goes first, and questions belong to the lane whichever
door asked. Seen in what reaches the models, in what the call says and on the A2A wire."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any

import pytest

from livekit.agents import Agent, InputRequired, LiveSession, RunContext
from livekit.agents.a2a import A2ASessionContext
from livekit.agents.delegation import A2ADelegate, InputRequiredCheck
from livekit.agents.llm import InputSpeechStartedEvent
from livekit.agents.llm.tool_context import SAY_EXACTLY
from livekit.agents.storage import AgentDBStorage
from livekit.agents.storage.resolve import Thread
from livekit.agents.voice.audio_recognition import (
    _EndOfTurnInfo,
    _EndOfTurnMetrics,
    _PreemptiveGenerationInfo,
)

from .a2a_driver import AGENT, Door, Reply, ScriptedLLM, call, no_project_env  # noqa: F401
from .fake_io import FakeAudioOutput
from .fake_realtime import ScriptedRealtimeModel
from .test_call_lane import held
from .test_call_progress import answering, instructed, outputs, slow_expert, soon, spoken, told
from .test_live_inbox import said

pytestmark = [pytest.mark.unit]


def asking(*questions: str, answer: str, heard: list[tuple[str, str]]) -> Callable[..., Any]:
    """An expert's handler whose agent asks ``questions`` in turn, as written, then answers
    ``answer``; ``heard`` keeps each message it got with its task id."""
    asked = iter(questions)
    replies = [Reply(tool_calls=[call("check", f"k{n}")]) for n in range(len(questions))]
    agent = Agent(instructions="expert", llm=ScriptedLLM(*replies, Reply(answer)))

    @agent.tool
    async def check(ctx: RunContext) -> InputRequired:
        """Check what the expert needs."""
        return InputRequired(next(asked), verbatim=True)

    async def handler(ctx: A2ASessionContext[Any]) -> Any:
        heard.append((ctx.task_id, ctx.message.text_content or ""))
        result = await agent.run(ctx.message, message_history=ctx.message_history)
        return result.input_required or result.output

    return handler


RELAYS = Reply(tool_calls=[call("ask", "a2", request="tuesday")], when=said("tuesday"))
"""The model's step on the person's answer: it relays it to the delegate that asked."""


@pytest.mark.rule("R38")
async def test_the_models_call_of_the_delegate_that_asked_continues_its_task(
    expert: Callable[..., A2ADelegate],
) -> None:
    heard: list[tuple[str, str]] = []
    model = ScriptedLLM(
        Reply(tool_calls=[call("ask", "a1", request="change my visit")]),
        RELAYS,
        Reply("You are booked for Tuesday.", when=lambda c: "Booked." in str(c.items[-1])),
    )
    delegate = expert(asking("Which day?", answer="Booked.", heard=heard), announce=False)
    async with LiveSession(llm=model) as session:
        await session.start(Agent(instructions="front desk", tools=[delegate]))
        await session.run(user_input="change my visit")
        await soon(lambda: "Which day?" in spoken(session), "the question")
        assert "call ask with their answer" in str(outputs(session.history)), "the model is told"
        await session.run(user_input="tuesday")

    (asked, _), (answered, text) = heard
    assert answered == asked and text == "tuesday"
    assert not isinstance(model.tool_choices[-2], dict), "the model is not forced"
    assert spoken(session) == ["Which day?", "You are booked for Tuesday."]


@pytest.mark.rule("R38")
async def test_a_turn_the_model_does_not_relay_leaves_the_question_open(
    expert: Callable[..., A2ADelegate],
) -> None:
    heard: list[tuple[str, str]] = []
    model = ScriptedLLM(
        Reply(tool_calls=[call("ask", "a1", request="change my visit")]),
        Reply("We open at nine.", when=said("when do you open?")),
        RELAYS,
        Reply("You are booked for Tuesday.", when=lambda c: "Booked." in str(c.items[-1])),
    )
    delegate = expert(asking("Which day?", answer="Booked.", heard=heard), announce=False)
    async with LiveSession(llm=model) as session:
        await session.start(Agent(instructions="front desk", tools=[delegate]))
        await session.run(user_input="change my visit")
        await soon(lambda: "Which day?" in spoken(session), "the question")
        await session.run(user_input="when do you open?")
        await session.run(user_input="tuesday")

    (asked, _), (answered, text) = heard
    assert answered == asked and text == "tuesday"


@pytest.mark.rule("R37")
async def test_a_question_the_person_passes_over_lets_the_next_be_asked(
    expert: Callable[..., A2ADelegate],
) -> None:
    gate = asyncio.Event()
    fares_heard: list[tuple[str, str]] = []
    model = ScriptedLLM(
        Reply(
            tool_calls=[
                call("fares", "f1", request="a flight"),
                call("seats", "s1", request="a seat"),
            ]
        ),
        Reply("We open at nine.", when=said("when do you open?")),
        Reply(tool_calls=[call("fares", "f2", request="tuesday")], when=said("tuesday")),
    )
    fares = asking("Which date?", answer="Fare booked.", heard=fares_heard)
    seats_agent = asking("Which seat?", answer="Seat booked.", heard=[])

    async def seats(ctx: A2ASessionContext[Any]) -> Any:
        await gate.wait()  # its question comes second
        return await seats_agent(ctx)

    desk = Agent(
        instructions="front desk",
        tools=[
            expert(fares, "fares", tool="fares", announce=False),
            expert(seats, "seats", tool="seats", announce=False),
        ],
    )
    async with LiveSession(llm=model) as session:
        await session.start(desk)
        await session.run(user_input="book a flight and a seat")
        await soon(lambda: "Which date?" in spoken(session), "the first question")
        gate.set()
        await asyncio.sleep(0.3)
        assert "Which seat?" not in spoken(session), "one question at a time"
        await session.run(user_input="when do you open?")
        await soon(lambda: "Which seat?" in spoken(session), "the next question")
        await session.run(user_input="tuesday")
        await soon(lambda: len(fares_heard) > 1, "the answer")

    (asked, _), (answered, text) = fares_heard
    assert answered == asked and text == "tuesday", "the passed question still takes its answer"


@pytest.mark.rule("R38")
@pytest.mark.parametrize("verdict", ["first", "new", "unclear"])
async def test_the_answer_detector_picks_the_question_a_call_answers(
    expert: Callable[..., A2ADelegate], verdict: str
) -> None:
    heard: list[tuple[str, str]] = []
    picked: list[str] = []

    async def detector(check: InputRequiredCheck) -> Any:
        picked.append(check.questions[0].ref)
        return check.questions[0] if verdict == "first" else verdict

    model = ScriptedLLM(
        Reply(
            tool_calls=[call("ask", "a1", request="a flight"), call("ask", "a2", request="a seat")]
        ),
        Reply(tool_calls=[call("ask", "a3", request="tuesday")], when=said("tuesday")),
    )
    delegate = expert(
        asking("Which date?", "Which seat?", answer="Booked.", heard=heard),
        announce=False,
        answer_detector=detector,
    )
    async with LiveSession(llm=model) as session:
        await session.start(Agent(instructions="front desk", tools=[delegate]))
        await session.run(user_input="book a flight and a seat")
        await soon(lambda: len([s for s in spoken(session) if s.endswith("?")]) == 1, "a question")
        await session.run(user_input="hmm")  # passes over it: the other is asked
        await soon(lambda: len([s for s in spoken(session) if s.endswith("?")]) == 2, "the next")
        await session.run(user_input="tuesday")
        await asyncio.sleep(0.3)

    if verdict == "unclear":
        assert len(heard) == 2, "nothing sent"
        assert any("could answer more than one" in o for o in outputs(session.history))
    else:
        task, text = heard[-1]
        assert text == "tuesday"
        assert task == picked[0] if verdict == "first" else task not in {t for t, _ in heard[:2]}


@pytest.mark.rule("R40")
async def test_the_expert_reads_the_persons_words_in_the_thread(
    door: Callable[..., Door], expert: Callable[..., A2ADelegate]
) -> None:
    read: list[str] = []

    async def reads(ctx: A2ASessionContext[Any]) -> str:
        history = await ctx.thread.chat_history(ctx.caller_lane)
        read.append([i.text_content for i in history.items if i.type == "message"][-1] or "")
        return "Noted."

    model = ScriptedLLM(Reply(tool_calls=[call("ask", "a1", request="a visit on tuesday")]))
    desk = door()
    thread = await desk.thread()
    lane = Thread(desk.storage, thread, agent_name=AGENT)
    async with LiveSession(llm=model, thread=lane) as session:
        await session.start(Agent(instructions="front desk", tools=[expert(reads, announce=False)]))
        await session.run(user_input="book me for tuesday")
        await soon(lambda: read, "the expert's turn")

    assert read == ["book me for tuesday"]


@pytest.mark.rule("B99")
async def test_a_spoken_answer_on_a_pipeline_call_is_one_turn(
    expert: Callable[..., A2ADelegate],
) -> None:
    """The STT's early transcript and the committed turn are one answer."""
    heard: list[tuple[str, str]] = []
    model = ScriptedLLM(
        Reply(tool_calls=[call("ask", "a1", request="change my visit")]),
        RELAYS,
        Reply("You are booked for Tuesday.", when=lambda c: "Booked." in str(c.items[-1])),
    )
    delegate = expert(asking("Which day?", answer="Booked.", heard=heard), announce=False)
    async with LiveSession(llm=model) as session:
        await session.start(Agent(instructions="front desk", tools=[delegate]))
        await session.run(user_input="change my visit")
        await soon(lambda: "Which day?" in spoken(session), "the question")
        activity = session._activity
        assert activity is not None
        # what the STT's audio recognition calls: an early transcript, then the turn's end
        activity.on_preemptive_generation(_PreemptiveGenerationInfo("tuesday", 1.0, time.time()))
        activity.on_end_of_turn(
            _EndOfTurnInfo(False, "tuesday", 1.0, _EndOfTurnMetrics(None, None, None, None))
        )
        await soon(lambda: "You are booked for Tuesday." in spoken(session), "the result")
        await asyncio.sleep(0.2)

    (asked, _), (answered, text) = heard
    assert answered == asked and text == "tuesday"
    users = [i for i in session.history.items if i.type == "message" and i.role == "user"]
    assert [i.text_content for i in users].count("tuesday") == 1


@pytest.mark.rule("R37")
@pytest.mark.parametrize("progress", [False, True], ids=["at-once", "after-progress"])
async def test_a_follow_up_of_the_answered_task_goes_before_the_queue(
    expert: Callable[..., A2ADelegate], progress: bool
) -> None:
    """Then its result is told before the queued question, also when progress lets the
    answer's call run on in the background."""
    gate = asyncio.Event()
    fares_heard: list[tuple[str, str]] = []
    seats_heard: list[tuple[str, str]] = []
    fares_agent = asking(
        "Which date?", "Morning or evening?", answer="Fare booked.", heard=fares_heard
    )

    async def fares(ctx: A2ASessionContext[Any]) -> Any:
        if progress and fares_heard:
            await ctx.update("Checking fares.")
            await asyncio.sleep(0.3)
        return await fares_agent(ctx)

    seats_agent = asking("Which seat?", answer="Seat booked.", heard=seats_heard)

    async def seats(ctx: A2ASessionContext[Any]) -> Any:
        if not seats_heard:
            await ctx.update("Looking at the seat map.")
            await gate.wait()
        return await seats_agent(ctx)

    model = ScriptedLLM(
        Reply(
            tool_calls=[
                call("fares", "f1", request="a flight"),
                call("seats", "s1", request="a seat"),
            ]
        ),
        Reply(tool_calls=[call("fares", "f2", request="tuesday")], when=said("tuesday")),
        Reply(tool_calls=[call("fares", "f3", request="morning")], when=said("morning")),
        Reply("Your fare is booked.", when=lambda c: "Fare booked." in str(c.items)),
    )
    desk = Agent(
        instructions="front desk",
        tools=[
            expert(fares, "fares", tool="fares", announce=False),
            expert(seats, "seats", tool="seats", announce=False, streamed=True),
        ],
    )
    async with LiveSession(llm=model) as session:
        await session.start(desk)
        await session.run(user_input="book a flight and a seat")
        await soon(lambda: "Which date?" in spoken(session), "the first question")
        gate.set()
        await soon(lambda: seats_heard, "the seats expert")
        await asyncio.sleep(0.5)  # its question waits behind the one in front
        await session.run(user_input="tuesday")
        await session.run(user_input="morning")
        await soon(lambda: "Which seat?" in spoken(session), "the queued question")

    assert spoken(session)[-4:] == [
        "Which date?",
        "Morning or evening?",
        "Your fare is booked.",
        "Which seat?",
    ]


@pytest.mark.rule("R42")
async def test_a_question_asked_by_text_is_answered_on_a_call(
    door: Callable[..., Door], expert: Callable[..., A2ADelegate]
) -> None:
    heard: list[tuple[str, str]] = []
    delegate = expert(asking("Which day?", answer="Booked.", heard=heard), announce=False)
    texting = Agent(
        instructions="front desk",
        llm=ScriptedLLM(Reply(tool_calls=[call("ask", "a1", request="change my visit")])),
        tools=[delegate],
    )

    async def handler(ctx: A2ASessionContext[Any]) -> Any:
        return await texting.run(ctx.message, message_history=ctx.message_history)

    desk = door(handler)
    asked = await desk.send("change my visit")
    thread = asked.task["contextId"]
    model = ScriptedLLM(
        RELAYS, Reply("You are booked.", when=lambda c: "Booked." in str(c.items[-1]))
    )
    calling = Agent(instructions="front desk", tools=[delegate])
    async with LiveSession(llm=model, thread=Thread(desk.storage, thread, agent_name=AGENT)) as s:
        await s.start(calling)
        await s.run(user_input="tuesday")

    (first, _), (answered, text) = heard
    assert asked.state == "input_required"
    assert answered == first and text == "tuesday"
    assert (await desk.get_task(asked.task["id"], thread))["status"]["state"] == (
        "TASK_STATE_COMPLETED"
    )


@pytest.mark.rule("R42")
async def test_a_question_asked_on_a_call_is_answered_by_text(
    agentdb_storage: AgentDBStorage, door: Callable[..., Door], expert: Callable[..., A2ADelegate]
) -> None:
    heard: list[tuple[str, str]] = []
    expert_of = asking("Which day?", answer="Booked.", heard=heard)
    # one delegate per door: the call's end closes its own (B55)
    delegate = expert(expert_of, announce=False)
    texting = Agent(
        instructions="front desk",
        llm=ScriptedLLM(
            RELAYS, Reply("You are booked.", when=lambda c: "Booked." in str(c.items[-1]))
        ),
        tools=[delegate],
    )

    async def handler(ctx: A2ASessionContext[Any]) -> Any:
        return await texting.run(ctx.message, message_history=ctx.message_history)

    desk = door(handler)
    thread = await desk.thread()
    model = ScriptedLLM(Reply(tool_calls=[call("ask", "a1", request="change my visit")]))
    calling = Agent(instructions="front desk", tools=[expert(expert_of, announce=False)])
    async with LiveSession(llm=model, thread=Thread(desk.storage, thread, agent_name=AGENT)) as s:
        await s.start(calling)
        await s.run(user_input="change my visit")
        await soon(lambda: "Which day?" in spoken(s), "the question")

    answered = await desk.send("tuesday", context_id=thread)

    (first, _), (second, text) = heard
    assert second == first and text == "tuesday"
    assert answered.answer == "You are booked."


@pytest.mark.rule("B68")
async def test_a_text_during_a_call_answers_the_question_in_front(
    door: Callable[..., Door], expert: Callable[..., A2ADelegate]
) -> None:
    heard: list[tuple[str, str]] = []
    model = ScriptedRealtimeModel(
        Reply("", [call("ask", "a1", request="book sunday")]),
        Reply("", [call("ask", "a2", request="yes")], when=answering("yes")),
    )
    delegate = expert(asking("Sunday at 14:00?", answer="Booked.", heard=heard), announce=False)
    desk = door(inbox="live")
    thread = await desk.thread()
    async with LiveSession(llm=model, thread=held(desk, thread)) as session:
        await session.start(Agent(instructions="front desk", tools=[delegate]))
        model.active_session.hear("book sunday")
        await soon(lambda: "Sunday at 14:00?" in spoken(session), "the question")
        await asyncio.wait_for(desk.send("yes", context_id=thread), 20)

    (asked, _), (answered, text) = heard
    assert answered == asked and text == "yes"
    states = [t["status"]["state"] for t in await desk.list_tasks(thread)]
    assert "TASK_STATE_INPUT_REQUIRED" not in states


@pytest.mark.rule("R61")
async def test_a_verbatim_question_on_a_model_with_no_voice_is_asked_through_a_reply(
    expert: Callable[..., A2ADelegate],
) -> None:
    heard: list[tuple[str, str]] = []
    model = ScriptedRealtimeModel(
        Reply("", [call("ask", "a1", request="change my visit")]),
        Reply("Which day?", when=lambda c: "Which day?" in instructed(c)),
        say=False,
    )
    delegate = expert(asking("Which day?", answer="Booked.", heard=heard), announce=False)
    async with LiveSession(llm=model) as session:
        session.output.audio = FakeAudioOutput()
        await session.start(Agent(instructions="front desk", tools=[delegate]))
        model.active_session.hear("change my visit")
        await soon(lambda: "Which day?" in spoken(session), "the question")

    asking_reply = model.requests[-1]
    assert instructed(asking_reply) == SAY_EXACTLY.format(text="Which day?")
    # the realtime model held the question, once, when it was asked to say it
    held = [i for i in asking_reply.items if i.type == "function_call_output"]
    assert [o.output for o in held].count("Which day?") <= 1
    assert spoken(session).count("Which day?") == 1


@pytest.mark.rule("B95")
async def test_a_result_that_lands_while_a_question_waits_is_told_after_the_persons_turn(
    expert: Callable[..., A2ADelegate],
) -> None:
    gate, ended, heard = asyncio.Event(), list[str](), list[tuple[str, str]]()
    model = ScriptedRealtimeModel(
        Reply("", [call("slots", "s1", request="slots")], when=answering("any slots?")),
        Reply("", [call("ask", "a1", request="book sunday")], when=answering("book sunday")),
        Reply("We have 10 slots.", when=told("s1_final")),
        Reply("", [call("ask", "a2", request="yes")], when=answering("yes")),
    )
    slots = expert(slow_expert(gate, ended), "slots", tool="slots")
    booking = expert(asking("Sunday at 14:00?", answer="Booked.", heard=heard), announce=False)
    async with LiveSession(llm=model) as session:
        await session.start(Agent(instructions="front desk", tools=[slots, booking]))
        rt = model.active_session
        rt.hear("any slots?")
        await soon(lambda: [i for i in session.history.items if i.type == "function_call"], "slots")
        rt.hear("book sunday")
        await soon(lambda: "Sunday at 14:00?" in spoken(session), "the question")
        gate.set()
        await soon(lambda: "There are 10 slots." in outputs(session.history), "the result")
        await asyncio.sleep(0.3)
        assert spoken(session)[-1] == "Sunday at 14:00?"
        assert not any(told("s1_final")(r) for r in model.requests), "told over the question"
        rt.hear("yes")
        await soon(lambda: "We have 10 slots." in spoken(session), "the result told")


@pytest.mark.rule("B95")
async def test_a_call_with_no_words_since_the_question_is_a_new_request(
    expert: Callable[..., A2ADelegate],
) -> None:
    heard: list[tuple[str, str]] = []
    model = ScriptedRealtimeModel(
        Reply("", [call("ask", "a1", request="book sunday")], when=answering("book sunday")),
        Reply(
            "",
            [call("ask", "a2", request="the patient confirms yes")],
            when=answering("book sunday"),
        ),
        Reply("", [call("ask", "a3", request="yes")], when=answering("yes")),
    )
    questions = asking("Sunday at 14:00?", "Sunday at 14:00?", answer="Booked.", heard=heard)
    async with LiveSession(llm=model) as session:
        await session.start(
            Agent(instructions="front desk", tools=[expert(questions, announce=False)])
        )
        rt = model.active_session
        rt.hear("book sunday")
        await soon(lambda: "Sunday at 14:00?" in spoken(session), "the question")
        rt.noise()
        await soon(lambda: len(heard) > 1, "the model's call")
        rt.hear("yes")
        await soon(lambda: heard[-1][1] == "yes", "the answer")

    first = heard[0][0]
    assert [text for task, text in heard if task == first] == ["book sunday", "yes"]


@pytest.mark.rule("B95")
async def test_an_empty_transcript_after_a_question_is_nothing_said_at_once(
    expert: Callable[..., A2ADelegate],
) -> None:
    heard: list[tuple[str, str]] = []
    model = ScriptedRealtimeModel(
        Reply("", [call("ask", "a1", request="book sunday")], when=answering("book sunday")),
        Reply("", [call("ask", "a2", request="the patient confirms yes")]),
    )
    questions = asking("Sunday at 14:00?", "Sunday at 14:00?", answer="Booked.", heard=heard)
    async with LiveSession(llm=model) as session:
        await session.start(
            Agent(instructions="front desk", tools=[expert(questions, announce=False)])
        )
        rt = model.active_session
        rt.hear("book sunday")
        await soon(lambda: "Sunday at 14:00?" in spoken(session), "the question")
        rt.hear("")  # noise the provider transcribed as nothing
        await asyncio.wait_for(soon(lambda: len(heard) > 1, "the model's call"), 1.0)

    assert heard[1][1] == "the patient confirms yes"


@pytest.mark.rule("B98")
async def test_a_transcript_of_a_turn_before_the_question_does_not_answer_it(
    expert: Callable[..., A2ADelegate],
) -> None:
    heard: list[tuple[str, str]] = []
    model = ScriptedRealtimeModel(
        Reply("", [call("ask", "a1", request="book sunday")], when=answering("book sunday")),
        Reply("", [call("ask", "a2", request="yes")], when=answering("yes")),
    )
    delegate = expert(asking("Sunday at 14:00?", answer="Booked.", heard=heard), announce=False)
    async with LiveSession(llm=model) as session:
        await session.start(Agent(instructions="front desk", tools=[delegate]))
        rt = model.active_session
        before = time.time()
        rt.hear("book sunday")
        await soon(lambda: "Sunday at 14:00?" in spoken(session), "the question")
        rt.hear("ok, let me tell you", started_at=before)  # a turn from before the question
        await asyncio.sleep(0.2)
        rt.hear("yes")
        await soon(lambda: len(heard) > 1, "the answer")

    (asked, _), (answered, text) = heard
    assert answered == asked and text == "yes"
    assert "ok, let me tell you" not in outputs(session.history)


@pytest.mark.rule("B98")
async def test_a_question_cut_before_any_of_it_played_is_asked_again(
    expert: Callable[..., A2ADelegate],
) -> None:
    heard: list[tuple[str, str]] = []
    model = ScriptedRealtimeModel(
        Reply("", [call("ask", "a1", request="book sunday")], when=answering("book sunday")),
        Reply("", hold=asyncio.Event(), when=lambda c: "Sunday at 14:00?" in instructed(c)),
        Reply("Sunday at 14:00?", when=lambda c: "Sunday at 14:00?" in instructed(c)),
        Reply("", [call("ask", "a2", request="yes")], when=answering("yes")),
        say=False,
    )
    delegate = expert(asking("Sunday at 14:00?", answer="Booked.", heard=heard), announce=False)
    async with LiveSession(llm=model) as session:
        session.output.audio = FakeAudioOutput()
        await session.start(Agent(instructions="front desk", tools=[delegate]))
        rt = model.active_session
        rt.hear("book sunday")
        await soon(lambda: any("Sunday" in instructed(r) for r in model.requests), "the asking")
        cut = time.time()
        rt.emit("input_speech_started", InputSpeechStartedEvent())  # before any of it played
        await soon(lambda: "Sunday at 14:00?" in spoken(session), "the question asked again")
        rt.hear("hold on", started_at=cut)  # the cutting turn's transcript, late
        rt.hear("yes")
        await soon(lambda: len(heard) > 1, "the answer")

    (asked, _), (answered, text) = heard
    assert answered == asked and text == "yes"
