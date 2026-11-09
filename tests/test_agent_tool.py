"""An agent called as a tool outside a TextSession: the child runs in process as the call's
delegate, in a direct run and on a call."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

import pytest
from pydantic import BaseModel

from livekit.agents import Agent, InputRequired, RunContext, VoiceSession

from .test_text_session import Reply, Script, call

pytestmark = pytest.mark.unit


class Booking(BaseModel):
    reference: str


class Request(BaseModel):
    nights: int


@dataclass
class Case:
    booked: int = 0


def _booking(model: Script) -> Agent[Case, Booking]:
    child = Agent(
        name="booking",
        instructions="Book the room.",
        context_type=Case,
        output_type=Booking,
        llm=model,
    )

    @child.tool
    async def book(ctx: RunContext[Case], nights: int) -> str:
        """Book the room.

        Args:
            nights: how many nights
        """
        ctx.context.booked = nights
        return "booked"

    return child


async def test_a_direct_run_reads_the_childs_typed_output_and_counts_its_usage() -> None:
    child = _booking(
        Script(
            call("book", {"nights": 2}, "book-1"),
            call("final_result", {"reference": "B-42"}, "final-1"),
        )
    )
    parent_model = Script(call("booking", {"nights": 2}, "delegate"), Reply("Booked: B-42."))
    parent = Agent(
        instructions="Delegate bookings.",
        context_type=Case,
        tools=[child.as_tool(parameters=Request)],
        llm=parent_model,
    )
    case = Case()

    result = await parent.run("Two nights please", context=case)

    assert result.status == "completed" and result.output == "Booked: B-42."
    assert case.booked == 2, "the child shares the parent's context"
    outputs = [i for i in result.all_messages() if i.type == "function_call_output"]
    assert [json.loads(o.output) for o in outputs] == [{"reference": "B-42"}]
    # the child's history stays its own: the parent's model never saw the child's tool call
    seen = [i for ctx in parent_model.requests for i in ctx.items if i.type == "function_call"]
    assert {i.name for i in seen} == {"booking"}
    assert sum(model.requests for model in result.usage.models) == 4


async def test_a_waiting_child_passes_its_question_up_and_resumes_on_the_next_call() -> None:
    child_model = Script(
        Reply("For which date?"),
        call("final_result", {"reference": "B-7"}, "final-1"),
    )
    child = _booking(child_model)
    parent = Agent(
        instructions="Delegate bookings.",
        context_type=Case,
        tools=[child.as_tool(parameters=Request, chat_history="fork")],
        llm=Script(
            call("booking", {"nights": 1}, "first"),
            Reply("Which date would you like?"),
            call("booking", {"nights": 1}, "second"),
            Reply("Booked: B-7."),
        ),
    )

    first = await parent.run("One night", context=Case())
    assert first.status == "input_required"
    assert first.question == "Which date would you like?", "the parent words the child's question"

    second = await parent.run("Friday", message_history=first.all_messages(), context=Case())
    assert second.status == "completed" and second.output == "Booked: B-7."
    resumed = child_model.requests[1].items
    assert any(
        i.type == "message" and i.role == "assistant" and i.text_content == "For which date?"
        for i in resumed
    ), "the child resumed with its own history"
    calls = [i for i in second.all_messages() if i.type == "function_call"]
    asked = [i for i in first.all_messages() if i.type == "function_call"]
    assert "lk.agent_tool" in asked[0].extra, "the waiting child is kept"
    assert all("lk.agent_tool" not in c.extra for c in calls), "an answered child is not kept"


async def test_only_the_latest_waiting_child_is_kept() -> None:
    child = _booking(
        Script(Reply("For which date?"), Reply("Which room?"), Reply("Smoking or not?"))
    )
    parent = Agent(
        instructions="Delegate bookings.",
        context_type=Case,
        tools=[child.as_tool(parameters=Request)],
        llm=Script(
            call("booking", {"nights": 1}, "first"),
            Reply("Which date?"),
            call("booking", {"nights": 1}, "second"),
            Reply("Which room?"),
        ),
    )
    first = await parent.run("One night", context=Case())
    second = await parent.run("Friday", message_history=first.all_messages(), context=Case())
    assert second.status == "input_required"
    calls = [i for i in second.all_messages() if i.type == "function_call"]
    assert ["lk.agent_tool" in c.extra for c in calls] == [False, True]
    kept = json.dumps(calls[1].extra["lk.agent_tool"])
    assert "For which date?" in kept and "Which room?" in kept


async def test_cancelling_the_parent_cancels_and_settles_the_child() -> None:
    blocked = asyncio.Event()
    child = _booking(Script(Reply(blocked=blocked)))
    parent = Agent(
        instructions="Delegate bookings.",
        tools=[child.as_tool(parameters=Request)],
        llm=Script(call("booking", {"nights": 1}, "delegate")),
    )
    async with parent.run_stream("One night") as stream:
        consumer = asyncio.create_task(stream.result())
        await asyncio.wait_for(blocked.wait(), timeout=5)
        stream.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(consumer, timeout=5)


async def test_agent_tools_nest_up_to_the_depth_limit() -> None:
    # an agent calling itself is refused past the limit instead of recursing forever
    model = Script(*[call("recursive", {}, f"r{i}") for i in range(9)], *[Reply("bottom")] * 9)
    recursive = Agent(name="recursive", instructions="Recurse.", output_type=str, llm=model)
    recursive.tool(recursive.as_tool())
    result = await recursive.run("go")
    assert result.status == "completed" and result.output == "bottom"
    deepest = model.requests[9].items
    refusals = [i.output for i in deepest if i.type == "function_call_output"]
    assert refusals == ["agent-tool call exceeds max_depth=8"]


async def test_on_a_call_the_childs_result_reaches_the_call() -> None:
    from livekit.agents.llm import FunctionToolCall

    from .fake_llm import FakeLLM, FakeLLMResponse

    child = _booking(Script(call("final_result", {"reference": "B-9"}, "final-1")))
    delegate = FunctionToolCall(name="booking", arguments='{"nights": 3}', call_id="c1")
    model = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="Three nights", content="", ttft=0, duration=0, tool_calls=[delegate]
            )
        ]
    )
    desk = Agent(instructions="Delegate.", name="desk", tools=[child.as_tool(parameters=Request)])
    async with VoiceSession(
        llm=model, vad=None, context=Case(), turn_handling={"turn_detection": "manual"}
    ) as session:
        await session.start(desk)
        await session._run(user_input="Three nights")

        async def result() -> str:
            while True:
                for item in session.history.items:
                    if item.type == "function_call_output" and "B-9" in item.output:
                        return item.output
                await asyncio.sleep(0.01)

        assert "B-9" in await asyncio.wait_for(result(), 5)


async def test_on_a_call_the_persons_answer_resumes_the_waiting_child() -> None:
    from livekit.agents.llm import FunctionToolCall

    from .fake_io import FakeAudioInput, FakeAudioOutput
    from .fake_llm import FakeLLM, FakeLLMResponse
    from .fake_stt import FakeSTT, FakeUserSpeech
    from .fake_tts import FakeTTS
    from .fake_vad import FakeVAD

    child_model = Script(
        call("ask_date", {}, "ask"), call("final_result", {"reference": "B-5"}, "f")
    )
    child = _booking(child_model)

    @child.tool
    async def ask_date(ctx: RunContext[Case]) -> InputRequired:
        """Ask for the date."""
        return InputRequired("For which date?", verbatim=True)

    delegate = FunctionToolCall(name="booking", arguments='{"nights": 1}', call_id="c1")
    model = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="one night", content="", ttft=0, duration=0, tool_calls=[delegate]
            )
        ]
    )
    speech = [
        FakeUserSpeech(start_time=0.1, end_time=0.4, transcript="one night", stt_delay=0.05),
        FakeUserSpeech(start_time=2.0, end_time=2.3, transcript="friday", stt_delay=0.05),
    ]
    session = VoiceSession(
        vad=FakeVAD(fake_user_speeches=speech, min_silence_duration=0.1),
        stt=FakeSTT(fake_user_speeches=speech),
        llm=model,
        tts=FakeTTS(fake_audio_duration=0.1),
        context=Case(),
        turn_handling={"turn_detection": None, "endpointing": {"min_delay": 0.1}},
        aec_warmup_duration=None,
    )
    audio_input = FakeAudioInput()
    session.input.audio = audio_input
    session.output.audio = FakeAudioOutput()
    desk = Agent(instructions="Delegate.", name="desk", tools=[child.as_tool(parameters=Request)])
    async with session:
        await session.start(desk)
        audio_input.push(3.0)

        async def result() -> str:
            while True:
                for item in session.history.items:
                    if item.type == "function_call_output" and "B-5" in item.output:
                        return item.output
                await asyncio.sleep(0.01)

        assert "B-5" in await asyncio.wait_for(result(), 10)
    answer = child_model.requests[1].items[-1]
    assert answer.type == "message" and answer.text_content == "friday"


class _ChildModel(Script):
    @property
    def model(self) -> str:
        return "child-model"


async def test_on_a_call_the_childs_usage_counts_once_in_the_sessions() -> None:
    from livekit.agents.llm import FunctionToolCall

    from .fake_llm import FakeLLM, FakeLLMResponse

    child = _booking(_ChildModel(call("final_result", {"reference": "B-3"}, "final-1")))
    delegate = FunctionToolCall(name="booking", arguments='{"nights": 2}', call_id="c1")
    response = FakeLLMResponse(input="Two nights", content="", ttft=0, duration=0)
    model = FakeLLM(fake_responses=[response.model_copy(update={"tool_calls": [delegate]})])
    desk = Agent(instructions="Delegate.", name="desk", tools=[child.as_tool(parameters=Request)])
    async with VoiceSession(
        llm=model, vad=None, context=Case(), turn_handling={"turn_detection": "manual"}
    ) as session:
        await session.start(desk)
        await session._run(user_input="Two nights")

        async def child_requests() -> int:
            while True:
                counted = [m for m in session.usage.models if m.model == "child-model"]
                if counted:
                    return counted[0].requests
                await asyncio.sleep(0.01)

        assert await asyncio.wait_for(child_requests(), 5) == 1
        await asyncio.sleep(0.2)
        assert [m.requests for m in session.usage.models if m.model == "child-model"] == [1]
