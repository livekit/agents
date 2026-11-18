"""Agents that delegate: a child agent called as a tool during a call, and a remote A2A agent
called as a tool."""

from __future__ import annotations

import asyncio
import json

import pytest
from pydantic import BaseModel

from livekit.agents import (
    Agent,
    APIConnectOptions,
    Completed,
    DelegationOptions,
    InputRequired,
    RunContext,
    VoiceSession,
    VoiceTask,
)
from livekit.agents.a2a import RemoteAgent
from livekit.agents.llm import ChatContext, FunctionToolCall
from livekit.agents.storage import ContextStore
from tests.fakes import (
    FakeAudioInput,
    FakeLLM,
    FakeSTT,
    FakeVAD,
    Hang,
    last_input,
    tool_call,
    until,
)

from .a2a_peer import Peer


class Booking(BaseModel):
    reference: str


class Request(BaseModel):
    nights: int


def _results(session: VoiceSession) -> list[object]:
    """The JSON tool results on the call, past the in-progress updates."""
    outputs = [i.output for i in session.history.items if i.type == "function_call_output"]
    return [json.loads(o) for o in outputs if o.startswith("{")]


# Catches: a child agent's accepted output lost on a call, or not handed back as the tool result.
async def test_on_a_call_a_child_agents_typed_result_reaches_the_call() -> None:
    child = Agent(
        name="booking",
        instructions="Book the room.",
        output_type=Booking,
        llm=FakeLLM([tool_call("final_result", reference="B-9")]),
    )
    desk = Agent(instructions="Delegate.", name="desk", tools=[child.as_tool(parameters=Request)])
    delegate = tool_call("booking", nights=3)
    model = FakeLLM(lambda ctx: delegate if last_input(ctx) == "Three nights" else "Okay.")
    async with VoiceSession(
        llm=model, vad=None, turn_handling={"turn_detection": "manual"}
    ) as session:
        await session.start(desk)
        await session.generate_reply(user_input="Three nights")
        await until(lambda: _results(session), "the child's result never reached the call")

        assert _results(session) == [{"reference": "B-9"}]


# Catches: the person's answer on a call not forwarded to the child waiting on it.
async def test_on_a_call_the_persons_answer_resumes_the_waiting_child() -> None:
    child_model = FakeLLM([tool_call("ask_date"), tool_call("final_result", reference="B-6")])
    child = Agent(
        name="booking",
        instructions="Book the room.",
        output_type=Booking,
        llm=child_model,
    )

    @child.tool
    async def ask_date(ctx: RunContext) -> InputRequired:
        """Ask for the date."""
        return InputRequired("For which date?", verbatim=True)

    desk = Agent(instructions="Delegate.", name="desk", tools=[child.as_tool(parameters=Request)])
    delegate = tool_call("booking", nights=1)
    model = FakeLLM(lambda ctx: delegate if last_input(ctx) == "One night" else "Okay.")
    session = VoiceSession(
        vad=FakeVAD(),
        stt=FakeSTT(["One night", "Friday"]),
        llm=model,
        turn_handling={"turn_detection": None, "endpointing": {"min_delay": 0.1}},
        aec_warmup_duration=None,
    )
    microphone = FakeAudioInput()
    session.input.audio = microphone
    async with session:
        await session.start(desk)
        microphone.speak(0.4)
        await until(
            lambda: any(
                i.type == "message"
                and i.role == "assistant"
                and i.text_content == "For which date?"
                for i in session.history.items
            ),
            "the child's question was never asked",
        )
        microphone.speak(0.4)
        await until(lambda: _results(session), "the child's result never reached the call")

        assert _results(session) == [{"reference": "B-6"}]
    answer = child_model.requests[1].items[-1]
    assert answer.type == "message" and answer.text_content == "Friday"


# Catches: a failed remote task crashing the run instead of being the tool's error outcome.
async def test_a_remote_agents_failure_is_the_calling_tools_outcome(peer: Peer) -> None:
    peer.fail = "coverage unavailable"
    remote = RemoteAgent(
        peer.url,
        name="coverage_expert",
        description="Checks account coverage.",
        conn_options=APIConnectOptions(max_retry=0),
    )
    agent = Agent(
        instructions="Ask coverage_expert.",
        tools=[remote.as_tool()],
        llm=FakeLLM([tool_call("coverage_expert", request="Check"), "Coverage is down."]),
    )

    result = await agent.run("Check")

    assert result.status == "completed" and result.output == "Coverage is down."
    output = next(i for i in result.new_messages() if i.type == "function_call_output")
    assert output.is_error and output.output == "coverage unavailable"


@pytest.mark.parametrize("bound", [False, True])
@pytest.mark.parametrize("change_agent", [False, True])
async def test_a_voice_side_question_leaves_the_remote_input_open_until_the_person_answers(
    peer: Peer,
    bound: bool,
    change_agent: bool,
) -> None:
    peer.question = "Which date?"
    remote = RemoteAgent(
        peer.url,
        name="coverage_expert",
        description="Checks account coverage.",
        conn_options=APIConnectOptions(max_retry=0),
    )

    def frontend(history: ChatContext) -> str | FunctionToolCall:
        latest = next(
            (
                item.text_content
                for item in reversed(history.items)
                if item.type == "message" and item.role == "user"
            ),
            None,
        )
        if latest == "Why do you need the date?":
            return "To check which coverage applies."
        if latest == "Friday":
            if any(
                item.type == "function_call_output" and "Confirmed" in item.output
                for item in history.items
            ):
                return "Confirmed."
            return "I'll pass that on."
        if any(
            item.type == "message"
            and item.role != "user"
            and "Which date?" in (item.text_content or "")
            for item in history.items
        ):
            return "Which date?"
        if any(item.type == "function_call" for item in history.items):
            return "I'm checking."
        return tool_call("coverage_expert", request="Check coverage")

    store = ContextStore()
    agent = Agent(
        name="desk", instructions="Help.", tools=[remote.as_tool()], llm=FakeLLM(frontend)
    )
    root_entered = asyncio.Event()

    @agent.on_enter
    async def entered_root(session: VoiceSession) -> None:
        root_entered.set()

    def said(session: VoiceSession, text: str) -> bool:
        return any(
            item.type == "message" and item.role == "assistant" and item.text_content == text
            for item in session.history.items
        )

    microphone = FakeAudioInput()
    async with VoiceSession(
        context=store.bind() if bound else None,
        delegation=DelegationOptions(
            routing_model=FakeLLM(
                [
                    tool_call("route_input_response", candidates=[], unclear=False),
                    tool_call("route_input_response", candidates=[0], unclear=False),
                ]
            )
        ),
        vad=FakeVAD(),
        stt=FakeSTT(["Check coverage", "Why do you need the date?", "Friday"]),
        turn_handling={"turn_detection": None, "endpointing": {"min_delay": 0.1}},
        aec_warmup_duration=None,
    ) as session:
        session.input.audio = microphone
        await session.start(agent)
        microphone.speak(0.4)
        await until(lambda: said(session, "Which date?"), "the remote question was not presented")

        microphone.speak(0.4)
        await until(
            lambda: said(session, "To check which coverage applies."),
            "the frontend did not answer the side question",
        )
        assert len(peer.messages) == 1, "a side question must not answer the remote task"
        if bound:
            state = await store.load_session()
            task = state.remote_agents[peer.url].tasks["remote-task"]
            assert [request.state for request in task.input_requests.values()] == ["open"]

        if change_agent:
            helper_entered = asyncio.Event()
            helper = Agent(
                name="helper",
                instructions="Help.",
                llm=FakeLLM(["I'll keep that for the travel desk."]),
            )

            @helper.on_enter
            async def entered_helper(session: VoiceSession) -> None:
                helper_entered.set()

            root_entered.clear()
            session.update_agent(helper)
            await until(helper_entered.is_set, "the agent without the remote tool never took over")

        microphone.speak(0.4)
        if change_agent:
            await until(
                lambda: said(session, "I'll keep that for the travel desk."),
                "the current agent did not handle the whole response",
            )
            assert len(peer.messages) == 1, "a response cannot dispatch through an ineligible agent"
            if bound:
                state = await store.load_session()
                task = state.remote_agents[peer.url].tasks["remote-task"]
                assert [request.state for request in task.input_requests.values()] == ["responding"]
            session.update_agent(agent)
            await until(root_entered.is_set, "the eligible agent never returned")
        await until(lambda: said(session, "Confirmed."), "the remote result was not presented")
        assert len(peer.messages) == 2
        assert peer.messages[1]["taskId"] == "remote-task"
        assert peer.messages[1]["parts"] == [{"text": "Friday"}]


async def test_interrupting_voice_input_routing_keeps_the_remote_request_open(peer: Peer) -> None:
    peer.question = "Which date?"
    detector = Hang()

    def reply(history: ChatContext) -> str | FunctionToolCall:
        if any(
            item.type == "message"
            and item.role != "user"
            and "Which date?" in (item.text_content or "")
            for item in history.items
        ):
            return "Which date?"
        if any(item.type == "function_call" for item in history.items):
            return "I'm checking."
        return tool_call("travel", request="Book a flight")

    frontend = FakeLLM(reply)
    agent = Agent(
        name="desk",
        instructions="Help.",
        llm=frontend,
        tools=[RemoteAgent(peer.url, name="travel").as_tool()],
    )
    store = ContextStore()
    async with VoiceSession(
        context=store.bind(),
        vad=None,
        delegation=DelegationOptions(routing_model=FakeLLM([detector])),
    ) as session:
        await session.start(agent)
        session.generate_reply(user_input="Book a flight")
        await until(
            lambda: any(
                item.type == "message"
                and item.role == "assistant"
                and item.text_content == "Which date?"
                for item in session.history.items
            ),
            "the question was not presented",
        )
        requests_before_response = len(frontend.requests)
        speech = session.generate_reply(user_input="Friday")
        await until(detector.started.is_set, "the input detector never started")
        await session.interrupt()
        await speech
        await until(detector.ended.is_set, "interruption left the detector stream open")
        assert len(frontend.requests) == requests_before_response, (
            "a canceled detector must not start a frontend reply"
        )
        assert len(peer.messages) == 1, "the canceled verdict cannot authorize remote delivery"
        state = await store.load_session()
        task = state.remote_agents[peer.url].tasks["remote-task"]
        assert [request.state for request in task.input_requests.values()] == ["open"]


async def test_a_voice_task_keeps_its_answer_out_of_an_earlier_remote_question(peer: Peer) -> None:
    """A foreground collector owns the person's answer, even when an expert also needs a date."""
    peer.question = "Which date?"
    collector = Agent(
        name="callback",
        instructions="Collect a callback date.",
        output_type=str,
        llm=FakeLLM(["When should I call you?", tool_call("finish", date="Friday")]),
    )

    @collector.tool
    async def finish(ctx: RunContext, date: str) -> Completed[str]:
        """Complete the callback arrangement."""
        return Completed(date)

    def frontend(history: ChatContext) -> str | FunctionToolCall:
        human = next(
            item.text_content
            for item in reversed(history.items)
            if item.type == "message" and item.role == "user"
        )
        if any(
            item.type == "function_call_output"
            and item.name == "arrange_callback"
            and "Friday" in item.output
            for item in history.items
        ):
            return "I'll call on Friday."
        if human == "Call me":
            return tool_call("arrange_callback")
        if any(
            item.type == "message" and "Which date?" in (item.text_content or "")
            for item in history.items
        ):
            return "Which date?"
        if any(item.type == "function_call" for item in history.items):
            return "I'm checking."
        return tool_call("travel", request="Book a flight")

    agent = Agent(
        name="desk",
        instructions="Help.",
        llm=FakeLLM(frontend),
        tools=[RemoteAgent(peer.url, name="travel").as_tool()],
    )

    @agent.tool
    async def arrange_callback(ctx: RunContext) -> str:
        """Collect a convenient callback date."""
        return await VoiceTask(agent=collector).run()

    store = ContextStore()
    routing = FakeLLM([tool_call("route_input_response", candidates=[], unclear=False)])
    async with VoiceSession(
        context=store.bind(), vad=None, delegation=DelegationOptions(routing_model=routing)
    ) as session:
        await session.start(agent)

        def said(text: str) -> bool:
            return any(
                item.type == "message" and item.role == "assistant" and item.text_content == text
                for item in session.history.items
            )

        session.generate_reply(user_input="Book a flight")
        await until(lambda: said("Which date?"), "the remote question was not presented")
        callback = session.generate_reply(user_input="Call me")
        await until(lambda: said("When should I call you?"), "the collector did not take the floor")
        await session.generate_reply(user_input="Friday")
        await callback
        await until(lambda: said("I'll call on Friday."), "the collector's output was lost")
        state = await store.load_session()
        task = state.remote_agents[peer.url].tasks["remote-task"]
        assert [request.state for request in task.input_requests.values()] == ["open"]
        assert len(peer.messages) == 1, "the callback answer also reached the unrelated remote task"
        assert len(routing.requests) == 1, "a VoiceTask's answer must stay with that collector"
