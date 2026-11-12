"""Agents that delegate: a child agent called as a tool during a call, and a remote A2A agent
called as a tool."""

from __future__ import annotations

import json

from pydantic import BaseModel

from livekit.agents import Agent, APIConnectOptions, InputRequired, RunContext, VoiceSession
from livekit.agents.a2a import RemoteAgent
from tests.fakes import (
    FakeAudioInput,
    FakeLLM,
    FakeSTT,
    FakeVAD,
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
