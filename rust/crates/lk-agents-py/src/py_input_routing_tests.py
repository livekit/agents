"""Human journeys through the native owner, a separate routing model and generated state views."""

import asyncio
import json
from dataclasses import dataclass

import pytest

from livekit.agents import (
    Agent,
    APIStatusError,
    DelegationOptions,
    InputRequired,
    RunContext,
    SessionState,
    TextSession,
    persisted,
)
from livekit.agents.a2a import RemoteAgent
from livekit.agents.storage import ContextStore
from tests.fakes import FakeLLM, Hang, tool_call, until
from tests.host.a2a_peer import Peer

pytest_plugins = ["tests.host.a2a_peer"]

pytestmark = pytest.mark.unit


@dataclass
class App:
    session: SessionState = persisted(default_factory=SessionState)


def request_state(app: App) -> str:
    current = app.session.current_agent
    assert current is not None
    return next(iter(current.input_requests.values())).state


async def test_a_side_question_does_not_consume_the_date_and_a_later_response_is_routed() -> None:
    store = ContextStore()
    app = await store.load_context(App())
    frontend = FakeLLM(
        [tool_call("date"), "I need it to check availability.", "Friday is selected."]
    )
    routing = FakeLLM(
        [
            tool_call("route_input_response", candidates=[], unclear=False),
            tool_call("route_input_response", candidates=[0], unclear=False),
        ]
    )
    agent = Agent(name="desk", instructions="Help.", llm=frontend)

    @agent.tool
    async def date() -> InputRequired:
        """Ask for the booking date."""
        return InputRequired("Which date?", verbatim=True)

    delegation = DelegationOptions(routing_model=routing)
    result = await agent.run("Book a room", context=store.bind(app), delegation=delegation)
    assert result.status == "input_required" and result.question == "Which date?"
    assert routing.requests == [], "unseen questions cannot trigger classification"

    result = await TextSession(agent=agent, delegation=delegation).run(
        "Why?", context=store.bind(app)
    )
    assert result.output == "I need it to check availability."
    assert result.usage.requests == 2, "routing and frontend consumption belong to the turn"
    assert request_state(app) == "open", (
        "a normal root reply does not end the conversation's question"
    )
    assert len(routing.requests) == 1
    current = app.session.current_agent
    assert current is not None
    assert next(
        item.output for item in current.chat_ctx.items if item.type == "function_call_output"
    ).startswith("waiting for the person's answer to:"), (
        "a side question cannot rewrite the tool's recorded outcome"
    )

    result = await agent.run("Friday", context=store.bind(app), delegation=delegation)
    assert result.output == "Friday is selected."
    assert request_state(app) == "responded"
    assert len(routing.requests) == 2 and len(frontend.requests) == 3
    human = [item.text_content for item in app.session.chat_history.items if item.role == "user"]
    assert human == ["Book a room", "Why?", "Friday"]
    assert all(item.role != "system" for item in app.session.chat_history.items)
    prompt = json.loads(routing.requests[-1].items[-1].text_content)
    assert prompt["latest_message"] == "Friday"
    assert [candidate["question"] for candidate in prompt["candidates"]] == ["Which date?"]


@pytest.mark.parametrize(
    "verdict",
    [
        tool_call("route_input_response", candidates=[9], unclear=False),
        tool_call("route_input_response", candidates=[], unclear=True),
        RuntimeError("router unavailable"),
    ],
)
async def test_an_invalid_unclear_or_failed_detector_clarifies_without_consuming_input(
    verdict: object,
) -> None:
    store = ContextStore()
    app = await store.load_context(App())
    routing = FakeLLM([verdict])
    agent = Agent(
        name="desk",
        instructions="Help.",
        llm=FakeLLM([tool_call("date"), "Which date did you mean?"]),
    )

    @agent.tool
    async def date() -> InputRequired:
        """Ask for the date."""
        return InputRequired("Which date?", verbatim=True)

    session = TextSession(agent=agent, delegation=DelegationOptions(routing_model=routing))
    await session.run("Book", context=store.bind(app))
    result = await session.run("Yes", context=store.bind(app))
    assert result.output == "Which date did you mean?"
    assert request_state(app) == "open"
    assert len(routing.requests) == 1, "the verdict must come from the separate routing model"


async def test_canceling_a_turn_waiting_on_its_detector_closes_that_stream() -> None:
    store = ContextStore()
    app = await store.load_context(App())
    gate = Hang()
    routing = FakeLLM([gate])
    frontend = FakeLLM([tool_call("date")])
    agent = Agent(name="desk", instructions="Help.", llm=frontend)

    @agent.tool
    async def date() -> InputRequired:
        """Ask for the date."""
        return InputRequired("Which date?", verbatim=True)

    session = TextSession(agent=agent, delegation=DelegationOptions(routing_model=routing))
    await session.run("Book", context=store.bind(app))
    turn = asyncio.create_task(session.run("Friday", context=store.bind(app)))
    await until(gate.started.is_set, "the routing model did not start")
    turn.cancel()
    with pytest.raises(asyncio.CancelledError):
        await turn
    await until(gate.ended.is_set, "canceling the turn left the routing stream open")
    assert len(frontend.requests) == 1, "the canceled detector must not start the frontend turn"
    assert request_state(app) == "open"


async def test_phrasing_failure_saves_the_source_question_without_marking_it_presented() -> None:
    store = ContextStore()
    app = await store.load_context(App())
    failure = APIStatusError("phrasing failed", status_code=401)
    frontend = FakeLLM([tool_call("city"), failure, "How can I help?"])
    routing = FakeLLM([])
    agent = Agent(name="desk", instructions="Help.", llm=frontend)

    @agent.tool
    async def city() -> InputRequired:
        """Ask for the missing city."""
        return InputRequired("Which city?", verbatim=False)

    session = TextSession(agent=agent, delegation=DelegationOptions(routing_model=routing))
    with pytest.raises(APIStatusError) as raised:
        await session.run("Book", context=store.bind(app))
    assert raised.value is failure
    current = app.session.current_agent
    assert current is not None
    assert len(current.input_requests) == 1, "a failed phrasing call cannot discard its source"
    request = next(iter(current.input_requests.values()))
    assert request.state == "open" and request.source_message.text_content == "Which city?"
    assert all(item.role != "assistant" for item in app.session.chat_history.items)
    result = await session.run("Hello?", context=store.bind(app))
    assert result.output == "How can I help?" and routing.requests == [], (
        "a question the person never received cannot be offered to the routing model"
    )


async def test_a_remote_date_response_keeps_its_task_and_the_persons_exact_words(
    peer: Peer,
) -> None:
    peer.question = "Which date?"
    store = ContextStore()
    app = await store.load_context(App())
    routing = FakeLLM(
        [
            tool_call("route_input_response", candidates=[], unclear=False),
            tool_call("route_input_response", candidates=[0], unclear=False),
        ]
    )
    frontend = FakeLLM(
        [
            tool_call("coverage_expert", request="Check coverage"),
            "Which date?",
            "The service needs it to check coverage.",
            tool_call("coverage_expert", request="Saturday"),
            "Confirmed for Friday.",
        ]
    )
    agent = Agent(
        name="desk",
        instructions="Help.",
        llm=frontend,
        tools=[
            RemoteAgent(peer.url, name="coverage_expert", description="Check coverage.").as_tool(),
        ],
    )
    session = TextSession(agent=agent, delegation=DelegationOptions(routing_model=routing))
    first = await session.run("Check", context=store.bind(app))
    assert first.status == "input_required" and first.question == "Which date?"
    current = app.session.current_agent
    assert current is not None and current.input_requests == {}, "one remote question has one owner"
    observed = app.session.remote_agents[peer.url].tasks["remote-task"]
    assert len(observed.input_requests) == 1
    await session.run("Why do you need the date?", context=store.bind(app))
    assert len(peer.messages) == 1, "a side question cannot become a remote answer"
    result = await session.run("Friday", context=store.bind(app))
    assert result.output == "Confirmed for Friday."
    assert len(peer.messages) == 2
    assert peer.messages[1]["taskId"] == "remote-task"
    assert peer.messages[1]["contextId"] == "remote-context"
    assert peer.messages[1]["parts"][0] == {"text": "Friday"}, (
        "model arguments cannot rewrite the selected human response"
    )
    assert next(iter(observed.input_requests.values())).state == "responded"
    assert len(routing.requests) == 2


@pytest.mark.parametrize("correct_before_return", [False, True])
async def test_a_response_held_during_a_local_child_can_be_sent_by_another_eligible_agent(
    peer: Peer,
    correct_before_return: bool,
) -> None:
    peer.question = "Which date?"
    store = ContextStore()
    app = await store.load_context(App())
    local_model = FakeLLM([tool_call("topic")])
    if correct_before_return:
        local_model.extend([tool_call("topic")])
    local_model.extend([tool_call("continue_local"), "Policy explained."])
    local = Agent(
        name="policy",
        instructions="Explain the policy.",
        output_type=str,
        llm=local_model,
    )

    @local.tool
    async def topic() -> InputRequired:
        """Ask which policy to explain."""
        return InputRequired("Which policy?", verbatim=True)

    @local.tool
    async def continue_local(ctx: RunContext[App]) -> str:
        """Continue the local conversation before returning the floor."""
        remote = ctx.context.session.remote_agents[peer.url].tasks["remote-task"]
        held = next(iter(remote.input_requests.values()))
        assert held.state == "responding" and held.response_message_id is None
        assert len(peer.messages) == 1, "the local child has no remote tool"
        return "Continue explaining."

    sender = Agent(
        name="followup",
        instructions="Follow up on the remote request.",
        output_type=str,
        tools=[
            RemoteAgent(peer.url, name="coverage_expert", description="Check coverage.").as_tool()
        ],
        llm=FakeLLM([tool_call("coverage_expert", request="Saturday"), "Confirmed."]),
    )
    root = Agent(
        name="desk",
        instructions="Help.",
        tools=[
            RemoteAgent(peer.url, name="coverage_expert", description="Check coverage.").as_tool(),
            local.as_tool(),
            sender.as_tool(),
        ],
        llm=FakeLLM(
            [
                tool_call("coverage_expert", request="Check coverage"),
                "Which date?",
                tool_call("policy"),
                tool_call("followup"),
                "Confirmed.",
            ]
        ),
    )
    routing = FakeLLM(
        [
            tool_call("route_input_response", candidates=[], unclear=False),
            tool_call("route_input_response", candidates=[1], unclear=False),
        ]
    )
    if correct_before_return:
        routing.extend([tool_call("route_input_response", candidates=[2], unclear=False)])
    session = TextSession(agent=root, delegation=DelegationOptions(routing_model=routing))
    await session.run("Check coverage", context=store.bind(app))
    await session.run("Explain the hotel policy", context=store.bind(app))
    assert app.session.current_agent is not None
    assert app.session.current_agent.agent_name == "policy"
    result = await session.run("Friday", context=store.bind(app))
    if correct_before_return:
        assert result.status == "input_required" and len(peer.messages) == 1
        result = await session.run("Actually Saturday", context=store.bind(app))
    assert result.output == "Confirmed."
    assert len(peer.messages) == 2
    assert peer.messages[1]["taskId"] == "remote-task"
    assert peer.messages[1]["contextId"] == "remote-context"
    assert peer.messages[1]["parts"][0] == {
        "text": "Actually Saturday" if correct_before_return else "Friday"
    }
    if correct_before_return:
        prompt = json.loads(routing.requests[-1].items[-1].text_content)
        assert prompt["candidates"][2]["held_response"] == "Friday"


async def test_routed_approval_still_requires_a_validated_repeat() -> None:
    store = ContextStore()
    app = await store.load_context(App())
    body_calls = []
    routing = FakeLLM(
        [
            tool_call("route_input_response", candidates=[], unclear=False),
            tool_call("route_input_response", candidates=[0], unclear=False),
        ]
    )
    frontend = FakeLLM(
        [
            tool_call("reserve", nights=2),
            "It reserves the room for those two nights.",
            tool_call("reserve", nights=2),
            "Reserved.",
        ]
    )
    agent = Agent(name="desk", instructions="Help.", llm=frontend)

    @agent.tool(needs_approval=True)
    async def reserve(nights: int) -> str:
        """Reserve the room."""
        body_calls.append(nights)
        return "confirmed"

    session = TextSession(agent=agent, delegation=DelegationOptions(routing_model=routing))
    result = await session.run("Reserve two nights", context=store.bind(app))
    assert result.status == "input_required" and body_calls == []
    await session.run("What does that do?", context=store.bind(app))
    assert body_calls == [] and request_state(app) == "open"
    result = await session.run("Yes, reserve two nights", context=store.bind(app))
    assert result.output == "Reserved." and body_calls == [2]


@pytest.mark.parametrize("ask_another_agent", [False, True])
async def test_unanswered_agent_approval_survives_a_side_question(ask_another_agent: bool) -> None:
    store = ContextStore()
    app = await store.load_context(App())
    child_model = FakeLLM(["reserved"])
    child = Agent(name="booking", instructions="Book.", output_type=str, llm=child_model)
    details = Agent(
        name="details",
        instructions="Explain.",
        output_type=str,
        llm=FakeLLM(["It starts a booking."]),
    )
    replies = [tool_call("booking")]
    if ask_another_agent:
        replies.append(tool_call("details"))
    replies.extend(["That starts a booking.", tool_call("booking"), "Booked."])
    root = Agent(
        name="desk",
        instructions="Help.",
        tools=[child.as_tool(needs_approval=True), details.as_tool()],
        llm=FakeLLM(replies),
    )
    routing = FakeLLM(
        [
            tool_call("route_input_response", candidates=[], unclear=False),
            tool_call("route_input_response", candidates=[0], unclear=False),
        ]
    )
    session = TextSession(agent=root, delegation=DelegationOptions(routing_model=routing))
    result = await session.run("Book", context=store.bind(app))
    assert result.status == "input_required"
    current = app.session.current_agent
    assert current is not None
    original = next(iter(current.calls))
    assert current.calls[original].status == "approval" and child_model.requests == []
    await session.run("What does that do?", context=store.bind(app))
    current = app.session.current_agent
    assert current is not None and original in current.calls, (
        "an unrelated turn cannot discard the unresolved agent call"
    )
    assert current.calls[original].status == "approval" and child_model.requests == []
    result = await session.run("Yes, start it", context=store.bind(app))
    assert result.output == "Booked." and len(child_model.requests) == 1
    current = app.session.current_agent
    assert current is not None and current.calls == {}
