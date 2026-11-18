"""A remote tool addresses saved work without exposing input-occurrence IDs to the model."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

import pytest
from a2a.types import a2a_pb2 as pb

from livekit.agents import (
    Agent,
    DelegationOptions,
    SessionState,
    TextSession,
    VoiceSession,
    persisted,
)
from livekit.agents.a2a import RemoteAgent
from livekit.agents.llm import ChatContext, FunctionToolCall
from livekit.agents.storage import ContextStore
from tests.fakes import FakeAudioInput, FakeLLM, FakeSTT, FakeVAD, last_input, tool_call, until

from .a2a_peer import Peer


@dataclass
class _Conversation:
    session: SessionState = persisted(default_factory=SessionState)


async def test_a_bound_task_cancels_through_its_live_owner_while_a_snapshot_cannot(
    peer: Peer,
) -> None:
    peer.hold = True
    remote = RemoteAgent(peer.url, name="travel", description="Book flights.")
    agent = Agent(
        name="desk",
        instructions="Help.",
        tools=[remote.as_tool()],
        llm=FakeLLM(
            [tool_call("travel", request="Find a flight"), "I'm checking.", "The search stopped."]
        ),
    )
    store = ContextStore()
    app = await store.load_context(_Conversation())
    async with VoiceSession(context=store.bind(app), vad=None) as session:
        await session.start(agent)
        await session.generate_reply(user_input="Find a flight")
        await until(
            lambda: (
                peer.url in app.session.remote_agents
                and "remote-task" in app.session.remote_agents[peer.url].tasks
            ),
            "the remote working task was not retained",
        )
        task = app.session.remote_agents[peer.url].tasks["remote-task"]
        assert task.task.status.state == pb.TASK_STATE_WORKING

        detached = await store.load_session()
        with pytest.raises(ValueError, match="active bound conversation owner"):
            await detached.remote_agents[peer.url].tasks["remote-task"].cancel()
        assert not peer.cancelled

        await asyncio.gather(task.cancel(), task.cancel())

        assert peer.cancelled == ["remote-task"]
        assert task.cancel_requested
        assert task.task.status.state == pb.TASK_STATE_CANCELED
        saved = await store.load_session()
        assert (
            saved.remote_agents[peer.url].tasks["remote-task"].task.status.state
            == pb.TASK_STATE_CANCELED
        )
        requests = [
            entry
            for entry in app.session.remote_agents[peer.url].events
            if entry.kind == "request_recorded" and isinstance(entry.payload, pb.CancelTaskRequest)
        ]
        assert len(requests) == 1
        assert requests[0].payload.id == "remote-task"
    with pytest.raises(ValueError, match="owner"):
        await task.cancel()
    assert peer.cancelled == ["remote-task"], (
        "closing the frontend must not send another cancellation"
    )


def routing_input(history: ChatContext) -> dict[str, Any]:
    value = last_input(history)
    assert value is not None
    return dict(json.loads(value))


def call_route(
    proposal: dict[str, Any], action: str, task_id: str | None = None
) -> FunctionToolCall:
    candidate = (
        next(item["index"] for item in proposal["candidates"] if item["task_id"] == task_id)
        if task_id is not None
        else None
    )
    return tool_call(
        "route_remote_call",
        action=action,
        candidate=candidate,
        response=0 if action == "respond" else None,
    )


def clarification_route(history: ChatContext) -> FunctionToolCall:
    proposal = routing_input(history)
    if "proposed_call" in proposal:
        return call_route(proposal, "continue", "booking-42")
    return tool_call("route_input_response", candidates=[], unclear=False)


def waiting_task(message: dict[str, Any]) -> list[dict[str, Any]]:
    """The peer keeps asking for a date; clarification does not complete that business work."""
    response = message["parts"][0].get("text") == "Friday"
    return [
        {
            "task": {
                "id": "booking-42",
                "contextId": message.get("contextId", "trip"),
                "status": {
                    "state": "TASK_STATE_COMPLETED" if response else "TASK_STATE_INPUT_REQUIRED",
                    "message": {
                        "messageId": "booking-completed" if response else "date-question",
                        "role": "ROLE_AGENT",
                        "parts": [{"text": "Booked for Friday" if response else "Which date?"}],
                    },
                },
            }
        }
    ]


@pytest.mark.parametrize("surface", ["text", "run", "stream"])
@pytest.mark.parametrize("detect", [True, False])
async def test_task_id_continues_saved_work_for_clarification_without_answering_it(
    peer: Peer, surface: str, detect: bool
) -> None:
    peer.events = waiting_task
    remote = RemoteAgent(peer.url, name="flights", description="Book flights.")
    agent = Agent(
        name="desk",
        instructions="Help.",
        tools=[remote.as_tool()],
        llm=FakeLLM(
            [
                tool_call("flights", request="Book a flight"),
                "Which date?",
                tool_call("flights", request="Why do you need the date?", task_id="booking-42"),
                "Which date?",
            ]
        ),
    )
    store = ContextStore()
    delegation = DelegationOptions(routing_model=FakeLLM(clarification_route) if detect else None)
    session = TextSession(agent=agent, delegation=delegation)

    async def turn(text: str) -> None:
        if surface == "text":
            result = await session.run(text, context=store.bind())
        elif surface == "run":
            result = await agent.run(text, context=store.bind(), delegation=delegation)
        else:
            async with agent.run_stream(
                text, context=store.bind(), delegation=delegation
            ) as stream:
                async for _ in stream.stream_text(delta=True):
                    pass
                result = await stream.result()
        assert result.status == "input_required"

    await turn("Book a flight")
    before = await store.load_session()
    await turn("Why do you need the date?")
    assert len(peer.messages) == 2
    assert "taskId" not in peer.messages[0]
    assert peer.messages[1]["taskId"] == "booking-42"
    assert peer.messages[1]["contextId"] == "trip"
    assert peer.messages[1]["parts"][0] == {"text": "Why do you need the date?"}
    after = await store.load_session()
    previous = before.remote_agents[peer.url].tasks["booking-42"]
    current = after.remote_agents[peer.url].tasks["booking-42"]
    assert list(current.input_requests) == list(previous.input_requests)
    assert [request.state for request in current.input_requests.values()] == ["open"]


async def test_custom_remote_schema_keeps_business_data_and_strips_task_routing(peer: Peer) -> None:
    peer.input_schema = {
        "type": "object",
        "properties": {"destination": {"type": "string"}},
        "required": ["destination"],
        "additionalProperties": False,
    }
    peer.events = waiting_task
    remote = RemoteAgent(peer.url)
    agent = Agent(
        name="desk",
        instructions="Help.",
        tools=[remote.as_tool()],
        llm=FakeLLM(
            [
                tool_call("coverage_expert", destination="Paris", task_id=None),
                "Which date?",
                tool_call("coverage_expert", destination="Rome", task_id="booking-42"),
                "Which date?",
            ]
        ),
    )
    session = TextSession(
        agent=agent,
        delegation=DelegationOptions(routing_model=FakeLLM(clarification_route)),
    )
    store = ContextStore()
    await session.run("Paris", context=store.bind())
    await session.run("Actually Rome", context=store.bind())
    assert [message["parts"][0] for message in peer.messages] == [
        {"data": {"destination": "Paris"}},
        {"data": {"destination": "Rome"}},
    ]
    assert peer.messages[1]["taskId"] == "booking-42"
    assert peer.messages[1]["contextId"] == "trip"


@pytest.mark.parametrize("task_id", ["missing", "remote-task"])
async def test_unknown_or_terminal_task_returns_tool_feedback_without_dispatch(
    peer: Peer, task_id: str
) -> None:
    remote = RemoteAgent(peer.url, name="flights", description="Book flights.")
    agent = Agent(
        name="desk",
        instructions="Help.",
        tools=[remote.as_tool()],
        llm=FakeLLM(
            [
                tool_call("flights", request="Book"),
                "Booked.",
                tool_call("flights", request="Correct it", task_id=task_id),
                "That task cannot be continued.",
            ]
        ),
    )
    store = ContextStore()
    await agent.run("Book", context=store.bind())
    result = await agent.run("Correct it", context=store.bind())
    outputs = [item for item in result.new_messages() if item.type == "function_call_output"]
    assert outputs[-1].is_error
    assert ("unknown" if task_id == "missing" else "ended") in outputs[-1].output
    assert len(peer.messages) == 1, "invalid task routing must fail before the network send"


async def test_reused_provider_call_ids_do_not_recover_an_earlier_business_request(
    peer: Peer,
) -> None:
    def complete(message: dict[str, Any]) -> list[dict[str, Any]]:
        task_id = "flight" if "flight" in message["parts"][0]["text"] else "hotel"
        return [
            {
                "task": {
                    "id": task_id,
                    "contextId": message.get("contextId", "trip"),
                    "status": {
                        "state": "TASK_STATE_COMPLETED",
                        "message": {
                            "messageId": f"{task_id}-completed",
                            "role": "ROLE_AGENT",
                            "parts": [{"text": f"Booked {task_id}"}],
                        },
                    },
                }
            }
        ]

    peer.events = complete
    remote = RemoteAgent(peer.url, name="travel", description="Book travel.")
    agent = Agent(
        name="desk",
        instructions="Help.",
        tools=[remote.as_tool()],
        llm=FakeLLM(
            [
                tool_call("travel", request="Book flight", call_id="reused"),
                "Flight booked.",
                tool_call("travel", request="Book hotel", call_id="reused"),
                "Hotel booked.",
            ]
        ),
    )
    store = ContextStore()
    await agent.run("Book flight", context=store.bind())
    result = await agent.run("Book hotel", context=store.bind())
    assert peer.methods == ["SendStreamingMessage", "SendStreamingMessage"], (
        "a new call must not reconcile the earlier call's task"
    )
    outputs = [item for item in result.new_messages() if item.type == "function_call_output"]
    assert not outputs[-1].is_error and "Booked hotel" in outputs[-1].output
    assert [message["parts"][0] for message in peer.messages] == [
        {"text": "Book flight"},
        {"text": "Book hotel"},
    ]
    assert "taskId" not in peer.messages[1]
    assert peer.messages[1]["contextId"] == "trip"
    shared = next(part["data"]["items"] for part in peer.messages[1]["parts"] if "data" in part)
    assert any(item["type"] == "function_call" and item["call_id"] == "reused" for item in shared)
    assert any(
        item["type"] == "function_call_output" and "Booked flight" in item["output"]
        for item in shared
    )


@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize("acknowledge", [True, False])
async def test_voice_clarification_reuses_the_live_task_without_claiming_a_response(
    peer: Peer, structured: bool, acknowledge: bool
) -> None:
    def events(message: dict[str, Any]) -> list[dict[str, Any]]:
        result = waiting_task(message)
        if "taskId" not in message:
            result.insert(
                0,
                {
                    "task": {
                        "id": "booking-42",
                        "contextId": "trip",
                        "status": {"state": "TASK_STATE_WORKING"},
                    }
                },
            )
        return result

    peer.events = events
    peer.response_gate = asyncio.Event()
    if structured:
        peer.input_schema = {
            "type": "object",
            "properties": {"destination": {"type": "string"}},
            "required": ["destination"],
        }
    remote = RemoteAgent(peer.url, name="flights")
    initial_reply = asyncio.Event()

    def frontend(history: ChatContext) -> str | FunctionToolCall:
        latest = next(
            (
                item.text_content
                for item in reversed(history.items)
                if item.type == "message" and item.role == "user"
            ),
            None,
        )
        calls = [item for item in history.items if item.type == "function_call"]
        if latest == "Friday":
            if not any("Friday" in item.arguments for item in calls):
                arguments = (
                    {"destination": "Friday, please."}
                    if structured
                    else {"request": "Friday, please."}
                )
                return tool_call("flights", **arguments, task_id="booking-42")
            if any(
                item.type == "function_call_output" and "Booked for Friday" in item.output
                for item in history.items
            ):
                return "Booked for Friday"
            return "I'll pass that on."
        if latest == "Why do you need the date?" and not any(
            "task_id" in item.arguments for item in calls
        ):
            arguments = {"destination": "Rome"} if structured else {"request": latest}
            return tool_call("flights", **arguments, task_id="booking-42")
        if any(
            item.type == "message"
            and item.role != "user"
            and "Which date?" in (item.text_content or "")
            for item in history.items
        ):
            return "Which date?"
        if calls:
            initial_reply.set()
            return "I'm checking." if acknowledge else ""
        return tool_call(
            "flights", **({"destination": "Paris"} if structured else {"request": "Book"})
        )

    store = ContextStore()
    microphone = FakeAudioInput()

    def route(history: ChatContext) -> FunctionToolCall:
        proposal = routing_input(history)
        if "proposed_call" in proposal:
            action = "respond" if "Friday" in json.dumps(proposal["proposed_call"]) else "continue"
            return call_route(proposal, action, "booking-42")
        return tool_call(
            "route_input_response",
            candidates=[0] if proposal["latest_message"] == "Friday" else [],
            unclear=False,
        )

    async with VoiceSession(
        context=store.bind(),
        delegation=DelegationOptions(routing_model=FakeLLM(route)),
        vad=FakeVAD(),
        stt=FakeSTT(["Book", "Why do you need the date?", "Friday"]),
        turn_handling={"turn_detection": None, "endpointing": {"min_delay": 0.1}},
        aec_warmup_duration=None,
    ) as session:
        session.input.audio = microphone
        await session.start(
            Agent(
                name="desk", instructions="Help.", tools=[remote.as_tool()], llm=FakeLLM(frontend)
            )
        )
        microphone.speak(0.4)
        try:
            await until(
                initial_reply.is_set, "the frontend did not take its initial reply opportunity"
            )
            if acknowledge:
                await until(
                    lambda: any(
                        item.type == "message" and item.text_content == "I'm checking."
                        for item in session.history.items
                    ),
                    "the frontend's acknowledgement was not delivered",
                )
        finally:
            peer.response_gate.set()
        await until(
            lambda: any(
                item.type == "message"
                and item.role == "assistant"
                and item.text_content == "Which date?"
                for item in session.history.items
            ),
            "the task's original question was not presented",
        )
        original = next(item for item in session.history.items if item.type == "function_call")
        pending = next(
            item
            for item in session.history.items
            if item.type == "function_call_output" and item.id == f"{original.id}_out"
        )
        assert "The remote request is running" in pending.output
        assert (
            any(
                item.type == "message"
                and item.role == "assistant"
                and item.text_content == "I'm checking."
                for item in session.history.items
            )
            == acknowledge
        )
        microphone.speak(0.4)
        await until(
            lambda: len(peer.messages) == 2, "the clarification did not reach the live task"
        )
        assert peer.messages[1]["taskId"] == "booking-42"
        assert peer.messages[1]["contextId"] == "trip"
        assert peer.messages[1]["parts"][0] == (
            {"data": {"destination": "Rome"}}
            if structured
            else {"text": "Why do you need the date?"}
        )
        saved = await store.load_session()
        task = saved.remote_agents[peer.url].tasks["booking-42"]
        assert [request.state for request in task.input_requests.values()] == ["open"]
        assert (
            sum(event.kind == "request_recorded" for event in saved.remote_agents[peer.url].events)
            == 2
        ), "a second observer must not be created"
        await until(
            lambda: (
                sum(
                    item.type == "message"
                    and item.role == "assistant"
                    and item.text_content == "Which date?"
                    for item in session.history.items
                )
                >= 2
            ),
            "the still-open question was not presented after clarification",
        )
        microphone.speak(0.4)
        await until(
            lambda: any(
                item.type == "function_call_output"
                and item.id == f"{original.id}_out"
                and "Booked for Friday" in item.output
                for item in session.history.items
            ),
            "the task's accepted result did not resolve the original call",
        )
        assert len(peer.messages) == 3
        assert peer.messages[2]["taskId"] == "booking-42"
        assert peer.messages[2]["parts"][0] == {"text": "Friday"}, (
            "forward the reserved words exactly"
        )
        saved = await store.load_session()
        task = saved.remote_agents[peer.url].tasks["booking-42"]
        assert [request.state for request in task.input_requests.values()] == ["responded"]


def travel_task(
    message: dict[str, Any], task_id: str, text: str, *, completed: bool = False
) -> list[dict[str, Any]]:
    return [
        {
            "task": {
                "id": task_id,
                "contextId": message.get("contextId", "trip"),
                "status": {
                    "state": "TASK_STATE_COMPLETED" if completed else "TASK_STATE_INPUT_REQUIRED",
                    "message": {
                        "messageId": f"{task_id}-{'completed' if completed else 'question'}",
                        "role": "ROLE_AGENT",
                        "parts": [{"text": text}],
                    },
                },
            }
        }
    ]


async def test_one_answer_is_forwarded_once_while_the_same_turn_starts_hotel_work(
    peer: Peer,
) -> None:
    """An already-forwarded flight answer cannot blanket-block independent hotel work."""

    def events(message: dict[str, Any]) -> list[dict[str, Any]]:
        if message.get("taskId") == "flight":
            return travel_task(message, "flight", "Flight confirmed", completed=True)
        if "hotel" in message["parts"][0]["text"].lower():
            return travel_task(message, "hotel", "Hotel confirmed", completed=True)
        return travel_task(message, "flight", "Which flight date?")

    peer.events = events
    human = "Friday. Also book a hotel."

    def route(history: ChatContext) -> FunctionToolCall:
        proposal = routing_input(history)
        if "proposed_call" not in proposal:
            return tool_call("route_input_response", candidates=[0], unclear=False)
        if proposal["proposed_call"]["request"] == "Book a hotel":
            return call_route(proposal, "new")
        return call_route(proposal, "respond", "flight")

    model = FakeLLM(
        [
            tool_call("travel", request="Book a flight"),
            "Which flight date?",
            tool_call("travel", request="Friday, please."),
            tool_call("travel", request="Friday, please.", call_id="duplicate-answer"),
            tool_call("travel", request="Book a hotel"),
            "Flight and hotel confirmed.",
        ]
    )
    agent = Agent(
        name="desk",
        instructions="Help with travel.",
        tools=[RemoteAgent(peer.url, name="travel").as_tool()],
        llm=model,
    )
    store = ContextStore()
    session = TextSession(agent=agent, delegation=DelegationOptions(routing_model=FakeLLM(route)))
    first = await session.run("Book a flight", context=store.bind())
    assert first.status == "input_required"
    result = await session.run(human, context=store.bind())
    assert result.status == "completed" and result.output == "Flight and hotel confirmed."
    assert [message["parts"][0] for message in peer.messages] == [
        {"text": "Book a flight"},
        {"text": human},
        {"text": "Book a hotel"},
    ]
    assert peer.messages[1]["taskId"] == "flight"
    assert "taskId" not in peer.messages[2]
    assert peer.messages[2]["contextId"] == "trip"
    duplicate = next(
        item
        for item in result.new_messages()
        if item.type == "function_call_output" and item.call_id == "duplicate-answer"
    )
    assert not duplicate.is_error
    saved = await store.load_session()
    remote = saved.remote_agents[peer.url]
    assert set(remote.tasks) == {"flight", "hotel"}
    assert [request.state for request in remote.tasks["flight"].input_requests.values()] == [
        "responded"
    ]
    assert sum(event.kind == "request_recorded" for event in remote.events) == 3


async def test_parallel_calls_for_one_reserved_answer_succeed_without_sending_it_twice(
    peer: Peer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both calls can classify the same answer before either outgoing intent is prepared."""
    bookings: list[str] = []

    def events(message: dict[str, Any]) -> list[dict[str, Any]]:
        if message.get("taskId") == "booking-42":
            bookings.append(message["parts"][0]["text"])
        return waiting_task(message)

    peer.events = events
    proposed: list[dict[str, Any]] = []

    def route(history: ChatContext) -> FunctionToolCall:
        proposal = routing_input(history)
        if "proposed_call" not in proposal:
            return tool_call("route_input_response", candidates=[0], unclear=False)
        proposed.append(proposal)
        return call_route(proposal, "respond", "booking-42")

    agent = Agent(
        name="desk",
        instructions="Help with flights.",
        tools=[RemoteAgent(peer.url, name="travel", description="Book flights.").as_tool()],
        llm=FakeLLM(
            [
                tool_call("travel", request="Book a flight"),
                "Which date?",
                [
                    tool_call("travel", request="Friday, please.", call_id="first-answer"),
                    tool_call("travel", request="Friday, please.", call_id="second-answer"),
                ],
                "Booked for Friday.",
            ]
        ),
    )
    store = ContextStore()
    session = TextSession(agent=agent, delegation=DelegationOptions(routing_model=FakeLLM(route)))
    first = await session.run("Book a flight", context=store.bind())
    assert first.status == "input_required"

    discovering = 0
    first_admitted, duplicate_tracked, release_discovery = (
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
    )
    respond = peer._respond

    async def gated_discovery(writer: asyncio.StreamWriter, kind: str, body: bytes) -> None:
        nonlocal discovering
        if kind == "application/json" and "supportedInterfaces" in json.loads(body):
            discovering += 1
            first_admitted.set()
            await release_discovery.wait()
        await respond(writer, kind, body)

    # Discovery follows native admission, before Journal Prepare. While the owner is held
    # here, its duplicate must finish through shared admission without opening another client.
    monkeypatch.setattr(peer, "_respond", gated_discovery)
    async with session.run_stream("Friday", context=store.bind()) as stream:

        async def observe() -> None:
            async for event in stream.stream_events():
                if (
                    event.type == "tool_result"
                    and event.outcome == "completed"
                    and isinstance(event.output, str)
                    and "already tracked" in event.output
                ):
                    duplicate_tracked.set()

        observing = asyncio.create_task(observe())
        try:
            await until(first_admitted.is_set, "the original answer did not reach admission")
            await until(
                duplicate_tracked.is_set, "the parallel duplicate was not tracked before Prepare"
            )
            assert discovering == 1, "the duplicate must not open another task observer"
            assert len(proposed) == 2
            assert all(
                proposal["candidates"][0]["responses"][0]["held_response"] == "Friday"
                for proposal in proposed
            )
            held = (await store.load_session()).remote_agents[peer.url].tasks["booking-42"]
            assert all(
                request.response_message_id is None for request in held.input_requests.values()
            ), "coalescing must succeed while the original response is still unprepared"
            assert len(peer.messages) == 1, "the response cannot Prepare before the gate"
            release_discovery.set()
            await observing
            result = await stream.result()
        finally:
            release_discovery.set()
            if not observing.done():
                observing.cancel()
            await asyncio.gather(observing, return_exceptions=True)

    assert result.status == "completed" and result.output == "Booked for Friday."
    assert bookings == ["Friday"], "the response must have one business effect"
    assert [message["parts"][0] for message in peer.messages] == [
        {"text": "Book a flight"},
        {"text": "Friday"},
    ]
    outputs = {
        item.call_id: item for item in result.new_messages() if item.type == "function_call_output"
    }
    assert set(outputs) == {"first-answer", "second-answer"}
    assert all(not output.is_error for output in outputs.values()), {
        call_id: output.output for call_id, output in outputs.items()
    }
    assert all(
        "Booked for Friday" in output.output or "already tracked" in output.output
        for output in outputs.values()
    )
    saved = await store.load_session()
    task = saved.remote_agents[peer.url].tasks["booking-42"]
    assert [request.state for request in task.input_requests.values()] == ["responded"]


async def test_two_open_tasks_keep_explicit_clarification_and_answers_separate(peer: Peer) -> None:
    def events(message: dict[str, Any]) -> list[dict[str, Any]]:
        task = message.get("taskId")
        text = message["parts"][0]["text"]
        if task == "flight":
            return travel_task(message, "flight", "Flight confirmed", completed=True)
        if task == "hotel" or text == "Book a hotel":
            return travel_task(message, "hotel", "What hotel budget?")
        return travel_task(message, "flight", "Which flight date?")

    peer.events = events

    def route(history: ChatContext) -> FunctionToolCall:
        proposal = routing_input(history)
        if "proposed_call" not in proposal:
            selected = (
                [item["index"] for item in proposal["candidates"] if "flight" in item["question"]]
                if proposal["latest_message"] == "Friday for the flight"
                else []
            )
            return tool_call("route_input_response", candidates=selected, unclear=False)
        request = proposal["proposed_call"]["request"]
        if request == "Book a hotel":
            return call_route(proposal, "new")
        task = "flight" if request == "Friday" else "hotel"
        assert proposal["selected_task"] == 0
        assert [candidate["task_id"] for candidate in proposal["candidates"]] == [task]
        return call_route(proposal, "respond" if task == "flight" else "continue", task)

    agent = Agent(
        name="desk",
        instructions="Help with travel.",
        tools=[RemoteAgent(peer.url, name="travel").as_tool()],
        llm=FakeLLM(
            [
                tool_call("travel", request="Book a flight"),
                "Which flight date?",
                tool_call("travel", request="Book a hotel"),
                "What hotel budget?",
                tool_call("travel", request="Choose an airport hotel", task_id="hotel"),
                "What hotel budget?",
                tool_call("travel", request="Friday", task_id="flight"),
                "Flight confirmed; the hotel still needs a budget.",
            ]
        ),
    )
    store = ContextStore()
    session = TextSession(agent=agent, delegation=DelegationOptions(routing_model=FakeLLM(route)))
    await session.run("Book a flight", context=store.bind())
    await session.run("Also book a hotel", context=store.bind())
    before = await store.load_session()
    assert set(before.remote_agents[peer.url].tasks) == {"flight", "hotel"}
    await session.run("Choose an airport hotel", context=store.bind())
    clarified = await store.load_session()
    for task in ("flight", "hotel"):
        earlier = before.remote_agents[peer.url].tasks[task].input_requests
        current = clarified.remote_agents[peer.url].tasks[task].input_requests
        assert list(current) == list(earlier)
        assert [request.state for request in current.values()] == ["open"]
    await session.run("Friday for the flight", context=store.bind())
    assert [message.get("taskId") for message in peer.messages] == [None, None, "hotel", "flight"]
    assert peer.messages[2]["parts"][0] == {"text": "Choose an airport hotel"}
    assert peer.messages[3]["parts"][0] == {"text": "Friday for the flight"}
    saved = await store.load_session()
    assert [
        request.state
        for request in saved.remote_agents[peer.url].tasks["hotel"].input_requests.values()
    ] == ["open"]
    assert [
        request.state
        for request in saved.remote_agents[peer.url].tasks["flight"].input_requests.values()
    ] == ["responded"]


def latest_human(history: ChatContext) -> str | None:
    return next(
        (
            item.text_content
            for item in reversed(history.items)
            if item.type == "message" and item.role == "user"
        ),
        None,
    )


def said(session: VoiceSession, text: str) -> bool:
    return any(
        item.type == "message" and item.role == "assistant" and item.text_content == text
        for item in session.history.items
    )


async def test_a_working_task_correction_keeps_the_original_call_as_the_only_result_owner(
    peer: Peer,
) -> None:
    peer.hold = True

    def frontend(history: ChatContext) -> str | FunctionToolCall:
        calls = [item for item in history.items if item.type == "function_call"]
        if latest_human(history) == "Actually Saturday":
            if not any("Saturday" in item.arguments for item in calls):
                return tool_call("travel", request="Actually Saturday")
            if any(
                item.type == "function_call_output" and "Confirmed" in item.output
                for item in history.items
            ):
                return "Saturday confirmed."
            return "I'll pass on Saturday."
        if not calls:
            return tool_call("travel", request="Book a flight for Friday")
        return "I'm checking."

    def route(history: ChatContext) -> FunctionToolCall:
        proposal = routing_input(history)
        assert proposal["proposed_call"] == {"request": "Actually Saturday"}
        return call_route(proposal, "continue", "remote-task")

    store = ContextStore()
    async with VoiceSession(
        context=store.bind(),
        vad=None,
        delegation=DelegationOptions(routing_model=FakeLLM(route)),
    ) as session:
        await session.start(
            Agent(
                name="desk",
                instructions="Help with travel.",
                tools=[RemoteAgent(peer.url, name="travel").as_tool()],
                llm=FakeLLM(frontend),
            )
        )
        await session.generate_reply(user_input="Book a flight for Friday")
        await until(lambda: len(peer.messages) == 1, "the original flight request never started")
        original = next(item for item in session.history.items if item.type == "function_call")
        peer.hold = False
        await session.generate_reply(user_input="Actually Saturday")
        await until(
            lambda: any(
                item.type == "function_call_output"
                and item.id == f"{original.id}_out"
                and "Confirmed" in item.output
                for item in session.history.items
            ),
            "the correction's result did not resolve the original tool call",
        )
        await session.wait_for_idle()
        assert said(session, "Saturday confirmed.")
        assert peer.methods == ["SendStreamingMessage", "SendStreamingMessage"]
        assert [message["parts"][0] for message in peer.messages] == [
            {"text": "Book a flight for Friday"},
            {"text": "Actually Saturday"},
        ]
        assert peer.messages[1]["taskId"] == "remote-task"
        calls = [item for item in session.history.items if item.type == "function_call"]
        assert len(calls) == 2
        outputs = [item for item in session.history.items if item.type == "function_call_output"]
        accepted = [item for item in outputs if "Confirmed" in item.output]
        assert len(accepted) == 1 and accepted[0].id == f"{original.id}_out"
        correction = next(item for item in outputs if item.id == f"{calls[1].id}_out")
        assert not correction.is_error and "already tracked" in correction.output
        saved = await store.load_session()
        remote = saved.remote_agents[peer.url]
        assert sum(event.kind == "request_recorded" for event in remote.events) == 2
        assert [
            artifact.artifact_id for artifact in remote.tasks["remote-task"].task.artifacts
        ] == ["answer"]
    assert peer.cancelled == []


async def test_a_forwarded_voice_answer_is_not_reasked_when_the_agent_returns(peer: Peer) -> None:
    """Native answer forwarding clears presentation even while the remote task keeps working."""
    peer.question = "Which date?"

    def frontend(history: ChatContext) -> str | FunctionToolCall:
        if (last_input(history) or "").startswith("Ask the person this"):
            return "Which date?"
        calls = [item for item in history.items if item.type == "function_call"]
        if latest_human(history) == "Any news?":
            return "The Friday search is still running."
        if latest_human(history) == "Friday":
            if not any("Friday" in item.arguments for item in calls):
                return tool_call("travel", request="Friday, please.")
            return "I'll keep checking Friday."
        if not calls:
            return tool_call("travel", request="Book a flight")
        return "I'm checking."

    def route(history: ChatContext) -> FunctionToolCall:
        proposal = routing_input(history)
        if "proposed_call" in proposal:
            return call_route(proposal, "respond", "remote-task")
        return tool_call(
            "route_input_response",
            candidates=[0] if proposal["latest_message"] == "Friday" else [],
            unclear=False,
        )

    desk = Agent(
        name="desk",
        instructions="Help with travel.",
        tools=[RemoteAgent(peer.url, name="travel").as_tool()],
        llm=FakeLLM(frontend),
    )
    helper = Agent(name="helper", instructions="Help.", llm=FakeLLM(["How can I help?"]))
    desk_entered, helper_entered = asyncio.Event(), asyncio.Event()

    @desk.on_enter
    async def returned_to_desk(session: VoiceSession) -> None:
        desk_entered.set()

    @helper.on_enter
    async def entered_helper(session: VoiceSession) -> None:
        helper_entered.set()

    store = ContextStore()
    async with VoiceSession(
        context=store.bind(),
        vad=None,
        delegation=DelegationOptions(routing_model=FakeLLM(route)),
    ) as session:
        await session.start(desk)
        session.generate_reply(user_input="Book a flight")
        await until(lambda: said(session, "Which date?"), "the date question was not presented")
        peer.hold = True
        await session.generate_reply(user_input="Friday")
        await until(lambda: len(peer.messages) == 2, "the answer was not forwarded")
        await session.wait_for_idle()
        saved = await store.load_session()
        task = saved.remote_agents[peer.url].tasks["remote-task"]
        assert [request.state for request in task.input_requests.values()] == ["responded"]
        assert task.task.status.state == pb.TASK_STATE_WORKING

        session.update_agent(helper)
        await until(helper_entered.is_set, "the helper did not take the floor")
        desk_entered.clear()
        session.update_agent(desk)
        await until(desk_entered.is_set, "the travel desk did not return")
        await session.wait_for_idle()
        await session.generate_reply(user_input="Any news?")
        await session.wait_for_idle()
        assert said(session, "The Friday search is still running.")
        assert (
            sum(
                item.type == "message"
                and item.role == "assistant"
                and item.text_content == "Which date?"
                for item in session.history.items
            )
            == 1
        ), "an answered question was presented again after the agent returned"
        assert len(peer.messages) == 2
    assert peer.cancelled == [], "ending the frontend must only detach its observer"
    saved = await store.load_session()
    task = saved.remote_agents[peer.url].tasks["remote-task"]
    assert task.task.status.state == pb.TASK_STATE_WORKING
    assert not task.cancel_requested


async def test_a_new_request_merged_into_an_idle_task_resolves_only_its_original_call(
    peer: Peer,
) -> None:
    """A peer can attach new work to a task whose original live call is waiting for input."""

    def events(message: dict[str, Any]) -> list[dict[str, Any]]:
        completed = len(peer.messages) == 2
        events = travel_task(
            message,
            "flight",
            "Flight and hotel confirmed" if completed else "Which flight date?",
            completed=completed,
        )
        peer.task_snapshot = events[0]["task"]
        return events

    peer.events = events

    def frontend(history: ChatContext) -> str | FunctionToolCall:
        if (last_input(history) or "").startswith("Ask the person this"):
            return "Which flight date?"
        calls = [item for item in history.items if item.type == "function_call"]
        if latest_human(history) == "Also find a hotel":
            if not any("hotel" in item.arguments for item in calls):
                return tool_call("travel", request="Find a hotel")
            if any(
                item.type == "function_call_output" and "Flight and hotel confirmed" in item.output
                for item in history.items
            ):
                return "Both requests confirmed."
            return "I'll track both requests."
        if not calls:
            return tool_call("travel", request="Book a flight")
        return "I'm checking."

    def route(history: ChatContext) -> FunctionToolCall:
        proposal = routing_input(history)
        if "proposed_call" in proposal:
            assert proposal["proposed_call"] == {"request": "Find a hotel"}
            return call_route(proposal, "new")
        assert proposal["latest_message"] == "Also find a hotel"
        return tool_call("route_input_response", candidates=[], unclear=False)

    async with VoiceSession(
        vad=None,
        delegation=DelegationOptions(routing_model=FakeLLM(route)),
    ) as session:
        await session.start(
            Agent(
                name="desk",
                instructions="Help with travel.",
                tools=[RemoteAgent(peer.url, name="travel").as_tool()],
                llm=FakeLLM(frontend),
            )
        )
        session.generate_reply(user_input="Book a flight")
        await until(
            lambda: said(session, "Which flight date?"),
            "the original task did not enter its waiting-for-input state",
        )
        original = next(item for item in session.history.items if item.type == "function_call")
        await session.generate_reply(user_input="Also find a hotel")
        await until(
            lambda: any(
                item.type == "function_call_output"
                and item.id == f"{original.id}_out"
                and "Flight and hotel confirmed" in item.output
                for item in session.history.items
            ),
            "merging a new request stranded the original task's result owner",
        )
        await session.wait_for_idle()
        calls = [item for item in session.history.items if item.type == "function_call"]
        assert len(calls) == 2
        outputs = [item for item in session.history.items if item.type == "function_call_output"]
        final = [item for item in outputs if "Flight and hotel confirmed" in item.output]
        assert len(final) == 1 and final[0].id == f"{original.id}_out"
        merged = next(item for item in outputs if item.id == f"{calls[1].id}_out")
        assert not merged.is_error and "already tracked" in merged.output
        assert peer.methods.count("SendStreamingMessage") == 2, "business input was resent"
        assert peer.methods.count("GetTask") == 1, "the sole owner reconciles the merged task"
        assert "SubscribeToTask" not in peer.methods
        assert [message["parts"][0] for message in peer.messages] == [
            {"text": "Book a flight"},
            {"text": "Find a hotel"},
        ]
    assert peer.cancelled == []
