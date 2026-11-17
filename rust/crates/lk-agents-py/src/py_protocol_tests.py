"""Native observations use the upstream SDK records and retain every stream event."""

from dataclasses import dataclass

import pytest
from a2a.types import a2a_pb2 as pb
from google.protobuf.json_format import MessageToDict

from livekit.agents import Agent, SessionState, TextSession, persisted
from livekit.agents.a2a import AgentClient, RemoteAgent
from livekit.agents.storage import ContextStore
from livekit.agents.storage.session_state import RemoteAgentState, RemoteContextState, RemoteTask
from tests.fakes import FakeLLM, tool_call
from tests.host.a2a_peer import Peer

pytest_plugins = ["tests.host.a2a_peer"]

pytestmark = pytest.mark.unit


async def test_raw_a2a_observations_retain_artifacts_as_sdk_events(peer: Peer) -> None:
    client = AgentClient(peer.url)
    try:
        reply = await client.send("check")
        events = []
        while (event := await reply._core_next_event()) is not None:
            assert isinstance(event, pb.StreamResponse)
            events.append(event)
        assert [event.WhichOneof("payload") for event in events] == [
            "task",
            "status_update",
            "artifact_update",
            "status_update",
        ]
        artifact = events[2].artifact_update
        assert artifact.task_id == "remote-task"
        assert artifact.artifact.parts[0].text == "Confirmed"
        assert events[-1].status_update.status.state == pb.TASK_STATE_COMPLETED
    finally:
        await client.aclose()


async def test_bound_remote_work_journals_protocol_before_returning_the_tool_result(
    peer: Peer,
) -> None:
    @dataclass
    class App:
        session: SessionState = persisted(default_factory=SessionState)

    store = ContextStore()
    app = await store.load_context(App())
    remote = RemoteAgent(peer.url, name="coverage_expert", description="Check coverage.")
    agent = Agent(
        name="desk",
        instructions="Help.",
        tools=[remote.as_tool()],
        llm=FakeLLM(
            [
                tool_call("coverage_expert", call_id="check_coverage", request="Check"),
                "Confirmed.",
                tool_call("coverage_expert", call_id="recheck", request="Check again"),
                "Still confirmed.",
            ]
        ),
    )
    session = TextSession(agent=agent)
    result = await session.run("Check coverage.", context=store.bind(app))
    assert result.output == "Confirmed."
    saved = app.session._snapshot_data()
    assert saved is not None
    observed = saved.remote_agents[peer.url]
    assert len(peer.messages) == 1 and peer.cancelled == []
    assert [event.kind for event in observed.events] == [
        "request_recorded",
        "response_received",
        "response_received",
        "response_received",
        "response_received",
    ]
    assert isinstance(observed.events[0].payload, pb.SendMessageRequest)
    assert all(isinstance(event.payload, pb.StreamResponse) for event in observed.events[1:])
    assert observed.tasks["remote-task"].task.status.state == pb.TASK_STATE_COMPLETED
    assert observed.tasks["remote-task"].task.artifacts[0].parts[0].text == "Confirmed"
    request = next(iter(observed.requests.values()))
    assert request.dispatch_started and request.ended
    assert request.origin is not None and request.origin.call_id == "check_coverage"
    bound = app.session.remote_agents[peer.url]
    detached = app.session.snapshot().remote_agents[peer.url]
    assert isinstance(bound, RemoteAgentState)
    assert bound.endpoint == peer.url and len(bound.events) == 5
    context = bound.contexts["remote-context"]
    assert isinstance(context, RemoteContextState)
    task = context.tasks["remote-task"]
    assert isinstance(task, RemoteTask) and isinstance(task.task, pb.Task)
    assert task.task.status.state == pb.TASK_STATE_COMPLETED
    assert task.input_requests == {} and not task.cancel_requested
    returned = task.task
    returned.id = "reader mutation"
    assert task.task.id == "remote-task", "protocol inspection cannot mutate the conversation"
    assert all(isinstance(message, pb.Message) for message in bound.messages.values())
    await session.run("Check again.", context=store.bind(app))
    assert "contextId" not in peer.messages[0]
    assert peer.messages[1]["contextId"] == "remote-context", "new work reuses the saved context"
    assert "taskId" not in peer.messages[1], "new work does not resume the completed task"
    assert len(bound.events) == 10, "a retained bound view follows the owner's checkpoints"
    assert len(detached.events) == 5, "an inspection snapshot stays fixed"


async def test_a_contextless_direct_message_is_not_a_synthetic_task(peer: Peer) -> None:
    peer.direct_reply = {
        "messageId": "direct",
        "role": "ROLE_AGENT",
        "parts": [
            {"text": "Here is the receipt"},
            {"data": {"receipt": {"amount": 42}}},
        ],
    }
    client = AgentClient(peer.url)
    try:
        reply = await client.send("receipt")
        event = await reply._core_next_event()
        assert isinstance(event, pb.StreamResponse)
        assert event.WhichOneof("payload") == "message"
        assert event.message.message_id == "direct"
        assert not event.message.context_id and not event.message.task_id
        assert MessageToDict(event.message) == peer.direct_reply
        assert await reply._core_next_event() is None
    finally:
        await client.aclose()
