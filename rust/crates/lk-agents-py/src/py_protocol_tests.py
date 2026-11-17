"""Native observations use the upstream SDK records and retain every stream event."""

from dataclasses import dataclass

import pytest
from a2a.types import a2a_pb2 as pb
from google.protobuf.json_format import MessageToDict

from livekit.agents import Agent, SessionState, TextSession, VoiceSession, persisted
from livekit.agents.a2a import AgentClient, RemoteAgent
from livekit.agents.llm import ChatContext, FunctionToolCall
from livekit.agents.storage import ContextStore
from livekit.agents.storage.session_state import RemoteAgentState, RemoteContextState, RemoteTask
from tests.fakes import FakeLLM, tool_call, until
from tests.host.a2a_peer import Peer

pytest_plugins = ["tests.host.a2a_peer"]

pytestmark = pytest.mark.unit


async def test_voice_checkpoints_remote_sdk_observations_before_presenting_the_result(
    peer: Peer,
) -> None:
    @dataclass
    class App:
        session: SessionState = persisted(default_factory=SessionState)

    def frontend(history: ChatContext) -> str | FunctionToolCall:
        if any(
            item.type == "function_call_output" and "Confirmed" in item.output
            for item in history.items
        ):
            return "Coverage confirmed."
        if any(item.type == "function_call" for item in history.items):
            return "I'm checking."
        return tool_call("coverage_expert", request="Check coverage.")

    store = ContextStore()
    app = await store.load_context(App())
    remote = RemoteAgent(peer.url, name="coverage_expert", description="Check coverage.")
    agent = Agent(
        name="desk", instructions="Help.", tools=[remote.as_tool()], llm=FakeLLM(frontend)
    )
    async with VoiceSession(context=store.bind(app), vad=None) as voice:
        await voice.start(agent)
        await voice.generate_reply(user_input="Check coverage.")
        await until(
            lambda: any(
                item.type == "message" and item.text_content == "Coverage confirmed."
                for item in voice.history.items
            ),
            "the remote result was not presented",
        )
        observed = app.session.remote_agents[peer.url]
        assert observed.tasks["remote-task"].task.status.state == pb.TASK_STATE_COMPLETED
        assert isinstance(observed.events[0].payload, pb.SendMessageRequest)
        assert all(isinstance(event.payload, pb.StreamResponse) for event in observed.events[1:])
        assert len(observed.events) == 5, "the native owner retains every observed event"
        assert peer.messages[0]["messageId"] in observed.messages
    saved = await store.load_session()
    assert saved.remote_agents[peer.url].tasks["remote-task"].task.artifacts[0].parts[0].text == (
        "Confirmed"
    )


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


async def test_authentication_state_uses_the_sdk_enum_and_keeps_the_remote_task_paused(
    peer: Peer,
) -> None:
    peer.auth_required = True
    client = AgentClient(peer.url)
    try:
        reply = await client.send("check")
        updates = [update async for update in reply]
        assert [update.state for update in updates] == [
            pb.TASK_STATE_SUBMITTED,
            pb.TASK_STATE_WORKING,
            pb.TASK_STATE_AUTH_REQUIRED,
        ]
        assert updates[-1].text == "Sign in to continue."
        assert not updates[-1].progress
    finally:
        await client.aclose()
    assert not peer.cancelled


async def test_a_remote_tool_reports_authentication_without_inventing_a_task_failure(
    peer: Peer,
) -> None:
    @dataclass
    class App:
        session: SessionState = persisted(default_factory=SessionState)

    peer.auth_required = True
    store = ContextStore()
    app = await store.load_context(App())
    agent = Agent(
        name="desk",
        instructions="Help.",
        tools=[
            RemoteAgent(peer.url, name="coverage_expert", description="Check coverage.").as_tool()
        ],
        llm=FakeLLM([tool_call("coverage_expert", request="Check"), "Please sign in to continue."]),
    )
    result = await agent.run("Check", context=store.bind(app))
    assert result.output == "Please sign in to continue."
    output = next(item for item in result.new_messages() if item.type == "function_call_output")
    assert output.is_error and output.output == "Sign in to continue."
    task = app.session.remote_agents[peer.url].tasks["remote-task"]
    assert task.task.status.state == pb.TASK_STATE_AUTH_REQUIRED
    assert not task.input_requests, "authentication is not automatically a question for the person"
    assert not task.cancel_requested and not peer.cancelled


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
