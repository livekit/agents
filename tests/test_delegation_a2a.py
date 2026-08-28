"""Reaching a delegate over A2A: the mapping, and talking to an agent that is not ours.

The point of A2A here is interop, so the server in these tests is a **foreign** one — a
hand-written ASGI app emitting spec-shaped SSE frames, using none of our code. Pointing our
client at our own server would prove only that we agree with ourselves.

Nothing binds a socket: the app runs in this loop behind a streaming httpx transport, so the
real REST binding and the real SSE framing are exercised under virtual time.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from a2a.types import a2a_pb2 as pb

from livekit.agents import (
    Agent,
    AgentSession,
    ChatContext,
    DelegationRequest,
    DelegationUpdate,
)
from livekit.agents.llm import FunctionToolCall
from livekit.agents.voice.a2a import (
    CHAT_CTX_PART_KIND,
    DELEGATION_ID_KEY,
    A2ADelegate,
    from_a2a_events,
    from_a2a_request,
    to_a2a_events,
    to_a2a_request,
)
from livekit.agents.voice.delegation import DELEGATE_TOOL_NAME

from .fake_llm import FakeLLM, FakeLLMResponse

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]


BASE_URL = "http://experts.test"


def _request(task: str = "change it to Tuesday") -> DelegationRequest:
    chat_ctx = ChatContext.empty()
    chat_ctx.add_message(role="user", content="book a flight")
    return DelegationRequest(
        task=task, chat_ctx=chat_ctx, delegation_id="call_1", metadata={"customer_id": "c-42"}
    )


async def _replay(events: list[Any]) -> AsyncIterator[Any]:
    for event in events:
        yield event


# -- the request mapping -----------------------------------------------------------------------


def test_the_request_carries_the_task_the_conversation_and_the_call_id() -> None:
    message = to_a2a_request(_request(), context_id="conv_1").message

    assert [p.text for p in message.parts if p.WhichOneof("content") == "text"] == [
        "change it to Tuesday"
    ]
    data = [p for p in message.parts if p.WhichOneof("content") == "data"]
    assert len(data) == 1
    assert data[0].metadata["kind"] == CHAT_CTX_PART_KIND
    # the server assigns the task id, so the caller's own id for the delegation rides beside it
    assert not message.task_id
    assert message.context_id == "conv_1"


def test_the_request_metadata_carries_the_application_data() -> None:
    request = to_a2a_request(_request(), context_id="conv_1")
    assert request.metadata["customer_id"] == "c-42"
    assert request.metadata[DELEGATION_ID_KEY] == "call_1"


def test_a_request_round_trips() -> None:
    original = _request()
    back = from_a2a_request(to_a2a_request(original, context_id="conv_1"))

    assert back.task == original.task
    assert back.delegation_id == original.delegation_id
    assert back.metadata == original.metadata
    assert back.chat_ctx.to_dict() == original.chat_ctx.to_dict()


def test_a_request_from_a_foreign_caller_needs_only_text() -> None:
    """What makes a plain A2A client usable against a published delegate."""
    request = pb.SendMessageRequest(
        message=pb.Message(message_id="m1", role=pb.Role.ROLE_USER, parts=[pb.Part(text="hi")])
    )
    delegation = from_a2a_request(request)

    assert delegation.task == "hi"
    assert delegation.delegation_id == "m1"
    assert delegation.chat_ctx.items == []
    assert delegation.metadata == {}


# -- the event mapping -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "event",
    [
        DelegationUpdate("cancelling the earlier booking first"),
        DelegationUpdate("Rebooked for Tuesday.", state="completed"),
        DelegationUpdate("", state="completed"),
        DelegationUpdate("the refund had already been issued", state="canceled"),
        DelegationUpdate("the booking system is down", state="failed"),
        DelegationUpdate("which booking reference?", state="input-required"),
    ],
    ids=lambda e: f"{e.state}-{'text' if e.text else 'empty'}",
)
@pytest.mark.asyncio
async def test_an_event_round_trips(event: DelegationUpdate) -> None:
    """`to_a2a_events` and `from_a2a_events` are inverses over the whole vocabulary.

    A terminal state with something to say becomes two A2A events — the artifact and the
    status — and arrives back as the one event it started as.
    """
    wire = to_a2a_events(event, task_id="t1", context_id="c1")
    assert [e async for e in from_a2a_events(_replay(wire))] == [event]


@pytest.mark.asyncio
async def test_a_working_tick_with_nothing_to_say_surfaces_as_nothing() -> None:
    wire = to_a2a_events(DelegationUpdate(), task_id="t1", context_id="c1")
    assert [e async for e in from_a2a_events(_replay(wire))] == []


def test_a_terminal_state_with_an_answer_sends_the_artifact_before_the_status() -> None:
    wire = to_a2a_events(
        DelegationUpdate("Rebooked for Tuesday.", state="completed"), task_id="t1", context_id="c1"
    )
    assert isinstance(wire[0], pb.TaskArtifactUpdateEvent)
    assert wire[0].last_chunk is True
    assert isinstance(wire[1], pb.TaskStatusUpdateEvent)
    assert wire[1].status.state == pb.TaskState.TASK_STATE_COMPLETED


# -- what a foreign agent can send that we do not -----------------------------------------------


@pytest.mark.asyncio
async def test_a_chunked_artifact_is_reassembled() -> None:
    """A2A streams a long answer in pieces; the caller sees one answer."""

    def chunk(text: str, *, append: bool, last: bool) -> pb.TaskArtifactUpdateEvent:
        return pb.TaskArtifactUpdateEvent(
            task_id="t1",
            artifact=pb.Artifact(artifact_id="a1", parts=[pb.Part(text=text)]),
            append=append,
            last_chunk=last,
        )

    wire = [
        chunk("Rebooked ", append=False, last=False),
        chunk("for Tuesday, ", append=True, last=False),
        chunk("ref QX7.", append=True, last=True),
        pb.TaskStatusUpdateEvent(
            task_id="t1", status=pb.TaskStatus(state=pb.TaskState.TASK_STATE_COMPLETED)
        ),
    ]
    assert [e async for e in from_a2a_events(_replay(wire))] == [
        DelegationUpdate("Rebooked for Tuesday, ref QX7.", state="completed")
    ]


@pytest.mark.asyncio
async def test_two_artifacts_do_not_run_into_each_other() -> None:
    """An agent may stream several artifacts at once, and `append` names the one it belongs
    to. Accumulating on one buffer concatenates a summary's chunks onto the answer's."""

    def chunk(
        aid: str, name: str, text: str, *, append: bool = False
    ) -> pb.TaskArtifactUpdateEvent:
        return pb.TaskArtifactUpdateEvent(
            task_id="t1",
            artifact=pb.Artifact(artifact_id=aid, name=name, parts=[pb.Part(text=text)]),
            append=append,
        )

    wire = [
        chunk("a1", "summary", "the trip was "),
        chunk("a2", "answer", "Rebooked "),
        chunk("a1", "summary", "rearranged", append=True),
        chunk("a2", "answer", "for Tuesday.", append=True),
        pb.TaskStatusUpdateEvent(
            task_id="t1", status=pb.TaskStatus(state=pb.TaskState.TASK_STATE_COMPLETED)
        ),
    ]
    # the artifact named `answer` is the answer; the summary is not spliced into it
    assert [e async for e in from_a2a_events(_replay(wire))] == [
        DelegationUpdate("Rebooked for Tuesday.", state="completed")
    ]


@pytest.mark.asyncio
async def test_unnamed_artifacts_join_in_arrival_order() -> None:
    def art(aid: str, text: str) -> pb.TaskArtifactUpdateEvent:
        return pb.TaskArtifactUpdateEvent(
            task_id="t1", artifact=pb.Artifact(artifact_id=aid, parts=[pb.Part(text=text)])
        )

    wire = [
        art("a1", "first"),
        art("a2", "second"),
        pb.TaskStatusUpdateEvent(
            task_id="t1", status=pb.TaskStatus(state=pb.TaskState.TASK_STATE_COMPLETED)
        ),
    ]
    assert [e async for e in from_a2a_events(_replay(wire))] == [
        DelegationUpdate("first\nsecond", state="completed")
    ]


@pytest.mark.asyncio
async def test_an_agent_that_answers_without_opening_a_task() -> None:
    """A2A allows a bare Message reply. One output, so it is the answer."""
    wire = [pb.Message(message_id="m1", role=pb.Role.ROLE_AGENT, parts=[pb.Part(text="240 USD")])]
    assert [e async for e in from_a2a_events(_replay(wire))] == [
        DelegationUpdate("240 USD", state="completed")
    ]


@pytest.mark.asyncio
async def test_states_we_do_not_model_map_onto_ones_we_do() -> None:
    for state, expected in (
        (pb.TaskState.TASK_STATE_REJECTED, "failed"),
        (pb.TaskState.TASK_STATE_AUTH_REQUIRED, "input-required"),
    ):
        wire = [pb.TaskStatusUpdateEvent(task_id="t1", status=pb.TaskStatus(state=state))]
        assert [e async for e in from_a2a_events(_replay(wire))] == [
            DelegationUpdate("", state=expected)
        ], state


# -- a foreign A2A server, over the real wire ---------------------------------------------------


class ForeignAgent:
    """A minimal A2A endpoint written from the spec, using none of our code.

    It serves an agent card and answers ``POST /message:stream`` with SSE frames whose JSON is
    hand-built, so anything our client understands here it would understand from any other
    implementation of the protocol.
    """

    def __init__(self, frames: list[dict[str, Any]]) -> None:
        self._frames = frames
        self.seen: list[dict[str, Any]] = []

    def _card(self) -> dict[str, Any]:
        return {
            "name": "fare-desk",
            "description": "flights",
            "version": "1.0.0",
            "capabilities": {"streaming": True},
            "defaultInputModes": ["text/plain"],
            "defaultOutputModes": ["text/plain"],
            "skills": [{"id": "delegate", "name": "delegate", "description": "x", "tags": ["x"]}],
            "supportedInterfaces": [
                {"url": BASE_URL, "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"}
            ],
        }

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        assert scope["type"] == "http"

        if scope["path"].endswith("/.well-known/agent-card.json"):
            await _respond(send, 200, b"application/json", json.dumps(self._card()).encode())
            return

        if scope["path"] != "/message:stream":
            await _respond(send, 404, b"text/plain", b"not found")
            return

        body = b""
        while True:
            message = await receive()
            body += message.get("body", b"")
            if not message.get("more_body", False):
                break
        self.seen.append(json.loads(body))

        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/event-stream")],
            }
        )
        for frame in self._frames:
            await send(
                {
                    "type": "http.response.body",
                    "body": f"data: {json.dumps(frame)}\n\n".encode(),
                    "more_body": True,
                }
            )
            await asyncio.sleep(0.01)  # a real endpoint does not emit everything at once
        await send({"type": "http.response.body", "body": b"", "more_body": False})


async def _respond(send: Any, status: int, content_type: bytes, body: bytes) -> None:
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [(b"content-type", content_type)],
        }
    )
    await send({"type": "http.response.body", "body": body, "more_body": False})


class _ASGIStream(httpx.AsyncByteStream):
    def __init__(self, chunks: asyncio.Queue[bytes | None], app_task: asyncio.Task[None]) -> None:
        self._chunks = chunks
        self._app_task = app_task

    async def __aiter__(self) -> AsyncIterator[bytes]:
        while (chunk := await self._chunks.get()) is not None:
            yield chunk

    async def aclose(self) -> None:
        self._app_task.cancel()
        try:
            await self._app_task
        except asyncio.CancelledError:
            pass


class StreamingASGITransport(httpx.AsyncBaseTransport):
    """Runs an ASGI app in this loop and yields its body as it is produced.

    httpx's own ``ASGITransport`` joins the whole body before returning, which would collapse
    the SSE stream into one delivery and hide whether progress really arrives before the
    answer.
    """

    def __init__(self, app: Any) -> None:
        self._app = app
        self._running: set[asyncio.Task[None]] = set()

    async def aclose(self) -> None:
        for task in list(self._running):
            task.cancel()
        await asyncio.gather(*self._running, return_exceptions=True)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        body = await request.aread()
        chunks: asyncio.Queue[bytes | None] = asyncio.Queue()
        started: asyncio.Future[tuple[int, list[tuple[bytes, bytes]]]] = (
            asyncio.get_running_loop().create_future()
        )
        sent = False
        done = asyncio.Event()

        async def receive() -> dict[str, Any]:
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            await done.wait()
            return {"type": "http.disconnect"}

        async def send(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                started.set_result((message["status"], message.get("headers", [])))
            elif message["type"] == "http.response.body":
                if chunk := message.get("body", b""):
                    chunks.put_nowait(chunk)
                if not message.get("more_body", False):
                    chunks.put_nowait(None)

        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": request.method,
            "headers": [(k.lower(), v) for k, v in request.headers.raw],
            "scheme": request.url.scheme,
            "path": request.url.path,
            "raw_path": request.url.raw_path.split(b"?")[0],
            "query_string": request.url.query,
            "server": (request.url.host, request.url.port),
            "client": ("127.0.0.1", 1234),
            "root_path": "",
        }

        async def run() -> None:
            try:
                await self._app(scope, receive, send)
            except BaseException as exc:
                if not started.done():
                    started.set_exception(exc)
                raise
            finally:
                done.set()
                chunks.put_nowait(None)

        app_task = asyncio.create_task(run())
        self._running.add(app_task)
        app_task.add_done_callback(self._running.discard)
        status, headers = await started
        return httpx.Response(status, headers=headers, stream=_ASGIStream(chunks, app_task))


def _status_frame(state: str, text: str = "") -> dict[str, Any]:
    status: dict[str, Any] = {"state": state}
    if text:
        status["message"] = {"messageId": "m", "role": "ROLE_AGENT", "parts": [{"text": text}]}
    return {"statusUpdate": {"taskId": "t1", "contextId": "c1", "status": status}}


def _artifact_frame(text: str) -> dict[str, Any]:
    return {
        "artifactUpdate": {
            "taskId": "t1",
            "contextId": "c1",
            "artifact": {"artifactId": "a1", "name": "answer", "parts": [{"text": text}]},
            "lastChunk": True,
        }
    }


_SUBMITTED = {"task": {"id": "t1", "contextId": "c1", "status": {"state": "TASK_STATE_SUBMITTED"}}}


@contextlib.asynccontextmanager
async def _connected(agent: ForeignAgent) -> AsyncIterator[A2ADelegate]:
    transport = StreamingASGITransport(agent)
    http = httpx.AsyncClient(transport=transport, timeout=None)
    delegate = A2ADelegate(BASE_URL, httpx_client=http)
    try:
        yield delegate
    finally:
        await asyncio.wait_for(delegate.aclose(), timeout=10.0)
        await asyncio.wait_for(http.aclose(), timeout=10.0)
        await asyncio.wait_for(transport.aclose(), timeout=10.0)


async def _collect(delegate: A2ADelegate, request: DelegationRequest) -> list[DelegationUpdate]:
    async with delegate(request) as stream:
        return [ev async for ev in stream]


def _answers(session: AgentSession) -> list[str]:
    """What the `delegate` calls returned, which each records under `<call_id>_final`."""
    return [
        item.output
        for item in session.history.items
        if item.type == "function_call_output"
        and item.name == DELEGATE_TOOL_NAME
        and item.call_id.endswith("_final")
    ]


async def _wait_for(predicate: Any, *, timeout: float = 10.0) -> None:
    async def _poll() -> None:
        while not predicate():
            await asyncio.sleep(0.05)

    await asyncio.wait_for(_poll(), timeout=timeout)


@pytest.mark.asyncio
async def test_a_foreign_agent_answers_a_delegation() -> None:
    agent = ForeignAgent(
        [
            _SUBMITTED,
            _status_frame("TASK_STATE_WORKING"),
            _status_frame("TASK_STATE_WORKING", "cancelling the earlier booking first"),
            _artifact_frame("Rebooked for Tuesday, ref QX7."),
            _status_frame("TASK_STATE_COMPLETED"),
        ]
    )
    async with _connected(agent) as delegate:
        events = await asyncio.wait_for(_collect(delegate, _request()), timeout=10.0)

    assert events == [
        DelegationUpdate("cancelling the earlier booking first"),
        DelegationUpdate("Rebooked for Tuesday, ref QX7.", state="completed"),
    ]

    # and it received a request it could act on without knowing anything about this framework
    assert len(agent.seen) == 1
    parts = agent.seen[0]["message"]["parts"]
    assert parts[0]["text"] == "change it to Tuesday"
    assert agent.seen[0]["metadata"][DELEGATION_ID_KEY] == "call_1"


@pytest.mark.asyncio
async def test_a_foreign_agent_that_fails_the_task() -> None:
    agent = ForeignAgent(
        [_SUBMITTED, _status_frame("TASK_STATE_FAILED", "the booking system is down")]
    )
    async with _connected(agent) as delegate:
        events = await asyncio.wait_for(_collect(delegate, _request()), timeout=10.0)

    assert events == [DelegationUpdate("the booking system is down", state="failed")]


@pytest.mark.asyncio
async def test_a_foreign_agent_that_asks_for_more_input() -> None:
    """A2A treats `input-required` as interrupted, not terminal: the task stays alive and the
    caller is meant to resume it with the same taskId. We end the delegation instead and tell
    the conversation model what is missing, so it asks the user. See DELEGATION_DESIGN.md."""
    agent = ForeignAgent(
        [
            _SUBMITTED,
            _status_frame("TASK_STATE_WORKING", "looking you up"),
            _status_frame("TASK_STATE_INPUT_REQUIRED", "which booking reference?"),
        ]
    )
    async with _connected(agent) as delegate:
        events = await asyncio.wait_for(_collect(delegate, _request()), timeout=10.0)

    assert events == [
        DelegationUpdate("looking you up"),
        DelegationUpdate("which booking reference?", state="input-required"),
    ]


@pytest.mark.asyncio
async def test_asking_for_more_input_reaches_the_conversation_as_an_error() -> None:
    """End to end, so the whole path is covered and not only the mapping."""
    agent = ForeignAgent(
        [_SUBMITTED, _status_frame("TASK_STATE_INPUT_REQUIRED", "which booking reference?")]
    )
    async with _connected(agent) as delegate:
        session = AgentSession(
            llm=FakeLLM(
                fake_responses=[
                    FakeLLMResponse(
                        input="how much",
                        content="one sec",
                        ttft=0.0,
                        duration=0.0,
                        tool_calls=[
                            FunctionToolCall(
                                type="function",
                                name=DELEGATE_TOOL_NAME,
                                arguments='{"task": "what is the fare"}',
                                call_id="d_1",
                            )
                        ],
                    )
                ]
            ),
            delegate=delegate,
        )
        await session.start(agent=Agent(instructions="you are the airline's voice"))
        await asyncio.wait_for(session.run(user_input="how much"), timeout=10.0)
        await _wait_for(
            lambda: any(
                item.type == "function_call_output" and item.is_error
                for item in session.history.items
            )
        )
        await asyncio.wait_for(session.aclose(), timeout=10.0)

    errors = " ".join(
        item.output
        for item in session.history.items
        if item.type == "function_call_output" and item.is_error
    )
    assert "which booking reference?" in errors


@pytest.mark.asyncio
async def test_a_foreign_agent_answers_the_conversation() -> None:
    """End to end: the voice session delegates, and a foreign expert's answer comes back."""
    agent = ForeignAgent(
        [
            _SUBMITTED,
            _status_frame("TASK_STATE_WORKING", "checking the fare"),
            _artifact_frame("It is 240 USD."),
            _status_frame("TASK_STATE_COMPLETED"),
        ]
    )
    async with _connected(agent) as delegate:
        session = AgentSession(
            llm=FakeLLM(
                fake_responses=[
                    FakeLLMResponse(
                        input="how much",
                        content="one sec",
                        ttft=0.0,
                        duration=0.0,
                        tool_calls=[
                            FunctionToolCall(
                                type="function",
                                name=DELEGATE_TOOL_NAME,
                                arguments='{"task": "what is the fare"}',
                                call_id="d_1",
                            )
                        ],
                    )
                ]
            ),
            delegate=delegate,
        )
        await session.start(agent=Agent(instructions="you are the airline's voice"))
        await asyncio.wait_for(session.run(user_input="how much"), timeout=10.0)
        # the delegate tool releases the turn, so the run returns before the answer arrives
        await _wait_for(lambda: _answers(session) == ["It is 240 USD."])
        await asyncio.wait_for(session.aclose(), timeout=10.0)

    outputs = [item.output for item in session.history.items if item.type == "function_call_output"]
    assert any("checking the fare" in output for output in outputs)


# -- the boundary -------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_local_delegation_never_touches_a2a() -> None:
    """A2A is a transport. A delegate that runs here is reached directly, so a local
    delegation must not import anything from the protocol."""
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        """
        import asyncio, sys
        sys.path.insert(0, ".")
        from livekit.agents import Agent, AgentSession
        from livekit.agents.llm import FunctionToolCall
        from livekit.agents.voice.delegation import DELEGATE_TOOL_NAME, AgentDelegate
        from tests.fake_llm import FakeLLM, FakeLLMResponse

        def expert():
            llm = FakeLLM(fake_responses=[
                FakeLLMResponse(input="what is the fare", content="It is 240 USD.",
                                ttft=0.0, duration=0.0),
            ])
            return Agent(instructions="expert", llm=llm)

        async def main():
            session = AgentSession(
                llm=FakeLLM(fake_responses=[
                    FakeLLMResponse(
                        input="how much", content="one sec", ttft=0.0, duration=0.0,
                        tool_calls=[FunctionToolCall(
                            type="function", name=DELEGATE_TOOL_NAME,
                            arguments='{"task": "what is the fare"}', call_id="d1")],
                    ),
                ]),
                delegate=AgentDelegate(expert),
            )
            await session.start(agent=Agent(instructions="voice"))
            await asyncio.wait_for(session.run(user_input="how much"), timeout=30)

            def answered():
                return [i.output for i in session.history.items
                        if i.type == "function_call_output"
                        and i.name == DELEGATE_TOOL_NAME and i.call_id.endswith("_final")]

            async def wait():
                while answered() != ["It is 240 USD."]:
                    await asyncio.sleep(0.05)

            # the delegate tool releases the turn, so the run returns before the answer lands
            await asyncio.wait_for(wait(), timeout=30)
            await asyncio.wait_for(session.aclose(), timeout=30)

        asyncio.run(main())
        leaked = sorted(m for m in sys.modules if m == "a2a" or m.startswith("a2a."))
        assert not leaked, leaked
        print("clean")
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=180
    )
    assert out.returncode == 0, out.stderr[-3000:]
    assert "clean" in out.stdout
