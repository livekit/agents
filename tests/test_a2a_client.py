"""A2AClient against a foreign endpoint written from the spec, over the real wire."""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from livekit.agents import a2a
from livekit.agents.a2a.extension import EXTENSION_URI, KIND, REASON, as_dict, pb

pytestmark = pytest.mark.unit

BASE_URL = "http://foreign.test"


class ForeignAgent:
    """A minimal A2A endpoint written from the spec, using none of our code.

    It serves an agent card and answers ``POST /message:stream`` with SSE frames whose JSON
    is hand-built, so anything our client understands here it would understand from any
    other implementation of the protocol.
    """

    def __init__(self, frames: list[dict[str, Any]], *, offers_extension: bool = False) -> None:
        self._frames = frames
        self._offers_extension = offers_extension
        self.requests: list[dict[str, Any]] = []
        self.headers: list[dict[str, str]] = []
        self.cancels: list[dict[str, Any]] = []
        self.hold: asyncio.Event | None = None
        """When set, a request is recorded and then held before its first frame."""

    def _card(self) -> dict[str, Any]:
        capabilities: dict[str, Any] = {"streaming": True}
        if self._offers_extension:
            capabilities["extensions"] = [{"uri": EXTENSION_URI, "required": False}]
        return {
            "name": "fare-desk",
            "description": "flights",
            "version": "1.0.0",
            "capabilities": capabilities,
            "defaultInputModes": ["text/plain"],
            "defaultOutputModes": ["text/plain"],
            "skills": [{"id": "delegate", "name": "delegate", "description": "x", "tags": ["x"]}],
            "supportedInterfaces": [
                {"url": BASE_URL, "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"}
            ],
        }

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        assert scope["type"] == "http"
        path = scope["path"]
        headers = {k.decode(): v.decode() for k, v in scope["headers"]}

        if path.endswith("/.well-known/agent-card.json"):
            await _respond(send, 200, json.dumps(self._card()).encode())
            return

        body = b""
        while True:
            message = await receive()
            body += message.get("body", b"")
            if not message.get("more_body", False):
                break

        if path.endswith(":cancel"):
            self.cancels.append({"path": path, "body": json.loads(body or b"{}")})
            await _respond(send, 200, json.dumps({"id": "t1", "contextId": "c1"}).encode())
            return

        if not path.endswith("/message:stream"):
            await _respond(send, 404, b"not found", content_type=b"text/plain")
            return

        self.requests.append(json.loads(body))
        self.headers.append(headers)
        if self.hold is not None:
            await self.hold.wait()

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
            await asyncio.sleep(0)  # a real endpoint does not emit everything at once
        await send({"type": "http.response.body", "body": b"", "more_body": False})


async def _respond(
    send: Any, status: int, body: bytes, *, content_type: bytes = b"application/json"
) -> None:
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
        with contextlib.suppress(asyncio.CancelledError):
            await self._app_task


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


def _status_frame(state: str, text: str = "", task_id: str = "t1") -> dict[str, Any]:
    status: dict[str, Any] = {"state": state}
    if text:
        status["message"] = {"messageId": "m", "role": "ROLE_AGENT", "parts": [{"text": text}]}
    return {"statusUpdate": {"taskId": task_id, "contextId": "c1", "status": status}}


def _artifact_frame(text: str, task_id: str = "t1") -> dict[str, Any]:
    return {
        "artifactUpdate": {
            "taskId": task_id,
            "contextId": "c1",
            "artifact": {"artifactId": "a1", "name": "answer", "parts": [{"text": text}]},
            "lastChunk": True,
        }
    }


def _submitted(task_id: str = "t1") -> dict[str, Any]:
    return {"task": {"id": task_id, "contextId": "c1", "status": {"state": "TASK_STATE_SUBMITTED"}}}


ANSWERED = [
    _submitted(),
    _status_frame("TASK_STATE_WORKING", "Checking Tuesday."),
    _artifact_frame("The change fee is $75."),
    _status_frame("TASK_STATE_COMPLETED"),
]


@contextlib.asynccontextmanager
async def _client(agent: ForeignAgent) -> AsyncIterator[a2a.A2AClient]:
    transport = StreamingASGITransport(agent)
    http = httpx.AsyncClient(transport=transport, timeout=None)
    client = a2a.A2AClient(BASE_URL, httpx_client=http)
    try:
        yield client
    finally:
        await asyncio.wait_for(client.aclose(), timeout=10.0)
        await asyncio.wait_for(http.aclose(), timeout=10.0)
        await asyncio.wait_for(transport.aclose(), timeout=10.0)


def _asks(agent: ForeignAgent) -> list[dict[str, Any]]:
    """What the client asked for, which is every message but the goodbye."""
    return [
        request
        for request in agent.requests
        if (request["message"].get("metadata") or {}).get(KIND) != "close"
    ]


async def _collect(client: a2a.A2AClient, task_input: a2a.TaskInput) -> list[a2a.TaskUpdate]:
    async with client.send(task_input) as stream:
        return [update async for update in stream]


async def test_a_foreign_endpoint_answers_over_the_real_wire() -> None:
    agent = ForeignAgent(ANSWERED)
    async with _client(agent) as client:
        updates = await _collect(client, a2a.TaskInput(instruction="find the change fee"))

    assert [(u.state, u.text) for u in updates] == [
        ("working", "Checking Tuesday."),
        ("completed", "The change fee is $75."),
    ]
    # the card offered no extension, so the client did not ask for one
    assert client.extension_active is False
    assert "a2a-extensions" not in agent.headers[0]


async def test_the_extension_is_activated_only_where_it_is_offered() -> None:
    agent = ForeignAgent(ANSWERED, offers_extension=True)
    async with _client(agent) as client:
        await _collect(client, a2a.TaskInput(instruction="find it"))

    assert client.extension_active is True
    assert agent.headers[0]["a2a-extensions"] == EXTENSION_URI


async def test_a_supplied_http_client_keeps_its_own_headers() -> None:
    """The header goes per request: a caller's client reaches other endpoints too."""
    agent = ForeignAgent(ANSWERED, offers_extension=True)
    transport = StreamingASGITransport(agent)
    http = httpx.AsyncClient(transport=transport, timeout=None)
    client = a2a.A2AClient(BASE_URL, httpx_client=http)
    try:
        await _collect(client, a2a.TaskInput(instruction="find it"))
    finally:
        await asyncio.wait_for(client.aclose(), timeout=10.0)
        await asyncio.wait_for(http.aclose(), timeout=10.0)
        await asyncio.wait_for(transport.aclose(), timeout=10.0)

    assert agent.headers[0]["a2a-extensions"] == EXTENSION_URI
    assert "a2a-extensions" not in http.headers


async def test_the_request_carries_the_context_and_the_delegation_tag() -> None:
    agent = ForeignAgent(ANSWERED, offers_extension=True)
    async with _client(agent) as client:
        await _collect(client, a2a.TaskInput(instruction="find it"))
        context_id = client.context_id

    (sent,) = _asks(agent)
    assert sent["message"]["contextId"] == context_id
    assert sent["message"]["metadata"][KIND] == "delegation"


async def test_the_task_id_is_taken_from_the_first_event() -> None:
    agent = ForeignAgent(ANSWERED)
    async with _client(agent) as client, client.send(a2a.TaskInput(text="hi")) as stream:
        assert stream.task_id == ""  # nothing sent yet
        await stream.__anext__()
        assert stream.task_id == "t1"


async def test_a_question_is_referenced_by_the_next_send() -> None:
    frames = [
        _submitted(),
        _status_frame("TASK_STATE_INPUT_REQUIRED", "Which Tuesday flight?"),
    ]
    agent = ForeignAgent(frames + ANSWERED)
    async with _client(agent) as client:
        first = await _collect(client, a2a.TaskInput(instruction="change it"))
        assert [u.state for u in first] == ["input-required"]

        agent._frames = ANSWERED
        await _collect(client, a2a.TaskInput(text="the 9am one"))

    assert _asks(agent)[0]["message"].get("referenceTaskIds") is None
    # the answer says what it might be answering; a resume it is not
    assert _asks(agent)[1]["message"]["referenceTaskIds"] == ["t1"]


async def test_a_question_is_referenced_once() -> None:
    agent = ForeignAgent([_submitted(), _status_frame("TASK_STATE_INPUT_REQUIRED", "Which?")])
    async with _client(agent) as client:
        await _collect(client, a2a.TaskInput(instruction="change it"))
        agent._frames = ANSWERED
        await _collect(client, a2a.TaskInput(text="the 9am one"))
        await _collect(client, a2a.TaskInput(text="thanks"))

    assert _asks(agent)[1]["message"]["referenceTaskIds"] == ["t1"]
    assert _asks(agent)[2]["message"].get("referenceTaskIds") is None


async def _settle() -> None:
    for _ in range(20):
        await asyncio.sleep(0)


async def test_one_unacknowledged_send_per_context() -> None:
    """Two HTTP requests carry no ordering between them, so the second waits for the first
    task to come back. The endpoint here never acknowledges until released."""
    agent = ForeignAgent(ANSWERED)
    agent.hold = asyncio.Event()

    async with _client(agent) as client:
        await client._connect()  # resolving the card is not what we are timing

        first = client.send(a2a.TaskInput(text="one"))
        second = client.send(a2a.TaskInput(text="two"))
        try:
            reading_first = asyncio.create_task(first.__anext__())
            await _settle()
            assert [r["message"]["parts"][0]["text"] for r in _asks(agent)] == ["one"]

            reading_second = asyncio.create_task(second.__anext__())
            await _settle()
            # the first is still unacknowledged, so the second has not reached the wire
            assert [r["message"]["parts"][0]["text"] for r in _asks(agent)] == ["one"]

            agent.hold.set()
            await reading_first
            await reading_second
            assert [r["message"]["parts"][0]["text"] for r in _asks(agent)] == ["one", "two"]
        finally:
            await first.aclose()
            await second.aclose()


async def test_a_send_that_never_reaches_the_wire_releases_the_turn() -> None:
    """A stream closed before it is read must not strand the context."""
    agent = ForeignAgent(ANSWERED)
    async with _client(agent) as client:
        abandoned = client.send(a2a.TaskInput(text="one"))
        await abandoned.aclose()
        updates = await _collect(client, a2a.TaskInput(text="two"))

    assert [u.state for u in updates] == ["working", "completed"]
    assert [r["message"]["parts"][0]["text"] for r in _asks(agent)] == ["two"]


async def test_cancel_names_the_task_and_says_why() -> None:
    agent = ForeignAgent(ANSWERED)
    async with _client(agent) as client, client.send(a2a.TaskInput(text="hi")) as stream:
        await stream.__anext__()
        await stream.cancel("user_interrupted")

    (cancel,) = agent.cancels
    assert cancel["path"].endswith("/tasks/t1:cancel")
    assert cancel["body"]["metadata"][REASON] == "user_interrupted"


async def test_cancelling_before_the_task_is_known_does_nothing() -> None:
    agent = ForeignAgent(ANSWERED)
    async with _client(agent) as client, client.send(a2a.TaskInput(text="hi")) as stream:
        await stream.cancel("too early")
    assert agent.cancels == []


async def test_closing_says_goodbye_once() -> None:
    agent = ForeignAgent(ANSWERED)
    async with _client(agent) as client:
        await _collect(client, a2a.TaskInput(text="hi"))

    kinds = [(r["message"].get("metadata") or {}).get(KIND) for r in agent.requests]
    assert kinds == [None, "close"]


async def test_a_context_nothing_was_sent_on_says_nothing() -> None:
    """Closing a client that never connected must not open one to say goodbye."""
    agent = ForeignAgent(ANSWERED)
    async with _client(agent):
        pass
    assert agent.requests == []


async def test_a_vanilla_reply_without_a_task_is_the_answer() -> None:
    """A server that answers with a Message and opens no task has one output."""
    frames = [
        {
            "message": {
                "messageId": "m1",
                "role": "ROLE_AGENT",
                "parts": [{"text": "the fee is $75"}],
            }
        }
    ]
    agent = ForeignAgent(frames)
    async with _client(agent) as client:
        updates = await _collect(client, a2a.TaskInput(text="what is the fee?"))

    assert [(u.state, u.text) for u in updates] == [("completed", "the fee is $75")]


async def test_the_history_rides_along_and_comes_back_intact() -> None:
    agent = ForeignAgent(ANSWERED, offers_extension=True)
    chat_ctx = a2a.TaskInput(text="x").chat_ctx
    chat_ctx.add_message(role="user", content="change my Monday flight", id="m1")

    async with _client(agent) as client:
        await _collect(client, a2a.TaskInput(instruction="find it", chat_ctx=chat_ctx))

    (sent,) = _asks(agent)
    request = pb.SendMessageRequest()
    from google.protobuf import json_format

    json_format.ParseDict(sent, request)
    received = a2a.from_a2a_request(request)
    assert [item.id for item in received.chat_ctx.items] == ["m1"]
    assert as_dict(request.message.metadata)[KIND] == "delegation"
