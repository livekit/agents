"""A behavior-test driver for the text door: it reaches the system only through public surfaces.

``door(handler, inbox=...)`` serves ``a2a_session`` in process on the fake agentstore and speaks
JSON-RPC to it; ``Door.send`` streams one message and reads its task back with ``tasks/get``, and
``get_task``, ``list_tasks`` and ``cancel`` are the A2A methods of the same names. ``expert(handler)``
is a second door on the same store with an ``A2ADelegate`` pointing at it. ``ScriptedLLM`` answers
by order or by a predicate on the request and keeps every request.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest
from a2a.client import ClientConfig, ClientFactory
from a2a.client.card_resolver import A2ACardResolver
from a2a.types import a2a_pb2 as pb
from a2a.utils.constants import TransportProtocol

from livekit.agents import AgentServer
from livekit.agents.a2a import A2AEndpoint, A2ASessionContext, Credentials
from livekit.agents.a2a.codec import metadata_from_wire
from livekit.agents.delegation import A2ADelegate
from livekit.agents.llm import ChatContext, FunctionToolCall, LLMStream, Tool
from livekit.agents.storage import AgentDBStorage
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

from .fake_llm import FakeLLM, FakeLLMStream

PROJECT_ENV = (
    "LIVEKIT_URL",
    "LIVEKIT_AGENT_NAME",
    "LIVEKIT_AGENT_NAME_OVERRIDE",
    "LIVEKIT_AGENT_DEPLOYMENT",
    "LIVEKIT_AGENT_VERSION",
)


@pytest.fixture(autouse=True)
def no_project_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """These tests build their servers in memory and address them as ``agents.test``; a
    developer's exported demo environment (``LIVEKIT_URL`` shapes the card's URLs, the agent
    variables its name and version) must not reach them."""
    for name in PROJECT_ENV:
        monkeypatch.delenv(name, raising=False)


AGENT = "front-desk"
API_KEY = "APItest"
API_SECRET = "secret-secret-secret-secret-secret"
CREDENTIALS = Credentials(API_KEY, API_SECRET)
ENDPOINT = "desk"


async def echo(ctx: A2ASessionContext[Any]) -> str:
    await ctx.update("looking")
    return "reply to: " + ctx.message.text_content


def make_server(
    storage: AgentDBStorage,
    handler: Callable[[A2ASessionContext[Any]], Awaitable[Any]] = echo,
    **kw: Any,
) -> tuple[AgentServer, A2AEndpoint[Any]]:
    server = AgentServer(storage=storage, agent_name=AGENT, api_key=API_KEY, api_secret=API_SECRET)
    kw.setdefault("description", "Answers front desk questions.")
    endpoint = server.a2a_session(endpoint=ENDPOINT, **kw)(handler)
    return server, endpoint


def http_client(server: AgentServer, **headers: str) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.http.app),
        base_url="http://agents.test",
        headers=headers,
        timeout=30.0,
    )


async def stock_client(http: httpx.AsyncClient, *, extensions: str | None = None) -> Any:
    """A vanilla a2a-sdk client over JSON-RPC; ``extensions`` is the header it sends."""
    card = await A2ACardResolver(http, f"http://agents.test/{ENDPOINT}").get_agent_card()
    client = ClientFactory(
        ClientConfig(
            httpx_client=http,
            streaming=True,
            supported_protocol_bindings=[TransportProtocol.JSONRPC],
            accepted_output_modes=["text/plain"],
        )
    ).create(card)
    if extensions:
        from a2a.client import ClientCallContext

        client._lk_context = ClientCallContext(service_parameters={"A2A-Extensions": extensions})
    else:
        client._lk_context = None
    return client


def user_message(
    text: str, message_id: str, *, context_id: str | None = None, **metadata: Any
) -> pb.Message:
    msg = pb.Message(message_id=message_id, role=pb.Role.ROLE_USER, parts=[pb.Part(text=text)])
    if context_id:
        msg.context_id = context_id
    if metadata:
        from livekit.agents.a2a.extension import struct, widen

        msg.metadata.CopyFrom(struct({widen(k): v for k, v in metadata.items()}))
    return msg


async def send(client: Any, message: pb.Message) -> list[Any]:
    request = pb.SendMessageRequest(message=message)
    return [e async for e in client.send_message(request, context=client._lk_context)]


def payloads(events: list[Any]) -> list[Any]:
    return [getattr(e, e.WhichOneof("payload")) for e in events]


async def rpc(
    http: httpx.AsyncClient, method: str, params: Mapping[str, Any], **headers: str
) -> httpx.Response:
    return await http.post(
        f"/{ENDPOINT}/v1",
        json={"jsonrpc": "2.0", "id": method, "method": method, "params": dict(params)},
        headers=headers,
    )


def sse_results(response: httpx.Response) -> list[dict[str, Any]]:
    out = []
    for block in response.text.split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data: "):
                out.append(json.loads(line[len("data: ") :]))
    return out


def wire(
    text: str, message_id: str, *, context_id: str | None = None, **metadata: Any
) -> dict[str, Any]:
    from livekit.agents.a2a.extension import widen

    msg: dict[str, Any] = {"messageId": message_id, "role": "ROLE_USER", "parts": [{"text": text}]}
    if context_id:
        msg["contextId"] = context_id
    if metadata:
        msg["metadata"] = {widen(k): v for k, v in metadata.items()}
    return msg


def state_of(result: dict[str, Any]) -> str | None:
    update = result.get("statusUpdate")
    return update["status"]["state"] if update else None


# ---------------------------------------------------------------------------------------------
# the door
# ---------------------------------------------------------------------------------------------


@dataclass
class Sent:
    """One message's answer: the results its stream carried, and its task read back after."""

    frames: list[dict[str, Any]]
    task: dict[str, Any]

    @property
    def state(self) -> str:
        return str(self.task["status"]["state"]).removeprefix("TASK_STATE_").lower()

    @property
    def status_text(self) -> str | None:
        """The text of the task's last status message: a question, a reason, progress."""
        parts = self.task["status"].get("message", {}).get("parts", [])
        return "".join(p.get("text", "") for p in parts) or None

    @property
    def answer(self) -> str | None:
        """The text of the task's last artifact."""
        artifacts = self.task.get("artifacts", [])
        return "".join(p.get("text", "") for p in artifacts[-1]["parts"]) if artifacts else None

    @property
    def metadata(self) -> dict[str, Any]:
        """The task's metadata, short keys."""
        return dict(metadata_from_wire(self.task.get("metadata")))


class Door:
    """One ``a2a_session`` served in process, and a JSON-RPC client of it."""

    def __init__(self, storage: AgentDBStorage, handler: Any, **kw: Any) -> None:
        self.storage = storage
        self.server, self.endpoint = make_server(storage, handler, **kw)
        self.http = http_client(self.server)

    async def thread(self) -> str:
        """A new thread from the threads service, for messages that name it from the start."""
        return (await self.storage.threads.create_thread()).thread_id

    async def send(
        self,
        text: str,
        *,
        context_id: str | None = None,
        task_id: str | None = None,
        message_id: str | None = None,
        **metadata: Any,
    ) -> Sent:
        """``message/stream`` to the end, then ``tasks/get`` of the task it answered on."""
        message = wire(text, message_id or uuid.uuid4().hex, context_id=context_id, **metadata)
        if task_id:
            message["taskId"] = task_id
        response = await rpc(self.http, "message/stream", {"message": message})
        frames = [r["result"] for r in sse_results(response) if "result" in r]
        first = frames[0]["task"]
        return Sent(frames, await self.get_task(first["id"], first["contextId"]))

    async def get_task(self, task_id: str, context_id: str) -> dict[str, Any]:
        r = await rpc(self.http, "tasks/get", {"id": task_id, "contextId": context_id})
        return dict(r.json()["result"])

    async def list_tasks(self, context_id: str) -> list[dict[str, Any]]:
        r = await rpc(self.http, "tasks/list", {"contextId": context_id})
        # an empty list is left out of the response
        return list(r.json()["result"].get("tasks", []))

    async def cancel(self, task_id: str, context_id: str, **metadata: Any) -> dict[str, Any]:
        """``tasks/cancel``: the JSON-RPC response, its ``result`` or its ``error``."""
        params: dict[str, Any] = {"id": task_id, "contextId": context_id}
        if metadata:
            params["metadata"] = wire("", "", **metadata)["metadata"]
        return dict((await rpc(self.http, "tasks/cancel", params)).json())


EXPERT = "expert"


@pytest.fixture
async def door(agentdb_storage: AgentDBStorage) -> AsyncIterator[Callable[..., Door]]:
    """``door(handler, inbox=...)``: a door on the fake store; the keywords go to
    ``a2a_session``."""
    doors: list[Door] = []

    def make(handler: Any = echo, **kw: Any) -> Door:
        doors.append(Door(agentdb_storage, handler, **kw))
        return doors[-1]

    yield make
    for d in doors:
        await d.http.aclose()


@pytest.fixture
async def expert(agentdb_storage: AgentDBStorage) -> AsyncIterator[Callable[..., A2ADelegate]]:
    """``expert(handler, name=, announce=, tool=)``: an ``A2ADelegate`` to an expert's door, its
    own lane, on the same store; ``tool`` names its tool."""
    clients: list[httpx.AsyncClient] = []
    delegates: list[A2ADelegate] = []

    def make(
        handler: Any,
        name: str = EXPERT,
        announce: bool = True,
        tool: str | None = None,
        **kw: Any,
    ) -> A2ADelegate:
        server = AgentServer(
            storage=agentdb_storage, agent_name=name, api_key=API_KEY, api_secret=API_SECRET
        )
        kw.setdefault("description", "Answers what the front desk cannot.")
        server.a2a_session(endpoint=name, **kw)(handler)
        clients.append(
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server.http.app),
                base_url="http://experts.test",
                timeout=30.0,
            )
        )
        url = f"http://experts.test/{name}"
        delegates.append(A2ADelegate(url, httpx_client=clients[-1], announce=announce, name=tool))
        return delegates[-1]

    yield make
    for delegate in delegates:
        await delegate.aclose()
    for client in clients:
        await client.aclose()


# ---------------------------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------------------------


@dataclass
class Reply:
    """What the model answers: text, tool calls, and when, if not simply next in order; with
    ``hold`` the stream stops after the text until it is set."""

    content: str = ""
    tool_calls: list[FunctionToolCall] = field(default_factory=list)
    when: Callable[[ChatContext], bool] | None = None
    hold: asyncio.Event | None = None


def call(name: str, call_id: str, **arguments: Any) -> FunctionToolCall:
    return FunctionToolCall(name=name, arguments=json.dumps(arguments), call_id=call_id)


def output_of(call_id: str) -> Callable[[ChatContext], bool]:
    """A request whose last item is the output of the call ``call_id``."""

    def matches(ctx: ChatContext) -> bool:
        last = ctx.items[-1] if ctx.items else None
        return last is not None and last.type == "function_call_output" and last.call_id == call_id

    return matches


class ScriptedLLM(FakeLLM):
    """Each request takes the first reply left whose ``when`` matches it, or that has none; with
    none left the answer is empty. ``requests`` keeps every request's chat context, and
    ``tool_choices`` its tool choice."""

    def __init__(self, *replies: Reply) -> None:
        super().__init__()
        self.replies = list(replies)
        self.requests: list[ChatContext] = []
        self.tool_choices: list[Any] = []

    def chat(  # type: ignore[override]
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        **kwargs: Any,
    ) -> LLMStream:
        self.requests.append(chat_ctx.copy())
        self.tool_choices.append(kwargs.get("tool_choice"))
        reply = next((r for r in self.replies if r.when is None or r.when(chat_ctx)), None)
        if reply is not None:
            self.replies.remove(reply)
        return ScriptedStream(
            self, reply, chat_ctx=chat_ctx, tools=tools or [], conn_options=conn_options
        )


class ScriptedStream(FakeLLMStream):
    def __init__(self, llm: ScriptedLLM, reply: Reply | None, **kw: Any) -> None:
        super().__init__(llm, **kw)
        self._reply = reply

    async def _run(self) -> None:
        if self._reply is not None:
            self._send_chunk(delta=self._reply.content, tool_calls=self._reply.tool_calls)
            if self._reply.hold is not None:
                await self._reply.hold.wait()
