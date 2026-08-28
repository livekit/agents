"""Reaching a delegate over the A2A protocol (https://a2a-protocol.org).

An expert written in LangGraph, ADK or anything else that speaks A2A becomes usable as a
delegate here, unchanged::

    session = AgentSession(delegate=a2a.A2ADelegate("https://experts.internal/fare-desk"))

A2A is a transport, not the framework's contract. A delegate that runs in this process is
reached directly and never touches this module. Two things live here: the mapping
(:func:`to_a2a_request` / :func:`from_a2a_request` and :func:`to_a2a_events` /
:func:`from_a2a_events`, which are inverses) and the client that carries it. Nothing outside
this module names an A2A type.

Needs the ``a2a`` extra: ``pip install 'livekit-agents[a2a]'``.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterable, AsyncIterator, Iterable
from contextlib import aclosing
from typing import Any, cast

from ..llm.chat_context import ChatContext
from ..log import logger
from ..utils import shortuuid
from .delegation import DelegationRequest, DelegationState, DelegationStream, DelegationUpdate

try:
    import httpx
    from a2a.client import Client, ClientConfig, ClientFactory
    from a2a.types import a2a_pb2 as pb
    from a2a.utils.constants import PROTOCOL_VERSION_1_0, TransportProtocol
    from google.protobuf import json_format, struct_pb2
except ImportError as e:
    raise ImportError(
        "The 'a2a-sdk' package is required to reach a delegate over A2A but is not installed.\n"
        "To fix this, install the optional dependency: pip install 'livekit-agents[a2a]'"
    ) from e


DELEGATION_ID_KEY = "livekit.delegation_id"
"""Request metadata key carrying the caller's ``delegate`` call id.

The task id belongs to the server, so the caller's own id travels beside the request instead.
"""

CHAT_CTX_PART_KIND = "livekit.chat_ctx"
"""``Part.metadata["kind"]`` marking the data part that carries the conversation.

An agent that does not know this framework ignores the part and works from the task text
alone, which is why the ``delegate`` tool is told to state the request in full.
"""

ANSWER_ARTIFACT_NAME = "answer"
"""Name of the artifact carrying what the conversation model phrases."""


_STATE_FROM_A2A: dict[Any, DelegationState] = {
    pb.TaskState.TASK_STATE_SUBMITTED: "working",
    pb.TaskState.TASK_STATE_WORKING: "working",
    pb.TaskState.TASK_STATE_COMPLETED: "completed",
    pb.TaskState.TASK_STATE_CANCELED: "canceled",
    pb.TaskState.TASK_STATE_FAILED: "failed",
    # a delegate that declines is a delegate that could not answer, and the conversation model
    # has the same thing to do about either
    pb.TaskState.TASK_STATE_REJECTED: "failed",
    pb.TaskState.TASK_STATE_INPUT_REQUIRED: "input-required",
    pb.TaskState.TASK_STATE_AUTH_REQUIRED: "input-required",
}

_STATE_TO_A2A: dict[DelegationState, Any] = {
    "working": pb.TaskState.TASK_STATE_WORKING,
    "completed": pb.TaskState.TASK_STATE_COMPLETED,
    "canceled": pb.TaskState.TASK_STATE_CANCELED,
    "failed": pb.TaskState.TASK_STATE_FAILED,
    "input-required": pb.TaskState.TASK_STATE_INPUT_REQUIRED,
}


def _text(parts: Iterable[Any]) -> str:
    return "\n".join(part.text for part in parts if part.WhichOneof("content") == "text")


def _struct(value: dict[str, Any]) -> struct_pb2.Struct:
    return json_format.ParseDict(value, struct_pb2.Struct())


def _agent_message(text: str, *, task_id: str, context_id: str) -> pb.Message:
    return pb.Message(
        message_id=shortuuid("msg-"),
        task_id=task_id,
        context_id=context_id,
        role=pb.Role.ROLE_AGENT,
        parts=[pb.Part(text=text)],
    )


def delegate_agent_card(
    *,
    url: str = "",
    name: str = "delegate",
    description: str = "Handles reasoning, lookups and actions for a conversation.",
    version: str = "1.0.0",
) -> pb.AgentCard:
    """The card describing a delegate: one skill, text in and text out.

    One skill because there is one delegate and nothing to route to — several specialties live
    inside it, as sub-agents it hands off to.
    """
    return pb.AgentCard(
        name=name,
        description=description,
        version=version,
        capabilities=pb.AgentCapabilities(streaming=True),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[
            pb.AgentSkill(
                id="delegate", name="delegate", description=description, tags=["delegation"]
            )
        ],
        supported_interfaces=[
            pb.AgentInterface(
                url=url,
                protocol_binding=TransportProtocol.HTTP_JSON,
                protocol_version=PROTOCOL_VERSION_1_0,
            )
        ],
    )


# -- the mapping -------------------------------------------------------------------------------


def to_a2a_request(request: DelegationRequest, *, context_id: str) -> pb.SendMessageRequest:
    """One delegation as an A2A message: the task as text, the conversation as a data part."""
    chat_ctx = json_format.ParseDict(request.chat_ctx.to_dict(), struct_pb2.Value())
    return pb.SendMessageRequest(
        # no task id: the server assigns it, so `delegation_id` travels in the metadata instead
        message=pb.Message(
            message_id=request.delegation_id,
            context_id=context_id,
            role=pb.Role.ROLE_USER,
            parts=[
                pb.Part(text=request.task),
                pb.Part(data=chat_ctx, metadata=_struct({"kind": CHAT_CTX_PART_KIND})),
            ],
        ),
        metadata=_struct({**request.metadata, DELEGATION_ID_KEY: request.delegation_id}),
        configuration=pb.SendMessageConfiguration(accepted_output_modes=["text/plain"]),
    )


def from_a2a_request(request: pb.SendMessageRequest) -> DelegationRequest:
    """The delegation an incoming A2A request carries — the inverse of :func:`to_a2a_request`.

    A caller that sends only text still produces a valid request, with an empty conversation,
    which is what makes a plain A2A client usable against a published delegate.
    """
    message = request.message
    chat_ctx = ChatContext.empty()
    for part in message.parts:
        if part.WhichOneof("content") != "data":
            continue
        if json_format.MessageToDict(part.metadata).get("kind") != CHAT_CTX_PART_KIND:
            continue
        chat_ctx = ChatContext.from_dict(json_format.MessageToDict(part.data))

    metadata = dict(json_format.MessageToDict(request.metadata))
    delegation_id = metadata.pop(DELEGATION_ID_KEY, None) or message.message_id
    return DelegationRequest(
        task=_text(message.parts),
        chat_ctx=chat_ctx,
        delegation_id=str(delegation_id),
        metadata=metadata,
    )


def to_a2a_events(event: DelegationUpdate, *, task_id: str, context_id: str) -> list[Any]:
    """One delegation update as the A2A events that carry it.

    A terminal state with something to say is two events, because the protocol separates the
    deliverable from the lifecycle: the artifact, then the status that ends the task.
    """
    status = pb.TaskStatusUpdateEvent(
        task_id=task_id,
        context_id=context_id,
        status=pb.TaskStatus(state=_STATE_TO_A2A[event.state]),
    )

    if event.state in ("working", "failed", "input-required"):
        # nothing was produced, so whatever the delegate said travels as the status message
        if event.text:
            status.status.message.CopyFrom(
                _agent_message(event.text, task_id=task_id, context_id=context_id)
            )
        return [status]

    # completed and canceled hand over what the delegation produced. a delegation that finished
    # with nothing to add completes carrying no artifact, the way a tool may return None
    if not event.text:
        return [status]

    artifact = pb.TaskArtifactUpdateEvent(
        task_id=task_id,
        context_id=context_id,
        artifact=pb.Artifact(
            artifact_id=shortuuid("art-"),
            name=ANSWER_ARTIFACT_NAME,
            parts=[pb.Part(text=event.text)],
        ),
        last_chunk=True,
    )
    return [artifact, status]


async def from_a2a_events(events: AsyncIterable[Any]) -> AsyncIterator[DelegationUpdate]:
    """An A2A task's event stream as delegation updates — the inverse of :func:`to_a2a_events`.

    Artifacts accumulate until a terminal status arrives, so the deliverable and the lifecycle
    reach the caller as one update. The artifact is the answer wherever there is one; a
    terminal state that produced none says why in its status message.
    """
    # keyed per artifact id: an agent may stream several at once, and `append` refers to the
    # one being named, so keying on nothing concatenates one artifact's chunks onto another's
    artifacts: dict[str, str] = {}
    names: dict[str, str] = {}

    async for event in events:
        if isinstance(event, pb.StreamResponse):
            event = getattr(event, event.WhichOneof("payload"))

        if isinstance(event, pb.Message):
            # an agent that replies without opening a task has one output, so it is the answer
            yield DelegationUpdate(_text(event.parts), state="completed")
            return

        if isinstance(event, pb.TaskArtifactUpdateEvent):
            artifact_id = event.artifact.artifact_id
            chunk = _text(event.artifact.parts)
            artifacts[artifact_id] = (
                artifacts.get(artifact_id, "") + chunk if event.append else chunk
            )
            if event.artifact.name:
                names[artifact_id] = event.artifact.name
            continue

        if not isinstance(event, (pb.Task, pb.TaskStatusUpdateEvent)):
            continue

        state = _STATE_FROM_A2A.get(event.status.state, "working")
        text = _text(event.status.message.parts)
        if state == "working":
            if text:
                yield DelegationUpdate(text)
            continue

        # where an agent produced more than one, the one named `answer` is the answer, and
        # otherwise they join in arrival order
        named = [
            body
            for artifact_id, body in artifacts.items()
            if names.get(artifact_id) == ANSWER_ARTIFACT_NAME
        ]
        answer = "\n".join(named or artifacts.values())
        yield DelegationUpdate(answer or text, state=state)
        return


# -- the client --------------------------------------------------------------------------------


class _A2ADelegationStream(DelegationStream):
    def __init__(self, request: DelegationRequest, *, delegate: A2ADelegate) -> None:
        self._delegate = delegate
        super().__init__(request)

    async def _run(self) -> str | None:
        client = await self._delegate._connect()
        a2a_request = to_a2a_request(self._request, context_id=self._delegate._context_id)
        # the SDK under-declares its stream as an AsyncIterator; it is a generator, and until
        # it is closed it holds its HTTP connection
        stream = cast("AsyncGenerator[Any, None]", client.send_message(a2a_request))
        # the terminal update is one of these, so nothing is returned to complete with
        async with aclosing(stream) as events:
            async for update in from_a2a_events(events):
                self.send(update)

        return None


class A2ADelegate:
    """A delegate reached over A2A on an HTTP endpoint.

    The card at ``<url>/.well-known/agent-card.json`` is read once, on the first delegation, so
    the endpoint describes its own skills and auth rather than being a URL with an implicit
    contract. Closing it stays the caller's job: a delegate may serve one session or every
    session in a worker, and the framework cannot tell which.
    """

    def __init__(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        httpx_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._url = url
        # no read timeout: a delegation is as long as the work it describes, and the task
        # stream is what reports progress meanwhile
        self._http = httpx_client or httpx.AsyncClient(headers=headers or {}, timeout=None)
        self._owns_http = httpx_client is None
        self._client: Client | None = None
        # delegations run concurrently, and the first two would otherwise each resolve the card
        self._connect_lock = asyncio.Lock()
        # the tasks of one caller are grouped under one context
        self._context_id = shortuuid("lk-delegation-")

    def __call__(self, request: DelegationRequest) -> DelegationStream:
        return _A2ADelegationStream(request, delegate=self)

    async def _connect(self) -> Client:
        async with self._connect_lock:
            if self._client is None:
                factory = ClientFactory(
                    ClientConfig(
                        httpx_client=self._http,
                        streaming=True,
                        supported_protocol_bindings=[TransportProtocol.HTTP_JSON],
                        accepted_output_modes=["text/plain"],
                    )
                )
                self._client = await factory.create_from_url(self._url)
                logger.debug("resolved the delegate's agent card", extra={"url": self._url})
            return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None
        if self._owns_http:
            await self._http.aclose()
