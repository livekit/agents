"""``AgentClient``: a session endpoint from outside the agent, thin over the stock a2a-sdk.

It reads the card once, speaks JSON-RPC over HTTP+SSE, asks for the Live extension when the
card offers it, and reads both profiles' keys in what comes back. Nothing here is required to
talk to an endpoint: any A2A client works.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator, Callable, Mapping

import httpx
from a2a.client import Client, ClientCallContext, ClientConfig, ClientFactory
from a2a.client.card_resolver import A2ACardResolver
from a2a.client.errors import A2AClientError
from google.protobuf import json_format
from pydantic import JsonValue

from ..llm.chat_context import ChatContext
from ..log import logger
from ..utils import shortuuid
from .codec import (
    StoredMessage,
    TaskUpdate,
    chat_ctx_part,
    message_to_pb,
    metadata_to_wire,
    updates_from_events,
)
from .extension import (
    EXTENSION_URI,
    EXTENSIONS_HEADER,
    JSONRPC_BINDING,
    REASON,
    Extensions,
    pb,
)


class Reply:
    """One ``send``: the task it opened, known as soon as ``send`` returns, and its updates
    as they arrive, the progress then the answer."""

    def __init__(self, task_id: str, thread_id: str, updates: AsyncIterator[TaskUpdate]):
        self.task_id = task_id
        self.thread_id = thread_id
        self._updates = updates

    def __aiter__(self) -> AsyncIterator[TaskUpdate]:
        return self._updates


class AgentClient:
    """A text endpoint, as a backend or a test talks to it.

    ::

        client = AgentClient(endpoint_url, token=token)
        reply = await client.send("Where is my order?", thread_id=thread_id)
        async for update in reply: ...
    """

    def __init__(
        self,
        endpoint_url: str,
        token: str | Callable[[], str | None] | None = None,
        *,
        httpx_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._url = endpoint_url.rstrip("/")
        self._token = token
        self._thread_id: str | None = None
        self._echo: Extensions | None = None
        self._http = httpx_client or httpx.AsyncClient(timeout=None)
        self._owns_http = httpx_client is None
        self._http.event_hooks.setdefault("response", []).append(self._on_response)
        self._client: Client | None = None
        self._card: pb.AgentCard | None = None
        self._lock = asyncio.Lock()

    @property
    def extensions(self) -> Extensions | None:
        """What the server echoed on the last response, ``None`` before the first."""
        return self._echo

    @property
    def card(self) -> pb.AgentCard | None:
        """The expert's card, once connected."""
        return self._card

    @property
    def offers_extension(self) -> bool:
        """Whether the card offers the Live extension: decides what the client sends."""
        if self._card is None:
            return False
        return any(ext.uri == EXTENSION_URI for ext in self._card.capabilities.extensions)

    @property
    def transport(self) -> str | None:
        """``JSONRPC`` once connected."""
        return None if self._client is None else JSONRPC_BINDING

    async def connect(self) -> Client:
        """Reads the card and builds the client; a no-op once connected."""
        async with self._lock:
            if self._client is not None:
                return self._client

            self._card = await self._resolve_card()
            self._client = ClientFactory(self._config()).create(self._card)
            logger.debug(
                "resolved an agent card",
                extra={"url": self._url, "card": self._card.name, "transport": self.transport},
            )
            return self._client

    def _bearer(self) -> str | None:
        """The token for the next request; a callable is asked each time, so it can mint."""
        return self._token() if callable(self._token) else self._token

    async def _resolve_card(self) -> pb.AgentCard:
        token = self._bearer()
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        resolver = A2ACardResolver(self._http, self._url)
        return await resolver.get_agent_card(http_kwargs={"headers": headers} if headers else None)

    def _config(self) -> ClientConfig:
        return ClientConfig(
            httpx_client=self._http,
            streaming=True,
            supported_protocol_bindings=[JSONRPC_BINDING],
            accepted_output_modes=["text/plain"],
        )

    async def send(
        self,
        message: str | StoredMessage,
        *,
        thread_id: str | None = None,
        task_id: str | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
        chat_ctx: ChatContext | None = None,
    ) -> Reply:
        """``message/stream``: a text or a message dict (storage shape, short keys), on a session
        when given, continuing a task when ``task_id`` is given. Returns once the server has
        acknowledged the task."""
        client = await self.connect()
        outgoing = self._outgoing_message(message, thread_id, task_id, metadata, chat_ctx)
        request = pb.SendMessageRequest(
            message=message_to_pb(outgoing),
            configuration=pb.SendMessageConfiguration(accepted_output_modes=["text/plain"]),
        )

        events = client.send_message(request, context=self._call_context(thread_id))
        updates = updates_from_events(events)
        first = await updates.__anext__()
        if first.task_id:
            self._thread_id = first.context_id or self._thread_id

        if first.task_id and first.state == "submitted":
            return Reply(first.task_id, first.context_id, updates)

        # a server that opened no task, or answered without an acknowledgment: the first
        # update is content
        return Reply(first.task_id, first.context_id, _replay(first, updates))

    async def cancel(
        self, task_id: str, *, thread_id: str | None = None, reason: str | None = None
    ) -> pb.Task:
        """``tasks/cancel``. The task's session rides as ``contextId`` in the params when the
        card offers the extension (plan 3.3.8); a stock server gets the stock request."""
        await self.connect()
        params: dict[str, JsonValue] = {"id": task_id}
        context_id = thread_id or self._thread_id
        if context_id and self.offers_extension:
            params["contextId"] = context_id
        if reason:
            params["metadata"] = metadata_to_wire({REASON: reason})

        result = await self._rpc("CancelTask", params)
        return json_format.ParseDict(result, pb.Task(), ignore_unknown_fields=True)

    async def _rpc(self, method: str, params: dict[str, JsonValue]) -> dict[str, JsonValue]:
        """One JSON-RPC call the a2a-sdk's typed requests cannot carry; the result object."""
        headers = self._call_context(None).service_parameters
        body = {"jsonrpc": "2.0", "id": shortuuid("rpc_"), "method": method, "params": params}
        response = await self._http.post(self._rpc_url, json=body, headers=headers)
        response.raise_for_status()

        answer: dict[str, JsonValue] = response.json()
        if "error" in answer:
            raise A2AClientError(f"{method}: {json.dumps(answer['error'])}")
        result = answer.get("result")
        if not isinstance(result, dict):
            raise A2AClientError(f"{method}: the result is not an object")
        return result

    @property
    def _rpc_url(self) -> str:
        """The card's JSON-RPC interface, else the endpoint URL the client was given."""
        if self._card is None:
            return self._url
        for interface in self._card.supported_interfaces:
            if interface.protocol_binding == JSONRPC_BINDING and interface.url:
                return str(interface.url)
        return self._url

    async def aclose(self) -> None:
        if self._client is not None:
            with contextlib.suppress(Exception):
                await self._client.close()
            self._client = None

        if self._owns_http:
            await self._http.aclose()

    def _outgoing_message(
        self,
        message: str | StoredMessage,
        thread_id: str | None,
        task_id: str | None,
        metadata: Mapping[str, JsonValue] | None,
        chat_ctx: ChatContext | None,
    ) -> StoredMessage:
        msg = _as_stored_message(message)

        context_id = thread_id or msg.get("contextId") or self._thread_id
        if context_id:
            msg["contextId"] = context_id
        if task_id:
            msg["taskId"] = task_id

        if metadata:
            msg["metadata"] = {**(msg.get("metadata") or {}), **metadata}
        if chat_ctx is not None and chat_ctx.items and self.offers_extension:
            msg["parts"] = [*msg.get("parts", []), chat_ctx_part(chat_ctx)]
        return msg

    def _call_context(self, thread_id: str | None) -> ClientCallContext:
        # a stock server gets a stock request: the extension header goes only to a card that
        # offers the extension; the session is in the body, never in a header
        headers: dict[str, str] = {"A2A-Version": "1.0"}
        if self.offers_extension:
            headers[EXTENSIONS_HEADER] = EXTENSION_URI
        if token := self._bearer():
            headers["Authorization"] = f"Bearer {token}"
        return ClientCallContext(service_parameters=headers)

    async def _on_response(self, response: httpx.Response) -> None:
        echoed = response.headers.get(EXTENSIONS_HEADER)
        if echoed is not None or response.request.method == "POST":
            self._echo = Extensions.from_header(echoed)


def _as_stored_message(message: str | StoredMessage) -> StoredMessage:
    """A text as a one-part user message; a dict copied, with the ids every message needs."""
    if isinstance(message, str):
        msg: StoredMessage = {"role": "user", "parts": [{"kind": "text", "text": message}]}
    else:
        msg = dict(message)

    msg.setdefault("messageId", shortuuid("msg_"))
    msg.setdefault("role", "user")
    return msg


async def _replay(first: TaskUpdate, rest: AsyncIterator[TaskUpdate]) -> AsyncIterator[TaskUpdate]:
    yield first
    async for update in rest:
        yield update


__all__ = ["AgentClient", "Reply"]
