"""Talking to an A2A endpoint: one context, one task at a time."""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import AsyncGenerator, AsyncIterator
from types import TracebackType
from typing import Any, cast

from ..log import logger
from ..utils import shortuuid
from .codec import from_a2a_events, to_a2a_request
from .extension import EXTENSION_URI, REASON, offers_extension, pb, struct
from .types import TaskInput, TaskUpdate

try:
    import httpx
    from a2a.client import Client, ClientCallContext, ClientConfig, ClientFactory
    from a2a.client.card_resolver import A2ACardResolver
    from a2a.extensions.common import HTTP_EXTENSION_HEADER
    from a2a.utils.constants import TransportProtocol
except ImportError as e:
    raise ImportError(
        "The 'a2a-sdk' package is required to speak A2A but is not installed.\n"
        "To fix this, install the optional dependency: pip install 'livekit-agents[a2a]'"
    ) from e


class TaskStream:
    """One task: the updates it produces, and the handle to cancel it.

    Read it under ``async with``, which closes the HTTP stream when the caller stops
    listening; :attr:`task_id` is empty until the first event, and cancelling before then
    does nothing.
    """

    def __init__(self, client: A2AClient, task_input: TaskInput) -> None:
        self._client = client
        self._input = task_input
        self._task_id = ""
        self._updates: AsyncGenerator[TaskUpdate, None] | None = None
        self._raw: AsyncGenerator[Any, None] | None = None
        self._holds_turn = False

    @property
    def task_id(self) -> str:
        """The server's id for this task, once its first event has arrived."""
        return self._task_id

    @property
    def task_input(self) -> TaskInput:
        return self._input

    async def cancel(self, reason: str = "") -> None:
        """Ask the server to stop. Best-effort: work can finish between the two."""
        if self._task_id:
            await self._client.cancel(self._task_id, reason=reason)

    async def aclose(self) -> None:
        try:
            # closing the outermost generator propagates down to the SDK's, which is what
            # holds the HTTP connection; closing only that one strands the two wrapping it
            if self._updates is not None:
                await self._updates.aclose()
            elif self._raw is not None:
                await self._raw.aclose()
        finally:
            self._updates = self._raw = None
            self._release_turn()

    async def _start(self) -> None:
        client = await self._client._connect()
        # one unacknowledged send per context: two HTTP requests carry no ordering between
        # them, so the next one waits for this one's task to come back
        await self._client._turn.acquire()
        self._holds_turn = True
        try:
            task_input = self._input
            if task_input.context_id:
                # a later request on this client continues the context without naming it
                self._client._context_id = task_input.context_id
            else:
                task_input = dataclasses.replace(task_input, context_id=self._client.context_id)
            if not self._client.extension_active:
                # the conversation id is ours to share only with an endpoint that joins it
                task_input = dataclasses.replace(
                    task_input, conversation_id=None, caller_session_id=None
                )
            request = to_a2a_request(
                task_input, reference_task_ids=self._client._take_open_questions()
            )
            # the SDK under-declares its stream as an AsyncIterator; it is a generator, and
            # until it is closed it holds its HTTP connection
            self._raw = cast(
                "AsyncGenerator[Any, None]",
                client.send_message(request, context=self._client._call_context),
            )
            self._updates = cast(
                "AsyncGenerator[TaskUpdate, None]", from_a2a_events(self._acknowledge(self._raw))
            )
        except BaseException:
            self._release_turn()
            raise

    async def _acknowledge(self, events: AsyncGenerator[Any, None]) -> AsyncIterator[Any]:
        """Pass every event through, taking the server's task id off the first one."""
        try:
            async for event in events:
                payload = event
                if isinstance(payload, pb.StreamResponse):
                    payload = getattr(payload, payload.WhichOneof("payload"))
                if not self._task_id:
                    # a Task names itself `id`; every later event names it `task_id`
                    task_id = getattr(payload, "id", "") or getattr(payload, "task_id", "")
                    if task_id:
                        self._task_id = task_id
                        if context_id := getattr(payload, "context_id", ""):
                            self._client._context_id = context_id
                        self._release_turn()
                yield event
        finally:
            self._release_turn()

    def _release_turn(self) -> None:
        if self._holds_turn:
            self._holds_turn = False
            self._client._turn.release()

    async def __anext__(self) -> TaskUpdate:
        if self._updates is None:
            await self._start()
        assert self._updates is not None
        update = await self._updates.__anext__()
        if update.state == "input-required" and self._task_id:
            self._client._open_questions.add(self._task_id)
        return update

    def __aiter__(self) -> AsyncIterator[TaskUpdate]:
        return self

    async def __aenter__(self) -> TaskStream:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.aclose()


class A2AClient:
    """An A2A endpoint, as one context.

    The card is read once on the first send, and the extension is activated only where that
    card offers it; one instance is one ``context_id``, so give each session its own.
    """

    def __init__(
        self,
        url: str,
        *,
        context_id: str | None = None,
        headers: dict[str, str] | None = None,
        httpx_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._url = url
        # no read timeout: a task is as long as the work it describes, and its event stream is
        # what reports progress meanwhile
        self._http = httpx_client or httpx.AsyncClient(headers=headers or {}, timeout=None)
        self._owns_http = httpx_client is None
        self._client: Client | None = None
        self._extension_active = False
        self._call_context: ClientCallContext | None = None
        self._connect_lock = asyncio.Lock()
        self._turn = asyncio.Semaphore(1)
        self._context_id = context_id or shortuuid("lk-ctx-")
        self._open_questions: set[str] = set()

    @property
    def url(self) -> str:
        return self._url

    @property
    def context_id(self) -> str:
        """The context. The server finds or creates its side by this."""
        return self._context_id

    @property
    def extension_active(self) -> bool:
        """Whether the endpoint offered the profile. False until the card is read."""
        return self._extension_active

    def send(self, task_input: TaskInput) -> TaskStream:
        """Send one message and read the task it opens, on the request's context when it names
        one, which this client then keeps, and on this client's otherwise."""
        return TaskStream(self, task_input)

    async def cancel(self, task_id: str, *, reason: str = "") -> None:
        client = await self._connect()
        request = pb.CancelTaskRequest(id=task_id)
        if reason:
            request.metadata.CopyFrom(struct({REASON: reason}))
        try:
            await client.cancel_task(request, context=self._call_context)
        except Exception:
            # best-effort by contract: a server may have finished, or may not support it
            logger.debug("the endpoint did not cancel the task", extra={"task_id": task_id})

    def _take_open_questions(self) -> list[str]:
        """Tasks that ended asking something, handed over once for the server to match."""
        open_questions = sorted(self._open_questions)
        self._open_questions.clear()
        return open_questions

    async def _connect(self) -> Client:
        async with self._connect_lock:
            if self._client is not None:
                return self._client

            config = ClientConfig(
                httpx_client=self._http,
                streaming=True,
                supported_protocol_bindings=[TransportProtocol.HTTP_JSON],
                accepted_output_modes=["text/plain"],
            )
            card = await A2ACardResolver(self._http, self._url).get_agent_card()
            self._extension_active = offers_extension(card)
            if self._extension_active:
                # asking is what activates it; a server that does not echo it back has not.
                # per request rather than on the client, which the caller may share with
                # endpoints that never offered the profile
                self._call_context = ClientCallContext(
                    service_parameters={HTTP_EXTENSION_HEADER: EXTENSION_URI}
                )

            self._client = ClientFactory(config).create(card)
            logger.debug(
                "resolved the agent card",
                extra={
                    "url": self._url,
                    "card": card.name,
                    "extension": self._extension_active,
                },
            )
            return self._client

    async def close_context(self) -> None:
        """Tell the endpoint the context is over, so it need not wait for idle.

        Best-effort: a server keeps its own idle policy, and this only saves it the wait.
        """
        if self._client is None:
            return  # nothing was ever sent on this context
        try:
            async with self.send(TaskInput(closing=True)) as stream:
                async for _ in stream:
                    pass
        except Exception:
            logger.debug("the endpoint did not take the goodbye", extra={"url": self._url})

    async def aclose(self) -> None:
        await self.close_context()
        if self._client is not None:
            await self._client.close()
            self._client = None
        if self._owns_http:
            await self._http.aclose()
