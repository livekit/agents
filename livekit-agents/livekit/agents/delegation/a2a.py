"""``A2ADelegate``: an expert on another deployment, or any A2A server, as a toolset.

A stock A2A client with the Live extension on top, JSON-RPC over HTTP+SSE. The caller degrades
from what the card offers: with the extension, ``livekit/kind: delegation`` and, outside the
caller's session, the conversation as a ``chat_ctx`` part; without it, the request as text, the
expert's ``working`` status messages as progress, the artifact text as the result.

Attached to an ``Agent`` as ``tools=[A2ADelegate(url)]`` it is one tool, named and described
from the expert's card, the same in a text run and on a call (docs/multi-agent.md).
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

import httpx
from a2a.client.errors import A2AClientError
from a2a.utils.errors import A2AError
from pydantic import JsonValue

from ..llm.chat_context import ChatContext, ChatItem, FunctionCall
from ..llm.tool_context import (
    ANSWERS_KEY,
    FunctionTool,
    InputRequired,
    Tool,
    ToolError,
    ToolResult,
    Toolset,
    function_tool,
)
from ..log import logger
from ..types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from ..voice.agent_run import current_run_hooks
from ..voice.events import RunContext
from ..voice.run_result import RunUsage
from .delegate import (
    CONTEXT_ID_KEY,
    ENDPOINT_KEY,
    TASK_ID_KEY,
    Delegate,
    DelegationFailed,
    DelegationResult,
    Progress,
    answer_message,
    record_delegation,
    report,
)

ASK = (
    "State the request in full, with everything the person said that bears on it; the "
    "answer, or the question it has for the person, is what you say next."
)
"""The second paragraph of the tool's description, after the card's own."""

UNREAD = (
    "Its description could not be read; a call still tries it, and says why if it cannot answer."
)
"""The second paragraph of the tool's description while the card cannot be read, after the
expert's name."""

DISPATCHED = (
    'Acknowledge in a few natural words, "one moment", "sure, let me check", "okay, looking '
    'now", varying the wording, restating none of the request, asking nothing and promising '
    "nothing about the outcome. The answer is a separate entry, not this one."
)
"""What the model reads on a call while the expert works: the line it acknowledges with."""

if TYPE_CHECKING:
    from ..a2a.client import AgentClient, Reply
    from ..a2a.codec import TaskUpdate


class A2ADelegate(Delegate, Toolset):
    """An expert served at an A2A endpoint, ours or not.

    ::

        ELIGIBILITY = "https://acme.livekit.cloud/agents/eligibility/production/eligibility"
        front_desk = Agent(instructions=..., tools=[A2ADelegate(ELIGIBILITY)])

        @front_desk.tool
        async def coverage(ctx: RunContext, payer: str) -> str:
            result = await A2ADelegate(ELIGIBILITY).run(f"{payer} coverage", progress=ctx.update)
            return result.output

    As a toolset it is one tool, ``ask``, described by the card, read in ``setup()`` (or by a call,
    while it could not be read); ``name=`` tells two experts on one agent apart. The tool hands the
    request on with the run's thread and history, passes the expert's progress to ``ctx.update()``
    and its exact text to ``ctx.say()``, records the expert's task on the call and returns the
    answer, or the question the expert has for the person; an answer or a question the expert
    flagged reaches the person as written. On a call ``announce`` releases the turn with a line the
    model acknowledges while the expert works.

    An expert whose card offers the Live extension gets the caller's session id as the
    ``contextId`` and the caller's lane as ``livekit/callerLane``, and writes its own lane of
    that session; outside a turn, with no session to name, it gets the conversation as a
    ``chat_ctx`` part and keeps a session of its own. Any other A2A server gets the request as
    text and nothing else, and keeps a context of its own. A context the expert keeps is
    recorded on the delegating call, and the next delegation to the same expert reads it from
    the conversation: one delegate serves every conversation in the process and keeps none
    (docs/multi-agent.md). ``token`` omitted, an expert on the project's own host
    (``LIVEKIT_URL``) gets a project token minted per request from ``LIVEKIT_API_KEY`` and
    ``LIVEKIT_API_SECRET``; an expert anywhere else gets none. A dropped answer stream is read
    again up to ``conn_options.max_retry`` times.
    """

    def __init__(
        self,
        url: str,
        *,
        token: str | None = None,
        httpx_client: httpx.AsyncClient | None = None,
        name: str | None = None,
        announce: bool = True,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        self._url = url.rstrip("/")
        self._token = token
        self._httpx_client = httpx_client
        self._conn_options = conn_options
        self._client: AgentClient | None = None
        self._cancels: set[asyncio.Future[None]] = set()
        self._tool_name = name
        self._announce = announce
        self._tool: FunctionTool | None = None
        # the tool names the expert only: its card could not be read yet
        self._unread = False
        Toolset.__init__(self, id=self._url.rsplit("/", 1)[-1] or self._url)

    # -- the toolset --

    @property
    def tools(self) -> Sequence[Tool | Toolset]:
        return [self._tool] if self._tool is not None else []

    async def setup(self) -> A2ADelegate:
        """Reads the card and builds the tool from it, once per delegate. While the card cannot
        be read the tool names the expert and the error is raised for the caller to log."""
        if self._tool is not None and not self._unread:
            return self

        try:
            client = await self._connect()
        except A2AError:
            # the run logs a failed setup and keeps the tools as they are: offer this one first
            if self._tool is None:
                label = self._tool_name or self.id
                self._tool = function_tool(
                    self._ask,
                    name=self._tool_name or "ask",
                    description=f"{label}\n\n{UNREAD}\n\n{ASK}",
                )
                self._unread = True
            raise

        self._build_tool(client)
        return self

    def _build_tool(self, client: AgentClient) -> None:
        """The tool as the card describes it, in place of the one that names the expert."""
        card = client.card
        self._id = card.name if card is not None else self.id
        about = card.description.strip() if card is not None and card.description else ""
        description = f"{about}\n\n{ASK}" if about else ASK
        self._tool = function_tool(
            self._ask, name=self._tool_name or "ask", description=description
        )
        if self._unread:
            self._unread = False
            self._tools_changed()

    async def _ask(self, ctx: RunContext, request: str) -> str | InputRequired | ToolResult:
        """The tool: the request to the expert with what the run knows, the answer back.

        Args:
            request: the request, in full
        """
        session = ctx._session
        history: ChatContext
        if session is not None:
            history = session.history
        else:
            assert ctx._run is not None
            history = ctx._run.chat_ctx
        asked = _asking_call(ctx)
        if session is not None and self._announce and asked is None:
            await ctx.update(DISPATCHED)
        try:
            result = await self.run(
                request,
                progress=ctx.update,
                say=ctx.say,
                context_id=session.thread.id if session and session.thread else None,
                message_history=history,
                usage=ctx.usage,
                caller_lane=session.thread.agent_name if session and session.thread else None,
                task_id=asked,
            )
        except DelegationFailed as e:
            raise ToolError(str(e) or "the delegation failed") from e

        record_delegation(ctx.function_call, result)
        if result.input_required:
            # the expert asked the person something: asked as it wrote it when it said so, and
            # what they say next continues its task, as they said it
            verbatim = result.verbatim is not None
            return InputRequired(result.input_required, tool=self._tool, verbatim=verbatim)
        if result.verbatim is not None:
            # the person gets the answer as written, and the model does not word it
            return ToolResult(result.verbatim, verbatim=True, reply_required=False)
        return result.output if isinstance(result.output, str) else str(result.output or "")

    @property
    def transport(self) -> str | None:
        return self._client.transport if self._client is not None else None

    @property
    def extension_active(self) -> bool | None:
        """Whether the expert echoed the Live extension on the last response; ``None`` before."""
        if self._client is None or self._client.extensions is None:
            return None
        return self._client.extensions.livekit

    async def run(
        self,
        prompt: str,
        *,
        progress: Progress | None = None,
        say: Progress | None = None,
        context_id: str | None = None,
        message_history: ChatContext | Sequence[ChatItem] | None = None,
        usage: RunUsage | None = None,
        caller_lane: str | None = None,
        task_id: str | None = None,
    ) -> DelegationResult:
        """One request, or with ``task_id`` the answer to the question that expert task asked,
        which continues it; ``caller_lane`` is the caller's lane of ``context_id``, which an
        expert of this project may read. Inside a turn both default to the turn's. ``say`` takes
        the expert's exact text, ``progress`` when it is not given; ``DelegationFailed`` when the
        expert failed the task or never took it."""
        if context_id is None and (hooks := current_run_hooks()) is not None:
            context_id, caller_lane = hooks.thread_id, hooks.lane
        try:
            client = await self._connect()
            reply = await self._send(
                client, prompt, context_id, message_history, caller_lane, task_id
            )
        except A2AError as e:
            # unreachable, or refused: the caller's model hears why
            raise DelegationFailed(f"the expert did not take the request: {e}") from e
        if self._unread:
            # the expert is back: the next request offers the tool as its card describes it
            self._build_tool(client)
        offers = client.offers_extension

        try:
            update, text = await _answer(
                reply,
                progress,
                say or progress,
                lambda: client.resubscribe(reply.task_id, thread_id=reply.thread_id or None),
                self._conn_options,
            )
        except asyncio.CancelledError:
            self._cancel_in_background(reply.task_id, reply.thread_id)
            raise

        if update.state == "failed":
            raise DelegationFailed(text or "the delegation failed")
        asked = update.state == "input-required"

        return DelegationResult(
            output=text,
            state=update.state,
            task_id=update.task_id or reply.task_id,
            context_id=update.context_id or reply.thread_id,
            input_required=text if asked else None,
            new_messages=answer_message(text),
            usage=usage,
            interrupted=update.state == "canceled",
            endpoint=self._url,
            transport=client.transport or "",
            profile="livekit" if offers else "a2a",
            verbatim=text if update.verbatim and text else None,
        )

    async def _send(
        self,
        client: AgentClient,
        prompt: str,
        context_id: str | None,
        message_history: ChatContext | Sequence[ChatItem] | None,
        caller_lane: str | None,
        task_id: str | None,
    ) -> Reply:
        """The request as the card allows: the caller's session and lane, the delegation tag
        and, with no session to name, the conversation, all only when the expert offers the
        extension. An answer goes to the task that asked, alone."""
        from ..a2a.extension import CALLER_LANE, KIND, KIND_DELEGATION

        offers = client.offers_extension
        shared = context_id if offers else None
        metadata: dict[str, JsonValue] = {KIND: KIND_DELEGATION}
        if shared is not None and caller_lane:
            metadata[CALLER_LANE] = caller_lane
        own = None
        if shared is None:
            # the latest `FunctionCall` to this endpoint holds `lk.delegation.context_id` in `extra`
            items = (
                message_history.items
                if isinstance(message_history, ChatContext)
                else message_history or []
            )
            own = next(
                (
                    item.extra[CONTEXT_ID_KEY]
                    for item in reversed(items)
                    if isinstance(item, FunctionCall)
                    and item.extra.get(ENDPOINT_KEY) == self._url
                    and item.extra.get(CONTEXT_ID_KEY)
                ),
                None,
            )
        return await client.send(
            prompt,
            thread_id=shared or own,
            task_id=task_id,
            metadata=metadata if offers else None,
            chat_ctx=(
                _conversation(message_history)
                if offers and shared is None and task_id is None
                else None
            ),
        )

    def _cancel_in_background(self, task_id: str, context_id: str) -> None:
        """The caller gave up (a barge-in, a closed call): tell the expert why, off the turn."""
        if not task_id:
            return

        cancel = self.cancel(task_id, context_id=context_id, reason="caller-canceled")
        self._cancels.add(asyncio.ensure_future(cancel))

    async def cancel(self, task_id: str, *, context_id: str = "", reason: str = "") -> None:
        """Ask the expert to stop a task of ``context_id``; best-effort, a running tool
        finishes."""
        if self._client is None or not task_id:
            return

        try:
            await self._client.cancel(task_id, thread_id=context_id or None, reason=reason or None)
        except Exception as e:
            logger.debug(
                "the expert did not cancel the task",
                extra={"url": self._url, "taskId": task_id, "error": str(e)},
            )

    async def aclose(self) -> None:
        if self._cancels:
            await asyncio.gather(*self._cancels, return_exceptions=True)
            self._cancels.clear()

        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _connect(self) -> AgentClient:
        if self._client is None:
            from ..a2a.client import AgentClient

            # a minted token expires, so the client mints one per request rather than keeping it
            token = self._token if self._token is not None else self._project_token
            self._client = AgentClient(self._url, token, httpx_client=self._httpx_client)

        await self._client.connect()
        return self._client

    def _project_token(self) -> str | None:
        """A project token when the expert is on the project's own host, else ``None``."""
        livekit_url = os.environ.get("LIVEKIT_URL") or ""
        api_key = os.environ.get("LIVEKIT_API_KEY") or ""
        api_secret = os.environ.get("LIVEKIT_API_SECRET") or ""
        if not (livekit_url and api_key and api_secret):
            return None
        if urlsplit(self._url).netloc.lower() != urlsplit(livekit_url).netloc.lower():
            return None

        from ..a2a.auth import Credentials

        return Credentials(api_key, api_secret).project_token(identity="delegate")


def _asking_call(ctx: RunContext) -> str | None:
    """The expert's task whose question this call answers, when the framework made the call
    with the person's answer: the task the asking call reached."""
    asked_by = ctx.function_call.extra.get(ANSWERS_KEY)
    if not asked_by or ctx._run is None:
        return None
    for item in reversed(ctx._run.chat_ctx.items):
        if isinstance(item, FunctionCall) and item.call_id == asked_by:
            task_id = item.extra.get(TASK_ID_KEY)
            return task_id if isinstance(task_id, str) and task_id else None
    return None


def _conversation(
    message_history: ChatContext | Sequence[ChatItem] | None,
) -> ChatContext | None:
    """The caller's history as the expert may see it: the turns, no calls or instructions."""
    if message_history is None:
        return None

    if not isinstance(message_history, ChatContext):
        message_history = ChatContext(list(message_history))
    return message_history.copy(
        exclude_function_call=True,
        exclude_handoff=True,
        exclude_config_update=True,
        exclude_instructions=True,
    )


async def _answer(
    reply: Reply,
    progress: Progress | None,
    say: Progress | None,
    resubscribe: Callable[[], Reply],
    conn_options: APIConnectOptions,
) -> tuple[TaskUpdate, str]:
    """Hands the progress to ``progress`` and the exact text to ``say`` as they arrive; returns
    the terminal update and its own text, never a progress line. A stream that drops is read
    again from the start with ``resubscribe``, and what was handed on is not handed on twice."""
    reported = 0
    updates = reply
    for attempt in range(conn_options.max_retry + 1):
        skip = reported
        try:
            async for update in updates:
                if update.state not in ("working", "submitted"):
                    return update, update.text

                if not update.text:
                    continue
                if skip > 0:
                    skip -= 1
                    continue
                reported += 1
                await report(say if update.verbatim else progress, update.text)
        except (A2AClientError, httpx.HTTPError) as e:
            logger.info(
                "an expert's answer stream dropped",
                extra={"taskId": reply.task_id, "attempt": attempt, "error": str(e)},
            )
        if not reply.task_id or attempt == conn_options.max_retry:
            break
        await asyncio.sleep(conn_options._interval_for_retry(attempt))
        updates = resubscribe()

    raise DelegationFailed("the delegation ended without an answer")


__all__ = ["A2ADelegate"]
