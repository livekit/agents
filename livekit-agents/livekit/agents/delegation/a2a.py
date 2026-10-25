"""``A2ADelegate``: an expert on another deployment, or any A2A server, as a toolset.

A stock A2A client with the Live extension on top, JSON-RPC over HTTP+SSE. The caller degrades
from what the card offers: with the extension, ``livekit/kind: delegation`` and, outside the
caller's session, the conversation as a ``chat_ctx`` part; without it, the request as text, the
expert's ``working`` status messages as progress, the artifact text as the result.

Attached to an ``Agent`` as ``tools=[A2ADelegate(url)]`` it is one tool, named and described
from the expert's card, the same in a text run and on a call (docs/multi-agent.md).
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Literal
from urllib.parse import urlsplit

import httpx
from a2a.client.errors import A2AClientError
from pydantic import JsonValue

from ..llm.chat_context import ChatContext, ChatItem, FunctionCall
from ..llm.tool_context import (
    ANSWERS_KEY,
    HEARD_KEY,
    FunctionTool,
    InputRequired,
    Tool,
    ToolError,
    ToolResult,
    Toolset,
    function_tool,
)
from ..log import logger
from ..voice.agent_run import current_run_hooks
from ..voice.events import RunContext
from ..voice.run_result import RunUsage
from .delegate import (
    Delegate,
    DelegationFailed,
    DelegationResult,
    Progress,
    answer_message,
    report,
)
from .tool import CONTEXT_ID_KEY, DISPATCHED, TASK_ID_KEY, record_delegation

RESUBSCRIBE_ATTEMPTS = 3
"""How many times a dropped answer stream is read again before the delegation fails."""
NO_ANSWER = "the expert finished with no answer"
"""The delegate's result when the expert's task ended with no text: that fact, not its progress."""

ACKNOWLEDGE_WITH_CALL = (
    "In the same reply as this call, say a few words that you are on it, so the person hears "
    "something at once; promise nothing about the outcome."
)

CUT_OFF = "(The person's message cut your question off: they had heard {heard} of it.)"
"""The line before an answer that cut the expert's question off, so a stock expert's model reads
it too."""

ASK = (
    "State the request in full, with everything the person said that bears on it; the "
    "answer, or the question it has for the person, is what you say next."
)

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

    As a toolset it is one tool, ``ask``, described by the card, read once in ``setup()``;
    ``name=`` tells two experts on one agent apart. The tool hands the request on with the run's thread and
    history, speaks the expert's progress through ``ctx.update()``, records the expert's task on
    the call and returns the answer, or the question the expert has for the person. On a call
    ``announce`` releases the turn with a line the model acknowledges while the expert works;
    ``announce="with_call"`` asks the model to acknowledge in the reply that makes the call
    instead, so the person hears it at once, on a call and in text. The answer to the expert's
    question continues the expert's task. The tool writes the person's message to the caller's
    lane before it sends, and names that lane (``livekit/callerLane``), so an expert of this
    project can read the person's own words.

    An expert whose card offers the Live extension gets the caller's session id as the
    ``contextId`` and writes its own lane of that session; outside a turn, with no session to
    name, it gets the conversation as a ``chat_ctx`` part and keeps a session of its own. Any
    other A2A server gets the request as text and nothing else, and keeps a context of its own
    (docs/multi-agent.md). ``token`` omitted, an expert on the project's
    own host (``LIVEKIT_URL``) gets a project token minted per request from ``LIVEKIT_API_KEY``
    and ``LIVEKIT_API_SECRET``; an expert anywhere else gets none.
    """

    def __init__(
        self,
        url: str,
        *,
        token: str | None = None,
        httpx_client: httpx.AsyncClient | None = None,
        name: str | None = None,
        announce: bool | Literal["with_call"] = True,
    ) -> None:
        self._url = url.rstrip("/")
        self._token = token
        self._httpx_client = httpx_client
        self._client: AgentClient | None = None
        # a stock expert keeps a context of its own: one per caller thread
        self._own_contexts: dict[str | None, str] = {}
        self._tool_name = name
        self._announce = announce
        self._tool: FunctionTool | None = None
        Toolset.__init__(self, id=self._url.rsplit("/", 1)[-1] or self._url)

    # -- the toolset --

    @property
    def tools(self) -> Sequence[Tool | Toolset]:
        return [self._tool] if self._tool is not None else []

    async def setup(self) -> A2ADelegate:
        """Reads the card and builds the tool from it; once per delegate."""
        if self._tool is None:
            client = await self._connect()
            card = client.card
            title = card.name if card is not None else self.id
            self._id = title
            name = self._tool_name or "ask"
            about = card.description.strip() if card is not None and card.description else ""
            description = f"{about}\n\n{ASK}" if about else ASK
            if self._announce == "with_call":
                description = f"{description} {ACKNOWLEDGE_WITH_CALL}"
            self._tool = function_tool(self._ask, name=name, description=description)
            # the person's answer to the expert's question comes back to this tool as said, and
            # the expert's own model reads it
            self._tool._relays_answer = True  # type: ignore[attr-defined]
        return self

    async def _ask(self, ctx: RunContext, request: str) -> str | InputRequired | ToolResult:
        """The tool: the request to the expert with what the run knows, the answer back. A call
        the framework made with the person's answer to the expert's question continues the
        expert's task with it, as the person said it.

        Args:
            request: the request, in full
        """
        session = ctx._session
        asked = _asking_call(ctx)
        if session is not None and self._announce is True and asked is None:
            await ctx.update(DISPATCHED)
        hooks = current_run_hooks()
        # the expert may read the caller's lane: the person's message is on it first
        if hooks is not None and hooks.record_message is not None:
            await hooks.record_message()
        elif session is not None:
            await session._record_message()
        caller_lane = (
            session.thread.agent_name
            if session is not None and session.thread is not None
            else (hooks.lane if hooks is not None else None)
        )
        try:
            result = await self.run(
                request,
                progress=ctx.update,
                context_id=session.thread.id if session and session.thread else None,
                message_history=_history_of(ctx),
                usage=ctx.usage,
                caller_lane=caller_lane,
                task_id=_str(asked.extra.get(TASK_ID_KEY)) if asked is not None else None,
                expert_context=_str(asked.extra.get(CONTEXT_ID_KEY)) if asked is not None else None,
                heard=_str(ctx.function_call.extra.get(HEARD_KEY), empty=True),
            )
        except DelegationFailed as e:
            raise ToolError(str(e) or "the delegation failed") from e

        record_delegation(ctx.function_call, result)
        exact = result.verbatim
        if result.input_required:
            # the expert asked the person something: asked as it wrote it when it said so, and
            # what they say next continues the expert's task, as they said it
            verbatim = exact is not None and exact.strip() == result.input_required.strip()
            return InputRequired(result.input_required, verbatim=verbatim)
        answer = result.output if isinstance(result.output, str) else str(result.output or "")
        if exact and answer.startswith(exact):
            # the person gets the exact part as written; the rest is the model's to phrase
            rest = answer[len(exact) :].strip()
            said = ToolResult(exact, verbatim=True, reply_required=bool(rest))
            said._more = rest  # type: ignore[attr-defined]
            return said
        return answer

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
        context_id: str | None = None,
        message_history: ChatContext | Sequence[ChatItem] | None = None,
        usage: RunUsage | None = None,
        task_id: str | None = None,
        expert_context: str | None = None,
        caller_lane: str | None = None,
        heard: str | None = None,
    ) -> DelegationResult:
        """One request, or with ``task_id`` the answer to a question the expert's task asked,
        which continues that task; ``expert_context`` is the context the expert kept for it,
        ``heard`` what the person had of the question when their answer cut it off.
        ``caller_lane`` names the caller's lane in the shared thread, for the expert to read."""
        client = await self._connect()
        offers = client.offers_extension
        context_id = self._caller_session(context_id)
        reply = await self._send(
            client, prompt, context_id, message_history, task_id, expert_context, caller_lane, heard
        )
        # a caller that gives up (a barge-in, a closed call) leaves the expert's task running:
        # only an explicit cancel stops it; a stream that drops is read again, the task is not
        # affected
        update, text = await _answer(
            reply,
            progress,
            lambda: client.resubscribe(reply.task_id, thread_id=reply.thread_id or None),
        )

        if update.state == "failed":
            raise DelegationFailed(text or "the delegation failed")
        # an expert task that ended with no text says so: its last progress is not its answer
        text = text or NO_ANSWER

        return DelegationResult(
            output=text,
            state=update.state,
            task_id=update.task_id or reply.task_id,
            context_id=update.context_id or reply.thread_id,
            input_required=text if update.state == "input-required" else None,
            new_messages=answer_message(text),
            usage=usage,
            interrupted=update.state == "canceled",
            endpoint=self._url,
            transport=client.transport or "",
            profile="livekit" if offers else "a2a",
            verbatim=_verbatim(update.metadata),
        )

    async def _send(
        self,
        client: AgentClient,
        prompt: str,
        context_id: str | None,
        message_history: ChatContext | Sequence[ChatItem] | None,
        task_id: str | None,
        expert_context: str | None,
        caller_lane: str | None,
        heard: str | None = None,
    ) -> Reply:
        """The request as the card allows: the caller's session, the delegation tag and, with
        no session to name, the conversation, all only when the expert offers the extension. An
        answer goes to the task that asked, in the context that task is in, and alone; one that
        cut the question off says what the person had of it, in a line and in ``HEARD``."""
        from ..a2a.extension import CALLER_LANE, HEARD, INTERRUPT, KIND, KIND_DELEGATION

        offers = client.offers_extension
        shared = context_id if offers else None
        metadata: dict[str, JsonValue] | None = None
        if offers:
            # a delegation never interrupts what the expert is producing
            metadata = {KIND: KIND_DELEGATION, INTERRUPT: False}
            if shared is not None and caller_lane:
                metadata[CALLER_LANE] = caller_lane
            if task_id and heard is not None:
                metadata[HEARD] = heard
        if task_id and heard is not None:
            line = CUT_OFF.format(heard=f'only "{heard}"' if heard else "none")
            prompt = f"{line}\n{prompt}"
        reply = await client.send(
            prompt,
            thread_id=(expert_context if task_id else None)
            or shared
            or self._own_contexts.get(context_id),
            task_id=task_id,
            metadata=metadata,
            chat_ctx=(
                _conversation(message_history)
                if offers and shared is None and task_id is None
                else None
            ),
        )

        if shared is None and reply.thread_id:
            self._own_contexts[context_id] = reply.thread_id
        return reply

    def _caller_session(self, context_id: str | None) -> str | None:
        """From a tool inside a turn, the session the run belongs to; else what was given."""
        if context_id is not None:
            return context_id

        hooks = current_run_hooks()
        return hooks.thread_id if hooks is not None else None

    async def cancel(self, task_id: str, *, reason: str = "") -> None:
        """``tasks/cancel`` of an expert's task, for a tool the developer gives the model: its
        tools stop at their next wait point, and the task ends ``canceled``."""
        if self._client is None or not task_id:
            return

        try:
            await self._client.cancel(task_id, reason=reason or None)
        except Exception as e:
            logger.debug(
                "the expert did not cancel the task",
                extra={"url": self._url, "taskId": task_id, "error": str(e)},
            )

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _connect(self) -> AgentClient:
        if self._client is None:
            from ..a2a.client import AgentClient
            from ..http import LOCAL_ROUTES_ENV

            # a minted token expires, so the client mints one per request rather than keeping it
            token = self._token if self._token is not None else self._project_token
            http = self._httpx_client
            routes = [
                tuple(route.split("=", 1))
                for route in (os.environ.get(LOCAL_ROUTES_ENV) or "").split(",")
                if "=" in route
            ]
            if http is None and any(self._url.startswith(public) for public, _ in routes):
                # the expert is a door of this call's own worker: reached over the loopback,
                # its answer streamed as written, not through the node
                http = httpx.AsyncClient(
                    timeout=None,
                    transport=_LocalRoutes(dict(routes)),  # type: ignore[arg-type]
                    limits=httpx.Limits(max_keepalive_connections=0),
                )
                logger.info(
                    "a delegation to this worker goes over the loopback", extra={"url": self._url}
                )
            self._client = AgentClient(self._url, token, httpx_client=http)

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


class _LocalRoutes(httpx.AsyncBaseTransport):
    """Sends a request under a local route's public prefix to the worker's loopback listener."""

    def __init__(self, routes: dict[str, str]) -> None:
        self._routes = routes
        self._transport = httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        for public, local in self._routes.items():
            if url.startswith(public):
                request.url = httpx.URL(local + url[len(public) :])
                break
        return await self._transport.handle_async_request(request)

    async def aclose(self) -> None:
        await self._transport.aclose()


def _asking_call(ctx: RunContext) -> FunctionCall | None:
    """The call whose question this call answers, when the framework made it with the person's
    answer: it names the expert's task to continue."""
    asked_by = ctx.function_call.extra.get(ANSWERS_KEY)
    if not asked_by:
        return None
    if ctx._session is not None:
        history = ctx._session.history
    elif ctx._run is not None:
        history = ctx._run.chat_ctx
    else:
        return None
    for item in reversed(history.items):
        if isinstance(item, FunctionCall) and item.call_id == asked_by:
            return item
    return None


def _str(value: object, *, empty: bool = False) -> str | None:
    return value if isinstance(value, str) and (value or empty) else None


def _history_of(ctx: RunContext) -> ChatContext:
    """The conversation as the expert may see it: the turns, no calls or instructions."""
    history: ChatContext
    if ctx._session is not None:
        history = ctx._session.history
    else:
        assert ctx._run is not None
        history = ctx._run.chat_ctx
    return history.copy(
        exclude_function_call=True,
        exclude_handoff=True,
        exclude_config_update=True,
        exclude_instructions=True,
    )


def _conversation(
    message_history: ChatContext | Sequence[ChatItem] | None,
) -> ChatContext | None:
    """The caller's history as the expert may see it: the turns, no calls or instructions."""
    if message_history is None:
        return None

    if not isinstance(message_history, ChatContext):
        message_history = ChatContext(list(message_history))
    return message_history.copy(
        exclude_function_call=True, exclude_handoff=True, exclude_instructions=True
    )


def _verbatim(metadata: object) -> str | None:
    from ..a2a.extension import VERBATIM

    exact = metadata.get(VERBATIM) if isinstance(metadata, dict) else None
    return exact if isinstance(exact, str) and exact else None


async def _answer(
    reply: Reply, progress: Progress | None, resubscribe: Callable[[], Reply] | None = None
) -> tuple[TaskUpdate, str]:
    """Reports the progress as it arrives; returns the terminal update and its own text, never
    a progress line. A stream that ends early or fails is read again with ``resubscribe``, from the start, and
    the progress already reported is not reported twice."""
    reported = 0
    updates: Reply = reply
    for attempt in range(RESUBSCRIBE_ATTEMPTS + 1):
        skip = reported
        try:
            async for update in updates:
                if update.state not in ("working", "submitted"):
                    return update, update.text

                if update.text:
                    if skip > 0:
                        skip -= 1
                        continue
                    reported += 1
                    await report(progress, update.text)
        except (A2AClientError, httpx.HTTPError) as e:
            logger.info("an expert's answer stream dropped", extra={"error": str(e)})
        if resubscribe is None or not reply.task_id or attempt == RESUBSCRIBE_ATTEMPTS:
            break
        updates = resubscribe()

    raise DelegationFailed("the delegation ended without an answer")


__all__ = ["A2ADelegate"]
