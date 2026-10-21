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
from collections.abc import Sequence
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from ..llm.chat_context import ChatContext, ChatItem
from ..llm.tool_context import (
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
from .tool import DISPATCHED, record_delegation

ASK = (
    "State the request in full, with everything the person said that bears on it; the "
    "answer, or the question it has for the person, is what you say next."
)

if TYPE_CHECKING:
    import httpx

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
    ``announce`` releases the turn with a line the model acknowledges while the expert works.

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
        announce: bool = True,
    ) -> None:
        self._url = url.rstrip("/")
        self._token = token
        self._httpx_client = httpx_client
        self._client: AgentClient | None = None
        self._own_context: str | None = None
        self._cancels: set[asyncio.Future[None]] = set()
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
            self._tool = function_tool(self._ask, name=name, description=description)
        return self

    async def _ask(self, ctx: RunContext, request: str) -> str | InputRequired | ToolResult:
        """The tool: the request to the expert with what the run knows, the answer back.

        Args:
            request: the request, in full
        """
        session = ctx._session
        if session is not None and self._announce:
            await ctx.update(DISPATCHED)
        try:
            result = await self.run(
                request,
                progress=ctx.update,
                context_id=session.thread.id if session and session.thread else None,
                message_history=_history_of(ctx),
                usage=ctx.usage,
            )
        except DelegationFailed as e:
            raise ToolError(str(e) or "the delegation failed") from e

        record_delegation(ctx.function_call, result)
        exact = result.verbatim
        if result.input_required:
            # the expert asked the person something: asked as it wrote it when it said so, and
            # what they say next is this tool's next request, as they said it
            verbatim = exact is not None and exact.strip() == result.input_required.strip()
            return InputRequired(result.input_required, tool=self._tool, verbatim=verbatim)
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
    ) -> DelegationResult:
        client = await self._connect()
        offers = client.offers_extension
        context_id = self._caller_session(context_id)
        reply = await self._send(client, prompt, context_id, message_history)

        try:
            update, text = await _answer(reply, progress)
        except asyncio.CancelledError:
            self._cancel_in_background(reply.task_id)
            raise

        if update.state == "failed":
            raise DelegationFailed(text or "the delegation failed")

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
    ) -> Reply:
        """The request as the card allows: the caller's session, the delegation tag and, with
        no session to name, the conversation, all only when the expert offers the extension."""
        from ..a2a.extension import KIND, KIND_DELEGATION

        offers = client.offers_extension
        shared = context_id if offers else None
        reply = await client.send(
            prompt,
            thread_id=shared or self._own_context,
            metadata={KIND: KIND_DELEGATION} if offers else None,
            chat_ctx=_conversation(message_history) if offers and shared is None else None,
        )

        if shared is None:
            self._own_context = reply.thread_id or self._own_context
        return reply

    def _caller_session(self, context_id: str | None) -> str | None:
        """From a tool inside a turn, the session the run belongs to; else what was given."""
        if context_id is not None:
            return context_id

        hooks = current_run_hooks()
        return hooks.thread_id if hooks is not None else None

    def _cancel_in_background(self, task_id: str) -> None:
        """The caller gave up (a barge-in, a closed call): tell the expert why, off the turn."""
        if not task_id:
            return

        cancel = self.cancel(task_id, reason="caller-canceled")
        self._cancels.add(asyncio.ensure_future(cancel))

    async def cancel(self, task_id: str, *, reason: str = "") -> None:
        """Ask the expert to stop a task; best-effort, a running tool finishes."""
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


async def _answer(reply: Reply, progress: Progress | None) -> tuple[TaskUpdate, str]:
    """Reports the progress as it arrives; returns the terminal update and the answer text,
    the last progress message when the terminal update carries none."""
    last_text = ""

    async for update in reply:
        if update.state not in ("working", "submitted"):
            return update, update.text or last_text

        if update.text:
            last_text = update.text
            await report(progress, update.text)

    raise DelegationFailed("the delegation ended without an answer")


__all__ = ["A2ADelegate"]
