from __future__ import annotations

import asyncio
import inspect
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from typing_extensions import TypedDict

from ..llm.chat_context import ChatContext
from ..llm.tool_context import FunctionTool, ToolError, function_tool
from ..log import logger
from ..utils import aio

# imported at runtime: the delegate tool's signature is resolved with get_type_hints() when a
# call arrives, so RunContext has to be a real name by then
from .events import RunContext
from .run_result import ChatMessageEvent

if TYPE_CHECKING:
    from .agent import Agent
    from .agent_session import AgentSession
    from .run_result import RunResult


MESSAGE_SOURCE_KEY = "message_source"
"""``ChatMessage.extra`` key: what produced an assistant message.

``"turn_end"`` for a generation step that issued no tool call, ``"tool_call"`` for one that
did, and ``"say"`` for a line from ``AgentSession.say``.
"""

DELEGATE_TOOL_NAME = "lk_agents_delegate"
"""The builtin tool a delegate is reached through, fixed so a duplex model can synthesize it."""

TOOL_DESCRIPTION = """Hand a request to the expert that handles reasoning, lookups and actions.

Default to using this. Delegate anything that is not small talk, not already answered earlier
in this conversation, and not covered by one of your other tools. Never guess, never answer
from memory, and never tell the user something is impossible before asking.

Delegate: account, order, billing and status questions; anything needing a lookup, a
calculation or a change; anything with a rule or policy behind it; anything you are unsure of.
Do not delegate: greetings, chit-chat, acknowledgements, or repeating what was already said.

State the request in full — the expert sees the conversation but not your intent.
The expert is never mentioned to the user: no consulting anyone, no handing anything over, no
passing it on — as far as they are concerned, this is you doing the work."""

# with `announce` off nothing else acknowledges, so the line comes free in the same completion
ACK_DIRECTIVE = """
In the same turn as the call, say one short line so the user is not left in silence —
"one sec", "on it", "okay, looking now", etc. — varying the wording. Do not restate the
request and do not promise an outcome."""

TOOL_DESCRIPTION_WITH_ACK = TOOL_DESCRIPTION + "\n" + ACK_DIRECTIVE

# no reply is generated from this one: the model's own line alongside the call acknowledges
DISPATCHED_SILENT = "Started. The answer is a separate entry, not this one."

# with `announce` on, the model answers this instead of writing its own line
DISPATCHED = (
    'Acknowledge in a few natural words — "one moment", "sure, let me check", "okay, looking '
    'now" — varying the wording, restating none of the request and promising nothing about '
    "the outcome. The answer is a separate entry, not this one."
)

DelegationState = Literal["working", "completed", "failed", "canceled", "input-required"]
"""How far a delegation has got, in the task states of the A2A protocol.

``working`` is intermediate and repeats; the rest are terminal.
"""


class DelegationOptions(TypedDict, total=False):
    """Configuration for delegation.

    Can be passed as a plain dict::

        AgentSession(
            delegate=...,
            delegation_options={"metadata": {"customer_id": "c-42"}},
        )
    """

    metadata: dict[str, Any]
    """Application data attached to every request. JSON-serializable. Defaults to ``{}``."""
    announce: bool
    """Whether answering the dispatch note is what acknowledges a delegation. Defaults to True.

    A model cannot be relied on to write a line alongside the tool call, and realtime models
    routinely emit the call and no speech. Turn it off for a model that does write one — the
    ``delegate`` tool then asks for it, which costs no round trip.
    """


def _resolve_delegation_options(config: DelegationOptions | None = None) -> DelegationOptions:
    """Fill in defaults for missing keys."""
    opts = DelegationOptions(metadata={}, announce=True)
    opts.update(config or {})
    return opts


@dataclass
class DelegationRequest:
    """One request handed to a delegate, the same whether it runs here or behind an endpoint."""

    task: str
    """What the conversation model asked for, in its words."""
    chat_ctx: ChatContext
    """The conversation so far, as the expert should see it."""
    delegation_id: str
    """The ``delegate`` call id. Stable for the life of one delegation."""
    metadata: dict[str, Any] = field(default_factory=dict)
    """Application data from ``DelegationOptions``. JSON-serializable."""


@dataclass
class DelegationUpdate:
    """What comes back from a delegate: where it has got to, and anything it has to say."""

    text: str = ""
    state: DelegationState = "working"


class DelegationStream(ABC):
    """The updates from one delegation, until it declares a terminal state.

    Subclasses do the work in :meth:`_run`. Closing the stream ends the delegation wherever
    it runs, so a caller reads it under ``async with``::

        async with delegate(request) as stream:
            async for update in stream:
                ...
    """

    def __init__(self, request: DelegationRequest) -> None:
        self._request = request
        self._event_ch = aio.Chan[DelegationUpdate]()
        self._task = asyncio.create_task(self._main_task(), name="DelegationStream._run")
        self._task.add_done_callback(lambda _: self._event_ch.close())

    @property
    def request(self) -> DelegationRequest:
        return self._request

    @abstractmethod
    async def _run(self) -> str | None:
        """Do the work: report progress with :meth:`send`, and answer by returning the text.

        A delegate that has to declare anything other than ``completed`` — a cancellation, or
        a request for more input — sends that terminal update itself and returns ``None``.
        Raising fails the delegation.
        """

    async def _main_task(self) -> None:
        # returning is the ordinary way to complete; an empty answer is still an answer, the
        # way a tool may return None
        if (answer := await self._run()) is not None:
            self.send(DelegationUpdate(answer, state="completed"))

    def send(self, update: DelegationUpdate) -> None:
        """Hand one update to the caller."""
        self._event_ch.send_nowait(update)

    async def aclose(self) -> None:
        await aio.cancel_and_wait(self._task)
        self._event_ch.close()

    async def __anext__(self) -> DelegationUpdate:
        try:
            return await self._event_ch.__anext__()
        except StopAsyncIteration:
            if not self._task.cancelled() and (exc := self._task.exception()):
                raise exc  # noqa: B904

            raise StopAsyncIteration from None

    def __aiter__(self) -> AsyncIterator[DelegationUpdate]:
        return self

    async def __aenter__(self) -> DelegationStream:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.aclose()


@runtime_checkable
class Delegate(Protocol):
    """Starts one delegation, and answers it or hands back the stream of its updates.

    A plain ``async def handler(request) -> str`` is a delegate, which is why an HTTP endpoint
    that returns text is one too. A :class:`DelegationStream` subclass is also a delegate on
    its own, since constructing one starts the work; a delegate with a lifetime of its own — an
    open HTTP connection, say — implements ``__call__`` and returns a fresh stream per call.
    """

    def __call__(self, request: DelegationRequest) -> DelegationStream | Awaitable[str]: ...


class _AgentDelegationStream(DelegationStream):
    def __init__(self, request: DelegationRequest, *, agent: Agent, session: AgentSession) -> None:
        self._agent = agent
        self._session = session
        super().__init__(request)

    async def _run(self) -> str | None:
        self._agent._chat_ctx = self._request.chat_ctx
        # the last turn the expert concluded; anything before it announced what followed
        answer = ""

        async def _forward(run: RunResult[Any]) -> None:
            """Progress out as it arrives; the last turn the expert concluded is the answer."""
            nonlocal answer
            # only what the expert said surfaces — its tool traffic is work that happened, and
            # a report through ctx.update() carries framing written for its own model
            async for ev in run:
                if not isinstance(ev, ChatMessageEvent) or ev.item.role != "assistant":
                    continue
                if not (text := ev.item.text_content):
                    continue

                if ev.item.extra.get(MESSAGE_SOURCE_KEY) != "turn_end":
                    self.send(DelegationUpdate(text))
                    continue

                if answer:
                    self.send(DelegationUpdate(answer))
                answer = text
            await run

        try:
            # on_enter speaks before the request is put, and a run has to be drained before
            # the next one starts, so its turn is read out here rather than dropped
            await _forward(
                await self._session.start(agent=self._agent, record=False, capture_run=True)
            )
            await _forward(self._session.run(user_input=self._request.task))
        except Exception as exc:
            logger.exception("the expert's run failed", extra={"task": self._request.task})
            self.send(DelegationUpdate(str(exc) or type(exc).__name__, state="failed"))
            return None

        return answer

    async def aclose(self) -> None:
        # the session is closed here rather than in `_run`, so the close is not itself running
        # under the cancellation that ended the work
        await super().aclose()
        await self._session.aclose()


class AgentDelegate:
    """A delegate that runs an :class:`Agent` in its own session, one per delegation.

    ``factory`` runs for each delegation because an agent binds to a single session and
    delegations run concurrently::

        delegate=AgentDelegate(FareDesk)
        delegate=AgentDelegate(lambda: FareDesk(customer_id=customer_id))
    """

    def __init__(
        self,
        factory: Callable[[], Agent],
        *,
        session_factory: Callable[[], AgentSession] | None = None,
    ) -> None:
        self._factory = factory
        self._session_factory = session_factory

    def __call__(self, request: DelegationRequest) -> DelegationStream:
        from .agent_session import AgentSession

        return _AgentDelegationStream(
            request,
            agent=self._factory(),
            session=self._session_factory() if self._session_factory else AgentSession(),
        )


def build_delegate_tool(description: str | None = None, *, announce: bool = True) -> FunctionTool:
    """The one tool the conversation model gains, to reach whichever delegate is in force."""

    async def delegate(ctx: RunContext, task: str) -> str:
        session = ctx.session
        activity = session.current_agent._get_activity_or_raise()
        handler = activity.delegate
        if handler is None:
            raise RuntimeError("the delegate tool ran with no delegate configured")

        # releases the turn so the conversation model keeps talking while the expert works. a
        # model that speaks whatever is pushed to it cannot be released quietly, so with
        # `announce` off it holds the turn rather than acknowledging against instruction
        rt_session = activity.realtime_llm_session
        if announce or rt_session is None or not rt_session.capabilities.auto_tool_reply_generation:
            await ctx.update(DISPATCHED if announce else DISPATCHED_SILENT, silent=not announce)

        request = DelegationRequest(
            task=task,
            chat_ctx=session.current_agent.chat_ctx.copy(
                exclude_function_call=True,
                exclude_handoff=True,
                exclude_config_update=True,
                exclude_instructions=True,
            ),
            delegation_id=ctx.function_call.call_id,
            metadata=dict(session._opts.delegation_options["metadata"]),
        )

        result = handler(request)
        if inspect.isawaitable(result):
            # one answer and no lifetime to release: awaiting it is the whole delegation
            return await result

        # a terminal state leaves the delegation running, holding a session here or an open
        # HTTP stream elsewhere, until the stream is closed
        async with result as stream:
            async for ev in stream:
                if ev.state == "working":
                    if ev.text:
                        await ctx.update(ev.text)
                    continue
                if ev.state == "failed":
                    raise ToolError(ev.text or "the delegation failed")
                if ev.state == "input-required":
                    # not routed to the user in v0: the conversation model is told what is
                    # missing and asks for it, as it would for a tool that could not proceed
                    raise ToolError(
                        ev.text or "the delegation needs information the request did not carry"
                    )
                # completed and canceled both answer. a cancelled delegation still has to say
                # what happened, including side effects that already landed
                return ev.text

        raise ToolError("the delegate ended without declaring a result")

    # not CANCELLABLE, since the expert owns its work. duplicates are allowed because the check
    # keys on the function name, which would make every delegation a duplicate of every other
    # one; the in-flight placeholder for the pending call is what stops the model re-delegating
    return function_tool(
        delegate,
        name=DELEGATE_TOOL_NAME,
        # exactly one of the two acknowledges: the dispatch note when `announce` is on, the
        # model's own line in the same completion when it is off
        description=description or (TOOL_DESCRIPTION if announce else TOOL_DESCRIPTION_WITH_ACK),
    )
